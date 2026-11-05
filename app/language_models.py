"""Pluggable answer-generation backends.

Four backends are available, all behind :class:`LanguageModelBackend`:

* :class:`OllamaBackend` — a quantised model served locally by Ollama. Nothing
  leaves the machine, which matters because the corpus is internal accreditation
  material. Qwen3 8B at Q4_K_M occupies roughly 5 GB, so it runs entirely on an
  8 GB card with headroom left for the context window.
* :class:`ClaudeCodeBackend` — the Claude Code CLI (``claude``).
* :class:`CodexBackend` — the OpenAI Codex CLI (``codex exec``).
* :class:`AntigravityBackend` — the Antigravity CLI (``agy``).

The three CLI backends share :class:`CommandLineAgentBackend`, which owns the
subprocess lifecycle, prompt delivery and stream decoding; a subclass supplies
only the executable and its arguments. Adding another agent CLI is therefore a
small subclass plus one registry entry.

**Privacy note.** Only the Ollama backend is fully local. Each CLI backend sends
the retrieved document excerpts to its vendor, so the choice of backend is a
data-handling decision, not just a quality one.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from .config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass
class BackendStatus:
    """Whether a backend can currently serve requests, and what it is running."""

    is_available: bool
    backend_name: str
    model_name: str
    detail: str = ""
    #: False for cloud backends, so the interface can warn before excerpts leave.
    runs_locally: bool = False
    #: Reasoning depth, reported separately from the model name so the interface
    #: can show it as its own badge rather than hiding it inside a string.
    effort: str = ""

    def as_dictionary(self) -> dict:
        """Represent the status as JSON-serialisable data for the interface."""
        return {
            "is_available": self.is_available,
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "detail": self.detail,
            "runs_locally": self.runs_locally,
            "effort": self.effort,
        }


class ReasoningTagStripper:
    """Removes chain-of-thought markup that reasoning models emit inline.

    Qwen3 wraps its deliberation in ``<think>`` tags. Ollama can suppress this
    server-side, but older builds ignore the flag, so the stream is filtered
    here as well. Filtering happens incrementally: text inside a tag is dropped,
    and a partial tag straddling two stream chunks is buffered until resolved.
    """

    _COMPLETE_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    OPENING_TAG = "<think>"
    CLOSING_TAG = "</think>"

    def __init__(self) -> None:
        """Start outside any reasoning block with an empty carry-over buffer."""
        self._inside_block = False
        self._pending_text = ""

    def strip_complete_text(self, text: str) -> str:
        """Remove every complete reasoning block from a finished response."""
        without_blocks = self._COMPLETE_BLOCK.sub("", text)
        # A block left unclosed by a truncated response would otherwise survive.
        opening_position = without_blocks.lower().find(self.OPENING_TAG)
        if opening_position != -1:
            without_blocks = without_blocks[:opening_position]
        return without_blocks.strip()

    def filter_stream_chunk(self, chunk: str) -> str:
        """Return the visible portion of one streamed chunk.

        Text inside a reasoning block is dropped. A tag split across two chunks
        is handled by holding back any trailing fragment that could still grow
        into a tag, so ``"...answer <thi"`` never reaches the browser.
        """
        self._pending_text += chunk
        visible_output = ""

        while True:
            if self._inside_block:
                closing_position = self._pending_text.lower().find(self.CLOSING_TAG)
                if closing_position == -1:
                    # Discard the reasoning text, but keep a possible partial tag.
                    keep_length = self._partial_tag_suffix_length(self._pending_text)
                    self._pending_text = (
                        self._pending_text[len(self._pending_text) - keep_length :]
                        if keep_length
                        else ""
                    )
                    break
                self._pending_text = self._pending_text[closing_position + len(self.CLOSING_TAG) :]
                self._inside_block = False
                continue

            opening_position = self._pending_text.lower().find(self.OPENING_TAG)
            if opening_position == -1:
                keep_length = self._partial_tag_suffix_length(self._pending_text)
                emit_length = len(self._pending_text) - keep_length
                visible_output += self._pending_text[:emit_length]
                self._pending_text = self._pending_text[emit_length:]
                break
            visible_output += self._pending_text[:opening_position]
            self._pending_text = self._pending_text[opening_position + len(self.OPENING_TAG) :]
            self._inside_block = True

        return visible_output

    def _partial_tag_suffix_length(self, text: str) -> int:
        """Length of the trailing fragment that could be the start of a tag.

        Returns 0 when the text cannot end mid-tag, meaning all of it is safe to
        emit (or, inside a block, safe to discard).
        """
        longest_tag = max(len(self.OPENING_TAG), len(self.CLOSING_TAG))
        for candidate_length in range(min(longest_tag - 1, len(text)), 0, -1):
            tail = text[-candidate_length:].lower()
            if self.OPENING_TAG.startswith(tail) or self.CLOSING_TAG.startswith(tail):
                return candidate_length
        return 0

    def flush(self) -> str:
        """Emit whatever is still buffered once the stream has ended.

        Text held back as a possible partial tag turned out not to be one, so it
        is released. Anything still inside an unterminated reasoning block is
        discarded rather than shown.
        """
        remaining = "" if self._inside_block else self._pending_text
        self._pending_text = ""
        return remaining


class AgentEventTextExtractor:
    """Pulls assistant-visible text out of one NDJSON event from an agent CLI.

    The three CLIs stream different event vocabularies, but only a handful of
    shapes actually carry answer text. Each is matched explicitly rather than by
    walking the JSON for any ``text`` key, because a generic search would also
    pick up tool arguments, reasoning summaries and usage metadata.

    Verified shapes:

    * Claude Code — ``{"type": "assistant", "message": {"content": [{"type":
      "text", "text": "..."}]}}``
    * Codex — ``{"type": "item.completed", "item": {"type": "agent_message",
      "text": "..."}}``
    * Anthropic-style deltas — ``{"type": "content_block_delta", "delta":
      {"text": "..."}}``

    Anything else yields no text, so unrecognised events are silently ignored
    rather than corrupting the answer.
    """

    #: Item types in Codex-style events that represent the assistant's answer.
    ANSWER_ITEM_TYPES = frozenset({"agent_message", "assistant_message", "message"})

    def extract(self, event: dict) -> list[str]:
        """Return the answer-text fragments carried by one event."""
        if not isinstance(event, dict):
            return []
        event_type = event.get("type")

        if event_type == "assistant":
            return self._read_anthropic_message(event.get("message"))
        if event_type in {"content_block_delta", "message_delta"}:
            return self._read_delta(event.get("delta"))
        if event_type in {"item.completed", "item.updated"}:
            return self._read_codex_item(event.get("item"))
        # Antigravity keys its events on "event" rather than "type".
        if event.get("event") == "step_update":
            return self._read_antigravity_step(event.get("step_update"))
        return []

    def _read_antigravity_step(self, step) -> list[str]:
        """Read a streamed text delta from an Antigravity step update.

        Only ``agent_response`` steps carry the answer; the same event type also
        reports tool calls and planning, which must not reach the reader.
        """
        if not isinstance(step, dict):
            return []
        if step.get("step_type") != "agent_response":
            return []
        delta = step.get("text_delta")
        return [delta] if isinstance(delta, str) and delta else []

    def extract_final_result(self, event: dict) -> str:
        """Return terminal-event text, used when no incremental text arrived.

        Some CLI builds report the answer only once, in a closing event. The
        event is identified by carrying a string ``result`` rather than by its
        ``type``, because the type name differs between builds.
        """
        if not isinstance(event, dict):
            return ""
        result_value = event.get("result")
        if isinstance(result_value, str):
            return result_value
        # Antigravity nests the final answer as result.response.
        if isinstance(result_value, dict):
            response = result_value.get("response")
            if isinstance(response, str):
                return response
        return ""

    def describe_error(self, event: dict) -> str | None:
        """Return an error message when the event reports a failed turn."""
        if not isinstance(event, dict):
            return None
        if event.get("is_error") is True or event.get("subtype") == "error":
            return str(event.get("result") or event.get("error") or "CLI melaporkan galat")
        if event.get("type") in {"turn.failed", "error"} or event.get("event") == "error":
            error_detail = event.get("error") or event.get("message")
            return str(error_detail or "CLI melaporkan galat")
        return None

    def _read_anthropic_message(self, message) -> list[str]:
        """Read the text blocks of an Anthropic-shaped assistant message."""
        if not isinstance(message, dict):
            return []
        content_blocks = message.get("content")
        if isinstance(content_blocks, str):
            return [content_blocks]
        if not isinstance(content_blocks, list):
            return []
        return [
            block["text"]
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        ]

    def _read_delta(self, delta) -> list[str]:
        """Read an incremental text delta."""
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            return [delta["text"]]
        return []

    def _read_codex_item(self, item) -> list[str]:
        """Read a completed Codex item, keeping only assistant answers."""
        if not isinstance(item, dict):
            return []
        if item.get("type") not in self.ANSWER_ITEM_TYPES:
            return []
        text_value = item.get("text") or item.get("message")
        return [text_value] if isinstance(text_value, str) and text_value else []


class LanguageModelBackend(ABC):
    """Common interface every answer-generation backend implements."""

    def __init__(self, settings: Settings) -> None:
        """Record configuration; connections are opened per request."""
        self._settings = settings

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier matching the ``LLM_BACKEND`` setting."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Which model this backend will generate with."""

    @property
    @abstractmethod
    def runs_locally(self) -> bool:
        """True when no document text leaves this machine."""

    @abstractmethod
    def check_status(self) -> BackendStatus:
        """Probe the backend and report whether it can serve a request now."""

    @abstractmethod
    def generate_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield answer text incrementally as the model produces it."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Collect a complete answer by draining :meth:`generate_stream`."""
        return "".join(self.generate_stream(system_prompt, user_prompt))


class OllamaBackend(LanguageModelBackend):
    """Generates answers with a local model served by Ollama.

    Requests go to Ollama's ``/api/chat`` endpoint with streaming enabled, so
    the first tokens reach the browser while the rest are still being produced.
    """

    #: Seconds to wait for Ollama to accept a request before giving up.
    CONNECT_TIMEOUT_SECONDS = 10
    #: Seconds of silence mid-stream before the request is considered dead.
    READ_TIMEOUT_SECONDS = 300

    @property
    def name(self) -> str:
        """Identifier used in configuration and reported to the interface."""
        return "ollama"

    @property
    def model_name(self) -> str:
        """The Ollama model tag that will answer questions."""
        return self._settings.ollama_model_name

    @property
    def runs_locally(self) -> bool:
        """Ollama runs on this machine, so no excerpt ever leaves it."""
        return True

    def check_status(self) -> BackendStatus:
        """Confirm the Ollama service is up and the configured model is pulled."""
        import requests

        try:
            response = requests.get(
                f"{self._settings.ollama_base_url}/api/tags",
                timeout=self.CONNECT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            installed_models = [entry.get("name", "") for entry in response.json().get("models", [])]
        except Exception as error:
            return BackendStatus(
                is_available=False,
                backend_name=self.name,
                model_name=self.model_name,
                detail=f"Ollama tidak dapat dihubungi di {self._settings.ollama_base_url}: {error}",
                runs_locally=True,
            )

        if not self._is_model_installed(installed_models):
            return BackendStatus(
                is_available=False,
                backend_name=self.name,
                model_name=self.model_name,
                detail=f"Model belum diunduh. Jalankan: ollama pull {self.model_name}",
                runs_locally=True,
            )
        return BackendStatus(
            is_available=True,
            backend_name=self.name,
            model_name=self.model_name,
            detail="Berjalan lokal, dokumen tidak dikirim ke luar",
            runs_locally=True,
        )

    def _is_model_installed(self, installed_models: list[str]) -> bool:
        """True when the configured model appears in Ollama's model list.

        Ollama reports tags as ``name:tag``; a configured name without a tag is
        matched against the bare name so ``qwen3`` matches ``qwen3:8b``.
        """
        wanted = self.model_name
        for installed in installed_models:
            if installed == wanted or installed.split(":")[0] == wanted.split(":")[0]:
                return True
        return False

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Stream an answer from Ollama, filtering out reasoning markup."""
        import requests

        request_payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            # Qwen3 reasons by default; grounded extraction does not need it and
            # switching it off roughly halves time-to-answer.
            "think": False,
            "keep_alive": self._settings.ollama_keep_alive,
            "options": {
                "temperature": self._settings.generation_temperature,
                "num_ctx": self._settings.ollama_context_window,
                "num_predict": self._settings.maximum_output_tokens,
            },
        }

        stripper = ReasoningTagStripper()
        with requests.post(
            f"{self._settings.ollama_base_url}/api/chat",
            json=request_payload,
            stream=True,
            timeout=(self.CONNECT_TIMEOUT_SECONDS, self.READ_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                visible_text = self._extract_visible_text(raw_line, stripper)
                if visible_text:
                    yield visible_text
        trailing_text = stripper.flush()
        if trailing_text:
            yield trailing_text

    def _extract_visible_text(self, raw_line: str, stripper: ReasoningTagStripper) -> str:
        """Parse one NDJSON line from Ollama into displayable text."""
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            return ""
        if event.get("error"):
            raise RuntimeError(f"Ollama melaporkan galat: {event['error']}")
        message = event.get("message") or {}
        # The separate "thinking" field is discarded; only "content" is shown.
        return stripper.filter_stream_chunk(message.get("content", ""))


class CommandLineAgentBackend(LanguageModelBackend):
    """Shared machinery for backends that shell out to an agent CLI.

    Subclasses declare their executable and arguments; this class owns the parts
    that are identical across all of them:

    * writing the prompt to standard input rather than passing it as an
      argument, which sidesteps the ~32,000-character Windows command-line limit
      that RAG prompts routinely exceed;
    * decoding newline-delimited JSON events into answer text, skipping the
      banner and progress lines these tools interleave on stdout;
    * falling back to the terminal event's ``result`` when a CLI reports its
      answer only once at the end.
    """

    def __init__(self, settings: Settings) -> None:
        """Bind the backend and construct its shared event extractor."""
        super().__init__(settings)
        self._extractor = AgentEventTextExtractor()

    @property
    def runs_locally(self) -> bool:
        """CLI agents call a hosted model, so excerpts leave this machine."""
        return False

    @property
    @abstractmethod
    def executable(self) -> str:
        """Path or command name of the CLI to invoke."""

    @property
    def version_arguments(self) -> list[str]:
        """Arguments that make the CLI print its version and exit."""
        return ["--version"]

    @property
    def effort(self) -> str:
        """Reasoning depth this backend runs at, or "" when it has none."""
        return ""

    @abstractmethod
    def build_command(self) -> list[str]:
        """Assemble the flags for one non-interactive invocation, without the prompt."""

    @property
    def sends_prompt_via_stdin(self) -> bool:
        """Whether this CLI reads the prompt from standard input.

        Standard input is preferred wherever it is supported, because RAG
        prompts run to tens of kilobytes and Windows caps a whole command line
        at 32,767 characters. A CLI that only accepts the prompt as an argument
        overrides this and is length-checked instead.
        """
        return True

    #: Ceiling for argument-delivered prompts, leaving room for the other flags.
    MAXIMUM_PROMPT_ARGUMENT_LENGTH = 28000

    def build_invocation(self, prompt: str) -> list[str]:
        """Return the full argument vector, including the prompt when needed."""
        command = self.build_command()
        if self.sends_prompt_via_stdin:
            return command
        if len(prompt) > self.MAXIMUM_PROMPT_ARGUMENT_LENGTH:
            raise RuntimeError(
                f"Prompt terlalu panjang untuk {self.executable} "
                f"({len(prompt)} karakter; batas {self.MAXIMUM_PROMPT_ARGUMENT_LENGTH}). "
                f"Turunkan MAX_CONTEXT_CHARS atau TOP_K."
            )
        return [*command, prompt]

    @property
    def timeout_seconds(self) -> int:
        """How long to wait for a complete answer before giving up."""
        return self._settings.claude_cli_timeout_seconds

    @property
    def effort(self) -> str:
        """Reasoning depth this backend runs at."""
        return self._settings.claude_cli_effort

    @property
    def working_directory(self) -> str | None:
        """Directory to run the CLI in, or None to inherit the server's.

        Subclasses point this at an empty scratch directory when the agent would
        otherwise treat the project folder as a codebase to explore.
        """
        return None

    def check_status(self) -> BackendStatus:
        """Confirm the CLI is installed and responds to a version query."""
        try:
            completed = subprocess.run(
                [self.executable, *self.version_arguments],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
                check=False,
            )
        except FileNotFoundError:
            return BackendStatus(
                is_available=False,
                backend_name=self.name,
                model_name=self.model_name,
                detail=f"Perintah '{self.executable}' tidak ditemukan di PATH",
            )
        except subprocess.TimeoutExpired:
            return BackendStatus(
                is_available=False,
                backend_name=self.name,
                model_name=self.model_name,
                detail="CLI tidak merespons permintaan versi",
            )

        # Several of these CLIs print their version banner to stderr, not stdout.
        reported_version = (completed.stdout or completed.stderr or "").strip()
        first_line = reported_version.splitlines()[0] if reported_version else ""

        if completed.returncode != 0 and not first_line:
            return BackendStatus(
                is_available=False,
                backend_name=self.name,
                model_name=self.model_name,
                detail=f"CLI mengembalikan kode {completed.returncode}",
            )
        return BackendStatus(
            is_available=True,
            backend_name=self.name,
            model_name=self.model_name,
            detail=f"{first_line} — kutipan dokumen dikirim ke penyedia layanan",
            effort=self.effort,
        )

    #: Prepended for every agent CLI. These tools are coding agents by default
    #: and will reach for the filesystem or a shell unless told not to. In
    #: headless mode such a request is auto-denied and the agent then returns
    #: nothing at all, so the instruction is what makes the answer appear.
    NO_TOOL_DIRECTIVE = (
        "PENTING: Jangan gunakan perkakas (tool) apa pun — jangan menjalankan "
        "perintah, membaca berkas, mencari di disk, atau mengakses internet. "
        "Tidak ada perkakas yang tersedia dan permintaan semacam itu akan "
        "ditolak. Seluruh informasi yang Anda perlukan sudah ada pada teks di "
        "bawah ini. Jawab langsung dari teks tersebut.\n\n"
    )

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Stream an answer by running the CLI and decoding its event output."""
        combined_prompt = f"{self.NO_TOOL_DIRECTIVE}{system_prompt}\n\n{user_prompt}"
        try:
            process = subprocess.Popen(
                self.build_invocation(combined_prompt),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.working_directory,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Perintah '{self.executable}' tidak ditemukan. Pastikan CLI sudah terpasang."
            )

        # Killing the process on a timer is what actually bounds the request.
        # Reading stdout blocks indefinitely, so without this a CLI that hangs
        # — waiting on an expired login, say — would hold the HTTP connection
        # open forever. `process.kill` is passed as a bound method so the timer
        # needs no closure of its own.
        watchdog = threading.Timer(self.timeout_seconds, process.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            yield from self._stream_process_output(process, combined_prompt)
        finally:
            watchdog.cancel()
            if process.poll() is None:
                process.kill()

    def _stream_process_output(
        self, process: subprocess.Popen, combined_prompt: str
    ) -> Iterator[str]:
        """Feed the prompt in, then translate CLI events into answer text."""
        self._send_prompt(process, combined_prompt)

        emitted_any_text = False
        final_result_text = ""
        reported_error: str | None = None

        for raw_line in process.stdout:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            try:
                event = json.loads(stripped_line)
            except json.JSONDecodeError:
                # Banner and progress lines are not JSON; they are not answer text.
                continue

            reported_error = reported_error or self._extractor.describe_error(event)
            for fragment in self._extractor.extract(event):
                emitted_any_text = True
                yield fragment
            final_result_text = final_result_text or self._extractor.extract_final_result(event)

        return_code = process.wait()

        # A CLI that reports its answer only in the closing event still works.
        if not emitted_any_text and final_result_text:
            emitted_any_text = True
            yield final_result_text

        if not emitted_any_text:
            error_output = (process.stderr.read() or "").strip()
            raise RuntimeError(
                reported_error
                or f"{self.executable} selesai dengan kode {return_code} tanpa jawaban. "
                f"{error_output[:500]}"
            )

    def _send_prompt(self, process: subprocess.Popen, combined_prompt: str) -> None:
        """Deliver the prompt, then close standard input.

        Input is closed either way: a CLI that took the prompt as an argument
        still waits on standard input for interactive follow-ups, and would
        never terminate if the pipe were left open.
        """
        try:
            if self.sends_prompt_via_stdin:
                process.stdin.write(combined_prompt)
            process.stdin.close()
        except (BrokenPipeError, OSError) as error:
            # The CLI closed its input before the whole prompt was written —
            # almost always a sign it does not read stdin in this mode at all.
            raise RuntimeError(
                f"{self.executable} menutup masukan sebelum prompt selesai dikirim "
                f"({error}). CLI ini kemungkinan tidak membaca prompt dari stdin."
            )


class ClaudeCodeBackend(CommandLineAgentBackend):
    """Generates answers with the Claude Code CLI (``claude``).

    File and shell tools are disabled explicitly: every fact the model may use
    is already in the prompt, and the CLI must not wander into the filesystem —
    least of all into the read-only corpus — looking for more.
    """

    #: Tools disabled so the CLI answers purely from the supplied context.
    DISABLED_TOOLS = (
        "Read", "Write", "Edit", "NotebookEdit", "Bash",
        "Glob", "Grep", "WebFetch", "WebSearch", "Task",
    )

    @property
    def name(self) -> str:
        """Identifier used in configuration and reported to the interface."""
        return "claude_cli"

    @property
    def executable(self) -> str:
        """The configured path to the Claude Code CLI."""
        return self._settings.claude_cli_executable

    @property
    def model_name(self) -> str:
        """The model this backend will run.

        Just the model: reasoning depth is reported separately as
        :attr:`effort`, so the interface can show it as its own badge without
        the two appearing twice in the same line.
        """
        return self._settings.claude_cli_model or "(default Claude Code)"

    @property
    def timeout_seconds(self) -> int:
        """Answer timeout for this backend."""
        return self._settings.claude_cli_timeout_seconds

    def build_command(self) -> list[str]:
        """Assemble the CLI invocation for one non-interactive question."""
        command = [
            self.executable,
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--disallowed-tools", *self.DISABLED_TOOLS,
        ]
        if self._settings.claude_cli_model:
            command.extend(["--model", self._settings.claude_cli_model])
        if self._settings.claude_cli_effort:
            command.extend(["--effort", self._settings.claude_cli_effort])
        return command


class CodexBackend(CommandLineAgentBackend):
    """Generates answers with the OpenAI Codex CLI (``codex exec``).

    Two flags matter for safety and reliability. ``--sandbox read-only`` means
    the agent physically cannot write to disk, which protects the corpus even
    though nothing in the prompt asks it to write. ``--skip-git-repo-check``
    lets Codex run in a directory that is not a git repository, which this
    project is not.

    Codex reports the assistant's answer in a single ``item.completed`` event
    rather than streaming deltas, so answers from this backend appear all at
    once rather than word by word.
    """

    @property
    def name(self) -> str:
        """Identifier used in configuration and reported to the interface."""
        return "codex_cli"

    @property
    def executable(self) -> str:
        """The configured path to the Codex CLI."""
        return self._settings.codex_cli_executable

    @property
    def model_name(self) -> str:
        """The model this backend will run; depth is reported as `effort`."""
        return self._settings.codex_cli_model or "(default Codex)"

    @property
    def timeout_seconds(self) -> int:
        """Answer timeout for this backend."""
        return self._settings.codex_cli_timeout_seconds

    @property
    def effort(self) -> str:
        """Reasoning depth this backend runs at."""
        return self._settings.codex_cli_effort

    def build_command(self) -> list[str]:
        """Assemble the ``codex exec`` invocation, reading the prompt from stdin."""
        command = [
            self.executable,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
        ]
        if self._settings.codex_cli_model:
            command.extend(["--model", self._settings.codex_cli_model])
        if self._settings.codex_cli_effort:
            # Reasoning depth is a config override, not a flag. The value is
            # parsed as TOML, so it is quoted to arrive as a string.
            command.extend(
                ["-c", f'model_reasoning_effort="{self._settings.codex_cli_effort}"']
            )
        # A bare "-" tells Codex to read the prompt from standard input.
        command.append("-")
        return command


class AntigravityBackend(CommandLineAgentBackend):
    """Generates answers with the Antigravity CLI (``agy``).

    Slash-command expansion is switched off because a retrieved passage could
    otherwise begin with a ``/`` and be interpreted as a command rather than as
    document text.
    """

    def __init__(self, settings: Settings) -> None:
        """Bind the backend; the scratch directory is created on first use."""
        super().__init__(settings)
        self._scratch_directory: str | None = None

    @property
    def name(self) -> str:
        """Identifier used in configuration and reported to the interface."""
        return "antigravity_cli"

    @property
    def executable(self) -> str:
        """The configured path to the Antigravity CLI."""
        return self._settings.antigravity_cli_executable

    @property
    def version_arguments(self) -> list[str]:
        """``agy`` reports its version with a bare ``--version``."""
        return ["--version"]

    @property
    def sends_prompt_via_stdin(self) -> bool:
        """False: this CLI carries the prompt on the command line.

        Its ``--input-format text`` mode does not read standard input; only
        ``stream-json`` does. Writing to stdin produces a broken pipe as soon as
        the CLI exits.
        """
        return False

    def build_invocation(self, prompt: str) -> list[str]:
        """Return the argument vector with the prompt attached to ``--print``.

        ``--print`` *takes the prompt as its value* — the help lists ``--prompt``
        as an alias for it — rather than reading a trailing positional argument.
        Passing a bare ``--print`` makes the CLI swallow the next flag as the
        prompt, so it silently answers a question about that flag instead of
        about the documents. The ``--print=<prompt>`` form removes the ambiguity.
        """
        if len(prompt) > self.MAXIMUM_PROMPT_ARGUMENT_LENGTH:
            raise RuntimeError(
                f"Prompt terlalu panjang untuk {self.executable} "
                f"({len(prompt)} karakter; batas {self.MAXIMUM_PROMPT_ARGUMENT_LENGTH}). "
                f"Turunkan MAX_CONTEXT_CHARS atau TOP_K."
            )
        return [f"--print={prompt}" if part == "--print" else part
                for part in self.build_command()]

    #: Tier suffixes this CLI's catalogue appends to its model names.
    EMBEDDED_EFFORT_SUFFIXES = ("-low", "-medium", "-high")

    @property
    def model_name(self) -> str:
        """The model this backend will run, with the effort tier stripped.

        Alone among the three, this catalogue bakes the reasoning tier into the
        model name — ``gemini-3.1-pro-high``. The interface already shows effort
        as its own badge, so leaving the suffix in would state the same fact
        twice. Display only: :meth:`build_command` still sends the full
        catalogue name, which is what the CLI actually accepts.
        """
        model = self._settings.antigravity_cli_model or "(default Antigravity)"
        for suffix in self.EMBEDDED_EFFORT_SUFFIXES:
            if model.endswith(suffix):
                return model[: -len(suffix)]
        return model

    @property
    def timeout_seconds(self) -> int:
        """Answer timeout for this backend."""
        return self._settings.antigravity_cli_timeout_seconds

    @property
    def effort(self) -> str:
        """Reasoning depth this backend runs at."""
        return self._settings.antigravity_cli_effort

    @property
    def working_directory(self) -> str | None:
        """An empty scratch directory, created once per process.

        Run inside the project folder this agent treats the code as a workspace
        and reaches for shell tools, which headless mode auto-denies — leaving
        it to return nothing at all. An empty directory removes the temptation
        and, usefully, means it cannot see the corpus or the project either.
        """
        if self._scratch_directory is None:
            self._scratch_directory = tempfile.mkdtemp(prefix="djp-agy-")
        return self._scratch_directory

    def build_command(self) -> list[str]:
        """Assemble the ``agy --print`` invocation for one question."""
        command = [
            self.executable,
            "--print",
            "--output-format", "stream-json",
            "--input-format", "text",
            "--disable-slash-commands",
        ]
        if self._settings.antigravity_cli_model:
            command.extend(["--model", self._settings.antigravity_cli_model])
        if self._settings.antigravity_cli_effort:
            command.extend(["--effort", self._settings.antigravity_cli_effort])
        return command


class LanguageModelFactory:
    """Creates and caches backend instances by configured name."""

    #: Registry of every backend this system can construct.
    BACKEND_CLASSES = {
        "ollama": OllamaBackend,
        "claude_cli": ClaudeCodeBackend,
        "codex_cli": CodexBackend,
        "antigravity_cli": AntigravityBackend,
    }

    #: Order shown in the interface: local first, then the CLI agents.
    DISPLAY_ORDER = ("ollama", "claude_cli", "codex_cli", "antigravity_cli")

    def __init__(self, settings: Settings) -> None:
        """Bind the factory to configuration and start with an empty cache."""
        self._settings = settings
        self._instances: dict[str, LanguageModelBackend] = {}

    @property
    def available_backend_names(self) -> list[str]:
        """Every backend name this factory knows how to build, in display order."""
        return list(self.DISPLAY_ORDER)

    def create(self, backend_name: str | None = None) -> LanguageModelBackend:
        """Return the backend for ``backend_name``, defaulting to configuration.

        Raises:
            ValueError: when the name matches no registered backend.
        """
        resolved_name = (backend_name or self._settings.language_model_backend).strip().lower()
        if resolved_name not in self.BACKEND_CLASSES:
            raise ValueError(
                f"Backend '{resolved_name}' tidak dikenal. "
                f"Pilihan: {', '.join(self.available_backend_names)}"
            )
        if resolved_name not in self._instances:
            backend_class = self.BACKEND_CLASSES[resolved_name]
            self._instances[resolved_name] = backend_class(self._settings)
        return self._instances[resolved_name]

    def check_all_statuses(self) -> list[dict]:
        """Probe every registered backend, for the interface's status panel."""
        return [
            self.create(backend_name).check_status().as_dictionary()
            for backend_name in self.available_backend_names
        ]
