"""Question answering over the indexed corpus.

:class:`RagEngine` is the orchestration layer the web server talks to. One
question flows through four steps: retrieve candidate passages, assemble them
into a numbered context block, stream an answer from the configured language
model, and hand back the citations that back it up.

The prompt in :class:`PromptBuilder` is written in Indonesian and is deliberately
strict about grounding. Accreditation work is audited, so an answer that quietly
invents a decree number is worse than no answer at all — the model is instructed
to say when the corpus does not contain something, and to quote identifiers
verbatim rather than paraphrase them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

from .config import SETTINGS, Settings
from .conversations import ChatTurn, ConversationStore
from .documents import ExternalLinkExtractor, RetrievedPassage
from .language_models import LanguageModelFactory
from .retrieval import HybridRetriever

LOGGER = logging.getLogger(__name__)


@dataclass
class QuestionRequest:
    """Everything the engine needs to answer one question."""

    question: str
    #: Restricts retrieval to a single top-level folder, or None for the whole corpus.
    folder_filter: str | None = None
    #: Overrides the configured backend for this question only.
    backend_name: str | None = None
    #: Overrides how many passages are placed in the context.
    passage_count: int | None = None
    #: Conversation to continue and record this exchange in, or None for a
    #: one-off question that is not saved.
    session_id: str | None = None


@dataclass
class StreamEvent:
    """One message in the answer stream, delivered to the browser as SSE.

    ``name`` is the event type the browser listens for: ``sources`` carries the
    citations, ``token`` carries answer text, ``done`` closes the stream and
    ``error`` reports a failure the interface should surface.
    """

    name: str
    payload: dict


@dataclass
class AnswerCitation:
    """A passage shown beneath the answer as supporting evidence."""

    citation_number: int
    document_name: str
    relative_path: str
    document_id: str
    section_label: str
    excerpt: str
    produced_by_ocr: bool
    retrieval_methods: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    #: Live pages this passage points at — a SINTA or Scholar profile, say —
    #: so a reader can go to the source itself rather than the local copy.
    external_links: list[dict] = field(default_factory=list)

    def as_dictionary(self) -> dict:
        """Represent the citation as JSON-serialisable data."""
        return {
            "citation_number": self.citation_number,
            "document_name": self.document_name,
            "relative_path": self.relative_path,
            "document_id": self.document_id,
            "section_label": self.section_label,
            "excerpt": self.excerpt,
            "produced_by_ocr": self.produced_by_ocr,
            "retrieval_methods": self.retrieval_methods,
            "relevance_score": round(self.relevance_score, 4),
            "external_links": self.external_links,
        }


class PromptBuilder:
    """Assembles the system and user prompts for one grounded question.

    The context block is numbered so that a citation marker in the answer maps
    directly onto a passage card in the interface, and truncated to a character
    budget so a long retrieval can never overflow the model's context window.
    """

    SYSTEM_PROMPT = (
        "Anda adalah asisten informasi perpajakan Indonesia. Sumber Anda adalah "
        "salinan halaman dan dokumen publik situs Direktorat Jenderal Pajak "
        "(pajak.go.id): peraturan, materi Belajar Pajak, FAQ, artikel, siaran pers, "
        "formulir, serta profil dan layanan DJP. Anda bukan layanan resmi DJP.\n"
        "\n"
        "ATURAN MENJAWAB:\n"
        "1. Jawab HANYA berdasarkan KONTEKS yang diberikan. Jangan menambahkan "
        "pengetahuan dari luar konteks tersebut.\n"
        "2. Sertakan sitasi [n] pada setiap pernyataan faktual, merujuk nomor "
        "sumber di KONTEKS. Boleh lebih dari satu, misalnya [1][3].\n"
        "3. Jika KONTEKS tidak memuat jawabannya, nyatakan dengan jelas: "
        "\"Informasi tersebut tidak ditemukan pada dokumen yang terindeks.\" "
        "Kemudian sarankan di mana informasi itu mungkin tersedia, misalnya "
        "Kring Pajak 1500200 atau kantor pelayanan pajak.\n"
        "4. Untuk tarif, angka, tanggal, nomor peraturan, pasal, dan ayat: kutip "
        "PERSIS seperti tertulis pada sumber. Jangan membulatkan, menerjemahkan, "
        "atau menyingkat. Sebutkan nomor peraturan yang menjadi dasarnya.\n"
        "5. Gunakan Bahasa Indonesia yang formal dan ringkas. Jika pertanyaan "
        "diajukan dalam bahasa lain, jawab dalam bahasa pertanyaan tersebut.\n"
        "6. Bila sumber-sumber saling bertentangan, tunjukkan perbedaannya "
        "beserta sitasi masing-masing, jangan memilih salah satu secara diam-diam.\n"
        "7. Sumber bertanda (OCR) berasal dari pemindaian dan mungkin mengandung "
        "salah baca; beri catatan bila Anda mengutip angka dari sumber tersebut.\n"
        "8. Bila diminta alamat, tautan, URL, formulir, atau halaman layanan, "
        "tuliskan URL LENGKAP persis seperti pada KONTEKS — jangan disingkat, "
        "jangan diringkas menjadi nama situs saja, dan jangan direka. Tulis URL "
        "apa adanya agar dapat langsung diklik.\n"
        "9. Bila peraturan-peraturan di KONTEKS saling mengubah atau mencabut, "
        "utamakan yang terbaru dan sebutkan bahwa ketentuan lama telah diubah. "
        "Untuk keputusan atas kasus pribadi, sarankan konfirmasi ke kantor pajak."
    )

    #: Shown when retrieval returns nothing, instead of calling the model.
    EMPTY_RESULT_MESSAGE = (
        "Informasi tersebut tidak ditemukan pada dokumen yang terindeks.\n\n"
        "Saran:\n"
        "- Periksa ejaan atau gunakan istilah lain (misalnya singkatan resmi seperti "
        "PPh, PPN, PTKP, NPWP, SPT).\n"
        "- Hapus filter folder jika sedang aktif.\n"
        "- Pastikan proses indexing sudah dijalankan dan selesai."
    )

    def __init__(self, settings: Settings) -> None:
        """Bind the builder to the configured context budget."""
        self._settings = settings

    def build_system_prompt(self) -> str:
        """Return the grounding instructions given to every backend."""
        return self.SYSTEM_PROMPT

    def build_user_prompt(
        self,
        question: str,
        passages: list[RetrievedPassage],
        history: list[ChatTurn] | None = None,
    ) -> str:
        """Return the conversation so far, the numbered context, and the question.

        History comes first and is explicitly labelled as *not* a source, so the
        model can resolve a pronoun against it without treating an earlier
        answer as evidence — otherwise a mistake in turn one would be cited as
        fact in turn three.
        """
        sections: list[str] = []
        history_block = self._render_history(history or [])
        if history_block:
            sections.append(
                "RIWAYAT PERCAKAPAN (untuk memahami rujukan saja, BUKAN sumber fakta)\n"
                "===================================================================\n"
                f"{history_block}"
            )
        sections.append(
            "KONTEKS\n"
            "=======\n"
            f"{self._render_context(passages)}"
        )
        sections.append(
            "PERTANYAAN\n"
            "==========\n"
            f"{question}\n\n"
            "Jawab sesuai aturan di atas, dengan sitasi [n]."
        )
        return "\n\n".join(sections)

    def _render_history(self, history: list[ChatTurn]) -> str:
        """Render prior turns newest-last, within the history character budget."""
        if not history:
            return ""

        rendered: list[str] = []
        remaining = self._settings.maximum_history_characters
        # Walk backwards so that, when the budget runs out, it is the oldest
        # turns that are dropped rather than the most relevant recent ones.
        for turn in reversed(history):
            speaker = "Pengguna" if turn.is_question else "Asisten"
            body = " ".join(turn.content.split())
            if len(body) > 400:
                body = body[:400] + " ..."
            entry = f"{speaker}: {body}"
            if len(entry) > remaining:
                break
            rendered.insert(0, entry)
            remaining -= len(entry)
        return "\n".join(rendered)

    def _render_context(self, passages: list[RetrievedPassage]) -> str:
        """Render passages as numbered blocks within the character budget."""
        rendered_blocks: list[str] = []
        remaining_characters = self._settings.maximum_context_characters

        for passage in passages:
            block = self._render_passage(passage)
            if len(block) > remaining_characters:
                # Include a truncated tail of this passage rather than nothing,
                # unless there is no meaningful room left at all.
                if remaining_characters > 400:
                    rendered_blocks.append(block[:remaining_characters] + "\n[...dipotong...]")
                break
            rendered_blocks.append(block)
            remaining_characters -= len(block)

        return "\n\n".join(rendered_blocks)

    def _render_passage(self, passage: RetrievedPassage) -> str:
        """Render one passage with its citation number and provenance header."""
        ocr_marker = " (OCR)" if passage.produced_by_ocr else ""
        return (
            f"[{passage.citation_number}] {passage.document_name} — "
            f"{passage.section_label}{ocr_marker}\n"
            f"Lokasi: {passage.relative_path}\n"
            f"{passage.text}"
        )


class FollowUpQueryExpander:
    """Rewrites a short follow-up into something worth searching for.

    Retrieval sees only the current question, so a follow-up such as *"siapa
    saja mereka?"* carries almost no searchable signal and would return
    arbitrary passages. Prefixing the previous question restores the subject.

    The rule is deliberately conservative: it fires only on very short
    questions, because a fully-formed question is better retrieved on its own
    terms than diluted with an earlier, different topic.
    """

    def __init__(self, settings: Settings) -> None:
        """Bind the expander to the configured follow-up length threshold."""
        self._settings = settings

    def expand(self, question: str, history: list[ChatTurn]) -> str:
        """Return the query to search with, expanded only when warranted."""
        if not history:
            return question
        if len(question.split()) >= self._settings.follow_up_word_threshold:
            return question

        previous_question = self._find_previous_question(history)
        if not previous_question:
            return question
        return f"{previous_question} {question}"

    def _find_previous_question(self, history: list[ChatTurn]) -> str:
        """Return the most recent user question in the history, if any."""
        for turn in reversed(history):
            if turn.is_question and turn.content.strip():
                return " ".join(turn.content.split())
        return ""


class RagEngine:
    """Answers questions against the corpus, streaming results as they arrive.

    Citations are emitted *before* the first answer token, so the interface can
    render the evidence panel while the model is still writing. That ordering
    also means a user can start checking sources during generation, which is the
    slowest part of the request on a local model.
    """

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        model_factory: LanguageModelFactory,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        """Wire the engine to retrieval, the backend factory and chat storage.

        ``conversation_store`` is optional so the engine can be exercised — and
        used for one-off command-line questions — without persisting anything.
        """
        self._settings = settings
        self._retriever = retriever
        self._model_factory = model_factory
        self._conversation_store = conversation_store
        self._prompt_builder = PromptBuilder(settings)
        self._query_expander = FollowUpQueryExpander(settings)
        self._link_extractor = ExternalLinkExtractor()

    def stream_answer(self, request: QuestionRequest) -> Iterator[StreamEvent]:
        """Answer one question, yielding stream events in display order.

        The generator never raises into the web layer: a failure becomes an
        ``error`` event followed by ``done``, so the browser always sees a
        terminated stream and can re-enable its input box.
        """
        started_at = time.time()
        history = self._load_history(request.session_id)
        self._record_question(request)

        search_query = self._query_expander.expand(request.question, history)
        if search_query != request.question:
            LOGGER.info("Pertanyaan lanjutan diperluas untuk pencarian: %r", search_query)

        try:
            passages = self._retriever.retrieve(
                search_query, request.folder_filter, request.passage_count
            )
        except Exception as error:
            LOGGER.exception("Pencarian gagal")
            yield StreamEvent("error", {"message": f"Pencarian gagal: {error}"})
            yield StreamEvent("done", {"elapsed_seconds": round(time.time() - started_at, 2)})
            return

        citations = self._build_citations(passages)
        citation_payload = [item.as_dictionary() for item in citations]
        yield StreamEvent("sources", {"citations": citation_payload})

        if not passages:
            message = self._prompt_builder.EMPTY_RESULT_MESSAGE
            self._record_answer(request, message, [], "", "", time.time() - started_at)
            yield StreamEvent("token", {"text": message})
            yield StreamEvent(
                "done",
                {
                    "elapsed_seconds": round(time.time() - started_at, 2),
                    "passage_count": 0,
                    "backend_name": request.backend_name or self._settings.language_model_backend,
                    "session_id": request.session_id,
                },
            )
            return

        yield from self._stream_generated_answer(
            request, passages, citation_payload, history, started_at
        )

    def _stream_generated_answer(
        self,
        request: QuestionRequest,
        passages: list[RetrievedPassage],
        citation_payload: list[dict],
        history: list[ChatTurn],
        started_at: float,
    ) -> Iterator[StreamEvent]:
        """Run the language model, relay its output, and store the answer."""
        try:
            backend = self._model_factory.create(request.backend_name)
        except ValueError as error:
            yield StreamEvent("error", {"message": str(error)})
            yield StreamEvent("done", {"elapsed_seconds": round(time.time() - started_at, 2)})
            return

        system_prompt = self._prompt_builder.build_system_prompt()
        user_prompt = self._prompt_builder.build_user_prompt(
            request.question, passages, history
        )
        yield StreamEvent(
            "status",
            {"message": f"Menyusun jawaban dengan {backend.name} ({backend.model_name})..."},
        )

        token_count = 0
        answer_parts: list[str] = []
        try:
            for text_fragment in backend.generate_stream(system_prompt, user_prompt):
                if not text_fragment:
                    continue
                token_count += 1
                answer_parts.append(text_fragment)
                yield StreamEvent("token", {"text": text_fragment})
        except Exception as error:
            LOGGER.exception("Backend model bahasa gagal")
            yield StreamEvent(
                "error",
                {"message": f"Model bahasa gagal ({backend.name}): {error}"},
            )

        elapsed = time.time() - started_at
        # Stored even when generation failed part-way: a partial answer with its
        # sources is more use on resuming than a gap in the transcript.
        self._record_answer(
            request,
            "".join(answer_parts),
            citation_payload,
            backend.name,
            backend.model_name,
            elapsed,
        )

        yield StreamEvent(
            "done",
            {
                "elapsed_seconds": round(elapsed, 2),
                "passage_count": len(passages),
                "fragment_count": token_count,
                "backend_name": backend.name,
                "model_name": backend.model_name,
                "effort": getattr(backend, "effort", ""),
                "session_id": request.session_id,
            },
        )

    # --- Conversation persistence -------------------------------------------

    def _load_history(self, session_id: str | None) -> list[ChatTurn]:
        """Return prior turns of the session, or an empty list when unsaved."""
        if not session_id or self._conversation_store is None:
            return []
        return self._conversation_store.recent_turns(
            session_id, self._settings.conversation_history_turns
        )

    def _record_question(self, request: QuestionRequest) -> None:
        """Store the user's question and title the session if it is new."""
        if not request.session_id or self._conversation_store is None:
            return
        self._conversation_store.append_turn(
            request.session_id, ChatTurn(role="user", content=request.question)
        )
        self._conversation_store.ensure_titled(request.session_id, request.question)

    def _record_answer(
        self,
        request: QuestionRequest,
        answer_text: str,
        citations: list[dict],
        backend_name: str,
        model_name: str,
        elapsed_seconds: float,
    ) -> None:
        """Store the assistant's answer together with the sources behind it."""
        if not request.session_id or self._conversation_store is None:
            return
        self._conversation_store.append_turn(
            request.session_id,
            ChatTurn(
                role="assistant",
                content=answer_text,
                citations=citations,
                backend_name=backend_name,
                model_name=model_name,
                elapsed_seconds=elapsed_seconds,
            ),
        )

    def _build_citations(self, passages: list[RetrievedPassage]) -> list[AnswerCitation]:
        """Convert retrieved passages into the citation records shown in the UI."""
        return [
            AnswerCitation(
                citation_number=passage.citation_number,
                document_name=passage.document_name,
                relative_path=passage.relative_path,
                document_id=passage.document_id,
                section_label=passage.section_label,
                excerpt=self._build_excerpt(passage.text),
                produced_by_ocr=passage.produced_by_ocr,
                retrieval_methods=passage.retrieval_methods,
                relevance_score=passage.ranking_score,
                # Scanned from the full passage, not the shortened excerpt, so a
                # link is still offered when it falls outside the visible part.
                external_links=self._link_extractor.extract(passage.text),
            )
            for passage in passages
        ]

    def _build_excerpt(self, text: str, maximum_characters: int = 600) -> str:
        """Shorten passage text for the evidence card without cutting mid-word."""
        if len(text) <= maximum_characters:
            return text
        truncated = text[:maximum_characters]
        last_space = truncated.rfind(" ")
        if last_space > maximum_characters * 0.6:
            truncated = truncated[:last_space]
        return truncated + " ..."


#: Shared factory used by the server and the engine.
LANGUAGE_MODEL_FACTORY = LanguageModelFactory(SETTINGS)
