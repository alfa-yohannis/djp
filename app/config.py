"""Configuration objects for the DJP RAG system.

This module exposes two classes:

* :class:`EnvironmentReader` — converts raw environment variables into typed
  Python values, applying a fallback whenever a variable is missing or blank.
* :class:`Settings` — the single, fully typed configuration object that every
  other module reads from.

A ready-to-use instance is published as :data:`SETTINGS`. Overrides come from
the process environment or from a ``.env`` file in the project root; see
``.env.example`` for the full list of recognised names.

One setting is deliberately absent: there is no switch that permits writing to
the source corpus. That prohibition is enforced in :mod:`app.safety`.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class EnvironmentReader:
    """Reads typed configuration values from the process environment.

    A variable counts as "absent" when it is unset *or* set to an empty or
    whitespace-only string, so a blank entry in ``.env`` falls back to the
    default rather than producing an empty setting.
    """

    #: Values accepted as boolean true, compared case-insensitively.
    TRUTHY_WORDS = frozenset({"1", "true", "yes", "on", "enabled"})

    def __init__(self, env_file: Path | None = None) -> None:
        """Load ``env_file`` into the environment if python-dotenv is present."""
        if env_file is not None and load_dotenv is not None and env_file.exists():
            load_dotenv(env_file)

    def text(self, name: str, fallback: str) -> str:
        """Return the variable as a stripped string, or ``fallback``."""
        raw_value = os.getenv(name)
        if raw_value is None or not raw_value.strip():
            return fallback
        return raw_value.strip()

    def path(self, name: str, fallback: Path | str) -> Path:
        """Return the variable as a filesystem path, or ``fallback``."""
        return Path(self.text(name, str(fallback)))

    def text_list(self, name: str, fallback: list[str]) -> list[str]:
        """Return a semicolon-separated variable as a list of strings.

        Semicolon rather than comma because the values are often URLs, and a
        query string routinely contains commas.
        """
        raw_value = os.getenv(name)
        if raw_value is None or not raw_value.strip():
            return list(fallback)
        return [part.strip() for part in raw_value.split(";") if part.strip()]

    def path_list(self, name: str, fallback: list[Path]) -> list[Path]:
        """Return a semicolon-separated variable as a list of paths.

        Semicolon is the separator rather than comma because Windows paths
        routinely contain commas in folder names but never semicolons.
        """
        raw_value = os.getenv(name)
        if raw_value is None or not raw_value.strip():
            return list(fallback)
        return [Path(part.strip()) for part in raw_value.split(";") if part.strip()]

    def integer(self, name: str, fallback: int) -> int:
        """Return the variable as an int; malformed values fall back silently."""
        try:
            return int(self.text(name, str(fallback)))
        except ValueError:
            return fallback

    def decimal(self, name: str, fallback: float) -> float:
        """Return the variable as a float; malformed values fall back silently."""
        try:
            return float(self.text(name, str(fallback)))
        except ValueError:
            return fallback

    def flag(self, name: str, fallback: bool) -> bool:
        """Return the variable as a bool, recognising 1/true/yes/on/enabled."""
        default_word = "true" if fallback else "false"
        return self.text(name, default_word).lower() in self.TRUTHY_WORDS

    def lowercase_set(self, name: str, fallback: set[str]) -> set[str]:
        """Return a comma-separated variable as a set of lowercase tokens."""
        raw_value = os.getenv(name)
        if raw_value is None or not raw_value.strip():
            return set(fallback)
        return {token.strip().lower() for token in raw_value.split(",") if token.strip()}


class Settings:
    """Every tunable parameter of the system, resolved once at import time.

    Grouped by concern: corpus location, corpus filtering, chunking, embedding,
    reranking, retrieval, OCR, language model, and web server. Derived paths are
    exposed as read-only properties so that changing :attr:`data_directory`
    relocates the whole generated-artefact tree consistently.
    """

    #: Directory names skipped anywhere in the corpus tree.
    DEFAULT_EXCLUDED_DIRECTORIES = {
        "zdev",                # developer scratch scripts, not accreditation content
        "__pycache__",
        ".git",
        ".idea",
        ".vscode",
        "node_modules",
        ".ipynb_checkpoints",
        "venv",
        ".venv",
    }

    #: File extensions this system knows how to turn into text.
    DEFAULT_TEXT_EXTENSIONS = {
        ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx",
        ".txt", ".md", ".csv", ".html", ".htm", ".json", ".py",
    }

    #: Extensions indexed only when OCR is switched on.
    DEFAULT_IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
    }

    #: Extensions never indexed — media, archives, binaries, Drive pointers.
    DEFAULT_SKIPPED_EXTENSIONS = {
        ".mov", ".mp4", ".avi", ".mkv", ".wmv", ".mp3", ".wav", ".m4a",
        ".lnk", ".ini", ".pyc", ".zip", ".rar", ".7z", ".exe", ".dll", ".msi",
        ".gsheet", ".gdoc", ".gslides", ".gdraw", ".gform", ".gjam", ".gnote",
        ".db", ".sqlite", ".log", ".tmp", ".bak", ".crdownload", ".sh",
    }

    def __init__(self, reader: EnvironmentReader | None = None) -> None:
        """Resolve all settings, reading overrides through ``reader``."""
        if reader is None:
            reader = EnvironmentReader(PROJECT_ROOT / ".env")
        self._reader = reader

        # --- Corpus location ------------------------------------------------
        #: The document corpus: snapshots of pajak.go.id written by
        #: `main.py crawl`. Ingest and the server only ever read it.
        self.source_directory: Path = reader.path(
            "SOURCE_DIR", PROJECT_ROOT / "corpus-pajak"
        )
        #: Extra directories indexed alongside the corpus, separated by ";".
        #: A missing extra root is skipped, not an error.
        self.additional_source_directories: list[Path] = reader.path_list(
            "EXTRA_SOURCE_DIRS", []
        )
        #: Root for every artefact this system generates. Must sit outside the corpus.
        self.data_directory: Path = reader.path("DATA_DIR", PROJECT_ROOT / "data")

        # --- pajak.go.id crawl -----------------------------------------------
        #: Courtesy pause between requests to pajak.go.id.
        self.pajak_crawl_delay_seconds: float = reader.decimal("PAJAK_CRAWL_DELAY", 1.0)
        self.pajak_crawl_timeout_seconds: int = reader.integer("PAJAK_CRAWL_TIMEOUT", 45)
        #: Attachments larger than this are not downloaded.
        self.pajak_attachment_maximum_megabytes: float = reader.decimal(
            "PAJAK_ATTACHMENT_MAX_MB", 30.0
        )

        # --- External page snapshots ----------------------------------------
        #: Where snapshots of SINTA / Scholar pages are written. Sits inside
        #: `corpus-lokal` so the ordinary ingest picks them up automatically.
        self.web_snapshot_directory: Path = reader.path(
            "WEB_SNAPSHOT_DIR", PROJECT_ROOT / "corpus-lokal" / "eksternal"
        )
        self.web_snapshot_timeout_seconds: int = reader.integer("WEB_SNAPSHOT_TIMEOUT", 45)

        # --- Browser-driven sources -----------------------------------------
        # PDDikti sits behind a Cloudflare check that rejects any browser
        # Playwright launches itself, so Chrome is started as an ordinary
        # process and attached to over the DevTools protocol instead.
        self.browser_executable: Path = reader.path(
            "BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        )
        self.browser_debug_port: int = reader.integer("BROWSER_DEBUG_PORT", 9222)
        #: Chrome refuses remote debugging on its default profile directory, so
        #: the cookie-bearing files are copied here first.
        self.browser_profile_source: Path = reader.path(
            "BROWSER_PROFILE_SOURCE",
            Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data",
        )
        self.browser_profile_working_copy: Path = reader.path(
            "BROWSER_PROFILE_COPY", PROJECT_ROOT / "data" / "browser-profile"
        )
        self.browser_timeout_milliseconds: int = reader.integer("BROWSER_TIMEOUT_MS", 90000)
        self.browser_settle_milliseconds: int = reader.integer("BROWSER_SETTLE_MS", 2500)
        #: How long to wait for the bot-check interstitial to hand over.
        self.browser_bot_check_seconds: int = reader.integer("BROWSER_BOT_CHECK_SECONDS", 60)
        #: Upper bound on paginated result pages, per listing.
        self.browser_maximum_pages: int = reader.integer("BROWSER_MAX_PAGES", 15)
        #: Upper bound on detail pages followed from one listing. The lecturer
        #: directory alone runs to roughly eighty pages.
        self.browser_maximum_detail_pages: int = reader.integer("BROWSER_MAX_DETAILS", 120)

        #: Pages fetched through the browser regardless of the corpus, separated
        #: by ";". These are institution-level sources with no document in the
        #: corpus to harvest them from.
        #:
        #: Each entry is either a bare URL, or ``url::regex`` where the regex
        #: selects which links to follow from that page. The pattern is per-seed
        #: because the sites differ: PDDikti hides its substance behind
        #: ``/detail-…/`` pages, while the university site lists one page per
        #: lecturer under ``/praktisi-akademisi/dosen/``. A single shared
        #: pattern would silently follow nothing on one of them.
        self.browser_seed_urls: list[str] = reader.text_list(
            "BROWSER_SEED_URLS",
            [
                "https://pddikti.kemdiktisaintek.go.id/search/Universitas%20Pradita"
                "::/detail-(dosen|prodi|pt|mahasiswa)/",
                "https://www.pradita.ac.id/praktisi-akademisi"
                "::/praktisi-akademisi/dosen/",
                "https://www.pradita.ac.id/",
            ],
        )
        #: Fallback pattern for a seed that does not carry its own.
        self.browser_detail_link_pattern: str = reader.text(
            "BROWSER_DETAIL_PATTERN", r"/detail-(dosen|prodi|pt|mahasiswa)/"
        )

        # --- SINTA sign-in --------------------------------------------------
        #: Signing in is what makes pages beyond the first of a SINTA
        #: publication list visible. Read from .env; never written anywhere.
        self.sinta_username: str = reader.text("SINTA_USER", "")
        self.sinta_password: str = reader.text("SINTA_PASS", "")
        self.sinta_login_url: str = reader.text(
            "SINTA_LOGIN_URL", "https://sinta.kemdiktisaintek.go.id/logins"
        )
        #: Courtesy pause between requests; Scholar refuses back-to-back traffic.
        self.web_snapshot_delay_seconds: float = reader.decimal("WEB_SNAPSHOT_DELAY", 3.0)
        #: A page yielding less text than this was probably blocked or needs login.
        self.web_snapshot_minimum_characters: int = reader.integer(
            "WEB_SNAPSHOT_MIN_CHARS", 400
        )

        # --- Corpus filtering -----------------------------------------------
        self.excluded_directories: set[str] = reader.lowercase_set(
            "EXCLUDE_DIRS", self.DEFAULT_EXCLUDED_DIRECTORIES
        )
        self.text_extensions: set[str] = reader.lowercase_set(
            "TEXT_EXTENSIONS", self.DEFAULT_TEXT_EXTENSIONS
        )
        self.image_extensions: set[str] = reader.lowercase_set(
            "IMAGE_EXTENSIONS", self.DEFAULT_IMAGE_EXTENSIONS
        )
        self.skipped_extensions: set[str] = reader.lowercase_set(
            "SKIP_EXTENSIONS", self.DEFAULT_SKIPPED_EXTENSIONS
        )
        #: Files above this size are skipped, guarding against runaway extraction.
        self.maximum_file_megabytes: float = reader.decimal("MAX_FILE_MB", 200.0)
        #: While the server runs, check the corpus this often and index whatever
        #: the crawler has added since the last run. 0 turns it off.
        self.auto_ingest_seconds: int = reader.integer("AUTO_INGEST_SECONDS", 60)

        # --- Chunking -------------------------------------------------------
        #: Target chunk length, measured with the embedding model's tokenizer.
        self.chunk_token_count: int = reader.integer("CHUNK_TOKENS", 450)
        #: Tokens repeated between neighbouring chunks so facts are not split.
        self.chunk_overlap_token_count: int = reader.integer("CHUNK_OVERLAP_TOKENS", 80)
        #: Chunks shorter than this are dropped as noise (page numbers, headers).
        self.minimum_chunk_characters: int = reader.integer("MIN_CHUNK_CHARS", 80)

        # --- Embedding ------------------------------------------------------
        #: Multilingual embedding model; BGE-M3 handles Indonesian well.
        self.embedding_model_name: str = reader.text("EMBED_MODEL", "BAAI/bge-m3")
        #: "auto" resolves to cuda when a GPU is present, else cpu.
        self.embedding_device: str = reader.text("EMBED_DEVICE", "auto")
        self.embedding_batch_size: int = reader.integer("EMBED_BATCH_SIZE", 8)
        #: Output width of the embedding model; BGE-M3 produces 1024 dimensions.
        self.embedding_dimensions: int = reader.integer("EMBED_DIM", 1024)

        # --- Reranking ------------------------------------------------------
        self.reranking_enabled: bool = reader.flag("RERANK_ENABLED", True)
        self.reranking_model_name: str = reader.text(
            "RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        #: Defaults to CPU so the 8 GB GPU stays reserved for the language model.
        self.reranking_device: str = reader.text("RERANK_DEVICE", "cpu")

        # --- Retrieval ------------------------------------------------------
        #: Candidates pulled from the vector index before fusion.
        self.dense_candidate_count: int = reader.integer("DENSE_CANDIDATES", 40)
        #: Candidates pulled from the keyword index before fusion.
        self.sparse_candidate_count: int = reader.integer("SPARSE_CANDIDATES", 40)
        #: How many fused candidates the reranker scores.
        self.reranking_candidate_count: int = reader.integer("RERANK_CANDIDATES", 24)
        #: Passages finally shown to the language model and cited in the answer.
        self.answer_passage_count: int = reader.integer("TOP_K", 6)
        #: Reciprocal-rank-fusion smoothing constant; 60 is the published default.
        self.fusion_constant: int = reader.integer("RRF_K", 60)
        #: Hard ceiling on context characters, protecting the model's window.
        self.maximum_context_characters: int = reader.integer("MAX_CONTEXT_CHARS", 14000)

        # --- Conversations --------------------------------------------------
        #: Prior turns replayed to the model so follow-up questions make sense.
        #: Four covers two exchanges, enough for pronouns without crowding out
        #: the retrieved passages.
        self.conversation_history_turns: int = reader.integer("HISTORY_TURNS", 4)
        #: Characters of history allowed in the prompt before older turns are cut.
        self.maximum_history_characters: int = reader.integer("MAX_HISTORY_CHARS", 2500)
        #: A follow-up shorter than this many words is expanded with the previous
        #: question before retrieval, so "siapa saja mereka?" still finds documents.
        self.follow_up_word_threshold: int = reader.integer("FOLLOW_UP_WORDS", 6)

        # --- Optical character recognition ----------------------------------
        self.ocr_enabled: bool = reader.flag("OCR_ENABLED", False)
        #: A PDF page yielding fewer characters than this is treated as scanned.
        self.ocr_page_character_threshold: int = reader.integer("OCR_PAGE_CHAR_THRESHOLD", 60)
        self.ocr_render_dpi: int = reader.integer("OCR_DPI", 200)
        #: Safety valve so one huge scanned PDF cannot stall an entire ingest.
        self.ocr_maximum_pages_per_document: int = reader.integer("OCR_MAX_PAGES_PER_DOC", 80)

        # --- Language model -------------------------------------------------
        #: One of "ollama" (fully local), "claude_cli", "codex_cli",
        #: "antigravity_cli". Only "ollama" keeps document text on this machine.
        self.language_model_backend: str = reader.text("LLM_BACKEND", "ollama")
        self.ollama_base_url: str = reader.text("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.ollama_model_name: str = reader.text("OLLAMA_MODEL", "qwen3:8b")
        #: Context window requested from Ollama; 8192 fits comfortably in 8 GB VRAM.
        self.ollama_context_window: int = reader.integer("OLLAMA_NUM_CTX", 8192)
        #: How long Ollama keeps the model resident between questions.
        self.ollama_keep_alive: str = reader.text("OLLAMA_KEEP_ALIVE", "10m")
        #: Low temperature keeps grounded answers close to the retrieved text.
        self.generation_temperature: float = reader.decimal("TEMPERATURE", 0.2)
        self.maximum_output_tokens: int = reader.integer("MAX_OUTPUT_TOKENS", 1536)

        # Agent CLIs. A blank model setting means "whatever that CLI is already
        # configured to use", which is the sensible default for a tool the user
        # has already signed in to and tuned.
        # Each CLI is pinned to its strongest model at its highest reasoning
        # setting. Accreditation answers are audited, so depth matters more than
        # latency; the timeouts below are widened to match. Model names were
        # taken from each CLI's own catalogue, not assumed.
        self.claude_cli_executable: str = reader.text("CLAUDE_CLI_PATH", "claude")
        self.claude_cli_model: str = reader.text("CLAUDE_CLI_MODEL", "claude-opus-5")
        #: Accepts low | medium | high | xhigh | max.
        self.claude_cli_effort: str = reader.text("CLAUDE_CLI_EFFORT", "max")
        self.claude_cli_timeout_seconds: int = reader.integer("CLAUDE_CLI_TIMEOUT", 900)

        self.codex_cli_executable: str = reader.text("CODEX_CLI_PATH", "codex")
        self.codex_cli_model: str = reader.text("CODEX_CLI_MODEL", "gpt-5.6-sol")
        #: "max" is the deepest setting; "ultra" also delegates to sub-agents,
        #: which is unnecessary for answering from a supplied context.
        self.codex_cli_effort: str = reader.text("CODEX_CLI_EFFORT", "max")
        self.codex_cli_timeout_seconds: int = reader.integer("CODEX_CLI_TIMEOUT", 900)

        self.antigravity_cli_executable: str = reader.text("ANTIGRAVITY_CLI_PATH", "agy")
        #: Its catalogue embeds the effort tier in the model name.
        self.antigravity_cli_model: str = reader.text(
            "ANTIGRAVITY_CLI_MODEL", "gemini-3.1-pro-high"
        )
        #: This CLI accepts only low | medium | high.
        self.antigravity_cli_effort: str = reader.text("ANTIGRAVITY_CLI_EFFORT", "high")
        self.antigravity_cli_timeout_seconds: int = reader.integer("ANTIGRAVITY_CLI_TIMEOUT", 900)

        # --- Web server -----------------------------------------------------
        #: Loopback by default; the corpus is internal material.
        self.server_host: str = reader.text("HOST", "127.0.0.1")
        self.server_port: int = reader.integer("PORT", 8028)
        #: Lets the browser ask the local server to open a cited file natively.
        self.file_opening_enabled: bool = reader.flag("ALLOW_OPEN_FILE", True)

        # --- Sign-in ----------------------------------------------------------
        # One account per role. Passwords are stored only as PBKDF2 hashes;
        # generate a new one with `python main.py hash-password`.
        self.admin_username: str = reader.text("ADMIN_USERNAME", "admin")
        self.admin_password_hash: str = reader.text(
            "ADMIN_PASSWORD_HASH",
            "pbkdf2_sha256$600000$ec9fc80d6e7d1eb87c4f30113a757425"
            "$e746f7893455bc15a14bf69650e19969d1d8af1ad1f01aa9f0fd82ecbb558053",
        )
        self.guest_username: str = reader.text("GUEST_USERNAME", "guest")
        self.guest_password_hash: str = reader.text(
            "GUEST_PASSWORD_HASH",
            "pbkdf2_sha256$600000$4ee3b06adeedf63cb957c26f9b2b1b4d"
            "$bcbd20e52e300d33c039333e971d7542b74a5bbdecbc618c21feb55093d9a8e1",
        )
        #: Key that signs session cookies. Blank means a random key persisted
        #: under the data directory, which is right for a single server.
        self.session_secret: str = reader.text("SESSION_SECRET", "")
        #: How long a sign-in lasts.
        self.session_lifetime_hours: int = reader.integer("SESSION_HOURS", 12)

    # --- Derived paths ------------------------------------------------------

    @property
    def database_path(self) -> Path:
        """SQLite file holding document metadata, chunk text and the keyword index."""
        return self.data_directory / "index.db"

    @property
    def vector_store_path(self) -> Path:
        """Compressed NumPy archive holding the normalised chunk embeddings."""
        return self.data_directory / "vectors.npz"

    @property
    def model_cache_directory(self) -> Path:
        """Local cache for downloaded embedding and reranking models."""
        return self.data_directory / "models"

    @property
    def web_directory(self) -> Path:
        """Static assets served as the browser interface."""
        return PROJECT_ROOT / "web"

    def indexable_extensions(self) -> set[str]:
        """Extensions eligible for indexing under the current OCR setting."""
        eligible = set(self.text_extensions)
        if self.ocr_enabled:
            eligible |= self.image_extensions
        return eligible

    def create_directories(self) -> None:
        """Create the generated-artefact directories if they do not exist yet."""
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.model_cache_directory.mkdir(parents=True, exist_ok=True)

    def redirect_model_cache(self) -> None:
        """Point HuggingFace downloads at the project cache, not the user profile."""
        os.environ.setdefault("HF_HOME", str(self.model_cache_directory))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(self.model_cache_directory))


#: Process-wide configuration instance shared by every module.
SETTINGS = Settings()
SETTINGS.create_directories()
SETTINGS.redirect_model_cache()
