"""Walking the corpus and building the search index.

The ingest path is deliberately incremental. :class:`CorpusScanner` enumerates
eligible files and computes a cheap size-and-modification-time signature for
each; :class:`IngestionPipeline` compares those signatures against the stored
index and only re-reads what actually changed, so a second run over an
unchanged corpus finishes in seconds instead of re-embedding 700 files.

:class:`IngestionRunner` hosts the pipeline on a background thread and exposes
:class:`IngestionProgress`, which the web interface polls to draw its progress
bar. :class:`AutomaticIngestion` starts that runner by itself whenever the
corpus changes, so crawled pages become searchable while the crawl is still
going. Nothing here ever opens a corpus file for writing — all reads funnel
through the read-only guard.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import TokenAwareChunker
from .config import SETTINGS, Settings
from .documents import SourceDocument
from .embeddings import TextEmbedder
from .extractors import EXTRACTOR_REGISTRY, ExtractorRegistry
from .safety import SOURCE_GUARD, SourceTreeGuard
from .storage import DOCUMENT_CATALOGUE, VECTOR_INDEX, DocumentCatalogue, VectorIndex

LOGGER = logging.getLogger(__name__)


class FileTypeSniffer:
    """Identifies documents saved without a file extension.

    Google Drive exports and downloads occasionally lose their extension, which
    would otherwise make a genuine PDF invisible to the extension-based router.
    Reading the first few bytes recovers the real format.
    """

    #: Leading byte sequences that identify a format unambiguously.
    MAGIC_SIGNATURES = (
        (b"%PDF", ".pdf"),
        (b"\xff\xd8\xff", ".jpg"),
        (b"\x89PNG\r\n\x1a\n", ".png"),
        (b"GIF8", ".gif"),
        (b"BM", ".bmp"),
    )

    #: Entry-name prefixes inside an OOXML zip that reveal the specific format.
    ZIP_CONTENT_MARKERS = (
        ("word/", ".docx"),
        ("xl/", ".xlsx"),
        ("ppt/", ".pptx"),
    )

    #: Files smaller than this are almost certainly not real documents.
    MINIMUM_INTERESTING_BYTES = 1024

    def __init__(self, guard: SourceTreeGuard) -> None:
        """Bind the sniffer to the read-only guard used for all corpus reads."""
        self._guard = guard

    def detect_extension(self, file_path: Path, size_bytes: int) -> str | None:
        """Return a guessed extension for an extensionless file, or None."""
        if size_bytes < self.MINIMUM_INTERESTING_BYTES:
            return None
        try:
            with self._guard.open_for_reading(file_path, "rb") as file_handle:
                header = file_handle.read(8)
        except OSError:
            return None

        for signature, extension in self.MAGIC_SIGNATURES:
            if header.startswith(signature):
                return extension
        if header.startswith(b"PK\x03\x04"):
            return self._detect_ooxml_extension(file_path)
        return None

    def _detect_ooxml_extension(self, file_path: Path) -> str | None:
        """Distinguish Word, Excel and PowerPoint inside an OOXML zip container."""
        try:
            with zipfile.ZipFile(file_path) as archive:
                entry_names = archive.namelist()
        except (zipfile.BadZipFile, OSError):
            return None
        for marker, extension in self.ZIP_CONTENT_MARKERS:
            if any(name.startswith(marker) for name in entry_names):
                return extension
        return None


@dataclass(frozen=True)
class CorpusRoot:
    """One directory tree that contributes documents to the index.

    ``label`` is prefixed onto every relative path found beneath this root. That
    keeps document identifiers unique when two roots happen to contain the same
    sub-path, and makes the origin of a citation visible in the interface. The
    primary corpus carries an empty label, so its paths read exactly as they do
    on the shared drive.
    """

    directory: Path
    label: str = ""
    #: A missing primary corpus is an error; a missing extra root is skipped.
    is_required: bool = True

    def relative_path_for(self, file_path: Path) -> str:
        """Return ``file_path`` relative to this root, including its label."""
        relative = file_path.relative_to(self.directory)
        return str(Path(self.label) / relative) if self.label else str(relative)


class CorpusScanner:
    """Enumerates the files eligible for indexing across every corpus root.

    Filtering happens in this one place: excluded directories are pruned during
    the walk (so their contents are never even stat-ed), and files are rejected
    by extension or size before any extractor sees them.

    Two roots are scanned by default — the read-only shared drive, and the
    project's ``corpus-lokal`` directory holding material copied out of it.
    """

    def __init__(
        self, settings: Settings, guard: SourceTreeGuard, registry: ExtractorRegistry
    ) -> None:
        """Bind the scanner to configuration, the guard, and the extractor set."""
        self._settings = settings
        self._guard = guard
        self._registry = registry
        self._sniffer = FileTypeSniffer(guard)

    def build_roots(self) -> list[CorpusRoot]:
        """Return every root to scan, primary corpus first."""
        roots = [CorpusRoot(Path(self._settings.source_directory), "", is_required=True)]
        for extra_directory in self._settings.additional_source_directories:
            extra_path = Path(extra_directory)
            roots.append(CorpusRoot(extra_path, extra_path.name, is_required=False))
        return roots

    def scan(self) -> list[SourceDocument]:
        """Return every indexable document currently present across all roots."""
        discovered: list[SourceDocument] = []
        for root in self.build_roots():
            if not root.directory.exists():
                if root.is_required:
                    raise FileNotFoundError(
                        f"Direktori sumber tidak ditemukan: {root.directory}"
                    )
                LOGGER.info("Direktori tambahan tidak ada, dilewati: %s", root.directory)
                continue
            self._walk_directory(root.directory, root, discovered)
        discovered.sort(key=self._sort_key)
        return discovered

    def _sort_key(self, document: SourceDocument) -> str:
        """Order documents by path so ingest progress reads predictably."""
        return document.relative_path.lower()

    def _walk_directory(
        self, directory: Path, root: CorpusRoot, discovered: list[SourceDocument]
    ) -> None:
        """Recursively collect eligible files, pruning excluded directories."""
        try:
            entries = list(directory.iterdir())
        except (OSError, PermissionError) as error:
            LOGGER.warning("Tidak dapat membaca direktori %s: %s", directory, error)
            return

        for entry in entries:
            try:
                if entry.is_dir():
                    if entry.name.lower() not in self._settings.excluded_directories:
                        self._walk_directory(entry, root, discovered)
                    continue
                document = self._build_document(entry, root)
                if document is not None:
                    discovered.append(document)
            except (OSError, PermissionError) as error:
                LOGGER.warning("Melewati %s: %s", entry, error)

    def _build_document(self, file_path: Path, root: CorpusRoot) -> SourceDocument | None:
        """Describe one file, or return None when it should not be indexed."""
        if file_path.name.startswith("~$") or file_path.name.lower() == "desktop.ini":
            return None

        file_status = file_path.stat()
        size_megabytes = file_status.st_size / (1024 * 1024)
        if size_megabytes > self._settings.maximum_file_megabytes:
            LOGGER.info("Melewati berkas besar (%.1f MB): %s", size_megabytes, file_path.name)
            return None

        extension = file_path.suffix.lower()
        if extension in self._settings.skipped_extensions:
            return None
        if not extension:
            sniffed = self._sniffer.detect_extension(file_path, file_status.st_size)
            if sniffed is None:
                return None
            extension = sniffed
        if extension not in self._settings.indexable_extensions():
            return None
        if not self._registry.supports(extension):
            return None

        return SourceDocument(
            absolute_path=file_path,
            relative_path=root.relative_path_for(file_path),
            display_name=file_path.name,
            extension=extension,
            size_bytes=file_status.st_size,
            modified_timestamp=file_status.st_mtime,
        )


@dataclass
class IngestionProgress:
    """Live state of an ingest run, polled by the web interface.

    Mutated only by :class:`IngestionPipeline` and read through
    :meth:`as_dictionary`, which takes a snapshot so a poll can never observe a
    half-updated set of counters.
    """

    is_running: bool = False
    phase: str = "idle"
    current_document: str = ""
    documents_total: int = 0
    documents_processed: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    documents_removed: int = 0
    chunks_created: int = 0
    embeddings_done: int = 0
    embeddings_total: int = 0
    scanned_pages_found: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error_message: str | None = None
    recent_messages: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    #: How many log lines the interface keeps visible.
    MESSAGE_HISTORY_LIMIT = 60

    def reset(self, documents_total: int) -> None:
        """Clear all counters at the start of a run."""
        with self._lock:
            self.is_running = True
            self.phase = "starting"
            self.current_document = ""
            self.documents_total = documents_total
            self.documents_processed = 0
            self.documents_indexed = 0
            self.documents_skipped = 0
            self.documents_failed = 0
            self.documents_removed = 0
            self.chunks_created = 0
            self.embeddings_done = 0
            self.embeddings_total = 0
            self.scanned_pages_found = 0
            self.started_at = time.time()
            self.finished_at = 0.0
            self.error_message = None
            self.recent_messages = []

    def record_message(self, message: str) -> None:
        """Append a human-readable line to the run log."""
        with self._lock:
            self.recent_messages.append(message)
            if len(self.recent_messages) > self.MESSAGE_HISTORY_LIMIT:
                self.recent_messages = self.recent_messages[-self.MESSAGE_HISTORY_LIMIT :]

    def set_phase(self, phase: str) -> None:
        """Record which stage of the pipeline is currently active."""
        with self._lock:
            self.phase = phase

    def finish(self, error_message: str | None = None) -> None:
        """Mark the run complete, optionally recording why it failed."""
        with self._lock:
            self.is_running = False
            self.phase = "failed" if error_message else "completed"
            self.finished_at = time.time()
            self.error_message = error_message

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock duration of the run so far, or of the finished run."""
        if self.started_at == 0.0:
            return 0.0
        end_time = self.finished_at or time.time()
        return round(end_time - self.started_at, 1)

    def as_dictionary(self) -> dict:
        """Snapshot the progress state as plain JSON-serialisable data."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "phase": self.phase,
                "current_document": self.current_document,
                "documents_total": self.documents_total,
                "documents_processed": self.documents_processed,
                "documents_indexed": self.documents_indexed,
                "documents_skipped": self.documents_skipped,
                "documents_failed": self.documents_failed,
                "documents_removed": self.documents_removed,
                "chunks_created": self.chunks_created,
                "embeddings_done": self.embeddings_done,
                "embeddings_total": self.embeddings_total,
                "scanned_pages_found": self.scanned_pages_found,
                "elapsed_seconds": self.elapsed_seconds,
                "error_message": self.error_message,
                "recent_messages": list(self.recent_messages),
            }


class IngestionPipeline:
    """Builds the index from the corpus, re-reading only what changed.

    One run performs five steps: scan the corpus, reconcile the two stores,
    extract and chunk each new or modified document, embed its chunks, and
    persist both stores. Failures are recorded per document so a single
    unreadable file never aborts the run.
    """

    #: Documents indexed between flushes of the vector archive to disk.
    VECTOR_FLUSH_INTERVAL = 25

    def __init__(
        self,
        settings: Settings,
        catalogue: DocumentCatalogue,
        vector_index: VectorIndex,
        embedder: TextEmbedder,
        registry: ExtractorRegistry,
        guard: SourceTreeGuard,
    ) -> None:
        """Wire the pipeline to every collaborator it needs."""
        self._settings = settings
        self._catalogue = catalogue
        self._vector_index = vector_index
        self._embedder = embedder
        self._registry = registry
        self._scanner = CorpusScanner(settings, guard, registry)
        self._last_flush_document_count = 0
        self.progress = IngestionProgress()

    def run(self, force_full_rebuild: bool = False) -> dict:
        """Execute one ingest pass and return a summary of what changed.

        Args:
            force_full_rebuild: Discard the existing index and re-read every
                document, which is needed after changing the embedding model or
                the chunking parameters.
        """
        try:
            return self._execute(force_full_rebuild)
        except Exception as error:
            LOGGER.exception("Proses indexing gagal")
            self.progress.finish(error_message=f"{type(error).__name__}: {error}")
            raise

    def corpus_fingerprint(self) -> str:
        """Digest of every indexable file's path, size and modification time.

        Only directory entries are read, never file contents, so it is cheap
        enough to take every minute. Two equal digests mean an incremental run
        would find nothing to do.
        """
        digest = hashlib.sha1()
        for document in self._scanner.scan():
            line = f"{document.relative_path}\0{document.content_signature}\n"
            digest.update(line.encode("utf-8", "replace"))
        return digest.hexdigest()

    def _execute(self, force_full_rebuild: bool) -> dict:
        """Run the five ingest steps in order."""
        self._last_flush_document_count = 0
        self.progress.reset(documents_total=0)
        self.progress.set_phase("scanning")
        self.progress.record_message("Memindai direktori sumber (baca-saja)...")

        discovered_documents = self._scanner.scan()
        self.progress.reset(documents_total=len(discovered_documents))
        self.progress.record_message(
            f"Ditemukan {len(discovered_documents)} berkas yang dapat diindeks."
        )

        if force_full_rebuild:
            self._clear_existing_index()

        self._remove_deleted_documents(discovered_documents)
        stored_signatures = self._catalogue.load_content_signatures()
        self._forget_documents_missing_vectors(stored_signatures)

        chunker = TokenAwareChunker(self._settings, self._embedder.build_token_counter())
        self.progress.set_phase("indexing")

        for document in discovered_documents:
            self._process_document(document, stored_signatures, chunker)
            self._save_vectors_periodically()

        self.progress.set_phase("saving")
        self._discard_orphan_vectors()
        self.progress.record_message("Menyimpan indeks vektor...")
        self._vector_index.save()
        self._catalogue.optimise()

        self.progress.finish()
        self.progress.record_message("Selesai.")
        return self.progress.as_dictionary()

    def _forget_documents_missing_vectors(self, stored_signatures: dict[str, str]) -> None:
        """Drop the stored signature of any document whose vectors were lost.

        Removing the signature is what makes the incremental pass re-read the
        document. Without it, an ingest run interrupted between the last vector
        flush and the end would leave those documents indexed in SQLite but
        absent from semantic search, and every later run would skip them.
        """
        stranded_ids = self._catalogue.find_documents_with_missing_vectors(
            self._vector_index.known_chunk_ids()
        )
        if not stranded_ids:
            return
        for document_id in stranded_ids:
            stored_signatures.pop(document_id, None)
        self.progress.record_message(
            f"Memulihkan {len(stranded_ids)} dokumen yang kehilangan vektor "
            f"(indexing sebelumnya terputus)."
        )

    def _is_scan_implausibly_small(self, discovered_documents: list[SourceDocument]) -> bool:
        """True when the scan found too few files to be trusted for deletion."""
        indexed_count = len(self._catalogue.list_document_ids())
        if indexed_count == 0:
            return False

        survivors = len(discovered_documents)
        if survivors >= indexed_count * self.MINIMUM_SURVIVING_SHARE:
            return False

        message = (
            f"Pemindaian hanya menemukan {survivors} berkas padahal indeks berisi "
            f"{indexed_count} dokumen. Penghapusan dilewati — kemungkinan besar "
            f"shared drive belum tersinkron penuh. Jalankan ulang setelah Google "
            f"Drive siap, atau pakai --rebuild bila memang disengaja."
        )
        LOGGER.warning(message)
        self.progress.record_message(message)
        return True

    def _discard_orphan_vectors(self) -> None:
        """Drop vectors whose chunk row no longer exists.

        The mirror image of the missing-vector recovery above. A vector is
        stranded when its document was re-indexed or deleted between periodic
        flushes; retrieval already skips such rows, but they occupy memory and
        crowd out real candidates in the fixed-size top-k, quietly shrinking the
        result set.
        """
        stored_chunk_ids = self._catalogue.list_chunk_ids()
        orphans = [
            chunk_id
            for chunk_id in self._vector_index.known_chunk_ids()
            if chunk_id not in stored_chunk_ids
        ]
        if not orphans:
            return
        self._vector_index.remove(orphans)
        self.progress.record_message(f"Membuang {len(orphans)} vektor yatim.")

    def _save_vectors_periodically(self) -> None:
        """Flush the vector archive every so often during a long run.

        Bounds how much embedding work an interruption can cost. Anything newer
        than the last flush is recovered by the reconciliation above.
        """
        if self.progress.documents_indexed == 0:
            return
        if self.progress.documents_indexed % self.VECTOR_FLUSH_INTERVAL != 0:
            return
        if self.progress.documents_indexed == self._last_flush_document_count:
            return
        self._last_flush_document_count = self.progress.documents_indexed
        self._vector_index.save()

    def _clear_existing_index(self) -> None:
        """Drop the whole index ahead of a forced full rebuild."""
        self.progress.record_message("Membangun ulang dari nol: menghapus indeks lama...")
        self._catalogue.clear()
        self._vector_index.clear()

    #: Refuse to purge when a scan finds less than this share of the indexed
    #: documents. A network drive that is unmounted or still syncing looks
    #: exactly like a corpus where everything was deleted.
    MINIMUM_SURVIVING_SHARE = 0.5

    def _remove_deleted_documents(self, discovered_documents: list[SourceDocument]) -> None:
        """Purge index entries whose source file has disappeared.

        This deliberately reads *every* stored identifier rather than the
        signature map, because that map omits previously failed documents. A
        renamed folder produces exactly that combination — entries that are both
        failed and gone — and using the signature map here would leave them in
        the index permanently.

        Purging is skipped entirely when the scan returns implausibly few files.
        The corpus lives on a Google shared drive that can be half-synced or
        briefly unmounted, and in that state an ordinary run would otherwise
        delete a working index and force a full re-embed.
        """
        if self._is_scan_implausibly_small(discovered_documents):
            return
        present_ids = {document.document_id for document in discovered_documents}
        vanished_ids = [
            document_id
            for document_id in self._catalogue.list_document_ids()
            if document_id not in present_ids
        ]
        if not vanished_ids:
            return
        removed_chunk_ids = self._catalogue.delete_documents(vanished_ids)
        self._vector_index.remove(removed_chunk_ids)
        self.progress.documents_removed = len(vanished_ids)
        self.progress.record_message(
            f"Menghapus {len(vanished_ids)} dokumen yang sudah tidak ada di sumber."
        )

    def _process_document(
        self,
        document: SourceDocument,
        stored_signatures: dict[str, str],
        chunker: TokenAwareChunker,
    ) -> None:
        """Extract, chunk, embed and store one document if it changed."""
        self.progress.current_document = document.relative_path
        self.progress.documents_processed += 1

        if stored_signatures.get(document.document_id) == document.content_signature:
            self.progress.documents_skipped += 1
            return

        # The document is new or edited: drop any previous version first.
        stale_chunk_ids = self._catalogue.delete_documents([document.document_id])
        self._vector_index.remove(stale_chunk_ids)

        outcome = self._registry.extract(document)
        self.progress.scanned_pages_found += outcome.scanned_section_count

        if outcome.error_message is not None:
            self._store_unusable_document(document, outcome, status="failed")
            self.progress.documents_failed += 1
            self.progress.record_message(
                f"GAGAL {document.relative_path}: {outcome.error_message}"
            )
            return

        chunks = chunker.chunk_sections(document.document_id, outcome.sections)
        if not chunks:
            self._store_unusable_document(document, outcome, status="empty")
            self.progress.documents_skipped += 1
            if outcome.scanned_section_count:
                self.progress.record_message(
                    f"KOSONG {document.relative_path}: "
                    f"{outcome.scanned_section_count} halaman terindikasi hasil pindai "
                    f"(aktifkan OCR untuk membacanya)."
                )
            return

        self.progress.embeddings_total += len(chunks)
        chunk_texts = [chunk.text for chunk in chunks]
        vectors = self._embedder.encode_passages(chunk_texts)

        assigned_chunk_ids = self._catalogue.save_document(
            document=document,
            chunks=chunks,
            section_count=len(outcome.sections),
            character_count=outcome.total_characters,
            scanned_section_count=outcome.scanned_section_count,
            ocr_section_count=outcome.ocr_section_count,
            status="indexed",
        )
        self._vector_index.add(assigned_chunk_ids, vectors)

        self.progress.documents_indexed += 1
        self.progress.chunks_created += len(chunks)
        self.progress.embeddings_done += len(chunks)

    def _store_unusable_document(self, document, outcome, status: str) -> None:
        """Record a document that produced no chunks, with the reason why."""
        self._catalogue.save_document(
            document=document,
            chunks=[],
            section_count=len(outcome.sections),
            character_count=outcome.total_characters,
            scanned_section_count=outcome.scanned_section_count,
            ocr_section_count=outcome.ocr_section_count,
            status=status,
            error_message=outcome.error_message,
        )


class IngestionRunner:
    """Runs the ingest pipeline on a background thread, one run at a time.

    The web server returns immediately after starting a run and the browser
    polls :meth:`current_progress`, so a 700-file ingest never blocks an HTTP
    request or ties up the event loop.
    """

    def __init__(self, pipeline: IngestionPipeline) -> None:
        """Bind the runner to the pipeline it will execute."""
        self._pipeline = pipeline
        self._worker_thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """True while a background ingest run is in flight."""
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def start(self, force_full_rebuild: bool = False) -> bool:
        """Begin a run, returning False when one is already in progress."""
        with self._start_lock:
            if self.is_running:
                return False
            self._worker_thread = threading.Thread(
                target=self._run_pipeline,
                args=(force_full_rebuild,),
                name="djp-ingest",
                daemon=True,
            )
            self._worker_thread.start()
            return True

    def _run_pipeline(self, force_full_rebuild: bool) -> None:
        """Thread body: execute one pipeline run, swallowing the exception.

        Failures are already recorded in the progress object, which is what the
        interface displays; re-raising here would only kill the worker thread.
        """
        try:
            self._pipeline.run(force_full_rebuild=force_full_rebuild)
        except Exception:
            LOGGER.exception("Proses indexing latar belakang berhenti")

    def current_progress(self) -> dict:
        """Snapshot the running (or last completed) run's progress."""
        return self._pipeline.progress.as_dictionary()


class AutomaticIngestion:
    """Indexes on its own whenever the corpus changes, while the server runs.

    The crawler keeps adding pajak.go.id pages to the corpus. Without this they
    would stay unsearchable until someone pressed "Perbarui indeks". Every
    ``interval_seconds`` the corpus is fingerprinted and, when the fingerprint
    differs from the one last handed to a run, an incremental run starts on the
    shared runner. Sharing the runner with the button means the two never
    overlap, and because the run happens inside the server process, its
    in-memory vectors are current the moment the run ends.
    """

    def __init__(
        self, pipeline: IngestionPipeline, runner: IngestionRunner, interval_seconds: float
    ) -> None:
        """Bind the watcher to the pipeline it fingerprints and the runner it starts."""
        self._pipeline = pipeline
        self._runner = runner
        self._interval_seconds = interval_seconds
        #: Fingerprint of the corpus as the last started run saw it. None until
        #: the first check, so the first check always runs once to catch up on
        #: anything added while the server was down.
        self._indexed_fingerprint: str | None = None
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> bool:
        """Begin watching; False when switched off (interval 0) or already started."""
        if self._interval_seconds <= 0 or self._worker_thread is not None:
            return False
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._watch, name="djp-auto-ingest", daemon=True
        )
        self._worker_thread.start()
        LOGGER.info(
            "Indexing otomatis aktif: korpus diperiksa tiap %s detik", self._interval_seconds
        )
        return True

    def stop(self) -> None:
        """Stop watching; a run already in progress is left to finish."""
        self._stop_event.set()
        self._worker_thread = None

    def check_once(self) -> bool:
        """Start a run if the corpus changed since the last one; True when started."""
        if self._runner.is_running:
            return False
        fingerprint = self._pipeline.corpus_fingerprint()
        if fingerprint == self._indexed_fingerprint:
            return False
        if not self._runner.start():
            return False
        self._indexed_fingerprint = fingerprint
        LOGGER.info("Korpus berubah; indexing otomatis dimulai")
        return True

    def _watch(self) -> None:
        """Thread body: check the corpus every interval until stopped."""
        while not self._stop_event.wait(self._interval_seconds):
            try:
                self.check_once()
            except Exception:
                LOGGER.exception("Pemeriksaan indexing otomatis gagal")


#: Shared embedder used by both ingest and query paths, so weights load once.
TEXT_EMBEDDER = TextEmbedder(SETTINGS)

#: Shared pipeline and background runner bound to the process-wide stores.
INGESTION_PIPELINE = IngestionPipeline(
    settings=SETTINGS,
    catalogue=DOCUMENT_CATALOGUE,
    vector_index=VECTOR_INDEX,
    embedder=TEXT_EMBEDDER,
    registry=EXTRACTOR_REGISTRY,
    guard=SOURCE_GUARD,
)
INGESTION_RUNNER = IngestionRunner(INGESTION_PIPELINE)

#: Watcher the web server starts, indexing crawled pages as they arrive.
AUTOMATIC_INGESTION = AutomaticIngestion(
    INGESTION_PIPELINE, INGESTION_RUNNER, SETTINGS.auto_ingest_seconds
)
