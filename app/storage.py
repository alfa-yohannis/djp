"""Persistence for the index: document metadata, chunk text and vectors.

The index is split across two stores that are written together and kept
consistent by :class:`app.ingestion.IngestionPipeline`:

* :class:`DocumentCatalogue` — a SQLite database holding document metadata,
  chunk text, and an FTS5 full-text index used for BM25 keyword search.
* :class:`VectorIndex` — a NumPy archive of L2-normalised chunk embeddings.

SQLite plus NumPy is a deliberate choice over a dedicated vector database. At
this corpus size the whole matrix is a few tens of megabytes, so an exhaustive
dot product answers a query in single-digit milliseconds with exact recall,
while SQLite gives durable metadata and BM25 with no extra dependency.

Every write target passes through :data:`app.safety.SOURCE_GUARD` first, so a
mis-configured data directory can never place index files inside the corpus.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import SETTINGS, Settings
from .documents import RetrievedPassage, SourceDocument, TextChunk
from .safety import SOURCE_GUARD, SourceTreeGuard

LOGGER = logging.getLogger(__name__)


class KeywordQueryBuilder:
    """Converts a natural-language question into a safe FTS5 MATCH expression.

    FTS5 treats characters such as ``"``, ``*``, ``(`` and ``-`` as operators, so
    a raw question routinely produces a syntax error. This builder keeps only
    word characters, drops very short and very common Indonesian stop words, and
    joins the remainder with ``OR`` so partial matches still rank.
    """

    _WORD_PATTERN = re.compile(r"[\wÀ-ɏ]+", re.UNICODE)

    #: High-frequency Indonesian and English function words that add no signal.
    STOP_WORDS = frozenset({
        "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah",
        "atau", "ini", "itu", "dalam", "oleh", "sebagai", "juga", "akan", "ada",
        "tidak", "bisa", "dapat", "apa", "siapa", "bagaimana", "berapa", "kapan",
        "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "is", "are",
        "what", "how", "who", "when", "which",
    })

    #: Words shorter than this carry too little signal to be worth matching.
    MINIMUM_WORD_LENGTH = 3

    def build(self, question: str) -> str:
        """Return an FTS5 expression for ``question``, or "" when unusable."""
        candidate_words = self._WORD_PATTERN.findall(question.lower())
        meaningful_words = [
            word
            for word in candidate_words
            if len(word) >= self.MINIMUM_WORD_LENGTH and word not in self.STOP_WORDS
        ]
        if not meaningful_words:
            # Every word was filtered out; fall back to the raw words so a
            # question like "apa itu EWMP" still matches on "EWMP".
            meaningful_words = [word for word in candidate_words if len(word) > 1]
        if not meaningful_words:
            return ""
        # Quoting each term makes it a literal, neutralising FTS5 operators.
        return " OR ".join(f'"{word}"' for word in dict.fromkeys(meaningful_words))


class DocumentCatalogue:
    """SQLite-backed store of documents, their chunks, and a keyword index.

    A single connection is shared across threads under a lock. The web server
    handles one question at a time per request and ingest runs in a single
    worker thread, so contention is negligible and this avoids the complexity of
    per-thread connection pools.
    """

    #: Schema version, bumped when a migration would be required.
    SCHEMA_VERSION = 1

    _SCHEMA_STATEMENTS = (
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id           TEXT PRIMARY KEY,
            relative_path         TEXT NOT NULL,
            absolute_path         TEXT NOT NULL,
            display_name          TEXT NOT NULL,
            extension             TEXT NOT NULL,
            top_level_folder      TEXT NOT NULL,
            size_bytes            INTEGER NOT NULL,
            modified_timestamp    REAL NOT NULL,
            content_signature     TEXT NOT NULL,
            section_count         INTEGER NOT NULL DEFAULT 0,
            chunk_count           INTEGER NOT NULL DEFAULT 0,
            character_count       INTEGER NOT NULL DEFAULT 0,
            scanned_section_count INTEGER NOT NULL DEFAULT 0,
            ocr_section_count     INTEGER NOT NULL DEFAULT 0,
            status                TEXT NOT NULL,
            error_message         TEXT,
            indexed_at            TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id     TEXT NOT NULL,
            ordinal         INTEGER NOT NULL,
            section_label   TEXT NOT NULL,
            text            TEXT NOT NULL,
            token_count     INTEGER NOT NULL,
            produced_by_ocr INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS index_chunks_by_document ON chunks(document_id)",
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search
        USING fts5(text, tokenize='unicode61 remove_diacritics 2')
        """,
        """
        CREATE TABLE IF NOT EXISTS index_metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
    )

    def __init__(self, settings: Settings, guard: SourceTreeGuard) -> None:
        """Open (creating if needed) the catalogue database."""
        self._settings = settings
        self._database_path: Path = guard.assert_writable(settings.database_path)
        self._access_lock = threading.RLock()
        self._query_builder = KeywordQueryBuilder()
        self._connection = sqlite3.connect(
            self._database_path, check_same_thread=False, timeout=30.0
        )
        self._connection.row_factory = sqlite3.Row
        self._configure_connection()
        self._create_schema()

    def _configure_connection(self) -> None:
        """Apply pragmas that suit a single-writer, read-heavy local index."""
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA temp_store=MEMORY")

    def _create_schema(self) -> None:
        """Create tables and the full-text index if this is a fresh database."""
        with self._access_lock:
            for statement in self._SCHEMA_STATEMENTS:
                self._connection.execute(statement)
            self._connection.execute(
                "INSERT OR REPLACE INTO index_metadata(key, value) VALUES(?, ?)",
                ("schema_version", str(self.SCHEMA_VERSION)),
            )
            self._connection.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        with self._access_lock:
            self._connection.close()

    # --- Change detection ---------------------------------------------------

    def load_content_signatures(self) -> dict[str, str]:
        """Return ``document_id -> content_signature`` for reusable documents.

        The ingest pipeline compares this against the filesystem to decide which
        documents actually changed, so an unchanged corpus re-indexes instantly.

        Documents whose last ingest **failed** are deliberately excluded, so the
        next run retries them. Most extraction failures on a shared drive are
        transient — a colleague renaming a folder mid-scan, a workbook held open
        in Excel, Google Drive re-syncing. Keeping their signature would let one
        unlucky moment exclude a document from the index permanently.
        """
        with self._access_lock:
            rows = self._connection.execute(
                "SELECT document_id, content_signature FROM documents "
                "WHERE status != 'failed'"
            ).fetchall()
        return {row["document_id"]: row["content_signature"] for row in rows}

    def list_document_ids(self) -> set[str]:
        """Return the identifiers of every document currently indexed."""
        with self._access_lock:
            rows = self._connection.execute("SELECT document_id FROM documents").fetchall()
        return {row["document_id"] for row in rows}

    def list_chunk_ids(self) -> set[int]:
        """Return every chunk identifier currently stored."""
        with self._access_lock:
            rows = self._connection.execute("SELECT chunk_id FROM chunks").fetchall()
        return {int(row["chunk_id"]) for row in rows}

    def find_documents_with_missing_vectors(self, known_chunk_ids: set[int]) -> set[str]:
        """Return documents whose chunks are absent from the vector index.

        The two stores can drift apart if an ingest run is interrupted: chunk
        rows are committed per document, but the vector archive is only flushed
        periodically. Without this reconciliation the affected documents would
        keep their unchanged content signature, be skipped by every later
        incremental run, and stay permanently invisible to semantic search.
        """
        with self._access_lock:
            rows = self._connection.execute(
                "SELECT document_id, chunk_id FROM chunks"
            ).fetchall()
        return {
            row["document_id"]
            for row in rows
            if int(row["chunk_id"]) not in known_chunk_ids
        }

    # --- Writing ------------------------------------------------------------

    def delete_documents(self, document_ids: list[str]) -> list[int]:
        """Remove documents and their chunks, returning the deleted chunk ids.

        The caller uses the returned ids to drop matching rows from the vector
        index, keeping the two stores in step.
        """
        if not document_ids:
            return []
        with self._access_lock:
            placeholders = ",".join("?" for _ in document_ids)
            chunk_rows = self._connection.execute(
                f"SELECT chunk_id FROM chunks WHERE document_id IN ({placeholders})",
                document_ids,
            ).fetchall()
            removed_chunk_ids = [row["chunk_id"] for row in chunk_rows]

            for chunk_id in removed_chunk_ids:
                self._connection.execute(
                    "DELETE FROM chunk_search WHERE rowid = ?", (chunk_id,)
                )
            self._connection.execute(
                f"DELETE FROM chunks WHERE document_id IN ({placeholders})", document_ids
            )
            self._connection.execute(
                f"DELETE FROM documents WHERE document_id IN ({placeholders})", document_ids
            )
            self._connection.commit()
        return removed_chunk_ids

    def save_document(
        self,
        document: SourceDocument,
        chunks: list[TextChunk],
        section_count: int,
        character_count: int,
        scanned_section_count: int,
        ocr_section_count: int,
        status: str,
        error_message: str | None = None,
    ) -> list[int]:
        """Insert a document and its chunks, returning the new chunk ids.

        The returned ids line up positionally with ``chunks``, so the caller can
        pair them with the embedding matrix rows it just computed.
        """
        with self._access_lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, relative_path, absolute_path, display_name, extension,
                    top_level_folder, size_bytes, modified_timestamp, content_signature,
                    section_count, chunk_count, character_count,
                    scanned_section_count, ocr_section_count,
                    status, error_message, indexed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    document.document_id,
                    document.relative_path,
                    str(document.absolute_path),
                    document.display_name,
                    document.extension,
                    document.top_level_folder,
                    document.size_bytes,
                    document.modified_timestamp,
                    document.content_signature,
                    section_count,
                    len(chunks),
                    character_count,
                    scanned_section_count,
                    ocr_section_count,
                    status,
                    error_message,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

            assigned_chunk_ids: list[int] = []
            for chunk in chunks:
                cursor = self._connection.execute(
                    """
                    INSERT INTO chunks
                        (document_id, ordinal, section_label, text, token_count, produced_by_ocr)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.section_label,
                        chunk.text,
                        chunk.token_count,
                        1 if chunk.produced_by_ocr else 0,
                    ),
                )
                chunk_id = int(cursor.lastrowid)
                assigned_chunk_ids.append(chunk_id)
                # rowid is set explicitly so the FTS row and the chunk row share an id.
                self._connection.execute(
                    "INSERT INTO chunk_search(rowid, text) VALUES (?, ?)",
                    (chunk_id, chunk.text),
                )
            self._connection.commit()
        return assigned_chunk_ids

    def optimise(self) -> None:
        """Compact the full-text index after a large ingest run."""
        with self._access_lock:
            try:
                self._connection.execute(
                    "INSERT INTO chunk_search(chunk_search) VALUES('optimize')"
                )
                self._connection.commit()
            except sqlite3.OperationalError as error:
                LOGGER.warning("Optimisasi indeks teks gagal: %s", error)

    def clear(self) -> list[int]:
        """Delete the entire index, returning every removed chunk id."""
        with self._access_lock:
            rows = self._connection.execute("SELECT chunk_id FROM chunks").fetchall()
            removed_chunk_ids = [row["chunk_id"] for row in rows]
            self._connection.execute("DELETE FROM chunk_search")
            self._connection.execute("DELETE FROM chunks")
            self._connection.execute("DELETE FROM documents")
            self._connection.commit()
        return removed_chunk_ids

    # --- Reading ------------------------------------------------------------

    def search_by_keyword(
        self, question: str, limit: int, folder_filter: str | None = None
    ) -> list[RetrievedPassage]:
        """Return the best BM25 matches for ``question``.

        SQLite's ``bm25()`` returns increasingly negative numbers for better
        matches; the sign is flipped so callers always sort descending.
        """
        match_expression = self._query_builder.build(question)
        if not match_expression:
            return []

        sql = """
            SELECT c.chunk_id, c.document_id, c.section_label, c.text, c.produced_by_ocr,
                   d.display_name, d.relative_path, d.absolute_path,
                   bm25(chunk_search) AS relevance
            FROM chunk_search
            JOIN chunks    c ON c.chunk_id = chunk_search.rowid
            JOIN documents d ON d.document_id = c.document_id
            WHERE chunk_search MATCH ?
        """
        parameters: list = [match_expression]
        if folder_filter:
            sql += " AND d.top_level_folder = ?"
            parameters.append(folder_filter)
        sql += " ORDER BY relevance LIMIT ?"
        parameters.append(limit)

        with self._access_lock:
            try:
                rows = self._connection.execute(sql, parameters).fetchall()
            except sqlite3.OperationalError as error:
                LOGGER.warning("Pencarian kata kunci gagal: %s", error)
                return []

        return [self._row_to_passage(row, sparse_score=-float(row["relevance"])) for row in rows]

    def fetch_passages_by_id(self, chunk_ids: list[int]) -> dict[int, RetrievedPassage]:
        """Load full passage records for a set of chunk ids, keyed by id."""
        if not chunk_ids:
            return {}
        with self._access_lock:
            placeholders = ",".join("?" for _ in chunk_ids)
            rows = self._connection.execute(
                f"""
                SELECT c.chunk_id, c.document_id, c.section_label, c.text, c.produced_by_ocr,
                       d.display_name, d.relative_path, d.absolute_path
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        return {int(row["chunk_id"]): self._row_to_passage(row) for row in rows}

    def find_texts_containing(self, needle: str, limit: int = 5000) -> list[tuple[str, str]]:
        """Return ``(document_name, chunk_text)`` for chunks containing ``needle``.

        Used to harvest outbound links already present in the corpus, so the
        external pages those documents point at can be snapshotted and indexed
        alongside them.
        """
        with self._access_lock:
            rows = self._connection.execute(
                """
                SELECT d.display_name, c.text
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                WHERE c.text LIKE ?
                LIMIT ?
                """,
                (f"%{needle}%", limit),
            ).fetchall()
        return [(row["display_name"], row["text"]) for row in rows]

    def fetch_chunk_ids_for_folder(self, folder_name: str) -> set[int]:
        """Return every chunk id belonging to one top-level folder."""
        with self._access_lock:
            rows = self._connection.execute(
                """
                SELECT c.chunk_id FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                WHERE d.top_level_folder = ?
                """,
                (folder_name,),
            ).fetchall()
        return {int(row["chunk_id"]) for row in rows}

    def _row_to_passage(
        self, row: sqlite3.Row, sparse_score: float | None = None
    ) -> RetrievedPassage:
        """Map a joined chunk/document row onto a retrieval result object."""
        return RetrievedPassage(
            chunk_id=int(row["chunk_id"]),
            document_id=row["document_id"],
            document_name=row["display_name"],
            relative_path=row["relative_path"],
            absolute_path=row["absolute_path"],
            section_label=row["section_label"],
            text=row["text"],
            produced_by_ocr=bool(row["produced_by_ocr"]),
            sparse_score=sparse_score,
        )

    # --- Statistics ---------------------------------------------------------

    def collect_statistics(self) -> dict:
        """Summarise the index for the status panel in the web interface."""
        with self._access_lock:
            totals = self._connection.execute(
                """
                SELECT
                    COUNT(*)                              AS document_count,
                    COALESCE(SUM(chunk_count), 0)         AS chunk_count,
                    COALESCE(SUM(character_count), 0)     AS character_count,
                    COALESCE(SUM(size_bytes), 0)          AS total_bytes,
                    COALESCE(SUM(scanned_section_count),0) AS scanned_section_count,
                    COALESCE(SUM(ocr_section_count), 0)   AS ocr_section_count,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
                    SUM(CASE WHEN status = 'empty'  THEN 1 ELSE 0 END) AS empty_count,
                    MAX(indexed_at)                       AS last_indexed_at
                FROM documents
                """
            ).fetchone()
            folder_rows = self._connection.execute(
                """
                SELECT top_level_folder AS folder,
                       COUNT(*)                      AS document_count,
                       COALESCE(SUM(chunk_count), 0) AS chunk_count
                FROM documents
                GROUP BY top_level_folder
                ORDER BY document_count DESC
                """
            ).fetchall()
            extension_rows = self._connection.execute(
                """
                SELECT extension, COUNT(*) AS document_count
                FROM documents
                GROUP BY extension
                ORDER BY document_count DESC
                """
            ).fetchall()

        return {
            "document_count": int(totals["document_count"] or 0),
            "chunk_count": int(totals["chunk_count"] or 0),
            "character_count": int(totals["character_count"] or 0),
            "total_megabytes": round(float(totals["total_bytes"] or 0) / (1024 * 1024), 1),
            "scanned_section_count": int(totals["scanned_section_count"] or 0),
            "ocr_section_count": int(totals["ocr_section_count"] or 0),
            "failed_count": int(totals["failed_count"] or 0),
            "empty_count": int(totals["empty_count"] or 0),
            "last_indexed_at": totals["last_indexed_at"],
            "folders": [dict(row) for row in folder_rows],
            "extensions": [dict(row) for row in extension_rows],
        }

    def list_documents(self, status_filter: str | None = None, limit: int = 1000) -> list[dict]:
        """List indexed documents, newest first, optionally filtered by status."""
        sql = """
            SELECT document_id, display_name, relative_path, absolute_path, extension,
                   top_level_folder, size_bytes, chunk_count, section_count,
                   scanned_section_count, ocr_section_count, status, error_message, indexed_at
            FROM documents
        """
        parameters: list = []
        if status_filter:
            sql += " WHERE status = ?"
            parameters.append(status_filter)
        sql += " ORDER BY top_level_folder, display_name LIMIT ?"
        parameters.append(limit)

        with self._access_lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_folders(self) -> list[str]:
        """Return the distinct top-level folders present in the index."""
        with self._access_lock:
            rows = self._connection.execute(
                "SELECT DISTINCT top_level_folder FROM documents ORDER BY top_level_folder"
            ).fetchall()
        return [row["top_level_folder"] for row in rows]

    def find_document_path(self, document_id: str) -> str | None:
        """Return the absolute path of one indexed document, or None."""
        with self._access_lock:
            row = self._connection.execute(
                "SELECT absolute_path FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return row["absolute_path"] if row else None


class VectorIndex:
    """In-memory matrix of normalised chunk embeddings, persisted as an ``.npz``.

    Because every stored vector is unit length, cosine similarity reduces to a
    dot product, so a search is one matrix-vector multiply over the whole index.
    At a few tens of thousands of chunks that costs single-digit milliseconds
    and, unlike an approximate index, never misses a relevant passage.
    """

    def __init__(self, settings: Settings, guard: SourceTreeGuard) -> None:
        """Load the persisted vectors, or start with an empty index."""
        self._settings = settings
        self._archive_path: Path = guard.assert_writable(settings.vector_store_path)
        self._access_lock = threading.RLock()
        self._chunk_ids: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._vectors: np.ndarray = np.zeros(
            (0, settings.embedding_dimensions), dtype=np.float32
        )
        self.load()

    @property
    def size(self) -> int:
        """Number of vectors currently held."""
        return int(self._chunk_ids.shape[0])

    @property
    def dimensions(self) -> int:
        """Width of the stored vectors."""
        return int(self._vectors.shape[1]) if self._vectors.size else self._settings.embedding_dimensions

    def known_chunk_ids(self) -> set[int]:
        """Return the chunk ids this index currently holds vectors for."""
        with self._access_lock:
            return {int(chunk_id) for chunk_id in self._chunk_ids}

    def load(self) -> None:
        """Read the vector archive from disk if it exists."""
        if not self._archive_path.exists():
            return
        try:
            with np.load(self._archive_path) as archive:
                self._chunk_ids = archive["chunk_ids"].astype(np.int64)
                self._vectors = archive["vectors"].astype(np.float32)
            LOGGER.info("Memuat %d vektor dari %s", self.size, self._archive_path)
        except Exception as error:
            LOGGER.warning("Gagal memuat indeks vektor, memulai kosong: %s", error)
            self._chunk_ids = np.zeros((0,), dtype=np.int64)
            self._vectors = np.zeros((0, self._settings.embedding_dimensions), dtype=np.float32)

    def save(self) -> None:
        """Atomically persist the vectors, via a temporary file and replace.

        The archive is written through an open file handle rather than a path:
        given a path, ``savez`` appends its own ``.npz`` suffix, which would
        leave the temporary file under a name the replace step cannot find.

        It is stored uncompressed. Embeddings barely compress (about 8%), and
        at 100k vectors compressing takes 20 s against 0.4 s, all of it under
        the lock that searches wait on — while ingest saves every 25 documents.
        """
        with self._access_lock:
            temporary_path = self._archive_path.with_name(self._archive_path.name + ".tmp")
            with open(temporary_path, "wb") as archive_file:
                np.savez(archive_file, chunk_ids=self._chunk_ids, vectors=self._vectors)
            temporary_path.replace(self._archive_path)
            LOGGER.info("Menyimpan %d vektor ke %s", self.size, self._archive_path)

    def add(self, chunk_ids: list[int], vectors: np.ndarray) -> None:
        """Append newly embedded chunks to the index."""
        if not chunk_ids or vectors.size == 0:
            return
        with self._access_lock:
            incoming_ids = np.asarray(chunk_ids, dtype=np.int64)
            incoming_vectors = np.asarray(vectors, dtype=np.float32)
            if self._vectors.size == 0:
                self._chunk_ids = incoming_ids
                self._vectors = incoming_vectors
            else:
                self._chunk_ids = np.concatenate([self._chunk_ids, incoming_ids])
                self._vectors = np.vstack([self._vectors, incoming_vectors])

    def remove(self, chunk_ids: list[int]) -> None:
        """Drop vectors for chunks that were deleted or are being re-indexed."""
        if not chunk_ids or self.size == 0:
            return
        with self._access_lock:
            doomed = np.isin(self._chunk_ids, np.asarray(chunk_ids, dtype=np.int64))
            keep = ~doomed
            self._chunk_ids = self._chunk_ids[keep]
            self._vectors = self._vectors[keep]

    def clear(self) -> None:
        """Empty the index entirely."""
        with self._access_lock:
            self._chunk_ids = np.zeros((0,), dtype=np.int64)
            self._vectors = np.zeros((0, self._settings.embedding_dimensions), dtype=np.float32)

    def search(
        self, question_vector: np.ndarray, limit: int, allowed_chunk_ids: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Return ``(chunk_id, cosine_similarity)`` for the closest vectors.

        Args:
            question_vector: A unit-length query embedding.
            limit: Maximum number of results.
            allowed_chunk_ids: When given, restricts the search to these chunks,
                which is how the folder filter is applied.
        """
        with self._access_lock:
            if self.size == 0:
                return []
            candidate_ids = self._chunk_ids
            candidate_vectors = self._vectors

            if allowed_chunk_ids is not None:
                selection = np.isin(
                    candidate_ids, np.fromiter(allowed_chunk_ids, dtype=np.int64)
                )
                candidate_ids = candidate_ids[selection]
                candidate_vectors = candidate_vectors[selection]
                if candidate_ids.size == 0:
                    return []

            if candidate_vectors.shape[1] != question_vector.shape[0]:
                LOGGER.error(
                    "Dimensi vektor tidak cocok (indeks %d, kueri %d); jalankan reindex penuh",
                    candidate_vectors.shape[1],
                    question_vector.shape[0],
                )
                return []

            similarities = candidate_vectors @ question_vector.astype(np.float32)
            result_count = min(limit, similarities.shape[0])
            # argpartition finds the top-k without fully sorting the array.
            top_positions = np.argpartition(-similarities, result_count - 1)[:result_count]
            top_positions = top_positions[np.argsort(-similarities[top_positions])]

        return [
            (int(candidate_ids[position]), float(similarities[position]))
            for position in top_positions
        ]


#: Shared stores bound to the active settings and read-only guard.
DOCUMENT_CATALOGUE = DocumentCatalogue(SETTINGS, SOURCE_GUARD)
VECTOR_INDEX = VectorIndex(SETTINGS, SOURCE_GUARD)
