"""Tests for the SQLite catalogue, the vector index and keyword querying."""

from __future__ import annotations

import numpy as np
import pytest

from app.documents import SourceDocument, TextChunk
from app.storage import DocumentCatalogue, KeywordQueryBuilder, VectorIndex


@pytest.fixture
def catalogue(isolated_settings, guard) -> DocumentCatalogue:
    """A catalogue backed by a throwaway database file."""
    instance = DocumentCatalogue(isolated_settings, guard)
    yield instance
    instance.close()


@pytest.fixture
def vector_index(isolated_settings, guard) -> VectorIndex:
    """A vector index with a small dimension count, for fast assertions."""
    isolated_settings.embedding_dimensions = 4
    return VectorIndex(isolated_settings, guard)


def build_document(relative_path: str, size_bytes: int = 100) -> SourceDocument:
    """Construct a SourceDocument without touching the filesystem."""
    from pathlib import Path

    return SourceDocument(
        absolute_path=Path("/corpus") / relative_path,
        relative_path=relative_path,
        display_name=Path(relative_path).name,
        extension=Path(relative_path).suffix.lower(),
        size_bytes=size_bytes,
        modified_timestamp=1_700_000_000.0,
    )


def build_chunks(document_id: str, texts: list[str]) -> list[TextChunk]:
    """Construct chunks for a document from a list of texts."""
    return [
        TextChunk(
            document_id=document_id,
            ordinal=index,
            section_label=f"Halaman {index + 1}",
            text=text,
            token_count=len(text.split()),
        )
        for index, text in enumerate(texts)
    ]


class TestKeywordQueryBuilder:
    """Turning questions into safe FTS5 expressions."""

    def test_joins_meaningful_words_with_or(self):
        expression = KeywordQueryBuilder().build("beban mengajar dosen")
        assert expression == '"beban" OR "mengajar" OR "dosen"'

    def test_removes_indonesian_stop_words(self):
        expression = KeywordQueryBuilder().build("berapa jumlah dosen yang ada")
        assert '"yang"' not in expression
        assert '"dosen"' in expression

    def test_neutralises_fts5_operator_characters(self):
        """Raw punctuation would otherwise be a MATCH syntax error."""
        expression = KeywordQueryBuilder().build('cari "SK-023/PRA" (penting) AND*')
        assert '(' not in expression and '*' not in expression
        assert expression.count('"') % 2 == 0

    def test_keeps_short_acronyms_when_nothing_else_survives(self):
        """"apa itu SK" must still search for "SK" rather than nothing."""
        expression = KeywordQueryBuilder().build("apa itu SK")
        assert '"sk"' in expression

    def test_returns_empty_string_for_unusable_input(self):
        assert KeywordQueryBuilder().build("!!! ???") == ""

    def test_deduplicates_repeated_words(self):
        expression = KeywordQueryBuilder().build("dosen dosen dosen tetap")
        assert expression.count('"dosen"') == 1


class TestCatalogueWriting:
    """Storing and removing documents."""

    def test_saving_returns_one_chunk_id_per_chunk(self, catalogue):
        document = build_document("a/berkas.pdf")
        chunks = build_chunks(document.document_id, ["isi pertama", "isi kedua"])
        chunk_ids = catalogue.save_document(
            document, chunks, section_count=2, character_count=22,
            scanned_section_count=0, ocr_section_count=0, status="indexed",
        )
        assert len(chunk_ids) == 2
        assert all(isinstance(chunk_id, int) for chunk_id in chunk_ids)

    def test_statistics_reflect_saved_documents(self, catalogue):
        document = build_document("a/berkas.pdf", size_bytes=2048)
        catalogue.save_document(
            document, build_chunks(document.document_id, ["satu", "dua", "tiga"]),
            section_count=3, character_count=12,
            scanned_section_count=1, ocr_section_count=0, status="indexed",
        )
        statistics = catalogue.collect_statistics()
        assert statistics["document_count"] == 1
        assert statistics["chunk_count"] == 3
        assert statistics["scanned_section_count"] == 1

    def test_deleting_a_document_returns_its_chunk_ids(self, catalogue):
        document = build_document("a/berkas.pdf")
        chunk_ids = catalogue.save_document(
            document, build_chunks(document.document_id, ["x", "y"]),
            section_count=2, character_count=2,
            scanned_section_count=0, ocr_section_count=0, status="indexed",
        )
        removed = catalogue.delete_documents([document.document_id])
        assert sorted(removed) == sorted(chunk_ids)
        assert catalogue.collect_statistics()["document_count"] == 0

    def test_deleting_also_clears_the_keyword_index(self, catalogue):
        document = build_document("a/berkas.pdf")
        catalogue.save_document(
            document, build_chunks(document.document_id, ["akreditasi institusi unggul"]),
            section_count=1, character_count=27,
            scanned_section_count=0, ocr_section_count=0, status="indexed",
        )
        catalogue.delete_documents([document.document_id])
        assert catalogue.search_by_keyword("akreditasi", limit=10) == []

    def test_clear_removes_everything(self, catalogue):
        for index in range(3):
            document = build_document(f"a/berkas{index}.pdf")
            catalogue.save_document(
                document, build_chunks(document.document_id, ["isi"]),
                section_count=1, character_count=3,
                scanned_section_count=0, ocr_section_count=0, status="indexed",
            )
        catalogue.clear()
        assert catalogue.collect_statistics()["document_count"] == 0

    def test_deleting_nothing_is_a_no_op(self, catalogue):
        assert catalogue.delete_documents([]) == []


class TestChangeDetection:
    """Signatures that drive incremental re-indexing."""

    def test_signature_is_stored_and_returned(self, catalogue):
        document = build_document("a/berkas.pdf", size_bytes=555)
        catalogue.save_document(
            document, build_chunks(document.document_id, ["isi"]),
            section_count=1, character_count=3,
            scanned_section_count=0, ocr_section_count=0, status="indexed",
        )
        signatures = catalogue.load_content_signatures()
        assert signatures[document.document_id] == document.content_signature

    def test_signature_changes_when_the_file_is_edited(self):
        original = build_document("a/berkas.pdf", size_bytes=100)
        edited = build_document("a/berkas.pdf", size_bytes=140)
        assert original.document_id == edited.document_id
        assert original.content_signature != edited.content_signature

    def test_failed_documents_are_excluded_so_they_get_retried(self, catalogue):
        """A transient read failure must not exclude a document permanently.

        Folder renames, Excel locks and Drive re-syncs all cause one-off
        failures; keeping the signature would make the next incremental run skip
        the document forever.
        """
        document = build_document("a/gagal.pdf")
        catalogue.save_document(
            document, [], section_count=0, character_count=0,
            scanned_section_count=0, ocr_section_count=0,
            status="failed", error_message="FileNotFoundError",
        )
        assert document.document_id not in catalogue.load_content_signatures()

    def test_indexed_documents_are_still_skipped(self, catalogue):
        document = build_document("a/baik.pdf")
        catalogue.save_document(
            document, build_chunks(document.document_id, ["isi"]),
            section_count=1, character_count=3,
            scanned_section_count=0, ocr_section_count=0, status="indexed",
        )
        assert document.document_id in catalogue.load_content_signatures()

    def test_empty_documents_are_still_skipped(self, catalogue):
        """An image-only PDF is stably empty; re-reading it every run is waste."""
        document = build_document("a/pindai.pdf")
        catalogue.save_document(
            document, [], section_count=4, character_count=0,
            scanned_section_count=4, ocr_section_count=0, status="empty",
        )
        assert document.document_id in catalogue.load_content_signatures()


class TestCatalogueSearching:
    """Keyword retrieval and passage hydration."""

    @pytest.fixture
    def populated_catalogue(self, catalogue):
        """Two documents in different folders, with distinguishable text."""
        first = build_document("1. Budaya Mutu/ewmp.pdf")
        catalogue.save_document(
            first,
            build_chunks(first.document_id, ["Rata-rata beban EWMP dosen adalah 12,5 SKS"]),
            section_count=1, character_count=42,
            scanned_section_count=0, ocr_section_count=0, status="indexed",
        )
        second = build_document("3. Relevansi Penelitian/publikasi.pdf")
        catalogue.save_document(
            second,
            build_chunks(second.document_id, ["Publikasi penelitian dosen tahun 2025"]),
            section_count=1, character_count=37,
            scanned_section_count=0, ocr_section_count=0, status="indexed",
        )
        return catalogue

    def test_finds_the_matching_document(self, populated_catalogue):
        results = populated_catalogue.search_by_keyword("EWMP", limit=10)
        assert len(results) == 1
        assert "EWMP" in results[0].text

    def test_populates_the_sparse_score(self, populated_catalogue):
        results = populated_catalogue.search_by_keyword("publikasi", limit=10)
        assert results[0].sparse_score is not None

    def test_folder_filter_restricts_results(self, populated_catalogue):
        results = populated_catalogue.search_by_keyword(
            "dosen", limit=10, folder_filter="1. Budaya Mutu"
        )
        assert len(results) == 1
        assert results[0].relative_path.startswith("1. Budaya Mutu")

    def test_unmatched_query_returns_nothing(self, populated_catalogue):
        assert populated_catalogue.search_by_keyword("kriptografi kuantum", limit=10) == []

    def test_malformed_query_does_not_raise(self, populated_catalogue):
        assert populated_catalogue.search_by_keyword('"" OR ( AND *', limit=10) == []

    def test_fetching_passages_by_id_returns_full_records(self, populated_catalogue):
        found = populated_catalogue.search_by_keyword("EWMP", limit=10)
        hydrated = populated_catalogue.fetch_passages_by_id([found[0].chunk_id])
        passage = hydrated[found[0].chunk_id]
        assert passage.document_name == "ewmp.pdf"
        assert passage.section_label == "Halaman 1"

    def test_listing_folders_returns_distinct_names(self, populated_catalogue):
        assert populated_catalogue.list_folders() == [
            "1. Budaya Mutu",
            "3. Relevansi Penelitian",
        ]

    def test_chunk_ids_can_be_fetched_per_folder(self, populated_catalogue):
        chunk_ids = populated_catalogue.fetch_chunk_ids_for_folder("1. Budaya Mutu")
        assert len(chunk_ids) == 1

    def test_document_path_lookup(self, populated_catalogue):
        documents = populated_catalogue.list_documents()
        found = populated_catalogue.find_document_path(documents[0]["document_id"])
        assert found == documents[0]["absolute_path"]

    def test_document_path_lookup_returns_none_when_absent(self, populated_catalogue):
        assert populated_catalogue.find_document_path("tidak-ada") is None


class TestVectorIndex:
    """The NumPy-backed dense index."""

    def test_starts_empty(self, vector_index):
        assert vector_index.size == 0

    def test_add_grows_the_index(self, vector_index):
        vectors = np.eye(3, 4, dtype=np.float32)
        vector_index.add([1, 2, 3], vectors)
        assert vector_index.size == 3

    def test_search_ranks_the_nearest_vector_first(self, vector_index):
        vectors = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32
        )
        vector_index.add([10, 20, 30], vectors)

        query = np.array([0, 1, 0, 0], dtype=np.float32)
        results = vector_index.search(query, limit=3)

        assert results[0][0] == 20
        assert results[0][1] == pytest.approx(1.0)

    def test_search_respects_the_limit(self, vector_index):
        vector_index.add([1, 2, 3], np.eye(3, 4, dtype=np.float32))
        assert len(vector_index.search(np.array([1, 0, 0, 0], dtype=np.float32), limit=2)) == 2

    def test_search_on_empty_index_returns_nothing(self, vector_index):
        assert vector_index.search(np.zeros(4, dtype=np.float32), limit=5) == []

    def test_allowed_ids_restrict_the_search(self, vector_index):
        vector_index.add([1, 2, 3], np.eye(3, 4, dtype=np.float32))
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = vector_index.search(query, limit=5, allowed_chunk_ids={2, 3})
        assert {chunk_id for chunk_id, _score in results} == {2, 3}

    def test_empty_allowed_ids_returns_nothing(self, vector_index):
        vector_index.add([1, 2], np.eye(2, 4, dtype=np.float32))
        assert vector_index.search(np.zeros(4, dtype=np.float32), 5, allowed_chunk_ids=set()) == []

    def test_remove_drops_only_the_named_vectors(self, vector_index):
        vector_index.add([1, 2, 3], np.eye(3, 4, dtype=np.float32))
        vector_index.remove([2])
        assert vector_index.size == 2
        remaining = vector_index.search(np.array([0, 1, 0, 0], dtype=np.float32), limit=3)
        assert 2 not in {chunk_id for chunk_id, _score in remaining}

    def test_clear_empties_the_index(self, vector_index):
        vector_index.add([1, 2], np.eye(2, 4, dtype=np.float32))
        vector_index.clear()
        assert vector_index.size == 0

    def test_dimension_mismatch_is_reported_rather_than_raised(self, vector_index):
        """A stale index must degrade to no results, not crash a live query."""
        vector_index.add([1], np.ones((1, 4), dtype=np.float32))
        assert vector_index.search(np.ones(8, dtype=np.float32), limit=3) == []

    def test_vectors_survive_a_save_and_reload(self, isolated_settings, guard):
        isolated_settings.embedding_dimensions = 4
        first = VectorIndex(isolated_settings, guard)
        first.add([7, 8], np.eye(2, 4, dtype=np.float32))
        first.save()

        second = VectorIndex(isolated_settings, guard)
        assert second.size == 2
        results = second.search(np.array([1, 0, 0, 0], dtype=np.float32), limit=1)
        assert results[0][0] == 7

    def test_saving_refuses_a_path_inside_the_corpus(self, isolated_settings, guard):
        """A DATA_DIR pointed at the corpus must fail at construction."""
        from app.safety import SourceWriteAttempt

        isolated_settings.data_directory = isolated_settings.source_directory / "idx"
        with pytest.raises(SourceWriteAttempt):
            VectorIndex(isolated_settings, guard)


class TestWritingOnlyWhatChanged:
    """An indexing run that found nothing new must not rewrite the stores.

    On a large corpus the vector archive runs to hundreds of megabytes and the
    full-text index's 'optimize' rewrites it whole; doing either for nothing
    kept a slow disk busy for minutes after every automatic run.
    """

    def test_unchanged_vectors_are_not_saved(self, vector_index, isolated_settings):
        vector_index.save()
        assert not isolated_settings.vector_store_path.exists()

    def test_added_vectors_are_saved_once(self, vector_index, isolated_settings):
        vector_index.add([1, 2], np.eye(2, 4, dtype=np.float32))
        assert vector_index.has_unsaved_changes
        vector_index.save()
        written_at = isolated_settings.vector_store_path.stat().st_mtime_ns
        assert not vector_index.has_unsaved_changes

        vector_index.save()
        assert isolated_settings.vector_store_path.stat().st_mtime_ns == written_at

    def test_removing_unknown_ids_changes_nothing(self, vector_index):
        vector_index.add([1], np.eye(1, 4, dtype=np.float32))
        vector_index.save()
        vector_index.remove([99])
        assert not vector_index.has_unsaved_changes

    def test_removing_a_vector_marks_the_index_changed(self, vector_index):
        vector_index.add([1, 2], np.eye(2, 4, dtype=np.float32))
        vector_index.save()
        vector_index.remove([2])
        assert vector_index.has_unsaved_changes

    def test_optimise_runs_only_after_passages_change(self, catalogue):
        statements: list[str] = []
        catalogue._connection.set_trace_callback(statements.append)

        catalogue.optimise()
        assert not [s for s in statements if "optimize" in s]

        document = build_document("a.txt")
        catalogue.save_document(document, build_chunks(document.document_id, ["satu dua tiga"]),
                                section_count=1, character_count=13, scanned_section_count=0,
                                ocr_section_count=0, status="indexed")
        catalogue.optimise()
        catalogue.optimise()
        assert len([s for s in statements if "optimize" in s]) == 1
