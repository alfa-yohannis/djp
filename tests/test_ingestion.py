"""Tests for incremental indexing and store reconciliation.

The pipeline is exercised with a stub embedder so no model is downloaded; what
is under test here is the bookkeeping — which documents get re-read, which are
skipped, and how the two stores are kept consistent when a run is interrupted.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.extractors import ExtractorRegistry
from app.ingestion import (
    AutomaticIngestion,
    IngestionPipeline,
    IngestionProgress,
    IngestionRunner,
)
from app.storage import DocumentCatalogue, VectorIndex


class StubEmbedder:
    """Produces deterministic unit vectors without loading a model."""

    def __init__(self, dimensions: int = 4) -> None:
        """Record the vector width and start with an empty call log."""
        self._dimensions = dimensions
        self.encoded_batches: list[list[str]] = []

    def build_token_counter(self):
        """Use the tokenizer-free counter, as the real embedder does when unloaded."""
        from app.chunking import CharacterRatioTokenCounter

        return CharacterRatioTokenCounter()

    def encode_passages(self, passages, progress_callback=None):
        """Return one unit vector per passage, derived from its length."""
        self.encoded_batches.append(list(passages))
        vectors = np.zeros((len(passages), self._dimensions), dtype=np.float32)
        for index, passage in enumerate(passages):
            vectors[index][len(passage) % self._dimensions] = 1.0
        return vectors

    def encode_question(self, question: str):
        """Return a unit vector for a question."""
        vector = np.zeros(self._dimensions, dtype=np.float32)
        vector[len(question) % self._dimensions] = 1.0
        return vector


@pytest.fixture
def pipeline(isolated_settings, guard, sample_corpus):
    """A pipeline wired to throwaway stores and a stub embedder."""
    isolated_settings.embedding_dimensions = 4
    # The fixture documents are deliberately short; the production floor of 80
    # characters would classify them all as empty and defeat the point.
    isolated_settings.minimum_chunk_characters = 10
    catalogue = DocumentCatalogue(isolated_settings, guard)
    vector_index = VectorIndex(isolated_settings, guard)
    embedder = StubEmbedder(dimensions=4)
    registry = ExtractorRegistry(isolated_settings, guard)

    instance = IngestionPipeline(
        settings=isolated_settings,
        catalogue=catalogue,
        vector_index=vector_index,
        embedder=embedder,
        registry=registry,
        guard=guard,
    )
    instance.catalogue = catalogue
    instance.vector_index = vector_index
    instance.embedder = embedder
    yield instance
    catalogue.close()


class TestIngestionProgress:
    """The counters the interface polls during a run."""

    def test_starts_idle(self):
        progress = IngestionProgress()
        assert progress.is_running is False
        assert progress.phase == "idle"

    def test_reset_marks_the_run_active(self):
        progress = IngestionProgress()
        progress.reset(documents_total=10)
        assert progress.is_running is True
        assert progress.documents_total == 10

    def test_finish_records_completion(self):
        progress = IngestionProgress()
        progress.reset(documents_total=1)
        progress.finish()
        assert progress.is_running is False
        assert progress.phase == "completed"

    def test_finish_with_an_error_records_failure(self):
        progress = IngestionProgress()
        progress.reset(documents_total=1)
        progress.finish(error_message="disk penuh")
        assert progress.phase == "failed"
        assert progress.error_message == "disk penuh"

    def test_message_history_is_capped(self):
        """An unbounded log would grow without limit on a long run."""
        progress = IngestionProgress()
        progress.reset(documents_total=1)
        for index in range(200):
            progress.record_message(f"baris {index}")
        assert len(progress.as_dictionary()["recent_messages"]) <= (
            IngestionProgress.MESSAGE_HISTORY_LIMIT
        )

    def test_snapshot_is_plain_data(self):
        progress = IngestionProgress()
        progress.reset(documents_total=3)
        snapshot = progress.as_dictionary()
        assert isinstance(snapshot, dict)
        assert "_lock" not in snapshot


class TestFirstRun:
    """Building an index from an empty state."""

    def test_indexes_every_eligible_document(self, pipeline):
        summary = pipeline.run()
        assert summary["documents_indexed"] > 0
        assert summary["documents_failed"] == 0

    def test_creates_chunks_and_vectors_in_step(self, pipeline):
        pipeline.run()
        statistics = pipeline.catalogue.collect_statistics()
        assert statistics["chunk_count"] > 0
        assert pipeline.vector_index.size == statistics["chunk_count"]

    def test_excludes_the_developer_directory(self, pipeline):
        pipeline.run()
        paths = [record["relative_path"] for record in pipeline.catalogue.list_documents()]
        assert not any("zdev" in path for path in paths)

    def test_persists_the_vector_archive(self, pipeline, isolated_settings):
        pipeline.run()
        assert isolated_settings.vector_store_path.exists()

    def test_leaves_the_corpus_unmodified(self, pipeline, sample_corpus):
        """A full ingest must not alter a single byte of the source."""
        before = {p: p.read_bytes() for p in sample_corpus.rglob("*") if p.is_file()}
        pipeline.run()
        after = {p: p.read_bytes() for p in sample_corpus.rglob("*") if p.is_file()}
        assert before == after


class TestIncrementalRun:
    """Re-running over a mostly unchanged corpus."""

    def test_second_run_skips_unchanged_documents(self, pipeline):
        first = pipeline.run()
        second = pipeline.run()
        assert second["documents_indexed"] == 0
        assert second["documents_skipped"] == first["documents_total"]

    def test_second_run_does_not_re_embed(self, pipeline):
        pipeline.run()
        calls_after_first = len(pipeline.embedder.encoded_batches)
        pipeline.run()
        assert len(pipeline.embedder.encoded_batches) == calls_after_first

    def test_a_modified_document_is_re_indexed(self, pipeline, sample_corpus):
        pipeline.run()
        target = sample_corpus / "laporan.md"
        target.write_text("# Laporan Akreditasi\n\nIsi yang sudah diperbarui.", encoding="utf-8")
        summary = pipeline.run()
        assert summary["documents_indexed"] == 1

    def test_a_new_document_is_picked_up(self, pipeline, sample_corpus):
        pipeline.run()
        (sample_corpus / "tambahan.txt").write_text(
            "Dokumen tambahan tentang capaian pembelajaran lulusan.", encoding="utf-8"
        )
        assert pipeline.run()["documents_indexed"] == 1

    def test_a_deleted_document_is_purged(self, pipeline, sample_corpus):
        pipeline.run()
        (sample_corpus / "laporan.md").unlink()
        summary = pipeline.run()
        assert summary["documents_removed"] == 1

    def test_a_failed_document_that_vanished_is_purged(self, pipeline, sample_corpus):
        """The renamed-folder case: entries that are both failed and gone.

        Failed documents are excluded from the signature map so they get
        retried; purging must therefore read the full identifier list, or these
        would remain in the index forever.
        """
        pipeline.run()
        target = sample_corpus / "laporan.md"
        document_id = next(
            record["document_id"]
            for record in pipeline.catalogue.list_documents()
            if record["display_name"] == "laporan.md"
        )
        # Mark it failed the way a transient read error would.
        pipeline.catalogue._connection.execute(
            "UPDATE documents SET status='failed', error_message='FileNotFoundError' "
            "WHERE document_id = ?",
            (document_id,),
        )
        pipeline.catalogue._connection.commit()
        target.unlink()

        summary = pipeline.run()
        assert summary["documents_removed"] == 1
        assert document_id not in pipeline.catalogue.list_document_ids()

    def test_a_failed_document_still_present_is_retried(self, pipeline, sample_corpus):
        """The complement: a failure that is still on disk must be re-read."""
        pipeline.run()
        document_id = next(
            record["document_id"]
            for record in pipeline.catalogue.list_documents()
            if record["display_name"] == "laporan.md"
        )
        pipeline.catalogue._connection.execute(
            "UPDATE documents SET status='failed' WHERE document_id = ?", (document_id,)
        )
        pipeline.catalogue._connection.commit()

        summary = pipeline.run()
        assert summary["documents_indexed"] == 1
        assert summary["documents_removed"] == 0

    def test_purging_also_removes_its_vectors(self, pipeline, sample_corpus):
        pipeline.run()
        vectors_before = pipeline.vector_index.size
        (sample_corpus / "laporan.md").unlink()
        pipeline.run()
        assert pipeline.vector_index.size < vectors_before

    def test_a_half_synced_drive_does_not_purge_the_index(self, pipeline, sample_corpus):
        """A shared drive mid-sync looks identical to a corpus that was emptied.

        Deleting on that evidence would destroy a working index and force a full
        re-embed, so an implausibly small scan is refused instead.
        """
        first = pipeline.run()
        assert first["documents_indexed"] > 0

        for path in sample_corpus.rglob("*"):
            if path.is_file() and path.suffix in {".txt", ".csv", ".md", ".json"}:
                path.unlink()

        summary = pipeline.run()
        assert summary["documents_removed"] == 0
        assert pipeline.catalogue.collect_statistics()["document_count"] > 0

    def test_a_genuine_deletion_is_still_purged(self, pipeline, sample_corpus):
        """The guard must not block ordinary, proportionate deletions."""
        pipeline.run()
        (sample_corpus / "laporan.md").unlink()
        assert pipeline.run()["documents_removed"] == 1

    def test_rebuild_re_reads_everything(self, pipeline):
        first = pipeline.run()
        rebuilt = pipeline.run(force_full_rebuild=True)
        assert rebuilt["documents_indexed"] == first["documents_indexed"]


class TestStoreReconciliation:
    """Recovering when the catalogue and the vector archive drift apart."""

    def chunk_ids_by_document(self, pipeline) -> dict[str, list[int]]:
        """Group every indexed chunk id by the document it belongs to."""
        known_ids = list(pipeline.vector_index.known_chunk_ids())
        grouped: dict[str, list[int]] = {}
        for chunk_id, passage in pipeline.catalogue.fetch_passages_by_id(known_ids).items():
            grouped.setdefault(passage.document_id, []).append(chunk_id)
        return grouped

    def test_documents_missing_vectors_are_detected(self, pipeline):
        """Simulates a run killed between the last vector flush and the end."""
        pipeline.run()
        documents_with_chunks = set(self.chunk_ids_by_document(pipeline))
        pipeline.vector_index.clear()

        stranded = pipeline.catalogue.find_documents_with_missing_vectors(
            pipeline.vector_index.known_chunk_ids()
        )
        assert stranded == documents_with_chunks

    def test_stranded_documents_are_re_indexed_not_skipped(self, pipeline):
        """Without reconciliation these would be skipped forever."""
        first = pipeline.run()
        pipeline.vector_index.clear()

        recovered = pipeline.run()
        assert recovered["documents_indexed"] == first["documents_indexed"]

    def test_reconciliation_restores_vector_coverage(self, pipeline):
        pipeline.run()
        expected_chunks = pipeline.catalogue.collect_statistics()["chunk_count"]
        pipeline.vector_index.clear()
        pipeline.run()
        assert pipeline.vector_index.size == expected_chunks

    def test_orphan_vectors_are_discarded(self, pipeline):
        """A vector whose chunk row is gone crowds out real top-k candidates."""
        pipeline.run()
        import numpy as np

        pipeline.vector_index.add([999_001, 999_002], np.eye(2, 4, dtype=np.float32))
        assert pipeline.vector_index.size == pipeline.catalogue.collect_statistics()["chunk_count"] + 2

        pipeline.run()
        assert pipeline.vector_index.size == pipeline.catalogue.collect_statistics()["chunk_count"]

    def test_a_consistent_index_needs_no_recovery(self, pipeline):
        pipeline.run()
        stranded = pipeline.catalogue.find_documents_with_missing_vectors(
            pipeline.vector_index.known_chunk_ids()
        )
        assert stranded == set()

    def test_partial_vector_loss_only_re_indexes_the_affected_document(self, pipeline):
        """Recovery is targeted: unaffected documents are still skipped."""
        pipeline.run()
        grouped = self.chunk_ids_by_document(pipeline)
        victim_document_id = sorted(grouped)[0]
        pipeline.vector_index.remove(grouped[victim_document_id])

        summary = pipeline.run()
        assert summary["documents_indexed"] == 1


class TestMultipleRootsIngestion:
    """Indexing the primary corpus together with the local copy."""

    def test_documents_from_both_roots_reach_the_index(self, pipeline, extra_corpus_root):
        pipeline.run()
        paths = [record["relative_path"] for record in pipeline.catalogue.list_documents()]
        assert any(path.startswith("corpus-lokal") for path in paths)
        assert any(not path.startswith("corpus-lokal") for path in paths)

    def test_the_local_copy_appears_as_its_own_folder_filter(self, pipeline, extra_corpus_root):
        pipeline.run()
        assert "corpus-lokal" in pipeline.catalogue.list_folders()

    def test_the_lecturer_mapping_is_searchable(self, pipeline, extra_corpus_root):
        """The SINTA and Scholar mapping is why the extra root exists at all."""
        pipeline.run()
        results = pipeline.catalogue.search_by_keyword("sinta scholar dosen", limit=10)
        assert any("sinta" in passage.text.lower() for passage in results)


class StubRunner:
    """Records start requests instead of indexing, and can pretend to be busy."""

    def __init__(self) -> None:
        """Start idle with no runs requested."""
        self.is_running = False
        self.starts = 0

    def start(self, force_full_rebuild: bool = False) -> bool:
        """Count the request; refuse it while busy, as the real runner does."""
        if self.is_running:
            return False
        self.starts += 1
        return True


class TestAutomaticIngestion:
    """Indexing on its own when the crawler adds files to the corpus."""

    def test_the_first_check_catches_up(self, pipeline):
        runner = StubRunner()
        watcher = AutomaticIngestion(pipeline, runner, interval_seconds=60)
        assert watcher.check_once() is True
        assert runner.starts == 1

    def test_an_unchanged_corpus_starts_nothing(self, pipeline):
        runner = StubRunner()
        watcher = AutomaticIngestion(pipeline, runner, interval_seconds=60)
        watcher.check_once()
        assert watcher.check_once() is False
        assert runner.starts == 1

    def test_a_new_file_starts_a_run(self, pipeline, sample_corpus):
        runner = StubRunner()
        watcher = AutomaticIngestion(pipeline, runner, interval_seconds=60)
        watcher.check_once()
        (sample_corpus / "tambahan.txt").write_text(
            "Halaman baru hasil crawl tentang tarif PPh Pasal 21.", encoding="utf-8"
        )
        assert watcher.check_once() is True
        assert runner.starts == 2

    def test_a_busy_runner_is_left_alone(self, pipeline):
        runner = StubRunner()
        runner.is_running = True
        watcher = AutomaticIngestion(pipeline, runner, interval_seconds=60)
        assert watcher.check_once() is False
        runner.is_running = False
        assert watcher.check_once() is True

    def test_an_interval_of_zero_switches_it_off(self, pipeline):
        watcher = AutomaticIngestion(pipeline, StubRunner(), interval_seconds=0)
        assert watcher.start() is False

    def test_a_started_run_makes_the_new_file_searchable(self, pipeline, sample_corpus):
        runner = IngestionRunner(pipeline)
        watcher = AutomaticIngestion(pipeline, runner, interval_seconds=60)
        watcher.check_once()
        runner._worker_thread.join(timeout=60)
        (sample_corpus / "tambahan.txt").write_text(
            "Halaman baru hasil crawl tentang tarif PPh Pasal 21.", encoding="utf-8"
        )
        assert watcher.check_once() is True
        runner._worker_thread.join(timeout=60)
        results = pipeline.catalogue.search_by_keyword("tarif PPh Pasal 21", limit=5)
        assert any("tarif" in passage.text.lower() for passage in results)
