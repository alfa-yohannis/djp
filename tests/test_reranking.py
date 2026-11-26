"""Tests for the reranker's backends: PyTorch and OpenVINO.

No model is loaded: the OpenVINO runtime, the tokenizer and sentence-
transformers' CrossEncoder are replaced by small stand-ins, so the suite stays
offline and fast. What is checked is the wiring -- which backend is chosen,
that a missing OpenVINO model falls back to PyTorch, and that scores come back
in the order of the pairs, batch after batch.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from app.documents import RetrievedPassage
from app.reranking import CrossEncoderReranker, OpenVinoCrossEncoder


def make_passage(chunk_id: int, text: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id, document_id=f"doc-{chunk_id}", document_name=f"Dokumen {chunk_id}",
        relative_path=f"doc-{chunk_id}.pdf", absolute_path=f"/x/doc-{chunk_id}.pdf",
        section_label="Halaman 1", text=text,
    )


class LengthScoringModel:
    """Scores a pair by the length of its passage: longer is more relevant."""

    def __init__(self, *args, **kwargs) -> None:
        self.batch_sizes: list[int] = []

    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        self.batch_sizes.append(batch_size)
        return np.array([len(passage) for _, passage in pairs], dtype=np.float32)


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    """Replace sentence-transformers' CrossEncoder with LengthScoringModel."""
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = LengthScoringModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


class TestBackendSelection:
    def test_default_backend_is_openvino(self, isolated_settings):
        assert isolated_settings.reranking_backend == "openvino"

    def test_torch_backend_when_configured(self, isolated_settings, fake_cross_encoder):
        isolated_settings.reranking_backend = "torch"
        reranker = CrossEncoderReranker(isolated_settings)
        reranker.load()
        assert reranker.backend_name == "torch"

    def test_openvino_backend_is_used_when_its_model_loads(
        self, isolated_settings, monkeypatch, fake_cross_encoder
    ):
        isolated_settings.reranking_backend = "openvino"
        monkeypatch.setattr("app.reranking.OpenVinoCrossEncoder.__init__",
                            lambda self, directory, max_length=512: None)
        reranker = CrossEncoderReranker(isolated_settings)
        reranker.load()
        assert reranker.backend_name == "openvino"

    def test_missing_openvino_model_falls_back_to_torch(
        self, isolated_settings, tmp_path, fake_cross_encoder
    ):
        isolated_settings.reranking_backend = "openvino"
        isolated_settings.reranking_openvino_directory = tmp_path / "does-not-exist"
        reranker = CrossEncoderReranker(isolated_settings)
        passages = [make_passage(1, "pendek"), make_passage(2, "bagian yang jauh lebih panjang")]

        kept = reranker.rerank("pertanyaan", passages, keep_count=1)

        assert reranker.backend_name == "torch"
        assert reranker.unavailable_reason is None
        assert [p.chunk_id for p in kept] == [2]


class TestOpenVinoCrossEncoder:
    def make_encoder(self, batch_log: list[int]) -> OpenVinoCrossEncoder:
        """An encoder whose tokenizer and compiled model are stand-ins.

        The fake "model" returns, for each row, the number of non-padding
        tokens, so the expected score of every pair is known.
        """
        encoder = OpenVinoCrossEncoder.__new__(OpenVinoCrossEncoder)
        encoder._max_length = 512

        def tokenizer(questions, passages, **kwargs):
            lengths = [len(q.split()) + len(p.split()) for q, p in zip(questions, passages)]
            width = max(lengths)
            mask = np.array([[1] * n + [0] * (width - n) for n in lengths], dtype=np.int64)
            return {"input_ids": mask.copy(), "attention_mask": mask}

        def compiled(inputs):
            batch_log.append(inputs["attention_mask"].shape[0])
            return {"logits": inputs["attention_mask"].sum(axis=1, keepdims=True).astype(np.float32)}

        encoder._tokenizer = tokenizer
        encoder._compiled = compiled
        encoder._output = "logits"
        return encoder

    def test_scores_follow_pair_order_across_batches(self):
        batches: list[int] = []
        encoder = self.make_encoder(batches)
        pairs = [("a", "satu"), ("a b", "dua tiga"), ("a", "x y z w"), ("q", "r")]

        scores = encoder.predict(pairs, batch_size=3)

        assert batches == [3, 1]
        assert scores.tolist() == [2.0, 4.0, 5.0, 2.0]

    def test_no_pairs_gives_no_scores(self):
        encoder = self.make_encoder([])
        assert encoder.predict([], batch_size=8).shape == (0,)


class RecordingModel:
    """Scores a pair by the length of its passage, and records what it was sent."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        self.sent.extend(passage for _, passage in pairs)
        return np.array([len(passage) for _, passage in pairs], dtype=np.float32)


class TestScoringOnlyWhatIsNeeded:
    """Duplicates are scored once and texts go shortest first; rankings stay the same."""

    def make_reranker(self, isolated_settings, model):
        reranker = CrossEncoderReranker(isolated_settings)
        reranker._model = model
        reranker._load_attempted = True
        return reranker

    def test_each_distinct_text_is_scored_once_shortest_first(self, isolated_settings):
        model = RecordingModel()
        reranker = self.make_reranker(isolated_settings, model)
        passages = [make_passage(1, "teks yang paling panjang di sini"), make_passage(2, "pendek"),
                    make_passage(3, "teks yang paling panjang di sini"), make_passage(4, "sedang saja")]

        reranker.rerank("pertanyaan", passages, keep_count=4)

        assert model.sent == ["pendek", "sedang saja", "teks yang paling panjang di sini"]

    def test_duplicates_share_their_text_score_and_order_is_kept(self, isolated_settings):
        reranker = self.make_reranker(isolated_settings, RecordingModel())
        passages = [make_passage(1, "sedang saja"), make_passage(2, "teks yang paling panjang"),
                    make_passage(3, "pendek"), make_passage(4, "teks yang paling panjang")]

        kept = reranker.rerank("pertanyaan", passages, keep_count=3)

        assert [p.chunk_id for p in kept] == [2, 4, 1]
        assert kept[0].reranked_score == kept[1].reranked_score == len("teks yang paling panjang")
