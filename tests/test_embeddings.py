"""Tests for the question encoder's backends and the indexing child process.

No model is loaded. sentence-transformers and the OpenVINO runtime are replaced
by stand-ins, and the child process runs a few lines of Python that speak the
worker's protocol, so what is checked is the wiring: which backend is chosen,
that a missing OpenVINO model falls back to PyTorch, and that the child
encodes batch by batch, reports progress and ends when unloaded.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from app.embeddings import OpenVinoSentenceEncoder, SubprocessEmbedder, TextEmbedder


class FixedDevice:
    """A device resolver that answers the same device whatever is asked."""

    def __init__(self, device: str) -> None:
        self.device = device

    def resolve(self, requested_device: str) -> str:
        return self.device


class StandInSentenceTransformer:
    """Encodes a text as a one-hot vector chosen by its length."""

    def __init__(self, *args, **kwargs) -> None:
        self.tokenizer = object()

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, **options):
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, len(text) % 4] = 1.0
        return vectors


@pytest.fixture
def stand_in_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StandInSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


class TestQuestionEncoderBackend:
    def test_default_backend_is_torch(self, isolated_settings):
        """int8 saved no RAM for the question encoder, so it is opt-in."""
        assert isolated_settings.query_embedding_backend == "torch"

    def test_missing_openvino_model_falls_back_to_torch(
        self, isolated_settings, tmp_path, stand_in_sentence_transformers
    ):
        embedder = TextEmbedder(isolated_settings, FixedDevice("cpu"), device="cpu",
                                backend="openvino", openvino_directory=tmp_path / "missing")
        vector = embedder.encode_question("apa itu EWMP")
        assert embedder.backend_name == "torch"
        assert vector.shape == (4,)

    def test_openvino_is_used_when_its_model_loads(
        self, isolated_settings, tmp_path, monkeypatch, stand_in_sentence_transformers
    ):
        monkeypatch.setattr(OpenVinoSentenceEncoder, "__init__",
                            lambda self, directory, max_length=512: None)
        embedder = TextEmbedder(isolated_settings, FixedDevice("cpu"), device="cpu",
                                backend="openvino", openvino_directory=tmp_path)
        embedder.load()
        assert embedder.backend_name == "openvino"

    def test_openvino_is_not_used_on_the_gpu(
        self, isolated_settings, tmp_path, monkeypatch, stand_in_sentence_transformers
    ):
        monkeypatch.setattr(OpenVinoSentenceEncoder, "__init__",
                            lambda self, directory, max_length=512: None)
        embedder = TextEmbedder(isolated_settings, FixedDevice("cuda"), device="cuda",
                                backend="openvino", openvino_directory=tmp_path)
        embedder.load()
        assert embedder.backend_name == "torch"


class TestOpenVinoSentenceEncoder:
    def make_encoder(self, batch_log: list[int]) -> OpenVinoSentenceEncoder:
        """An encoder whose tokenizer and compiled model are stand-ins.

        The stand-in model returns, for each text, a vector whose first entry
        is the number of tokens and whose second is 1, so results can be
        checked after normalisation.
        """
        encoder = OpenVinoSentenceEncoder.__new__(OpenVinoSentenceEncoder)
        encoder._max_length = 512

        def tokenizer(texts, **options):
            lengths = [len(text.split()) for text in texts]
            width = max(lengths)
            mask = np.array([[1] * n + [0] * (width - n) for n in lengths], dtype=np.int64)
            return {"input_ids": mask.copy(), "attention_mask": mask}

        def compiled(inputs):
            batch_log.append(inputs["attention_mask"].shape[0])
            counts = inputs["attention_mask"].sum(axis=1).astype(np.float32)
            return {"vectors": np.stack([counts, np.ones_like(counts)], axis=1)}

        encoder.tokenizer = tokenizer
        encoder._compiled = compiled
        encoder._output = "vectors"
        encoder.get_sentence_embedding_dimension = lambda: 2
        return encoder

    def test_vectors_follow_text_order_across_batches_and_are_unit_length(self):
        batches: list[int] = []
        encoder = self.make_encoder(batches)
        vectors = encoder.encode(["a", "a b c", "a b"], batch_size=2)

        assert batches == [2, 1]
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
        expected = np.array([[1, 1], [3, 1], [2, 1]], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        assert np.allclose(vectors, expected)

    def test_no_texts_gives_no_vectors(self):
        assert self.make_encoder([]).encode([]).shape == (0, 2)


#: A stand-in for app.embedding_worker that speaks the same protocol.
FAKE_WORKER = r"""
import pickle, sys
import numpy as np
requests, replies = sys.stdin.buffer, sys.stdout.buffer
configuration = pickle.load(requests)
if configuration["model"] == "fails-to-load":
    pickle.dump(("error", "cannot load"), replies); replies.flush(); sys.exit(1)
pickle.dump(("ready", 4), replies); replies.flush()
while True:
    try:
        passages = pickle.load(requests)
    except EOFError:
        break
    if passages is None:
        break
    vectors = np.zeros((len(passages), 4), dtype=np.float32)
    for row, text in enumerate(passages):
        vectors[row, len(text) % 4] = 1.0
    pickle.dump(("vectors", vectors), replies); replies.flush()
"""


class TestSubprocessEmbedder:
    def make_embedder(self, settings) -> SubprocessEmbedder:
        return SubprocessEmbedder(settings, FixedDevice("cuda"),
                                  command=[sys.executable, "-c", FAKE_WORKER])

    def test_encodes_batch_by_batch_and_reports_progress(self, isolated_settings):
        isolated_settings.embedding_batch_size = 2
        embedder = self.make_embedder(isolated_settings)
        progress = []
        try:
            vectors = embedder.encode_passages(
                ["a", "bb", "ccc"], lambda done, total: progress.append((done, total))
            )
            assert embedder.is_loaded
            assert embedder.dimensions == 4
        finally:
            embedder.unload()

        assert vectors.shape == (3, 4)
        assert [int(np.argmax(row)) for row in vectors] == [1, 2, 3]
        assert progress == [(2, 3), (3, 3)]

    def test_unload_ends_the_child_process(self, isolated_settings):
        embedder = self.make_embedder(isolated_settings)
        embedder.load()
        process = embedder._process
        embedder.unload()
        assert not embedder.is_loaded
        assert process.poll() is not None

    def test_a_child_that_fails_to_load_raises_and_is_cleaned_up(self, isolated_settings):
        isolated_settings.embedding_model_name = "fails-to-load"
        embedder = self.make_embedder(isolated_settings)
        with pytest.raises(RuntimeError, match="cannot load"):
            embedder.load()
        assert not embedder.is_loaded

    def test_no_passages_starts_nothing_heavy(self, isolated_settings):
        embedder = self.make_embedder(isolated_settings)
        try:
            assert embedder.encode_passages([]).shape == (0, 4)
        finally:
            embedder.unload()

    def test_reports_the_device_it_encodes_on(self, isolated_settings):
        assert self.make_embedder(isolated_settings).device == "cuda"


class TestQuestionEncoderThreads:
    def test_cpu_threads_limit_pytorch_on_the_cpu(
        self, isolated_settings, monkeypatch, stand_in_sentence_transformers
    ):
        import torch

        calls = []
        monkeypatch.setattr(torch, "set_num_threads", calls.append)
        TextEmbedder(isolated_settings, FixedDevice("cpu"), device="cpu", cpu_threads=3).load()
        assert calls == [3]

    def test_no_limit_by_default(self, isolated_settings, monkeypatch, stand_in_sentence_transformers):
        import torch

        calls = []
        monkeypatch.setattr(torch, "set_num_threads", calls.append)
        TextEmbedder(isolated_settings, FixedDevice("cpu"), device="cpu").load()
        assert calls == []

    def test_openmp_threads_do_not_spin_after_work(self):
        import os

        import app.config  # noqa: F401  (sets the default on import)

        assert os.environ.get("KMP_BLOCKTIME") is not None
