"""Cross-encoder reranking of retrieved candidates.

Hybrid retrieval is tuned for recall: it deliberately returns more candidates
than the answer needs. :class:`CrossEncoderReranker` supplies the precision
half, scoring each ``(question, passage)`` pair jointly instead of comparing
two independently produced vectors. That catches passages that merely share
vocabulary with the question, which is the dominant failure mode when many
documents in a corpus use near-identical boilerplate.

The reranker runs on the CPU, because the GPU's 8 GB belong to the language
model. Run through PyTorch, scoring the 24 candidates takes about 9.4 s per
question on the 20-core i7-14700F (measured in the akreditasi sister project, ``evaluation_results/``); run through
OpenVINO as an int8 model (``RERANK_BACKEND=openvino``) the same model takes
about 3.1 s and keeps 95 % of the same passages. The int8 model is built once
with ``evaluation/openvino_reranker.py`` and ``evaluation/ov_quantize.py``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from .config import Settings
from .documents import RetrievedPassage
from .embeddings import ComputeDeviceResolver

LOGGER = logging.getLogger(__name__)


class OpenVinoCrossEncoder:
    """The reranking model run by OpenVINO instead of PyTorch.

    Offers the one method the reranker uses from sentence-transformers'
    ``CrossEncoder`` -- ``predict(pairs, batch_size=...)`` -- and tokenises the
    same way, so the two backends are interchangeable.

    Args:
        model_directory: Folder holding ``model.xml``/``model.bin``; the
            tokenizer is read from the sibling ``tokenizer`` folder.
        max_length: Token limit per question-passage pair, as for PyTorch.
    """

    def __init__(self, model_directory: Path, max_length: int = 512) -> None:
        import openvino as ov
        from transformers import AutoTokenizer

        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_directory.parent / "tokenizer")
        self._compiled = ov.Core().compile_model(
            str(model_directory / "model.xml"), "CPU", {"PERFORMANCE_HINT": "LATENCY"}
        )
        self._output = self._compiled.output(0)

    def predict(self, pairs, batch_size: int = 8, show_progress_bar: bool = False) -> np.ndarray:
        """Score each ``(question, passage)`` pair; higher means more relevant."""
        scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = self._tokenizer(
                [question for question, _ in batch],
                [passage for _, passage in batch],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="np",
            )
            logits = self._compiled(
                {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
            )[self._output]
            scores.append(np.asarray(logits, dtype=np.float32).reshape(-1))
        return np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)


class CrossEncoderReranker:
    """Re-scores candidate passages against the question, most relevant first.

    Disabled or failed loading is not fatal: :meth:`rerank` then returns the
    candidates in their original fused order, so retrieval still works without
    the reranking model present.
    """

    def __init__(
        self, settings: Settings, device_resolver: ComputeDeviceResolver | None = None
    ) -> None:
        """Record configuration without loading the cross-encoder."""
        self._settings = settings
        self._device_resolver = device_resolver or ComputeDeviceResolver()
        self._model = None
        self._load_attempted = False
        self._load_lock = threading.Lock()
        self._unavailable_reason: str | None = None

    @property
    def is_enabled(self) -> bool:
        """True when reranking is switched on in configuration."""
        return self._settings.reranking_enabled

    @property
    def is_loaded(self) -> bool:
        """True once the cross-encoder weights are resident in memory."""
        return self._model is not None

    @property
    def unavailable_reason(self) -> str | None:
        """Why reranking is not running, or None when it is."""
        if not self._settings.reranking_enabled:
            return "Reranking dinonaktifkan (RERANK_ENABLED=0)"
        return self._unavailable_reason

    def load(self) -> None:
        """Load the cross-encoder once, recording failure rather than raising."""
        if self._model is not None or self._load_attempted:
            return
        with self._load_lock:
            if self._model is not None or self._load_attempted:
                return
            self._load_attempted = True
            if self._settings.reranking_backend == "openvino" and self._load_openvino():
                return
            try:
                from sentence_transformers import CrossEncoder

                device = self._device_resolver.resolve(self._settings.reranking_device)
                LOGGER.info(
                    "Memuat model reranking %s pada %s",
                    self._settings.reranking_model_name,
                    device,
                )
                self._model = CrossEncoder(
                    self._settings.reranking_model_name,
                    device=device,
                    max_length=512,
                    cache_folder=str(self._settings.model_cache_directory),
                )
            except Exception as error:
                self._unavailable_reason = f"Gagal memuat model reranking: {error}"
                LOGGER.warning(self._unavailable_reason)

    def _load_openvino(self) -> bool:
        """Try the OpenVINO model; on any failure, log it and use PyTorch instead.

        Falling back rather than disabling reranking keeps answers the same
        quality when the converted model or the ``openvino`` package is missing.
        """
        model_directory = self._settings.reranking_openvino_directory
        try:
            LOGGER.info("Memuat model reranking OpenVINO dari %s", model_directory)
            self._model = OpenVinoCrossEncoder(model_directory)
            return True
        except Exception as error:
            LOGGER.warning(
                "Reranker OpenVINO tidak dapat dimuat (%s); memakai PyTorch.", error
            )
            return False

    @property
    def backend_name(self) -> str:
        """'openvino' or 'torch' once loaded, for the status panel and logs."""
        if isinstance(self._model, OpenVinoCrossEncoder):
            return "openvino"
        return "torch"

    def rerank(
        self, question: str, candidates: list[RetrievedPassage], keep_count: int
    ) -> list[RetrievedPassage]:
        """Return the ``keep_count`` candidates most relevant to ``question``.

        Each returned passage carries its cross-encoder score in
        :attr:`RetrievedPassage.reranked_score`. When reranking is unavailable,
        the original fused ordering is preserved and scores stay unset.
        """
        if not candidates:
            return []
        if not self._settings.reranking_enabled:
            return candidates[:keep_count]

        self.load()
        if self._model is None:
            return candidates[:keep_count]

        try:
            scores = self._score(question, [candidate.text for candidate in candidates])
        except Exception as error:
            LOGGER.warning("Reranking gagal, memakai urutan awal: %s", error)
            return candidates[:keep_count]

        for candidate, score in zip(candidates, scores):
            candidate.reranked_score = float(score)
        candidates.sort(key=self._sort_key, reverse=True)
        return candidates[:keep_count]

    def _score(self, question: str, texts: list[str]) -> list[float]:
        """Score each passage text against the question, distinct texts once.

        Copies of one document bring the same passage text more than once --
        a fifth of the 24 candidates on the akreditasi sister project -- and identical
        input gives an identical score, so each text is scored once. Texts go
        to the model shortest first, so each batch pads to a similar length
        and less padding is computed. Neither changes the ranking (measured on
        20 questions: the same six passages); together they cut reranking time.
        """
        distinct_texts = sorted(set(texts), key=len)
        distinct_scores = self._model.predict(
            [(question, text) for text in distinct_texts], show_progress_bar=False, batch_size=8
        )
        score_by_text = {text: float(score) for text, score in zip(distinct_texts, distinct_scores)}
        return [score_by_text[text] for text in texts]

    def _sort_key(self, passage: RetrievedPassage) -> float:
        """Ordering key for reranked results: highest cross-encoder score first."""
        return passage.reranked_score if passage.reranked_score is not None else 0.0
