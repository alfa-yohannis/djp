"""Dense vector encoding of chunks and questions.

:class:`TextEmbedder` wraps a SentenceTransformer model — BGE-M3 by default,
chosen because it handles Indonesian well and produces a single 1024-dimension
vector per passage. Vectors are L2-normalised on the way out, which lets the
vector index compute cosine similarity as a plain dot product.

:class:`ComputeDeviceResolver` centralises the ``"auto"`` device rule so the
embedder and the reranker resolve hardware the same way.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .config import Settings

LOGGER = logging.getLogger(__name__)


class ComputeDeviceResolver:
    """Turns a configured device name into a concrete torch device string."""

    def resolve(self, requested_device: str) -> str:
        """Return a usable device, expanding ``"auto"`` to cuda when present."""
        normalised = (requested_device or "auto").strip().lower()
        if normalised != "auto":
            return normalised
        return "cuda" if self.is_cuda_available() else "cpu"

    def is_cuda_available(self) -> bool:
        """True when a CUDA-capable GPU is visible to torch."""
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def describe_hardware(self) -> str:
        """Human-readable summary of the active accelerator, for the status panel."""
        try:
            import torch

            if not torch.cuda.is_available():
                return "CPU"
            device_name = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return f"{device_name} ({total_bytes / (1024 ** 3):.1f} GB VRAM)"
        except Exception:
            return "CPU"


class TextEmbedder:
    """Encodes passages and questions into normalised dense vectors.

    The model is loaded lazily and guarded by a lock, because the web server may
    field concurrent requests while an ingest run is still warming the model up.
    Passages and queries share one encoder; BGE-M3 needs no instruction prefix,
    unlike the E5 family.
    """

    def __init__(
        self,
        settings: Settings,
        device_resolver: ComputeDeviceResolver | None = None,
        device: str | None = None,
    ) -> None:
        """Record configuration without touching the GPU or downloading weights.

        Args:
            device: Overrides ``settings.embedding_device`` for this instance.
                The server embeds questions with a CPU instance and indexes
                with a GPU one; both load the same model, so a question's
                vector is comparable with the indexed passages either way.
        """
        self._settings = settings
        self._requested_device = device or settings.embedding_device
        self._device_resolver = device_resolver or ComputeDeviceResolver()
        self._model = None
        self._resolved_device: str | None = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """True once the model weights are resident in memory."""
        return self._model is not None

    @property
    def device(self) -> str:
        """The device the model is running on, or the one it would use."""
        if self._resolved_device is not None:
            return self._resolved_device
        return self._device_resolver.resolve(self._requested_device)

    @property
    def dimensions(self) -> int:
        """Width of the vectors this embedder produces."""
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return self._settings.embedding_dimensions

    def load(self) -> None:
        """Download if needed and load the model, at most once per process."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            self._resolved_device = self._device_resolver.resolve(self._requested_device)
            LOGGER.info(
                "Memuat model embedding %s pada %s",
                self._settings.embedding_model_name,
                self._resolved_device,
            )
            self._model = SentenceTransformer(
                self._settings.embedding_model_name,
                device=self._resolved_device,
                cache_folder=str(self._settings.model_cache_directory),
            )

    def unload(self) -> None:
        """Release the model and free GPU memory for the language model."""
        with self._load_lock:
            self._model = None
        self.release_gpu_memory()

    def release_gpu_memory(self) -> None:
        """Return cached CUDA blocks to the driver without unloading weights."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            LOGGER.debug("Tidak dapat mengosongkan cache CUDA", exc_info=True)

    def build_token_counter(self):
        """Expose the model's tokenizer as a chunking token counter.

        Falls back to the character-ratio estimator when the model has not been
        loaded, so chunking never forces a multi-gigabyte download.
        """
        from .chunking import CharacterRatioTokenCounter, TransformerTokenCounter

        if self._model is None:
            return CharacterRatioTokenCounter()
        try:
            return TransformerTokenCounter(self._model.tokenizer)
        except AttributeError:
            return CharacterRatioTokenCounter()

    def encode_passages(self, passages: list[str], progress_callback=None) -> np.ndarray:
        """Encode indexed passages into a normalised float32 matrix.

        Args:
            passages: Chunk texts to encode.
            progress_callback: Optional callable receiving ``(done, total)``
                after each batch, used to drive the ingest progress bar.

        Returns:
            An ``(len(passages), dimensions)`` array of unit-length vectors.
        """
        self.load()
        if not passages:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        batch_size = max(1, self._settings.embedding_batch_size)
        encoded_batches: list[np.ndarray] = []
        for start_index in range(0, len(passages), batch_size):
            batch = passages[start_index : start_index + batch_size]
            batch_vectors = self._model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            encoded_batches.append(np.asarray(batch_vectors, dtype=np.float32))
            if progress_callback is not None:
                progress_callback(min(start_index + batch_size, len(passages)), len(passages))
        return np.vstack(encoded_batches)

    def encode_question(self, question: str) -> np.ndarray:
        """Encode one question into a single normalised vector."""
        self.load()
        vector = self._model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)[0]
