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
import pickle
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT, Settings

LOGGER = logging.getLogger(__name__)


class OpenVinoSentenceEncoder:
    """BGE-M3 run by OpenVINO (int8) on the CPU, for encoding questions.

    The converted graph already applies BGE-M3's own pooling -- the first
    token's vector, L2-normalised -- so its output is the sentence vector.
    Offers the parts of sentence-transformers' ``SentenceTransformer`` that
    :class:`TextEmbedder` uses (``encode``, ``tokenizer``,
    ``get_sentence_embedding_dimension``), so the two are interchangeable.

    Args:
        model_directory: Folder holding ``model.xml``/``model.bin``; the
            tokenizer is read from the sibling ``tokenizer`` folder.
        max_length: Token limit per text. Questions are far shorter.
    """

    def __init__(self, model_directory: Path, max_length: int = 512) -> None:
        import openvino as ov
        from transformers import AutoTokenizer

        self._max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory.parent / "tokenizer")
        self._compiled = ov.Core().compile_model(
            str(model_directory / "model.xml"), "CPU", {"PERFORMANCE_HINT": "LATENCY"}
        )
        self._output = self._compiled.output(0)

    def get_sentence_embedding_dimension(self) -> int:
        return int(self._output.get_partial_shape()[-1].get_length())

    def encode(self, texts, batch_size: int = 8, convert_to_numpy: bool = True,
               normalize_embeddings: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        """Return one unit-length vector per text, as sentence-transformers does."""
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self._max_length,
                return_tensors="np",
            )
            output = self._compiled(
                {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
            )[self._output]
            vectors.append(np.asarray(output, dtype=np.float32))
        if not vectors:
            return np.zeros((0, self.get_sentence_embedding_dimension()), dtype=np.float32)
        matrix = np.vstack(vectors)
        if normalize_embeddings:
            # The graph normalises already; this only removes rounding from int8.
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix


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
        backend: str = "torch",
        openvino_directory: Path | None = None,
        cpu_threads: int = 0,
    ) -> None:
        """Record configuration without touching the GPU or downloading weights.

        Args:
            device: Overrides ``settings.embedding_device`` for this instance.
                The server embeds questions with a CPU instance and indexes
                with a GPU one; both load the same model, so a question's
                vector is comparable with the indexed passages either way.
            backend: ``"openvino"`` runs the int8 model in
                ``openvino_directory`` on the CPU (smaller and faster, for
                questions); ``"torch"``, or an OpenVINO model that fails to
                load, uses sentence-transformers.
            cpu_threads: When above 0 and the model runs on the CPU through
                PyTorch, the number of threads PyTorch may use. The server's
                question encoder sets it so it does not fight the reranker.
        """
        self._settings = settings
        self._requested_device = device or settings.embedding_device
        self._device_resolver = device_resolver or ComputeDeviceResolver()
        self._backend = (backend or "torch").lower()
        self._openvino_directory = openvino_directory
        self._cpu_threads = cpu_threads
        self._model = None
        self._resolved_device: str | None = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """True once the model weights are resident in memory."""
        return self._model is not None

    @property
    def backend_name(self) -> str:
        """'openvino' or 'torch' once loaded, for the status panel and logs."""
        if isinstance(self._model, OpenVinoSentenceEncoder):
            return "openvino"
        return "torch"

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
            # Renamed to get_embedding_dimension in newer sentence-transformers.
            dimension = (getattr(self._model, "get_embedding_dimension", None)
                         or self._model.get_sentence_embedding_dimension)
            return int(dimension())
        return self._settings.embedding_dimensions

    def load(self) -> None:
        """Download if needed and load the model, at most once per process."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._resolved_device = self._device_resolver.resolve(self._requested_device)
            if self._backend == "openvino" and self._resolved_device == "cpu" and self._load_openvino():
                return
            from sentence_transformers import SentenceTransformer

            if self._cpu_threads > 0 and self._resolved_device == "cpu":
                import torch

                torch.set_num_threads(self._cpu_threads)

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

    def _load_openvino(self) -> bool:
        """Try the OpenVINO model; on any failure, log it and use PyTorch instead."""
        directory = self._openvino_directory
        try:
            if directory is None:
                raise FileNotFoundError("tidak ada direktori model OpenVINO")
            LOGGER.info("Memuat model embedding OpenVINO dari %s", directory)
            self._model = OpenVinoSentenceEncoder(directory)
            return True
        except Exception as error:
            LOGGER.warning("Embedding OpenVINO tidak dapat dimuat (%s); memakai PyTorch.", error)
            return False

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


class SubprocessEmbedder:
    """Encodes passages for indexing in a child process that ends with the run.

    Used by the web server for the indexing it starts itself (the admin button,
    automatic indexing). The child, ``python -m app.embedding_worker``, loads
    the model on ``EMBED_DEVICE`` -- the GPU when there is one -- so indexing
    keeps its speed, and :meth:`unload` ends the child, taking with it the
    CUDA context of about 1.4 GB that would otherwise stay in the server and
    crowd the answering model off the 8 GB card.

    Offers what :class:`IngestionPipeline` uses from :class:`TextEmbedder`.
    Chunking counts tokens with the character-ratio estimate, as the in-process
    embedder does whenever its model is not yet loaded at the start of a run.
    """

    #: Seconds to wait for the child to exit after being asked to.
    EXIT_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        settings: Settings,
        device_resolver: ComputeDeviceResolver | None = None,
        command: list[str] | None = None,
    ) -> None:
        """Record configuration; the child process starts on first use.

        Args:
            command: The child's command line; tests replace the worker.
        """
        self._settings = settings
        self._device_resolver = device_resolver or ComputeDeviceResolver()
        self._command = command or [sys.executable, "-m", "app.embedding_worker"]
        self._process: subprocess.Popen | None = None
        self._dimensions: int | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        """The device the child encodes on."""
        return self._device_resolver.resolve(self._settings.embedding_device)

    @property
    def is_loaded(self) -> bool:
        """True while the child process is running."""
        return self._process is not None and self._process.poll() is None

    @property
    def dimensions(self) -> int:
        """Width of the vectors this embedder produces."""
        return self._dimensions or self._settings.embedding_dimensions

    def build_token_counter(self):
        """The character-ratio token estimate, as for an unloaded embedder."""
        from .chunking import CharacterRatioTokenCounter

        return CharacterRatioTokenCounter()

    def load(self) -> None:
        """Start the child and wait until its model is ready."""
        with self._lock:
            if self.is_loaded:
                return
            device = self.device
            LOGGER.info(
                "Memulai proses embedding terpisah (%s pada %s)",
                self._settings.embedding_model_name,
                device,
            )
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
            )
            reply = self._exchange({
                "model": self._settings.embedding_model_name,
                "device": device,
                "cache_folder": str(self._settings.model_cache_directory),
                "batch_size": max(1, self._settings.embedding_batch_size),
            })
            if reply[0] != "ready":
                self._stop_process()
                raise RuntimeError(f"Proses embedding gagal dimulai: {reply[1]}")
            self._dimensions = int(reply[1])

    def encode_passages(self, passages: list[str], progress_callback=None) -> np.ndarray:
        """Encode passages in the child, batch by batch, reporting progress."""
        self.load()
        if not passages:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        batch_size = max(1, self._settings.embedding_batch_size)
        encoded_batches: list[np.ndarray] = []
        for start_index in range(0, len(passages), batch_size):
            batch = list(passages[start_index : start_index + batch_size])
            with self._lock:
                reply = self._exchange(batch)
            if reply[0] != "vectors":
                raise RuntimeError(f"Proses embedding gagal: {reply[1]}")
            encoded_batches.append(np.asarray(reply[1], dtype=np.float32))
            if progress_callback is not None:
                progress_callback(min(start_index + batch_size, len(passages)), len(passages))
        return np.vstack(encoded_batches)

    def encode_question(self, question: str) -> np.ndarray:
        """Encode one text; indexing does not need it, but the interface has it."""
        return self.encode_passages([question])[0]

    def unload(self) -> None:
        """End the child process, freeing everything it held on the GPU."""
        with self._lock:
            self._stop_process()

    def _exchange(self, message):
        """Send one message to the child and return its reply."""
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("Proses embedding tidak berjalan")
        try:
            pickle.dump(message, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            process.stdin.flush()
            return pickle.load(process.stdout)
        except (EOFError, BrokenPipeError, OSError) as error:
            code = process.poll()
            raise RuntimeError(f"Proses embedding berhenti (kode {code}): {error}") from error

    def _stop_process(self) -> None:
        """Ask the child to exit, and make sure it has."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                pickle.dump(None, process.stdin)
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=self.EXIT_TIMEOUT_SECONDS)
        except Exception:
            process.kill()
            process.wait()
        LOGGER.info("Proses embedding terpisah selesai; memorinya dilepas")
