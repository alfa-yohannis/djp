"""Encode passages for indexing in a separate, short-lived process.

Started by :class:`app.embeddings.SubprocessEmbedder` as
``python -m app.embedding_worker`` for one indexing run, and ended with it.

Why a separate process: once a model has run on the GPU, its process keeps a
CUDA context of about 1.4 GB until the process exits -- unloading the model does
not free it. Run inside the web server, that leftover crowded the answering
model off the 8 GB card and made every later answer several times slower. Here
the leftover dies with the worker, so the server itself never touches the GPU.

Messages are pickled, one per line of the conversation, over stdin/stdout:

* parent -> worker: ``{"model", "device", "cache_folder", "batch_size"}``, once;
* worker -> parent: ``("ready", dimensions)`` or ``("error", message)``;
* parent -> worker: a list of passages; worker -> parent: ``("vectors", array)``
  or ``("error", message)``; repeated;
* parent -> worker: ``None``, and the worker exits (as it does on end of input).

The pipe is private to the parent, which started this process itself.
"""

from __future__ import annotations

import os
import pickle
import sys


def main() -> int:
    """Serve encoding requests until told to stop; return the exit code."""
    requests = sys.stdin.buffer
    # Keep a private handle on the real stdout for replies, then point file
    # descriptor 1 at stderr: a library printing to stdout must not corrupt
    # the pickled replies.
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def send(message) -> None:
        pickle.dump(message, replies, protocol=pickle.HIGHEST_PROTOCOL)
        replies.flush()

    try:
        configuration = pickle.load(requests)
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            configuration["model"],
            device=configuration["device"],
            cache_folder=configuration["cache_folder"],
        )
        dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        send(("ready", int(dimension())))
    except Exception as error:  # reported to the parent, which logs it
        send(("error", f"{type(error).__name__}: {error}"))
        return 1

    while True:
        try:
            passages = pickle.load(requests)
        except EOFError:
            return 0
        if passages is None:
            return 0
        try:
            vectors = model.encode(
                passages,
                batch_size=configuration["batch_size"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            send(("vectors", np.asarray(vectors, dtype=np.float32)))
        except Exception as error:
            send(("error", f"{type(error).__name__}: {error}"))


if __name__ == "__main__":
    sys.exit(main())
