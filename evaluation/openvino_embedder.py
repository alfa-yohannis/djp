"""Build the int8 OpenVINO copy of BGE-M3 used to encode questions.

Same two steps as ``openvino_reranker.py``, because NNCF needs an older NumPy
than the app and therefore runs in a separate environment:

1. **export** (app environment): wrap the Hugging Face model with BGE-M3's own
   pooling -- the first token's vector, L2-normalised, as sentence-transformers
   computes it -- convert that to OpenVINO in full precision, check it against
   sentence-transformers, and write calibration texts from this project's index;
2. **quantize** (``evaluation/ov_quantize.py`` in the separate environment):
   write ``int8/`` beside the full-precision model.

Only questions go through the int8 model. Passages in the index keep their
full-precision vectors, so the two are compared exactly as before, up to the
small rounding int8 introduces; ``verify`` measures how much that changes the
40 passages search by meaning returns.

    .venv\\Scripts\\python.exe main.py convert-embedder
    .venv\\Scripts\\python.exe -m evaluation.openvino_embedder verify
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SETTINGS  # noqa: E402


def calibration_texts(count: int, seed: int = 13) -> list[str]:
    """Half question-like snippets, half whole passages, from this index."""
    connection = sqlite3.connect(f"file:{SETTINGS.database_path.as_posix()}?mode=ro", uri=True)
    rows = [row[0] for row in connection.execute("SELECT text FROM chunks WHERE length(text) > 80")]
    connection.close()
    generator = random.Random(seed)
    texts = []
    while len(texts) < count:
        text = generator.choice(rows)
        if len(texts) % 2 == 0:
            words = text.split()
            start = generator.randint(0, max(0, len(words) - 14))
            text = " ".join(words[start:start + generator.randint(5, 14)])
        texts.append(text)
    return texts


def export(directory: Path, calibration_size: int = 300) -> None:
    """Convert BGE-M3 with its pooling to OpenVINO and write calibration data."""
    import openvino as ov
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    name, cache = SETTINGS.embedding_model_name, str(SETTINGS.model_cache_directory)
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache)
    transformer = AutoModel.from_pretrained(name, cache_dir=cache)
    transformer.eval()

    class PooledEncoder(torch.nn.Module):
        """The transformer plus BGE-M3's pooling: first token, unit length."""

        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return torch.nn.functional.normalize(hidden[:, 0], p=2, dim=1)

    encoder = PooledEncoder(transformer).eval()
    example = tokenizer(["contoh pertanyaan", "bagian dokumen yang lebih panjang"],
                        padding=True, return_tensors="pt")
    with torch.no_grad():
        converted = ov.convert_model(
            encoder,
            example_input={"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]},
            input=[("input_ids", [-1, -1], ov.Type.i64), ("attention_mask", [-1, -1], ov.Type.i64)],
        )
    fp32_directory = directory / "fp32"
    fp32_directory.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, str(fp32_directory / "model.xml"), compress_to_fp16=False)
    tokenizer.save_pretrained(directory / "tokenizer")

    # The converted model must give sentence-transformers' own vectors.
    texts = calibration_texts(16, seed=99)
    reference = SentenceTransformer(name, device="cpu", cache_folder=cache).encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    del transformer
    compiled = ov.Core().compile_model(converted, "CPU")
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    vectors = np.asarray(compiled({"input_ids": encoded["input_ids"],
                                   "attention_mask": encoded["attention_mask"]})[0])
    cosine = (vectors * reference).sum(axis=1)
    print(f"OpenVINO fp32 vs sentence-transformers: lowest cosine {cosine.min():.6f}")

    arrays = {}
    for index, text in enumerate(calibration_texts(calibration_size)):
        one = tokenizer([text], truncation=True, max_length=512, return_tensors="np")
        arrays[f"input_ids_{index}"] = one["input_ids"].astype(np.int64)
        arrays[f"attention_mask_{index}"] = one["attention_mask"].astype(np.int64)
    np.savez_compressed(directory / "calibration.npz", count=np.array(calibration_size), **arrays)
    print(f"saved {fp32_directory / 'model.xml'} and {calibration_size} calibration texts")


def convert_for_app() -> int:
    """Build the int8 question encoder the app uses, end to end."""
    target = SETTINGS.query_embedding_openvino_directory
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    print(f"1/3 Mengekspor {SETTINGS.embedding_model_name} ke OpenVINO di {directory} ...")
    export(directory)
    environment = directory.parent / "quantize-env"
    python = environment / "Scripts" / "python.exe"
    if not python.exists():
        print(f"2/3 Menyiapkan lingkungan kuantisasi terpisah di {environment} ...")
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q", "openvino", "nncf"], check=True)
    else:
        print(f"2/3 Memakai lingkungan kuantisasi yang sudah ada di {environment}")
    print("3/3 Mengkuantisasi ke int8 (beberapa menit) ...")
    subprocess.run([str(python), str(Path(__file__).with_name("ov_quantize.py")), str(directory),
                    target.name], check=True)
    print(f"Selesai: {target}. Dipakai otomatis (QUERY_EMBED_BACKEND=openvino).")
    return 0


def verify(question_count: int = 20, questions_file: Path | None = None) -> None:
    """Compare int8 questions with full-precision ones on this project's index.

    Questions: those in ``questions_file`` (``{"queries": [{"text": ...}]}``,
    as evaluation/queries.json) when given, otherwise the distinct ones in this
    project's conversation log, topped up with question-like snippets. Reports
    the cosine between the two vectors of each question, how many of the 40
    passages search by meaning returns stay the same, and the time per
    question. Prints numbers only.
    """
    import json

    from app.embeddings import OpenVinoSentenceEncoder, TextEmbedder
    from app.storage import VECTOR_INDEX

    questions = []
    if questions_file is not None:
        questions = [q["text"] for q in json.loads(questions_file.read_text(encoding="utf-8"))["queries"]]
    else:
        try:
            connection = sqlite3.connect(
                f"file:{(SETTINGS.data_directory / 'conversations.db').as_posix()}?mode=ro", uri=True)
            questions = [row[0] for row in connection.execute(
                "SELECT DISTINCT content FROM turns WHERE role='user' AND length(content) > 10")]
            connection.close()
        except Exception:
            pass
    logged = min(len(questions), question_count)
    questions = questions[:question_count] + calibration_texts(question_count - logged, seed=21)[: question_count - logged]

    reference = TextEmbedder(SETTINGS, device="cpu")
    quantised = OpenVinoSentenceEncoder(SETTINGS.query_embedding_openvino_directory)
    reference.encode_question("pemanasan")
    quantised.encode(["pemanasan"])
    cosines, overlaps, reference_ms, quantised_ms = [], [], [], []
    for question in questions:
        started = time.perf_counter()
        expected = reference.encode_question(question)
        reference_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        actual = quantised.encode([question])[0]
        quantised_ms.append((time.perf_counter() - started) * 1000)
        cosines.append(float(np.dot(expected, actual)))
        top = lambda vector: {chunk_id for chunk_id, _ in VECTOR_INDEX.search(vector, 40)}  # noqa: E731
        overlaps.append(len(top(expected) & top(actual)) / 40)
    source = questions_file.name if questions_file is not None else "the conversation log"
    print(f"{len(questions)} questions ({logged} from {source}), {VECTOR_INDEX.size:,} indexed passages")
    print(f"  cosine int8 vs full precision: mean {np.mean(cosines):.4f}, lowest {min(cosines):.4f}")
    print(f"  same 40 passages by meaning: mean {np.mean(overlaps) * 100:.1f} %, lowest {min(overlaps) * 100:.0f} %")
    print(f"  time per question: PyTorch {np.median(reference_ms):.0f} ms, OpenVINO int8 {np.median(quantised_ms):.0f} ms")

    # What the answering model reads: the six passages after fusion and reranking.
    from app.retrieval import CROSS_ENCODER_RERANKER, HybridRetriever
    from app.storage import DOCUMENT_CATALOGUE

    class Int8Questions:
        def encode_question(self, question):
            return quantised.encode([question])[0]

    def final_six(embedder, question):
        retriever = HybridRetriever(SETTINGS, DOCUMENT_CATALOGUE, VECTOR_INDEX, embedder, CROSS_ENCODER_RERANKER)
        return {passage.chunk_id for passage in retriever.retrieve(question, None, 6)}

    finals = [len(final_six(reference, q) & final_six(Int8Questions(), q)) / 6 for q in questions]
    print(f"  same six passages for the answering model ({CROSS_ENCODER_RERANKER.backend_name} reranker): "
          f"mean {np.mean(finals) * 100:.1f} %, lowest {min(finals) * 100:.0f} %, "
          f"all six identical in {sum(f == 1 for f in finals)} of {len(finals)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export", help="convert to OpenVINO and write calibration data")
    verify_parser = sub.add_parser("verify", help="compare int8 with full-precision questions")
    verify_parser.add_argument("--questions", type=Path, default=None,
                               help='JSON file {"queries": [{"text": ...}]}; default: the conversation log')
    arguments = parser.parse_args()
    if arguments.command == "export":
        export(SETTINGS.query_embedding_openvino_directory.parent)
    else:
        verify(questions_file=arguments.questions)
