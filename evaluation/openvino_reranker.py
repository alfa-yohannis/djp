"""Run the reranker through OpenVINO instead of PyTorch.

The model itself does not change: it is BAAI/bge-reranker-v2-m3, converted to
OpenVINO's format. Two steps, because the int8 quantiser (NNCF) needs an older
NumPy than the app uses and therefore lives in a separate environment:

1. **export** (app environment): convert the Hugging Face model to an
   OpenVINO model in full precision, and write a calibration set of tokenised
   question-passage pairs::

       .venv\\Scripts\\python.exe -m evaluation.openvino_reranker export

2. **quantize** (separate environment with ``openvino`` and ``nncf``): see
   ``evaluation/ov_quantize.py``, which writes the int8 model next to the
   full-precision one.

``OpenVinoReranker`` then scores pairs with either model, tokenising exactly as
sentence-transformers' ``CrossEncoder`` does, so it can stand in for the
PyTorch reranker in ``bench_pipeline.py --reranker-backend openvino``.

Calibration pairs are built from the indexed passages, not from the benchmark
questions, so the questions used for timing are never seen during
quantisation.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SETTINGS  # noqa: E402

MODEL_ID = "BAAI/bge-reranker-v2-m3"
OPENVINO_DIRECTORY = PROJECT_ROOT / "data" / "eval" / "openvino" / "bge-reranker-v2-m3"


class OpenVinoReranker:
    """Cross-encoder scoring with an OpenVINO model on the CPU."""

    def __init__(self, model_directory: Path, batch_size: int, max_length: int,
                 performance_hint: str = "LATENCY") -> None:
        import openvino as ov
        from transformers import AutoTokenizer

        self.model_id = f"{MODEL_ID} (OpenVINO, {model_directory.name})"
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory.parent / "tokenizer")
        core = ov.Core()
        self.compiled = core.compile_model(str(model_directory / "model.xml"), "CPU",
                                           {"PERFORMANCE_HINT": performance_hint})
        self.output = self.compiled.output(0)

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            encoded = self.tokenizer([q for q, _ in batch], [p for _, p in batch],
                                     padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="np")
            logits = self.compiled({"input_ids": encoded["input_ids"],
                                    "attention_mask": encoded["attention_mask"]})[self.output]
            scores.append(np.asarray(logits).reshape(-1))
        return np.concatenate(scores)

    def rerank(self, question: str, passages: list, keep: int) -> list:
        scores = self.predict([(question, passage.text) for passage in passages])
        ranked = sorted(zip(passages, scores), key=lambda pair: float(pair[1]), reverse=True)
        kept = []
        for passage, score in ranked[:keep]:
            passage.reranked_score = float(score)
            kept.append(passage)
        return kept


def calibration_pairs(count: int, seed: int = 7) -> list[tuple[str, str]]:
    """Question-like snippets paired with passages, half from the same document.

    A pseudo-question is the opening words of one passage; its partner is
    either a neighbouring passage of the same document (a likely match) or a
    random passage (a likely non-match), so activations cover both cases.
    """
    connection = sqlite3.connect(f"file:{SETTINGS.database_path.as_posix()}?mode=ro", uri=True)
    rows = connection.execute("SELECT document_id, text FROM chunks").fetchall()
    connection.close()
    by_document: dict[str, list[str]] = {}
    for document_id, text in rows:
        by_document.setdefault(document_id, []).append(text)
    generator = random.Random(seed)
    pairs = []
    while len(pairs) < count:
        document_id, text = generator.choice(rows)
        question = " ".join(text.split()[:generator.randint(5, 14)])
        if generator.random() < 0.5 and len(by_document[document_id]) > 1:
            partner = generator.choice(by_document[document_id])
        else:
            partner = generator.choice(rows)[1]
        pairs.append((question, partner))
    return pairs


def export(calibration_size: int, directory: Path = OPENVINO_DIRECTORY) -> None:
    """Convert the model to OpenVINO (full precision) and write calibration data."""
    import openvino as ov
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache = str(SETTINGS.model_cache_directory)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, cache_dir=cache)
    model.eval()

    example = tokenizer(["contoh pertanyaan"] * 2, ["contoh bagian dokumen yang lebih panjang"] * 2,
                        padding=True, return_tensors="pt")
    with torch.no_grad():
        converted = ov.convert_model(
            model,
            example_input={"input_ids": example["input_ids"],
                           "attention_mask": example["attention_mask"]},
            input=[("input_ids", [-1, -1], ov.Type.i64),
                   ("attention_mask", [-1, -1], ov.Type.i64)],
        )
    fp32_directory = directory / "fp32"
    fp32_directory.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, str(fp32_directory / "model.xml"), compress_to_fp16=False)
    tokenizer.save_pretrained(directory / "tokenizer")

    # Check the converted model against PyTorch on a few real pairs.
    pairs = calibration_pairs(16, seed=99)
    encoded = tokenizer([q for q, _ in pairs], [p for _, p in pairs], padding=True,
                        truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        reference = model(**encoded).logits.reshape(-1).numpy()
    compiled = ov.Core().compile_model(converted, "CPU")
    converted_scores = np.asarray(compiled({"input_ids": encoded["input_ids"].numpy(),
                                            "attention_mask": encoded["attention_mask"].numpy()})[0]).reshape(-1)
    print(f"max |PyTorch - OpenVINO fp32| logit difference: {np.abs(reference - converted_scores).max():.5f}")

    # One pair per sample, unpadded, as NNCF expects.
    samples = calibration_pairs(calibration_size)
    arrays = {}
    for index, (question, passage) in enumerate(samples):
        one = tokenizer([question], [passage], truncation=True, max_length=512, return_tensors="np")
        arrays[f"input_ids_{index}"] = one["input_ids"].astype(np.int64)
        arrays[f"attention_mask_{index}"] = one["attention_mask"].astype(np.int64)
    np.savez_compressed(directory / "calibration.npz", count=np.array(len(samples)), **arrays)
    print(f"saved {fp32_directory / 'model.xml'} and {len(samples)} calibration pairs")


def convert_for_app() -> int:
    """Build the int8 model the app uses (``RERANK_BACKEND=openvino``), end to end.

    Exports to OpenVINO, then quantises in a separate environment created on
    first use under ``data/models/openvino/quantize-env`` (NNCF needs an
    older NumPy than the app), writing ``int8/`` where the app looks for it.
    """
    import subprocess

    target = SETTINGS.reranking_openvino_directory
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    print(f"1/3 Mengekspor {MODEL_ID} ke OpenVINO di {directory} ...")
    export(300, directory)

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
    print(f"Selesai: {target}. Aktifkan dengan RERANK_BACKEND=openvino di .env.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export", help="convert to OpenVINO and write calibration data")
    export_parser.add_argument("--calibration-size", type=int, default=300)
    arguments = parser.parse_args()
    if arguments.command == "export":
        export(arguments.calibration_size)
