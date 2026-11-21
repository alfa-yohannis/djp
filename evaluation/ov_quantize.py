"""Quantise the OpenVINO reranker to int8 with NNCF.

Run in a *separate* environment that has only ``openvino`` and ``nncf``
(NNCF needs NumPy < 2.5, older than the app's). It imports nothing from the
app, PyTorch or transformers: it reads the full-precision model and the
calibration pairs written by ``openvino_reranker.py export`` and writes
``int8/model.xml`` beside them::

    python -m venv data\\eval\\ov-quantize-env
    data\\eval\\ov-quantize-env\\Scripts\\python.exe -m pip install openvino nncf
    data\\eval\\ov-quantize-env\\Scripts\\python.exe evaluation\\ov_quantize.py

Post-training static quantisation: weights *and* activations become int8,
with activation ranges taken from the calibration pairs. ``ModelType.
TRANSFORMER`` keeps the numerically sensitive operations (softmax, layer norm)
in floating point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nncf
import numpy as np
import openvino as ov

DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "eval" / "openvino" / "bge-reranker-v2-m3"


def main() -> None:
    # Optional arguments: the model folder (default: the evaluation copy) and
    # the name of the output folder inside it (default: "int8").
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DIRECTORY
    output_name = sys.argv[2] if len(sys.argv) > 2 else "int8"
    model = ov.Core().read_model(str(directory / "fp32" / "model.xml"))
    with np.load(directory / "calibration.npz") as archive:
        count = int(archive["count"])
        samples = [{"input_ids": archive[f"input_ids_{i}"],
                    "attention_mask": archive[f"attention_mask_{i}"]} for i in range(count)]
    quantized = nncf.quantize(
        model,
        nncf.Dataset(samples),
        model_type=nncf.ModelType.TRANSFORMER,
        subset_size=len(samples),
    )
    output = directory / output_name
    output.mkdir(exist_ok=True)
    ov.save_model(quantized, str(output / "model.xml"))
    print(f"saved {output / 'model.xml'} from {count} calibration pairs")


if __name__ == "__main__":
    main()
