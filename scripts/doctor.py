"""Check local dependencies and required model files."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "numpy",
    "cv2",
    "scipy",
    "matplotlib",
    "PIL",
    "onnxruntime",
    "mediapipe",
)
MODELS = (
    "models/pose_landmarker_full.task",
    "models/birefnet_lite.onnx",
    "models/dexined_2024sep.onnx",
    "models/rtmpose-m-distill/onnx/rtmpose-m-distill-256x256.onnx",
    "models/schp-lip-20/onnx/schp-lip-20-int8-static.onnx",
    "models/schp-atr-18/onnx/schp-atr-18-int8-static.onnx",
)


def main():
    missing_packages = [
        name for name in PACKAGES if importlib.util.find_spec(name) is None
    ]
    missing_models = [
        relative for relative in MODELS
        if not (ROOT / relative).is_file()
    ]
    print(f"project root: {ROOT}")
    print(f"python: {Path(__import__('sys').executable)}")
    print(
        "dependencies: "
        + ("ok" if not missing_packages else
           "missing " + ", ".join(missing_packages))
    )
    print(
        "models: "
        + ("ok" if not missing_models else
           "missing " + ", ".join(missing_models))
    )
    if missing_packages or missing_models:
        print("Run `python -m pip install -r requirements.txt` and add models.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
