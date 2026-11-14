"""Inference with the exported ONNX model (CPU, no PyTorch needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "model.onnx"


class AIRestorer:
    """Callable: float32 RGB (H, W, 3) in [0, 1] -> restored image, same shape."""

    def __init__(self, path):
        import onnxruntime as ort

        self.path = str(path)
        self.session = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, img):
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None].astype(np.float32))
        y = self.session.run(None, {self.input_name: x})[0]
        return np.clip(y[0].transpose(1, 2, 0), 0.0, 1.0).astype(np.float32)


def load_ai(path=None):
    """Return an AIRestorer, or None when no model file is available."""
    path = Path(path) if path else DEFAULT_MODEL
    if not path.exists():
        return None
    try:
        return AIRestorer(path)
    except Exception as exc:  # corrupt file, onnxruntime missing, ...
        print(f"[warn] could not load AI model {path}: {exc}")
        return None
