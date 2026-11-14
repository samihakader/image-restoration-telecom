"""The restoration network (PyTorch) and ONNX export helpers.

RestoreNet is a small DnCNN-style residual CNN with dilated convolutions, so its
receptive field (about 49 px) is large enough to "see across" a lost 8x8 block.
It is fully convolutional: it accepts images of any size.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RestoreNet(nn.Module):
    def __init__(self, channels=3, features=48, depth=12, dilations=(1, 2, 4)):
        super().__init__()
        layers = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for i in range(depth - 2):
            d = dilations[i % len(dilations)]
            layers += [
                nn.Conv2d(features, features, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(features, channels, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        # residual learning: the network predicts a correction to its input
        return x + self.body(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def export_onnx(model, path, patch=64):
    """Export ``model`` to ONNX with dynamic batch / height / width."""
    model = model.eval().cpu()
    dummy = torch.rand(1, 3, patch, patch)
    kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )
    last_error = None
    # Newer PyTorch versions default to a different exporter; try the classic one first.
    for extra in ({"dynamo": False}, {}):
        try:
            torch.onnx.export(model, dummy, str(path), **kwargs, **extra)
            return
        except Exception as exc:  # noqa: BLE001 - try the next exporter
            last_error = exc
    raise RuntimeError(f"ONNX export failed: {last_error}") from last_error


def verify_onnx(model, path, size=(80, 96)):
    """Max absolute difference between PyTorch and ONNX Runtime on a random image."""
    import onnxruntime as ort

    model = model.eval().cpu()
    x = torch.rand(1, 3, *size)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x.numpy().astype(np.float32)})[0]
    return float(np.abs(ref - out).max())


def load_checkpoint(path, device="cpu"):
    """Load a ``model.pt`` written by train.py."""
    ckpt = torch.load(str(path), map_location=device)
    model = RestoreNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
