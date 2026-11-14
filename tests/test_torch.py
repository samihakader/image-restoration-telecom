"""Tests that need PyTorch (skipped automatically when it is not installed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from restoration.model import RestoreNet, export_onnx, verify_onnx  # noqa: E402


def test_model_shapes_and_any_size():
    model = RestoreNet(features=8, depth=6).eval()
    for h, w in [(32, 32), (45, 70)]:
        x = torch.rand(2, 3, h, w)
        assert model(x).shape == x.shape


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    model = RestoreNet(features=8, depth=6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    clean = torch.rand(4, 3, 32, 32)
    noisy = (clean + 0.2 * torch.randn_like(clean)).clamp(0, 1)
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.l1_loss(model(noisy), clean)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("onnxruntime")
    model = RestoreNet(features=8, depth=6).eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path, patch=32)
    assert verify_onnx(model, path, size=(40, 56)) < 1e-4


def test_train_script_smoke(tmp_path):
    """End-to-end: tiny training run on a folder of images, then ONNX file exists."""
    from PIL import Image

    import train
    from restoration import pipeline

    folder = tmp_path / "imgs"
    folder.mkdir()
    for i, p in enumerate(pipeline.sample_paths()[:3]):
        Image.open(p).convert("RGB").save(folder / f"{i}.png")
    out = tmp_path / "out"
    train.main(["--dataset", "folder", "--data-dir", str(folder), "--epochs", "1",
                "--patch", "48", "--repeat", "3", "--batch", "4", "--workers", "0",
                "--features", "8", "--depth", "6", "--out", str(out)])
    assert (out / "model.pt").exists()
    assert (out / "train_log.csv").exists()
