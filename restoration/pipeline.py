"""High level helpers shared by the app, the evaluation script and the tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import channel
from .classic import classic_restore
from .metrics import psnr, ssim

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "samples"


def load_image(src, max_side=320):
    """Load a path / file-like object as float32 RGB in [0, 1], longest side <= max_side."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def to_u8(img):
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def sample_paths():
    return sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def run_demo(clean, kind, level, seed=0, ai=None):
    """Damage ``clean`` and restore it with every available method.

    Returns a dict with the images and the PSNR / SSIM of each stage.
    """
    rng = np.random.default_rng(seed)
    damaged = channel.apply_channel(clean, kind, level, rng)
    outputs = {"damaged": damaged, "classic": classic_restore(damaged, kind)}
    if ai is not None:
        outputs["ai"] = ai(damaged)
    scores = {name: {"psnr": psnr(clean, im), "ssim": ssim(clean, im)}
              for name, im in outputs.items()}
    return {"clean": clean, "images": outputs, "scores": scores,
            "label": channel.describe(kind, level)[0]}


def sweep(clean, kind, levels, seed=0, ai=None):
    """Quality of every method across severity ``levels``. Returns a list of row dicts."""
    rows = []
    for level in levels:
        res = run_demo(clean, kind, level, seed=seed, ai=ai)
        _, x, xlabel = channel.describe(kind, level)
        for method, s in res["scores"].items():
            rows.append({"kind": kind, "level": float(level), "x": float(x),
                         "xlabel": xlabel, "method": method,
                         "psnr": s["psnr"], "ssim": s["ssim"]})
    return rows
