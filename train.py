"""Train the restoration network.

Examples
--------
Quick smoke test (about a minute, even on CPU):
    python train.py --dataset cifar10 --epochs 1 --max-images 300 --out models_smoke

Full run on a Colab / Kaggle GPU (about 10-15 minutes):
    python train.py --dataset stl10 --epochs 30 --out models

Outputs (in --out): model.pt (PyTorch weights), model.onnx (used by the app),
train_log.csv, train_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from restoration.channel import CHANNELS, apply_channel
from restoration.model import RestoreNet, count_parameters, export_onnx, verify_onnx


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_images(name, root, patch, max_images=None, seed=0):
    """Return a list of uint8 RGB arrays (H, W, 3), each at least ``patch`` px per side."""
    if name == "stl10":
        from torchvision.datasets import STL10

        data = np.asarray(STL10(root=root, split="train", download=True).data)  # (N,3,96,96)
        images = [np.ascontiguousarray(a.transpose(1, 2, 0)) for a in data]
    elif name == "cifar10":
        from torchvision.datasets import CIFAR10

        data = np.asarray(CIFAR10(root=root, train=True, download=True).data)  # (N,32,32,3)
        images = [np.ascontiguousarray(a) for a in data]
    elif name == "folder":
        import cv2

        paths = sorted(p for p in Path(root).rglob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        if not paths:
            raise SystemExit(f"no images found in {root}")
        images = []
        for p in paths:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if min(h, w) < patch:
                s = patch / min(h, w)
                rgb = cv2.resize(rgb, (max(patch, round(w * s)), max(patch, round(h * s))))
            images.append(rgb)
    else:
        raise SystemExit(f"unknown dataset {name!r}")

    order = np.random.default_rng(seed).permutation(len(images))
    images = [images[i] for i in order]
    if max_images:
        images = images[:max_images]
    return images


class CorruptedDataset(Dataset):
    """Yields (damaged, clean) tensors; the damage is generated on the fly."""

    def __init__(self, images, patch, repeat, train, seed=0):
        self.images = images
        self.patch = patch
        self.repeat = repeat
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.images) * self.repeat

    def __getitem__(self, i):
        img = self.images[i % len(self.images)]
        # training: fresh randomness every call; validation: fixed per index
        rng = np.random.default_rng(None if self.train else self.seed + i)
        h, w = img.shape[:2]
        p = min(self.patch, h, w)
        if self.train:
            y = int(rng.integers(0, h - p + 1))
            x = int(rng.integers(0, w - p + 1))
        else:
            y, x = (h - p) // 2, (w - p) // 2
        crop = img[y:y + p, x:x + p].astype(np.float32) / 255.0
        if self.train:
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            if rng.random() < 0.5:
                crop = crop[::-1, :]
        crop = np.ascontiguousarray(crop)
        kind = CHANNELS[int(rng.integers(len(CHANNELS)))]
        level = float(rng.random())
        damaged = apply_channel(crop, kind, level, rng)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))  # noqa: E731
        return to_t(damaged), to_t(crop)


def psnr_per_image(pred, target):
    mse = ((pred.clamp(0, 1) - target) ** 2).mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def evaluate(model, loader, device):
    model.eval()
    sums = {"in": 0.0, "out": 0.0}
    n = 0
    with torch.no_grad():
        for damaged, clean in loader:
            damaged, clean = damaged.to(device), clean.to(device)
            sums["in"] += psnr_per_image(damaged, clean).sum().item()
            sums["out"] += psnr_per_image(model(damaged), clean).sum().item()
            n += damaged.shape[0]
    return sums["in"] / n, sums["out"] / n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="stl10", choices=["stl10", "cifar10", "folder"])
    ap.add_argument("--data-dir", default="data", help="download location, or image folder for --dataset folder")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patch", type=int, default=64, help="training crop size (px)")
    ap.add_argument("--repeat", type=int, default=4, help="random crops per image per epoch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--features", type=int, default=48)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=None, help="use only N images (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    images = load_images(args.dataset, args.data_dir, args.patch, args.max_images, args.seed)
    n_val = max(10, len(images) // 20)
    val_imgs, train_imgs = images[:n_val], images[n_val:]
    print(f"{len(train_imgs)} training images, {len(val_imgs)} validation images")

    train_ds = CorruptedDataset(train_imgs, args.patch, args.repeat, train=True)
    val_ds = CorruptedDataset(val_imgs, args.patch, 4, train=False, seed=12345)
    loader_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, **loader_kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **loader_kw)

    config = {"features": args.features, "depth": args.depth}
    model = RestoreNet(**config).to(device)
    print(f"parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, args.epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    loss_fn = nn.L1Loss()

    log_rows, best = [], -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for damaged, clean in train_dl:
            damaged, clean = damaged.to(device), clean.to(device)
            loss = loss_fn(model(damaged), clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            steps += 1
        psnr_in, psnr_out = evaluate(model, val_dl, device)
        row = {"epoch": epoch, "train_l1": running / max(1, steps),
               "val_psnr_damaged": psnr_in, "val_psnr_restored": psnr_out,
               "seconds": round(time.time() - start, 1)}
        log_rows.append(row)
        flag = ""
        if psnr_out > best:
            best = psnr_out
            torch.save({"state_dict": model.state_dict(), "config": config}, out / "model.pt")
            flag = "  <- best, saved"
        print(f"epoch {epoch:3d}/{args.epochs}  L1 {row['train_l1']:.4f}  "
              f"val PSNR {psnr_in:.2f} -> {psnr_out:.2f} dB  ({row['seconds']}s){flag}")

    with open(out / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    # export the best checkpoint to ONNX for the app
    ckpt = torch.load(str(out / "model.pt"), map_location="cpu")
    best_model = RestoreNet(**ckpt["config"])
    best_model.load_state_dict(ckpt["state_dict"])
    summary = {"best_val_psnr_restored": best, "val_psnr_damaged": log_rows[-1]["val_psnr_damaged"],
               "parameters": count_parameters(best_model), "epochs": args.epochs,
               "dataset": args.dataset, "onnx": False}
    try:
        export_onnx(best_model, out / "model.onnx", patch=args.patch)
        diff = verify_onnx(best_model, out / "model.onnx")
        summary["onnx"] = True
        summary["onnx_max_abs_diff"] = diff
        print(f"ONNX export OK -> {out / 'model.onnx'} (max diff vs PyTorch: {diff:.2e})")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] ONNX export failed: {exc}\n"
              f"       model.pt was saved; see README for the manual export command.")
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
