"""Evaluate damaged vs classic vs AI restoration and produce the report material.

    python evaluate.py                       # uses samples/ and models/model.onnx (if present)
    python evaluate.py --images my_photos/   # your own test images
    python evaluate.py --levels 0,0.25,0.5,0.75,1 --seeds 3

Writes into --out (default: results/):
    results.csv        every measurement
    summary.md         tables ready to paste into the report
    curves.png         quality vs channel severity, one panel per damage type
    examples.png       damaged / classic / AI images side by side
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from restoration import channel, pipeline  # noqa: E402
from restoration.infer import load_ai  # noqa: E402

METHOD_NAMES = {"damaged": "Damaged", "classic": "Classic (OpenCV)", "ai": "AI (CNN)"}
COLORS = {"damaged": "#9aa0a6", "classic": "#e8710a", "ai": "#1a73e8"}


def collect_images(folder, max_side):
    if folder:
        paths = sorted(p for p in Path(folder).iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    else:
        paths = pipeline.sample_paths()
    if not paths:
        raise SystemExit("no test images found")
    return [(p.stem, pipeline.load_image(p, max_side)) for p in paths]


def run(images, ai, levels, seeds):
    rows = []
    for name, img in images:
        for kind in channel.CHANNELS:
            for seed in range(seeds):
                for r in pipeline.sweep(img, kind, levels, seed=seed, ai=ai):
                    r.update(image=name, seed=seed)
                    rows.append(r)
    return rows


def aggregate(rows):
    """Mean over images and seeds -> {kind: {method: [(x, psnr, ssim), ...]}}"""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["kind"], r["method"], r["level"])].append((r["x"], r["psnr"], r["ssim"]))
    out = defaultdict(lambda: defaultdict(list))
    for (kind, method, level), vals in sorted(acc.items()):
        arr = np.array(vals)
        out[kind][method].append((level, arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()))
    return out


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_curves(agg, path):
    kinds = list(channel.CHANNELS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, kind in zip(axes.ravel(), kinds):
        xlabel = channel.describe(kind, 0.5)[2]
        for method, pts in agg[kind].items():
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax.plot(xs, ys, marker="o", lw=2, label=METHOD_NAMES[method], color=COLORS[method])
        ax.set_title(channel.LABELS[kind], fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Image quality after a noisy channel: higher is better", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_summary(agg, path, n_images, seeds, has_ai):
    methods = ["damaged", "classic"] + (["ai"] if has_ai else [])
    lines = [f"# Results\n",
             f"Mean over {n_images} test image(s) x {seeds} random seed(s). "
             "PSNR in dB (higher is better), SSIM in [0, 1].\n"]
    for kind in channel.CHANNELS:
        lines.append(f"\n## {channel.LABELS[kind]}\n")
        xlabel = channel.describe(kind, 0.5)[2]
        header = f"| {xlabel} | " + " | ".join(f"{METHOD_NAMES[m]} PSNR | {METHOD_NAMES[m]} SSIM"
                                              for m in methods) + " |"
        lines += [header, "|" + "---|" * (1 + 2 * len(methods))]
        n_pts = len(agg[kind]["damaged"])
        for i in range(n_pts):
            x = agg[kind]["damaged"][i][1]
            cells = []
            for m in methods:
                _, _, p, s = agg[kind][m][i]
                cells += [f"{p:.2f}", f"{s:.3f}"]
            lines.append(f"| {x:.1f} | " + " | ".join(cells) + " |")
    if has_ai:
        gains = []
        for kind in channel.CHANNELS:
            g = np.mean([a[2] - c[2] for a, c in zip(agg[kind]["ai"], agg[kind]["classic"])])
            gains.append((kind, g))
        lines.append("\n## Average PSNR gain of AI over classic\n")
        lines += ["| Damage | Gain (dB) |", "|---|---|"]
        lines += [f"| {channel.LABELS[k]} | {g:+.2f} |" for k, g in gains]
    Path(path).write_text("\n".join(lines) + "\n")


def _ci_halfwidth(values):
    """95% confidence half-width of the mean (t distribution); None if < 2 values."""
    n = len(values)
    if n < 2:
        return None
    se = float(np.std(values, ddof=1)) / np.sqrt(n)
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
    except Exception:  # noqa: BLE001 - scipy missing: normal approximation
        t = 1.96
    return t * se


def paired_section(rows):
    """Is the AI's advantage real, or an accident of a few test images?

    For every damage type the AI-minus-baseline PSNR difference is computed on the SAME
    damaged image, averaged per test image, and then summarised across images
    (mean, 95% confidence interval, and how many images the AI wins on).
    A gain is only convincing when its interval does not include 0.
    """
    by_case = defaultdict(dict)
    for r in rows:
        by_case[(r["kind"], r["image"], r["seed"], r["level"])][r["method"]] = r["psnr"]

    lines = ["\n## Is the AI's advantage real? (paired statistics)\n",
             "Differences are computed on the same damaged image, averaged per test image, then "
             "summarised across images. A gain is only convincing when its 95% interval stays "
             "above 0 and the AI wins on most images.\n"]
    for baseline, title in (("classic", "AI minus classic OpenCV"),
                            ("damaged", "AI minus doing nothing (damaged image)")):
        lines += [f"\n### {title}\n",
                  "| Damage | Mean gain (dB) | 95% interval | Images AI wins | Cases AI wins |",
                  "|---|---|---|---|---|"]
        for kind in channel.CHANNELS:
            per_image, case_wins, case_n = defaultdict(list), 0, 0
            for (k, image, _seed, _level), v in by_case.items():
                if k != kind or "ai" not in v or baseline not in v:
                    continue
                diff = v["ai"] - v[baseline]
                per_image[image].append(diff)
                case_wins += diff > 0
                case_n += 1
            if not per_image:
                continue
            means = np.array([np.mean(d) for d in per_image.values()])
            half = _ci_halfwidth(means)
            interval = "n/a (1 image)" if half is None else \
                f"{means.mean() - half:+.2f} to {means.mean() + half:+.2f}"
            lines.append(f"| {channel.LABELS[kind]} | {means.mean():+.2f} | {interval} | "
                         f"{int((means > 0).sum())}/{len(means)} | "
                         f"{100.0 * case_wins / case_n:.0f}% |")
    return lines


def plot_examples(img, ai, path, level=0.6, seed=3):
    cols = ["clean", "damaged", "classic"] + (["ai"] if ai is not None else [])
    kinds = list(channel.CHANNELS)
    fig, axes = plt.subplots(len(kinds), len(cols), figsize=(3.2 * len(cols), 3.0 * len(kinds)))
    for r, kind in enumerate(kinds):
        res = pipeline.run_demo(img, kind, level, seed=seed, ai=ai)
        for c, col in enumerate(cols):
            ax = axes[r, c]
            im = res["clean"] if col == "clean" else res["images"][col]
            ax.imshow(pipeline.to_u8(im))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == "clean":
                ax.set_ylabel(kind, fontsize=11)
                ax.set_title("Original" if r == 0 else "")
            else:
                sc = res["scores"][col]
                title = f"{METHOD_NAMES[col]}\n{sc['psnr']:.1f} dB / SSIM {sc['ssim']:.2f}"
                ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", default=None, help="folder of test images (default: samples/)")
    ap.add_argument("--model", default=None, help="path to model.onnx (default: models/model.onnx)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--levels", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-side", type=int, default=256)
    args = ap.parse_args()

    levels = [float(v) for v in args.levels.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.images, args.max_side)
    ai = load_ai(args.model)
    print(f"{len(images)} image(s), AI model: {'yes' if ai else 'NOT FOUND (classic baseline only)'}")

    rows = run(images, ai, levels, args.seeds)
    agg = aggregate(rows)
    write_csv(rows, out / "results.csv")
    plot_curves(agg, out / "curves.png")
    write_summary(agg, out / "summary.md", len(images), args.seeds, ai is not None)
    if ai is not None:
        section = paired_section(rows)
        with open(out / "summary.md", "a") as f:
            f.write("\n".join(section) + "\n")
        print("\n".join(section))
    plot_examples(images[0][1], ai, out / "examples.png")
    print(f"wrote results.csv, summary.md, curves.png, examples.png to {out}/")


if __name__ == "__main__":
    main()
