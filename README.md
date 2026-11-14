# 📡 Noisy Channel Image Rescue

Send an image through a **simulated bad telecom channel**, then repair it two ways:

1. **Classic OpenCV filters**: the standard tool for each type of damage.
2. **A small neural network** (residual CNN, ~0.2 M parameters) trained on mixed damage.

Quality is measured with **PSNR** and **SSIM** against the original image. A Streamlit app lets you play with the
channel live; `evaluate.py` produces the tables and graphs for your report.

```
image ──> [ channel: noise / lost packets / JPEG ] ──> damaged ──┬─> classic OpenCV ──┐
                                                                  └─> CNN (ONNX)  ─────┴─> PSNR / SSIM
```

## The channel (the telecom part)

| Damage | What it models | Severity range |
|---|---|---|
| Gaussian noise (AWGN) | thermal noise on the link | SNR 30 dB → 5 dB |
| Salt & pepper | impulse noise / bit errors | 2% → 30% of pixels |
| Packet loss | lost 8×8 blocks (packets that never arrive) | 2% → 30% of blocks |
| JPEG artifacts | lossy source coding | quality 75 → 10 |
| Combined | JPEG → AWGN → packet loss, in that order | one severity slider |

## Fair comparison (important for your viva)

- The **classic baseline is told what kind of damage happened** and, for packet loss, even receives the exact map of
  lost blocks. It also uses a tuned filter per damage type.
- The **neural network is blind**: one model, no hint about damage type or severity.

So any win for the CNN is an honest win, and any loss is an honest limitation. Report whatever you measure; do
**not** quote numbers you did not produce. Where the CNN loses (for example, lost blocks, where classic inpainting is
told exactly where to look), say so and explain why. That is a finding, not a failure.

## Data

- **Training:** the first 20,000 training images of CIFAR-10 (Krizhevsky, 2009), damaged on the fly by the channel
  simulator. No other training data.
- **Evaluation:** 28 photos collected from the internet, never used for training and not redistributed here.
  Only aggregate statistics are published (`results/summary.md`, `results/curves.png`).
- **`samples/`:** five example images bundled with scikit-image, used for the app demo.

## Project layout

```
app.py                  Streamlit app (live demo)
evaluate.py             tables + graphs: results/summary.md, curves.png, examples.png, results.csv
train.py                trains the CNN, writes models/model.pt and models/model.onnx
export_onnx.py          re-export model.pt to ONNX if needed
restoration/            channel.py  classic.py  model.py  metrics.py  infer.py  pipeline.py
notebooks/train_colab.ipynb   one-click training on free Colab
tests/                  automated tests (also run by GitHub Actions)
samples/                five demo images
models/                 put model.onnx (and model.pt) here
results/                evaluation output
docs/DEMO_SCRIPT.md     5-minute speaking script + likely questions
```

## Quick start (your laptop)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Without `models/model.onnx` the app still works and shows damaged vs classic. The AI panel appears as soon as the
model file is present.

## Train the model (free)

**Easiest: Colab.** Open `notebooks/train_colab.ipynb` in [Google Colab](https://colab.research.google.com),
choose a **T4 GPU** runtime and run the cells in order. At the end it downloads `trained_outputs.zip`
containing `models/` and `results/`. Copy them into this repo.

Kaggle works the same way (turn on GPU and Internet in the notebook settings).

On any machine with PyTorch, this is the exact recipe of the final model:

```bash
pip install -r requirements-train.txt
python train.py --dataset cifar10 --max-images 20000 --repeat 2 --epochs 40 --lr 5e-4 --out models
python evaluate.py                                           # uses models/model.onnx
```

About 18 minutes on a Colab T4 GPU. Notes from the experiments:

- STL-10 (`--dataset stl10`) is a 2.6 GB download and was too slow on free Colab, so CIFAR-10 was used.
- A first CIFAR-10 run with the default learning rate (2e-3) stalled: it learned to fill lost blocks but not to
  remove noise (validation 24.7 dB). With `--lr 5e-4` the model learned both (validation 30.5 dB).
- Enlarging the training images 2x (to look more like normal photos) was tried and did **not** help; it is
  documented as an optional experiment at the end of the notebook.

## Final results (28 unseen photos, 3 noise seeds)

The model was tested on 28 photos collected from the internet that were never used for training (they are not included in this repository). Mean PSNR gain of the AI over the
classic OpenCV baseline, averaged over the severity range, with a 95% confidence interval:

| Damage | Mean gain (dB) | 95% interval | Photos where AI wins |
|---|---|---|---|
| Gaussian noise | +1.88 | +1.39 to +2.38 | 25/28 |
| Salt & pepper | +2.55 | +0.92 to +4.19 | 22/28 |
| Packet loss | -1.95 | -3.23 to -0.68 | 8/28 |
| JPEG | -0.23 | -0.54 to +0.08 (tie) | 10/28 |
| Combined | +0.89 | +0.56 to +1.22 | 24/28 |

The AI wins on noise and combined damage, loses on packet loss (the classic baseline is given the exact positions
of the lost blocks, the AI is blind) and ties on JPEG. Against doing nothing, the AI improves every damage type
except JPEG. Caveats: 28 photos is a small test set of unknown origin and mixed quality, and the choice between two
candidate models (v2 vs v3) was made on the same photos.

## Deploy for free (MLOps flow: train → commit model → auto-deploy)

1. Create a GitHub repo and push this folder, including `models/model.onnx` (a few MB).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **Create app**, pick the repo,
   branch `main` and main file `app.py`.
3. Wait for the build. You get a public link.
4. To update: retrain, replace `models/model.onnx`, `git push`. The app redeploys by itself.

The app uses `onnxruntime` on CPU, so it stays light enough for the free tier. The free tier puts the app to sleep
when idle: **open the link 30 minutes before presenting**, and keep `streamlit run app.py` working on your laptop as a
backup.

Every push also runs the tests on GitHub Actions (`.github/workflows/tests.yml`), which is your CI step.

## Tests

```bash
pip install pytest
pytest -q tests
```

Tests that need PyTorch or Streamlit skip themselves when those are not installed.

## Known limitations (say these yourself before anyone asks)

- The CNN only knows the damage types it was trained on.
- Trained on small natural photos (STL-10); other image styles may do worse.
- Packet loss is simulated as lost 8×8 blocks and the baseline gets the block map for free.
- PSNR/SSIM measure fidelity, not how pretty an image looks.
- Test images in `samples/` are few. For stronger claims, evaluate on more images (`--images your_folder/`).
