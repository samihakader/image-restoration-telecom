import numpy as np
import pytest

from restoration import channel, classic, pipeline
from restoration.infer import load_ai
from restoration.metrics import psnr, ssim


@pytest.fixture(scope="module")
def img():
    return pipeline.load_image(pipeline.sample_paths()[0], max_side=128)


def test_images_are_valid(img):
    assert img.dtype == np.float32 and img.ndim == 3 and img.shape[2] == 3
    assert 0.0 <= img.min() and img.max() <= 1.0


@pytest.mark.parametrize("kind", channel.CHANNELS)
@pytest.mark.parametrize("level", [0.0, 0.5, 1.0])
def test_channel_output_shape_range_and_damage(img, kind, level):
    out = channel.apply_channel(img, kind, level, np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert psnr(img, out) < 60  # something was damaged


def test_awgn_hits_requested_snr(img):
    for snr in (5, 15, 25):
        noisy = channel.awgn(img, snr, np.random.default_rng(1))
        noise_power = np.mean((noisy - img) ** 2)
        measured = 10 * np.log10(np.mean(img ** 2) / noise_power)
        # Clipping pixels to [0, 1] removes a little noise power, so the measured SNR
        # lands slightly ABOVE the requested one (most visibly at very low SNR).
        assert snr - 0.5 < measured < snr + 2.5


def test_packet_loss_fraction_and_detection(img):
    out = channel.packet_loss(img, 0.3, np.random.default_rng(2))
    mask = channel.detect_lost_blocks(out)
    frac = mask.mean()
    assert 0.15 < frac < 0.45
    assert (out[mask] == 0).all()


def test_severity_is_monotonic(img):
    scores = [psnr(img, channel.apply_channel(img, "awgn", lv, np.random.default_rng(0)))
              for lv in (0.0, 0.5, 1.0)]
    assert scores[0] > scores[1] > scores[2]


@pytest.mark.parametrize("kind", ["awgn", "salt_pepper", "packet_loss", "combined"])
def test_classic_improves_quality(img, kind):
    damaged = channel.apply_channel(img, kind, 0.7, np.random.default_rng(3))
    fixed = classic.classic_restore(damaged, kind)
    assert fixed.shape == img.shape
    assert psnr(img, fixed) > psnr(img, damaged) + 1.0


def test_metrics_identity(img):
    assert psnr(img, img) == 99.0
    assert ssim(img, img) == pytest.approx(1.0)


def test_run_demo_and_sweep_without_ai(img):
    res = pipeline.run_demo(img, "awgn", 0.5, seed=0, ai=None)
    assert set(res["images"]) == {"damaged", "classic"}
    rows = pipeline.sweep(img, "packet_loss", [0.0, 1.0], seed=0, ai=None)
    assert len(rows) == 4 and {r["method"] for r in rows} == {"damaged", "classic"}


def test_run_demo_with_fake_ai(img):
    res = pipeline.run_demo(img, "jpeg", 0.5, seed=0, ai=lambda x: x)
    assert "ai" in res["images"] and "ai" in res["scores"]


def test_missing_model_returns_none(tmp_path):
    assert load_ai(tmp_path / "nope.onnx") is None


def test_paired_statistics_report_clear_win_and_loss():
    import evaluate

    rows = []
    for image in ("a", "b", "c"):
        for kind in channel.CHANNELS:
            ai = 28.0 if kind == "awgn" else 22.0
            for method, value in (("damaged", 20.0), ("classic", 25.0), ("ai", ai)):
                rows.append(dict(kind=kind, image=image, seed=0, level=0.5,
                                 method=method, psnr=value))
    text = "\n".join(evaluate.paired_section(rows))
    assert "| Gaussian noise (AWGN) | +3.00 |" in text   # AI beats classic by 3 dB
    assert "| Packet loss (lost 8x8 blocks) | -3.00 |" in text  # and loses by 3 dB here
    assert "3/3" in text
