"""Simulated noisy telecom channel.

Images are float32 RGB arrays in [0, 1] with shape (H, W, 3).

Every damage type can be driven by a single ``level`` (severity) in [0, 1], which
keeps training, evaluation and the app consistent. ``describe`` converts a level
back into real units (SNR in dB, % of blocks lost, JPEG quality, ...).
"""
from __future__ import annotations

import numpy as np
import cv2

CHANNELS = ("awgn", "salt_pepper", "packet_loss", "jpeg", "combined")

LABELS = {
    "awgn": "Gaussian noise (AWGN)",
    "salt_pepper": "Impulse noise (salt & pepper)",
    "packet_loss": "Packet loss (lost 8x8 blocks)",
    "jpeg": "JPEG compression artifacts",
    "combined": "Combined (JPEG + AWGN + packet loss)",
}

BLOCK = 8  # a "packet" is an 8x8 pixel block


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


# --------------------------------------------------------------------------- #
# Individual impairments (real units)
# --------------------------------------------------------------------------- #
def awgn(img, snr_db, rng=None):
    """Add white Gaussian noise so that signal power / noise power = snr_db."""
    rng = _rng(rng)
    power = float(np.mean(img.astype(np.float64) ** 2)) + 1e-12
    sigma = np.sqrt(power / (10.0 ** (snr_db / 10.0)))
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)


def salt_pepper(img, density, rng=None):
    """Corrupt a fraction ``density`` of the pixels to pure black or pure white."""
    rng = _rng(rng)
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < density / 2.0] = 0.0
    out[(r >= density / 2.0) & (r < density)] = 1.0
    return out


def block_mask(shape_hw, loss_rate, rng=None, block=BLOCK):
    """Boolean (H, W) mask, True where a block ("packet") was lost."""
    rng = _rng(rng)
    h, w = shape_hw
    gh, gw = -(-h // block), -(-w // block)
    lost = rng.random((gh, gw)) < loss_rate
    return lost.repeat(block, axis=0).repeat(block, axis=1)[:h, :w]


def packet_loss(img, loss_rate, rng=None, block=BLOCK):
    """Zero out randomly chosen blocks (lost packets)."""
    out = img.copy()
    out[block_mask(img.shape[:2], loss_rate, rng, block)] = 0.0
    return out


def detect_lost_blocks(img, block=BLOCK):
    """Receiver-side detection of lost packets: blocks that are exactly zero."""
    h, w = img.shape[:2]
    gh, gw = -(-h // block), -(-w // block)
    nonzero = (img > 0).any(axis=2)
    padded = np.zeros((gh * block, gw * block), dtype=bool)
    padded[:h, :w] = nonzero
    block_has_data = padded.reshape(gh, block, gw, block).any(axis=(1, 3))
    lost = ~block_has_data
    return lost.repeat(block, axis=0).repeat(block, axis=1)[:h, :w]


def jpeg(img, quality):
    """Round-trip through JPEG at the given quality (1-100)."""
    u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return (cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)) / 255.0


# --------------------------------------------------------------------------- #
# Severity -> real units
# --------------------------------------------------------------------------- #
def params(kind, level):
    """Map severity level in [0, 1] to the real-unit parameters of ``kind``."""
    level = float(np.clip(level, 0.0, 1.0))
    if kind == "awgn":
        return {"snr_db": 30.0 - 25.0 * level}
    if kind == "salt_pepper":
        return {"density": 0.02 + 0.28 * level}
    if kind == "packet_loss":
        return {"loss_rate": 0.02 + 0.28 * level}
    if kind == "jpeg":
        return {"quality": int(round(75 - 65 * level))}
    if kind == "combined":
        return {
            "quality": int(round(90 - 55 * level)),
            "snr_db": 30.0 - 18.0 * level,
            "loss_rate": 0.02 + 0.18 * level,
        }
    raise ValueError(f"unknown channel kind: {kind!r}")


def apply_channel(img, kind, level, rng=None):
    """Damage ``img`` with channel ``kind`` at severity ``level`` in [0, 1]."""
    rng = _rng(rng)
    p = params(kind, level)
    if kind == "awgn":
        return awgn(img, p["snr_db"], rng)
    if kind == "salt_pepper":
        return salt_pepper(img, p["density"], rng)
    if kind == "packet_loss":
        return packet_loss(img, p["loss_rate"], rng)
    if kind == "jpeg":
        return jpeg(img, p["quality"])
    # combined: source coding (JPEG) -> noisy link (AWGN) -> lost packets
    out = jpeg(img, p["quality"])
    out = awgn(out, p["snr_db"], rng)
    return packet_loss(out, p["loss_rate"], rng)


def describe(kind, level):
    """Return (human readable text, x value, x axis label) for plots and the app."""
    p = params(kind, level)
    if kind == "awgn":
        return f"AWGN, SNR = {p['snr_db']:.1f} dB", p["snr_db"], "SNR (dB)"
    if kind == "salt_pepper":
        return (f"Salt & pepper, {p['density'] * 100:.0f}% of pixels",
                p["density"] * 100, "Corrupted pixels (%)")
    if kind == "packet_loss":
        return (f"Packet loss, {p['loss_rate'] * 100:.0f}% of blocks",
                p["loss_rate"] * 100, "Lost blocks (%)")
    if kind == "jpeg":
        return f"JPEG quality {p['quality']}", p["quality"], "JPEG quality"
    text = (f"JPEG q={p['quality']} + AWGN {p['snr_db']:.0f} dB + "
            f"{p['loss_rate'] * 100:.0f}% blocks lost")
    return text, float(np.clip(level, 0, 1)) * 100, "Severity (%)"
