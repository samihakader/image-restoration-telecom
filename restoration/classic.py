"""Classic (non-learned) restoration baselines built on OpenCV.

``classic_restore`` is deliberately a *strong* baseline: it is told which kind of
damage happened and picks the standard textbook tool for it. The neural network
has no such side information, so any win for the network is an honest win.
"""
from __future__ import annotations

import numpy as np
import cv2

from .channel import detect_lost_blocks


def _u8(img):
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def _f32(u8):
    return u8.astype(np.float32) / 255.0


def median(img, k=3):
    return _f32(cv2.medianBlur(_u8(img), k))


def gaussian(img, sigma=1.0):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def bilateral(img, d=7, sigma_color=40, sigma_space=5):
    return _f32(cv2.bilateralFilter(_u8(img), d, sigma_color, sigma_space))


def nlm(img, h):
    """Non-local means denoising (colour version, filter strength h)."""
    bgr = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2BGR)
    out = cv2.fastNlMeansDenoisingColored(bgr, None, float(h), float(h), 7, 21)
    return _f32(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def inpaint(img, mask, radius=3):
    """Fill the masked pixels from their surroundings (Telea fast marching)."""
    bgr = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return _f32(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def estimate_noise_sigma(img):
    """Blind noise level estimate (Immerkaer's method), in the [0, 1] intensity scale."""
    gray = img.mean(axis=2).astype(np.float32)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    resp = cv2.filter2D(gray, cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REFLECT)
    resp = resp[1:-1, 1:-1]
    # the colour average of 3 independent channels has sigma / sqrt(3)
    return float(np.sqrt(np.pi / 2.0) * np.abs(resp).mean() / 6.0 * np.sqrt(3.0))


def _impulse_fraction(img):
    extreme = (img == 0).all(axis=2) | (img == 1).all(axis=2)
    return float(extreme.mean())


def denoise_gaussian_noise(img):
    """Blind denoiser: bilateral filter for light noise, NLM for heavy noise."""
    sigma = estimate_noise_sigma(img)
    if sigma < SIGMA_SWITCH:
        return bilateral(img, 7, 40, 5)
    return nlm(img, max(2.0, 255.0 * sigma * NLM_GAIN))


def classic_restore(img, kind):
    """Best classic tool for the given damage type."""
    if kind == "awgn":
        return denoise_gaussian_noise(img)
    if kind == "salt_pepper":
        k = 3 if _impulse_fraction(img) < IMPULSE_SWITCH else 5
        return median(img, k)
    if kind == "packet_loss":
        return inpaint(img, detect_lost_blocks(img), radius=INPAINT_RADIUS)
    if kind == "jpeg":
        return bilateral(img, 5, 20, 3)
    if kind == "combined":
        filled = inpaint(img, detect_lost_blocks(img), radius=INPAINT_RADIUS)
        return denoise_gaussian_noise(filled)
    raise ValueError(f"unknown channel kind: {kind!r}")


# Settings picked by a small sweep on the bundled sample images, so the baseline is
# as strong as simple classic tools allow (see README, "Fair baseline").
NLM_GAIN = 0.7          # NLM strength relative to the estimated noise sigma
SIGMA_SWITCH = 0.10     # below this estimated sigma use bilateral, above it NLM
IMPULSE_SWITCH = 0.19   # above this fraction of extreme pixels use a 5x5 median
INPAINT_RADIUS = 3
