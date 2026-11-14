"""Image quality metrics (inputs: float32 RGB in [0, 1])."""
from __future__ import annotations

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref, img):
    """PSNR in dB (capped at 99 so identical images do not give infinity)."""
    return min(float(peak_signal_noise_ratio(ref, img, data_range=1.0)), 99.0)


def ssim(ref, img):
    return float(structural_similarity(ref, img, data_range=1.0, channel_axis=2))
