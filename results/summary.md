# Results

Mean over 28 test image(s) x 3 random seed(s). PSNR in dB (higher is better), SSIM in [0, 1].


## Gaussian noise (AWGN)

| SNR (dB) | Damaged PSNR | Damaged SSIM | Classic (OpenCV) PSNR | Classic (OpenCV) SSIM | AI (CNN) PSNR | AI (CNN) SSIM |
|---|---|---|---|---|---|---|
| 30.0 | 36.14 | 0.870 | 37.82 | 0.956 | 38.71 | 0.943 |
| 25.0 | 31.22 | 0.734 | 36.06 | 0.932 | 37.30 | 0.945 |
| 20.0 | 26.35 | 0.560 | 32.11 | 0.810 | 34.58 | 0.914 |
| 15.0 | 21.60 | 0.387 | 28.17 | 0.759 | 31.38 | 0.845 |
| 10.0 | 17.02 | 0.239 | 24.57 | 0.698 | 27.84 | 0.712 |
| 5.0 | 12.82 | 0.133 | 20.26 | 0.587 | 20.49 | 0.579 |

## Impulse noise (salt & pepper)

| Corrupted pixels (%) | Damaged PSNR | Damaged SSIM | Classic (OpenCV) PSNR | Classic (OpenCV) SSIM | AI (CNN) PSNR | AI (CNN) SSIM |
|---|---|---|---|---|---|---|
| 2.0 | 21.59 | 0.653 | 33.15 | 0.927 | 28.94 | 0.826 |
| 7.6 | 15.74 | 0.278 | 31.98 | 0.921 | 35.82 | 0.912 |
| 13.2 | 13.35 | 0.172 | 30.04 | 0.907 | 34.93 | 0.902 |
| 18.8 | 11.85 | 0.127 | 27.77 | 0.882 | 33.06 | 0.868 |
| 24.4 | 10.75 | 0.101 | 27.82 | 0.852 | 31.23 | 0.824 |
| 30.0 | 9.84 | 0.082 | 27.26 | 0.845 | 29.36 | 0.769 |

## Packet loss (lost 8x8 blocks)

| Lost blocks (%) | Damaged PSNR | Damaged SSIM | Classic (OpenCV) PSNR | Classic (OpenCV) SSIM | AI (CNN) PSNR | AI (CNN) SSIM |
|---|---|---|---|---|---|---|
| 2.0 | 23.88 | 0.960 | 43.31 | 0.996 | 39.33 | 0.985 |
| 7.6 | 17.72 | 0.856 | 35.76 | 0.983 | 33.74 | 0.954 |
| 13.2 | 15.13 | 0.755 | 32.85 | 0.969 | 31.21 | 0.926 |
| 18.8 | 13.31 | 0.651 | 30.86 | 0.952 | 29.43 | 0.895 |
| 24.4 | 12.12 | 0.564 | 29.40 | 0.935 | 28.10 | 0.864 |
| 30.0 | 11.19 | 0.490 | 28.16 | 0.916 | 26.82 | 0.834 |

## JPEG compression artifacts

| JPEG quality | Damaged PSNR | Damaged SSIM | Classic (OpenCV) PSNR | Classic (OpenCV) SSIM | AI (CNN) PSNR | AI (CNN) SSIM |
|---|---|---|---|---|---|---|
| 75.0 | 34.71 | 0.948 | 35.18 | 0.954 | 34.56 | 0.948 |
| 62.0 | 33.44 | 0.935 | 33.89 | 0.942 | 33.71 | 0.937 |
| 49.0 | 32.63 | 0.925 | 33.06 | 0.934 | 32.72 | 0.926 |
| 36.0 | 31.71 | 0.911 | 32.10 | 0.921 | 31.89 | 0.914 |
| 23.0 | 30.29 | 0.881 | 30.61 | 0.893 | 30.66 | 0.885 |
| 10.0 | 27.45 | 0.809 | 27.63 | 0.821 | 27.54 | 0.811 |

## Combined (JPEG + AWGN + packet loss)

| Severity (%) | Damaged PSNR | Damaged SSIM | Classic (OpenCV) PSNR | Classic (OpenCV) SSIM | AI (CNN) PSNR | AI (CNN) SSIM |
|---|---|---|---|---|---|---|
| 0.0 | 22.75 | 0.808 | 34.54 | 0.941 | 34.32 | 0.916 |
| 20.0 | 18.26 | 0.658 | 31.85 | 0.911 | 31.97 | 0.900 |
| 40.0 | 15.80 | 0.510 | 29.66 | 0.855 | 30.06 | 0.868 |
| 60.0 | 14.36 | 0.385 | 27.27 | 0.734 | 28.55 | 0.820 |
| 80.0 | 13.08 | 0.281 | 24.61 | 0.596 | 26.90 | 0.753 |
| 100.0 | 11.95 | 0.197 | 23.65 | 0.636 | 25.14 | 0.661 |

## Average PSNR gain of AI over classic

| Damage | Gain (dB) |
|---|---|
| Gaussian noise (AWGN) | +1.88 |
| Impulse noise (salt & pepper) | +2.55 |
| Packet loss (lost 8x8 blocks) | -1.95 |
| JPEG compression artifacts | -0.23 |
| Combined (JPEG + AWGN + packet loss) | +0.89 |

## Is the AI's advantage real? (paired statistics)

Differences are computed on the same damaged image, averaged per test image, then summarised across images. A gain is only convincing when its 95% interval stays above 0 and the AI wins on most images.


### AI minus classic OpenCV

| Damage | Mean gain (dB) | 95% interval | Images AI wins | Cases AI wins |
|---|---|---|---|---|
| Gaussian noise (AWGN) | +1.88 | +1.39 to +2.38 | 25/28 | 83% |
| Impulse noise (salt & pepper) | +2.55 | +0.92 to +4.19 | 22/28 | 77% |
| Packet loss (lost 8x8 blocks) | -1.95 | -3.23 to -0.68 | 8/28 | 39% |
| JPEG compression artifacts | -0.23 | -0.54 to +0.08 | 10/28 | 42% |
| Combined (JPEG + AWGN + packet loss) | +0.89 | +0.56 to +1.22 | 24/28 | 78% |

### AI minus doing nothing (damaged image)

| Damage | Mean gain (dB) | 95% interval | Images AI wins | Cases AI wins |
|---|---|---|---|---|
| Gaussian noise (AWGN) | +7.52 | +6.89 to +8.16 | 28/28 | 98% |
| Impulse noise (salt & pepper) | +18.37 | +17.48 to +19.26 | 28/28 | 100% |
| Packet loss (lost 8x8 blocks) | +15.88 | +14.11 to +17.65 | 28/28 | 99% |
| JPEG compression artifacts | +0.14 | -0.17 to +0.46 | 26/28 | 89% |
| Combined (JPEG + AWGN + packet loss) | +13.46 | +12.08 to +14.83 | 28/28 | 100% |
