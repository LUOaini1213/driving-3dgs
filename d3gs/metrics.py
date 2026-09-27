"""Image metrics in NumPy (evaluation) — PSNR and SSIM.

Images are float arrays in [0, 1], shape (H, W, 3). SSIM follows Wang et al. 2004
with an 11x11 Gaussian window (sigma 1.5), K1=0.01, K2=0.03, computed per channel
on the 'valid' region and averaged — the same recipe used by the 3DGS paper's
evaluation code.
"""
from __future__ import annotations

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _filter_valid(img: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Separable 'valid' correlation of a 2-D image with 1-D kernel g."""
    k = len(g)
    H, W = img.shape
    tmp = np.zeros((H, W - k + 1))
    for i in range(k):
        tmp += g[i] * img[:, i:i + W - k + 1]
    out = np.zeros((H - k + 1, W - k + 1))
    for i in range(k):
        out += g[i] * tmp[i:i + H - k + 1, :]
    return out


def _ssim_maps(a: np.ndarray, b: np.ndarray, data_range: float) -> list[np.ndarray]:
    """Per-channel SSIM maps on the 'valid' region, shape (H - 10, W - 10)."""
    g = _gaussian_kernel()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    maps = []
    for c in range(a.shape[-1]):
        x, y = a[..., c], b[..., c]
        mx, my = _filter_valid(x, g), _filter_valid(y, g)
        sxx = _filter_valid(x * x, g) - mx * mx
        syy = _filter_valid(y * y, g) - my * my
        sxy = _filter_valid(x * y, g) - mx * my
        maps.append(((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2)))
    return maps


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    return float(np.mean([m.mean() for m in _ssim_maps(a, b, data_range)]))


def psnr_masked(a: np.ndarray, b: np.ndarray, keep: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR over the pixels where ``keep`` (H, W) is True (all channels of those pixels)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    if a.shape != b.shape or keep.shape != a.shape[:2]:
        raise ValueError("shape mismatch")
    if not keep.any():
        return float("nan")
    return psnr(a[keep], b[keep], data_range)


def ssim_masked(a: np.ndarray, b: np.ndarray, keep: np.ndarray, data_range: float = 1.0) -> float:
    """Mean of the SSIM map over window centres where ``keep`` is True (same recipe as ``ssim``)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    r = (len(_gaussian_kernel()) - 1) // 2
    kc = keep[r:keep.shape[0] - r, r:keep.shape[1] - r]   # 'valid' map is offset by the window radius
    if not kc.any():
        return float("nan")
    return float(np.mean([m[kc].mean() for m in _ssim_maps(a, b, data_range)]))
