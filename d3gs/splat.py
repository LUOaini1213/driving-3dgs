"""Web export: the de-facto '.splat' format (antimatter15/splat), 32 bytes per Gaussian.

  float32 x, y, z | float32 sx, sy, sz (linear scale, i.e. exp of the log-scale) |
  uint8 r, g, b, a (SH band-0 colour, sigmoid opacity) | uint8 qw, qx, qy, qz (unit quaternion, q*128+128)

Only the view-independent colour (SH degree 0) survives this format, so the web asset is a lossy copy
of the trained model; ``scripts/export.py`` measures how lossy (held-out PSNR of exactly what is exported).
"""
from __future__ import annotations

import numpy as np

C0 = 0.28209479177387814
BYTES_PER_SPLAT = 32


def select_for_web(opacity: np.ndarray, log_scales: np.ndarray, max_n: int, min_opacity: float) -> np.ndarray:
    """Indices kept for the web asset: opacity >= min_opacity, then the ``max_n`` with the largest
    opacity * (projected-area proxy = product of the two largest scales). Sorted by that score, descending."""
    s = np.exp(log_scales)
    s.sort(axis=1)
    score = opacity * s[:, 1] * s[:, 2]
    idx = np.where(opacity >= min_opacity)[0]
    idx = idx[np.argsort(-score[idx], kind="stable")]
    return idx[:max_n]


def encode_splat(means, log_scales, quats, opacity_logit, sh0) -> bytes:
    """Encode Gaussians (raw trained parameters, gsplat layout) to .splat bytes, in the given order."""
    n = len(means)
    buf = np.zeros(n, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    buf["pos"] = means
    buf["scale"] = np.exp(log_scales)
    rgb = np.clip(np.asarray(sh0).reshape(n, 3) * C0 + 0.5, 0, 1)
    a = 1 / (1 + np.exp(-np.asarray(opacity_logit, dtype=np.float64)))
    buf["rgba"][:, :3] = np.round(rgb * 255)
    buf["rgba"][:, 3] = np.round(a * 255)
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)           # (w, x, y, z)
    buf["rot"] = np.clip(np.round(q * 128 + 128), 0, 255)
    return buf.tobytes()


def decode_splat(data: bytes) -> dict[str, np.ndarray]:
    buf = np.frombuffer(data, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    q = (buf["rot"].astype(np.float64) - 128) / 128
    return {"means": buf["pos"].astype(np.float64), "scales": buf["scale"].astype(np.float64),
            "rgb": buf["rgba"][:, :3] / 255.0, "alpha": buf["rgba"][:, 3] / 255.0,
            "quats": q / np.linalg.norm(q, axis=1, keepdims=True)}
