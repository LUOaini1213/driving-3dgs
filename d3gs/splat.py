"""Web export: the de-facto '.splat' format (antimatter15/splat), 32 bytes per Gaussian.

  float32 x, y, z | float32 sx, sy, sz (linear scale, i.e. exp of the log-scale) |
  uint8 r, g, b, a (SH band-0 colour, sigmoid opacity) | uint8 qw, qx, qy, qz (unit quaternion, q*128+128)

Only view-independent colour (SH degree 0) survives this format. The historical ``web_subset_sh0``
evaluation used pre-quantisation parameters; the new ``decoded_asset_sh0`` evaluation in
``scripts/export.py`` renders the decoded bytes with gsplat (not the browser renderer).
"""
from __future__ import annotations

import numpy as np

from .geometry import normalise_quaternion

C0 = 0.28209479177387814
BYTES_PER_SPLAT = 32


def _finite_array(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value


def select_for_web(opacity: np.ndarray, log_scales: np.ndarray, max_n: int, min_opacity: float) -> np.ndarray:
    """Indices kept for the web asset: opacity >= min_opacity, then the ``max_n`` with the largest
    opacity * (projected-area proxy = product of the two largest scales). Sorted by that score, descending."""
    opacity = np.asarray(opacity, dtype=np.float64)
    if opacity.ndim != 1 or not np.isfinite(opacity).all() or ((opacity < 0) | (opacity > 1)).any():
        raise ValueError("opacity must be a finite vector in [0, 1]")
    if isinstance(max_n, (bool, np.bool_)) or not isinstance(max_n, (int, np.integer)) or max_n < 0:
        raise ValueError("max_n must be a nonnegative integer")
    if np.ndim(min_opacity) or not np.isfinite(min_opacity) or not 0 <= min_opacity <= 1:
        raise ValueError("min_opacity must be finite and in [0, 1]")
    s = _finite_array(log_scales, (len(opacity), 3), "log_scales").copy()
    s.sort(axis=1)
    # The log-score has the same order without exp/product overflow or underflow.
    with np.errstate(divide="ignore"):
        score = np.log(opacity) + s[:, 1] + s[:, 2]
    idx = np.where(opacity >= min_opacity)[0]
    idx = idx[np.argsort(-score[idx], kind="stable")]
    return idx[:max_n]


def encode_splat(means, log_scales, quats, opacity_logit, sh0) -> bytes:
    """Encode Gaussians (raw trained parameters, gsplat layout) to .splat bytes, in the given order."""
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    n = len(means)
    means = _finite_array(means, (n, 3), "means")
    log_scales = _finite_array(log_scales, (n, 3), "log_scales")
    quats = _finite_array(quats, (n, 4), "quaternions")
    sh0 = np.asarray(sh0, dtype=np.float64)
    if sh0.shape not in ((n, 3), (n, 1, 3)) or not np.isfinite(sh0).all():
        raise ValueError("sh0 must be finite with shape (N, 3) or (N, 1, 3)")
    opacity_logit = np.asarray(opacity_logit, dtype=np.float64)
    if opacity_logit.shape != (n,) or np.isnan(opacity_logit).any():
        raise ValueError("opacity logits must have shape (N,) without NaN")
    # +/-infinite logits are the well-defined fully opaque/transparent endpoints.
    with np.errstate(over="ignore", under="ignore"):
        pos = means.astype('<f4')
        scales = np.exp(log_scales).astype('<f4')
    if not np.isfinite(pos).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("positions and positive scales must be representable in float32")
    buf = np.zeros(n, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    buf["pos"] = pos
    buf["scale"] = scales
    rgb = np.clip(np.asarray(sh0).reshape(n, 3) * C0 + 0.5, 0, 1)
    with np.errstate(under="ignore"):
        tail = np.exp(-np.abs(opacity_logit))
    a = np.where(opacity_logit >= 0, 1 / (1 + tail), tail / (1 + tail))
    buf["rgba"][:, :3] = np.round(rgb * 255)
    buf["rgba"][:, 3] = np.round(a * 255)
    q = normalise_quaternion(quats)                         # (w, x, y, z)
    buf["rot"] = np.clip(np.round(q * 128 + 128), 0, 255)
    return buf.tobytes()


def decode_splat(data: bytes) -> dict[str, np.ndarray]:
    if len(data) % BYTES_PER_SPLAT:
        raise ValueError("splat data must contain complete 32-byte records")
    buf = np.frombuffer(data, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    if not np.isfinite(buf['pos']).all() or not np.isfinite(buf['scale']).all() or (buf['scale'] <= 0).any():
        raise ValueError("splat records must have finite positions and finite positive scales")
    q = (buf["rot"].astype(np.float64) - 128) / 128
    return {"means": buf["pos"].astype(np.float64), "scales": buf["scale"].astype(np.float64),
            "rgb": buf["rgba"][:, :3] / 255.0, "alpha": buf["rgba"][:, 3] / 255.0,
            "quats": normalise_quaternion(q)}
