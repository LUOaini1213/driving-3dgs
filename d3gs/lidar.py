"""Lidar -> camera depth maps, depth-error metrics, and the held-out lidar leakage guard.

Leakage rule (used by prepare.py, train.py and evaluate.py)
-----------------------------------------------------------
For every held-out camera frame, the time-nearest lidar sweep is its *evaluation sweep*.
Evaluation sweeps may only be read by the depth evaluation of held-out frames. They must
never contribute to the Gaussian initialisation or to depth supervision; ``assert_no_lidar_leak``
raises if they do. AV2 sweeps are already motion-compensated to the sweep timestamp (checked on
this log: re-compensating each point with its ``offset_ns`` made consecutive sweeps align worse),
so a sweep is moved into the world with the ego pose at its own timestamp.
"""
from __future__ import annotations

import numpy as np

from .geometry import project, image_size


class LidarLeakError(RuntimeError):
    """A sweep reserved for held-out depth evaluation was used for training."""


def split_sweeps(sweep_ts, frame_ts, test_idx) -> tuple[list[int], list[int]]:
    """Return (train_sweep_idx, eval_sweep_idx) over ``sweep_ts``.

    eval = the time-nearest sweep of each held-out frame (ties -> the earlier sweep);
    train = every other sweep. Both lists are sorted indices into ``sweep_ts``.
    """
    sweep_ts = np.asarray(sweep_ts, dtype=np.int64)
    frame_ts = np.asarray(frame_ts, dtype=np.int64)
    if len(sweep_ts) == 0:
        return [], []
    ev = set()
    for i in test_idx:
        ev.add(int(np.argmin(np.abs(sweep_ts - frame_ts[i]))))
    eval_idx = sorted(ev)
    train_idx = [k for k in range(len(sweep_ts)) if k not in ev]
    return train_idx, eval_idx


def assert_no_lidar_leak(used_sweep_ts, eval_sweep_ts) -> None:
    """Raise LidarLeakError if any sweep timestamp used for training is an evaluation sweep."""
    overlap = {int(t) for t in used_sweep_ts} & {int(t) for t in eval_sweep_ts}
    if overlap:
        raise LidarLeakError(f"{len(overlap)} held-out evaluation sweep(s) used for training, e.g. {min(overlap)}")


def nearest_sweep(sweep_ts, t_ns: int, allowed) -> int:
    """Index (into sweep_ts) of the time-nearest sweep among ``allowed`` indices."""
    allowed = list(allowed)
    ts = np.asarray(sweep_ts, dtype=np.int64)[allowed]
    return allowed[int(np.argmin(np.abs(ts - int(t_ns))))]


def baseline_sweep(sweep_ts, t_ns: int, training_sweep_ts, eval_sweep_ts) -> int:
    """Lidar-only depth baseline: index of the time-nearest sweep the trained model could have seen.

    ``training_sweep_ts`` are the sweeps used for initialisation or depth supervision; the
    chosen sweep is checked against ``eval_sweep_ts`` so the baseline never reads held-out lidar.
    """
    sweep_ts = np.asarray(sweep_ts, dtype=np.int64)
    pos = {int(t): k for k, t in enumerate(sweep_ts)}
    allowed = sorted(pos[int(t)] for t in {int(t) for t in training_sweep_ts} if int(t) in pos)
    if not allowed:
        raise ValueError("no training sweep available for the lidar baseline")
    k = nearest_sweep(sweep_ts, t_ns, allowed)
    assert_no_lidar_leak([int(sweep_ts[k])], eval_sweep_ts)
    return k


def fill_nearest(D: np.ndarray) -> np.ndarray:
    """Dense depth from a sparse map: every empty pixel (0) takes the depth of the nearest pixel
    (Euclidean, image space) that has a lidar return. Pixels with a return keep their value."""
    from scipy.ndimage import distance_transform_edt
    D = np.asarray(D)
    empty = ~(D > 0)
    if empty.all():
        raise ValueError("cannot fill a depth map without any lidar return")
    if not empty.any():
        return D.copy()
    idx = distance_transform_edt(empty, return_distances=False, return_indices=True)
    return D[tuple(idx)]


def pooled_depth_errors(preds, gts, keeps=None, tol: float = 0.5,
                        bins=((0, 10), (10, 20), (20, 40), (40, 80))) -> dict:
    """Errors over the lidar pixels of several frames pooled (the ``heldout_depth`` layout of evaluate.py).

    ``keeps`` (optional, per frame) restricts the cohort, e.g. to static pixels. The result is
    ``depth_errors`` rounded to 4 decimals, plus ``by_range`` keyed by the lidar depth bin.
    """
    P, G = [], []
    for k, (pred, gt) in enumerate(zip(preds, gts)):
        v = np.asarray(gt) > 0
        if keeps is not None and keeps[k] is not None:
            v &= np.asarray(keeps[k], dtype=bool)
        P.append(np.asarray(pred)[v])
        G.append(np.asarray(gt)[v])
    P, G = np.concatenate(P), np.concatenate(G)

    def rnd(d: dict) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    out = rnd(depth_errors(P, G, tol=tol))
    out["by_range"] = {f"{lo}-{hi}m": rnd(depth_errors(P, np.where((G >= lo) & (G < hi), G, 0), tol=tol))
                       for lo, hi in bins}
    return out


def depth_map(K: np.ndarray, cam_SE3_world: np.ndarray, pts_world: np.ndarray, W: int, H: int,
              near: float = 0.5, far: float = 80.0) -> np.ndarray:
    """Sparse z-depth image (H, W) float32 from world points; 0 = no lidar return.

    A point lands in pixel (round(u), round(v)) (pixel centres at integer coordinates, the same
    convention as ``resize_intrinsics``); when several points hit one pixel the nearest wins.
    """
    W, H = image_size((W, H))
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError("depth planes must be finite with 0 < near < far")
    uv, z = project(K, cam_SE3_world, pts_world)
    ok = (z > near) & (z < far) & np.isfinite(uv).all(1)
    u = np.round(uv[ok, 0]).astype(np.int64)
    v = np.round(uv[ok, 1]).astype(np.int64)
    z = z[ok]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inb], v[inb], z[inb]
    D = np.full(H * W, np.inf)
    np.minimum.at(D, v * W + u, z)
    D[~np.isfinite(D)] = 0.0
    return D.reshape(H, W).astype(np.float32)


def depth_errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None, tol: float = 0.5) -> dict:
    """Errors on the fixed ``(gt > 0) & valid`` cohort; invalid predictions there raise.

    Predictions outside that cohort are irrelevant. An empty cohort returns n=0/null metrics.
    """
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError("prediction/ground-truth shape mismatch")
    if not np.isfinite(gt).all() or (gt < 0).any():
        raise ValueError("ground-truth depth must be finite and nonnegative")
    if np.ndim(tol) or isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol < 0:
        raise ValueError("depth tolerance must be finite and nonnegative")
    m = gt > 0
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != gt.shape or not np.isin(valid, [0, 1]).all():
            raise ValueError("valid mask must have the depth shape and boolean/0/1 values")
        m &= valid.astype(bool)
    if not np.isfinite(pred[m]).all():
        raise ValueError("nonfinite prediction on the evaluation depth cohort")
    if not m.any():
        return {"n": 0, "median_abs_m": None, "mean_abs_m": None, "within_tol": None, "abs_rel": None}
    e = np.abs(pred[m].astype(np.float64) - gt[m].astype(np.float64))
    return {"n": int(m.sum()), "median_abs_m": float(np.median(e)), "mean_abs_m": float(e.mean()),
            "within_tol": float((e <= tol).mean()), "abs_rel": float(np.mean(e / gt[m]))}
