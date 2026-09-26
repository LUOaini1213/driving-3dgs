"""Pose math: quaternions, SE(3), camera intrinsics resizing.

Conventions
-----------
* AV2 stores rotations as scalar-first quaternions (qw, qx, qy, qz).
* ``a_SE3_b`` is a 4x4 matrix mapping points expressed in frame ``b`` into frame ``a``
  (same naming as the AV2 API): p_a = a_SE3_b @ p_b.
* AV2 camera frames follow the OpenCV convention (x right, y down, z forward),
  which is also what gsplat expects for its world-to-camera ``viewmats``.
"""
from __future__ import annotations

import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Scalar-first unit quaternion(s) (..., 4) -> rotation matrix (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> scalar-first unit quaternion with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_from_quat_trans(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return se3(quat_to_rotmat(q), t)


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid transform (…, 4, 4)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (N, 3) points."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def interpolate_pose(ts: np.ndarray, quats: np.ndarray, trans: np.ndarray, t_query: int) -> np.ndarray:
    """Pose at ``t_query`` (ns) by slerp/lerp between bracketing samples.

    ``ts`` must be sorted ascending. Exact matches return the stored pose.
    Raises if the query lies outside the sampled range (no extrapolation).
    """
    ts = np.asarray(ts, dtype=np.int64)
    i = int(np.searchsorted(ts, t_query))
    if i < len(ts) and ts[i] == t_query:
        return se3_from_quat_trans(quats[i], trans[i])
    if i == 0 or i == len(ts):
        raise ValueError(f"timestamp {t_query} outside pose range [{ts[0]}, {ts[-1]}]")
    t0, t1 = ts[i - 1], ts[i]
    a = (t_query - t0) / (t1 - t0)
    q = slerp(quats[i - 1], quats[i], a)
    t = (1 - a) * np.asarray(trans[i - 1], float) + a * np.asarray(trans[i], float)
    return se3_from_quat_trans(q, t)


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def resize_intrinsics(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Rescale K for an image resized from ``src_wh`` to ``dst_wh``.

    Uses the pixel-centre convention of OpenCV (pixel i covers [i-0.5, i+0.5]),
    i.e. u' + 0.5 = s * (u + 0.5).
    """
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K2 = np.array(K, dtype=np.float64).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
    K2[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
    return K2


def project(K: np.ndarray, cam_SE3_world: np.ndarray, pts_world: np.ndarray):
    """Project world points; returns (uv (N,2), depth (N,))."""
    pc = transform_points(cam_SE3_world, pts_world)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, z
