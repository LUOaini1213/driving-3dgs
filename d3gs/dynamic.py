"""Dynamic-object masks from AV2 cuboid annotations.

* AV2 ``annotations.feather`` gives, at every lidar timestamp (10 Hz), each track's cuboid centre and
  orientation in the ego frame plus its size (length along object x, width along y, height along z).
* ``build_tracks`` moves every cuboid into the city frame and estimates a speed per sample from the
  centre displacement over a +-0.5 s window (annotation jitter is a few cm, so a single 0.1 s step is noisy).
* ``objects_at`` interpolates each track to a camera timestamp (lerp centre, slerp rotation).
* ``cuboid_mask`` rasterises a cuboid into the image: the 12 edges are clipped against the near plane
  *before* projection (corners behind the camera would otherwise project mirrored), the clipped points'
  convex hull is clipped to the image rectangle and filled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (quat_to_rotmat, se3, se3_from_quat_trans, slerp, rotmat_to_quat,
                       image_size, validate_intrinsics, validate_transform, timestamps_ns)

VEHICLE_CATEGORIES = {
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER",
    "ARTICULATED_BUS", "SCHOOL_BUS", "MOTORCYCLE", "BICYCLE", "MOTORCYCLIST", "BICYCLIST", "WHEELED_RIDER",
    "RAILED_VEHICLE", "MESSAGE_BOARD_TRAILER",
}

_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def cuboid_corners(dims) -> np.ndarray:
    """8 corners (object frame) of a cuboid centred at the origin; dims = (length_x, width_y, height_z)."""
    dims = np.asarray(dims, dtype=np.float64)
    if dims.shape != (3,) or not np.isfinite(dims).all() or (dims <= 0).any():
        raise ValueError("cuboid dimensions must be three finite positive lengths")
    l, w, h = (float(d) / 2 for d in dims)
    return np.array([[sx * l, sy * w, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def _clip_edges_near(pc: np.ndarray, near: float) -> np.ndarray:
    pts = [p for p in pc if p[2] >= near]
    for a, b in _EDGES:
        za, zb = pc[a, 2], pc[b, 2]
        if (za - near) * (zb - near) < 0:          # edge crosses the near plane
            t = (near - za) / (zb - za)
            pts.append(pc[a] + t * (pc[b] - pc[a]))
    return np.array(pts).reshape(-1, 3)


def _convex_hull(p: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; returns hull vertices counter-clockwise (no repeats)."""
    p = np.unique(p, axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def _clip_poly_rect(poly: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to an axis-aligned rectangle."""
    def clip(pts, inside, inter):
        out = []
        for k in range(len(pts)):
            cur, prev = pts[k], pts[k - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
        return out

    def ix(xc):
        return lambda a, b: a + (xc - a[0]) / (b[0] - a[0]) * (b - a)

    def iy(yc):
        return lambda a, b: a + (yc - a[1]) / (b[1] - a[1]) * (b - a)

    pts = [np.asarray(q, float) for q in poly]
    for inside, inter in ((lambda q: q[0] >= x0, ix(x0)), (lambda q: q[0] <= x1, ix(x1)),
                          (lambda q: q[1] >= y0, iy(y0)), (lambda q: q[1] <= y1, iy(y1))):
        if not pts:
            break
        pts = clip(pts, inside, inter)
    return np.array(pts).reshape(-1, 2)


def cuboid_mask(K: np.ndarray, cam_SE3_obj: np.ndarray, dims, W: int, H: int, near: float = 0.1) -> np.ndarray:
    """Boolean (H, W) mask of the pixels whose centre lies inside the cuboid's projection.

    Pixel centres are at integer coordinates (the convention of ``resize_intrinsics``).
    """
    W, H = image_size((W, H))
    K, cam_SE3_obj = validate_intrinsics(K), validate_transform(cam_SE3_obj)
    if cam_SE3_obj.shape != (4, 4) or not np.isfinite(near) or near <= 0:
        raise ValueError("cuboid projection needs one pose and a finite positive near plane")
    c = cuboid_corners(dims)
    pc = c @ cam_SE3_obj[:3, :3].T + cam_SE3_obj[:3, 3]
    pc = _clip_edges_near(pc, near)
    if len(pc) < 3:
        return np.zeros((H, W), bool)
    uv = pc[:, :2] / pc[:, 2:3] @ K[:2, :2].T + K[:2, 2]
    poly = _clip_poly_rect(_convex_hull(uv), -1.0, -1.0, W, H)
    return fill_convex(poly, W, H)


def fill_convex(poly: np.ndarray, W: int, H: int) -> np.ndarray:
    """Pixels (integer centres) inside a convex polygon, edges inclusive."""
    mask = np.zeros((H, W), bool)
    if len(poly) < 3:
        return mask
    x0, y0 = max(int(np.ceil(poly[:, 0].min())), 0), max(int(np.ceil(poly[:, 1].min())), 0)
    x1, y1 = min(int(np.floor(poly[:, 0].max())), W - 1), min(int(np.floor(poly[:, 1].max())), H - 1)
    if x1 < x0 or y1 < y0:
        return mask
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    area2 = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    sgn = 1.0 if area2 >= 0 else -1.0
    inside = np.ones(xs.shape, bool)
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        cr = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
        inside &= sgn * cr >= -1e-9
    mask[y0:y1 + 1, x0:x1 + 1] = inside
    return mask


@dataclass
class Track:
    uuid: str
    category: str
    ts: np.ndarray          # (N,) int64 ns
    city_SE3_obj: np.ndarray  # (N, 4, 4)
    dims: np.ndarray        # (N, 3) length, width, height
    speed: np.ndarray       # (N,) m/s, from centre displacement over +-window

    def __post_init__(self):
        self.ts = timestamps_ns(self.ts)
        self.city_SE3_obj = validate_transform(self.city_SE3_obj)
        self.dims, self.speed = np.asarray(self.dims, float), np.asarray(self.speed, float)
        n = len(self.ts)
        if self.city_SE3_obj.shape != (n, 4, 4) or self.dims.shape != (n, 3) or self.speed.shape != (n,):
            raise ValueError("track arrays must have one pose, size and speed per timestamp")
        if not np.isfinite(self.dims).all() or (self.dims <= 0).any() or not np.isfinite(self.speed).all() or (self.speed < 0).any():
            raise ValueError("track sizes must be positive and speeds nonnegative, all finite")


def track_speed(ts: np.ndarray, centres: np.ndarray, half_window_s: float = 0.5) -> np.ndarray:
    """Speed (m/s) per sample: |c(t+w) - c(t-w)| / dt using the samples nearest to t+-w inside the track."""
    ts = timestamps_ns(ts, allow_empty=True)
    centres = np.asarray(centres, dtype=np.float64)
    if centres.shape != (len(ts), 3) or not np.isfinite(centres).all():
        raise ValueError("track centres must have finite shape (N, 3)")
    if not np.isfinite(half_window_s) or half_window_s <= 0:
        raise ValueError("speed window must be finite and positive")
    # Subtract integer ns before conversion, preserving short intervals at epoch timestamps.
    t = np.array([(int(v) - int(ts[0])) / 1e9 for v in ts])
    sp = np.zeros(len(ts))
    for k in range(len(ts)):
        a = int(np.argmin(np.abs(t - (t[k] - half_window_s))))
        b = int(np.argmin(np.abs(t - (t[k] + half_window_s))))
        if b > a:
            sp[k] = np.linalg.norm(centres[b, :2] - centres[a, :2]) / (t[b] - t[a])
    return sp


def build_tracks(ann, poses, t_min: int | None = None, t_max: int | None = None) -> list[Track]:
    """Cuboid tracks in the city frame from an AV2 annotations DataFrame and a PoseTable."""
    if t_min is not None:
        ann = ann[ann.timestamp_ns >= t_min]
    if t_max is not None:
        ann = ann[ann.timestamp_ns <= t_max]
    ego_cache: dict[int, np.ndarray] = {}
    tracks = []
    for uuid, g in ann.groupby("track_uuid"):
        g = g.sort_values("timestamp_ns")
        ts = timestamps_ns(g.timestamp_ns.to_numpy())
        T = []
        for t, q, p in zip(ts, g[["qw", "qx", "qy", "qz"]].to_numpy(float), g[["tx_m", "ty_m", "tz_m"]].to_numpy(float)):
            if t not in ego_cache:
                ego_cache[t] = poses.city_SE3_ego(int(t))
            T.append(ego_cache[t] @ se3_from_quat_trans(q, p))
        T = np.stack(T)
        tracks.append(Track(str(uuid), str(g.category.iloc[0]), ts, T,
                            g[["length_m", "width_m", "height_m"]].to_numpy(float), track_speed(ts, T[:, :3, 3])))
    return tracks


def objects_at(tracks: list[Track], t_ns: int, max_gap_ns: int = 60_000_000):
    """[(track, city_SE3_obj, dims, speed)] interpolated to t_ns.

    Inside a track's time span: lerp the centre / slerp the rotation between bracketing samples,
    speed = max of the two. Outside the span by at most ``max_gap_ns``: nearest sample. Otherwise absent.
    """
    if isinstance(t_ns, (bool, np.bool_)) or not isinstance(t_ns, (int, np.integer)):
        raise ValueError("query timestamp must be integer nanoseconds")
    if isinstance(max_gap_ns, (bool, np.bool_)) or not isinstance(max_gap_ns, (int, np.integer)) or max_gap_ns < 0:
        raise ValueError("maximum gap must be nonnegative integer nanoseconds")
    out = []
    for tr in tracks:
        ts = tr.ts
        i = int(np.searchsorted(ts, t_ns))
        if i < len(ts) and ts[i] == t_ns:
            out.append((tr, tr.city_SE3_obj[i], tr.dims[i], float(tr.speed[i])))
        elif 0 < i < len(ts):
            t0, t1 = int(ts[i - 1]), int(ts[i])
            a = (t_ns - t0) / (t1 - t0)
            A, B = tr.city_SE3_obj[i - 1], tr.city_SE3_obj[i]
            q = slerp(rotmat_to_quat(A[:3, :3]), rotmat_to_quat(B[:3, :3]), a)
            T = se3(quat_to_rotmat(q), (1 - a) * A[:3, 3] + a * B[:3, 3])
            out.append((tr, T, tr.dims[i - 1], float(max(tr.speed[i - 1], tr.speed[i]))))
        else:
            j = 0 if i == 0 else len(ts) - 1
            if abs(int(ts[j]) - int(t_ns)) <= max_gap_ns:
                out.append((tr, tr.city_SE3_obj[j], tr.dims[j], float(tr.speed[j])))
    return out


def frame_masks(tracks, t_ns: int, K, city_SE3_cam: np.ndarray, W: int, H: int, speed_thresh: float = 1.0,
                pad_m: float = 0.25):
    """(moving_mask, vehicle_mask, n_moving_objects) for one camera frame.

    moving  = any object with speed > speed_thresh; vehicles = moving objects + every vehicle category.
    Each cuboid is padded by ``pad_m`` on every side to absorb annotation / timing slack.
    """
    from .geometry import se3_inverse
    W, H = image_size((W, H))
    K = validate_intrinsics(K)
    if not np.isfinite([speed_thresh, pad_m]).all() or speed_thresh < 0 or pad_m < 0:
        raise ValueError("speed threshold and padding must be finite and nonnegative")
    cam_SE3_city = se3_inverse(city_SE3_cam)
    mov = np.zeros((H, W), bool)
    veh = np.zeros((H, W), bool)
    n_mov = 0
    for tr, T, dims, sp in objects_at(tracks, t_ns):
        is_mov = sp > speed_thresh
        if not (is_mov or tr.category in VEHICLE_CATEGORIES):
            continue
        m = cuboid_mask(K, cam_SE3_city @ T, np.asarray(dims) + 2 * pad_m, W, H)
        if is_mov:
            mov |= m
            n_mov += int(m.any())
        veh |= m
    return mov, veh, n_mov
