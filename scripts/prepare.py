"""AV2 log window -> undistorted, cropped, downscaled frames + camera-to-world + lidar init.

Steps
  1. city_SE3_cam(t) = city_SE3_ego(t) @ ego_SE3_cam   (pose interpolated at the image timestamp)
  2. world = city shifted by the mean training-camera centre (keeps float32 precise)
  3. undistort (k1,k2,k3 radial) with the same K, crop the ego hood rows, resize
  4. hold out every Nth frame (deterministic)
  5. lidar init: sweeps -> world, keep points seen by >=1 TRAIN camera, colour them
     from the time-nearest train image; plus a far 'sky shell' sampled from train images.
     Held-out images are never read here.

Usage:
  python scripts/prepare.py --log-dir D:/driving-3dgs/data/<log> --out D:/driving-3dgs/work/<log>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d3gs.av2io import list_frames, list_sweeps, load_camera_calib, load_poses, load_sweep_xyz  # noqa: E402
from d3gs.geometry import project, resize_intrinsics, se3_inverse, transform_points  # noqa: E402
from d3gs.split import holdout_split  # noqa: E402


def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float):
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx], cols[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="ring_front_center")
    ap.add_argument("--crop-bottom", type=int, default=1824, help="keep rows [0, crop) of the undistorted image (drops ego hood)")
    ap.add_argument("--width", type=int, default=388)
    ap.add_argument("--height", type=int, default=456)
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--train-stride", type=int, default=1,
                    help="sparser-view variant: keep only train frames with idx %% k == 0 (held-out set unchanged)")
    ap.add_argument("--voxel", type=float, default=0.08)
    ap.add_argument("--max-lidar-pts", type=int, default=200_000)
    ap.add_argument("--n-sky", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log_dir, out = Path(args.log_dir), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    calib = load_camera_calib(log_dir, args.camera)
    poses = load_poses(log_dir)
    frames = list_frames(log_dir, args.camera)
    train, test = holdout_split(len(frames), args.holdout_every)
    train = [i for i in train if i % args.train_stride == 0]

    # --- camera poses ------------------------------------------------------
    city_SE3_cam = np.stack([poses.city_SE3_ego(t) @ calib.ego_SE3_cam for t, _ in frames])
    gaps = [poses.gap_ns(t) for t, _ in frames]
    origin = city_SE3_cam[train, :3, 3].mean(0)          # recentre on train cameras
    c2w = city_SE3_cam.copy()
    c2w[:, :3, 3] -= origin

    # --- images ------------------------------------------------------------
    K_full = calib.K
    src_wh = (calib.width, args.crop_bottom)          # cropping bottom rows keeps K unchanged
    K = resize_intrinsics(K_full, src_wh, (args.width, args.height))
    dist = np.array([calib.dist[0], calib.dist[1], 0.0, 0.0, calib.dist[2]])
    map1, map2 = cv2.initUndistortRectifyMap(K_full, dist, None, K_full, (calib.width, calib.height), cv2.CV_32FC1)
    small = []
    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)[: args.crop_bottom]
        sm = cv2.resize(und, (args.width, args.height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "images" / f"{i:04d}.png"), sm)
        small.append(sm[:, :, ::-1])  # RGB

    # --- lidar init (train cameras only) -------------------------------------
    t_frames = np.array([t for t, _ in frames])
    w2c = se3_inverse(c2w)
    all_p, all_c, n_raw = [], [], 0
    for ts, sp in list_sweeps(log_dir):
        xyz = load_sweep_xyz(sp)
        n_raw += len(xyz)
        r = np.linalg.norm(xyz[:, :2], axis=1)
        xyz = xyz[(r > 3.0) & (r < 80.0)]                # drop ego returns and far sparse points
        pw = transform_points(poses.city_SE3_ego(ts), xyz) - origin
        # colour from the time-nearest TRAIN frame that sees the point
        order = sorted(train, key=lambda j: abs(t_frames[j] - ts))
        col = np.full((len(pw), 3), -1.0)
        for j in order[:6]:
            uv, z = project(K, w2c[j], pw)
            ok = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < args.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < args.height - 1) & (col[:, 0] < 0)
            if ok.any():
                u, v = uv[ok, 0].round().astype(int), uv[ok, 1].round().astype(int)
                col[ok] = small[j][v, u] / 255.0
        keep = col[:, 0] >= 0
        all_p.append(pw[keep])
        all_c.append(col[keep])
    pts, cols = np.concatenate(all_p), np.concatenate(all_c)
    n_seen = len(pts)
    pts, cols = voxel_downsample(pts, cols, args.voxel)
    if len(pts) > args.max_lidar_pts:
        sel = rng.choice(len(pts), args.max_lidar_pts, replace=False)
        pts, cols = pts[sel], cols[sel]

    # --- far 'sky shell' from train images (upper 60% of rows) ---------------
    Kinv = np.linalg.inv(K)
    js = rng.choice(train, args.n_sky)
    u = rng.uniform(0, args.width - 1, args.n_sky)
    v = rng.uniform(0, 0.6 * (args.height - 1), args.n_sky)
    d = rng.uniform(150.0, 300.0, args.n_sky)
    rays = (Kinv @ np.stack([u, v, np.ones_like(u)])).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    sky_p = np.stack([transform_points(c2w[j], (rays[k] * d[k])[None])[0] for k, j in enumerate(js)])
    sky_c = np.stack([small[j][int(round(v[k])), int(round(u[k]))] / 255.0 for k, j in enumerate(js)])

    np.savez_compressed(out / "init_points.npz", xyz=np.concatenate([pts, sky_p]).astype(np.float32),
                        rgb=np.concatenate([cols, sky_c]).astype(np.float32),
                        n_lidar=len(pts), n_sky=args.n_sky)

    cams = {
        "log_dir": str(log_dir), "camera": args.camera, "width": args.width, "height": args.height,
        "K": K.tolist(), "city_origin": origin.tolist(), "holdout_every": args.holdout_every,
        "train": train, "test": test, "train_stride": args.train_stride, "crop_bottom": args.crop_bottom,
        "pose_gap_ms_max": max(gaps) / 1e6,
        "frames": [{"idx": i, "timestamp_ns": int(t), "image": f"images/{i:04d}.png", "c2w": c2w[i].tolist()}
                   for i, (t, _) in enumerate(frames)],
        "init": {"lidar_points_raw": int(n_raw), "lidar_points_seen_by_train": int(n_seen),
                 "lidar_points_after_voxel": int(len(pts)), "sky_points": args.n_sky, "voxel_m": args.voxel},
        "trajectory_length_m": float(np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum()),
        "duration_s": (t_frames[-1] - t_frames[0]) / 1e9,
    }
    (out / "cameras.json").write_text(json.dumps(cams, indent=1))
    print(json.dumps({k: v for k, v in cams.items() if k not in ("frames", "train", "test")}, indent=1))
    print(f"train {len(train)} / test {len(test)} frames")


if __name__ == "__main__":
    main()
