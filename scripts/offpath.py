"""Off-path views: render held-out poses shifted sideways (no ground-truth image exists there).

For each held-out frame the camera is moved along its own x axis (right = +) by each offset.
Two things are measured, both without a ground-truth image:
  * coverage: fraction of pixels with accumulated alpha < 0.5 (holes where nothing was reconstructed);
  * geometry: rendered expected depth vs the held-out frame's reserved lidar sweep projected into the
    SHIFTED camera (the sweep is 3D, so it can be re-projected; it was never used in training).
Writes a JSON of means per offset, and optionally a small comparison figure (< 300 KB).

Usage:
  python scripts/offpath.py --work D:/driving-3dgs/work/strict --runs D:/driving-3dgs/runs/strict_7k D:/driving-3dgs/runs/strict_depth_7k \
      --results results/offpath.json --fig docs/img/offpath.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d3gs.av2io import load_poses, load_sweep_xyz  # noqa: E402
from d3gs.geometry import se3_inverse, transform_points  # noqa: E402
from d3gs.lidar import depth_errors, depth_map, nearest_sweep  # noqa: E402
from d3gs.scene import Scene, render  # noqa: E402


def shifted_c2w(c2w: np.ndarray, dx: float) -> np.ndarray:
    """Move a camera by dx metres along its own x axis (image right), orientation unchanged."""
    out = np.array(c2w, dtype=np.float64).copy()
    out[:3, 3] = out[:3, 3] + dx * out[:3, 0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="strict work dir (has lidar_split.json)")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--offsets", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--fig-frames", type=int, nargs="+", default=[36, 116])
    ap.add_argument("--fig-scale", type=float, default=0.5)
    args = ap.parse_args()

    sc = Scene(args.work)
    split = sc.lidar_split
    log_dir = Path(sc.meta["log_dir"])
    poses = load_poses(log_dir)
    origin = np.array(sc.meta["city_origin"])
    sweep_ts = np.array(split["sweep_ts"], dtype=np.int64)
    eval_idx = [int(np.where(sweep_ts == t)[0][0]) for t in split["eval_sweep_ts"]]
    sweep_files = {int(p.stem): p for p in (log_dir / "sensors" / "lidar").glob("*.feather")}

    def eval_points(i: int) -> np.ndarray:
        k = nearest_sweep(sweep_ts, sc.meta["frames"][i]["timestamp_ns"], eval_idx)
        xyz = load_sweep_xyz(sweep_files[int(sweep_ts[k])])
        r = np.linalg.norm(xyz[:, :2], axis=1)
        return transform_points(poses.city_SE3_ego(int(sweep_ts[k])), xyz[(r > 3.0) & (r < 80.0)]) - origin

    pts = {i: eval_points(i) for i in sc.test}
    K = sc.K.cuda()
    Knp = sc.K.numpy().astype(np.float64)
    out = {"offsets_m": args.offsets, "frames": sc.test, "runs": {}}
    fig_tiles: dict[tuple, np.ndarray] = {}
    for run_dir in map(Path, args.runs):
        stats = json.loads((run_dir / "train_stats.json").read_text())
        params = {k: v.cuda() for k, v in torch.load(run_dir / "ckpt.pt").items()}
        per = {}
        for dx in args.offsets:
            holes, P, G = [], [], []
            for i in sc.test:
                c2w = shifted_c2w(sc.c2w[i].numpy(), dx)
                vm = torch.from_numpy(se3_inverse(c2w)).float().cuda()
                with torch.no_grad():
                    img, alpha, _, dep = render(params, vm, K, sc.W, sc.H, stats["sh_degree"], with_depth=True)
                a = alpha[..., 0].cpu().numpy()
                holes.append(float((a < 0.5).mean()))
                gd = depth_map(Knp, se3_inverse(c2w), pts[i], sc.W, sc.H)
                v = gd > 0
                P.append(dep.cpu().numpy()[v])
                G.append(gd[v])
                if args.fig and i in args.fig_frames:
                    fig_tiles[(run_dir.name, i, dx)] = (img.cpu().numpy() * 255).round().astype(np.uint8)
            e = depth_errors(np.concatenate(P), np.concatenate(G))
            per[f"{dx:+.0f}"] = {"hole_frac_mean": round(float(np.mean(holes)), 4),
                                 "depth_median_abs_m": round(e["median_abs_m"], 4),
                                 "depth_within_0_5m": round(e["within_tol"], 4), "depth_n": e["n"]}
            print(run_dir.name, dx, per[f"{dx:+.0f}"])
        out["runs"][run_dir.name] = {"mask": stats.get("mask", "none"), "depth_lambda": stats.get("depth_lambda", 0.0),
                                     "steps": stats["steps"], "per_offset": per}
        del params
        torch.cuda.empty_cache()
    Path(args.results).write_text(json.dumps(out, indent=1), encoding="utf-8")

    if args.fig:
        s = args.fig_scale
        w, h = int(sc.W * s), int(sc.H * s)
        rows = []
        for run_dir in map(Path, args.runs):
            for i in args.fig_frames:
                tiles = []
                for dx in args.offsets:
                    t = cv2.resize(fig_tiles[(run_dir.name, i, dx)], (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
                    cv2.rectangle(t, (0, 0), (w, 16), (0, 0, 0), -1)
                    lab = f"{run_dir.name} f{i} {dx:+.0f} m" if dx == args.offsets[0] else f"{dx:+.0f} m"
                    cv2.putText(t, lab, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(t)
                rows.append(np.hstack(tiles))
        fig = np.vstack(rows)
        q = 85
        while True:
            ok, enc = cv2.imencode(".jpg", fig, [cv2.IMWRITE_JPEG_QUALITY, q])
            if len(enc) < 290_000 or q <= 40:
                break
            q -= 5
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        Path(args.fig).write_bytes(enc.tobytes())
        print("figure", args.fig, len(enc), "bytes, quality", q)


if __name__ == "__main__":
    main()
