"""Lidar-only depth baseline for the held-out frames (CPU only, no torch, no GPU).

The headline depth result (`strict_depth_7k`) is scored against each held-out frame's reserved
evaluation sweep. This script asks how much of that is already explained by lidar the model was
allowed to see: for every held-out frame it takes the time-nearest TRAINING sweep (a sweep used for
the strict initialisation or for depth supervision, never an evaluation sweep), re-projects it into
the held-out camera with the same `depth_map` as prepare.py, fills empty pixels with the nearest
projected return (image space), and scores it with the same pooled `depth_errors` cohort as
evaluate.py (all lidar pixels / static pixels, and the same distance bins).

Self-checks before anything is written:
  * the re-projected evaluation sweep reproduces every stored eval_depth/*.npy bit for bit
    (same projection code path as the scored predictions);
  * the chosen sweep is never an evaluation sweep (`assert_no_lidar_leak`);
  * with --compare, the pooled cohort size equals the committed run's heldout_depth cohort.

Usage (data from scripts/download_av2.py, prepared dir from prepare.py --strict-lidar --masks):
  python scripts/depth_baseline.py --work "$DATA_ROOT/work/strict" \
      --log-dir "$DATA_ROOT/data/15ec0778-826e-3ed7-9775-54fbf66997f4" \
      --compare results/runs/strict_7k.json --results results/depth_baselines.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d3gs.av2io import list_sweeps, load_poses, load_sweep_xyz  # noqa: E402
from d3gs.geometry import se3_inverse, transform_points  # noqa: E402
from d3gs.image_io import read_image  # noqa: E402
from d3gs.lidar import (assert_no_lidar_leak, baseline_sweep, depth_map, fill_nearest,  # noqa: E402
                        nearest_sweep, pooled_depth_errors)
from d3gs.provenance import sha256  # noqa: E402

NAME = "lidar_nearest_train_sweep"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="prepared dir from prepare.py --strict-lidar --masks")
    ap.add_argument("--log-dir", default=None, help="raw AV2 log dir (default: log_dir recorded in cameras.json)")
    ap.add_argument("--compare", default=None, help="committed run JSON whose heldout_depth cohort must match")
    ap.add_argument("--results", default=None, help="write the baseline JSON here")
    args = ap.parse_args(argv)

    work = Path(args.work)
    meta = json.loads((work / "cameras.json").read_text(encoding="utf-8"))
    split = json.loads((work / "lidar_split.json").read_text(encoding="utf-8"))
    if split.get("policy") != "strict":
        raise ValueError("the baseline needs a strict prepared dir (evaluation sweeps reserved)")
    log_dir = Path(args.log_dir or meta["log_dir"])
    poses = load_poses(log_dir)
    sweeps = list_sweeps(log_dir)
    sweep_ts = np.array([t for t, _ in sweeps], dtype=np.int64)
    if sweep_ts.tolist() != split["sweep_ts"]:
        raise ValueError("raw lidar sweeps do not match the prepared lidar split")
    eval_ts = split["eval_sweep_ts"]
    training_ts = sorted(set(split["init_sweep_ts"]) | set(split["depth_sweep_ts"]))
    assert_no_lidar_leak(training_ts, eval_ts)
    eval_idx = [int(np.flatnonzero(sweep_ts == t)[0]) for t in eval_ts]

    K = np.asarray(meta["K"], dtype=np.float64)
    W, H = meta["width"], meta["height"]
    origin = np.asarray(meta["city_origin"], dtype=np.float64)
    c2w = np.array([f["c2w"] for f in meta["frames"]], dtype=np.float64)
    w2c = se3_inverse(c2w)
    t_frames = np.array([f["timestamp_ns"] for f in meta["frames"]], dtype=np.int64)

    def sweep_world(k: int) -> np.ndarray:   # identical to prepare.py
        xyz = load_sweep_xyz(sweeps[k][1])
        r = np.linalg.norm(xyz[:, :2], axis=1)
        return transform_points(poses.city_SE3_ego(int(sweep_ts[k])), xyz[(r > 3.0) & (r < 80.0)]) - origin

    preds, gts, keeps, frames = [], [], [], []
    for i in meta["test"]:
        gt = np.load(work / "eval_depth" / f"{i:04d}.npy", allow_pickle=False)
        ke = nearest_sweep(sweep_ts, t_frames[i], eval_idx)
        if not np.array_equal(depth_map(K, w2c[i], sweep_world(ke), W, H), gt):
            raise ValueError(f"re-projected evaluation sweep does not reproduce eval_depth/{i:04d}.npy")
        k = baseline_sweep(sweep_ts, int(t_frames[i]), training_ts, eval_ts)
        sparse = depth_map(K, w2c[i], sweep_world(k), W, H)
        pred = fill_nearest(sparse)
        mask = read_image(work / "masks" / "moving" / f"{i:04d}.png", cv2.IMREAD_GRAYSCALE) > 0
        on = gt > 0
        frames.append({"frame": i, "sweep_ts": int(sweep_ts[k]), "dt_ms": round((int(sweep_ts[k]) - int(t_frames[i])) / 1e6, 1),
                       "eval_px": int(on.sum()), "direct_hit_px": int((on & (sparse > 0)).sum())})
        preds.append(pred)
        gts.append(gt)
        keeps.append(~mask)

    res = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "log_id": log_dir.name,
        "method": {NAME: "time-nearest training sweep (strict init or depth-supervision sweep, never an evaluation "
                         "sweep) re-projected into the held-out camera with prepare.py's depth_map; empty pixels "
                         "take the nearest projected return in image space (scipy distance_transform_edt)"},
        "gt": "held-out frame's reserved time-nearest lidar sweep (eval_depth/*.npy), z-depth",
        "pooling": f"all lidar pixels of the {len(gts)} held-out frames pooled; same cohort and bins as evaluate.py",
        "tol_m": 0.5, "range_m": [0.5, 80.0],
        "eval_depth_reproduced": True,
        "eval_depth_sha256": {f"eval_depth/{i:04d}.npy": sha256(work / "eval_depth" / f"{i:04d}.npy") for i in meta["test"]},
        NAME: {"all": pooled_depth_errors(preds, gts), "static": pooled_depth_errors(preds, gts, keeps)},
        "per_frame": frames,
    }
    b = res[NAME]
    b["direct_hit_frac_all"] = round(sum(f["direct_hit_px"] for f in frames) / sum(f["eval_px"] for f in frames), 4)
    b["abs_dt_ms_median"] = float(np.median([abs(f["dt_ms"]) for f in frames]))
    if args.compare:
        ref = json.loads(Path(args.compare).read_text(encoding="utf-8"))["heldout_depth"]["3dgs"]
        for region in ("all", "static"):
            if ref[region]["n"] != b[region]["n"]:
                raise ValueError(f"cohort mismatch ({region}): {b[region]['n']} vs {ref[region]['n']} in {args.compare}")
        res["cohort_matches"] = Path(args.compare).as_posix()
    print(json.dumps({r: {k: b[r][k] for k in ("n", "median_abs_m", "within_tol")} for r in ("all", "static")}
                     | {"direct_hit_frac_all": b["direct_hit_frac_all"], "abs_dt_ms_median": b["abs_dt_ms_median"]}, indent=1))
    if args.results:
        Path(args.results).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
