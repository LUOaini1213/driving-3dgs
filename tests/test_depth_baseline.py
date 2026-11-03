"""Lidar-only depth baseline: sweep choice never leaks, filling, pooling, and the CPU script end to end."""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from d3gs.geometry import intrinsics_matrix, se3_inverse
from d3gs.lidar import (LidarLeakError, baseline_sweep, depth_errors, depth_map, fill_nearest,
                        pooled_depth_errors)

ROOT = Path(__file__).resolve().parents[1]
MS = 1_000_000


def script():
    spec = importlib.util.spec_from_file_location("depth_baseline", ROOT / "scripts" / "depth_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- sweep choice -------------------------------------------------------------------------

def test_baseline_sweep_skips_the_nearer_evaluation_sweep():
    sweeps = [0, 98 * MS, 200 * MS]
    # frame at 100 ms: the evaluation sweep (98 ms) is nearest, the baseline must take a training sweep
    assert baseline_sweep(sweeps, 100 * MS, [0, 200 * MS], [98 * MS]) in (0, 2)
    assert baseline_sweep(sweeps, 150 * MS, [0, 200 * MS], [98 * MS]) == 2


def test_baseline_sweep_refuses_an_evaluation_sweep_offered_as_training():
    with pytest.raises(LidarLeakError):
        baseline_sweep([0, 98 * MS, 200 * MS], 100 * MS, [0, 98 * MS], [98 * MS])


def test_baseline_sweep_needs_a_known_training_sweep():
    with pytest.raises(ValueError, match="no training sweep"):
        baseline_sweep([0, 98 * MS], 100 * MS, [7], [98 * MS])


# --- filling and pooling ------------------------------------------------------------------

def test_fill_nearest_copies_the_closest_return_and_keeps_returns():
    D = np.zeros((3, 5), np.float32)
    D[1, 0], D[1, 4] = 5.0, 9.0
    F = fill_nearest(D)
    assert F[1, 0] == 5.0 and F[1, 4] == 9.0
    assert (F[:, :2] == 5.0).all() and (F[:, 3:] == 9.0).all()
    assert F.min() > 0
    full = np.full((2, 2), 3.0)
    assert np.array_equal(fill_nearest(full), full)
    with pytest.raises(ValueError):
        fill_nearest(np.zeros((2, 2)))


def test_pooled_errors_match_depth_errors_on_concatenated_pixels():
    rng = np.random.default_rng(0)
    gts = [np.where(rng.random((6, 7)) < 0.5, rng.uniform(1, 70, (6, 7)), 0.0) for _ in range(3)]
    preds = [g + rng.normal(0, 0.4, g.shape) for g in gts]
    keeps = [rng.random((6, 7)) < 0.8 for _ in range(3)]
    out = pooled_depth_errors(preds, gts, keeps)
    v = [(g > 0) & k for g, k in zip(gts, keeps)]
    ref = depth_errors(np.concatenate([p[m] for p, m in zip(preds, v)]), np.concatenate([g[m] for g, m in zip(gts, v)]))
    assert out["n"] == ref["n"] and out["median_abs_m"] == round(ref["median_abs_m"], 4)
    assert sum(b["n"] for b in out["by_range"].values()) == ref["n"]


# --- the script on a synthetic AV2-like log -----------------------------------------------

W, H = 40, 30
K = intrinsics_matrix(20.0, 20.0, 19.5, 14.5)


def wall(z):
    x, y = np.meshgrid(np.linspace(-9, 9, 60), np.linspace(-6.5, 6.5, 45))
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], 1)


def make_scene(tmp_path, train_wall_z=10.2):
    log, work = tmp_path / "log", tmp_path / "work"
    (log / "sensors" / "lidar").mkdir(parents=True)
    ts = np.arange(0, 201, 10, dtype=np.int64) * MS
    pd.DataFrame({"timestamp_ns": ts, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                  "tx_m": 0.0, "ty_m": 0.0, "tz_m": 0.0}).to_feather(log / "city_SE3_egovehicle.feather")
    sweeps = {0: wall(train_wall_z), 98 * MS: wall(10.0), 200 * MS: wall(train_wall_z)}
    for t, xyz in sweeps.items():
        pd.DataFrame(xyz, columns=["x", "y", "z"]).to_feather(log / "sensors" / "lidar" / f"{t}.feather")
    frames = [{"idx": i, "timestamp_ns": i * 50 * MS, "image": f"images/{i:04d}.png", "c2w": np.eye(4).tolist()}
              for i in range(4)]
    (work / "eval_depth").mkdir(parents=True)
    (work / "masks" / "moving").mkdir(parents=True)
    keep = wall(10.0)
    keep = keep[np.linalg.norm(keep[:, :2], axis=1) > 3.0]       # prepare.py drops ego-near returns
    np.save(work / "eval_depth" / "0002.npy", depth_map(K, se3_inverse(np.eye(4)), keep, W, H))
    ok, png = cv2.imencode(".png", np.zeros((H, W), np.uint8))
    (work / "masks" / "moving" / "0002.png").write_bytes(png.tobytes())
    (work / "cameras.json").write_text(json.dumps({
        "log_dir": str(log), "width": W, "height": H, "K": K.tolist(), "city_origin": [0, 0, 0],
        "frames": frames, "train": [0, 1, 3], "test": [2]}))
    (work / "lidar_split.json").write_text(json.dumps({
        "policy": "strict", "sweep_ts": sorted(sweeps), "eval_sweep_ts": [98 * MS],
        "init_sweep_ts": [0], "depth_sweep_ts": [200 * MS]}))
    return log, work


def test_script_scores_the_training_sweep_on_the_evaluation_cohort(tmp_path):
    log, work = make_scene(tmp_path)
    out = tmp_path / "baseline.json"
    res = script().main(["--work", str(work), "--log-dir", str(log), "--results", str(out)])
    b = res["lidar_nearest_train_sweep"]
    gt = np.load(work / "eval_depth" / "0002.npy")
    assert b["all"]["n"] == int((gt > 0).sum()) == b["static"]["n"]
    # the training wall sits 0.2 m behind the evaluation wall; nothing else differs
    assert b["all"]["median_abs_m"] == pytest.approx(0.2, abs=1e-3) and b["all"]["within_tol"] == 1.0
    assert res["per_frame"][0]["sweep_ts"] in (0, 200 * MS)
    assert json.loads(out.read_text(encoding="utf-8"))["lidar_nearest_train_sweep"] == b
    assert str(tmp_path) not in out.read_text(encoding="utf-8")       # no local paths in the result


def test_script_rejects_a_stored_eval_depth_it_cannot_reproduce(tmp_path):
    log, work = make_scene(tmp_path)
    d = np.load(work / "eval_depth" / "0002.npy")
    d[d > 0] += 0.01
    np.save(work / "eval_depth" / "0002.npy", d)
    with pytest.raises(ValueError, match="does not reproduce"):
        script().main(["--work", str(work), "--log-dir", str(log)])


def test_script_rejects_a_split_whose_training_sweeps_include_an_evaluation_sweep(tmp_path):
    log, work = make_scene(tmp_path)
    split = json.loads((work / "lidar_split.json").read_text())
    split["depth_sweep_ts"].append(98 * MS)
    (work / "lidar_split.json").write_text(json.dumps(split))
    with pytest.raises(LidarLeakError):
        script().main(["--work", str(work), "--log-dir", str(log)])


def test_script_rejects_a_cohort_that_differs_from_the_compared_run(tmp_path):
    log, work = make_scene(tmp_path)
    ref = tmp_path / "run.json"
    ref.write_text(json.dumps({"heldout_depth": {"3dgs": {"all": {"n": 1}, "static": {"n": 1}}}}))
    with pytest.raises(ValueError, match="cohort mismatch"):
        script().main(["--work", str(work), "--log-dir", str(log), "--compare", str(ref)])


# --- the committed result ------------------------------------------------------------------

def test_committed_baseline_uses_the_same_cohort_as_every_strict_run():
    b = json.loads((ROOT / "results" / "depth_baselines.json").read_text(encoding="utf-8"))
    base = b["lidar_nearest_train_sweep"]
    assert b["eval_depth_reproduced"] is True
    strict = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "results" / "runs").glob("strict_*.json"))]
    assert strict
    for r in strict:
        for region in ("all", "static"):
            assert r["heldout_depth"]["3dgs"][region]["n"] == base[region]["n"]
    assert len(b["per_frame"]) == strict[0]["n_test"] == len(b["eval_depth_sha256"])
    assert all(abs(f["dt_ms"]) < 150 for f in b["per_frame"])
    assert sum(f["eval_px"] for f in b["per_frame"]) == base["all"]["n"]
