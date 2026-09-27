"""Mutation check for the pure-Python tests: break the code on purpose, confirm a test fails, restore.

Each mutation is a single exact string replacement. The original file bytes are restored in a
`finally` block and verified by SHA-256 afterwards. Writes MUTATION.md.

Usage:  python scripts/mutation_check.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTANTS = [
    ("d3gs/geometry.py", "R[..., 0, 1] = 2 * (x * y - w * z)", "R[..., 0, 1] = 2 * (x * y + w * z)",
     "quaternion->R: wrong sign in one off-diagonal term", "tests/test_geometry.py"),
    ("d3gs/geometry.py", "out[..., :3, 3] = -np.einsum", "out[..., :3, 3] = np.einsum",
     "SE(3) inverse: translation sign dropped", "tests/test_geometry.py"),
    ("d3gs/geometry.py", "K2[0, 2] = sx * (K[0, 2] + 0.5) - 0.5", "K2[0, 2] = sx * K[0, 2]",
     "resize_intrinsics: half-pixel centre convention dropped", "tests/test_geometry.py"),
    ("d3gs/geometry.py", "a = (t_query - t0) / (t1 - t0)", "a = (t1 - t_query) / (t1 - t0)",
     "pose interpolation: weight reversed", "tests/test_geometry.py"),
    ("d3gs/split.py", "    test = list(range(off, n_frames, every))\n",
     "    test = list(range(off, n_frames, every))\n    import random; test = random.sample(test, len(test))\n",
     "split: non-deterministic ordering (unseeded shuffle)", "tests/test_split.py"),
    ("d3gs/split.py", "train = [i for i in range(n_frames) if i not in test_set]", "train = list(range(n_frames))",
     "split: held-out frames leak into train", "tests/test_split.py"),
    ("d3gs/metrics.py", "return 10.0 * np.log10(data_range ** 2 / mse)", "return 20.0 * np.log10(data_range ** 2 / mse)",
     "PSNR: 20*log10 instead of 10*log10 on MSE", "tests/test_metrics.py"),
    ("d3gs/metrics.py", "C2 = (0.03 * data_range) ** 2", "C2 = (0.3 * data_range) ** 2",
     "SSIM: wrong K2 constant", "tests/test_metrics.py"),
    # --- added 2026-09-26 (second round): masks, masked metrics, lidar depth, leakage guard, web export ---
    ("d3gs/metrics.py", "return psnr(a[keep], b[keep], data_range)", "return psnr(a, b, data_range)",
     "masked PSNR: averages over all pixels instead of static ones", "tests/test_metrics.py"),
    ("d3gs/metrics.py", "kc = keep[r:keep.shape[0] - r, r:keep.shape[1] - r]", "kc = keep[:keep.shape[0] - 2 * r, :keep.shape[1] - 2 * r]",
     "masked SSIM: mask not aligned with the 'valid' SSIM map (off by the window radius)", "tests/test_metrics.py"),
    ("d3gs/dynamic.py", "    pc = _clip_edges_near(pc, near)\n", "",
     "cuboid mask: near-plane clipping removed (corners behind the camera project mirrored)", "tests/test_dynamic.py"),
    ("d3gs/dynamic.py", "l, w, h = (float(d) / 2 for d in dims)", "w, l, h = (float(d) / 2 for d in dims)",
     "cuboid corners: length and width swapped", "tests/test_dynamic.py"),
    ("d3gs/dynamic.py", "inside &= sgn * cr >= -1e-9", "inside |= sgn * cr >= -1e-9",
     "polygon fill: union of half-planes instead of intersection", "tests/test_dynamic.py"),
    ("d3gs/dynamic.py", "sp[k] = np.linalg.norm(centres[b, :2] - centres[a, :2]) / (t[b] - t[a])",
     "sp[k] = np.linalg.norm(centres[b, :2] - centres[a, :2]) / (2 * (t[b] - t[a]))",
     "track speed: wrong time base (half speed)", "tests/test_dynamic.py"),
    ("d3gs/dynamic.py", "a = (t_ns - t0) / (t1 - t0)", "a = (t1 - t_ns) / (t1 - t0)",
     "cuboid interpolation to camera time: weight reversed", "tests/test_dynamic.py"),
    ("d3gs/dynamic.py", "is_mov = sp > speed_thresh", "is_mov = sp >= 0",
     "moving mask: speed threshold ignored (parked cars masked as moving)", "tests/test_dynamic.py"),
    ("d3gs/lidar.py", "np.minimum.at(D, v * W + u, z)", "D[:] = 0; np.maximum.at(D, v * W + u, z)",
     "depth map: z-buffer keeps the farthest point", "tests/test_lidar.py"),
    ("d3gs/lidar.py", "u = np.round(uv[ok, 0]).astype(np.int64)", "u = np.floor(uv[ok, 0]).astype(np.int64)",
     "depth map: floor instead of round (pixel-centre convention broken in u)", "tests/test_lidar.py"),
    ("d3gs/lidar.py", "ok = (z > near) & (z < far) & np.isfinite(uv).all(1)", "ok = (z < far) & np.isfinite(uv).all(1)",
     "depth map: points behind the camera not rejected", "tests/test_lidar.py"),
    ("d3gs/lidar.py", "ev.add(int(np.argmin(np.abs(sweep_ts - frame_ts[i]))))", "ev.add(int(np.argmin(np.abs(sweep_ts - frame_ts[i]))) + 1)",
     "sweep split: reserves the sweep AFTER the held-out frame's nearest one", "tests/test_lidar.py"),
    ("d3gs/lidar.py", "    if overlap:", "    if False:",
     "leak guard disabled", "tests/test_lidar.py"),
    ("d3gs/scene.py", "if i not in self.train:", "if False:",
     "Scene.train_depth serves held-out frames' lidar", "tests/test_scene_guard.py"),
    ("d3gs/splat.py", 'buf["rot"] = np.clip(np.round(q * 128 + 128), 0, 255)', 'buf["rot"] = np.clip(np.round(q[:, [1, 2, 3, 0]] * 128 + 128), 0, 255)',
     ".splat: quaternion written as (x, y, z, w) instead of (w, x, y, z)", "tests/test_splat.py"),
    ("d3gs/splat.py", "idx = np.where(opacity >= min_opacity)[0]", "idx = np.arange(len(opacity))",
     "web pruning: opacity floor ignored", "tests/test_splat.py"),
]


def run_tests(test_file: str) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", test_file], cwd=ROOT,
                       capture_output=True, text=True)
    last = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1] if r.stdout.strip() else r.stderr[-200:]
    return r.returncode, last


def main() -> None:
    rows = []
    base_rc, base_msg = run_tests("tests")
    if base_rc != 0:
        raise SystemExit(f"baseline tests must pass first: {base_msg}")
    for rel, old, new, desc, test in MUTANTS:
        path = ROOT / rel
        orig = path.read_bytes()
        h0 = hashlib.sha256(orig).hexdigest()
        text = orig.decode("utf-8")
        if text.count(old) != 1:
            raise SystemExit(f"mutation anchor not unique/absent in {rel}: {old!r}")
        try:
            path.write_bytes(text.replace(old, new).encode("utf-8"))
            rc, msg = run_tests(test)
        finally:
            path.write_bytes(orig)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == h0, f"{rel} not restored!"
        caught = rc != 0
        rows.append((rel, desc, test, "caught" if caught else "SURVIVED", msg))
        print(f"[{'caught' if caught else 'SURVIVED'}] {rel}: {desc} -> {msg}")
    after_rc, after_msg = run_tests("tests")

    lines = ["# Mutation check", "",
             "Generated by `python scripts/mutation_check.py` (2026-09-27). Each row deliberately breaks one line of the",
             "pure-Python code, runs the relevant test file, and restores the file (SHA-256 verified).", "",
             f"Baseline before mutations: `{base_msg}`", "",
             "| # | file | mutation | test file | result | pytest summary |", "|---|---|---|---|---|---|"]
    for k, (rel, desc, test, res, msg) in enumerate(rows, 1):
        lines.append(f"| {k} | `{rel}` | {desc} | `{test}` | **{res}** | `{msg}` |")
    n_caught = sum(r[3] == "caught" for r in rows)
    lines += ["", f"Caught {n_caught}/{len(rows)} mutants. After restoring: `{after_msg}`", "",
              "## History", "",
              "First run (2026-09-26): mutant 4 (pose interpolation weight reversed, `a -> 1 - a`) **survived** —",
              "`test_interpolate_pose_exact_midpoint_and_range` only queried the midpoint, where `a = 1 - a = 0.5`.",
              "The test now also queries t = 25 of [0, 100] (expects translation 2.5 m and a 22.5 deg yaw), and the",
              "table above is the re-run after that fix. The other 7 mutants were caught on the first run.", "",
              "Second round (2026-09-27, mutants 9-24 for masks, masked metrics, lidar depth, the leakage guard and the",
              "web export): mutant 13 (polygon fill: union of half-planes instead of intersection) **survived** the first",
              "run - every cuboid test projected to an axis-aligned rectangle, which equals its own bounding box, so filling",
              "the whole bounding box looked correct. `test_rotated_box_fills_a_diamond_not_its_bounding_box` (a plate",
              "rotated 45 deg about the optical axis) was added, and the table above is the re-run after that fix. The",
              "other 15 new mutants were caught on the first run.", ""]
    (ROOT / "MUTATION.md").write_text("\n".join(lines), encoding="utf-8")
    if n_caught != len(rows) or after_rc != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
