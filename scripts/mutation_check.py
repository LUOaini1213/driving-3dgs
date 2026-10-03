"""Mutation-test the test suite: break the code on purpose, confirm the tests notice.

Each mutation runs in a temporary copy, leaving the working tree untouched. Clean tests must
pass; only actual test failures kill a mutation. Syntax, collection, fixture, timeout and runner
errors are inconclusive and fail this check. JUnit reports and logs are retained for inspection.

    python scripts/mutation_check.py            # all mutations, writes MUTATION.md
    python scripts/mutation_check.py --only M3  # one mutation, no MUTATION.md
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (id, file, original text, mutated text, what the bug means, tests to run)
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
     "    test = list(range(off, n_frames, every))\n    test.reverse()\n",
     "split: held-out frame ordering reversed", "tests/test_split.py"),
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

# Regressions found during the October review.
MUTANTS.extend([('d3gs/lidar.py',
  '    if not np.isfinite(pred[m]).all():',
  '    m &= np.isfinite(pred)\n    if not np.isfinite(pred[m]).all():',
  'depth: invalid predictions removed from the fixed ground-truth cohort',
  'tests/test_metric_integrity.py::test_missing_prediction_cannot_improve_a_fixed_depth_cohort'),
 ('d3gs/metrics.py',
  '    a, b = _pair(a, b, data_range, image=True, window=11)\n    keep = _mask',
  '    a, b = np.asarray(a), np.asarray(b)\n    keep = _mask',
  'masked SSIM: mismatched channels accepted',
  'tests/test_metric_integrity.py::test_masked_ssim_does_not_ignore_extra_error_channels'),
 ('d3gs/provenance.py',
  'if proof.get("checkpoint_sha256") != checkpoint_hash:',
  'if False:',
  'run identity: changed checkpoint accepted',
  'tests/test_scene_integrity.py::test_identity_binds_training_eval_init_split_and_checkpoint'),
 ('d3gs/web_asset.py',
  "if expected.get('mode') != mode or file_record(directory / name, text=mode == 'utf8-lf') != expected:",
  'if False:',
  'web asset: same-size content substitution accepted',
  'tests/test_web_evidence.py::test_manifest_rejects_same_size_changes_and_missing_files'),
 ('d3gs/report_keys.py',
  'return f"{int(value):+d}" if value.is_integer() else ("+" if value >= 0 else "") + repr(value)',
  'return f"{value:+.0f}"',
  'off-path report: sub-metre offsets collide',
  'tests/test_scene_integrity.py::test_offpath_actual_main_preserves_decimal_offsets'),
 ('d3gs/scene.py',
  'if not m["train"] or set(m["train"]) & set(m["test"]):',
  'if not m["train"]:',
  'scene: training and held-out cohort overlap accepted',
  'tests/test_scene_integrity.py::test_scene_rejects_invalid_metadata'),
 ('d3gs/provenance.py',
  'if proof.get("training") != recipe:',
  'if False:',
  'run identity: changed training recipe accepted',
  'tests/test_scene_integrity.py::test_recorded_training_metadata_is_bound'),
 ('d3gs/provenance.py',
  'if proof.get("scene") != current:',
  'if False:',
  'run identity: a different prepared scene accepted',
  'tests/test_scene_integrity.py::test_identity_copied_scene_accepted_but_changes_rejected'),
 ('scripts/train.py',
  'return (prediction[valid] - ground_truth[valid]).abs().mean() if valid.any() else prediction.sum() * 0',
  'return (prediction[valid] - ground_truth[valid]).abs().mean()',
  'training depth: empty support introduces NaN loss',
  'tests/test_scene_integrity.py::test_sparse_depth_empty_support_and_nonfinite_are_explicit')])

MUTATIONS = [(f"M{i}", *m) for i, m in enumerate(MUTANTS, 1)]
EQUIVALENT = ("M0", "d3gs/geometry.py", "Pose math: quaternions, SE(3), camera intrinsics resizing.",
              "Pose math: rotations, SE(3), camera intrinsics resizing.",
              "equivalent docstring edit - must survive", "tests/test_geometry.py")

def classify_report(returncode: int, report: Path) -> dict:
    """A pytest exit code alone cannot distinguish a killed mutant from a broken runner."""
    result = {"outcome": "error", "returncode": returncode,
              "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    try:
        root = ET.parse(report).getroot()
    except (OSError, ET.ParseError) as exc:
        return {**result, "reason": f"missing or invalid JUnit report: {type(exc).__name__}"}
    if root.tag not in ("testsuites", "testsuite"):
        return {**result, "reason": "unexpected JUnit root"}
    for case in root.iter("testcase"):
        if case.find("error") is not None:
            result["errors"] += 1
        elif case.find("failure") is not None:
            result["failed"] += 1
        elif case.find("skipped") is not None:
            result["skipped"] += 1
        else:
            result["passed"] += 1
    if result["errors"]:
        result["reason"] = "collection, setup or teardown error"
    elif returncode == 0 and result["passed"] > 0 and result["failed"] == 0:
        result["outcome"] = "passed"
    elif returncode == 1 and result["failed"] > 0:
        result["outcome"] = "failed"
    else:
        result["reason"] = "no executed tests or inconsistent pytest result"
    return result


def run_tests(tests: str, cwd: Path, evidence: Path, timeout: float) -> dict:
    evidence.mkdir(parents=True, exist_ok=False)
    report = evidence / "junit.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
           f"--junitxml={report}", *tests.split()]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(cwd)}
    # Host add-ons must not change the meaning of this project's test run.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    t0 = time.monotonic()
    try:
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                 else {"start_new_session": True})
        with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, **group) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                result = classify_report(process.returncode, report)
            except subprocess.TimeoutExpired:
                # A Windows venv launcher has a child Python process. Killing only the
                # launcher leaves pytest (and its children) alive in the temporary copy.
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=15)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                stdout, stderr = process.communicate(timeout=15)
                result = {"outcome": "error", "reason": "timeout", "returncode": None}
        (evidence / "stdout.log").write_bytes(stdout)
        (evidence / "stderr.log").write_bytes(stderr)
    except OSError as exc:
        result = {"outcome": "error", "reason": f"runner error: {type(exc).__name__}",
                  "returncode": None}
    result["seconds"] = round(time.monotonic() - t0, 3)
    (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def copy_snapshot(source: Path, target: Path, output_dir: Path) -> None:
    def ignore(directory, names):
        return [name for name in names if name in
                {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules",
                 "mutation-artifacts", ".mutation_backup"}
                or name.endswith(".pyc") or (Path(directory) / name).resolve() == output_dir.resolve()
                or (Path(directory) == source and name in {"data", "work", "runs", "export", "env"})]
    shutil.copytree(source, target, ignore=ignore)


def check_mutations(source: Path, mutations: list, equivalent: tuple,
                    output_dir: Path, timeout: float = 120) -> dict:
    """Run a frozen source copy and a fresh copy for every variant, even on interruption."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Separate run directories prevent stale XML from converting a runner error into a kill.
    evidence = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir)).resolve()
    summary = {"schema_version": 2, "status": "failed", "equivalent": None,
               "results": [], "source_sha256": {}}
    with tempfile.TemporaryDirectory(prefix="driving-mutations-") as tmp:
        snapshot = Path(tmp) / "snapshot"
        copy_snapshot(source, snapshot, output_dir)
        for index, (mid, rel, old, new, desc, tests) in enumerate([equivalent] + mutations):
            entry = {"id": mid, "file": rel, "description": desc, "tests": tests,
                     "outcome": "error"}
            original = (snapshot / rel).read_bytes()
            summary["source_sha256"][rel] = hashlib.sha256(original).hexdigest()
            # Git may check out CRLF on Windows; match logical source lines.
            code = original.decode("utf-8").replace("\r\n", "\n")
            if code.count(old) != 1:
                entry["reason"] = f"target found {code.count(old)} times (must be exactly 1)"
            else:
                variant = Path(tmp) / f"variant-{index}"
                copy_snapshot(snapshot, variant, output_dir)
                baseline = run_tests(tests, variant, evidence / f"{mid}-baseline", timeout)
                entry["baseline"] = baseline
                if baseline["outcome"] != "passed":
                    entry["reason"] = "clean baseline did not pass"
                else:
                    changed = code.replace(old, new)
                    try:
                        if Path(rel).suffix == ".py":
                            compile(changed, rel, "exec")
                    except (SyntaxError, ValueError) as exc:
                        entry["reason"] = f"invalid mutation: {type(exc).__name__}"
                    else:
                        (variant / rel).write_bytes(changed.encode("utf-8"))
                        trial = run_tests(tests, variant, evidence / f"{mid}-mutation", timeout)
                        entry["mutation"] = trial
                        entry["outcome"] = {"passed": "survived", "failed": "killed",
                                            "error": "error"}[trial["outcome"]]
                        if trial["outcome"] == "error":
                            entry["reason"] = trial["reason"]
            print(f"{mid} {entry['outcome'].upper()}  {desc}", flush=True)
            if index == 0:
                summary["equivalent"] = entry
                if entry["outcome"] != "survived":
                    break
            else:
                summary["results"].append(entry)
        summary["killed"] = sum(x["outcome"] == "killed" for x in summary["results"])
        summary["requested"] = len(mutations)
        if (summary["equivalent"]["outcome"] == "survived" and mutations
                and summary["killed"] == len(mutations)):
            summary["status"] = "passed"
    (evidence / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{summary['killed']}/{len(mutations)} killed; evidence: {evidence}", flush=True)
    return summary


HISTORY = """

First run (2026-09-26): mutant 4 (pose interpolation weight reversed, `a -> 1 - a`) **survived** —
`test_interpolate_pose_exact_midpoint_and_range` only queried the midpoint, where `a = 1 - a = 0.5`.
The test now also queries t = 25 of [0, 100] (expects translation 2.5 m and a 22.5 deg yaw), and the
table above is the re-run after that fix. The other 7 mutants were caught on the first run.

Second round (2026-09-27, mutants 9-24 for masks, masked metrics, lidar depth, the leakage guard and the
web export): mutant 13 (polygon fill: union of half-planes instead of intersection) **survived** the first
run - every cuboid test projected to an axis-aligned rectangle, which equals its own bounding box, so filling
the whole bounding box looked correct. `test_rotated_box_fills_a_diamond_not_its_bounding_box` (a plate
rotated 45 deg about the optical axis) was added, and the table above is the re-run after that fix. The
other 15 new mutants were caught on the first run.
"""

def report_markdown(summary: dict) -> str:
    lines = ["# Mutation check", "",
             "Every variant runs in a temporary source copy after a passing clean baseline.",
             "Only executed test-call failures count as kills. Syntax, collection, fixture,",
             "runner, empty-test and timeout errors fail the check. M0 changes a docstring",
             "and must survive. Working files and historical results are never mutated.", "",
             f"**{summary['killed']}/{summary['requested']} killed; check {summary['status']}.**", "",
             "JUnit, stdout, stderr, source hashes and the run manifest are saved under",
             "`mutation-artifacts/run-*/`; CI uploads these evidence files.", "",
             "| id | file | injected bug | tests | result |", "|---|---|---|---|---|"]
    for entry in [summary["equivalent"]] + summary["results"]:
        outcome = entry["outcome"]
        if entry.get("reason"):
            outcome += ": " + entry["reason"]
        lines.append(f"| {entry['id']} | `{entry['file']}` | {entry['description']} | "
                     f"`{entry['tests']}` | {outcome} |")
    lines += ["", "## Historical checks", "",
              "The September checks below used the original runner, which accepted any nonzero",
              "pytest exit. The current table above uses stricter failure classification.", HISTORY]
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=[m[0] for m in MUTATIONS])
    ap.add_argument("--output-dir", type=Path, default=ROOT / "mutation-artifacts")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--no-report", action="store_true", help="do not update MUTATION.md")
    args = ap.parse_args()
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    muts = [m for m in MUTATIONS if args.only in (None, m[0])]
    summary = check_mutations(ROOT, muts, EQUIVALENT, args.output_dir.resolve(), args.timeout)
    if args.only is None and not args.no_report:
        (ROOT / "MUTATION.md").write_text(report_markdown(summary), encoding="utf-8", newline="\n")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
