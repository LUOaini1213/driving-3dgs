# Mutation check

Every variant runs in a temporary source copy after a passing clean baseline.
Only executed test-call failures count as kills. Syntax, collection, fixture,
runner, empty-test and timeout errors fail the check. M0 changes a docstring
and must survive. Working files and historical results are never mutated.

**33/33 killed; check passed.**

JUnit, stdout, stderr, source hashes and the run manifest are saved under
`mutation-artifacts/run-*/`; CI uploads these evidence files.

| id | file | injected bug | tests | result |
|---|---|---|---|---|
| M0 | `d3gs/geometry.py` | equivalent docstring edit - must survive | `tests/test_geometry.py` | survived |
| M1 | `d3gs/geometry.py` | quaternion->R: wrong sign in one off-diagonal term | `tests/test_geometry.py` | killed |
| M2 | `d3gs/geometry.py` | SE(3) inverse: translation sign dropped | `tests/test_geometry.py` | killed |
| M3 | `d3gs/geometry.py` | resize_intrinsics: half-pixel centre convention dropped | `tests/test_geometry.py` | killed |
| M4 | `d3gs/geometry.py` | pose interpolation: weight reversed | `tests/test_geometry.py` | killed |
| M5 | `d3gs/split.py` | split: held-out frame ordering reversed | `tests/test_split.py` | killed |
| M6 | `d3gs/split.py` | split: held-out frames leak into train | `tests/test_split.py` | killed |
| M7 | `d3gs/metrics.py` | PSNR: 20*log10 instead of 10*log10 on MSE | `tests/test_metrics.py` | killed |
| M8 | `d3gs/metrics.py` | SSIM: wrong K2 constant | `tests/test_metrics.py` | killed |
| M9 | `d3gs/metrics.py` | masked PSNR: averages over all pixels instead of static ones | `tests/test_metrics.py` | killed |
| M10 | `d3gs/metrics.py` | masked SSIM: mask not aligned with the 'valid' SSIM map (off by the window radius) | `tests/test_metrics.py` | killed |
| M11 | `d3gs/dynamic.py` | cuboid mask: near-plane clipping removed (corners behind the camera project mirrored) | `tests/test_dynamic.py` | killed |
| M12 | `d3gs/dynamic.py` | cuboid corners: length and width swapped | `tests/test_dynamic.py` | killed |
| M13 | `d3gs/dynamic.py` | polygon fill: union of half-planes instead of intersection | `tests/test_dynamic.py` | killed |
| M14 | `d3gs/dynamic.py` | track speed: wrong time base (half speed) | `tests/test_dynamic.py` | killed |
| M15 | `d3gs/dynamic.py` | cuboid interpolation to camera time: weight reversed | `tests/test_dynamic.py` | killed |
| M16 | `d3gs/dynamic.py` | moving mask: speed threshold ignored (parked cars masked as moving) | `tests/test_dynamic.py` | killed |
| M17 | `d3gs/lidar.py` | depth map: z-buffer keeps the farthest point | `tests/test_lidar.py` | killed |
| M18 | `d3gs/lidar.py` | depth map: floor instead of round (pixel-centre convention broken in u) | `tests/test_lidar.py` | killed |
| M19 | `d3gs/lidar.py` | depth map: points behind the camera not rejected | `tests/test_lidar.py` | killed |
| M20 | `d3gs/lidar.py` | sweep split: reserves the sweep AFTER the held-out frame's nearest one | `tests/test_lidar.py` | killed |
| M21 | `d3gs/lidar.py` | leak guard disabled | `tests/test_lidar.py` | killed |
| M22 | `d3gs/scene.py` | Scene.train_depth serves held-out frames' lidar | `tests/test_scene_guard.py` | killed |
| M23 | `d3gs/splat.py` | .splat: quaternion written as (x, y, z, w) instead of (w, x, y, z) | `tests/test_splat.py` | killed |
| M24 | `d3gs/splat.py` | web pruning: opacity floor ignored | `tests/test_splat.py` | killed |
| M25 | `d3gs/lidar.py` | depth: invalid predictions removed from the fixed ground-truth cohort | `tests/test_metric_integrity.py::test_missing_prediction_cannot_improve_a_fixed_depth_cohort` | killed |
| M26 | `d3gs/metrics.py` | masked SSIM: mismatched channels accepted | `tests/test_metric_integrity.py::test_masked_ssim_does_not_ignore_extra_error_channels` | killed |
| M27 | `d3gs/provenance.py` | run identity: changed checkpoint accepted | `tests/test_scene_integrity.py::test_identity_binds_training_eval_init_split_and_checkpoint` | killed |
| M28 | `d3gs/web_asset.py` | web asset: same-size content substitution accepted | `tests/test_web_evidence.py::test_manifest_rejects_same_size_changes_and_missing_files` | killed |
| M29 | `d3gs/report_keys.py` | off-path report: sub-metre offsets collide | `tests/test_scene_integrity.py::test_offpath_actual_main_preserves_decimal_offsets` | killed |
| M30 | `d3gs/scene.py` | scene: training and held-out cohort overlap accepted | `tests/test_scene_integrity.py::test_scene_rejects_invalid_metadata` | killed |
| M31 | `d3gs/provenance.py` | run identity: changed training recipe accepted | `tests/test_scene_integrity.py::test_recorded_training_metadata_is_bound` | killed |
| M32 | `d3gs/provenance.py` | run identity: a different prepared scene accepted | `tests/test_scene_integrity.py::test_identity_copied_scene_accepted_but_changes_rejected` | killed |
| M33 | `scripts/train.py` | training depth: empty support introduces NaN loss | `tests/test_scene_integrity.py::test_sparse_depth_empty_support_and_nonfinite_are_explicit` | killed |

## Historical checks

The September checks below used the original runner, which accepted any nonzero
pytest exit. The current table above uses stricter failure classification.


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
