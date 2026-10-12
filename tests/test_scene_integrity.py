"""Prepared inputs and run identity, exercised with real files and CPU entry points."""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from d3gs.scene import Scene
from scripts import evaluate, offpath, prepare, train


def make_work(path):
    path.mkdir(parents=True)
    (path / "images").mkdir()
    frames = []
    for i in range(3):
        pose = np.eye(4)
        pose[0, 3] = i
        frames.append(dict(idx=i, timestamp_ns=i * 100, image=f"images/{i:04d}.png", c2w=pose.tolist()))
        ok, encoded = cv2.imencode(".png", np.full((16, 16, 3), 40 + i, np.uint8))
        assert ok
        (path / frames[-1]["image"]).write_bytes(encoded.tobytes())
    meta = dict(width=16, height=16, K=[[10, 0, 8], [0, 10, 8], [0, 0, 1]], frames=frames,
                train=[0, 2], test=[1], train_stride=1, camera="front", city_origin=[0, 0, 0],
                log_dir="relocatable/source", holdout_every=2, duration_s=0.1, trajectory_length_m=2,
                init={}, masks=dict(speed_thresh_mps=1, pad_m=.25))
    (path / "cameras.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez(path / "init_points.npz", xyz=np.array([[0, 0, 10], [1, 0, 10], [0, 1, 10], [1, 1, 10]], np.float32),
             rgb=np.full((4, 3), .2, np.float32))
    (path / "lidar_split.json").write_text(json.dumps(dict(policy="strict", sweep_ts=[0, 100, 200],
        eval_sweep_ts=[100], init_sweep_ts=[0, 200], depth_sweep_ts=[0, 200])), encoding="utf-8")
    for folder, ids in (("depth", [0, 2]), ("eval_depth", [1])):
        (path / folder).mkdir()
        for i in ids:
            np.save(path / folder / f"{i:04d}.npy", np.full((16, 16), 10, np.float32))
    return path


@pytest.fixture
def work(tmp_path):
    return make_work(tmp_path / "场景 with spaces")


@pytest.mark.parametrize("field,value", [("train", [0, 1]), ("test", [1, 1]), ("test", [3]),
    ("train", [True, 2]), ("train", []), ("width", 0), ("height", True), ("K", [[1, 2]]),
    ("K", [[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]])])
def test_scene_rejects_invalid_metadata(work, field, value):
    p = work / "cameras.json"
    meta = json.loads(p.read_text())
    meta[field] = value
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        Scene(work)


def test_scene_image_unicode_and_depth_shape(work):
    sc = Scene(work)
    assert sc.image(1).shape == (16, 16, 3)
    np.save(work / "depth/0000.npy", np.ones((2, 2)))
    with pytest.raises(ValueError, match="depth"):
        sc.train_depth(0)


def test_sparse_split_need_not_cover_all_frames(work):
    p = work / "cameras.json"
    meta = json.loads(p.read_text())
    meta["train"] = [0]
    p.write_text(json.dumps(meta))
    assert Scene(work).train == [0]


def make_run(path, scene, verified=True):
    path.mkdir()
    torch.save({"means": torch.ones(4, 3)}, path / "ckpt.pt")
    stats = dict(sh_degree=0, steps=1, mask="none", depth_lambda=0, work=str(scene.dir), lidar_policy="strict")
    if verified:
        from d3gs.provenance import capture_scene, make_provenance
        manifest = capture_scene(scene)
        stats["provenance"] = make_provenance(scene, path / "ckpt.pt", manifest, stats)
    (path / "train_stats.json").write_text(json.dumps(stats))
    return path


def test_identity_copied_scene_accepted_but_changes_rejected(work, tmp_path):
    from d3gs.provenance import verify_run
    run = make_run(tmp_path / "run", Scene(work))
    copied = tmp_path / "copy"
    shutil.copytree(work, copied)
    p = copied / "cameras.json"
    meta = json.loads(p.read_text())
    meta["log_dir"] = "elsewhere/same/source"
    p.write_text(json.dumps(meta, indent=2))
    assert verify_run(run, Scene(copied))["status"] == "verified"
    np.save(copied / "eval_depth/0001.npy", np.full((16, 16), 11, np.float32))
    with pytest.raises(ValueError, match="scene|content"):
        verify_run(run, Scene(copied), allow_legacy=True)


@pytest.mark.parametrize("target", ["images/0000.png", "images/0001.png", "init_points.npz", "lidar_split.json", "ckpt.pt"])
def test_identity_binds_training_eval_init_split_and_checkpoint(work, tmp_path, target):
    from d3gs.provenance import verify_run
    run = make_run(tmp_path / "run", Scene(work))
    path = (run if target == "ckpt.pt" else work) / target
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        verify_run(run, Scene(work), allow_legacy=True)


def test_legacy_is_explicitly_unverified(work, tmp_path):
    from d3gs.provenance import verify_run
    run = make_run(tmp_path / "run", Scene(work), verified=False)
    with pytest.raises(ValueError, match="legacy|provenance|unverified"):
        verify_run(run, Scene(work))
    assert verify_run(run, Scene(work), allow_legacy=True)["status"] == "unverified_legacy"


def test_evaluate_refuses_unproven_run_before_gpu_or_outputs(work, tmp_path, monkeypatch):
    run = make_run(tmp_path / "run", Scene(work), verified=False)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--work", str(work), "--run", str(run), "--results", str(tmp_path / "res.json")])
    monkeypatch.setattr(torch.nn.ParameterDict, "cuda", lambda *a, **k: pytest.fail("GPU entered before validation"))
    with pytest.raises(ValueError, match="legacy|provenance|unverified"):
        evaluate.main()
    assert not (run / "renders").exists()


def test_offpath_decimal_offsets_are_distinct():
    assert offpath.offset_key(.1) != offpath.offset_key(.2)
    assert offpath.offset_key(-2) == "-2"
    assert offpath.offset_key(0) == "+0"
    assert offpath.offset_key(.123456789) != offpath.offset_key(.123456788)


def test_empty_static_summary_is_missing_not_nan():
    row = {"3dgs": {"psnr": float("nan"), "ssim": None, "lpips": None}}
    result = evaluate.summarise([row], "3dgs")
    assert result["psnr"] is None
    assert result["counts"]["psnr"] == 0


def test_prepare_rejects_reused_output_before_reading_raw_inputs(work, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare", "--log-dir", "missing", "--out", str(work)])
    with pytest.raises(ValueError, match="empty|existing"):
        prepare.main()


def test_train_refuses_existing_run_before_gpu(work, tmp_path, monkeypatch):
    run = make_run(tmp_path / "run", Scene(work), verified=False)
    monkeypatch.setattr(sys, "argv", ["train", "--work", str(work), "--out", str(run), "--steps", "1"])
    with pytest.raises(ValueError, match="existing|already|empty"):
        train.main()


@pytest.mark.parametrize("key,value", [("steps", 30000), ("seed", 999), ("mask", "moving")])
def test_recorded_training_metadata_is_bound(work, tmp_path, key, value):
    from d3gs.provenance import verify_run
    run = make_run(tmp_path / "run", Scene(work))
    p = run / "train_stats.json"
    stats = json.loads(p.read_text())
    stats[key] = value
    p.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="metadata"):
        verify_run(run, Scene(work))


def cpu_render(params, viewmat, K, W, H, sh_degree, with_depth=False):
    image = torch.full((H, W, 3), .5)
    alpha = torch.ones((H, W, 1))
    return (image, alpha, {}, torch.full((H, W), 10.)) if with_depth else (image, alpha, {})


def cpu_evaluation(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(evaluate, "render", cpu_render)
    monkeypatch.setattr(offpath, "render", cpu_render)
    monkeypatch.setattr(evaluate, "init_params", lambda *a, **k: {"means": torch.zeros((4, 3))})
    monkeypatch.setattr(evaluate, "load_lpips", lambda *a: None)


@pytest.mark.parametrize("verified", [False, True])
def test_evaluate_actual_main_identity_and_empty_static(work, tmp_path, monkeypatch, verified):
    from d3gs.image_io import write_image
    (work / "masks/moving").mkdir(parents=True)
    for i in range(3):
        write_image(work / f"masks/moving/{i:04d}.png", np.full((16, 16), 255, np.uint8))
    run = make_run(tmp_path / "run", Scene(work), verified=verified)
    cpu_evaluation(monkeypatch)
    result = tmp_path / "评估结果.json"
    argv = ["evaluate", "--work", str(work), "--run", str(run), "--results", str(result), "--figs", str(tmp_path / "对比图")]
    if not verified:
        argv.append("--allow-unverified-run")
    monkeypatch.setattr(sys, "argv", argv)
    evaluate.main()
    data = json.loads(result.read_text())
    assert data["run_identity"]["status"] == ("verified" if verified else "unverified_legacy")
    assert data["heldout_depth"]["leak_free"] is (True if verified else None)
    assert data["lidar_policy"] == ("strict" if verified else "unverified")
    assert data["heldout_static_mean"]["3dgs"]["counts"] == dict(psnr=0, ssim=0, lpips=0)
    assert data["heldout_static_mean"]["3dgs"]["psnr"] is None
    assert data["per_frame"][0]["3dgs"]["static"]["n_pixels"] == 0
    assert "NaN" not in result.read_text()
    assert (run / "renders/test_0001.png").is_file()
    assert (tmp_path / "对比图/heldout_0001.jpg").is_file()


def add_raw_sources(work, tmp_path):
    from d3gs.provenance import capture_lidar_sources
    raw = tmp_path / "原始来源"
    (raw / "sensors/lidar").mkdir(parents=True)
    for name in ("city_SE3_egovehicle.feather", "sensors/lidar/100.feather"):
        (raw / name).write_bytes(b"fixture contents; loaders adapted in CPU tests")
    p = work / "cameras.json"
    meta = json.loads(p.read_text())
    meta["log_dir"] = str(raw)
    p.write_text(json.dumps(meta))
    p = work / "lidar_split.json"
    split = json.loads(p.read_text())
    split["eval_source_sha256"] = capture_lidar_sources(raw, split["eval_sweep_ts"])
    p.write_text(json.dumps(split))
    return raw


@pytest.mark.parametrize("verified", [False, True])
def test_offpath_actual_main_preserves_decimal_offsets(work, tmp_path, monkeypatch, verified):
    from types import SimpleNamespace
    add_raw_sources(work, tmp_path)
    run = make_run(tmp_path / "run", Scene(work), verified=verified)
    cpu_evaluation(monkeypatch)
    monkeypatch.setattr(offpath, "load_poses", lambda p: SimpleNamespace(city_SE3_ego=lambda t: np.eye(4)))
    monkeypatch.setattr(offpath, "load_sweep_xyz", lambda p: np.array([[4., 0, 10.], [5., 1, 10.]]))
    result = tmp_path / "偏移.json"
    argv = ["offpath", "--work", str(work), "--runs", str(run), "--offsets", ".1", ".2", "--results", str(result)]
    if not verified:
        argv.append("--allow-unverified-run")
    monkeypatch.setattr(sys, "argv", argv)
    offpath.main()
    data = json.loads(result.read_text())["runs"]["run"]
    assert list(data["per_offset"]) == ["+0.1", "+0.2"]
    assert data["leak_free"] is (True if verified else None)
    assert all(row["depth_n"] == 2 for row in data["per_offset"].values())


def test_raw_eval_source_change_is_rejected(work, tmp_path):
    from d3gs.provenance import verify_lidar_sources
    raw = add_raw_sources(work, tmp_path)
    assert verify_lidar_sources(Scene(work))["status"] == "verified"
    (raw / "sensors/lidar/100.feather").write_bytes(b"different data")
    with pytest.raises(ValueError, match="raw evaluation"):
        verify_lidar_sources(Scene(work), allow_legacy=True)


def test_aux_sparse_training_split_and_recentring_are_compatible(work, tmp_path):
    copied = tmp_path / "sparse"
    shutil.copytree(work, copied)
    p = copied / "cameras.json"
    meta = json.loads(p.read_text())
    meta["train"] = [0]
    meta["train_stride"] = 4
    meta["city_origin"] = [1, 0, 0]
    for frame in meta["frames"]:
        frame["c2w"][0][3] -= 1
    p.write_text(json.dumps(meta))
    evaluate.check_same_cameras(Scene(work), Scene(copied))
    meta["city_origin"] = [2, 0, 0]
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="cameras"):
        evaluate.check_same_cameras(Scene(work), Scene(copied))


def test_sparse_depth_empty_support_and_nonfinite_are_explicit():
    prediction = torch.ones((3, 3), requires_grad=True)
    loss = train.sparse_depth_loss(prediction, torch.zeros((3, 3)))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    for pred, gt in ((torch.full((3, 3), float("nan")), torch.zeros((3, 3))),
                     (torch.ones((3, 3)), torch.full((3, 3), float("nan")))):
        with pytest.raises(ValueError):
            train.sparse_depth_loss(pred, gt)


def test_prepare_actual_main_unicode_images_and_source_proof(tmp_path, monkeypatch):
    from d3gs.av2io import CameraCalib
    from d3gs.image_io import write_image
    from types import SimpleNamespace
    raw = tmp_path / "原始"
    raw.mkdir()
    frames, sweeps = [], []
    (raw / "sensors/lidar").mkdir(parents=True)
    (raw / "city_SE3_egovehicle.feather").write_bytes(b"pose fixture")
    for i in range(3):
        path = raw / f"相机{i}.png"
        write_image(path, np.full((16, 16, 3), 100 + i, np.uint8))
        frames.append((i * 100, path))
        path = raw / f"sensors/lidar/{i * 100}.feather"
        path.write_bytes(b"lidar fixture")
        sweeps.append((i * 100, path))
    calib = CameraCalib("front", np.array([[10., 0, 8], [0, 10, 8], [0, 0, 1]]), np.zeros(3), 16, 16, np.eye(4))
    monkeypatch.setattr(prepare, "load_camera_calib", lambda *a: calib)
    monkeypatch.setattr(prepare, "load_poses", lambda *a: SimpleNamespace(city_SE3_ego=lambda t: np.eye(4), gap_ns=lambda t: 0))
    monkeypatch.setattr(prepare, "list_frames", lambda *a: frames)
    monkeypatch.setattr(prepare, "list_sweeps", lambda *a: sweeps)
    monkeypatch.setattr(prepare, "load_sweep_xyz", lambda *a: np.array([[4., 0, 10.], [5., 0, 10.], [4., 1, 10.], [5., 1, 10.]]))
    out = tmp_path / "准备完成"
    monkeypatch.setattr(sys, "argv", ["prepare", "--log-dir", str(raw), "--out", str(out), "--width", "16", "--height", "16",
                                    "--crop-bottom", "16", "--holdout-every", "2", "--n-sky", "0"])
    prepare.main()
    scene = Scene(out)
    assert scene.image(1).shape == (16, 16, 3)
    assert scene.lidar_split["eval_source_sha256"]["sensors/lidar/100.feather"]
    with np.load(out / "init_points.npz") as points:
        assert points["xyz"].shape == (4, 3)


@pytest.mark.parametrize("fail_proof", [False, True])
def test_train_actual_cpu_step_records_proof_and_skips_empty_depth(work, tmp_path, monkeypatch, fail_proof):
    import types
    from d3gs.provenance import verify_run
    from d3gs.scene import init_params
    for i in (0, 2):
        np.save(work / f"depth/{i:04d}.npy", np.zeros((16, 16), np.float32))
    gsplat = types.ModuleType("gsplat")
    gsplat.__version__ = "CPU-test-adapter"
    strategy = types.ModuleType("gsplat.strategy")
    class Strategy:
        def __init__(self, **kwargs): pass
        def check_sanity(self, *args): pass
        def initialize_state(self, **kwargs): return {}
        def step_pre_backward(self, *args): pass
        def step_post_backward(self, *args, **kwargs): pass
    strategy.DefaultStrategy = Strategy
    monkeypatch.setitem(sys.modules, "gsplat", gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.strategy", strategy)
    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        if args and args[0] == "cuda":
            args = ("cpu",) + args[1:]
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(train, "init_params", lambda xyz, rgb, sh, dev: init_params(xyz, rgb, sh, "cpu"))
    def differentiable_render(params, vm, K, W, H, sh, with_depth=False):
        value = params["means"].mean()
        return value.sigmoid().expand(H, W, 3), torch.ones(H, W, 1), {}, (value + 10).expand(H, W)
    monkeypatch.setattr(train, "render", differentiable_render)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "CPU-test-adapter")
    run = tmp_path / "新训练"
    monkeypatch.setattr(sys, "argv", ["train", "--work", str(work), "--out", str(run), "--steps", "1", "--depth-lambda", ".1"])
    if fail_proof:
        def refused_proof(*args, **kwargs):
            raise ValueError("changed during training")
        monkeypatch.setattr(train, "make_provenance", refused_proof)
        with pytest.raises(ValueError, match="changed during training"):
            train.main()
        assert (run / "ckpt.pt").exists()
        assert not (run / "train_stats.json").exists()
        return
    train.main()
    stats = json.loads((run / "train_stats.json").read_text())
    assert np.isfinite(stats["log"][0]["loss"])
    assert verify_run(run, Scene(work))["status"] == "verified"


def test_offpath_nonfinite_alpha_cannot_count_as_perfect_coverage(work, tmp_path, monkeypatch):
    from types import SimpleNamespace
    add_raw_sources(work, tmp_path)
    run = make_run(tmp_path / "run", Scene(work))
    cpu_evaluation(monkeypatch)
    monkeypatch.setattr(offpath, "load_poses", lambda p: SimpleNamespace(city_SE3_ego=lambda t: np.eye(4)))
    monkeypatch.setattr(offpath, "load_sweep_xyz", lambda p: np.array([[4., 0, 10.]]))
    def bad_render(*args, **kwargs):
        image, alpha, info, depth = cpu_render(*args, **kwargs)
        alpha[:] = float("nan")
        return image, alpha, info, depth
    monkeypatch.setattr(offpath, "render", bad_render)
    result = tmp_path / "bad.json"
    monkeypatch.setattr(sys, "argv", ["offpath", "--work", str(work), "--runs", str(run), "--offsets", ".1", "--results", str(result)])
    with pytest.raises(ValueError, match="alpha"):
        offpath.main()
    assert not result.exists()


@pytest.mark.parametrize("xyz,rgb", [(np.ones((3, 3)), np.ones((3, 3))),
    (np.ones((4, 3)), np.ones((3, 3))), (np.full((4, 3), np.nan), np.ones((4, 3))),
    (np.ones((4, 3)), np.full((4, 3), 2))])
def test_initial_points_validation(work, xyz, rgb):
    np.savez(work / "init_points.npz", xyz=xyz, rgb=rgb)
    with pytest.raises(ValueError, match="initial points"):
        Scene(work).initial_points()


def test_make_proof_refuses_files_changed_during_training(work, tmp_path):
    from d3gs.provenance import capture_scene, make_provenance
    sc = Scene(work)
    before = capture_scene(sc)
    p = work / "cameras.json"
    meta = json.loads(p.read_text())
    meta["train"] = [0]
    p.write_text(json.dumps(meta))
    checkpoint = tmp_path / "new.pt"
    checkpoint.write_bytes(b"new checkpoint")
    with pytest.raises(ValueError, match="during training"):
        make_provenance(sc, checkpoint, before, dict(sh_degree=0))


@pytest.mark.parametrize("change", ["image", "camera", "pose", "test"])
def test_aux_rejects_wrong_eval_cohort(work, tmp_path, change):
    copied = tmp_path / "aux"
    shutil.copytree(work, copied)
    p = copied / "cameras.json"
    meta = json.loads(p.read_text())
    if change == "image":
        (copied / "images/0001.png").write_bytes(b"different")
    elif change == "camera":
        meta["camera"] = "other"
    elif change == "pose":
        meta["frames"][1]["c2w"][0][3] += 10
    else:
        meta["train"], meta["test"] = [0, 1], [2]
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="match"):
        evaluate.check_same_cameras(Scene(work), Scene(copied))
