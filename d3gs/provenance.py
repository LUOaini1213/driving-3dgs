"""Content identity for prepared scenes and checkpoints; paths are locations, not identity.

This verifies a locally recorded training declaration, not a signed attestation of an
external training process. Held-out files are hashed for identity, not optimization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def capture_scene(scene):
    """Bind metadata, every image, initialization, and all supplied depth/mask files."""
    # Only a raw-log location may change during a copy. All scientific metadata is bound.
    meta = {k: v for k, v in scene.meta.items() if k != "log_dir"}
    files = {"init_points.npz": sha256(scene.dir / "init_points.npz")}
    for frame in scene.meta["frames"]:
        files[frame["image"]] = sha256(scene.dir / frame["image"])
    for folder, ids, suffix in (("depth", scene.train, ".npy"), ("eval_depth", scene.test, ".npy"),
                                ("masks/moving", range(len(scene.meta["frames"])), ".png"),
                                ("masks/vehicles", range(len(scene.meta["frames"])), ".png")):
        directory = scene.dir / folder
        if directory.exists():
            for i in ids:
                relative = f"{folder}/{i:04d}{suffix}"
                files[relative] = sha256(scene.dir / relative)
            # Stray files are also bound: a prepared directory cannot change unnoticed.
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    files[path.relative_to(scene.dir).as_posix()] = sha256(path)
    lidar = scene.lidar_split
    if (scene.dir / "lidar_split.json").exists():
        files["lidar_split.json"] = sha256(scene.dir / "lidar_split.json")
    manifest = dict(metadata=meta, files=files, lidar=lidar)
    return dict(manifest=manifest, sha256=_digest(manifest))


def make_provenance(scene, checkpoint, initial, stats):
    current = capture_scene(scene)
    # Reload metadata too: the in-memory Scene must not hide edits during training.
    from .scene import Scene
    if current != initial or capture_scene(Scene(scene.dir)) != initial:
        raise ValueError("prepared scene content changed during training")
    return dict(schema=1, scene=initial, checkpoint_sha256=sha256(checkpoint),
                training={k: v for k, v in stats.items() if k not in ("work", "provenance")})


def verify_run(run_dir, scene, stats=None, *, allow_legacy=False):
    """Reject mismatches. Explicitly allowed legacy checkpoints remain unverified."""
    run_dir = Path(run_dir)
    if stats is None:
        stats = json.loads((run_dir / "train_stats.json").read_text(encoding="utf-8"))
    recipe = {k: v for k, v in stats.items() if k not in ("work", "provenance")}
    # Training logs may contain a legitimate +Infinity perfect-match PSNR.
    metadata_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    proof = stats.get("provenance")
    if proof is None:
        if not allow_legacy:
            raise ValueError("legacy run has no provenance; use --allow-unverified-run to report it as unverified")
        return dict(status="unverified_legacy", reason="no recorded checkpoint/scene binding",
                    checkpoint_sha256=sha256(run_dir / "ckpt.pt"), scene_sha256=capture_scene(scene)["sha256"],
                    metadata_sha256=metadata_hash, lidar=None)
    if not isinstance(proof, dict) or proof.get("schema") != 1:
        raise ValueError("unsupported run provenance")
    checkpoint_hash = sha256(run_dir / "ckpt.pt")
    if proof.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("checkpoint content does not match recorded run provenance")
    current = capture_scene(scene)
    if proof.get("scene") != current:
        raise ValueError("prepared scene content does not match recorded run provenance")
    if proof.get("training") != recipe:
        raise ValueError("training metadata does not match recorded run provenance")
    return dict(status="verified", checkpoint_sha256=checkpoint_hash, scene_sha256=current["sha256"],
                metadata_sha256=metadata_hash, lidar=current["manifest"]["lidar"])


def capture_eval_inputs(scene):
    """Record the exact auxiliary evaluation cohort, independently of its training split."""
    files = {}
    for i in scene.test:
        image = scene.meta["frames"][i]["image"]
        files[image] = sha256(scene.dir / image)
        for folder, suffix in (("eval_depth", ".npy"), ("masks/moving", ".png")):
            if (scene.dir / folder).exists():
                relative = f"{folder}/{i:04d}{suffix}"
                files[relative] = sha256(scene.dir / relative)
    manifest = dict(frames=[scene.meta["frames"][i] for i in scene.test], test=scene.test,
                    resolution=[scene.W, scene.H], K=scene.meta["K"], city_origin=scene.meta.get("city_origin"),
                    camera=scene.meta.get("camera"), lidar=scene.lidar_split, files=files)
    return dict(sha256=_digest(manifest), manifest=manifest)


def capture_lidar_sources(log_dir, eval_timestamps):
    log_dir = Path(log_dir)
    names = ["city_SE3_egovehicle.feather"] + [f"sensors/lidar/{t}.feather" for t in eval_timestamps]
    return {name: sha256(log_dir / name) for name in names}


def verify_lidar_sources(scene, *, allow_legacy=False):
    split = scene.lidar_split
    recorded = (split or {}).get("eval_source_sha256")
    if recorded is None:
        if not allow_legacy:
            raise ValueError("unverified raw lidar sources; use --allow-unverified-run for legacy data")
        return dict(status="unverified_legacy", reason="no recorded raw lidar source hashes")
    current = capture_lidar_sources(scene.meta["log_dir"], split["eval_sweep_ts"])
    if current != recorded:
        raise ValueError("raw evaluation lidar/pose content does not match prepared scene")
    return dict(status="verified", files=current)
