"""Build/check an explicitly allowlisted Pages artifact; no model or GPU required."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "build" / "pages"
BASELINE = "f3df5919154b60350c8f6a2b18d6bc2cbc57ff5b"
SCENE_SHA = "2dc35c0f05432772867395d2bad81101e76e30dead666671d37c5d1a4bdfb948"
SCENE_BYTES = 9_918_624
PREVIEW_SHA = "f76e953ad0c0a1051898d39ee1fa688fadd80f4181c127cb25964dfb860eace8"
VENDOR = {
    "vendor/gaussian-splats-3d.module.js": (
        "site/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js",
        "f273394134c9174e49bf42bb4ea0945caaad4fdd63023b8996cc6cd248f52a6b"),
    "vendor/gaussian-splats-3d.module.js.map": (
        "site/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js.map",
        "4a15cb505fd18c0f56a8264799d96e96705bcafcaf95dea40af87b723b6a3aa0"),
    "vendor/three.module.js": (
        "site/node_modules/three/build/three.module.js",
        "76dea8151bc9352aef3528b4262e249b2604f62543828328db978d060d61a495"),
}
TEXT = {
    name: "site/" + name for name in (
        "index.html", "viewer.css", "viewer.js", "viewer-frame.html", "viewer-frame.css",
        "viewer-frame.js", "scene-promise.js", "favicon.svg", "NOTICES.txt")
}
TEXT.update({
    "vendor/LICENSE-gaussian.txt": "site/node_modules/@mkkellogg/gaussian-splats-3d/LICENSE",
    "vendor/LICENSE-three.txt": "site/node_modules/three/LICENSE",
    "LICENSE-code.txt": "LICENSE",
})
BINARY = {"assets/full_7k.splat": "docs/splat/full_7k.splat", "assets/preview.jpg": "docs/img/viewer_headless.jpg"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_source(relative: str) -> bytes:
    source = ROOT / relative
    if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
        raise ValueError("source must be a regular repository file: " + relative)
    return source.read_bytes()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def expected(require_clean: bool = False) -> dict[str, bytes]:
    revision = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("cannot identify source commit")
    dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    if require_clean and dirty:
        raise ValueError("publishing requires a clean checkout; local previews remain labelled as uncommitted")
    scene = read_source(BINARY["assets/full_7k.splat"])
    if len(scene) != SCENE_BYTES or sha(scene) != SCENE_SHA:
        raise ValueError("the archived full_7k web asset changed; no new model export is authorized by this builder")
    if sha(read_source(BINARY["assets/preview.jpg"])) != PREVIEW_SHA:
        raise ValueError("the archived browser preview changed")
    metadata = json.loads(read_source("results/web_export.json"))
    if (metadata["run"], metadata["bytes"], metadata["n_exported"], metadata["initial_camera_frame"]) != (
            "full_7k", SCENE_BYTES, 309_957, 84):
        raise ValueError("archived export metadata changed")
    package = json.loads(read_source("site/package.json"))
    if package["dependencies"] != {"@mkkellogg/gaussian-splats-3d": "0.4.7", "three": "0.160.0"}:
        raise ValueError("viewer dependency versions changed")
    lock = json.loads(read_source("site/package-lock.json"))
    for package_name, version in package["dependencies"].items():
        if lock["packages"]["node_modules/" + package_name]["version"] != version:
            raise ValueError("dependency lock does not match " + package_name)
    output = {}
    origins = {}
    for name, relative in TEXT.items():
        content = read_source(relative).decode("utf-8").replace("\r\n", "\n")
        if name == "LICENSE-code.txt":
            content = content.split("\n---\n", 1)[0].rstrip() + "\n"
        if name == "index.html":
            replacements = {
                "__BASELINE_COMMIT__": BASELINE, "__SOURCE_COMMIT__": revision,
                "__SOURCE_SHORT__": revision[:7],
                "__SOURCE_LABEL__": "Local preview · uncommitted source" if dirty else "Published artifact source",
            }
            for token, value in replacements.items():
                content = content.replace(token, value)
            if re.search(r"__[A-Z_]+__", content):
                raise ValueError("unresolved publication token")
        output[name] = content.encode("utf-8")
        origins[name] = relative
    for name, relative in BINARY.items():
        output[name] = read_source(relative)
        origins[name] = relative
    for name, (relative, digest) in VENDOR.items():
        content = read_source(relative)
        if sha(content) != digest:
            raise ValueError("pinned npm dependency bytes changed: " + relative)
        output[name] = content
        origins[name] = relative
    index = {
        "schema": 1, "source_commit": revision, "source_dirty": dirty,
        "archived_measurement_commit": BASELINE,
        "scene": {"run": "full_7k", "bytes": SCENE_BYTES, "sha256": SCENE_SHA,
                  "gaussians": 309_957, "trained_gaussians": 557_802,
                  "camera_frame": 84, "colour": "zero-order spherical harmonics",
                  "data_license": "CC BY-NC-SA-4.0", "data_copyright": "2021 Argo AI, LLC"},
        "dependencies": package["dependencies"], "package_lock_sha256": sha(read_source("site/package-lock.json").replace(b"\r\n", b"\n")),
        "files": [{"path": name, "bytes": len(content), "sha256": sha(content), "source": origins[name]}
                  for name, content in sorted(output.items())],
        "scope": "Static viewer and one previously published pruned scene; no training data, checkpoints, logs or new performance measurements.",
    }
    output["source-index.json"] = (json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if sum(map(len, output.values())) > 15_000_000:
        raise ValueError("the isolated Pages artifact exceeds 15 MB")
    return output


def actual_files() -> dict[str, Path]:
    if OUTPUT.is_symlink() or not OUTPUT.resolve().is_relative_to(ROOT / "build"):
        raise ValueError("invalid artifact directory")
    files = {}
    if OUTPUT.exists():
        for path in OUTPUT.rglob("*"):
            if path.is_symlink():
                raise ValueError("artifact cannot contain a symlink")
            if path.is_file():
                files[path.relative_to(OUTPUT).as_posix()] = path
    return files


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true", help="read-only check of exact artifact bytes and inventory")
    parser.add_argument("--require-clean", action="store_true", help="refuse uncommitted source for a publishing build")
    args = parser.parse_args()
    try:
        files = expected(args.require_clean)
        current = actual_files()
        extras = sorted(set(current) - files.keys())
        if extras:
            raise ValueError("unexpected publish files (not uploaded or removed): " + ", ".join(extras))
        if args.check:
            missing = sorted(files.keys() - current.keys())
            changed = sorted(name for name in current if current[name].read_bytes() != files[name])
            if missing or changed:
                raise ValueError(f"artifact differs; missing={missing}, changed={changed}")
        else:
            for name, content in files.items():
                target = OUTPUT / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        print(f"PASS: {len(files)} allowlisted files; {sum(map(len, files.values())):,} bytes; archived scene SHA-256 verified")
        print("Local preview is marked source_dirty=true" if json.loads(files["source-index.json"])["source_dirty"] else "Clean source commit pinned in source-index.json")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
