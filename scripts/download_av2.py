"""Anonymous, partial download of ONE Argoverse 2 sensor log over plain HTTPS.

The AV2 bucket (s3://argoverse) allows anonymous ListObjectsV2 / GetObject, so no
AWS account, login or CLI is needed. We fetch only:
  * calibration/ (intrinsics + egovehicle_SE3_sensor)
  * city_SE3_egovehicle.feather (ego poses), annotations.feather (for reference)
  * a contiguous window of one camera's JPEGs
  * a handful of lidar sweeps inside that window (for Gaussian initialisation)

Usage:
  python scripts/download_av2.py --out D:/driving-3dgs/data --split val \
      --log 02678d04-cc9f-3148-9f95-1ba66347dff9 --start 100 --num-frames 160 --lidar-every 4
  python scripts/download_av2.py --out ... --split val --log ... --meta-only
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://argoverse.s3.amazonaws.com"
PREFIX = "datasets/av2/sensor"


def _get(url: str, retries: int = 5, timeout: float = 60.0) -> bytes:
    last = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccups: retry with backoff
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"failed to GET {url}: {last}")


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """All (key, size) under prefix, following continuation tokens."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        xml = _get(f"{BUCKET}/?{urllib.parse.urlencode(q)}").decode()
        for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", xml):
            out.append((k, int(s)))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if "<IsTruncated>true</IsTruncated>" in xml and m:
            token = m.group(1)
        else:
            return out


def fetch_all(keys: list[tuple[str, int]], root: Path, log_prefix: str, workers: int = 24) -> int:
    def one(item):
        key, size = item
        dst = root / key[len(log_prefix):]
        if dst.exists() and dst.stat().st_size == size:
            return size
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = _get(f"{BUCKET}/{urllib.parse.quote(key)}")
        if len(data) != size:
            raise RuntimeError(f"size mismatch for {key}: {len(data)} != {size}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dst)
        return size

    total = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, n in enumerate(ex.map(one, keys), 1):
            total += n
            if i % 20 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)} files, {total / 1e6:.1f} MB", flush=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--log", required=True)
    ap.add_argument("--camera", default="ring_front_center")
    ap.add_argument("--start", type=int, default=0, help="index of first camera frame in the log")
    ap.add_argument("--num-frames", type=int, default=160)
    ap.add_argument("--lidar-every", type=int, default=4, help="keep every k-th lidar sweep in the window")
    ap.add_argument("--meta-only", action="store_true")
    args = ap.parse_args()

    log_prefix = f"{PREFIX}/{args.split}/{args.log}/"
    root = Path(args.out) / args.log
    root.mkdir(parents=True, exist_ok=True)

    meta = [(k, s) for k, s in list_keys(log_prefix + "calibration/")]
    for name in ("city_SE3_egovehicle.feather", "annotations.feather"):
        meta += [(k, s) for k, s in list_keys(log_prefix + name)]
    wanted = list(meta)

    cams = sorted(list_keys(f"{log_prefix}sensors/cameras/{args.camera}/"))
    print(f"log has {len(cams)} {args.camera} frames")
    manifest = {"log": args.log, "split": args.split, "camera": args.camera,
                "n_camera_frames_in_log": len(cams)}
    if not args.meta_only:
        win = cams[args.start:args.start + args.num_frames]
        if len(win) < args.num_frames:
            raise SystemExit(f"window too short: {len(win)} frames")
        t0 = int(Path(win[0][0]).stem)
        t1 = int(Path(win[-1][0]).stem)
        lidar = sorted(list_keys(f"{log_prefix}sensors/lidar/"))
        lid_win = [kv for kv in lidar if t0 <= int(Path(kv[0]).stem) <= t1][:: args.lidar_every]
        wanted += win + lid_win
        manifest.update({"start": args.start, "num_frames": len(win), "t0_ns": t0, "t1_ns": t1,
                         "n_lidar_sweeps": len(lid_win), "lidar_every": args.lidar_every})

    t = time.time()
    total = fetch_all(wanted, root, log_prefix)
    manifest.update({"n_files": len(wanted), "bytes": total, "seconds": round(time.time() - t, 1),
                     "source": f"s3://argoverse/{log_prefix}", "files": [k[len(log_prefix):] for k, _ in wanted]})
    name = "manifest_meta.json" if args.meta_only else "manifest.json"
    (root / name).write_text(json.dumps(manifest, indent=1))
    print(f"done: {len(wanted)} files, {total / 1e6:.1f} MB in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
