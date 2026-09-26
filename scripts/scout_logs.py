"""How the log was chosen: rank AV2 val logs by how many annotated objects actually move.

Downloads only city_SE3_egovehicle.feather + annotations.feather (~0.5 MB per log) into memory.
A track counts as 'moving' if its centre (in the city frame) moves > 3 m over the log.

Usage:  python scripts/scout_logs.py --n 40
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_av2 import BUCKET, PREFIX, _get  # noqa: E402
from d3gs.geometry import quat_to_rotmat  # noqa: E402


def score(split: str, log: str):
    base = f"{BUCKET}/{PREFIX}/{split}/{log}"
    p = pd.read_feather(io.BytesIO(_get(base + "/city_SE3_egovehicle.feather"))).sort_values("timestamp_ns")
    a = pd.read_feather(io.BytesIO(_get(base + "/annotations.feather")))
    xyz = p[["tx_m", "ty_m", "tz_m"]].to_numpy()
    path = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    ts = p.timestamp_ns.to_numpy()
    idx = np.searchsorted(ts, a.timestamp_ns.to_numpy()).clip(0, len(ts) - 1)
    R = quat_to_rotmat(p[["qw", "qx", "qy", "qz"]].to_numpy()[idx])
    c = np.einsum("nij,nj->ni", R, a[["tx_m", "ty_m", "tz_m"]].to_numpy()) + xyz[idx]
    a = a.assign(cx=c[:, 0], cy=c[:, 1])
    g = a.groupby("track_uuid").agg(x0=("cx", "first"), x1=("cx", "last"), y0=("cy", "first"), y1=("cy", "last"))
    moving = int((np.hypot(g.x1 - g.x0, g.y1 - g.y0) > 3).sum())
    return log, round(path, 1), moving, a.track_uuid.nunique()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=40, help="first n logs (bucket listing order)")
    args = ap.parse_args()
    xml = _get(f"{BUCKET}/?list-type=2&prefix={PREFIX}/{args.split}/&delimiter=/").decode()
    logs = re.findall(rf"{args.split}/([0-9a-f-]{{36}})/", xml)[: args.n]
    with cf.ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda L: score(args.split, L), logs))
    print("log_id, ego_path_m, moving_tracks, all_tracks")
    for r in sorted(rows, key=lambda r: r[2]):
        print(*r, sep=", ")


if __name__ == "__main__":
    main()
