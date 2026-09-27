"""Evaluate a trained run on HELD-OUT frames and compare against baselines.

Methods compared on the same held-out frames (same resolution, same metric code):
  * 3dgs         : trained Gaussians rendered at the held-out pose
  * init_only    : the lidar/sky initial Gaussians rendered before any optimisation
  * copy_nearest : the time-nearest TRAIN image copied as the prediction (trivial baseline)
Also reports the trained model on TRAIN frames (to show the train/test gap).

With --aux <strict work dir> (default: --work) it also reports
  * static-only metrics: PSNR / SSIM / LPIPS over pixels NOT covered by a moving object (AV2 cuboids);
  * depth error vs lidar on held-out frames: the rendered expected depth against the held-out frame's
    reserved evaluation sweep (never used in training), over all lidar pixels and static-only pixels.
    heldout_depth.leak_free is False when the run's initialisation or depth supervision used any of those
    sweeps (the original `full` work dir did: its init read every downloaded sweep).

Usage:
  python scripts/evaluate.py --work D:/driving-3dgs/work/<log> --run D:/driving-3dgs/runs/<name> \
      --results results/results.json --figs docs/img
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d3gs.lidar import LidarLeakError, assert_no_lidar_leak, depth_errors  # noqa: E402
from d3gs.metrics import psnr, psnr_masked, ssim, ssim_masked  # noqa: E402
from d3gs.scene import Scene, init_params, render  # noqa: E402
from d3gs.split import nearest_train_index  # noqa: E402


def load_lpips(torch_home: str):
    try:
        os.environ["TORCH_HOME"] = torch_home
        import lpips
        return lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    except Exception as e:  # noqa: BLE001
        print("LPIPS unavailable:", e)
        return None


def lp(model, a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> float | None:
    """LPIPS; with ``keep`` (H, W bool) a spatial-LPIPS map is averaged over those pixels only."""
    if model is None:
        return None
    ta = torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda() * 2 - 1
    tb = torch.from_numpy(b).permute(2, 0, 1)[None].float().cuda() * 2 - 1
    with torch.no_grad():
        if keep is None:
            return float(model(ta, tb))
        m = model(ta, tb)[0, 0].cpu().numpy()      # spatial=True model: (H, W) map
        return float(m[keep].mean())


def check_same_cameras(a: Scene, b: Scene) -> None:
    """The aux work dir must describe the same frames: same K, held-out set, timestamps, rotations and
    relative camera positions (the recentring origin may differ, e.g. for the sparse-view variant)."""
    ta = [f["timestamp_ns"] for f in a.meta["frames"]]
    tb = [f["timestamp_ns"] for f in b.meta["frames"]]
    off = a.c2w[:, :3, 3] - b.c2w[:, :3, 3]
    ok = (ta == tb and a.test == b.test and torch.allclose(a.K, b.K)
          and torch.allclose(a.c2w[:, :3, :3], b.c2w[:, :3, :3], atol=1e-5)
          and float((off - off[0]).abs().max()) < 1e-3)
    if not ok:
        raise SystemExit("--aux work dir does not match --work cameras")


def throughput(log: list[dict]) -> dict | None:
    """Wall time per logged 500-step segment: median and the slowest one (stalls show up here)."""
    seg = [(b["step"], b["elapsed_s"] - a["elapsed_s"], b["step"] - a["step"]) for a, b in zip(log, log[1:])]
    seg = [(st, dt) for st, dt, n in seg if n == 500]
    if not seg:
        return None
    med = float(np.median([dt for _, dt in seg]))
    worst = max(seg, key=lambda x: x[1])
    excess = sum(dt - med for _, dt in seg if dt > 3 * med)   # time lost in stalled segments
    return {"segment_steps": 500, "median_segment_s": round(med, 1), "slowest_segment_s": round(worst[1], 1),
            "slowest_segment_end_step": worst[0], "stall_excess_s": round(excess, 1), "n_stalled_segments": sum(dt > 3 * med for _, dt in seg)}


def rnd(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def summarise(rows: list[dict], key: str) -> dict:
    out = {}
    for m in ("psnr", "ssim", "lpips"):
        vals = [r[key][m] for r in rows if r[key].get(m) is not None]
        if vals:
            out[m] = round(float(np.mean(vals)), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figs", default=None, help="folder for small comparison JPEGs")
    ap.add_argument("--n-figs", type=int, default=3)
    ap.add_argument("--torch-home", default="D:/driving-3dgs/torch_home")
    ap.add_argument("--aux", default=None, help="work dir with masks/ and eval_depth/ (prepare.py --strict-lidar)")
    args = ap.parse_args()

    import cv2
    run = Path(args.run)
    stats = json.loads((run / "train_stats.json").read_text())
    sc = Scene(args.work)
    sh = stats["sh_degree"]
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in torch.load(run / "ckpt.pt").items()}).cuda()
    init = np.load(sc.dir / "init_points.npz")
    p0 = init_params(init["xyz"], init["rgb"], sh, "cuda")
    K = sc.K.cuda()
    lpm = load_lpips(args.torch_home)
    (run / "renders").mkdir(exist_ok=True)
    aux = Scene(args.aux) if args.aux else sc
    if aux is not sc:
        check_same_cameras(sc, aux)
    has_masks = (aux.dir / "masks" / "moving").exists()
    has_depth = (aux.dir / "eval_depth").exists()
    lpm_sp = None
    if has_masks and lpm is not None:
        import lpips
        lpm_sp = lpips.LPIPS(net="alex", verbose=False, spatial=True).cuda().eval()
    # leakage bookkeeping for the depth evaluation
    leak_free = None
    if has_depth:
        ev = aux.lidar_split["eval_sweep_ts"]
        own = sc.lidar_split
        if own is not None:
            try:
                assert_no_lidar_leak(own["init_sweep_ts"] + own.get("depth_sweep_ts", []), ev)
                leak_free = True
            except LidarLeakError:
                leak_free = False
        if stats.get("depth_lambda", 0) > 0 and leak_free is not True:
            raise SystemExit("a depth-supervised run must be leak-free")

    def rend(p, i, deg, depth=False):
        with torch.no_grad():
            if depth:
                img, _, _, d = render(p, sc.viewmat(i).cuda(), K, sc.W, sc.H, deg, with_depth=True)
                return img.cpu().numpy().astype(np.float64), d.cpu().numpy()
            img, _, _ = render(p, sc.viewmat(i).cuda(), K, sc.W, sc.H, deg)
        return img.cpu().numpy().astype(np.float64)

    rows = []
    depth_cache: dict[int, tuple] = {}
    for i in sc.test:
        gt = sc.image(i) / 255.0
        j = nearest_train_index(i, sc.train)
        r3, d3 = rend(params, i, sh, depth=True)
        r0, d0 = rend(p0, i, 0, depth=True)
        preds = {"3dgs": r3, "init_only": r0, "copy_nearest": sc.image(j) / 255.0}
        row = {"frame": i, "nearest_train": j}
        keep = ~aux.mask(i, "moving") if has_masks else None
        for k, pr in preds.items():
            row[k] = {"psnr": round(psnr(pr, gt), 4), "ssim": round(ssim(pr, gt), 4), "lpips": lp(lpm, pr, gt)}
            if keep is not None:
                row[k]["static"] = {"psnr": round(psnr_masked(pr, gt, keep), 4),
                                    "ssim": round(ssim_masked(pr, gt, keep), 4), "lpips": lp(lpm_sp, pr, gt, keep)}
        if keep is not None:
            row["moving_px_frac"] = round(float(1 - keep.mean()), 5)
        if has_depth:
            depth_cache[i] = (aux.eval_depth(i), {"3dgs": d3, "init_only": d0}, keep)
        rows.append(row)
        cv2.imwrite(str(run / "renders" / f"test_{i:04d}.png"), (preds["3dgs"][:, :, ::-1] * 255).round().astype(np.uint8))
        print(i, {k: row[k]["psnr"] for k in preds})

    train_rows = []
    for i in sc.train[::7]:
        gt = sc.image(i) / 255.0
        pr = rend(params, i, sh)
        train_rows.append({"frame": i, "3dgs": {"psnr": round(psnr(pr, gt), 4), "ssim": round(ssim(pr, gt), 4)}})

    depth = None
    if has_depth:
        depth = {"leak_free": leak_free, "tol_m": 0.5, "range_m": [0.5, 80.0],
                 "gt": "held-out frame's reserved lidar sweep (time-nearest, never used in training), z-depth",
                 "pooling": "all lidar pixels of the 20 held-out frames pooled"}
        for k in ("3dgs", "init_only"):
            for region in ("all", "static"):
                P, G = [], []
                for i in sc.test:
                    gd, pd, keep = depth_cache[i]
                    v = gd > 0
                    if region == "static" and keep is not None:
                        v &= keep
                    P.append(pd[k][v])
                    G.append(gd[v])
                P, G = np.concatenate(P), np.concatenate(G)
                e = rnd(depth_errors(P, G))
                e["by_range"] = {f"{lo}-{hi}m": rnd(depth_errors(P, np.where((G >= lo) & (G < hi), G, 0)))
                                 for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 80))}
                depth.setdefault(k, {})[region] = e
        for r in rows:
            gd, pd, keep = depth_cache[r["frame"]]
            r["depth"] = {k: rnd(depth_errors(pd[k], gd)) for k in ("3dgs", "init_only")}

    def summarise_static(key: str) -> dict:
        out = {}
        for m in ("psnr", "ssim", "lpips"):
            vals = [r[key]["static"][m] for r in rows if r[key]["static"].get(m) is not None]
            if vals:
                out[m] = round(float(np.mean(vals)), 4)
        return out

    meta = sc.meta
    res = {
        "date": "2026-09-26",
        "dataset": "Argoverse 2 Sensor Dataset (val split)",
        "log_id": Path(meta["log_dir"]).name,
        "camera": meta["camera"],
        "n_frames": len(meta["frames"]), "n_train": len(sc.train), "n_test": len(sc.test),
        "holdout_every": meta["holdout_every"], "resolution": [sc.W, sc.H],
        "clip_duration_s": round(meta["duration_s"], 2), "trajectory_length_m": round(meta["trajectory_length_m"], 1),
        "init": meta["init"],
        "train_stride": sc.train_stride,
        "train": {k: stats.get(k, 0 if k == "seed" else None) for k in ("steps", "seed", "sh_degree", "n_gaussians_init", "n_gaussians_final", "train_wall_s",
                                        "vram_peak_allocated_mb", "vram_peak_reserved_mb", "gpu", "torch", "gsplat")},
        "train_throughput": throughput(stats.get("log", [])),
        "heldout_mean": {k: summarise(rows, k) for k in ("3dgs", "init_only", "copy_nearest")},
        "mask": stats.get("mask", "none"), "depth_lambda": stats.get("depth_lambda", 0.0),
        "lidar_policy": (sc.lidar_split or {}).get("policy", "unknown"),
        "train_views_mean_3dgs": summarise(train_rows, "3dgs"),
        "n_train_views_evaluated": len(train_rows),
        "lpips": "alex (lpips 0.1.4)" if lpm is not None else None,
        "per_frame": rows,
    }
    res["heldout_3dgs_beats_copy_nearest"] = sum(r["3dgs"]["psnr"] > r["copy_nearest"]["psnr"] for r in rows)
    if has_masks:
        res["heldout_static_mean"] = {k: summarise_static(k) for k in ("3dgs", "init_only", "copy_nearest")}
        res["heldout_static_3dgs_beats_copy_nearest"] = sum(
            r["3dgs"]["static"]["psnr"] > r["copy_nearest"]["static"]["psnr"] for r in rows)
        mk = aux.meta["masks"]
        res["static_mask"] = {"source": f"AV2 cuboids with speed > {mk['speed_thresh_mps']} m/s, padded {mk['pad_m']} m",
                              "heldout_moving_px_frac_mean": round(float(np.mean([r["moving_px_frac"] for r in rows])), 5),
                              "heldout_frames_with_moving": int(sum(r["moving_px_frac"] > 0 for r in rows))}
    if depth is not None:
        res["heldout_depth"] = depth
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(res, indent=1, default=lambda o: o.item()))
    print(json.dumps({k: res[k] for k in ("heldout_mean", "train_views_mean_3dgs", "heldout_3dgs_beats_copy_nearest")}, indent=1, default=lambda o: o.item()))

    if args.figs:
        figs = Path(args.figs)
        figs.mkdir(parents=True, exist_ok=True)
        pick = [sc.test[k] for k in np.linspace(0, len(sc.test) - 1, args.n_figs).round().astype(int)]
        for i in pick:
            gt = sc.image(i)
            r3 = cv2.imread(str(run / "renders" / f"test_{i:04d}.png"))[:, :, ::-1]
            cp = sc.image(nearest_train_index(i, sc.train))
            row = next(r for r in rows if r["frame"] == i)
            tiles = []
            for im, lab in ((gt, "held-out GT"), (r3, f"3DGS {row['3dgs']['psnr']:.2f} dB"),
                            (cp, f"copy-nearest {row['copy_nearest']['psnr']:.2f} dB")):
                t = np.ascontiguousarray(im[:, :, ::-1])
                cv2.rectangle(t, (0, 0), (sc.W, 22), (0, 0, 0), -1)
                cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(t)
            cv2.imwrite(str(figs / f"heldout_{i:04d}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
