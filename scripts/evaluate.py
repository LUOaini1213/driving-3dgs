"""Evaluate a trained run on HELD-OUT frames and compare against baselines.

Methods compared on the same held-out frames (same resolution, same metric code):
  * 3dgs         : trained Gaussians rendered at the held-out pose
  * init_only    : the lidar/sky initial Gaussians rendered before any optimisation
  * copy_nearest : the time-nearest TRAIN image copied as the prediction (trivial baseline)
Also reports the trained model on TRAIN frames (to show the train/test gap).

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
from d3gs.metrics import psnr, ssim  # noqa: E402
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


def lp(model, a: np.ndarray, b: np.ndarray) -> float | None:
    if model is None:
        return None
    ta = torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda() * 2 - 1
    tb = torch.from_numpy(b).permute(2, 0, 1)[None].float().cuda() * 2 - 1
    with torch.no_grad():
        return float(model(ta, tb))


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

    def rend(p, i, deg):
        with torch.no_grad():
            img, _, _ = render(p, sc.viewmat(i).cuda(), K, sc.W, sc.H, deg)
        return img.cpu().numpy().astype(np.float64)

    rows = []
    for i in sc.test:
        gt = sc.image(i) / 255.0
        j = nearest_train_index(i, sc.train)
        preds = {"3dgs": rend(params, i, sh), "init_only": rend(p0, i, 0), "copy_nearest": sc.image(j) / 255.0}
        row = {"frame": i, "nearest_train": j}
        for k, pr in preds.items():
            row[k] = {"psnr": round(psnr(pr, gt), 4), "ssim": round(ssim(pr, gt), 4), "lpips": lp(lpm, pr, gt)}
        rows.append(row)
        cv2.imwrite(str(run / "renders" / f"test_{i:04d}.png"), (preds["3dgs"][:, :, ::-1] * 255).round().astype(np.uint8))
        print(i, {k: row[k]["psnr"] for k in preds})

    train_rows = []
    for i in sc.train[::7]:
        gt = sc.image(i) / 255.0
        pr = rend(params, i, sh)
        train_rows.append({"frame": i, "3dgs": {"psnr": round(psnr(pr, gt), 4), "ssim": round(ssim(pr, gt), 4)}})

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
        "train": {k: stats[k] for k in ("steps", "sh_degree", "n_gaussians_init", "n_gaussians_final", "train_wall_s",
                                        "vram_peak_allocated_mb", "vram_peak_reserved_mb", "gpu", "torch", "gsplat")},
        "heldout_mean": {k: summarise(rows, k) for k in ("3dgs", "init_only", "copy_nearest")},
        "train_views_mean_3dgs": summarise(train_rows, "3dgs"),
        "n_train_views_evaluated": len(train_rows),
        "lpips": "alex (lpips 0.1.4)" if lpm is not None else None,
        "per_frame": rows,
    }
    res["heldout_3dgs_beats_copy_nearest"] = sum(r["3dgs"]["psnr"] > r["copy_nearest"]["psnr"] for r in rows)
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
