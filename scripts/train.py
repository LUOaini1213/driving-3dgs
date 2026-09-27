"""Train 3D Gaussian Splatting (gsplat rasteriser + gsplat DefaultStrategy densification).

Only TRAIN frames are ever loaded here. Writes ckpt.pt and train_stats.json to --out.

Usage:
  python scripts/train.py --work D:/driving-3dgs/work/<log> --out D:/driving-3dgs/runs/<name> --steps 7000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d3gs.lidar import assert_no_lidar_leak  # noqa: E402
from d3gs.scene import Scene, exp_lr, init_params, render, ssim_torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--refine-stop", type=int, default=None, help="default: 0.5 * steps (3DGS paper: 15k of 30k)")
    ap.add_argument("--grow-grad2d", type=float, default=0.0002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask", choices=["none", "moving", "vehicles"], default="none",
                    help="exclude pixels of moving objects (or all vehicles) from the photometric loss")
    ap.add_argument("--depth-lambda", type=float, default=0.0,
                    help=">0: add lambda * L1(rendered expected depth, lidar depth) on train pixels with a lidar return")
    args = ap.parse_args()

    import gsplat
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sc = Scene(args.work)
    init = np.load(sc.dir / "init_points.npz")
    params = init_params(init["xyz"], init["rgb"], args.sh_degree, dev)
    n_init = len(params["means"])

    split = sc.lidar_split
    if args.depth_lambda > 0:
        if split is None or split["policy"] != "strict":
            raise SystemExit("depth supervision needs a work dir prepared with --strict-lidar")
        assert_no_lidar_leak(split["init_sweep_ts"] + split["depth_sweep_ts"], split["eval_sweep_ts"])
    elif split is not None and split["policy"] == "strict":
        assert_no_lidar_leak(split["init_sweep_ts"], split["eval_sweep_ts"])

    imgs = {i: torch.from_numpy(sc.image(i)) for i in sc.train}  # CPU uint8, train only
    masks = ({i: torch.from_numpy(sc.mask(i, args.mask))[..., None] for i in sc.train} if args.mask != "none" else None)
    depths = ({i: torch.from_numpy(sc.train_depth(i)) for i in sc.train} if args.depth_lambda > 0 else None)
    K = sc.K.to(dev)
    viewmats = {i: sc.viewmat(i).to(dev) for i in sc.train}

    s = sc.scene_scale
    lrs = {"means": 1.6e-4 * s, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    refine_stop = args.refine_stop or int(0.5 * args.steps)
    strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=refine_stop, refine_every=100,
                               reset_every=3000, grow_grad2d=args.grow_grad2d, verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=s)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    order: list[int] = []
    log = []
    for step in range(args.steps):
        if not order:
            order = list(rng.permutation(sc.train))
        i = int(order.pop())
        gt = imgs[i].to(dev, non_blocking=True).float() / 255.0
        sh = min(step // 1000, args.sh_degree)
        if depths is not None:
            img, _, info, dep = render(params, viewmats[i], K, sc.W, sc.H, sh, with_depth=True)
        else:
            img, _, info = render(params, viewmats[i], K, sc.W, sc.H, sh)
        strategy.step_pre_backward(params, opts, state, step, info)
        if masks is not None:
            # masked pixels are replaced by the ground truth, so they contribute exactly zero loss/gradient
            m = masks[i].to(dev, non_blocking=True)
            img = torch.where(m, gt, img)
        l1 = (img - gt).abs().mean()
        loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * (1 - ssim_torch(img, gt))
        if depths is not None:
            dl = depths[i].to(dev, non_blocking=True)
            valid = dl > 0
            loss = loss + args.depth_lambda * (dep[valid] - dl[valid]).abs().mean()
        loss.backward()
        opts["means"].param_groups[0]["lr"] = exp_lr(step, args.steps, lrs["means"], lrs["means"] * 0.01)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, opts, state, step, info, packed=False)
        if step % 500 == 0 or step == args.steps - 1:
            with torch.no_grad():
                psnr = float(-10 * torch.log10(((img - gt) ** 2).mean()))
            rec = {"step": step, "loss": round(float(loss), 5), "train_psnr_1img": round(psnr, 2),
                   "n_gauss": len(params["means"]), "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                   "elapsed_s": round(time.time() - t0, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    torch.cuda.synchronize()
    wall = time.time() - t0

    torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "ckpt.pt")
    stats = {
        "steps": args.steps, "sh_degree": args.sh_degree, "ssim_lambda": args.ssim_lambda,
        "refine_stop": refine_stop, "grow_grad2d": args.grow_grad2d, "seed": args.seed,
        "mask": args.mask, "depth_lambda": args.depth_lambda,
        "lidar_policy": split["policy"] if split else "legacy (no lidar_split.json)",
        "n_gaussians_init": n_init, "n_gaussians_final": len(params["means"]),
        "train_wall_s": round(wall, 1),
        "vram_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "vram_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "gsplat": gsplat.__version__,
        "work": str(sc.dir), "scene_scale": s, "n_train": len(sc.train), "train_stride": sc.train_stride, "resolution": [sc.W, sc.H], "log": log,
    }
    (out / "train_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "log"}, indent=1))


if __name__ == "__main__":
    main()
