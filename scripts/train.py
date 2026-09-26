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

    imgs = {i: torch.from_numpy(sc.image(i)) for i in sc.train}  # CPU uint8, train only
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
        img, _, info = render(params, viewmats[i], K, sc.W, sc.H, sh)
        strategy.step_pre_backward(params, opts, state, step, info)
        l1 = (img - gt).abs().mean()
        loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * (1 - ssim_torch(img, gt))
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
