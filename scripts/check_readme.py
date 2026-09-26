"""Keep README numbers tied to results JSON.

The README contains generated blocks delimited by
  <!-- BEGIN:results --> ... <!-- END:results -->
  <!-- BEGIN:setup -->   ... <!-- END:setup -->
This script renders them from results/runs/*.json (+ results/data_manifest.json).

  python scripts/check_readme.py --check   # exit 1 if README differs from what the JSON implies
  python scripts/check_readme.py --write   # regenerate the blocks in place
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ORDER = ["full_7k", "full_30k", "sparse4_7k"]
METHOD_NAME = {"3dgs": "3DGS（本项目）", "copy_nearest": "基线：复制时间最近的训练帧", "init_only": "基线：仅初始化（0 次迭代）"}


def load_runs() -> dict[str, dict]:
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "results" / "runs").glob("*.json"))}
    return {k: runs[k] for k in RUN_ORDER if k in runs} | {k: v for k, v in runs.items() if k not in RUN_ORDER}


def f(x, nd):
    return "—" if x is None else f"{x:.{nd}f}"


def render_results(runs: dict[str, dict]) -> str:
    L = ["| 运行 | 训练帧 | 迭代 | 方法 | 留出帧 PSNR↑ | SSIM↑ | LPIPS↓ | 20 帧中胜过复制基线 |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        for m in ("3dgs", "copy_nearest", "init_only"):
            h = r["heldout_mean"][m]
            beats = f"{r['heldout_3dgs_beats_copy_nearest']}/{r['n_test']}" if m == "3dgs" else ""
            L.append(f"| `{name}` | {r['n_train']} | {r['train']['steps'] if m == '3dgs' else '—'} | {METHOD_NAME[m]} | "
                     f"{f(h.get('psnr'), 2)} | {f(h.get('ssim'), 3)} | {f(h.get('lpips'), 3)} | {beats} |")
    L += ["", "| 运行 | 训练用时 (s) | 峰值显存 allocated / reserved (MB) | 高斯数 初始 → 最终 | 训练视角 PSNR / SSIM（过拟合参照） |",
          "|---|---|---|---|---|"]
    for name, r in runs.items():
        t = r["train"]
        tv = r["train_views_mean_3dgs"]
        L.append(f"| `{name}` | {t['train_wall_s']:.1f} | {t['vram_peak_allocated_mb']} / {t['vram_peak_reserved_mb']} | "
                 f"{t['n_gaussians_init']:,} → {t['n_gaussians_final']:,} | {tv['psnr']:.2f} / {tv['ssim']:.3f} |")
    return "\n".join(L)


def render_setup(runs: dict[str, dict]) -> str:
    r = next(iter(runs.values()))
    man = json.loads((ROOT / "results" / "data_manifest.json").read_text(encoding="utf-8"))
    t = r["train"]
    return "\n".join([
        f"- 数据：{r['dataset']}，log `{r['log_id']}`，相机 `{r['camera']}`，"
        f"连续 {r['n_frames']} 帧（{r['clip_duration_s']} s，自车轨迹 {r['trajectory_length_m']} m）",
        f"- 下载：{man['n_files']} 个文件，共 {man['bytes'] / 1e6:.1f} MB（{man['num_frames']} 张 JPEG + {man['n_lidar_sweeps']} 帧激光雷达 + 标定/位姿/标注）",
        f"- 分辨率 {r['resolution'][0]}×{r['resolution'][1]}；每 {r['holdout_every']} 帧留出 1 帧 → 训练 {runs.get('full_7k', r)['n_train']} / 留出 {r['n_test']}",
        f"- 初始化：激光雷达点 {r['init']['lidar_points_after_voxel']:,} 个（体素 {r['init']['voxel_m']} m）+ 远景天空壳 {r['init']['sky_points']:,} 个",
        f"- 硬件/软件：{t['gpu']}，torch {t['torch']}，gsplat {t['gsplat']}",
    ])


def replace_block(text: str, tag: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{tag} -->\n)(.*?)(\n<!-- END:{tag} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"README missing block {tag}")
    return pat.sub(lambda m: m.group(1) + body + m.group(3), text)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()
    readme = ROOT / "README.md"
    cur = readme.read_text(encoding="utf-8")
    runs = load_runs()
    new = replace_block(replace_block(cur, "results", render_results(runs)), "setup", render_setup(runs))
    if args.write:
        readme.write_text(new, encoding="utf-8")
        print("README blocks regenerated")
        return
    # every image referenced from README must exist in the repo and be < 300 KB
    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", cur):
        p = ROOT / rel
        if not p.exists() or p.stat().st_size >= 300_000:
            raise SystemExit(f"README image missing or >= 300 KB: {rel}")
    if new != cur:
        import difflib
        sys.stdout.writelines(difflib.unified_diff(cur.splitlines(True), new.splitlines(True), "README.md", "from JSON"))
        print("\nREADME and results JSON DIVERGE", file=sys.stderr)
        raise SystemExit(1)
    else:
        print(f"README matches results JSON ({len(runs)} runs)")


if __name__ == "__main__":
    main()
