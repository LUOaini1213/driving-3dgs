"""Keep README numbers tied to results JSON.

The README contains generated blocks delimited by <!-- BEGIN:<tag> --> ... <!-- END:<tag> -->:
  setup     data / split / hardware          (results/data_manifest.json + results/runs/*.json)
  results   original runs vs baselines        (runs whose lidar_policy is legacy)
  ablation  strict-lidar runs: masks / depth  (runs whose lidar_policy is strict)
  depth     held-out depth error vs lidar     (heldout_depth of every run)
  web       splat web export + headless check (results/web_export.json, results/viewer_check.json)
  offpath   laterally shifted views           (results/offpath.json)
  findings  sentences whose numbers are differences computed from the JSON above

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
RUN_ORDER = ["full_7k", "full_30k", "sparse4_7k", "strict_7k", "strict_7k_seed1", "strict_mask_7k",
             "strict_depth_7k", "strict_mask_depth_7k"]
METHOD_NAME = {"3dgs": "3DGS（本项目）", "copy_nearest": "基线：复制时间最近的训练帧", "init_only": "基线：仅初始化（0 次迭代）"}
MASK_NAME = {"none": "—", "moving": "移动物体", "vehicles": "全部车辆"}
MAX_SPLAT_BYTES = 20_000_000


def load_json(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_runs() -> dict[str, dict]:
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "results" / "runs").glob("*.json"))}
    return {k: runs[k] for k in RUN_ORDER if k in runs} | {k: v for k, v in runs.items() if k not in RUN_ORDER}


def is_strict(r: dict) -> bool:
    return r.get("lidar_policy") == "strict"


def f(x, nd):
    return "—" if x is None else f"{x:.{nd}f}"


def sgn(x, nd):
    """Signed difference; tiny values print as ±0 instead of -0.000."""
    return f"±{0:.{nd}f}" if abs(x) < 0.5 * 10 ** -nd else f"{x:+.{nd}f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def render_results(runs: dict[str, dict]) -> str:
    legacy = {k: v for k, v in runs.items() if not is_strict(v)}
    L = ["| 运行 | 训练帧 | 迭代 | 方法 | 留出帧 PSNR↑ | SSIM↑ | LPIPS↓ | PSNR 高于复制基线的留出帧数 |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in legacy.items():
        for m in ("3dgs", "copy_nearest", "init_only"):
            h = r["heldout_mean"][m]
            beats = f"{r['heldout_3dgs_beats_copy_nearest']}/{r['n_test']}" if m == "3dgs" else ""
            L.append(f"| `{name}` | {r['n_train']} | {r['train']['steps'] if m == '3dgs' else '—'} | {METHOD_NAME[m]} | "
                     f"{f(h.get('psnr'), 2)} | {f(h.get('ssim'), 3)} | {f(h.get('lpips'), 3)} | {beats} |")
    L += ["", "| 运行 | 训练用时 (s) | 其中停顿多耗 (s)¹ | 峰值显存 allocated / reserved (MB，torch 统计，不含 CUDA 上下文) | 高斯数 初始 → 最终 | 训练视角 PSNR / SSIM（每 7 个训练帧抽 1 帧；过拟合参照） |",
          "|---|---|---|---|---|---|"]
    for name, r in runs.items():
        t = r["train"]
        tv = r["train_views_mean_3dgs"]
        th = r.get("train_throughput") or {}
        stall = f"{th['stall_excess_s']:.0f}（{th['n_stalled_segments']} 段）" if th.get("n_stalled_segments") else ("0" if th else "—")
        L.append(f"| `{name}` | {t['train_wall_s']:.1f} | {stall} | {t['vram_peak_allocated_mb']} / {t['vram_peak_reserved_mb']} | "
                 f"{t['n_gaussians_init']:,} → {t['n_gaussians_final']:,} | {tv['psnr']:.2f} / {tv['ssim']:.3f} |")
    L += ["", "¹ 训练日志每 500 步记一次时间；用时超过中位数 3 倍的段记为停顿，列出这些段超出中位数的总时长。"]
    return "\n".join(L)


def render_ablation(runs: dict[str, dict]) -> str:
    L = ["| 运行 | 训练时排除的像素 | 深度监督 λ | 种子 | 全图 PSNR↑ | 全图 SSIM↑ | 全图 LPIPS↓ | 静态像素 PSNR↑ | 静态 SSIM↑ | 静态 LPIPS↓ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    ref = None
    for name, r in runs.items():
        if "heldout_static_mean" not in r:
            continue
        h, s = r["heldout_mean"]["3dgs"], r["heldout_static_mean"]["3dgs"]
        seed = r["train"].get("seed", 0)
        L.append(f"| `{name}` | {MASK_NAME[r.get('mask', 'none')]} | {f(r.get('depth_lambda', 0.0), 2) if r.get('depth_lambda') else '—'} | {seed} | "
                 f"{f(h['psnr'], 2)} | {f(h['ssim'], 3)} | {f(h['lpips'], 3)} | {f(s['psnr'], 2)} | {f(s['ssim'], 3)} | {f(s['lpips'], 3)} |")
        if is_strict(r) and ref is None:
            ref = (name, r)
    if ref:
        name, r = ref
        h, s = r["heldout_mean"]["copy_nearest"], r["heldout_static_mean"]["copy_nearest"]
        L.append(f"| 基线：复制最近训练帧 | — | — | — | {f(h['psnr'], 2)} | {f(h['ssim'], 3)} | {f(h['lpips'], 3)} | "
                 f"{f(s['psnr'], 2)} | {f(s['ssim'], 3)} | {f(s['lpips'], 3)} |")
        sm = r["static_mask"]
        L += ["", f"- 静态像素 = 不在任何“移动物体”投影内的像素（{sm['source'].replace('AV2 cuboids with speed >', 'AV2 标注长方体，速度 >').replace('m/s, padded', 'm/s，每边外扩')}）；"
                  f"20 个留出帧中 {sm['heldout_frames_with_moving']} 帧有移动物体，移动像素平均占 {pct(sm['heldout_moving_px_frac_mean'])}。"]
    return "\n".join(L)


def render_depth(runs: dict[str, dict]) -> str:
    bins = ("0-10m", "10-20m", "20-40m", "40-80m")
    L = ["| 运行 | 初始化/监督是否用到留出帧的评测扫描 | 深度监督 λ | 方法 | 中位绝对误差 (m)↓ 全部 / 静态 | ≤ 0.5 m 占比↑ 全部 / 静态 | "
         "按激光距离分段的中位误差 (m，静态) 0–10 / 10–20 / 20–40 / 40–80 m | 激光点像素数 |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        d = r.get("heldout_depth")
        if not d:
            continue
        leak = {True: "否（严格）", False: "**是（泄漏，仅作对照）**", None: "未知"}[d["leak_free"]]
        for m, lab in (("3dgs", "3DGS 渲染深度"),):
            a, s = d[m]["all"], d[m]["static"]
            lam = f(r.get("depth_lambda", 0.0), 2) if r.get("depth_lambda") else "—"
            L.append(f"| `{name}` | {leak} | {lam} | {lab} | {f(a['median_abs_m'], 2)} / {f(s['median_abs_m'], 2)} | "
                     f"{pct(a['within_tol'])} / {pct(s['within_tol'])} | "
                     f"{' / '.join(f(s.get('by_range', {}).get(b, {}).get('median_abs_m'), 2) for b in bins)} | {a['n']:,} |")
    return "\n".join(L)


def render_web() -> str:
    w, v = load_json("results/web_export.json"), load_json("results/viewer_check.json")
    if not w:
        return "（尚未导出）"
    p = w["heldout_psnr_mean"]
    full_key = next(k for k in p if k.startswith("full_sh") and k != "full_sh0")
    L = [f"- 资产：`{w['file']}`，{w['n_exported']:,} / {w['n_gaussians_trained']:,} 个高斯（不透明度 ≥ {w['min_opacity']}，"
         f"按 不透明度×投影面积 取前 {w['max_splats']:,} 个），{w['bytes'] / 1e6:.1f} MB（每个 {w['bytes_per_splat']} 字节）",
         f"- 同一批留出帧上用 gsplat 渲染的 PSNR：完整模型 {p[full_key]:.2f} dB → 只保留 0 阶球谐（.splat 格式只存视角无关颜色）"
         f"{p['full_sh0']:.2f} dB → 网页实际加载的子集 {p['web_subset_sh0']:.2f} dB",
         f"- 渲染器：{w['renderer']}；初始视角为留出帧 {w['initial_camera_frame']} 的相机"]
    c = load_json("results/web_export_candidates.json")
    if c:
        L.append("- 选资产时比较过的候选（同一导出代码；网页子集在留出帧上的 PSNR）："
                 + "；".join(f"`{d['run']}` 保留 {d['n_exported']:,} 个 / {d['bytes'] / 1e6:.1f} MB → {d['heldout_psnr_mean']['web_subset_sh0']:.2f} dB"
                            for d in c["candidates"]))
    if v:
        cs = v.get("canvas_stats") or {}
        L.append(f"- 无头 Chromium（{' '.join(v['chromium_args'])}）检查：{'通过' if v['ok'] else '**未通过**'}；"
                 f"加载并渲染前 10 帧用时 {v['load_and_first_frames_s']} s，控制台错误 {len(v['console_errors'])} 条，"
                 f"失败请求 {len(v['failed_requests'])} 个；画布非背景像素 {pct(cs.get('nonbackground_frac'))}；"
                 f"WebGL 后端 `{cs.get('webgl')}`；截图 `{v['screenshot']}`")
    return "\n".join(L)


def render_offpath() -> str:
    o = load_json("results/offpath.json")
    if not o:
        return "（尚未运行）"
    offs = [f"{x:+.0f}" for x in o["offsets_m"]]
    L = ["| 运行 | 指标 | " + " | ".join(f"{x} m" for x in offs) + " |", "|---|---|" + "---|" * len(offs)]
    for name, r in o["runs"].items():
        po = r["per_offset"]
        L.append(f"| `{name}` | 深度中位误差 (m) vs 激光 | " + " | ".join(f(po[x]['depth_median_abs_m'], 2) for x in offs) + " |")
        L.append(f"| `{name}` | 深度 ≤ 0.5 m 占比 | " + " | ".join(pct(po[x]['depth_within_0_5m']) for x in offs) + " |")
        L.append(f"| `{name}` | 空洞像素（α < 0.5） | " + " | ".join(pct(po[x]['hole_frac_mean']) for x in offs) + " |")
    return "\n".join(L)


def render_findings(runs: dict[str, dict]) -> str:
    L = []
    g = runs.get

    def dp(a, b, key="psnr", region="heldout_mean"):
        return runs[a][region]["3dgs"][key] - runs[b][region]["3dgs"][key]

    if g("full_30k") and g("full_7k"):
        a, b = runs["full_30k"], runs["full_7k"]
        L.append(f"- 30k vs 7k（同一初始化）：留出 PSNR {sgn(dp('full_30k', 'full_7k'), 2)} dB，LPIPS "
                 f"{sgn(a['heldout_mean']['3dgs']['lpips'] - b['heldout_mean']['3dgs']['lpips'], 3)}；训练视角 PSNR "
                 f"{sgn(a['train_views_mean_3dgs']['psnr'] - b['train_views_mean_3dgs']['psnr'], 2)} dB；训练用时 ×{a['train']['train_wall_s'] / b['train']['train_wall_s']:.1f}，"
                 f"高斯数 ×{a['train']['n_gaussians_final'] / b['train']['n_gaussians_final']:.1f}")
    if g("strict_7k") and g("strict_7k_seed1"):
        L.append(f"- 同配置换种子（`strict_7k` vs `strict_7k_seed1`）：全图 PSNR 差 {abs(dp('strict_7k_seed1', 'strict_7k')):.2f} dB，"
                 f"静态 PSNR 差 {abs(dp('strict_7k_seed1', 'strict_7k', region='heldout_static_mean')):.2f} dB——小于这个量级的差异不当作效果")
    if g("strict_mask_7k") and g("strict_7k"):
        L.append(f"- 排除移动物体像素（`strict_mask_7k` − `strict_7k`）：全图 PSNR {sgn(dp('strict_mask_7k', 'strict_7k'), 2)} dB，"
                 f"静态 PSNR {sgn(dp('strict_mask_7k', 'strict_7k', region='heldout_static_mean'), 2)} dB，"
                 f"静态 LPIPS {sgn(dp('strict_mask_7k', 'strict_7k', 'lpips', 'heldout_static_mean'), 3)}")
    if g("strict_depth_7k") and g("strict_7k"):
        a, b = runs["strict_depth_7k"]["heldout_depth"]["3dgs"], runs["strict_7k"]["heldout_depth"]["3dgs"]
        L.append(f"- 激光深度监督（`strict_depth_7k` − `strict_7k`）：留出帧深度中位误差 {b['static']['median_abs_m']:.2f} → "
                 f"{a['static']['median_abs_m']:.2f} m（静态像素），≤ 0.5 m 占比 {pct(b['static']['within_tol'])} → {pct(a['static']['within_tol'])}；"
                 f"全图 PSNR {sgn(dp('strict_depth_7k', 'strict_7k'), 2)} dB，LPIPS {sgn(dp('strict_depth_7k', 'strict_7k', 'lpips'), 3)}")
    if g("full_7k") and g("strict_7k") and "heldout_depth" in runs["full_7k"]:
        a, b = runs["full_7k"]["heldout_depth"]["3dgs"]["static"], runs["strict_7k"]["heldout_depth"]["3dgs"]["static"]
        L.append(f"- 初始化是否含评测扫描（`full_7k` 含 / `strict_7k` 不含；两者初始化只差这 20 帧扫描）：留出帧深度中位误差 {a['median_abs_m']:.2f} / "
                 f"{b['median_abs_m']:.2f} m，≤ 0.5 m 占比 {pct(a['within_tol'])} / {pct(b['within_tol'])}；"
                 f"全图 PSNR {sgn(dp('full_7k', 'strict_7k'), 2)} dB（前者减后者）")
    return "\n".join(L) if L else "（尚无）"


def render_setup(runs: dict[str, dict]) -> str:
    r = runs.get("full_7k") or next(iter(runs.values()))
    man = json.loads((ROOT / "results" / "data_manifest.json").read_text(encoding="utf-8"))
    t = r["train"]
    lines = [
        f"- 数据：{r['dataset']}，log `{r['log_id']}`，相机 `{r['camera']}`，"
        f"连续 {r['n_frames']} 帧（{r['clip_duration_s']} s，自车轨迹 {r['trajectory_length_m']} m）",
        f"- 下载：{man['n_files']} 个文件，共 {man['bytes'] / 1e6:.1f} MB（{man['num_frames']} 张 JPEG + {man['n_lidar_sweeps']} 帧激光雷达 + 标定/位姿/标注）",
        f"- 分辨率 {r['resolution'][0]}×{r['resolution'][1]}；每 {r['holdout_every']} 帧留出 1 帧 → 训练 {r['n_train']} / 留出 {r['n_test']}",
        f"- 初始化（原始 `full_*`）：激光雷达点 {r['init']['lidar_points_after_voxel']:,} 个（体素 {r['init']['voxel_m']} m）+ 远景天空壳 {r['init']['sky_points']:,} 个",
    ]
    s = next((v for v in runs.values() if is_strict(v)), None)
    if s:
        lines.append(f"- 初始化（严格 `strict_*`）：去掉留出帧的评测扫描后，激光雷达点 {s['init']['lidar_points_after_voxel']:,} 个 + 天空壳 {s['init']['sky_points']:,} 个")
    lines.append(f"- 硬件/软件：{t['gpu']}，torch {t['torch']}，gsplat {t['gsplat']}")
    return "\n".join(lines)


def replace_block(text: str, tag: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{tag} -->)(.*?)(<!-- END:{tag} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"README missing block {tag}")
    return pat.sub(lambda m: m.group(1) + "\n" + body + "\n" + m.group(3), text)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()
    readme = ROOT / "README.md"
    cur = readme.read_text(encoding="utf-8")
    runs = load_runs()
    new = cur
    for tag, body in (("results", render_results(runs)), ("setup", render_setup(runs)), ("ablation", render_ablation(runs)),
                      ("depth", render_depth(runs)), ("web", render_web()), ("offpath", render_offpath()),
                      ("findings", render_findings(runs))):
        new = replace_block(new, tag, body)
    if args.write:
        readme.write_text(new, encoding="utf-8")
        print("README blocks regenerated")
        return
    # every image referenced from README must exist in the repo and be < 300 KB
    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", cur):
        p = ROOT / rel
        if not p.exists() or p.stat().st_size >= 300_000:
            raise SystemExit(f"README image missing or >= 300 KB: {rel}")
    # the committed web splat asset must stay small and match its JSON record
    w = load_json("results/web_export.json")
    if w:
        p = ROOT / w["file"]
        if not p.exists() or p.stat().st_size != w["bytes"] or p.stat().st_size >= MAX_SPLAT_BYTES:
            raise SystemExit(f"web asset missing, changed since export, or >= 20 MB: {w['file']}")
    if new != cur:
        import difflib
        sys.stdout.writelines(difflib.unified_diff(cur.splitlines(True), new.splitlines(True), "README.md", "from JSON"))
        print("\nREADME and results JSON DIVERGE", file=sys.stderr)
        raise SystemExit(1)
    print(f"README matches results JSON ({len(runs)} runs)")


if __name__ == "__main__":
    main()
