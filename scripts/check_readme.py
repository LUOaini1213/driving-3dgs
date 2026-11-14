"""Keep README numbers tied to results JSON.

The README contains generated blocks delimited by <!-- BEGIN:<tag> --> ... <!-- END:<tag> -->:
  setup     data / split / hardware          (results/data_manifest.json + results/runs/*.json)
  results   original runs vs baselines        (runs whose lidar_policy is legacy)
  ablation  strict-lidar runs: masks / depth  (runs whose lidar_policy is strict)
  depth     held-out depth error vs lidar     (heldout_depth of every run)
  web       splat web export + headless check (results/web_export.json, results/viewer_check.json)
  offpath   laterally shifted views           (results/offpath.json)
  findings  sentences whose numbers are differences computed from the JSON above
  tldr      the first-screen summary (numbers from the run JSONs and results/depth_baselines.json)

  python scripts/check_readme.py --check   # exit 1 if README differs from what the JSON implies
  python scripts/check_readme.py --write   # regenerate the blocks in place
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d3gs.web_asset import verify_manifest, verify_browser_report  # noqa: E402
from d3gs.report_keys import offset_key  # noqa: E402
RUN_ORDER = ["full_7k", "full_30k", "sparse4_7k", "strict_7k", "strict_7k_seed1", "strict_mask_7k",
             "strict_depth_7k", "strict_mask_depth_7k"]
METHOD_NAME = {"3dgs": "3DGS（本项目）", "copy_nearest": "基线：复制时间最近的训练帧", "init_only": "基线：仅初始化（0 次迭代）"}
MASK_NAME = {"none": "—", "moving": "移动物体", "vehicles": "全部车辆"}
MAX_SPLAT_BYTES = 20_000_000
VIEWER_URL = "https://luoaini1213.github.io/driving-3dgs/"
BASELINE = "lidar_nearest_train_sweep"


def load_json(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_runs() -> dict[str, dict]:
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "results" / "runs").glob("*.json"))}
    return {k: runs[k] for k in RUN_ORDER if k in runs} | {k: v for k, v in runs.items() if k not in RUN_ORDER}


def is_strict(r: dict) -> bool:
    return r.get("lidar_policy") == "strict"


def metric_number(value):
    if value in ("+Infinity", "Infinity", "-Infinity"):
        return float(value)
    return value


def difference(a, b):
    a, b = metric_number(a), metric_number(b)
    if a is None or b is None:
        return None
    value = a - b
    return None if math.isnan(value) else value


def f(x, nd):
    x = metric_number(x)
    if x is None or math.isnan(x):
        return "—"
    if math.isinf(x):
        return "+∞" if x > 0 else "−∞"
    return f"{x:.{nd}f}"


def sgn(x, nd):
    """Signed difference; tiny values print as ±0 instead of -0.000."""
    x = metric_number(x)
    if x is None or not math.isfinite(x):
        return f(x, nd)
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
    L = ["| 运行 | 初始化/监督是否用到留出帧的评测扫描 | 深度监督 λ | 方法 | 中位绝对误差 (m)↓ 全部 / 静态 | ≤ 0.5 m 占比↑ 全部 / 静态 | "
         "按激光距离分段的中位误差 (m，静态) 0–10 / 10–20 / 20–40 / 40–80 m | 激光点像素数 |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        d = r.get("heldout_depth")
        if not d:
            continue
        leak = {True: "否（严格）", False: "**是（泄漏，仅作对照）**", None: "未知"}[d["leak_free"]]
        lam = f(r.get("depth_lambda", 0.0), 2) if r.get("depth_lambda") else "—"
        L.append(depth_row(f"`{name}`", leak, lam, "3DGS 渲染深度", d["3dgs"]))
    # Every run also renders its initial Gaussians; runs sharing an initialisation give identical rows.
    groups: dict[str, list[str]] = {}
    for name, r in runs.items():
        d = r.get("heldout_depth")
        if d and "init_only" in d:
            groups.setdefault(json.dumps([d["init_only"], d["leak_free"]], sort_keys=True), []).append(name)
    for names in groups.values():
        d = runs[names[0]]["heldout_depth"]
        leak = {True: "否（严格）", False: "**是（泄漏，仅作对照）**", None: "未知"}[d["leak_free"]]
        label = f"`{names[0]}`" if len(names) == 1 else (f"`{names[0]}`、`{names[1]}`" if len(names) == 2
                                                          else f"`{names[0]}` 等 {len(names)} 个同初始化运行")
        L.append(depth_row(label, leak, "—", "基线：仅初始化（0 次迭代）", d["init_only"]))
    b = load_json("results/depth_baselines.json")
    if b:
        base = b[BASELINE]
        L.append(depth_row("—", "否（严格）", "—", "基线：只用激光（最近训练扫描重投影）", base))
        L += ["", f"- 只用激光的基线（`scripts/depth_baseline.py`，CPU）：每个留出帧取时间最近的**训练**扫描（严格初始化或深度监督用过的扫描，"
                  f"不含评测扫描；与留出帧的时间差中位 {base['abs_dt_ms_median']:.0f} ms），投影到留出帧相机，空像素取图像上最近的投影点深度；"
                  f"评测像素中 {pct(base['direct_hit_frac_all'])} 有直接投影点，其余靠填充。评测像素集合与 3DGS 行相同，"
                  "脚本先用同一投影代码逐位复现全部 `eval_depth`。"]
    else:
        L.append("")
    L.append("- 原始运行 JSON 里 `heldout_depth.gt` 的旧描述写着评测扫描 “never used in training”，那段文字早于泄漏发现；"
             "是否真正隔离以本表第二列（`leak_free`）为准，`full_*` / `sparse4_*` 的初始化读过这些扫描。")
    return "\n".join(L)


def depth_row(label: str, leak: str, lam: str, method: str, d: dict) -> str:
    bins = ("0-10m", "10-20m", "20-40m", "40-80m")
    a, s = d["all"], d["static"]
    return (f"| {label} | {leak} | {lam} | {method} | {f(a['median_abs_m'], 2)} / {f(s['median_abs_m'], 2)} | "
            f"{pct(a['within_tol'])} / {pct(s['within_tol'])} | "
            f"{' / '.join(f(s.get('by_range', {}).get(x, {}).get('median_abs_m'), 2) for x in bins)} | {a['n']:,} |")


def render_web() -> str:
    w, v = load_json("results/web_export.json"), load_json("results/viewer_check.json")
    if not w:
        return "（尚未导出）"
    p = w["heldout_psnr_mean"]
    full_key = next((k for k in p if k.startswith("full_sh") and k != "full_sh0"), "full_sh0")
    decoded = "decoded_asset_sh0" in p
    subset_key = "decoded_asset_sh0" if decoded else "web_subset_sh0"
    subset_label = "实际导出字节解码后的 SH0 参数" if decoded else "历史量化前筛选子集（未计入 RGBA/四元数量化）"
    L = [f"- 资产：`{w['file']}`，{w['n_exported']:,} / {w['n_gaussians_trained']:,} 个高斯（不透明度 ≥ {w['min_opacity']}，"
         f"按 不透明度×投影面积 取前 {w['max_splats']:,} 个），{w['bytes'] / 1e6:.1f} MB（每个 {w['bytes_per_splat']} 字节）",
         f"- 同一批留出帧上用 gsplat 渲染的 PSNR：完整模型 {f(p[full_key], 2)} dB → 只保留 0 阶球谐（.splat 格式只存视角无关颜色）"
         f"{f(p['full_sh0'], 2)} dB → {subset_label} {f(p[subset_key], 2)} dB；这些均不是浏览器渲染器的 PSNR",
         f"- 渲染器：{w['renderer']}；初始视角为留出帧 {w['initial_camera_frame']} 的相机"]
    if any(metric_number(value) == math.inf for value in p.values()):
        L.append("- +∞ 表示平均 PSNR 为正无穷（至少一个评测帧的 MSE 为零），不表示所有帧均完全匹配；JSON 用字符串 `+Infinity` 保留该值。")
    if w.get('viewer_check'):
        current = verify_browser_report(ROOT, w)
        cs = current['canvas_stats']
        L.append(f"- 本次绑定的无头 Chromium 检查（{current['date']}）：通过；加载并渲染前 10 帧用时 "
                 f"{current['load_and_first_frames_s']} s，控制台错误 {len(current['console_errors'])} 条，"
                 f"警告 {len(current['console_warnings'])} 条，失败请求 {len(current['failed_requests'])} 个；"
                 f"画布非背景像素 {pct(cs['nonbackground_frac'])}；报告 `{w['viewer_check']['file']}`，"
                 f"截图 `{current['screenshot']['file']}`。报告 SHA256、实际响应资产/页面字节和截图均已核验。")
    c = load_json("results/web_export_candidates.json")
    if c:
        L.append("- 历史候选比较（量化前筛选子集，用 gsplat 在留出帧上评测）："
                 + "；".join(f"`{d['run']}` 保留 {d['n_exported']:,} 个 / {d['bytes'] / 1e6:.1f} MB → {f(d['heldout_psnr_mean']['web_subset_sh0'], 2)} dB"
                            for d in c["candidates"]))
    if v:
        cs = v.get("canvas_stats") or {}
        L.append(f"- 历史无头 Chromium（{v.get('date', '日期未记录')}；{' '.join(v['chromium_args'])}）检查：{'通过' if v['ok'] else '**未通过**'}；"
                 f"加载并渲染前 10 帧用时 {v['load_and_first_frames_s']} s，控制台错误 {len(v['console_errors'])} 条，"
                 f"失败请求 {len(v['failed_requests'])} 个；画布非背景像素 {pct(cs.get('nonbackground_frac'))}；"
                 f"WebGL 后端 `{cs.get('webgl')}`；截图 `{v['screenshot']}`。旧记录未绑定资产/页面 SHA256，不证明当前页面通过检查。")
    L.append("- 当前 `docs/splat/asset_manifest.json` 仅绑定资产原始字节及页面 UTF-8/LF 内容；内容校验不等于新的训练、评测或浏览器验收。")
    return "\n".join(L)


def render_offpath() -> str:
    o = load_json("results/offpath.json")
    if not o:
        return "（尚未运行）"
    offs = [offset_key(x) for x in o["offsets_m"]]
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
        return difference(runs[a][region]["3dgs"][key], runs[b][region]["3dgs"][key])

    if g("full_30k") and g("full_7k"):
        a, b = runs["full_30k"], runs["full_7k"]
        L.append(f"- 30k vs 7k（同一初始化）：留出 PSNR {sgn(dp('full_30k', 'full_7k'), 2)} dB，LPIPS "
                 f"{sgn(difference(a['heldout_mean']['3dgs']['lpips'], b['heldout_mean']['3dgs']['lpips']), 3)}；训练视角 PSNR "
                 f"{sgn(difference(a['train_views_mean_3dgs']['psnr'], b['train_views_mean_3dgs']['psnr']), 2)} dB；训练用时 ×{a['train']['train_wall_s'] / b['train']['train_wall_s']:.1f}，"
                 f"高斯数 ×{a['train']['n_gaussians_final'] / b['train']['n_gaussians_final']:.1f}")
    if g("strict_7k") and g("strict_7k_seed1"):
        a, b = dp('strict_7k_seed1', 'strict_7k'), dp('strict_7k_seed1', 'strict_7k', region='heldout_static_mean')
        suffix = "小于这个量级的差异不当作效果" if a is not None and b is not None else "部分差值未知，不能据此判断效果"
        L.append(f"- 同配置换种子（`strict_7k` vs `strict_7k_seed1`）：全图 PSNR 差 {f(abs(a) if a is not None else None, 2)} dB，"
                 f"静态 PSNR 差 {f(abs(b) if b is not None else None, 2)} dB——{suffix}")
    if g("strict_mask_7k") and g("strict_7k"):
        L.append(f"- 排除移动物体像素（`strict_mask_7k` − `strict_7k`）：全图 PSNR {sgn(dp('strict_mask_7k', 'strict_7k'), 2)} dB，"
                 f"静态 PSNR {sgn(dp('strict_mask_7k', 'strict_7k', region='heldout_static_mean'), 2)} dB，"
                 f"静态 LPIPS {sgn(dp('strict_mask_7k', 'strict_7k', 'lpips', 'heldout_static_mean'), 3)}")
    if g("strict_depth_7k") and g("strict_7k"):
        a, b = runs["strict_depth_7k"]["heldout_depth"]["3dgs"], runs["strict_7k"]["heldout_depth"]["3dgs"]
        L.append(f"- 激光深度监督（`strict_depth_7k` − `strict_7k`）：留出帧深度中位误差 {f(b['static']['median_abs_m'], 2)} → "
                 f"{f(a['static']['median_abs_m'], 2)} m（静态像素），≤ 0.5 m 占比 {pct(b['static']['within_tol'])} → {pct(a['static']['within_tol'])}；"
                 f"全图 PSNR {sgn(dp('strict_depth_7k', 'strict_7k'), 2)} dB，LPIPS {sgn(dp('strict_depth_7k', 'strict_7k', 'lpips'), 3)}")
    b = load_json("results/depth_baselines.json")
    if g("strict_depth_7k") and b:
        a, c = runs["strict_depth_7k"]["heldout_depth"]["3dgs"]["static"], b[BASELINE]["static"]

        def near(x, k="0-10m"):
            return f(x["by_range"][k]["median_abs_m"], 2)
        L.append(f"- 深度监督模型 vs 只用激光的基线（`strict_depth_7k` / 最近训练扫描重投影，静态像素）：中位误差 "
                 f"{f(a['median_abs_m'], 2)} / {f(c['median_abs_m'], 2)} m，≤ 0.5 m 占比 {pct(a['within_tol'])} / {pct(c['within_tol'])}；"
                 f"0–10 m {near(a)} / {near(c)} m，20–40 m {near(a, '20-40m')} / {near(c, '20-40m')} m，"
                 f"40–80 m {near(a, '40-80m')} / {near(c, '40-80m')} m")
    if g("strict_7k") and "init_only" in runs["strict_7k"].get("heldout_depth", {}):
        a, c = runs["strict_7k"]["heldout_depth"]["init_only"]["static"], runs["strict_7k"]["heldout_depth"]["3dgs"]["static"]
        L.append(f"- 只用图像训练相对仅初始化（`strict_7k`，静态像素）：深度中位误差 {f(a['median_abs_m'], 2)} → {f(c['median_abs_m'], 2)} m，"
                 f"≤ 0.5 m 占比 {pct(a['within_tol'])} → {pct(c['within_tol'])}")
    if g("full_7k") and g("strict_7k") and "heldout_depth" in runs["full_7k"]:
        a, b = runs["full_7k"]["heldout_depth"]["3dgs"]["static"], runs["strict_7k"]["heldout_depth"]["3dgs"]["static"]
        L.append(f"- 初始化是否含评测扫描（`full_7k` 含 / `strict_7k` 不含；两者初始化只差这 20 帧扫描）：留出帧深度中位误差 {f(a['median_abs_m'], 2)} / "
                 f"{f(b['median_abs_m'], 2)} m，≤ 0.5 m 占比 {pct(a['within_tol'])} / {pct(b['within_tol'])}；"
                 f"全图 PSNR {sgn(dp('full_7k', 'strict_7k'), 2)} dB（前者减后者）")
    if any(re.search(r"(?<!—)—(?!—)", line) for line in L):
        L.append("- — 表示无有效支持或差值未定义（例如 +∞ − +∞），不代表零差异。")
    return "\n".join(L) if L else "（尚无）"


def render_tldr(runs: dict[str, dict]) -> str:
    """First-screen summary; every number comes from the same JSON as the tables below."""
    need = ("full_7k", "full_30k", "strict_7k", "strict_7k_seed1", "strict_mask_7k", "strict_depth_7k")
    b = load_json("results/depth_baselines.json")
    if any(k not in runs for k in need) or not b:
        return "（结果 JSON 不全，无法生成摘要）"
    r7, r30 = runs["full_7k"], runs["full_30k"]
    h7, c7, h30 = r7["heldout_mean"]["3dgs"], r7["heldout_mean"]["copy_nearest"], r30["heldout_mean"]["3dgs"]
    s0 = runs["strict_7k"]["heldout_depth"]["3dgs"]["static"]
    sd = runs["strict_depth_7k"]["heldout_depth"]["3dgs"]["static"]
    sb = b[BASELINE]["static"]

    def dpsnr(a, c="strict_7k", region="heldout_mean"):
        return difference(runs[a][region]["3dgs"]["psnr"], runs[c][region]["3dgs"]["psnr"])

    seed = dpsnr("strict_7k_seed1", region="heldout_static_mean")
    m = dpsnr("strict_mask_7k", region="heldout_static_mean")
    return "\n".join([
        f"- **What / 做了什么**：3D Gaussian Splatting (gsplat) on one {r7['clip_duration_s']} s Argoverse 2 front-camera clip "
        f"({r7['n_frames']} frames, {r7['n_test']} held out), scored against copy-nearest-frame and init-only baselines; "
        f"[live splat viewer / 在线查看器]({VIEWER_URL})。用一段前视视频训练 3DGS，在留出帧上与基线比较。",
        f"- **Images / 图像**：held-out PSNR {f(h7['psnr'], 2)} dB (7k) / {f(h30['psnr'], 2)} dB (30k) vs {f(c7['psnr'], 2)} dB "
        f"for copying the nearest training frame ({r7['heldout_3dgs_beats_copy_nearest']}/{r7['n_test']} frames higher); "
        f"at 7k the copy baseline still wins LPIPS ({f(c7['lpips'], 3)} vs {f(h7['lpips'], 3)})。7k 的 LPIPS 仍是复制基线更好。",
        f"- **Depth / 深度**：lidar depth supervision cuts held-out depth error {f(s0['median_abs_m'], 2)} → {f(sd['median_abs_m'], 2)} m "
        f"(static median) at {sgn(dpsnr('strict_depth_7k'), 2)} dB PSNR, but a lidar-only baseline (nearest training sweep re-projected) "
        f"also gets {f(sb['median_abs_m'], 2)} m (≤ 0.5 m: model {pct(sd['within_tol'])} vs baseline {pct(sb['within_tol'])}), so most of the gain "
        "is lidar geometry absorbed by the model。深度监督的增益大部分可由激光本身解释。",
        f"- **Masks / 掩膜**：excluding moving-object pixels changes static PSNR by "
        f"{sgn(m, 2)} dB "
        f"vs seed-to-seed {f(abs(seed) if seed is not None else None, 2)} dB: "
        + ("no measurable effect。移动物体掩膜没有可测收益。" if seed is not None and m is not None and abs(m) <= abs(seed)
           else "compare with the ablation table。见消融表。"),
        f"- **Limits / 局限**：one {r7['clip_duration_s']} s clip, one camera, {r7['resolution'][0]}×{r7['resolution'][1]}, one run per "
        "config (plus one seed repeat), floaters off the trajectory; data CC BY-NC-SA 4.0, non-commercial。"
        "单段、单相机、低分辨率、每配置一次运行；数据仅限非商用。",
    ])


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
                      ("findings", render_findings(runs)), ("tldr", render_tldr(runs))):
        new = replace_block(new, tag, body)
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
        try:
            manifest = verify_manifest(p.parent)
            if manifest['asset'] != p.name:
                raise ValueError('web report and asset manifest name disagree')
        except (ValueError, OSError, KeyError) as exc:
            raise SystemExit(f"web content binding failed: {exc}") from exc
    if args.write:
        readme.write_text(new, encoding="utf-8")
        print("README blocks regenerated")
        return
    if new != cur:
        import difflib
        sys.stdout.writelines(difflib.unified_diff(cur.splitlines(True), new.splitlines(True), "README.md", "from JSON"))
        print("\nREADME and results JSON DIVERGE", file=sys.stderr)
        raise SystemExit(1)
    print(f"README matches results JSON ({len(runs)} runs)")


if __name__ == "__main__":
    main()
