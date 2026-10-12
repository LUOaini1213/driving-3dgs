"""Export trained Gaussians: standard 3DGS PLY, a three.js point-cloud preview, and a web splat viewer.

* gaussians.ply  — the usual INRIA 3DGS layout (x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*),
                   loadable by common splat viewers.
* preview.html   — Gaussian CENTRES drawn as coloured points (not a splat renderer), plus the camera
                   trajectory (train = grey, held-out = red). Data is embedded as base64, so the file
                   opens straight from disk; three.js is loaded from the jsDelivr CDN.
* --splat-dir    — <name>.splat (pruned, SH degree 0; see d3gs/splat.py) + viewer.html using
                   @mkkellogg/gaussian-splats-3d from jsDelivr. Decoded exported bytes are evaluated with
                   gsplat at held-out poses. This is not a browser-renderer PSNR measurement.
                   Historical web_subset_sh0 results used the pre-quantization subset.

Usage:
  python scripts/export.py --run D:/driving-3dgs/runs/full_30k --out D:/driving-3dgs/export/full_30k
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d3gs.metrics import psnr  # noqa: E402
from d3gs.scene import Scene, render, sh0_to_rgb  # noqa: E402
from d3gs.splat import BYTES_PER_SPLAT, encode_splat, select_for_web  # noqa: E402
from d3gs.web_asset import decoded_parameters, make_manifest  # noqa: E402
from d3gs.provenance import verify_run  # noqa: E402


def asset_location(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix(), True
    except ValueError:
        return str(path), False


def psnr_mean_value(values):
    """Standard JSON: retain an infinite PSNR mean without emitting a bare token."""
    value = float(np.mean(values))
    if np.isposinf(value):
        return "+Infinity"
    if not np.isfinite(value):
        raise ValueError("PSNR mean is undefined")
    return round(value, 4)


def write_ply(path: Path, p: dict[str, np.ndarray]) -> None:
    n = len(p["means"])
    dc = p["sh0"].reshape(n, -1)
    rest = p["shN"].transpose(0, 2, 1).reshape(n, -1)  # channel-major, as the INRIA exporter does
    cols = [p["means"], np.zeros((n, 3), np.float32), dc, rest, p["opacities"][:, None], p["scales"], p["quats"]]
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(dc.shape[1])]
             + [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate(cols, 1).astype(np.float32)
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Driving 3DGS preview</title>
<link rel="icon" href="data:,">
<style>
 :root{--bg:#111418;--fg:#e8e8e8;--muted:#9aa3ad}
 html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
 #info{position:fixed;left:12px;top:10px;width:calc(100% - 24px);max-width:560px;box-sizing:border-box;z-index:10;pointer-events:none;background:rgba(12,16,22,.88);padding:10px 12px;border-radius:8px;overflow-wrap:anywhere}
 #info small{color:#d2d9e1} #status{display:block;color:#ffd479} canvas{display:block}
 @media(max-width:480px){#info{font-size:12px;padding:8px}#info small{font-size:11px}}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="info"><b>AV2 driving clip · 3DGS Gaussian centres</b><span id="status" role="status" aria-live="polite">loading dependencies…</span>
<small>__NPTS__ points shown (of __NTOTAL__ Gaussians, opacity &gt; __OPA__) · grey line = train cameras, red dots = held-out cameras ·
drag to orbit, scroll to zoom. Point cloud of centres only, not a splat render. Data © 2021 Argo AI, LLC, CC BY-NC-SA 4.0.</small></div>
<script type="module">
const st=document.getElementById('status');
window.__previewReady=false;
const fail=e=>{window.__previewReady=false;window.__previewError=String(e?.message||e);st.textContent='Preview failed: '+window.__previewError;console.error(e)};
window.addEventListener('error',e=>fail(e.error||e.message));
window.addEventListener('unhandledrejection',e=>fail(e.reason));
try {
const [THREE,{OrbitControls}]=await Promise.all([import('three'),import('three/addons/controls/OrbitControls.js')]);
const b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const D=__DATA__;
const q=new Int16Array(b64(D.pos).buffer), col=b64(D.col), n=col.length/3;
const pos=new Float32Array(n*3);
for(let i=0;i<n;i++)for(let k=0;k<3;k++)pos[3*i+k]=D.lo[k]+(q[3*i+k]+32768)/65535*(D.hi[k]-D.lo[k]);
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x111418);
const cam=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.1,2000); cam.up.set(0,0,1);
const r=new THREE.WebGLRenderer({antialias:true}); r.setPixelRatio(devicePixelRatio); r.setSize(innerWidth,innerHeight);
r.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();fail(new Error('WebGL context lost; reload to retry'))});
document.body.appendChild(r.domElement);
const g=new THREE.BufferGeometry();
g.setAttribute('position',new THREE.BufferAttribute(pos,3));
g.setAttribute('color',new THREE.BufferAttribute(new Float32Array(col).map(v=>v/255),3));
scene.add(new THREE.Points(g,new THREE.PointsMaterial({size:0.06,vertexColors:true})));
const tr=new THREE.BufferGeometry().setFromPoints(D.train.map(p=>new THREE.Vector3(...p)));
scene.add(new THREE.Line(tr,new THREE.LineBasicMaterial({color:0xaaaaaa})));
const te=new THREE.BufferGeometry().setFromPoints(D.test.map(p=>new THREE.Vector3(...p)));
scene.add(new THREE.Points(te,new THREE.PointsMaterial({color:0xff4040,size:0.6})));
const c0=D.train[0], c1=D.train[D.train.length-1];
cam.position.set(c0[0]-15,c0[1]-15,c0[2]+25);
const ctl=new OrbitControls(cam,r.domElement); ctl.target.set((c0[0]+c1[0])/2,(c0[1]+c1[1])/2,(c0[2]+c1[2])/2); ctl.update();
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();r.setSize(innerWidth,innerHeight)});
(function loop(){if(window.__previewError)return;try{ctl.update();r.render(scene,cam);window.__previewReady=true;st.textContent='Ready · point-cloud preview';requestAnimationFrame(loop)}catch(e){fail(e)}})();
} catch(e) { fail(e); }
</script></body></html>
"""


VIEWER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Driving 3DGS viewer</title>
<link rel="icon" href="data:,">
<style>
 :root{--bg:#0e1013;--fg:#e8e8e8;--muted:#9aa3ad}
 html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;overflow:hidden}
 #info{position:fixed;left:12px;top:10px;width:calc(100% - 24px);max-width:560px;box-sizing:border-box;z-index:10;pointer-events:none;background:rgba(12,16,22,.88);padding:10px 12px;border-radius:8px;overflow-wrap:anywhere}
 #info small{color:#d2d9e1} #status{display:block;color:#ffd479}
 @media(max-width:480px){#info{font-size:12px;padding:8px}#info small{font-size:11px}}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"@mkkellogg/gaussian-splats-3d":"https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"}}</script>
</head><body>
<div id="info"><b>AV2 driving clip &middot; 3D Gaussian Splatting (__RUN__)</b> <span id="status" role="status" aria-live="polite">loading __MB__ MB&hellip;</span>
<small>__N__ of __NTOTAL__ trained Gaussians (pruned for the web, view-independent colour only) &middot;
starts at held-out camera __FRAME__ &middot; drag to orbit, right-drag to pan, scroll to zoom.
Renderer: @mkkellogg/gaussian-splats-3d. Data &copy; 2021 Argo AI, LLC, CC BY-NC-SA 4.0 (non-commercial).</small></div>
<script type="module">
const st=document.getElementById('status');
window.__splatReady=false;
const fail=e=>{window.__splatReady=false;window.__splatError=String(e?.message||e);st.textContent='Viewer failed: '+window.__splatError;console.error(e)};
window.addEventListener('error',e=>fail(e.error||e.message));
window.addEventListener('unhandledrejection',e=>fail(e.reason));
try {
const GaussianSplats3D=await import('@mkkellogg/gaussian-splats-3d');
const viewer=new GaussianSplats3D.Viewer({
  cameraUp:[0,0,1], initialCameraPosition:__POS__, initialCameraLookAt:__LOOK__,
  sharedMemoryForWorkers:false, gpuAcceleratedSort:false, integerBasedSort:false,
  sceneRevealMode:GaussianSplats3D.SceneRevealMode.Instant, antialiased:true});
window.__viewer=viewer;
await viewer.addSplatScene(__FILE_JSON__,{format:GaussianSplats3D.SceneFormat.Splat,splatAlphaRemovalThreshold:0,
  showLoadingUI:false,progressiveLoad:false});
viewer.renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();fail(new Error('WebGL context lost; reload to retry'))});
viewer.start();
for(let n=0;n<10;n++){await new Promise(requestAnimationFrame);if(window.__splatError)throw new Error(window.__splatError);viewer.update();viewer.render();}
if(!window.__splatError){window.__splatReady=true;st.textContent='Ready · drag to orbit';}
} catch(e) { fail(e); }
</script></body></html>
"""


def export_web(args, ck: dict, sc: Scene, out_dir: Path, name: str, sh_degree: int, *, device="cuda") -> dict:
    """Evaluate decoded .splat bytes with gsplat, not the browser renderer."""
    import cv2  # noqa: F401  (Scene.image)
    opa = torch.sigmoid(torch.from_numpy(ck["opacities"])).numpy()
    idx = select_for_web(opa, ck["scales"], args.splat_max, args.splat_min_opacity)
    data = encode_splat(ck["means"][idx], ck["scales"][idx], ck["quats"][idx], ck["opacities"][idx], ck["sh0"][idx])
    decoded = decoded_parameters(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{name}.splat"
    (out_dir / fn).write_bytes(data)
    # initial camera = a held-out camera in the middle of the clip, looking along its optical axis
    fr = sc.test[len(sc.test) // 2]
    c2w = sc.c2w[fr].numpy()
    pos, look = c2w[:3, 3], c2w[:3, 3] + 10 * c2w[:3, 2]
    html = (VIEWER.replace("__FILE_JSON__", json.dumps(fn)).replace("__RUN__", name).replace("__N__", f"{len(idx):,}")
            .replace("__NTOTAL__", f"{len(opa):,}").replace("__MB__", f"{len(data) / 1e6:.1f}")
            .replace("__FRAME__", str(fr)).replace("__POS__", json.dumps(np.round(pos, 3).tolist()))
            .replace("__LOOK__", json.dumps(np.round(look, 3).tolist())))
    (out_dir / "viewer.html").write_text(html, encoding="utf-8", newline="\n")
    binding = make_manifest(out_dir / fn, out_dir / "viewer.html")
    (out_dir / "asset_manifest.json").write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Reconstruct quantized RGBA, quaternion and scale values from the actual bytes.
    dev = device
    full = {k: torch.from_numpy(v).to(dev) for k, v in ck.items()}
    sub = {k: torch.from_numpy(v).to(dev) for k, v in decoded.items()}
    K = sc.K.to(dev)
    res = {"full_sh%d" % sh_degree: [], "full_sh0": [], "decoded_asset_sh0": []}
    for i in sc.test:
        gt = sc.image(i) / 255.0
        variants = [("full_sh%d" % sh_degree, full, sh_degree), ("decoded_asset_sh0", sub, 0)]
        if sh_degree != 0:
            variants.insert(1, ("full_sh0", full, 0))
        for key, p, deg in variants:
            with torch.no_grad():
                img, _, _ = render(p, sc.viewmat(i).to(dev), K, sc.W, sc.H, deg)
            res[key].append(psnr(img.cpu().numpy().astype(np.float64), gt))
    location, portable = asset_location(out_dir / fn)
    return {"run": name, "file": location, "file_portable": portable, "bytes": len(data), "bytes_per_splat": BYTES_PER_SPLAT,
            "asset_binding": binding, "run_identity": getattr(args, "run_identity", None),
            "metric_semantics": "decoded .splat bytes, SH0, gsplat renderer; not browser-renderer PSNR",
            "psnr_encoding": "finite number or '+Infinity' (infinite mean; at least one frame has zero MSE)",
            "n_gaussians_trained": int(len(opa)), "n_exported": int(len(idx)),
            "min_opacity": args.splat_min_opacity, "max_splats": args.splat_max,
            "initial_camera_frame": int(fr), "renderer": "@mkkellogg/gaussian-splats-3d@0.4.7 + three@0.160.0 (jsDelivr)",
            "heldout_psnr_mean": {k: psnr_mean_value(v) for k, v in res.items()}}


def preview_html(args, ck, sc):
    if (args.max_points <= 0 or not np.isfinite(args.radius) or args.radius <= 0
            or not np.isfinite(args.min_opacity) or not 0 <= args.min_opacity <= 1):
        raise ValueError("preview requires positive max-points/radius and opacity in [0, 1]")
    centers = sc.c2w[:, :3, 3].numpy()
    m = ck["means"]
    opa = torch.sigmoid(torch.from_numpy(ck["opacities"])).numpy()
    rgb = np.clip(sh0_to_rgb(torch.from_numpy(ck["sh0"][:, 0])).numpy(), 0, 1)
    keep = (opa > args.min_opacity) & (np.linalg.norm(m - centers.mean(0), axis=1) < args.radius)
    idx = np.where(keep)[0]
    rng = np.random.default_rng(args.seed)
    if len(idx) > args.max_points:
        idx = np.sort(rng.choice(idx, args.max_points, replace=False))
    if not len(idx):
        raise ValueError("preview selection is empty; adjust opacity/radius/max-points")
    p, c = m[idx], (rgb[idx] * 255).round().astype(np.uint8)
    lo, hi = p.min(0), p.max(0)
    qz = (np.round((p - lo) / np.maximum(hi - lo, 1e-6) * 65535) - 32768).astype(np.int16)
    data = {"lo": lo.round(4).tolist(), "hi": hi.round(4).tolist(),
            "pos": base64.b64encode(qz.tobytes()).decode(), "col": base64.b64encode(c.tobytes()).decode(),
            "train": centers[sc.train].round(3).tolist(), "test": centers[sc.test].round(3).tolist()}
    html = (HTML.replace("__DATA__", json.dumps(data)).replace("__NPTS__", f"{len(idx):,}")
            .replace("__NTOTAL__", f"{len(m):,}").replace("__OPA__", str(args.min_opacity)))
    return html, len(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="prepared scene (default: read from run's train_stats)")
    ap.add_argument("--allow-unverified-run", action="store_true", help="explicitly permit legacy runs without identity proof")
    ap.add_argument("--max-points", type=int, default=150_000)
    ap.add_argument("--min-opacity", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=80.0, help="drop centres farther than this from the trajectory mean (sky shell)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-ply", action="store_true")
    ap.add_argument("--skip-preview", action="store_true")
    ap.add_argument("--splat-dir", default=None, help="write <run>.splat + viewer.html here (e.g. docs/splat)")
    ap.add_argument("--splat-max", type=int, default=400_000)
    ap.add_argument("--splat-min-opacity", type=float, default=0.05)
    ap.add_argument("--web-results", default=None, help="JSON with size + held-out PSNR of the exported subset")
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    stats = json.loads((run / "train_stats.json").read_text(encoding="utf-8"))
    work = args.work or stats.get("work")
    if not work:
        raise SystemExit("pass --work (prepared scene folder)")
    sc = Scene(work)
    args.run_identity = verify_run(run, sc, stats, allow_legacy=args.allow_unverified_run)
    ck = {k: v.numpy() for k, v in torch.load(run / "ckpt.pt", map_location="cpu", weights_only=True).items()}
    if not args.skip_preview:
        html, preview_count = preview_html(args, ck, sc)
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_ply:
        write_ply(out / "gaussians.ply", ck)
    if args.splat_dir:
        web = export_web(args, ck, sc, Path(args.splat_dir), run.name, stats["sh_degree"])
        print(json.dumps(web, indent=1, allow_nan=False))
        if args.web_results:
            Path(args.web_results).write_text(json.dumps(web, indent=1, allow_nan=False), encoding="utf-8")
    if args.skip_preview:
        return
    (out / "preview.html").write_text(html, encoding="utf-8")
    ply = out / "gaussians.ply"
    info = {"run_identity": args.run_identity, "ply": str(ply), "ply_mb": round(ply.stat().st_size / 1e6, 1) if ply.exists() else None,
            "n_gaussians": int(len(ck["means"])), "preview_points": int(preview_count),
            "preview_html_mb": round((out / "preview.html").stat().st_size / 1e6, 2)}
    (out / "export_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
