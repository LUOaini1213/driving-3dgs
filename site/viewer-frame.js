import { scenePromise } from "./scene-promise.js";

const channel = "driving-3dgs-viewer-v1";
const origin = location.origin;
const position = [2.059999942779541, 2.9579999446868896, -0.06700000166893005];
const target = [7.4670000076293945, -5.448999881744385, 0.2329999953508377];
let viewer = null;
let ready = false;
const report = (phase, extra = {}) => parent.postMessage({ channel, phase, ...extra }, origin);

function resetView() {
  if (!ready || !viewer) return;
  viewer.camera.up.set(0, 0, 1);
  viewer.camera.position.fromArray(position);
  viewer.controls.target.fromArray(target);
  viewer.camera.lookAt(viewer.controls.target);
  viewer.controls.update();
  report("reset");
}
window.addEventListener("message", (event) => {
  if (event.source !== parent || event.origin !== origin || event.data?.channel !== channel) return;
  if (event.data.command === "reset") {
    try { resetView(); } catch { report("error", { code: "load" }); }
  }
});
document.addEventListener("visibilitychange", () => {
  if (!ready || !viewer) return;
  if (document.hidden) viewer.stop(); else viewer.start();
});

async function start() {
  try {
    const probe = document.createElement("canvas");
    const gl = probe.getContext("webgl2");
    if (!gl || typeof WebAssembly !== "object") throw new Error("unsupported browser");
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    report("error", { code: "unsupported" });
    return;
  }
  try {
    const library = await import("./vendor/gaussian-splats-3d.module.js");
    viewer = new library.Viewer({
      cameraUp: [0, 0, 1], initialCameraPosition: position, initialCameraLookAt: target,
      sharedMemoryForWorkers: false, gpuAcceleratedSort: false, integerBasedSort: false,
      ignoreDevicePixelRatio: true, renderMode: library.RenderMode.OnChange,
      sceneRevealMode: library.SceneRevealMode.Instant, antialiased: true,
    });
    viewer.renderer.domElement.addEventListener("webglcontextlost", (event) => {
      event.preventDefault(); report("error", { code: "context" });
    });
    await scenePromise(viewer.addSplatScene("./assets/full_7k.splat", {
      format: library.SceneFormat.Splat, splatAlphaRemovalThreshold: 5,
      showLoadingUI: false, progressiveLoad: false,
      onProgress: (percent, label, loaderStatus) => {
        // The pinned renderer uses 0=Downloading, 1=Processing, 2=Done.
        report(loaderStatus === 0 ? "downloading" : "preparing", { percent });
      },
    }));
    document.querySelector(".frame-fallback").remove();
    viewer.start();
    // Keep the same first-render readiness boundary as the archived viewer check.
    for (let i = 0; i < 10; ++i) await new Promise(requestAnimationFrame);
    ready = true;
    if (document.hidden) viewer.stop();
    report("ready");
  } catch {
    report("error", { code: "load" });
  }
}
start();
