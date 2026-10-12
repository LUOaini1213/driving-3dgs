(() => {
  "use strict";
  const channel = "driving-3dgs-viewer-v1";
  const stage = document.getElementById("stage");
  const load = document.getElementById("load");
  const reset = document.getElementById("reset");
  const stop = document.getElementById("stop");
  const status = document.getElementById("status-text");
  const statusBox = document.getElementById("load-status");
  const progress = document.getElementById("progress");
  const progressLabel = document.getElementById("progress-label");
  let frame = null;
  let ready = false;
  let timeout = null;

  function clearDeadline() { if (timeout !== null) clearTimeout(timeout); timeout = null; }
  function describe(text, error = false) {
    status.textContent = text;
    statusBox.dataset.error = String(error);
  }
  function release() {
    clearDeadline();
    // Discard the whole renderer document, including its workers and WebGL context.
    // Late postMessages from that document cannot match the next iframe window.
    const old = frame;
    frame = null;
    ready = false;
    old?.remove();
    stage.dataset.mode = "preview";
    load.disabled = false;
    reset.disabled = true;
    stop.disabled = true;
    progress.hidden = true;
    progressLabel.textContent = "";
  }
  function fail(text) { release(); describe(text, true); }
  function armDeadline() {
    clearDeadline();
    if (frame && !ready && !document.hidden) timeout = setTimeout(() => {
      fail("The scene took too long to prepare. The preview remains available; try a current browser or reload with fewer tabs open.");
    }, 180000);
  }
  load.addEventListener("click", () => {
    if (frame) return;
    if (!["http:", "https:"].includes(location.protocol)) {
      fail("Open this page through HTTP or HTTPS. Direct file access cannot load the interactive scene.");
      return;
    }
    load.disabled = true;
    stop.disabled = false;
    stage.dataset.mode = "loading";
    progress.hidden = false;
    progress.removeAttribute("value");
    describe("Checking browser support and loading the renderer…");
    const next = document.createElement("iframe");
    next.title = "Interactive Argoverse 2 driving-scene reconstruction";
    next.tabIndex = -1;
    next.src = "viewer-frame.html";
    frame = next;
    stage.appendChild(next);
    armDeadline();
  });
  stop.addEventListener("click", () => {
    release();
    describe("Preview ready. The interactive frame has been closed.");
  });
  reset.addEventListener("click", () => {
    if (frame && ready) frame.contentWindow.postMessage({ channel, command: "reset" }, location.origin);
  });
  window.addEventListener("message", (event) => {
    if (!frame || event.source !== frame.contentWindow || event.origin !== location.origin) return;
    const data = event.data;
    if (!data || data.channel !== channel) return;
    if (data.phase === "downloading") {
      describe("Downloading the 9.9 MB scene…");
      if (typeof data.percent === "number" && Number.isFinite(data.percent)) {
        progress.value = Math.max(0, Math.min(100, data.percent));
        progressLabel.textContent = `${Math.floor(progress.value)}%`;
      }
    } else if (data.phase === "preparing") {
      describe("Scene downloaded. Preparing splats and the first view; this can take a while on slower devices…");
      progress.removeAttribute("value");
      progressLabel.textContent = "";
    } else if (data.phase === "ready") {
      ready = true;
      clearDeadline();
      stage.dataset.mode = "ready";
      progress.hidden = true;
      progressLabel.textContent = "";
      reset.disabled = false;
      frame.tabIndex = 0;
      describe("Interactive scene ready. Drag to explore, or reset to held-out camera 84.");
    } else if (data.phase === "reset") {
      describe("Returned to held-out camera 84.");
    } else if (data.phase === "error") {
      const messages = {
        unsupported: "This browser needs WebGL 2 and WebAssembly to display the scene. The recorded preview remains available.",
        context: "The browser lost its 3D rendering context. The preview remains available; close other tabs before trying again.",
        load: "The renderer or scene could not be loaded. The preview remains available; check the connection or try a current browser.",
      };
      fail(messages[data.code] ?? messages.load);
    }
  });
  document.addEventListener("visibilitychange", armDeadline);
  window.addEventListener("pagehide", release);
})();
