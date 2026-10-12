// In renderer 0.4.7, AbortablePromise.then accepts only a resolve callback.
// Direct await therefore cannot observe its rejection. Attach catch explicitly.
export function scenePromise(operation) {
  return new Promise((resolve, reject) => operation.then(resolve).catch(reject));
}

// The pinned Viewer replaces download/build causes with one generic load error.
// Observe the original cause before delegating; preserve its rejection behavior.
export function viewerWithDiagnostics(Viewer, reportFailure) {
  return class extends Viewer {
    updateError(error, ...args) {
      try { reportFailure(error); } catch { /* Diagnostics cannot replace the renderer error. */ }
      return super.updateError(error, ...args);
    }
  };
}

// fetch decodes HTTP gzip/br, whose Content-Length describes compressed bytes.
// A Blob URL exposes the decoded length to the pinned streaming splat loader.
export async function withDecodedScene(path, expectedBytes, loadScene) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Scene download failed: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== expectedBytes) throw new Error("Scene download has an unexpected decoded byte length");
  const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
  try { return await loadScene(url); }
  finally { URL.revokeObjectURL(url); }
}
