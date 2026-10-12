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
      reportFailure(error);
      return super.updateError(error, ...args);
    }
  };
}
