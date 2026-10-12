// In renderer 0.4.7, AbortablePromise.then accepts only a resolve callback.
// Direct await therefore cannot observe its rejection. Attach catch explicitly.
export function scenePromise(operation) {
  return new Promise((resolve, reject) => operation.then(resolve).catch(reject));
}
