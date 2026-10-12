import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { gzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { AbortablePromise, SplatLoader, Viewer } from "../site/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js";
import { Vector3 } from "../site/node_modules/three/build/three.module.js";
import { scenePromise, viewerWithDiagnostics, withDecodedScene } from "../site/scene-promise.js";

test("actual pinned renderer success crosses the native promise boundary", {timeout:1000}, async () => {
  const operation = new AbortablePromise(resolve => resolve("loaded"));
  assert.equal(await scenePromise(operation), "loaded");
});

test("actual pinned renderer download rejection reaches the caller promptly", {timeout:1000}, async () => {
  const failure = new Error("Controlled 404 scene download");
  const operation = new AbortablePromise((resolve, reject) => reject(failure));
  await assert.rejects(scenePromise(operation), error => error === failure);
});

test("native import or setup rejections keep the same failure boundary", {timeout:1000}, async () => {
  const failure = new Error("Controlled renderer initialization failure");
  await assert.rejects(scenePromise(Promise.reject(failure)), error => error === failure);
});

test("actual pinned Viewer retains the original cause before its generic load error", () => {
  const observed = [];
  const DiagnosticViewer = viewerWithDiagnostics(Viewer, error => observed.push(error));
  // updateError itself needs no WebGL context or renderer constructor.
  const instance = Object.create(DiagnosticViewer.prototype);
  const original = new TypeError("Controlled underlying scene fetch/build failure");
  const message = "Viewer::addSplatScene -> Could not load file ./assets/full_7k.splat";
  const expected = Viewer.prototype.updateError.call(instance, original, message);
  const actual = instance.updateError(original, message);
  assert.deepEqual(observed, [original]);
  assert.equal(actual.message, expected.message);
  assert.equal(actual.constructor, expected.constructor);
  assert.notEqual(actual, original);
});

test("diagnostic callback failure cannot replace the renderer rejection", () => {
  const DiagnosticViewer = viewerWithDiagnostics(Viewer, () => { throw new Error("logger failed"); });
  const instance = Object.create(DiagnosticViewer.prototype);
  const actual = instance.updateError(new Error("original cause"), "existing renderer error");
  assert.equal(actual.message, "existing renderer error");
});

test("actual HTTP gzip and pinned parser preserve the decoded scene across the Blob boundary", { timeout: 10000 }, async () => {
  // CPU parser/transport control only: no WebGL renderer or scene evaluation.
  const previousWindow = globalThis.window;
  globalThis.window = { setTimeout }; // The pinned parser defers CPU work via this timer.
  const rows = 8;
  const raw = Buffer.alloc(rows * 32);
  for (let i = 0; i < rows; ++i) {
    const offset = i * 32;
    raw.writeFloatLE(i / 4, offset);
    raw.writeFloatLE(i / 8, offset + 4);
    raw.writeFloatLE(0, offset + 8);
    for (const at of [12, 16, 20]) raw.writeFloatLE(0.1, offset + at);
    raw.set([255, 40, 80, 255, 255, 128, 128, 128], offset + 24);
  }
  const encoded = gzipSync(raw);
  assert.ok(encoded.length < raw.length);
  const hash = bytes => createHash("sha256").update(bytes).digest("hex");
  const server = createServer((request, response) => {
    if (request.url === "/missing.splat") {
      response.writeHead(404); response.end("controlled missing scene"); return;
    }
    response.writeHead(200, {
      "Content-Type": "application/octet-stream", "Content-Encoding": "gzip",
      "Content-Length": encoded.length,
    });
    response.end(encoded);
  });
  const bounded = (operation, name) => {
    let timer;
    const timedOut = new Promise((_resolve, reject) => {
      timer = setTimeout(() => {
        operation.abort?.("controlled parser timeout");
        reject(new Error(name + ": timeout; not a successful parser control"));
      }, 4000);
    });
    return Promise.race([scenePromise(operation), timedOut]).finally(() => clearTimeout(timer));
  };
  let successfulURL, failedURL;
  try {
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", resolve);
    });
    const url = `http://127.0.0.1:${server.address().port}/controlled.splat`;
    const response = await fetch(url, { signal: AbortSignal.timeout(4000) });
    const decoded = await response.arrayBuffer();
    assert.equal(response.headers.get("Content-Encoding"), "gzip");
    assert.equal(Number(response.headers.get("Content-Length")), encoded.length);
    assert.equal(decoded.byteLength, raw.length);
    assert.equal(hash(Buffer.from(decoded)), hash(raw));

    await assert.rejects(
      bounded(SplatLoader.loadFromURL(url, undefined, false, undefined, 5, 0, true), "original URL path"),
      error => error instanceof RangeError,
    );

    const splatBuffer = await bounded(withDecodedScene(url, raw.length, async blobURL => {
      successfulURL = blobURL;
      const blobResponse = await fetch(blobURL);
      assert.equal(Number(blobResponse.headers.get("Content-Length")), raw.length);
      assert.equal(blobResponse.headers.get("Content-Encoding"), null);
      const blobBytes = await blobResponse.arrayBuffer();
      assert.equal(hash(Buffer.from(blobBytes)), hash(raw));
      return bounded(SplatLoader.loadFromURL(blobURL, undefined, false, undefined, 5, 0, true),
                     "owned adapter pinned Blob URL path");
    }), "owned adapter decode/load");
    await assert.rejects(fetch(successfulURL));
    assert.equal(splatBuffer.getSplatCount(), rows);
    const centers = [];
    for (let i = 0; i < rows; ++i) {
      const center = new Vector3();
      splatBuffer.getSplatCenter(i, center);
      centers.push([center.x, center.y, center.z]);
    }
    centers.sort((a, b) => a[0] - b[0]);
    assert.deepEqual(centers, Array.from({ length: rows }, (_, i) => [i / 4, i / 8, 0]));

    let invalidSizeCallback = false;
    await assert.rejects(withDecodedScene(url, raw.length + 32, () => { invalidSizeCallback = true; }),
                         /unexpected decoded byte length/);
    assert.equal(invalidSizeCallback, false);
    let missingCallback = false;
    await assert.rejects(withDecodedScene(url.replace("controlled.splat", "missing.splat"), raw.length,
                         () => { missingCallback = true; }), /HTTP 404/);
    assert.equal(missingCallback, false);
    const controlledRejection = new Error("controlled existing Viewer rejection");
    await assert.rejects(withDecodedScene(url, raw.length, blobURL => {
      failedURL = blobURL; return Promise.reject(controlledRejection);
    }), error => error === controlledRejection);
    await assert.rejects(fetch(failedURL));
  } finally {
    if (successfulURL) URL.revokeObjectURL(successfulURL);
    if (failedURL) URL.revokeObjectURL(failedURL);
    if (previousWindow === undefined) delete globalThis.window; else globalThis.window = previousWindow;
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});
