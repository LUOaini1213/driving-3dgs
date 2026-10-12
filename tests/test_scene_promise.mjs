import assert from "node:assert/strict";
import test from "node:test";
import { AbortablePromise, Viewer } from "../site/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js";
import { scenePromise, viewerWithDiagnostics } from "../site/scene-promise.js";

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
