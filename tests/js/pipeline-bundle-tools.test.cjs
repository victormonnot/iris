"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const tools = require("../../src/iris/static/pipeline-bundle-tools.js");
const descriptor = { kind: "study", job_id: "study", sequence_id: "sequence", profile_sha256: "a".repeat(64) };
const source = { source: descriptor, name: "Saved study", detector: { architecture: "yolox_nano", min_score: .01, classes: [{ id: 7, name: "vehicle" }] }, profile: { algorithm: "botsort", class_ids: [7], low_threshold: .1, high_threshold: .5, buffer_updates: 30, gmc_method: "sparseOptFlow" }, selections: [{ id: "policy", name: "Continuity" }] };
const values = { name: " Export ", target_device: "cuda", selection_id: "policy" };
test("portable requests preserve exact source and independently selected detector target", () => {
  const request = tools.configuration(values, source);
  assert.equal(request.name, "Export"); assert.equal(request.target_device, "cuda"); assert.deepEqual(request.source, descriptor); assert.notEqual(request.source, descriptor);
  assert.deepEqual(Object.keys(request).sort(), ["name", "selection_id", "source", "target_device"]);
  assert.equal(tools.configuration({ ...values, selection_id: "" }, source).selection_id, null);
});
test("unavailable source, unknown policy, invalid target and malformed profile cannot be packaged", () => {
  assert.throws(() => tools.configuration(values, null), /exact saved/);
  assert.throws(() => tools.configuration(values, { ...source, available: false, reason: "Tiled input cannot be packaged" }), /Tiled/);
  assert.throws(() => tools.configuration({ ...values, selection_id: "other-profile" }, source), /exact source/);
  assert.throws(() => tools.configuration({ ...values, target_device: "jetson" }, source), /CPU or NVIDIA/);
  assert.throws(() => tools.configuration(values, { ...source, source: { ...descriptor, profile_sha256: "latest" } }), /frozen tracker/);
  for (const name of ["", "x".repeat(161), "a\nb"]) assert.throws(() => tools.configuration({ ...values, name }, source));
});
test("summary names external detector IDs and original pixel requirements without claiming execution", () => {
  assert.match(tools.sourceSummary(source), /vehicle \(7\)/); assert.match(tools.sourceSummary(source), /original BGR/);
  assert.equal(tools.sourceKey({ a: 1, b: 2 }), tools.sourceKey({ b: 2, a: 1 }));
  assert.equal(tools.bytes(1024 ** 2), "1.0 MiB"); assert.equal(tools.bytes(NaN), "Size unavailable");
  assert.equal(tools.active({ status: "cancelling" }), true);
  assert.match(tools.policySummary(null), /No selected-object policy/);
  assert.match(tools.policySummary({ min_score: .3, min_iou: .05, max_center_distance: 1, max_area_ratio: 3, max_lost_updates: 15, max_lost_seconds: null, recovery_confirmation_updates: 2 }), /does not prove physical identity/);
});
