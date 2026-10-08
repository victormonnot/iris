"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const tools = require("../../src/iris/static/tracking-selection-tools.js");
const source = { source: { kind: "comparison", job_id: "job", sequence_id: "sequence", profile_sha256: "a".repeat(64) }, sequence: { manifest: { taxonomy: { classes: [{ id: "person" }] }, frames: [{ frame_id: "first" }, { frame_id: "second" }] } }, replay: { profile: { class_ids: [1] }, passes: [{ frames: [{ frame_id: "first", observations: [{ detection_index: 5, track_id: 9, score: .9, confirmed: true }] }, { frame_id: "second", observations: [] }] }] } };
const request = (changes = {}) => ({ name: " One object ", selection: { frame_id: "first", detection_index: 5 }, release_frame_id: null, policy: { ...tools.defaults }, evaluation: null, max_seconds: 60, ...changes });

test("requests pin a measured detector index and source profile without using native ID as human identity", () => {
  const value = tools.configuration(request({ release_frame_id: "second" }), source);
  assert.equal(value.name, "One object"); assert.deepEqual(value.source, source.source);
  assert.deepEqual(value.selection, { frame_id: "first", detection_index: 5 }); assert.equal(value.evaluation, null);
  assert.notEqual(value.policy, tools.defaults); assert.equal(Object.hasOwn(value.selection, "track_id"), false);
});
test("missing, unconfirmed and low confidence observations cannot become an initial anchor", () => {
  assert.throws(() => tools.configuration(request({ selection: null }), source), /Click/);
  assert.throws(() => tools.configuration(request({ selection: { frame_id: "second", detection_index: 5 } }), source), /confirmed/);
  const changed = tools.clone(source); changed.replay.passes[0].frames[0].observations[0].confirmed = false;
  assert.throws(() => tools.configuration(request(), changed), /confirmed/);
  assert.throws(() => tools.configuration(request({ policy: { ...tools.defaults, min_score: .95 } }), source), /below/);
});
test("release must refer to a later available frame", () => {
  for (const id of ["first", "missing"]) assert.throws(() => tools.configuration(request({ release_frame_id: id }), source), /after/);
});
test("policy bounds preserve finite numbers and explicit source-time fallback", () => {
  assert.equal(tools.configuration(request({ policy: { ...tools.defaults, max_lost_seconds: null } }), source).policy.max_lost_seconds, null);
  for (const [key, value] of [["min_score", true], ["min_iou", NaN], ["max_center_distance", 11], ["max_area_ratio", .5], ["max_lost_updates", 0], ["recovery_confirmation_updates", 1], ["max_lost_seconds", 0]]) assert.throws(() => tools.configuration(request({ policy: { ...tools.defaults, [key]: value } }), source));
});
test("quality requires an explicit human identity and mapped taxonomy classes", () => {
  const evaluation = { reference_id: "reference", identity_id: "human-one", class_mapping: { "1": "person" }, iou_threshold: .5 };
  assert.deepEqual(tools.configuration(request({ evaluation }), source).evaluation, evaluation);
  for (const changed of [{ identity_id: "" }, { class_mapping: {} }, { class_mapping: { "1": null } }, { class_mapping: { "1": "unknown" } }, { iou_threshold: 0 }]) assert.throws(() => tools.configuration(request({ evaluation: { ...evaluation, ...changed } }), source));
});
test("source keys are canonical and recovery labels do not equate IDs with physical identity", () => {
  assert.equal(tools.sourceKey({ b: 2, a: 1 }), tools.sourceKey({ a: 1, b: 2 }));
  assert.equal(tools.stateLabel("recovering"), "Confirming recovery");
  assert.equal(tools.active({ status: "cancelling" }), true);
});
