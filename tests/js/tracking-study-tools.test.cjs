"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const tools = require("../../src/iris/static/tracking-study-tools.js");
const baseline = () => ({ schema: "iris-tracker-profile-v1", algorithm: "bytetrack", class_ids: [1], high_threshold: .5, low_threshold: .1, new_track_threshold: .6, match_threshold: .8, buffer_updates: 30, fuse_score: true, gmc_method: "none", gmc_downscale: 2, seed: 0, opencv_threads: 1, with_reid: false, time_policy: "one_update_per_available_frame" });
const request = () => ({ name: "Study", dataset_id: "dataset", sources: [{ sequence_id: "development", comparison_id: "comparison" }], baseline: { name: "Baseline", profile: baseline() }, candidates: [{ name: "Longer lost buffer", profile: { ...baseline(), buffer_updates: 60 } }], class_mapping: { 1: "person" }, iou_threshold: .5, repeats: 2, max_updates: 1000, max_seconds: 120 });

test("dataset preparation requires explicit human revision pins for development and validation, while test stays reserved", () => {
  const rows = [{ sequence_id: "train", selected: true, split: "train", reference_id: "r1" }, { sequence_id: "val", selected: true, split: "val", reference_id: "r2" }, { sequence_id: "test", selected: true, split: "test", reference_id: "" }, { sequence_id: "unused", selected: false, split: "train", reference_id: "" }];
  assert.deepEqual(tools.datasetRequest(" References ", rows), { name: "References", entries: [{ sequence_id: "train", split: "train", reference_id: "r1" }, { sequence_id: "val", split: "val", reference_id: "r2" }, { sequence_id: "test", split: "test", reference_id: null }] });
  rows[1].reference_id = "";
  assert.throws(() => tools.datasetRequest("References", rows), /Pin a human reference/);
  assert.throws(() => tools.datasetRequest("References", [rows[2]]), /Development|development/);
  assert.equal(tools.splitLabel("test"), "Reserved test · never evaluated");
});

test("editing ByteTrack keeps native derived thresholds and cannot mutate the frozen baseline", () => {
  const saved = baseline(), edited = tools.editProfile(saved, "high_threshold", .7);
  assert.equal(saved.high_threshold, .5);
  assert.equal(edited.low_threshold, .1);
  assert.equal(edited.new_track_threshold, .7 + .1);
  assert.equal(tools.profile(edited).gmc_method, "none");
  const switched = tools.editProfile({ ...saved, algorithm: "botsort", gmc_method: "sparseOptFlow", low_threshold: .05 }, "algorithm", "bytetrack");
  assert.equal(switched.gmc_method, "none"); assert.equal(switched.low_threshold, .1);
  assert.throws(() => tools.profile({ ...saved, high_threshold: .1, new_track_threshold: .2 }), /ByteTrack/);
  assert.throws(() => tools.profile({ ...saved, algorithm: "botsort", low_threshold: .6 }), /low < high/);
});

test("study profiles are unique even when object keys are reordered, and share the baseline classes", () => {
  const payload = request(), original = structuredClone(payload);
  assert.deepEqual(tools.configuration(payload), original);
  assert.deepEqual(payload, original);
  payload.candidates[0].profile = Object.fromEntries(Object.entries(baseline()).reverse());
  assert.throws(() => tools.configuration(payload), /Every included candidate must differ/);
  payload.candidates[0].profile = { ...baseline(), class_ids: [2] };
  assert.throws(() => tools.configuration(payload), /same native detector classes/);
  payload.candidates[0].profile = { ...baseline(), buffer_updates: 60 };
  payload.candidates[0].name = "x".repeat(81);
  assert.throws(() => tools.configuration(payload), /1–80/);
  payload.candidates[0].name = "Baseline";
  assert.throws(() => tools.configuration(payload), /names must be distinct/);
});

test("explicit bounded requests reject missing comparisons, excessive repetitions and unsupported work budgets", () => {
  for (const [key, value] of [["repeats", 4], ["max_updates", 20001], ["max_updates", .5], ["max_seconds", 601], ["iou_threshold", 0]]) {
    const payload = request(); payload[key] = value; assert.throws(() => tools.configuration(payload));
  }
  const payload = request(); payload.sources[0].comparison_id = "";
  assert.throws(() => tools.configuration(payload), /completed comparison/);
  payload.sources[0].comparison_id = "comparison"; payload.candidates = [];
  assert.throws(() => tools.configuration(payload), /1–7 candidate/);
});

test("native classes must be mapped explicitly and an entirely ignored study is rejected", () => {
  for (const mapping of [{}, { 1: "" }, { 1: null }, { 1: "person", 2: "car" }]) {
    const payload = request(); payload.class_mapping = mapping;
    assert.throws(() => tools.configuration(payload), /native class|Map at least/);
  }
  const payload = request(); payload.baseline.profile.class_ids = [1, 2]; payload.candidates[0].profile.class_ids = [1, 2]; payload.class_mapping = { 1: "person", 2: null };
  assert.deepEqual(tools.configuration(payload).class_mapping, { 1: "person", 2: null });
});

test("cost and quality display preserves undefined measurements and explicit baseline decisions", () => {
  assert.equal(tools.percentage(null), "Unavailable"); assert.equal(tools.milliseconds(null), "Unavailable");
  assert.equal(tools.percentage(0), "0.0%"); assert.equal(tools.milliseconds(-2.5), "-2.50 ms");
  assert.equal(tools.statusLabel("no_gain"), "No quality gain"); assert.equal(tools.statusLabel("unstable"), "Repeatability differs");
  assert.equal(tools.active({ status: "cancelled" }), false); assert.equal(tools.active({ status: "cancelling" }), true);
});
