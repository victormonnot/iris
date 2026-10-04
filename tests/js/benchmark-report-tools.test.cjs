"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/benchmark-report-tools.js");

test("missing quality is never promoted to a zero-error measurement", () => {
  const empty = tools.quality(null);
  assert.equal(empty.errors, "Incomplete · not scored");
  assert.equal(empty.precisionRecall, "N/A");
  assert.deepEqual(tools.quality({ complete: false, metrics: { summary: { fp: 0, fn: 0, precision: 1 } } }), empty);
  const measured = tools.quality({ complete: true, metrics: { summary: { fp: 0, fn: 0, class_conflicts: 0, precision: null, recall: null, matched_iou_mean: null } } });
  assert.equal(measured.errors, "0 extra / 0 missed");
  assert.equal(measured.iou, "N/A");
  assert.equal(measured.precisionRecall, "N/A / N/A");
});

test("single successful trial does not establish stability and repeated runs retain their shared dataset limitation", () => {
  assert.match(tools.repeatability({ measured: false, complete_count: 1, trial_count: 8 }), /Stability unavailable.*1 complete/);
  assert.match(tools.repeatability({ measured: true, complete_count: 1, trial_count: 8 }), /Stability unavailable/);
  assert.match(tools.repeatability({ measured: true, complete_count: 2, distinct_geometry_count: 1, identical_geometry: true }), /2 complete.*identical recorded geometry.*not independent datasets/);
  assert.match(tools.repeatability({ measured: true, complete_count: 2, distinct_geometry_count: 2 }), /2 distinct box sets/);
});

test("metric ranges disclose their sample count and do not coerce unavailable values to zero", () => {
  assert.equal(tools.metricRange(null), "Unavailable");
  assert.equal(tools.metricRange({ count: 0, min: null, max: null }), "Unavailable");
  assert.equal(tools.metricRange({ count: 1, min: 0, max: 0, mean: 0 }), "0 · n=1");
  assert.equal(tools.metricRange({ count: 2, min: 0.5, max: 1, mean: 0.75 }, tools.percentage), "50.0%–100.0% · mean 75.0% · n=2");
  for (const value of [undefined, null, NaN, Infinity, "0", false]) {
    assert.equal(tools.number(value), "N/A");
    assert.equal(tools.percentage(value), "N/A");
  }
});

test("comparison and immutable report membership requires the selected reference and role", () => {
  const comparison = { protocol: "iris-benchmark-comparison-v1", benchmark: { id: "first" }, role: "evaluation", reference: { frames: [] }, configs: [] };
  assert.equal(tools.validComparison(comparison, "first", "evaluation"), true);
  assert.equal(tools.validComparison(comparison, "second", "evaluation"), false);
  assert.equal(tools.validComparison(comparison, "first", "tuning"), false);
  assert.equal(tools.validComparison({ ...comparison, role: "test" }, "first"), false);
  const report = { id: "report", benchmark_id: "first", snapshot_sha256: "hash", snapshot: { comparison } };
  assert.equal(tools.validReport(report, "first"), true);
  assert.equal(tools.validReport({ ...report, benchmark_id: "second" }, "first"), false);
  assert.equal(tools.validReport({ ...report, snapshot: { comparison: { ...comparison, benchmark: { id: "second" } } } }, "first"), false);
  assert.equal(tools.validReport({ ...report, snapshot_sha256: null }, "first"), false);
});

test("image comparison never substitutes a different trial or frame when evidence is absent", () => {
  const reference = { frame_id: "image-one" };
  assert.equal(tools.frameFor({ reference: { frames: [reference] } }, "image-one"), reference);
  assert.equal(tools.frameFor({ reference: { frames: [reference] } }, "image-two"), null);
  const config = { trials: [{ id: "first", frames: [{ frame_id: "image-one", proposals: [] }] }, { id: "second", frames: [{ frame_id: "image-two", proposals: [{ id: "candidate" }] }] }] };
  assert.equal(tools.trialFrame(config, "first", "image-two").frame, null);
  assert.deepEqual(tools.trialFrame(config, "missing", "image-one"), { trial: null, frame: null });
  assert.deepEqual(tools.trialFrame(config, "first", "image-one").frame.proposals, []);
});

test("evidence declarations distinguish simulation from verified model performance", () => {
  assert.match(tools.evidence("not_declared"), /not declared/);
  assert.match(tools.evidence("simulation"), /Simulated.*does not measure real model performance/);
  assert.match(tools.evidence("real_data"), /declared by the author.*not independently verified/);
  assert.match(tools.evidence("unexpected"), /unavailable/);
  assert.equal(tools.approach("local_detector"), "Local detector control");
});
