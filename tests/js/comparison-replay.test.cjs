"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const {
  clock,
  sampleAtOrBefore,
  samplePosition,
  coverage,
  count,
} = require("../../src/iris/static/comparison-replay.js");

const sample = (id, time) => ({ frame_id: id, timestamp_seconds: time });

test("browser registration mounts nothing until create is called", () => {
  const window = {};
  vm.runInNewContext(
    readFileSync(
      join(__dirname, "../../src/iris/static/comparison-replay.js"),
      "utf8",
    ),
    { window },
  );
  assert.deepEqual(Object.keys(window.IRISComparisonReplay), ["create"]);
  assert.equal(typeof window.IRISComparisonReplay.create, "function");
});

test("before the first saved timestamp there is no current analysed sample", () => {
  const samples = [sample("first", 0.5), sample("later", 3.5)];
  assert.equal(sampleAtOrBefore(samples, 0), null);
  assert.equal(sampleAtOrBefore(samples, 0.499999999), null);
  assert.equal(sampleAtOrBefore(samples, 0.5).frame_id, "first");
});

test("following keeps the preceding saved sample throughout an unanalysed gap", () => {
  const samples = [
    sample("late", 6.5),
    sample("early", 1),
    sample("middle", 3.5),
  ];
  assert.equal(sampleAtOrBefore(samples, 3.499999999).frame_id, "early");
  assert.equal(sampleAtOrBefore(samples, 3.5).frame_id, "middle");
  assert.equal(sampleAtOrBefore(samples, 6.49).frame_id, "middle");
  assert.equal(sampleAtOrBefore(samples, 90).frame_id, "late");
});

test("duplicate timestamps retain an explicitly selected frame without choosing a future frame", () => {
  const samples = [sample("a", 2), sample("b", 2), sample("later", 5)];
  assert.equal(sampleAtOrBefore(samples, 2).frame_id, "a");
  assert.equal(sampleAtOrBefore(samples, 4.99, "b").frame_id, "b");
  assert.equal(sampleAtOrBefore(samples, 4.99, "later").frame_id, "a");
  assert.equal(sampleAtOrBefore(samples, 5, "b").frame_id, "later");
});

test("missing or invalid timestamps are never reconstructed from frame index or FPS", () => {
  const samples = [
    { ...sample("unknown", null), frame_index: 0, fps: 30 },
    sample("nan", NaN),
    sample("infinite", Infinity),
    sample("negative", -1),
    sample("valid", 2),
  ];
  assert.equal(sampleAtOrBefore(samples, 0), null);
  assert.equal(sampleAtOrBefore(samples, 10).frame_id, "valid");
  for (const time of [null, NaN, Infinity, -1])
    assert.equal(sampleAtOrBefore(samples, time), null);
});

test("sample context never treats a nearby playback instant as the recorded timestamp", () => {
  assert.deepEqual(samplePosition(3.5, 3.5), {
    kind: "recorded-timestamp",
    delta: 0,
  });
  for (const distance of [Number.EPSILON * 4, 0.001, 0.01, 0.05, 0.1]) {
    assert.equal(samplePosition(3.5 - distance, 3.5).kind, "before");
    assert.equal(samplePosition(3.5 + distance, 3.5).kind, "after");
  }
  assert.deepEqual(samplePosition(1, null), { kind: "unknown", delta: null });
});

test("coverage counts saved run outputs, including empty detections from full and tiled lanes", () => {
  const complete = {
    prediction_count: 2,
    predicted_run_ids: [
      "same-checkpoint-full-run",
      "same-checkpoint-tiled-run",
    ],
    complete: true,
    detections: [],
  };
  assert.equal(count(complete), 2);
  assert.equal(coverage(complete), "complete");
  assert.equal(coverage({ prediction_count: 1, complete: false }), "partial");
  assert.equal(coverage({ prediction_count: 0, complete: false }), "empty");
  assert.equal(coverage({ ...complete, image_available: false }), "complete");
});

test("timestamp labels carry rounded seconds to the next minute and reject invalid time", () => {
  assert.equal(clock(0), "00:00.00");
  assert.equal(clock(3.5), "00:03.50");
  assert.equal(clock(59.999), "01:00.00");
  assert.equal(clock(3600), "60:00.00");
  for (const value of [null, undefined, NaN, Infinity, -1, Number.MAX_VALUE])
    assert.equal(clock(value), "Unknown time");
});
