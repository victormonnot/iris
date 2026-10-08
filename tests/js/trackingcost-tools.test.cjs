"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/trackingcost-tools.js");
const base = { name: "Local measurement", lane_index: 0, device: "cpu", repeats: 2, policy: "offline_all", cadence_fps: null };

test("cost request preserves one explicit lane, device and repetitions without cached timings", () => {
  const input = { ...base, name: "  Measured run  ", cached_detector_ms: 12, tracker_ms: 3 };
  const result = tools.configuration(input);
  assert.deepEqual(result, { ...base, name: "Measured run" });
  assert.equal(input.name, "  Measured run  ");
  assert.equal(Object.hasOwn(result, "cached_detector_ms"), false);
  assert.deepEqual(tools.configuration({ ...base, lane_index: 1, device: "cuda", repeats: 5 }), { ...base, lane_index: 1, device: "cuda", repeats: 5 });
});

test("cost launch rejects invalid lane/device/repetition/name without silently tuning", () => {
  for (const value of [-1, 2, 0.5, "0", null]) assert.throws(() => tools.configuration({ ...base, lane_index: value }), /lane/);
  for (const value of [0, 6, 1.5, "2", NaN]) assert.throws(() => tools.configuration({ ...base, repeats: value }), /repetitions/);
  for (const value of ["auto", "cuda:0", "jetson", "paid-cloud"]) assert.throws(() => tools.configuration({ ...base, device: value }), /CPU or CUDA/);
  for (const value of ["", "   ", "n".repeat(161), null]) assert.throws(() => tools.configuration({ ...base, name: value }), /run name/);
});

test("arrival cadence is declared only for simulated latest scheduling", () => {
  assert.throws(() => tools.configuration({ ...base, cadence_fps: 30 }), /Offline/);
  assert.throws(() => tools.configuration({ ...base, policy: "camera_live" }), /policy/);
  for (const cadence of [null, 0, 0.09, 240.1, Infinity, NaN, "30"]) assert.throws(() => tools.configuration({ ...base, policy: "simulated_latest", cadence_fps: cadence }), /0.1–240/);
  for (const cadence of [0.1, 30, 240]) assert.equal(tools.configuration({ ...base, policy: "simulated_latest", cadence_fps: cadence }).cadence_fps, cadence);
  assert.match(tools.policyLabel("simulated_latest"), /Simulated/);
  assert.match(tools.policyLabel("offline_all"), /Offline/);
});

test("history lane filter reads the exact durable job and lightweight summary contracts", () => {
  assert.equal(tools.laneIndex({ job: { params: { config: { lane_index: 1 } } } }), 1);
  assert.equal(tools.laneIndex({ job: { result: { request: { lane_index: 0 } } } }), 0);
  assert.equal(tools.laneIndex({ report: { request: { lane_index: 1 } }, job: { params: { config: { lane_index: 1 } } } }), 1);
  assert.equal(tools.laneIndex({ job: { params: {} } }), null);
});

test("saved and imported reports must match both selected IDs and their frozen source", () => {
  const context = { comparison_id: "comparison-a", sequence_id: "sequence-a" };
  const record = { ...context, report: { source: { ...context } }, origin: "imported_declaration" };
  assert.equal(tools.matchesContext(record, context), true);
  assert.equal(tools.matchesContext({ ...context, report: null }, context), true, "pending durable jobs and lightweight history have no report");
  assert.equal(tools.matchesContext({ ...record, comparison_id: "comparison-b" }, context), false);
  assert.equal(tools.matchesContext({ ...record, sequence_id: "sequence-b" }, context), false);
  assert.equal(tools.matchesContext({ ...record, report: { source: { ...context, comparison_id: "comparison-b" } } }, context), false);
  assert.equal(tools.matchesContext({ ...record, report: {} }, context), false);
  assert.equal(tools.matchesContext(null, context), false);
});

test("missing memory and rates remain unavailable while measured zero stays explicit", () => {
  assert.equal(tools.memory(null), "Unavailable");
  assert.equal(tools.memory(0), "0.00 MiB");
  assert.equal(tools.memory(1048576), "1.00 MiB");
  assert.equal(tools.milliseconds(null), "Unavailable");
  assert.equal(tools.milliseconds(0), "0.00 ms");
  assert.equal(tools.milliseconds(1.256), "1.26 ms");
  assert.equal(tools.fps(null), "Undefined");
  assert.equal(tools.fps(0), "0.00 FPS");
  assert.equal(tools.fps(33.333), "33.33 FPS");
});

test("outer pipeline timing remains distinct from overlapping nested detector and tracker timers", () => {
  assert.equal(tools.stages.find(([key]) => key === "pipeline_ms")[2], "outer");
  assert.equal(tools.stages.find(([key]) => key === "detector_call_ms")[2], "outer");
  assert.equal(tools.stages.find(([key]) => key === "tracker_call_ms")[2], "outer");
  assert.equal(tools.stages.find(([key]) => key === "detector_inference_ms")[2], "nested");
  assert.equal(tools.stages.find(([key]) => key === "tracker_association_ms")[2], "nested");
  assert.equal(new Set(tools.stages.map(([key]) => key)).size, 14);
});

test("only durable active states expose measurement cancellation", () => {
  for (const status of ["queued", "running", "cancelling"]) assert.equal(tools.active({ status }), true);
  for (const status of ["succeeded", "failed", "cancelled", "interrupted"]) assert.equal(tools.active({ status }), false);
  assert.equal(tools.active(null), false);
});

test("report import previews parsed JSON while preserving the exact numeric source tokens", () => {
  const raw = '{\n "schema":"iris-tracking-cost-v1", "complete":true, "request":{"cadence_fps":120.0}, "frames":[{"timestamp_seconds":0.0,"timing":{"filter_ms":0.0},"index":0}], "metadata":{"scale":1.00e+0}\n}\n';
  const imported = tools.parseImport(raw);
  assert.equal(imported.report.request.cadence_fps, 120, "preview uses ordinary JavaScript values");
  assert.equal(imported.body, '{"report":' + raw + '}', "upload wraps the original JSON text without serializing its parsed numbers");
  assert.match(imported.body, /"cadence_fps":120\.0/);
  assert.match(imported.body, /"timestamp_seconds":0\.0/);
  assert.match(imported.body, /"filter_ms":0\.0/);
  assert.match(imported.body, /"scale":1\.00e\+0/);
  assert.deepEqual(JSON.parse(imported.body), { report: imported.report });
  assert.notEqual(imported.body, JSON.stringify({ report: imported.report }), "re-serialization would erase protocol-significant float tokens");
  imported.report.request.cadence_fps = 30;
  assert.match(imported.body, /"cadence_fps":120\.0/, "preview mutation cannot rewrite frozen import evidence");
});

test("raw import envelope rejects malformed, incomplete and unsupported JSON before upload", () => {
  for (const raw of ['{"schema":', '{}', 'null', '[]', '"text"', '{"schema":"iris-tracking-cost-v1","complete":false}', '{"schema":"other","complete":true}', '{"schema":"iris-tracking-cost-v1","complete":true} trailing', '{"schema":"iris-tracking-cost-v1","complete":true}{"injected":true}']) assert.throws(() => tools.parseImport(raw));
  assert.throws(() => tools.parseImport({ schema: "iris-tracking-cost-v1", complete: true }), /JSON file/);
});
