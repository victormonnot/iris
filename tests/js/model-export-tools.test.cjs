const test = require("node:test");
const assert = require("node:assert/strict");
const { MAX_MEASUREMENT_BYTES, selectionValid, selectionKey, findExport, findMeasurement, fileProblem, measurementPresentation } = require("../../src/iris/static/model-export-tools.js");

test("export confirmation binds checkpoint, evaluation, image selection and name", () => {
  const options = { trained_model_id: "trained-a", evaluation_id: "eval-a", name: "Target machine", frame_ids: ["frame-a", "frame-b"] };
  assert.equal(selectionValid(options), true);
  for (const change of [{ trained_model_id: "trained-b" }, { evaluation_id: "eval-b" }, { name: "Changed" }, { frame_ids: ["frame-b", "frame-a"] }, { frame_ids: ["frame-a"] }, { target_device: "cuda" }]) {
    assert.notEqual(selectionKey({ ...options, ...change }), selectionKey(options));
  }
  for (const change of [{ trained_model_id: "" }, { evaluation_id: "" }, { name: " " }, { frame_ids: [] }, { frame_ids: ["frame-a", "frame-a"] }, { frame_ids: Array.from({ length: 9 }, (_, i) => `frame-${i}`) }, { frame_ids: [null] }, { target_device: "mps" }]) {
    assert.equal(selectionValid({ ...options, ...change }), false);
  }
  assert.equal(selectionValid({ ...options, frame_ids: Array.from({ length: 8 }, (_, i) => `frame-${i}`) }), true);
});

test("export targets remain independent of training and reference devices", () => {
  const options = { trained_model_id: "trained", evaluation_id: "evaluation", name: "Target", frame_ids: ["frame"] };
  for (const trainingDevice of ["cpu", "cuda:0"]) {
    for (const referenceDevice of ["cpu", "cuda:0"]) {
      for (const targetDevice of ["cpu", "cuda"]) {
        assert.equal(selectionValid({ ...options, training_device: trainingDevice, reference_device: referenceDevice, target_device: targetDevice }), true);
      }
    }
  }
  assert.equal(selectionKey(options), selectionKey({ ...options, target_device: "cpu" }));
});

test("lost export acknowledgements only recover the exact request identity", () => {
  const rows = [{ id: "similar", request_id: "different", name: "Target machine" }, { id: "match", config: { request_id: "approved" } }];
  assert.equal(findExport(rows, "approved").id, "match");
  assert.equal(findExport(rows, "missing"), null);
  assert.equal(findExport(rows, null), null);
  assert.equal(findExport(null, "approved"), null);
  assert.equal(findExport([{ id: "top", request_id: "approved" }], "approved").id, "top");
});

test("lost measurement acknowledgements only recover checked content", () => {
  const rows = [{ id: "other", fingerprint: "other" }, { id: "saved", fingerprint: "checked" }];
  assert.equal(findMeasurement(rows, "checked").id, "saved");
  assert.equal(findMeasurement(rows, "missing"), null);
  assert.equal(findMeasurement(rows, undefined), null);
  assert.equal(findMeasurement(null, "checked"), null);
});

test("measurement imports enforce the bounded file size before reading content", () => {
  assert.match(fileProblem(null), /Choose/);
  assert.match(fileProblem({ size: 0 }), /empty/);
  assert.match(fileProblem({ size: NaN }), /unavailable/);
  assert.match(fileProblem({ size: MAX_MEASUREMENT_BYTES + 1 }), /8 MiB/);
  assert.equal(fileProblem({ size: MAX_MEASUREMENT_BYTES }), null);
  assert.equal(fileProblem({ size: 1 }), null);
});

test("measurement presentation separates exact parity from declared real execution", () => {
  assert.equal(measurementPresentation({ parity_passed: true }).parity, "Exact parity passed");
  assert.equal(measurementPresentation({ parity_passed: false }).parity, "Exact parity failed");
  assert.equal(measurementPresentation({ parity_passed: "true" }).parity, "Parity not verified");
  assert.match(measurementPresentation({ parity_passed: true }).evidence, /not independently verified/);
  assert.match(measurementPresentation({ parity_passed: true, evidence_kind: "simulation" }).evidence, /no real model performance measured/);
  assert.match(measurementPresentation({ provenance: { evidence_kind: "simulation" } }).evidence, /Simulated/);
  assert.match(measurementPresentation({ declaration: "simulation" }).evidence, /no real model performance/);
  assert.match(measurementPresentation({ declaration: "external_execution" }).evidence, /not independently verified/);
  assert.match(measurementPresentation({ evidence_kind: "unknown" }).evidence, /declared by the author/);
});
