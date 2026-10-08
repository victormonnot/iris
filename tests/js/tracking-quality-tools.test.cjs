"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/tracking-quality-tools.js");
const taxonomy = { id: "sequence-taxonomy", classes: [{ id: "person", name: "Person", coco_id: 0 }, { id: "vehicle", name: "Car", coco_id: 2 }] };

test("quality mapping uses explicit frozen class semantics and never name guesses", () => {
  const official = { class_contract: { taxonomy_id: "coco-2017-v1" }, classes: [{ id: 0, name: "Misleading name" }, { id: 99, name: "Person" }] };
  assert.deepEqual(tools.suggestions(official, taxonomy, [0, 2, 99]), { 0: "person", 2: "vehicle" });
  const trained = { class_contract: { taxonomy_id: taxonomy.id, taxonomy: { classes: taxonomy.classes, id: taxonomy.id }, output_class_mapping: { person: 7, vehicle: 11 } } };
  assert.deepEqual(tools.suggestions(trained, taxonomy, [7, 11]), { 7: "person", 11: "vehicle" }, "equivalent frozen taxonomy field order is immaterial");
  trained.class_contract.taxonomy = { ...taxonomy, id: "foreign-taxonomy" };
  assert.deepEqual(tools.suggestions(trained, taxonomy, [7, 11]), {});
  assert.deepEqual(tools.suggestions({ classes: [{ id: 0, name: "Person" }] }, taxonomy, [0]), {});
});

test("quality calculation requires an explicit revision and every native class mapping", () => {
  const mapping = { 0: "person", 2: null };
  const result = tools.configuration("reference-r3", [0, 2], mapping, 0.5, taxonomy);
  assert.deepEqual(result, { reference_id: "reference-r3", class_mapping: { 0: "person", 2: null }, iou_threshold: 0.5 });
  result.class_mapping[0] = "vehicle";
  assert.equal(mapping[0], "person", "request preparation does not mutate input form state");
  assert.throws(() => tools.configuration("", [0, 2], mapping, 0.5, taxonomy), /explicit saved/);
  assert.throws(() => tools.configuration("r", [0, 2], { 0: "person" }, 0.5, taxonomy), /every native/);
  assert.throws(() => tools.configuration("r", [0, 2], { ...mapping, 99: null }, 0.5, taxonomy), /every native/);
  assert.throws(() => tools.configuration("r", [0, 2], { 0: "", 2: null }, 0.5, taxonomy), /taxonomy class/);
  assert.throws(() => tools.configuration("r", [0, 2], { 0: null, 2: null }, 0.5, taxonomy), /at least one/);
  assert.throws(() => tools.configuration("r", [0, 2], { 0: "foreign", 2: null }, 0.5, taxonomy), /taxonomy class/);
});

test("IoU threshold follows the server interval without silently clipping", () => {
  for (const invalid of [0, -0.1, 1.01, NaN, Infinity, "0.5", null]) assert.throws(() => tools.configuration("r", [0], { 0: "person" }, invalid, taxonomy), /IoU/);
  for (const valid of [0.0001, 0.5, 1]) assert.equal(tools.configuration("r", [0], { 0: "person" }, valid, taxonomy).iou_threshold, valid);
});

test("saved report context requires matching outer and frozen comparison, sequence and reference IDs", () => {
  const context = { comparison_id: "comparison-a", sequence_id: "sequence-a" };
  const record = { ...context, reference_id: "reference-r3", report: { source: { ...context, reference_id: "reference-r3", reference_revision: 3 } } };
  assert.equal(tools.matchesContext(record, context), true);
  assert.equal(tools.matchesContext(record, { ...context, sequence_id: "sequence-b" }), false);
  assert.equal(tools.matchesContext(record, { ...context, comparison_id: "comparison-b" }), false);
  assert.equal(tools.matchesContext({ ...record, report: { source: { ...record.report.source, comparison_id: "comparison-b" } } }, context), false);
  assert.equal(tools.matchesContext({ ...record, reference_id: "reference-r4" }, context), false, "latest revision cannot replace the report's frozen reference");
  assert.equal(tools.matchesContext({ ...record, report: {} }, context), false);
  assert.equal(tools.matchesContext(null, context), false);
});

test("undefined rates and unavailable identity evidence never display as zero or perfect", () => {
  assert.equal(tools.percentage(null), "Not defined");
  assert.equal(tools.percentage(undefined), "Not defined");
  assert.equal(tools.percentage(NaN), "Not defined");
  assert.equal(tools.percentage(0), "0.0%");
  assert.equal(tools.percentage(1), "100.0%");
  assert.equal(tools.percentage(0.125), "12.5%");
  assert.match(tools.reason("incomplete_reference_coverage"), /fully dense.*whole source clip/);
  assert.match(tools.reason("no_evaluated_frames"), /No fully reviewed/);
  assert.match(tools.reason("no_ground_truth_identity_detections"), /No ground-truth/);
  assert.equal(tools.reason("future_protocol_reason"), "future protocol reason");
});

test("native scope unions both frozen profiles without relying on display filters", () => {
  const report = { lanes: [{ report: { profile: { class_ids: [2, 0] } } }, { report: { profile: { class_ids: [3, 2] } } }] };
  assert.deepEqual(tools.nativeClasses(report), [0, 2, 3]);
  assert.deepEqual(report.lanes[0].report.profile.class_ids, [2, 0]);
});

test("revision labels retain saved author and human coverage; event labels keep lane IDs local", () => {
  assert.equal(tools.referenceLabel({ revision: 3, summary: { human_complete_frames: 8, available_frames: 8 }, payload: { provenance: { author: "Victor" } } }), "Revision 3 · 8/8 complete human frames · Victor");
  assert.match(tools.eventText({ kind: "identity_switch", reference_identity: "ref_person", track_id: 0, previous_track_id: 9 }), /ref_person.*9 → 0/);
  assert.match(tools.eventText({ kind: "identity_transfer", reference_identity: "ref_b", previous_reference_identity: "ref_a", track_id: 0 }), /Tracker ID 0: ref_a → ref_b/);
  assert.match(tools.eventText({ kind: "fragment", reference_identity: "ref_person", track_id: 0 }), /matched again.*0.*missed observation/);
});
