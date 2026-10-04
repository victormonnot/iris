"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { taxonomyId, mlSupported, compatibleParents, revisionTokens, classRows,
  modelCompatibility, aggregateFilter, analysisScopes, displayedDetections } =
  require("../../src/iris/static/dataset-tools.js");
const taxonomyTools = require("../../src/iris/static/taxonomy-tools.js");

const custom = {
  id: "helmets-v2", version: 2,
  classes: [
    { id: "helmet", name: "Safety helmet", definition: "Visible protective helmet." },
    { id: "vehicle", name: "Car", definition: "Passenger cars only.", coco_id: 3 },
    { id: "cone", name: "Road cone", definition: "Traffic cones." },
  ],
};

test("release parent selection requires identical class version and project", () => {
  const records = [
    { id: "matching", project_id: "workshop", taxonomy_id: custom.id },
    { id: "old-definitions", project_id: "workshop", taxonomy_id: "helmets-v1" },
    { id: "other-project", project_id: "field", taxonomy_id: custom.id },
    { id: "original", project_id: "workshop", taxonomy_id: "iris-objects-v1" },
  ];
  assert.deepEqual(compatibleParents(records, custom.id, "workshop").map((item) => item.id), ["matching"]);
  assert.deepEqual(compatibleParents(records, "new-version", "workshop"), []);
});

test("runtime eligibility honors the API decision and keeps older records conservative", () => {
  assert.equal(mlSupported({ id: "legacy" }), true);
  assert.equal(taxonomyId({ id: "legacy" }), "iris-objects-v1");
  assert.equal(mlSupported({ taxonomy: custom }), false);
  assert.equal(mlSupported({ manifest: { taxonomy: custom } }), false);
  assert.equal(mlSupported({ taxonomy_id: custom.id }), false);
  assert.equal(mlSupported({ taxonomy_id: "iris-objects-v1", ml_supported: false }), false);
  assert.equal(mlSupported({ taxonomy: custom, ml_supported: true }), true);
  assert.equal(mlSupported(undefined), false);
});

test("official checkpoints train custom classes but evaluate only fully mapped taxonomies", () => {
  const dataset = { taxonomy: custom, ml_supported: true };
  const model = { origin: "official", training: true, classes: [{ id: 1 }, { id: 3 }] };
  assert.equal(modelCompatibility(dataset, model, "training").compatible, true);
  const unsupported = modelCompatibility(dataset, model);
  assert.equal(unsupported.compatible, false);
  assert.match(unsupported.reason, /Safety helmet.*Road cone/);
  const fullyMapped = { ...dataset, taxonomy: { id: "people-v2", classes: [{ id: "worker", name: "Worker", coco_id: 1 }] } };
  assert.equal(modelCompatibility(fullyMapped, model).compatible, true);
  assert.equal(modelCompatibility(dataset, { ...model, training: false }, "training").compatible, false);
});

test("trained parent and evaluation require the exact saved class snapshot", () => {
  const dataset = { taxonomy: custom, ml_supported: true };
  const model = { origin: "trained", taxonomy_id: custom.id, taxonomy: structuredClone(custom), training: true };
  assert.equal(modelCompatibility(dataset, model, "training").compatible, true);
  assert.equal(modelCompatibility(dataset, model).compatible, true);
  // Reordered JSON keys are semantically equal; edited definitions are not.
  model.taxonomy = { classes: custom.classes, version: 2, id: custom.id };
  assert.equal(modelCompatibility(dataset, model).compatible, true);
  model.taxonomy = structuredClone(custom);
  model.taxonomy.classes[0].definition = "A changed annotation rule.";
  assert.equal(modelCompatibility(dataset, model).compatible, false);
  assert.equal(modelCompatibility(dataset, { ...model, taxonomy: undefined }).compatible, false);
  assert.equal(modelCompatibility({ id: "legacy" }, { origin: "trained", taxonomy_id: "iris-objects-v1" }).compatible, true);
});

test("custom analysis keeps a class named all distinct from aggregate results", () => {
  const analysis = { aggregate_filter: "__all__", summary: { __all__: {}, all: {}, helmet: {} } };
  const taxonomy = { id: "edge-case", classes: [{ id: "all", name: "All marker" }, { id: "helmet", name: "Helmet" }] };
  assert.equal(aggregateFilter(analysis), "__all__");
  assert.deepEqual(analysisScopes(analysis, taxonomy), [
    { value: "__all__", label: "All classes" },
    { value: "all", label: "All marker" },
    { value: "helmet", label: "Helmet" },
  ]);
  assert.equal(aggregateFilter({ summary: { all: {} } }), "all");
});

test("overlays display custom labels without confusing ignored official output IDs", () => {
  const detections = [
    { label: "helmet", label_id: 1, score: 0.9 },
    { label: "vehicle", label_id: 2, score: 0.8 },
    { label: "car", label_id: 3, score: 0.95, ignored: true },
    { label: "helmet", label_id: 1, score: 0.99, ignored: true },
    { label: "cone", label_id: 3, score: 0.1 },
  ];
  assert.deepEqual(displayedDetections(custom, detections, 0.5).map((item) => [item.label, item.index]),
    [["helmet", 0], ["vehicle", 1]]);
});

test("freeze captures exactly the viewed revisions of included scene groups", () => {
  const selected = [
    { scene_group: "a", frames: [{ id: "frame-a", annotation_revision_id: "revision-3" }] },
    { scene_group: "b", frames: [{ id: "frame-b", annotation_revision_id: "revision-7" }] },
  ];
  const captured = revisionTokens(selected);
  selected[0].frames[0].annotation_revision_id = "revision-4";
  assert.deepEqual(captured, { "frame-a": "revision-3", "frame-b": "revision-7" });
  assert.throws(() => revisionTokens([{ frames: [{ id: "frame-c", revision: 4 }] }]), /Refresh dataset candidates/);
  assert.throws(() => revisionTokens([selected[0], selected[0]]), /Refresh dataset candidates/);
});

test("saved class display includes zero-count classes and separates all three numeric mappings", () => {
  const record = {
    taxonomy: custom,
    class_mapping: { helmet: 1, vehicle: 2, cone: 3 },
    coco_mapping: { helmet: 1, vehicle: 2, cone: 3 },
    summary: { class_counts: { helmet: 5, vehicle: 2 } },
  };
  const rows = classRows(record);
  assert.equal(rows.length, 3);
  assert.equal(rows[0].name, "Safety helmet");
  assert.equal(rows[0].source_coco_id, null);
  assert.equal(rows[1].class_id, 2);
  assert.equal(rows[1].export_id, 2);
  assert.equal(rows[1].source_coco_id, 3);
  assert.equal(rows[2].count, 0);
  assert.equal(rows[2].export_id, 3);
});

test("old release display preserves the original person/car export categories", () => {
  const rows = classRows({ summary: { class_counts: { person: 2 } } }, taxonomyTools.snapshot());
  assert.equal(rows[0].export_id, 1);
  assert.equal(rows[1].class_id, 2);
  assert.equal(rows[1].export_id, 3);
  assert.equal(rows[1].count, 0);
});
