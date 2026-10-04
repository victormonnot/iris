"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { taxonomyId, mlSupported, compatibleParents, revisionTokens, classRows } =
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

test("training and evaluation compatibility keeps legacy releases usable and excludes custom ones", () => {
  assert.equal(mlSupported({ id: "legacy" }), true);
  assert.equal(taxonomyId({ id: "legacy" }), "iris-objects-v1");
  assert.equal(mlSupported({ taxonomy: custom }), false);
  assert.equal(mlSupported({ manifest: { taxonomy: custom } }), false);
  assert.equal(mlSupported({ taxonomy_id: custom.id }), false);
  assert.equal(mlSupported({ taxonomy_id: "iris-objects-v1", ml_supported: false }), false);
  assert.equal(mlSupported(undefined), false);
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
