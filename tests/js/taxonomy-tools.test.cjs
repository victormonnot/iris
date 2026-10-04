"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { snapshot, editableClasses, className, hasClass, mappingComplete, versionLabel } =
  require("../../src/iris/static/taxonomy-tools.js");

const original = {
  id: "project-classes-v1", version: 1,
  classes: [
    { id: "helmet", name: "Helmet", definition: "A visible protective helmet." },
    { id: "vehicle", name: "Vehicle", definition: "Passenger cars only.", coco_id: 3 },
  ],
};

test("editing names and definitions leaves pinned annotation and import snapshots intact", () => {
  const pinned = snapshot(original);
  const edit = editableClasses(pinned);
  edit[0].name = "Safety helmet";
  edit[0].definition = "Include empty helmets as well as worn helmets.";
  edit[1].coco_id = null;
  edit.push({ id: "cone", name: "Cone", definition: "Road cones." });
  assert.equal(pinned.classes.length, 2);
  assert.equal(className(pinned, "helmet"), "Helmet");
  assert.equal(pinned.classes[0].definition, "A visible protective helmet.");
  assert.equal(pinned.classes[1].coco_id, 3);
  assert.equal(edit[0].id, pinned.classes[0].id);
  assert.equal(edit[0].coco_id, null);
});

test("COCO import needs an explicit choice for every source category in its pinned version", () => {
  const categories = [{ id: 8, name: "Helmet" }, { id: 13, name: "bicycle" }];
  assert.equal(mappingComplete(categories, {}, original), false);
  assert.equal(mappingComplete(categories, { 8: "helmet" }, original), false);
  assert.equal(mappingComplete(categories, { 8: "helmet", 13: "exclude" }, original), true);
  assert.equal(mappingComplete(categories, { 8: "Helmet", 13: "exclude" }, original), false);
  assert.equal(mappingComplete(categories, { 8: "cone", 13: "exclude" }, original), false);
  // Category names, COCO numbers and later project classes never imply a mapping.
  assert.equal(mappingComplete(categories, { 8: "1", 13: "exclude" }, original), false);
  assert.equal(mappingComplete([], {}, original), true);
});

test("removed builtin classes cannot be accepted as proposals for a custom-only image", () => {
  assert.equal(hasClass(original, "person"), false);
  assert.equal(hasClass(original, "car"), false);
  assert.equal(hasClass(original, "helmet"), true);
  assert.equal(className(original, "person"), "person");
  assert.equal(className(original, "vehicle"), "Vehicle");
});

test("historical imports without a snapshot use isolated copies of original definitions", () => {
  const first = snapshot();
  const second = snapshot();
  first.classes[0].name = "Changed in local form";
  assert.equal(second.id, "iris-objects-v1");
  assert.equal(second.classes[0].name, "Person");
  assert.equal(second.classes[1].coco_id, 3);
  assert.equal(versionLabel(second), "Original person / car classes");
  assert.equal(versionLabel(original), "Class version 1");
});
