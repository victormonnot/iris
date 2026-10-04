"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/benchmark-sam-tools.js");

const taxonomy = { id: "frozen-v2", classes: [
  { id: "helmet", name: "Safety helmet", definition: "A protective hard hat." },
  { id: "constructor", name: "Construction worker", definition: "A worker in the scene." },
] };

test("SAM prompts use frozen classes and survive polling only for the same benchmark snapshot", () => {
  const draft = tools.promptState(null, "reference-a", taxonomy);
  assert.equal(draft.values.get("helmet"), "Safety helmet");
  draft.values.set("helmet", "yellow hard hat");
  assert.equal(tools.promptState(draft, "reference-a", structuredClone(taxonomy)), draft);
  const other = tools.promptState(draft, "reference-b", taxonomy);
  assert.equal(other.values.get("helmet"), "Safety helmet");
  const revised = structuredClone(taxonomy); revised.classes[0].definition = "A changed class definition.";
  assert.notEqual(tools.promptState(draft, "reference-a", revised), draft);
});

test("prompt payloads include every saved class exactly and preserve literal class IDs", () => {
  const draft = tools.promptState(null, "reference", taxonomy);
  draft.values.set("helmet", "  protective helmet  "); draft.values.set("obsolete", "old class");
  const payload = tools.promptPayload(draft, taxonomy);
  assert.deepEqual(Object.keys(payload), ["helmet", "constructor"]);
  assert.equal(payload.helmet, "protective helmet");
  assert.equal(JSON.parse(JSON.stringify(payload)).constructor, "Construction worker");
  assert.equal(tools.promptError(draft, taxonomy), "");
  draft.values.delete("constructor");
  assert.match(tools.promptError(draft, taxonomy), /Construction worker/);
});

test("blank or overlong class prompts block configuration preparation", () => {
  const draft = tools.promptState(null, "reference", taxonomy);
  for (const text of ["", "   ", "x".repeat(121)]) {
    draft.values.set("helmet", text);
    assert.match(tools.promptError(draft, taxonomy), /Safety helmet/);
  }
  draft.values.set("helmet", "hard hat");
  assert.match(tools.promptError(draft, taxonomy, { max_classes: 1 }), /number of class prompts/);
  assert.match(tools.promptError(draft, null), /independent reference/);
});

test("SAM launch requires a positively checked local plan while unavailable previews block every approach", () => {
  assert.equal(tools.launchAllowed({ launch_allowed: true }, "segmentation"), true);
  for (const preview of [null, {}, { launch_allowed: false }, { launch_allowed: "true" }])
    assert.equal(tools.launchAllowed(preview, "segmentation"), false);
  for (const approach of ["local_detector", "multimodal", "segmentation"])
    assert.equal(tools.launchAllowed({ launch_allowed: false }, approach), false);
  assert.equal(tools.launchAllowed({}, "local_detector"), true, "legacy detector previews do not contain availability fields");
});

test("setup availability never claims verified inference and work is counted without duration estimates", () => {
  assert.match(tools.availability({ status: "missing_weights", reason: "Weights absent" }), /Setup required.*Weights absent/);
  assert.match(tools.availability({ status: "ready" }), /not evidence of a successful model run/);
  assert.match(tools.availability(null), /unavailable/);
  assert.equal(tools.workSummary({ image_encodings: 2, prompt_evaluations: 6, warmup_passes: 0 }), "2 image encodings · 6 class-prompt evaluations · 0 warm-up passes");
});
