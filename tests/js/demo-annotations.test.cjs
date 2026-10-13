"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const domain = require("../../src/iris/static/demo/annotations.js");
const seed = require("../../examples/web-demo/seed/fixture.json");

const sourceFrames = seed.responses[`/api/sessions/${seed.session_id}/frames`];
const [FIRST, SECOND] = sourceFrames.map((frame) => frame.id);
const fresh = () => ({ ...domain.createState(seed), epoch: "test-epoch" });
const document = (state, id = FIRST) => domain.document(seed, state, id);
const manual = (id = "manual") => ({ id, label: "person", box: [0, 0, 10, 20], suggestion_id: null });
const proposalBox = (item, id = item.id) => ({ id, label: item.label, box: [...item.box], suggestion_id: item.id });
const payload = (doc, changes = {}) => ({
  expected_revision: doc.revision, demo_epoch: doc.demo_epoch, taxonomy_id: doc.taxonomy_id,
  boxes: doc.boxes.map(({ id, label, box, suggestion_id }) => ({ id, label, box, suggestion_id })),
  decisions: { ...doc.decisions }, status: "draft", reviewer: "", notes: "", ...changes,
});

function rejectsUnchanged(state, input, status = 422, message) {
  const before = structuredClone(state);
  assert.throws(() => domain.save(seed, state, FIRST, input), (error) =>
    error.status === status && (!message || message.test(error.message)));
  assert.deepEqual(state, before);
}

test("browser annotation editing needs no network, DOM, or CommonJS", () => {
  const window = {};
  vm.runInNewContext(readFileSync(join(__dirname, "../../src/iris/static/demo/annotations.js"), "utf8"), {
    window, globalThis: window, structuredClone,
  });
  assert.equal(typeof window.IRISDemoAnnotations.save, "function");
  assert.equal(typeof window.IRISDemoAnnotations.document, "function");
});

test("initial documents preserve real images and all pending source proposals", () => {
  const state = fresh();
  assert.deepEqual(state.annotations, {});
  assert.deepEqual(state.selection, { [FIRST]: true, [SECOND]: true });
  assert.equal(Object.hasOwn(domain.createState(seed), "epoch"), false);
  const documents = [document(state), document(state, SECOND)];
  assert.deepEqual(documents.map((doc) => doc.suggestions), [FIRST, SECOND].map((id) => seed.responses[`/api/frames/${id}/annotation`].suggestions));
  for (const doc of documents) {
    assert.equal(doc.revision, 0);
    assert.equal(doc.status, "unannotated");
    assert.equal(doc.demo_epoch, state.epoch);
    assert.ok(doc.suggestions.every((item) => item.state === "pending"));
    assert.deepEqual(doc.boxes, []);
  }
  documents[0].suggestions[0].metadata.source_annotation.bbox[0] = -100;
  assert.ok(document(state).suggestions[0].metadata.source_annotation.bbox[0] >= 0);
  state.selection[FIRST] = false;
  assert.equal(document(state).frame.selected, false);
});

test("accept, correct, reject and manually add labels with immutable provenance and revisions", () => {
  // Exercise all decision paths even when the public sample has one proposal.
  const scenario = structuredClone(seed);
  const source = scenario.responses[`/api/frames/${FIRST}/annotation`];
  source.suggestions = Array.from({ length: 3 }, (_, index) => ({
    ...structuredClone(source.suggestions[0]), id: `synthetic-proposal-${index}`,
  }));
  const state = fresh();
  const original = domain.document(scenario, state, FIRST);
  const [accepted, corrected, rejected] = original.suggestions;
  const changed = proposalBox(corrected);
  changed.box[0] += 1;
  const input = payload(original, {
    boxes: [proposalBox(accepted), changed, manual()],
    decisions: { [accepted.id]: "accepted", [corrected.id]: "corrected", [rejected.id]: "rejected" },
    status: "validated", reviewer: "  Demo visitor  ", notes: "Reviewed in the browser",
  });
  const saved = domain.save(scenario, state, FIRST, input);
  assert.equal(saved.revision, 1);
  assert.equal(saved.status, "validated");
  assert.equal(saved.reviewer, "Demo visitor");
  assert.deepEqual(saved.boxes.map((box) => box.review_state), ["accepted", "corrected", "manual"]);
  assert.deepEqual(saved.suggestions.map((item) => item.state), ["accepted", "corrected", "rejected"]);
  assert.deepEqual(saved.boxes[0].source.metadata, accepted.metadata);
  assert.deepEqual(saved.boxes[1].source.metadata.source_annotation, corrected.metadata.source_annotation);
  assert.deepEqual(saved.boxes[2].source, { kind: "manual" });
  assert.equal(saved.history[0].box_count, 3);
  assert.equal(saved.history[0].revision, 1);
  assert.ok(Number.isFinite(Date.parse(saved.history[0].created_at)));
  assert.equal(document(state, SECOND).revision, 0);
  const firstRevision = domain.revision(scenario, state, FIRST, 1);
  const next = domain.save(scenario, state, FIRST, payload(saved, { notes: "Back to a draft", reviewer: "Visitor" }));
  assert.equal(next.status, "draft");
  assert.deepEqual(next.history.map((item) => item.revision), [2, 1]);
  assert.deepEqual(domain.revision(scenario, state, FIRST, 1), firstRevision);
  assert.equal(state.annotations[FIRST].revisions[0].boxes[0].source, undefined);
  input.boxes[0].box[0] = -50;
  saved.boxes[0].source.metadata.source_annotation.bbox[0] = -100;
  assert.deepEqual(domain.revision(scenario, state, FIRST, 1), firstRevision);
});

test("changing a proposed class is a correction even when coordinates stay identical", () => {
  const scenario = structuredClone(seed);
  scenario.responses[`/api/frames/${FIRST}/annotation`].taxonomy.classes.push({
    id: "synthetic_other", name: "Other", definition: "Synthetic alternative for this test.", coco_id: null,
  });
  const state = fresh();
  const doc = domain.document(scenario, state, FIRST);
  const suggestion = doc.suggestions[0];
  const alternate = doc.taxonomy.classes.find((item) => item.id !== suggestion.label);
  assert.ok(alternate, "The test needs a second declared class");
  const box = { ...proposalBox(suggestion), label: alternate.id };
  const before = structuredClone(state);
  assert.throws(() => domain.save(scenario, state, FIRST, payload(doc, {
    boxes: [box], decisions: { [suggestion.id]: "accepted" },
  })), (error) => error.status === 422);
  assert.deepEqual(state, before);
  const saved = domain.save(scenario, state, FIRST, payload(doc, {
    boxes: [box], decisions: { [suggestion.id]: "corrected" },
  }));
  assert.equal(saved.boxes[0].review_state, "corrected");
  assert.equal(saved.boxes[0].source.metadata.source_category.name, suggestion.metadata.source_category.name);
});

test("a complete reviewer decision can validate an empty image without inventing labels", () => {
  const state = fresh();
  const doc = document(state, SECOND);
  const saved = domain.save(seed, state, SECOND, payload(doc, {
    boxes: [], decisions: Object.fromEntries(doc.suggestions.map((item) => [item.id, "rejected"])),
    status: "validated", reviewer: "Visitor",
  }));
  assert.equal(saved.status, "validated");
  assert.deepEqual(saved.boxes, []);
  assert.ok(saved.suggestions.every((item) => item.state === "rejected"));
});

test("human validation requires a reviewer and a decision for every pending suggestion", () => {
  const state = fresh();
  const doc = document(state);
  rejectsUnchanged(state, payload(doc, { status: "validated" }), 422, /reviewer/);
  rejectsUnchanged(state, payload(doc, { status: "validated", reviewer: "Visitor" }), 422, /pending/);
  rejectsUnchanged(state, payload(doc, { status: "validated", reviewer: "  \n " }), 422, /reviewer/);
  assert.equal(domain.save(seed, state, FIRST, payload(doc)).status, "draft");
});

test("same-image stale saves and pre-reset editors conflict even when revision numbers coincide", () => {
  const state = fresh();
  const stale = payload(document(state), { boxes: [manual()] });
  domain.save(seed, state, FIRST, stale);
  rejectsUnchanged(state, stale, 409, /changed/);
  const reset = { ...domain.createState(seed), epoch: "new-epoch" };
  rejectsUnchanged(reset, stale, 409, /reset/);
  const missingEpoch = payload(document(reset));
  delete missingEpoch.demo_epoch;
  rejectsUnchanged(reset, missingEpoch, 409);
  rejectsUnchanged(reset, payload(document(reset), { taxonomy_id: "different-taxonomy" }), 409);
});

test("revision counters, text fields and statuses are validated before any mutation", () => {
  const state = fresh();
  const doc = document(state);
  for (const expected_revision of [true, -1, 0.5, "0", null, NaN, Infinity])
    rejectsUnchanged(state, payload(doc, { expected_revision }));
  for (const changes of [
    { status: "unannotated" }, { status: null }, { reviewer: {} }, { reviewer: "a".repeat(121) },
    { notes: null }, { notes: "n".repeat(4001) }, { boxes: null }, { decisions: [] },
  ]) rejectsUnchanged(state, payload(doc, changes));
});

test("text length limits count Unicode characters as the Python annotation API does", () => {
  const state = fresh();
  const doc = document(state);
  const reviewer = "🙂".repeat(120);
  const notes = "🙂".repeat(4000);
  const saved = domain.save(seed, state, FIRST, payload(doc, {
    reviewer, notes, boxes: [manual("🙂".repeat(128))],
  }));
  assert.equal(saved.reviewer, reviewer);
  assert.equal(saved.notes, notes);
  rejectsUnchanged(state, payload(saved, { reviewer: `${reviewer}🙂` }));
  rejectsUnchanged(state, payload(saved, { notes: `${notes}🙂` }));
});

test("boxes require finite original-image coordinates, positive area, valid classes and unique IDs", () => {
  const state = fresh();
  const doc = document(state);
  for (const box of [
    [NaN, 0, 10, 20], [0, 0, Infinity, 20], [false, 0, 10, 20], ["0", 0, 10, 20],
    [-1, 0, 10, 20], [0, -1, 10, 20], [0, 0, doc.frame.width + 1, 20], [0, 0, 10, doc.frame.height + 1],
    [10, 0, 10, 20], [11, 0, 10, 20], [0, 20, 10, 20], [0, 0, 1], null,
  ]) rejectsUnchanged(state, payload(doc, { boxes: [{ ...manual(), box }] }));
  for (const box of [
    { ...manual(), id: " " }, { ...manual(), id: "x".repeat(129) }, { ...manual(), id: 5 },
    { ...manual(), label: "truck" }, { ...manual(), label: 1 }, null,
  ]) rejectsUnchanged(state, payload(doc, { boxes: [box] }));
  rejectsUnchanged(state, payload(doc, { boxes: [manual(), manual()] }));
  rejectsUnchanged(state, payload(doc, { boxes: Array.from({ length: 501 }, (_, i) => manual(`id-${i}`)) }));
  const saved = domain.save(seed, state, FIRST, payload(doc, {
    boxes: [{ ...manual(), box: [0, 0, doc.frame.width, doc.frame.height] }],
  }));
  assert.deepEqual(saved.boxes[0].box, [0, 0, doc.frame.width, doc.frame.height]);
});

test("suggestion links, decisions and persisted origins cannot be forged or silently discarded", () => {
  const state = fresh();
  const doc = document(state);
  const suggestion = doc.suggestions[0];
  const accepted = proposalBox(suggestion);
  const foreign = document(state, SECOND).suggestions[0];
  rejectsUnchanged(state, payload(doc, { boxes: [proposalBox(foreign)], decisions: { [foreign.id]: "accepted" } }));
  rejectsUnchanged(state, payload(doc, { boxes: [accepted], decisions: {} }));
  rejectsUnchanged(state, payload(doc, { boxes: [], decisions: { [suggestion.id]: "accepted" } }));
  rejectsUnchanged(state, payload(doc, { boxes: [accepted], decisions: { [suggestion.id]: "rejected" } }));
  rejectsUnchanged(state, payload(doc, { boxes: [accepted, { ...accepted, id: "different-id" }], decisions: { [suggestion.id]: "accepted" } }));
  rejectsUnchanged(state, payload(doc, { decisions: { [suggestion.id]: "pending" } }));
  rejectsUnchanged(state, payload(doc, { decisions: { unknown: "rejected" } }));
  const saved = domain.save(seed, state, FIRST, payload(doc, {
    boxes: [{ ...accepted, source: { kind: "fabricated" }, review_state: "manual" }],
    decisions: { [suggestion.id]: "accepted" },
  }));
  assert.equal(saved.boxes[0].source.kind, "imported");
  rejectsUnchanged(state, payload(saved, { boxes: [], decisions: {} }), 422, /previous/);
  rejectsUnchanged(state, payload(saved, {
    boxes: [{ ...accepted, suggestion_id: null }], decisions: { [suggestion.id]: "rejected" },
  }), 422, /origin/);
  const deleted = domain.save(seed, state, FIRST, payload(saved, {
    boxes: [], decisions: { [suggestion.id]: "rejected" },
  }));
  assert.equal(deleted.suggestions[0].state, "rejected");
  assert.equal(domain.revision(seed, state, FIRST, 1).boxes[0].source.kind, "imported");
});

test("unknown images and revisions report 404 without creating data", () => {
  const state = fresh();
  assert.throws(() => domain.document(seed, state, "missing"), { status: 404 });
  assert.throws(() => domain.save(seed, state, "missing", {}), { status: 404 });
  for (const revision of [0, 1, "1", -1])
    assert.throws(() => domain.revision(seed, state, FIRST, revision), { status: 404 });
  assert.deepEqual(state.annotations, {});
});
