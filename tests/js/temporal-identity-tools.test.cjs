"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/temporal-identity-tools.js");

function sequence(indices = [0, 1, 2]) {
  return {
    manifest_sha256: "a".repeat(64),
    manifest: {
      id: "sequence-a", taxonomy: { id: "taxonomy-a" },
      clip: { start_frame: indices[0], end_frame: indices.at(-1) },
      frames: indices.map((frame_index) => ({ frame_index, frame_id: `frame-${frame_index}`, width: 100, height: 80 })),
    },
  };
}
function object(identity_id = "ref_a", patch = {}) {
  return { identity_id, label: "person", box: [1, 2, 10, 20], visibility: "visible", certainty: "certain", ...patch };
}
function fixture(indices = [0, 1, 2]) {
  let payload = tools.newIdentity(tools.blank(sequence(indices)), "person", "ref_a");
  payload.provenance = { author: "Initial author", origin: { track_mapping: { "7": "ref_a" }, comparison_id: "frozen-source" } };
  for (const frameIndex of indices) {
    payload = tools.addObject(payload, frameIndex, object());
    payload = tools.confirmFrame(payload, frameIndex, "Original reviewer", "complete");
  }
  return payload;
}
function freeze(value) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}
function reviewed(payload) {
  return payload.frames.filter((frame) => frame.review.status === "human_reviewed").map((frame) => frame.frame_index);
}

test("blank drafts bind the frozen manifest and preserve unavailable source gaps as unknown", () => {
  const record = freeze(sequence([0, 4, 9])), payload = tools.blank(record);
  assert.equal(payload.schema, "iris-temporal-reference-v2");
  assert.equal(payload.sequence_id, record.manifest.id);
  assert.equal(payload.sequence_sha256, record.manifest_sha256);
  assert.equal(payload.taxonomy_id, record.manifest.taxonomy.id);
  assert.deepEqual(payload.frames.map((frame) => frame.frame_index), [0, 4, 9]);
  assert.deepEqual(payload.provenance, { author: "", origin: null });
  assert.ok(payload.frames.every((frame) => frame.coverage === "unreviewed" && frame.review.status === "unreviewed"));
  assert.equal(tools.summary(payload, record.manifest).dense_human_reference, false);
  assert.throws(() => tools.blank({ manifest: record.manifest }), /manifest checksum/);
});

test("object correction resets only actual changed frame review and preserves immutable seed evidence", () => {
  const input = freeze(fixture()), before = tools.clone(input);
  const output = tools.editObject(input, 1, 0, { box: [2, 3, 11, 21] });
  assert.deepEqual(input, before);
  assert.deepEqual(reviewed(output), [0, 2]);
  assert.deepEqual(output.frames[1].review, { status: "unreviewed", reviewer: "" });
  assert.equal(output.frames[1].coverage, "unreviewed");
  assert.deepEqual(output.provenance, input.provenance);
  assert.notEqual(output.provenance, input.provenance);
  assert.deepEqual(tools.changedFrames(input, output), [1]);
  const noOp = tools.editObject(input, 1, 0, { box: [1, 2, 10, 20] });
  assert.deepEqual(reviewed(noOp), [0, 1, 2]);
  assert.deepEqual(tools.changedFrames(input, noOp), []);
});

test("visibility edits discard inferred positions and uncertainty never promotes human review", () => {
  const input = freeze(fixture());
  const unknown = tools.editObject(input, 1, 0, { visibility: "unknown" });
  assert.equal(unknown.frames[1].objects[0].box, null);
  assert.equal(unknown.frames[1].objects[0].certainty, "uncertain");
  assert.equal(unknown.frames[1].review.status, "unreviewed");
  const outside = tools.editObject(input, 1, 0, { visibility: "out_of_view" });
  assert.equal(outside.frames[1].objects[0].box, null);
  const occluded = tools.editObject(input, 1, 0, { visibility: "occluded", box: null });
  assert.equal(occluded.frames[1].objects[0].box, null);
  assert.equal(occluded.frames[1].objects[0].certainty, "certain");
  assert.deepEqual(reviewed(occluded), [0, 2]);
  const unidentified = tools.assignObject(input, 1, 0, null);
  assert.equal(unidentified.frames[1].objects[0].identity_id, null);
  assert.equal(unidentified.frames[1].objects[0].certainty, "uncertain");
  assert.throws(() => tools.editObject(input, 1, 0, { box: null }), /visible object needs/);
});

test("association keeps identity classes coherent and rejects same-frame duplicate identities atomically", () => {
  let input = tools.newIdentity(fixture(), "car", "ref_car");
  input = tools.newIdentity(input, "person", "ref_b");
  input = tools.addObject(input, 1, object("ref_b"));
  freeze(input);
  const before = tools.clone(input);
  assert.throws(() => tools.assignObject(input, 1, 1, "ref_a"), /twice in the same frame/);
  assert.deepEqual(input, before);
  const reassigned = tools.assignObject(input, 0, 0, "ref_car");
  assert.equal(reassigned.frames[0].objects[0].label, "car");
  assert.equal(reassigned.frames[0].review.status, "unreviewed");
  assert.throws(() => tools.assignObject(input, 0, 0, "missing"), /declared reference identity/);
  assert.throws(() => tools.editObject(input, 0, 0, { label: "car" }), /class must match/);
});

test("adding and removing an object invalidates reviewed negatives and never silently drops another object", () => {
  let input = tools.newIdentity(tools.blank(sequence()), "person", "ref_a");
  input = tools.confirmFrame(input, 1, "Reviewer", "complete");
  freeze(input);
  const added = tools.addObject(input, 1, object());
  assert.equal(added.frames[1].objects.length, 1);
  assert.equal(added.frames[1].review.status, "unreviewed");
  assert.deepEqual(tools.changedFrames(input, added), [1]);
  const removed = tools.removeObject(tools.confirmFrame(added, 1, "Reviewer", "complete"), 1, 0);
  assert.equal(removed.frames[1].objects.length, 0);
  assert.equal(removed.frames[1].coverage, "unreviewed", "removing the last box does not declare an empty negative");
  assert.throws(() => tools.removeObject(added, 1, 2), /existing reference object/);
  assert.throws(() => tools.addObject(added, 1, object()), /twice in the same frame/);
  assert.equal(input.frames[1].objects.length, 0);
});

test("split uses source indices across gaps and changes only the nonempty later segment", () => {
  const input = freeze(fixture([3, 20, 90]));
  const output = tools.splitIdentity(input, "ref_a", 20, "ref_b");
  assert.deepEqual(output.frames.map((frame) => frame.objects[0].identity_id), ["ref_a", "ref_b", "ref_b"]);
  assert.deepEqual(reviewed(output), [3]);
  assert.deepEqual(tools.changedFrames(input, output), [20, 90]);
  assert.deepEqual(output.identities, [{ id: "ref_a", label: "person" }, { id: "ref_b", label: "person" }]);
  assert.deepEqual(output.provenance, input.provenance);
  assert.deepEqual(input.frames.map((frame) => frame.objects[0].identity_id), ["ref_a", "ref_a", "ref_a"]);
  assert.throws(() => tools.splitIdentity(input, "ref_a", 3, "ref_b"), /both before and from/);
  assert.throws(() => tools.splitIdentity(input, "ref_a", 91, "ref_b"), /both before and from/);
  assert.throws(() => tools.splitIdentity(input, "ref_a", 20, "ref_a"), /new, distinct/);
  assert.throws(() => tools.splitIdentity(input, "ref_a", 1.5, "ref_b"), /nonnegative integer/);
});

test("merge preserves every observation, resets changed frames and retains historical seed mapping", () => {
  const input = freeze(tools.splitIdentity(fixture([2, 4, 8]), "ref_a", 4, "ref_b"));
  let reviewedInput = tools.confirmFrame(input, 4, "Second reviewer", "complete");
  reviewedInput = tools.confirmFrame(reviewedInput, 8, "Second reviewer", "complete");
  freeze(reviewedInput);
  const output = tools.mergeIdentities(reviewedInput, "ref_a", "ref_b");
  assert.equal(output.identities.length, 1);
  assert.equal(output.frames.flatMap((frame) => frame.objects).length, 3);
  assert.ok(output.frames.every((frame) => frame.objects[0].identity_id === "ref_a"));
  assert.deepEqual(reviewed(output), [2]);
  assert.deepEqual(tools.changedFrames(reviewedInput, output), [4, 8]);
  assert.deepEqual(output.provenance, reviewedInput.provenance);
});

test("merge refuses same-frame cooccurrence, differing classes and self-merges without mutating input", () => {
  let input = tools.newIdentity(fixture(), "person", "ref_b");
  input = tools.newIdentity(input, "car", "ref_car");
  input = tools.addObject(input, 2, object("ref_b", { visibility: "occluded", box: null }));
  freeze(input);
  const before = tools.clone(input);
  assert.throws(() => tools.mergeIdentities(input, "ref_a", "ref_b"), /source frame 2.*overlap/);
  assert.throws(() => tools.mergeIdentities(input, "ref_a", "ref_car"), /same class/);
  assert.throws(() => tools.mergeIdentities(input, "ref_a", "ref_a"), /two distinct/);
  assert.deepEqual(input, before);
});

test("complete human confirmation requires an explicit valid reviewer and known identity and visibility", () => {
  let input = tools.newIdentity(tools.blank(sequence()), "person", "ref_a");
  input = tools.addObject(input, 1, object("ref_a", { certainty: "uncertain" }));
  freeze(input);
  assert.throws(() => tools.confirmFrame(input, 1, "  ", "complete"), /Human reviewer/);
  assert.throws(() => tools.confirmFrame(input, 1, "Reviewer", "unreviewed"), /complete or partial/);
  const partial = tools.confirmFrame(input, 1, "Reviewer", "partial");
  assert.equal(partial.frames[1].objects[0].certainty, "uncertain");
  assert.equal(partial.frames[1].review.status, "human_reviewed");
  const complete = tools.confirmFrame(input, 1, "Reviewer", "complete");
  assert.equal(complete.frames[1].objects[0].certainty, "certain");
  assert.equal(complete.frames[1].coverage, "complete");
  assert.equal(input.frames[1].review.status, "unreviewed");
  const unknown = tools.editObject(input, 1, 0, { visibility: "unknown" });
  const unidentified = tools.assignObject(input, 1, 0, null);
  assert.throws(() => tools.confirmFrame(unknown, 1, "Reviewer", "complete"), /known identities and visibility/);
  assert.throws(() => tools.confirmFrame(unidentified, 1, "Reviewer", "complete"), /known identities and visibility/);
  assert.equal(tools.confirmFrame(input, 0, "Reviewer", "complete").frames[0].objects.length, 0, "an explicitly reviewed empty frame can be a negative");
});

test("summary distinguishes full human coverage from sparse, assistant, partial and omitted evidence", () => {
  const dense = fixture();
  assert.equal(tools.summary(dense, sequence().manifest).dense_human_reference, true);
  assert.equal(tools.summary(fixture([0, 2]), sequence([0, 2]).manifest).dense_human_reference, false);
  const mixed = tools.clone(dense);
  mixed.frames[0].review = { status: "assistant_reviewed", reviewer: "Assistance" };
  mixed.frames[1].coverage = "partial";
  mixed.frames[1].objects[0].certainty = "uncertain";
  mixed.frames.splice(2, 1);
  assert.deepEqual(tools.summary(mixed, sequence().manifest), {
    available_frames: 3, annotated_frames: 2, omitted_frames: 1, human_reviewed_frames: 1,
    assistant_reviewed_frames: 1, complete_frames: 1, human_complete_frames: 0, partial_frames: 1,
    unreviewed_frames: 1, identity_count: 1, object_count: 2, uncertain_objects: 1, dense_human_reference: false,
  });
});

test("identity catalog changes cannot silently delete observations or retain review after class correction", () => {
  const input = freeze(fixture());
  assert.throws(() => tools.removeIdentity(input, "ref_a"), /Reassign or remove/);
  const withUnused = tools.newIdentity(input, "car", "ref_car");
  assert.deepEqual(tools.changedFrames(input, withUnused), []);
  assert.deepEqual(tools.removeIdentity(withUnused, "ref_car"), input);
  const corrected = tools.updateIdentity(input, "ref_a", "car");
  assert.ok(corrected.frames.every((frame) => frame.objects[0].label === "car" && frame.review.status === "unreviewed"));
  assert.deepEqual(tools.changedFrames(input, corrected), [0, 1, 2]);
  assert.deepEqual(reviewed(tools.updateIdentity(input, "ref_a", "person")), [0, 1, 2]);
});

test("missing available frames can be materialized without inferring human negatives or filling intermediate gaps", () => {
  const input = tools.blank(sequence([1, 10, 20]));
  input.frames = [];
  freeze(input);
  const output = tools.ensureFrame(input, 10);
  assert.deepEqual(output.frames.map((frame) => frame.frame_index), [10]);
  assert.equal(output.frames[0].review.status, "unreviewed");
  assert.deepEqual(tools.changedFrames(input, output), [10]);
  assert.deepEqual(tools.changedFrames(output, tools.confirmFrame(output, 10, "Reviewer", "complete")), [], "review-only changes do not invalidate a pending confirmation");
  assert.deepEqual(input.frames, []);
});

test("invalid coordinate and tracker-field edits fail without damaging source state", () => {
  const input = freeze(fixture());
  for (const box of [[0, 0, Infinity, 2], [0, 0, NaN, 2], [-1, 0, 2, 2], [2, 0, 1, 2], [1, 1, 1, 1]]) {
    assert.throws(() => tools.editObject(input, 1, 0, { box }), /finite xyxy/);
  }
  assert.throws(() => tools.editObject(input, 1, 0, { track_id: 8 }), /supported reference fields/);
  assert.throws(() => tools.addObject(input, 1, { ...object(null), estimated_box: [0, 0, 2, 2] }), /supported fields/);
  assert.deepEqual(reviewed(input), [0, 1, 2]);
});
