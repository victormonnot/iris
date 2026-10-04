const test = require("node:test");
const assert = require("node:assert/strict");
const { filterFrames, selectionPayload, createUploadQueue } = require("../../src/iris/static/intake-tools.js");
const assets = [{ id: "video", filename: "Workshop take.mp4" }, { id: "image", filename: "Panel.jpg" }];
const frames = [
  { id: "late", asset_id: "video", timestamp_seconds: 8, frame_index: 80, selected: true },
  { id: "early", asset_id: "video", timestamp_seconds: 1, frame_index: 10, selected: false },
  { id: "still", asset_id: "image", selected: true },
  { id: "unknown", asset_id: "image", selected: true },
];
const insights = new Map([
  ["late", { review_status: "validated", negative: true, positive: false, no_target_predictions: true, similar_frame_ids: ["early"] }],
  ["early", { review_status: "pending_suggestions", negative: null, positive: null, low_confidence_count: 2, no_target_predictions: null }],
  ["still", { review_status: "validated", positive: true, negative: false, exact_duplicate_ids: ["unknown"] }],
]);
const ids = (filters) => filterFrames(frames, assets, insights, filters).map((frame) => frame.id);

test("source filters use chronological video order across extraction batches", () => {
  assert.deepEqual(ids({ source: "video" }), ["early", "late"]);
  assert.deepEqual(ids({ source: "video", selected: true }), ["late"]);
  assert.deepEqual(ids({ source: "video", search: "WORKSHOP", review: "pending" }), ["early"]);
  assert.deepEqual(ids({ search: "Panel", review: "validated_positive" }), ["still"]);
});

test("unknown and partially mapped predictions never qualify as negatives or zero detections", () => {
  assert.deepEqual(ids({ review: "validated_negative" }), ["late"]);
  assert.deepEqual(ids({ signal: "no_detections" }), ["late"]);
  assert.deepEqual(ids({ signal: "low_confidence" }), ["early"]);
  assert.deepEqual(ids({ review: "unreviewed" }), []);
  assert.deepEqual(ids({ signal: "exact_duplicates" }), ["still"]);
  assert.deepEqual(ids({ signal: "similar" }), ["late"]);
  assert.deepEqual(ids({ review: "validated_positive", signal: "low_confidence" }), []);
});

test("bulk requests capture exact pre-update values only for the filtered targets", () => {
  const visible = filterFrames(frames, assets, insights, { source: "video" });
  assert.deepEqual(selectionPayload(visible, true), { frame_ids: ["early"], selected: true, expected_selection: { early: false } });
  assert.deepEqual(selectionPayload(visible, false), { frame_ids: ["late"], selected: false, expected_selection: { late: true } });
  assert.equal(frames[0].selected, true);
});

const file = (name) => ({ name, size: 12 });
test("failed imports do not stop the batch and retry excludes successful or existing files", async () => {
  const calls = [];
  let failing = true;
  const queue = createUploadQueue({ upload: async (item, session) => {
    calls.push([item.name, session]);
    if (item.name === "broken" && failing) throw new Error("Invalid image");
    return { id: item.name, import_status: item.name === "known" ? "existing" : "created" };
  } });
  await queue.start([file("new"), file("broken"), file("known")], "captured-session");
  assert.deepEqual(queue.state.entries.map((entry) => entry.status), ["succeeded", "failed", "existing"]);
  assert.equal(queue.state.entries[0].file, null);
  assert.equal(queue.state.entries[1].error, "Invalid image");
  failing = false;
  await queue.retryFailed();
  assert.deepEqual(calls, [["new", "captured-session"], ["broken", "captured-session"], ["known", "captured-session"], ["broken", "captured-session"]]);
  assert.equal(queue.state.busy, false);
});

test("cancelling the queue leaves the active upload intact and never invents progress", async () => {
  let finish;
  let emit;
  const calls = [];
  const queue = createUploadQueue({ upload: (item, session, progress) => {
    calls.push([item.name, session]);
    emit = progress;
    return new Promise((resolve) => { finish = resolve; });
  } });
  const running = queue.start([file("active"), file("pending")], "original");
  assert.equal(queue.state.entries[0].loaded, null);
  assert.equal(queue.state.entries[0].status, "uploading");
  emit({ loaded: 5, total: 12 });
  assert.equal(queue.state.entries[0].loaded, 5);
  emit({ processing: true });
  assert.equal(queue.state.entries[0].status, "processing");
  queue.cancelRemaining();
  assert.equal(queue.state.busy, true);
  assert.equal(queue.state.entries[1].status, "cancelled");
  await queue.start([file("foreign")], "other-session");
  finish({ id: "saved", import_status: "created" });
  await running;
  assert.deepEqual(calls, [["active", "original"]]);
  assert.deepEqual(queue.state.entries.map((entry) => entry.status), ["succeeded", "cancelled"]);
  await queue.retryFailed();
  assert.deepEqual(calls, [["active", "original"]]);
});
