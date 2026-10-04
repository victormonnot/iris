const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/benchmark-tools.js");

test("benchmark roles include only checked images and require both bounded scene roles", () => {
  const groups = [
    { scene_group: "scene-a", frames: [{ id: "a1" }, { id: "a2" }] },
    { scene_group: "scene-b", frames: [{ id: "b" }] },
    { scene_group: "excluded", frames: [{ id: "x" }] },
  ];
  const roles = new Map([["scene-a", "tuning"], ["scene-b", "evaluation"]]);
  const chosen = new Set(["a2", "b", "x"]);
  const result = tools.referenceSelection(groups, roles, chosen);
  assert.deepEqual(result.frame_ids, ["a2", "b"]);
  assert.deepEqual({ ...result.roles }, { "scene-a": "tuning", "scene-b": "evaluation" });
  assert.equal(result.valid, true);
  assert.equal(tools.referenceSelection(groups, roles, new Set(["a2"])).valid, false);
  const large = [{ scene_group: "scene-a", frames: Array.from({ length: 26 }, (_, i) => ({ id: `a${i}` })) }, groups[1]];
  assert.equal(tools.referenceSelection(large, roles, new Set([...large[0].frames.map((frame) => frame.id), "b"])).valid, false);
});

test("arbitrary scene names remain literal role keys", () => {
  const groups = [{ scene_group: "__proto__", frames: [{ id: "a" }] }, { scene_group: "constructor", frames: [{ id: "b" }] }];
  const result = tools.referenceSelection(groups, new Map([["__proto__", "tuning"], ["constructor", "evaluation"]]), new Set(["a", "b"]));
  assert.equal(JSON.parse(JSON.stringify(result.roles))["__proto__"], "tuning");
  assert.equal(result.valid, true);
});

test("old successful outputs cannot become corrections for a newly selected benchmark or trial", () => {
  const old = { id: "trial-old", benchmark_id: "reference-old", frames: [{ output_id: "old-output", state: "ready" }] };
  assert.equal(tools.currentTrial(old, "reference-old", "trial-old"), true);
  assert.equal(tools.currentTrial(old, "reference-new", "trial-new"), false);
  assert.equal(tools.currentTrial(old, "reference-old", "trial-new"), false);
  assert.equal(tools.currentTrial(null, "reference-old", "trial-old"), false);
  assert.equal(tools.currentTrial(old, null, null), false);
});

test("timer ownership is explicit and missing duration is distinct from measured zero", () => {
  assert.equal(tools.ownsRunningTimer({ state: "running", owner_token: "owner" }, "owner"), true);
  assert.equal(tools.ownsRunningTimer({ state: "running", owner_token: "other-tab" }, "owner"), false);
  assert.equal(tools.ownsRunningTimer({ state: "paused", owner_token: "owner" }, "owner"), false);
  assert.equal(tools.ownsRunningTimer({ state: "running" }, undefined), false);
  for (const missing of [null, undefined, NaN, Infinity, -1]) assert.equal(tools.duration(missing), "Unmeasured");
  assert.equal(tools.duration(0), "0.0 s");
  assert.equal(tools.duration(61500), "1 min 1.5 s");
});

test("lost correction responses require a newer exact saved revision before acknowledging success", () => {
  const payload = { boxes: [{ id: "corrected", label: "helmet", box: [1, 2, 10, 20], proposal_id: "candidate" }], status: "reviewed", reviewer: "Human", notes: "Added missing edge" };
  const saved = { ...payload, revision: 3 };
  assert.equal(tools.correctionMatches(saved, payload, 2), true);
  assert.equal(tools.correctionMatches(saved, payload, 3), false);
  assert.equal(tools.correctionMatches({ ...saved, status: "draft" }, payload, 2), false);
  assert.equal(tools.correctionMatches({ ...saved, reviewer: "Other" }, payload, 2), false);
  assert.equal(tools.correctionMatches({ ...saved, boxes: [] }, payload, 2), false);
});

test("a masked editor cannot save a running timer and credit an uninspected interval", () => {
  const timer = { state: "running", owner_token: "this-editor", elapsed_ms: 1200 };
  assert.equal(tools.canSaveCorrection(timer, "this-editor", true), true);
  assert.equal(tools.canSaveCorrection(timer, "this-editor", false), false, "lost POST receipt or reload requires an explicit conservative pause");
  assert.equal(tools.canSaveCorrection(timer, "another-editor", true), false);
  assert.equal(tools.canSaveCorrection({ ...timer, state: "paused" }, "this-editor", false), true);
});

test("box movement and corner resizing preserve image bounds and positive extent", () => {
  assert.deepEqual(tools.boxAfterDrag([10, 10, 40, 40], "move", [15, 15], [-100, 1000], 100, 80), [0, 50, 30, 80]);
  assert.deepEqual(tools.boxAfterDrag([10, 10, 40, 40], "nw", [10, 10], [100, 100], 100, 80), [39, 39, 40, 40]);
  assert.deepEqual(tools.boxAfterDrag([10, 10, 40, 40], "se", [40, 40], [1000, 1000], 100, 80), [10, 10, 100, 80]);
});
