const test = require("node:test");
const assert = require("node:assert/strict");
const { validatePlan } = require("../../src/iris/static/partition-tools.js");
function fixture() {
  return {
    candidates: { taxonomy: { id: "classes-v2" }, groups: [
      { scene_group: "workshop", reserved_split: null, frames: [{ id: "a", annotation_revision_id: "a-review-2" }] },
      { scene_group: "field", reserved_split: "val", frames: [{ id: "b", annotation_revision_id: "b-review-1", reserved_split: "val" }] },
    ] },
    plan: { taxonomy_id: "classes-v2", can_freeze: true, blockers: [], frame_ids: ["b", "a"], expected_revisions: { a: "a-review-2", b: "b-review-1" }, splits: { workshop: "train", field: "val" } },
  };
}
test("an unchanged preview applies complete groups while retaining split reservations", () => {
  const { plan, candidates } = fixture();
  assert.deepEqual([...validatePlan(plan, candidates)], [["workshop", "train"], ["field", "val"]]);
  assert.equal(candidates.groups[0].split, undefined);
});
test("new, removed and rereviewed eligible frames invalidate a partition preview", () => {
  for (const change of [
    (c) => { c.groups[0].frames[0].annotation_revision_id = "a-review-3"; },
    (c) => { c.groups[0].frames.push({ id: "c", annotation_revision_id: "c-review-1" }); },
    (c) => { c.groups[0].frames = []; },
  ]) {
    const { plan, candidates } = fixture();change(candidates);
    assert.throws(() => validatePlan(plan, candidates), /Eligible frames or reviewed revisions changed/);
  }
});
test("changed definitions, reservations and blocking duplicate warnings prevent applying", () => {
  let { plan, candidates } = fixture();candidates.taxonomy.id = "classes-v3";
  assert.throws(() => validatePlan(plan, candidates), /Class definitions changed/);
  ({ plan, candidates } = fixture());candidates.groups[0].reserved_split = "test";
  assert.throws(() => validatePlan(plan, candidates), /Split reservations changed/);
  ({ plan, candidates } = fixture());plan.blockers.push({ code: "duplicates" });
  assert.throws(() => validatePlan(plan, candidates), /blockers/);
});
test("missing or extra revision tokens cannot conceal changes in eligible frames", () => {
  let { plan, candidates } = fixture();delete plan.expected_revisions.b;
  assert.throws(() => validatePlan(plan, candidates), /Eligible frames/);
  ({ plan, candidates } = fixture());plan.expected_revisions.foreign = "other-review";
  assert.throws(() => validatePlan(plan, candidates), /Eligible frames/);
});
