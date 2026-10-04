const test = require("node:test");
const assert = require("node:assert/strict");
const { history, dispatchPresentation, canContinue, findApprovedRequest, retryableBatchFrames } = require("../../src/iris/static/job-tools.js");

test("project history retains every active task while paging and filtering saved jobs", () => {
  const jobs = Array.from({ length: 12 }, (_, i) => ({ id: String(i), kind: "extract", status: "succeeded", created_at: `2026-10-${String(i + 1).padStart(2, "0")}`, message: "Saved source" }));
  jobs.push({ id: "active", kind: "assist", status: "running", created_at: "2026-09-01" });
  const result = history(jobs);
  assert.equal(result.rows.length, 9);
  assert.equal(result.rows[0].id, "active");
  assert.equal(result.more, true);
  assert.equal(history(jobs, { limit: 20 }).rows.length, 13);
  assert.deepEqual(history(jobs, { status: "active" }).rows.map((job) => job.id), ["active"]);
  assert.equal(history(jobs, { kind: "extract", query: "SOURCE" }).total, 12);
  assert.equal(history(jobs, { status: "attention" }).total, 0);
  assert.equal(jobs[0].id, "0", "view sorting must not mutate shared job records");
});

test("external delivery uncertainty and a recorded response never imply success or zero charge", () => {
  const unknown = dispatchPresentation({ state: "outcome_unknown", external: true });
  assert.equal(unknown.label, "Delivery outcome unknown");
  assert.match(unknown.explanation, /will not resend it automatically/);
  assert.match(unknown.explanation, /another charge/);
  const received = dispatchPresentation({ state: "response_received", external: true });
  assert.match(received.explanation, /does not by itself confirm/);
  assert.match(received.explanation, /final charge/);
  assert.doesNotMatch(dispatchPresentation({ state: "outcome_unknown", external: false }).explanation, /charge/);
  assert.equal(dispatchPresentation(null), null);
});

test("continuation needs a checked extraction fingerprint for this exact terminal job", () => {
  const detail = { job: { id: "parent", kind: "extract", status: "interrupted" }, recovery: { can_check: true }, dispatch: null };
  const preview = { source_job_id: "parent", mode: "continue_extraction", available: true, fingerprint: "frozen-plan", remaining_count: 3 };
  assert.equal(canContinue(detail, preview), true);
  for (const mutation of [{ source_job_id: "other" }, { fingerprint: null }, { remaining_count: 0 }, { available: false }]) assert.equal(canContinue(detail, { ...preview, ...mutation }), false);
  assert.equal(canContinue({ ...detail, job: { ...detail.job, status: "running" } }, preview), false);
  assert.equal(canContinue({ ...detail, job: { ...detail.job, kind: "train" } }, preview), false);
  assert.equal(canContinue({ ...detail, dispatch: { external: true } }, preview), false);
});

test("lost external acknowledgements reconcile by exact consumed preview identity only", () => {
  const records = [
    { id: "wrong", job_id: "unrelated", config: { consent: { preview_id: "other" } } },
    { id: "unsent", config: { consent: { preview_id: "approved" } } },
    { id: "recorded", job_id: "existing-job", config: { consent: { preview_id: "approved" } } },
  ];
  assert.equal(findApprovedRequest(records, "approved").job_id, "existing-job");
  assert.equal(findApprovedRequest(records, "missing"), null);
  assert.equal(findApprovedRequest(records, null), null);
  assert.equal(findApprovedRequest(null, "approved"), null);
});

test("local batch retry excludes successes and every frame with retained proposals", () => {
  const detail = { counts: { queued: 0, running: 0 }, frames: [
    { frame_id: "good", status: "succeeded", suggestions_created: 0 },
    { frame_id: "partial", status: "failed", suggestions_created: 2 },
    { frame_id: "failed", status: "failed", suggestions_created: 0 },
    { frame_id: "stopped", status: "cancelled", suggestions_created: 0 },
    { frame_id: "interrupted", status: "interrupted", suggestions_created: 0 },
  ] };
  assert.deepEqual(retryableBatchFrames(detail).map((frame) => frame.frame_id), ["failed", "stopped", "interrupted"]);
  assert.deepEqual(retryableBatchFrames({ ...detail, counts: { running: 1 } }), []);
});
