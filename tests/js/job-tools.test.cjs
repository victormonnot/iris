const test = require("node:test");
const assert = require("node:assert/strict");
const { history, dispatchPresentation, canContinue, continuationPresentation, findApprovedRequest, retryableBatchFrames } = require("../../src/iris/static/job-tools.js");

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

test("multi-image dispatch history keeps received, unknown and unstarted attempts distinct", () => {
  const result = dispatchPresentation({ state: "outcome_unknown", external: true,
    counts: { response_received: 2, dispatching: 0, outcome_unknown: 1, not_started: 3 } });
  assert.match(result.explanation, /2 responses recorded.*1 outcomes unknown.*3 not sent/);
  assert.match(result.explanation, /another charge/);
  assert.equal(history([{ id: "trial", kind: "benchmark", status: "succeeded" }], { query: "preannotation benchmark" }).total, 1);
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

test("temporal continuation binds the checked cache preview to its exact eligible job", () => {
  const detail = { job: { id: "cache-attempt", kind: "temporal_detect", status: "interrupted" }, recovery: { can_check: true }, dispatch: null };
  const preview = { source_job_id: "cache-attempt", mode: "continue_temporal_detection", available: true, fingerprint: "frozen-cache", remaining_count: 2 };
  assert.equal(canContinue(detail, preview), true);
  for (const status of ["failed", "cancelled"]) assert.equal(canContinue({ ...detail, job: { ...detail.job, status } }, preview), true);
  for (const status of ["queued", "running", "succeeded", "unknown"]) assert.equal(canContinue({ ...detail, job: { ...detail.job, status } }, preview), false);
  for (const mutation of [{ mode: "continue_extraction" }, { mode: "future" }, { source_job_id: "other" }, { available: false }, { fingerprint: " " }, { remaining_count: 0 }, { remaining_count: "2" }, { remaining_count: 0.5 }]) assert.equal(canContinue(detail, { ...preview, ...mutation }), false);
  assert.equal(canContinue({ ...detail, recovery: { can_check: false } }, preview), false);
  assert.equal(canContinue({ ...detail, dispatch: { external: true } }, preview), false);
  assert.equal(canContinue(null, preview), false);
  assert.equal(canContinue(detail, null), false);
  const extraction = { ...detail, job: { ...detail.job, kind: "extract" } };
  assert.equal(canContinue(extraction, preview), false, "temporal previews cannot enable extraction continuation");
});

test("recovery wording distinguishes cached detection frames from sampled extraction positions", () => {
  const preview = { completed_count: 1, remaining_count: 2, total_count: 3, reason: "Saved evidence checked." };
  const temporal = continuationPresentation({ job: { kind: "temporal_detect" } }, preview);
  assert.equal(temporal.label, "Continue remaining detections");
  assert.match(temporal.summary, /1 frames with saved results.*2 remaining of 3/);
  assert.match(temporal.summary, /no detections are also saved results/);
  assert.match(temporal.notice, /only the remaining frames.*saved detector settings/);
  assert.match(temporal.queuedMessage, /remaining detection frames/);
  assert.doesNotMatch(JSON.stringify(temporal), /extraction|skipped duplicates|sampled positions/);
  const extraction = continuationPresentation({ job: { kind: "extract" } }, preview);
  assert.equal(extraction.label, "Continue remaining extraction");
  assert.match(extraction.summary, /sampled positions.*skipped duplicates/);
  assert.match(extraction.queuedMessage, /remaining extraction positions/);
  assert.equal(continuationPresentation(null, null).label, "Continue remaining work");
  assert.equal(history([{ id: "cache", kind: "temporal_detect", status: "interrupted" }], { query: "temporal detector cache" }).total, 1);
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
