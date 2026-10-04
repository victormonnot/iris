"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const combined = require("../../src/iris/static/benchmark-combined-tools.js");
const external = require("../../src/iris/static/benchmark-external-tools.js");

const plan = () => ({ requests: [{
  frame_id: "first", image_url: "/api/benchmark-configs/config/frames/first/input-image",
  planning: { input: { image: { sent_width: 512, sent_height: 320 }, prompt: "Frozen class definitions", request_sha256: "exact-request" } },
  review: { template: { stage: "review", template: true, prompt: "Accept, reject or relabel existing IDs", max_dynamic_text_bytes: 131072, template_sha256: "frozen-template", dynamic_fields: ["planning_prompts", "candidates"] } },
}], estimate: { currency: "USD", upper_bound_usd: 0.1 } });

test("combined approval cannot treat a single-stage preview as a complete two-stage plan", () => {
  assert.equal(combined.planError(plan()), null);
  for (const value of [null, {}, { requests: [] }]) assert.match(combined.planError(value), /complete combined/);
  for (const change of [
    (request) => { delete request.planning; },
    (request) => { delete request.planning.input.image; },
    (request) => { request.planning.input.prompt = " "; },
    (request) => { delete request.planning.input.request_sha256; },
    (request) => { delete request.review; },
    (request) => { request.review.template = "Not a structured template"; },
    (request) => { request.review.template.max_dynamic_text_bytes = 262144; },
    (request) => { request.review.template.template = false; },
    (request) => { request.review.template.dynamic_fields = ["planning_prompts"]; },
    (request) => { delete request.review.template.template_sha256; },
  ]) {
    const value = plan(); change(value.requests[0]);
    assert.match(combined.planError(value), /incomplete/);
  }
});

test("one displayed image can cover two disclosed calls while the total bound and fresh consent remain required", () => {
  const now = Date.parse("2026-10-04T12:00:00Z");
  const preview = { external_plan: plan(), fingerprint: "plan", preview_token: "signed", expires_at: "2026-10-04T12:10:00Z", launch_allowed: true };
  const approval = { consent: true, budget: 0.1, loaded: new Set(["first"]), now };
  assert.equal(external.approval(preview, approval).allowed, true);
  assert.equal(external.approval(preview, { ...approval, budget: 0.05 }).allowed, false);
  assert.equal(external.approval(preview, { ...approval, consent: false }).allowed, false);
  assert.equal(external.approval({ ...preview, launch_allowed: false, launch_reason: "SAM setup required" }, approval).allowed, false);
  assert.match(combined.workSummary({ image_count: 3, request_count: 6, image_encodings: 3, prompt_evaluations: 9 }), /3 images.*6 external calls.*3 local image encodings.*9 class-prompt evaluations.*no iteration/);
});

test("stage evidence preserves independent raw and failed or unknown results without inventing completion", () => {
  const output = { metadata: { pipeline: { stages: { planning: { state: "success", result: { prompts: [{ class_id: "constructor", text: "Hard hat" }] } }, grounding: { state: "failed", error: "Invalid native output" }, review: { state: "outcome_unknown" } } } }, raw_response: { planning: { response: "raw" }, grounding: { invalid: true } } };
  const stages = combined.stages(output);
  assert.deepEqual(stages.map((stage) => stage.id), ["planning", "grounding", "review"]);
  assert.equal(stages[0].raw, output.raw_response.planning);
  assert.equal(stages[1].stage.error, "Invalid native output");
  assert.equal(stages[2].unknown, true);
  assert.equal(stages[2].raw, null);
  assert.ok(combined.stages(null).every((stage) => stage.state === "not_started" && stage.raw === null));
});

test("both external receipts for one image are retained and unrelated images cannot leak in", () => {
  const summary = { outputs: [
    { frame_id: "first", stage: "planning", state: "response_received", usage_cost_usd: 0.03 },
    { frame_id: "second", stage: "planning", state: "not_started" },
    { frame_id: "first", stage: "review", state: "outcome_unknown" },
  ], known_usage_cost_usd: 0.03, unknown_outcome_count: 1, usage_missing_count: 1 };
  assert.deepEqual(combined.dispatches(summary, "first").map((dispatch) => dispatch.stage), ["planning", "review"]);
  assert.deepEqual(combined.dispatches(summary, "absent"), []);
  assert.deepEqual(combined.dispatches(null, "first"), []);
  assert.match(external.costPresentation(summary), /subtotal.*0.03 USD.*total unknown/);
  assert.equal(combined.dispatches({ outputs: [{ frame_id: "legacy-a" }] }, "legacy-a").length, 1);
});

test("offline combined readiness never claims tested provider access or a successful model run", () => {
  assert.match(combined.availability({ status: "ready" }), /not evidence of verified model access or a successful run/);
  assert.match(combined.availability({ status: "missing_runtime", reason: "CUDA unavailable" }), /Setup required.*CUDA unavailable/);
  assert.match(combined.availability(null), /unavailable/);
});
