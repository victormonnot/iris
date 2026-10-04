"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { approval, findTrialReceipt, providerStatus, costPresentation, money } = require("../../src/iris/static/benchmark-external-tools.js");

const now = Date.parse("2026-10-04T12:00:00Z");
const preview = () => ({
  fingerprint: "approved-plan", preview_token: "signed-expiring-token", expires_at: "2026-10-04T12:10:00Z", launch_allowed: true,
  external_plan: { requests: [{ frame_id: "first" }, { frame_id: "second" }], estimate: { currency: "USD", upper_bound_usd: 0.035271 } },
});
const consent = (changes = {}) => ({ budget: 0.035271, consent: true, loaded: new Set(["first", "second"]), now, ...changes });

test("an external send needs an unexpired prepared receipt, a configured provider and a fresh explicit consent", () => {
  assert.equal(approval(preview(), consent()).allowed, true);
  for (const change of [{ preview_token: null }, { fingerprint: null }, { expires_at: "bad" }, { expires_at: new Date(now).toISOString() }, { launch_allowed: false }, { launch_allowed: undefined }])
    assert.equal(approval({ ...preview(), ...change }, consent()).allowed, false);
  assert.equal(approval(null, consent()).allowed, false);
  assert.equal(approval(preview(), consent({ consent: false })).allowed, false);
  assert.equal(approval(preview(), consent({ consent: "true" })).allowed, false);
  assert.match(approval({ ...preview(), launch_allowed: false, launch_reason: "No server key" }, consent()).reason, /No server key/);
});

test("the planning budget must cover the exact finite USD estimate without rounding it down", () => {
  assert.equal(approval(preview(), consent({ budget: 0.03527 })).allowed, false);
  assert.equal(approval(preview(), consent({ budget: 0.04 })).allowed, true);
  for (const budget of [NaN, Infinity, -1, "0.04", null, 1000.01]) assert.equal(approval(preview(), consent({ budget })).allowed, false);
  for (const estimate of [{ currency: "EUR", upper_bound_usd: 0.01 }, { currency: "USD", upper_bound_usd: null }, { currency: "USD", upper_bound_usd: NaN }, { currency: "USD", upper_bound_usd: -1 }]) {
    const value = preview(); value.external_plan.estimate = estimate;
    assert.equal(approval(value, consent()).allowed, false);
  }
});

test("every distinct outgoing image must load; an old or missing image cannot satisfy approval", () => {
  for (const loaded of [new Set(), new Set(["first"]), new Set(["first", "other-plan-image"])])
    assert.equal(approval(preview(), consent({ loaded })).allowed, false);
  const duplicate = preview(); duplicate.external_plan.requests = [{ frame_id: "first" }, { frame_id: "first" }];
  assert.equal(approval(duplicate, consent()).allowed, false);
  const empty = preview(); empty.external_plan.requests = [];
  assert.equal(approval(empty, consent()).allowed, false);
});

test("a lost external creation acknowledgement only reconciles the exact benchmark and fingerprint with a saved job", () => {
  const records = [
    { id: "other-benchmark", benchmark_id: "other", config: { fingerprint: "approved" }, job_id: "job" },
    { id: "other-preview", benchmark_id: "current", config: { fingerprint: "previous" }, job_id: "job" },
    { id: "no-job", benchmark_id: "current", config: { fingerprint: "approved" } },
    { id: "receipt", benchmark_id: "current", config: { fingerprint: "approved" }, job_id: "saved-job" },
  ];
  assert.equal(findTrialReceipt(records, "approved", "current").id, "receipt");
  assert.equal(findTrialReceipt(records, "absent", "current"), null);
  assert.equal(findTrialReceipt(records, null, "current"), null);
  assert.equal(findTrialReceipt(null, "approved", "current"), null);
});

test("unknown and missing usage remain unknown even when other image costs are recorded", () => {
  assert.equal(costPresentation(null), "Not measured");
  assert.match(costPresentation({ unknown_outcome_count: 1, known_usage_cost_usd: 0.02, usage_cost_usd: 0.02 }), /subtotal.*0.02 USD.*total unknown.*1 delivery outcome/);
  const missing = costPresentation({ usage_missing_count: 1, usage_cost_usd: 0 });
  assert.match(missing, /No recorded usage cost.*total unknown/);
  assert.doesNotMatch(missing, /0.00 USD/);
  assert.match(costPresentation({ usage_cost_usd: 0.037 }), /0.037 USD estimated from recorded usage/);
  assert.equal(costPresentation({}), "Usage cost not yet recorded");
  assert.equal(money(null), "Unknown");
});

test("offline provider readiness does not claim verified model access", () => {
  assert.match(providerStatus({ status: "ready" }), /model access not verified/);
  assert.match(providerStatus({ status: "missing_key" }), /No server API key/);
  assert.equal(providerStatus({ status: "invalid_config", reason: "Invalid settings" }), "Invalid settings");
});
