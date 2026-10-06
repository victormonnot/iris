const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/dinox-tools.js");

test("cloud approval requires the same preview, explicit consent and sufficient finite CNY budget", () => {
  const preview = { key: "saved-settings", fingerprint: "approved-inputs", eligible_count: 2, estimate: { currency: "CNY", total: 0.3 } };
  const approval = { key: "saved-settings", consent: true, budget: "0.30" };
  assert.equal(tools.approved(preview, approval), true);
  for (const budget of ["", " ", "0.29", "-1", "NaN", "Infinity", "invalid"])
    assert.equal(tools.approved(preview, { ...approval, budget }), false);
  assert.equal(tools.approved(preview, { ...approval, consent: false }), false);
  assert.equal(tools.approved(preview, { ...approval, consent: "true" }), false);
  assert.equal(tools.approved(preview, { ...approval, key: "changed-settings" }), false);
  for (const changed of [null, { ...preview, fingerprint: "" }, { ...preview, eligible_count: 0 }, { ...preview, estimate: { currency: "USD", total: 0.3 } }, { ...preview, estimate: { currency: "CNY", total: NaN } }])
    assert.equal(tools.approved(changed, approval), false);
});

test("free reuse and polling still require explicit approval and a typed budget", () => {
  const preview = { key: "reuse", fingerprint: "snapshot", eligible_count: 1, estimate: { currency: "CNY", total: 0 } };
  assert.equal(tools.approved(preview, { key: "reuse", consent: true, budget: "0" }), true);
  assert.equal(tools.approved(preview, { key: "reuse", consent: true, budget: "" }), false);
  assert.equal(tools.approved(preview, { key: "reuse", consent: false, budget: "0" }), false);
});

test("cached proposals work without a credential, while submissions and polls require one", () => {
  for (const provider of [null, { status: "missing_key" }, { status: "invalid_key" }]) {
    assert.equal(tools.canProcess({ request_count: 0, poll_count: 0, reuse_count: 1 }, provider), true);
    assert.equal(tools.canProcess({ request_count: 1, poll_count: 0 }, provider), false);
    assert.equal(tools.canProcess({ request_count: 0, poll_count: 1 }, provider), false);
    assert.equal(tools.canProcess({}, provider), false);
    assert.equal(tools.canProcess(null, provider), false);
  }
  assert.equal(tools.canProcess({ request_count: 1, poll_count: 1 }, { status: "ready" }), true);
});

test("custom prompts preserve exact class IDs and reject arrays or non-text values", () => {
  assert.equal(tools.parsePrompts(" "), null);
  assert.deepEqual(tools.parsePrompts('{"person":" human ","car":"car"}'), { car: "car", person: "human" });
  for (const invalid of ["bad JSON", "null", "[]", "{}", '{"person":2}', '{"person":"  "}', '{"":"car"}'])
    assert.throws(() => tools.parsePrompts(invalid));
  assert.deepEqual(Object.keys(tools.parsePrompts('{"__proto__":"object"}')), ["__proto__"]);
});

test("lost create responses reconcile only the exact saved fingerprint", () => {
  const batches = [{ id: "old", config: { fingerprint: "old" } }, { id: "recorded", config: { fingerprint: "approved" } }];
  assert.equal(tools.findReceipt(batches, "approved").id, "recorded");
  assert.equal(tools.findReceipt(batches, "missing"), null);
  assert.equal(tools.findReceipt(batches, null), null);
  assert.equal(tools.findReceipt(null, "approved"), null);
});

test("cloud outcome labels keep empty outputs and unknown delivery distinct", () => {
  assert.match(tools.frameStates.no_proposals, /inspect the whole image/);
  assert.match(tools.frameStates.outcome_unknown, /resubmission blocked/);
  assert.match(tools.actions.poll, /no resubmission/);
  assert.match(tools.actions.reuse, /no new request/);
  assert.equal(tools.money(0.15), "0.15 CNY");
  assert.equal(tools.money(null), "Unavailable");
});
