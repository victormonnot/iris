const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/preannotation-tools.js");

test("review hints require recorded evidence and never treat missing signals as omissions", () => {
  for (const frame of [{}, { hints: {} }, { hints: { possible_omission: null } }]) {
    assert.equal(tools.reviewMatches(frame, "uncertain"), false);
    assert.equal(tools.reviewMatches(frame, "possible_omissions"), false);
  }
  assert.equal(tools.reviewMatches({ hints: { uncertain_count: 1 } }, "uncertain"), true);
  assert.equal(tools.reviewMatches({ hints: { low_confidence_count: 2 } }, "uncertain"), true);
  assert.equal(tools.reviewMatches({ hints: { possible_omission: true } }, "possible_omissions"), true);
  assert.equal(tools.reviewMatches({ review_status: "validated", hints: { possible_omission: true } }, "needs_review"), false);
});

test("proposal score filters distinguish numeric detector scores from missing or model uncertainty", () => {
  for (const score of [null, undefined, "0.1", NaN, Infinity, -1, 0.5])
    assert.equal(tools.lowScore({ metadata: { score } }), false);
  assert.equal(tools.lowScore({ metadata: { score: 0 } }), true);
  assert.equal(tools.lowScore({ metadata: { score: 0.499 } }), true);
  const uncertain = { metadata: { recommendation: "uncertain" } };
  assert.equal(tools.proposalMatches(uncertain, "uncertain", "pending"), true);
  assert.equal(tools.proposalMatches(uncertain, "low_confidence", "pending"), false);
  assert.equal(tools.proposalMatches(uncertain, "pending", "rejected"), false);
  assert.equal(tools.proposalMatches(uncertain, "all", "rejected"), true);
});

test("lost create acknowledgements reconcile only the matching saved preview fingerprint", () => {
  const records = [
    { id: "old", config: { preannotation: { fingerprint: "different" } } },
    { id: "recorded", config: { preannotation: { fingerprint: "approved" } } },
  ];
  assert.equal(tools.findReceipt(records, "approved").id, "recorded");
  assert.equal(tools.findReceipt(records, "missing"), null);
  assert.equal(tools.findReceipt(records, null), null);
  assert.equal(tools.findReceipt(null, "approved"), null);
});

test("partial and empty outcomes describe saved evidence without implying human validation", () => {
  assert.match(tools.frameStates.no_proposals, /inspect the whole image/);
  assert.match(tools.frameStates.raw_saved, /not published/);
  assert.match(tools.frameStates.pending_review, /review/);
  assert.match(tools.frameStates.conflict, /changed/);
});
