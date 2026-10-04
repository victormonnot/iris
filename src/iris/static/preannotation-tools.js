"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISPreannotationTools = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const lowScore = (proposal) => typeof proposal?.metadata?.score === "number" &&
    Number.isFinite(proposal.metadata.score) && proposal.metadata.score >= 0 && proposal.metadata.score < 0.5;
  const uncertain = (proposal) => proposal?.metadata?.recommendation === "uncertain";
  function reviewMatches(frame, filter) {
    if (filter === "all") return true;
    if (filter === "needs_review") return frame.review_status !== "validated";
    if (filter === "uncertain") return frame.hints?.uncertain_count > 0 || frame.hints?.low_confidence_count > 0;
    if (filter === "possible_omissions") return frame.hints?.possible_omission === true;
    return frame.review_status === filter;
  }
  function proposalMatches(proposal, filter, decision) {
    if (filter === "pending") return decision === "pending";
    if (filter === "low_confidence") return lowScore(proposal);
    if (filter === "uncertain") return uncertain(proposal);
    return true;
  }
  function findReceipt(records, fingerprint) {
    if (!fingerprint) return null;
    return (records || []).find((record) =>
      record.config?.preannotation?.fingerprint === fingerprint || record.fingerprint === fingerprint,
    ) || null;
  }
  const frameStates = {
    queued: "Queued", running: "Running", pending_review: "Proposals ready for review",
    no_proposals: "No proposals · inspect the whole image", conflict: "Saved frame changed · review required",
    invalid_output: "Invalid output · inspect the raw record", raw_saved: "Raw output saved · proposals not published",
    cancelled: "Cancelled", failed: "Failed", interrupted: "Interrupted",
  };
  return { lowScore, uncertain, reviewMatches, proposalMatches, findReceipt, frameStates };
});
