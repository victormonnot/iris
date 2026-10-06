"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISDINOXTools = tools;
})(typeof window === "undefined" ? globalThis : window, () => {
  function parsePrompts(value) {
    if (!value.trim()) return null;
    let prompts;
    try { prompts = JSON.parse(value); } catch { throw new Error('Use a JSON object, for example {"person":"person"}.'); }
    if (!prompts || Array.isArray(prompts) || typeof prompts !== "object" || !Object.keys(prompts).length ||
        Object.entries(prompts).some(([key, prompt]) => !key.trim() || typeof prompt !== "string" || !prompt.trim()))
      throw new Error("Each class ID needs a nonempty text prompt.");
    return Object.fromEntries(Object.entries(prompts).sort(([a], [b]) => a.localeCompare(b)).map(([id, prompt]) => [id, prompt.trim()]));
  }
  function approved(preview, { key, consent, budget }) {
    const cost = preview?.estimate?.total;
    return Boolean(preview && preview.key === key && preview.fingerprint && preview.eligible_count > 0 &&
      preview.estimate?.currency === "CNY" && Number.isFinite(cost) && cost >= 0 && consent === true &&
      String(budget).trim() && Number.isFinite(Number(budget)) && Number(budget) >= cost);
  }
  function findReceipt(records, fingerprint) {
    if (!fingerprint || !Array.isArray(records)) return null;
    return records.find((record) => [record.fingerprint, record.config?.fingerprint,
      record.config?.dinox?.fingerprint, record.config?.consent?.preview_id].includes(fingerprint)) || null;
  }
  const canProcess = (preview, provider) => provider?.status === "ready" ||
    Boolean(preview && preview.request_count === 0 && preview.poll_count === 0);
  const active = (job) => ["queued", "running"].includes(job?.status);
  const money = (amount) => typeof amount === "number" && Number.isFinite(amount) ? `${amount.toFixed(2)} CNY` : "Unavailable";
  const actions = { submit: "New cloud request", reuse: "Reuse saved result · no new request", poll: "Fetch submitted result · no resubmission", blocked: "Blocked · no request" };
  const frameStates = {
    not_started: "Not sent · preview to continue", queued: "Queued", running: "Processing", submitting: "Sending to DINO-X", submitted: "Submitted to DINO-X",
    polling: "Fetching submitted result", ready: "Ready for human review", pending_review: "Proposals ready for review",
    no_proposals: "No proposals · inspect the whole image", succeeded: "Result saved · human review required",
    reused: "Saved result reused", failed: "Failed", cancelled: "Cancelled", interrupted: "Interrupted",
    conflict: "Saved frame changed · review required", invalid_output: "Invalid output · inspect the saved record",
    raw_saved: "Raw output saved · proposals not published", outcome_unknown: "Delivery unknown · resubmission blocked",
    blocked: "Blocked · inspect the saved record", excluded: "Excluded before launch",
  };
  return { parsePrompts, approved, canProcess, findReceipt, active, money, actions, frameStates };
});
