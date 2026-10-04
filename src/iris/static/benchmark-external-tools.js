"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISBenchmarkExternalTools = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const money = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0
    ? `${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 6 })} USD` : "Unknown";
  function approval(preview, { budget, consent, loaded = new Set(), now = Date.now() } = {}) {
    const plan = preview?.external_plan;
    if (!plan || !plan.requests?.length || !preview.preview_token || !preview.fingerprint)
      return { allowed: false, reason: "Prepare a complete external request preview first." };
    const expires = Date.parse(preview.expires_at);
    if (!Number.isFinite(expires) || expires <= now)
      return { allowed: false, reason: "This approval preview expired. Prepare a fresh preview." };
    if (preview.launch_allowed !== true)
      return { allowed: false, reason: preview.launch_reason || "The provider is not configured for this trial." };
    const upper = plan.estimate?.upper_bound_usd;
    if (plan.estimate?.currency !== "USD" || typeof upper !== "number" || !Number.isFinite(upper) || upper < 0)
      return { allowed: false, reason: "No valid USD planning estimate is available. Prepare a new preview." };
    if (new Set(plan.requests.map((request) => request.frame_id)).size !== plan.requests.length ||
        !plan.requests.every((request) => loaded.has(request.frame_id)))
      return { allowed: false, reason: "Wait until every outgoing image is displayed successfully." };
    if (typeof budget !== "number" || !Number.isFinite(budget) || budget < upper)
      return { allowed: false, reason: `Approve a budget of at least ${money(upper)} for this plan.` };
    if (budget > 1000)
      return { allowed: false, reason: "The trial planning budget must not exceed 1,000 USD." };
    if (consent !== true)
      return { allowed: false, reason: "Review the images, prompt and budget, then explicitly approve this trial." };
    return { allowed: true, reason: "This approval applies only to the current preview and budget." };
  }
  function findTrialReceipt(trials, fingerprint, benchmarkId) {
    if (!fingerprint || !benchmarkId) return null;
    return (Array.isArray(trials) ? trials : []).find((trial) => trial.benchmark_id === benchmarkId &&
      trial.config?.fingerprint === fingerprint && Boolean(trial.job_id || trial.job?.id)) || null;
  }
  function providerStatus(provider) {
    if (!provider) return "Server configuration unavailable";
    if (provider.status === "ready") return "Server key configured · model access not verified";
    return provider.reason || (provider.status === "missing_key" ? "No server API key configured" : "Server provider configuration is invalid");
  }
  function costPresentation(summary) {
    if (!summary) return "Not measured";
    const uncertain = summary.unknown_outcome_count > 0 || summary.usage_missing_count > 0;
    const known = typeof summary.known_usage_cost_usd === "number" && Number.isFinite(summary.known_usage_cost_usd) && summary.known_usage_cost_usd >= 0;
    if (uncertain) return `${known ? `Known usage subtotal: ${money(summary.known_usage_cost_usd)}` : "No recorded usage cost"} · total unknown${summary.unknown_outcome_count ? ` · ${summary.unknown_outcome_count} delivery outcomes unknown` : ""}`;
    if (typeof summary.usage_cost_usd === "number" && Number.isFinite(summary.usage_cost_usd) && summary.usage_cost_usd >= 0) return `${money(summary.usage_cost_usd)} estimated from recorded usage`;
    return known ? `${money(summary.known_usage_cost_usd)} known usage subtotal · total unknown` : "Usage cost not yet recorded";
  }
  return { money, approval, findTrialReceipt, providerStatus, costPresentation };
});
