"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISBenchmarkSAMTools = tools;
})(typeof window === "undefined" ? globalThis : window, () => {
  function promptState(previous, benchmarkId, taxonomy) {
    const classes = taxonomy?.classes || [];
    const key = JSON.stringify([benchmarkId, taxonomy]);
    if (previous?.key === key) return previous;
    return { key, values: new Map(classes.map((item) => [item.id, item.name || item.id])) };
  }
  function promptPayload(state, taxonomy) {
    return Object.fromEntries((taxonomy?.classes || []).map((item) => [item.id, (state?.values.get(item.id) || "").trim()]));
  }
  function promptError(state, taxonomy, limits = {}) {
    const classes = taxonomy?.classes || [], max = limits.max_length || 120;
    if (!classes.length) return "Choose an independent reference with saved class definitions.";
    if (classes.length > (limits.max_classes || 100)) return "This reference exceeds the supported number of class prompts.";
    for (const item of classes) {
      const text = (state?.values.get(item.id) || "").trim();
      if (text.length < (limits.min_length || 1) || text.length > max)
        return `Enter a prompt of ${limits.min_length || 1}–${max} characters for ${item.name || item.id}.`;
    }
    return "";
  }
  function launchAllowed(preview, approach) {
    if (!preview || preview.launch_allowed === false) return false;
    return approach !== "segmentation" || preview.launch_allowed === true;
  }
  function availability(provider) {
    if (!provider) return "Setup information unavailable. Refresh to check the local configuration.";
    return provider.status === "ready"
      ? "Available for a checked local trial. This status is not evidence of a successful model run."
      : `Setup required. ${provider.reason || "Check the isolated runtime, model weights and CUDA device."}`;
  }
  function workSummary(work) {
    if (!work) return "Local work plan unavailable";
    return `${work.image_encodings} image encodings · ${work.prompt_evaluations} class-prompt evaluations · ${work.warmup_passes} warm-up passes`;
  }
  return { promptState, promptPayload, promptError, launchAllowed, availability, workSummary };
});
