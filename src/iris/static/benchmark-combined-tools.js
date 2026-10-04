"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISBenchmarkCombinedTools = tools;
})(typeof window === "undefined" ? globalThis : window, () => {
  const stageNames = { planning: "1 · Astra plans class prompts", grounding: "2 · SAM generates native boxes locally", review: "3 · Astra reviews candidate IDs" };
  function availability(provider) {
    if (!provider) return "Combined setup information unavailable. Refresh to check the server configuration.";
    return provider.status === "ready" ? "Server key and local SAM setup available. This is not evidence of verified model access or a successful run."
      : `Setup required. ${provider.reason || "This approach requires both the server API key and the local SAM runtime, weights and CUDA."}`;
  }
  function stages(output) {
    return Object.entries(stageNames).map(([id, label]) => {
      const stage = output?.metadata?.pipeline?.stages?.[id];
      return { id, label, state: stage?.state || "not_started", stage: stage || null,
        raw: output?.raw_response?.[id] ?? null, unknown: stage?.state === "outcome_unknown" };
    });
  }
  function dispatches(summary, frameId) {
    return (summary?.outputs || []).filter((item) => item.frame_id === frameId);
  }
  function workSummary(work) {
    if (!work) return "Work plan unavailable";
    return `${work.image_count} images · at most ${work.request_count} external calls · ${work.image_encodings} local image encodings · ${work.prompt_evaluations} class-prompt evaluations · one pass, no iteration`;
  }
  function planError(plan) {
    if (!Array.isArray(plan?.requests) || !plan.requests.length)
      return "Prepare a complete combined request preview first.";
    for (const request of plan.requests) {
      const template = request.review?.template;
      if (!request.planning?.input?.image || typeof request.planning.input.prompt !== "string" ||
          !request.planning.input.prompt.trim() || !request.planning.input.request_sha256 ||
          !template || template.stage !== "review" || template.template !== true ||
          typeof template.prompt !== "string" || !template.prompt.trim() ||
          template.max_dynamic_text_bytes !== 131072 || !template.template_sha256 ||
          !Array.isArray(template.dynamic_fields) || !["planning_prompts", "candidates"].every((name) => template.dynamic_fields.includes(name)))
        return "The combined preview is incomplete. Inspect both the exact planning request and the bounded review template before approving.";
    }
    return null;
  }
  return { stageNames, availability, stages, dispatches, workSummary, planError };
});
