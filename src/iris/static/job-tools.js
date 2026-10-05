"use strict";
((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISJobTools = tools;
})(typeof window === "undefined" ? globalThis : window, () => {
  const active = (job) => ["queued", "running"].includes(job.status);
  const kindNames = { extract: "Frame extraction", infer: "Model comparison", assist: "Annotation assistance", train: "Detector training", evaluate: "Quality evaluation", model_export: "Model export", video_review: "Video passage review", benchmark: "Preannotation benchmark" };
  const statusName = (status) => String(status || "unknown").replaceAll("_", " ");
  function history(jobs, { status = "all", kind = "all", query = "", limit = 8 } = {}) {
    const search = query.trim().toLocaleLowerCase();
    const filtered = jobs.filter((job) => {
      if (kind !== "all" && job.kind !== kind) return false;
      if (status === "active" && !active(job)) return false;
      if (status === "attention" && !["failed", "interrupted"].includes(job.status)) return false;
      if (!["all", "active", "attention"].includes(status) && job.status !== status) return false;
      return !search || `${job.id} ${kindNames[job.kind] || job.kind} ${job.message || ""} ${job.error || ""}`.toLocaleLowerCase().includes(search);
    }).sort((a, b) => Number(active(b)) - Number(active(a)) || String(b.created_at).localeCompare(String(a.created_at)));
    const current = filtered.filter(active);
    const complete = filtered.filter((job) => !active(job));
    return { rows: [...current, ...complete.slice(0, limit)], total: filtered.length, more: complete.length > limit };
  }
  function dispatchPresentation(dispatch) {
    if (!dispatch) return null;
    const labels = { not_started: "Not sent", dispatching: "Request in progress", response_received: "Response received", outcome_unknown: "Delivery outcome unknown" };
    let explanation = dispatch.state === "outcome_unknown"
      ? `The provider may have processed this request. IRIS will not resend it automatically. Check the saved evidence before preparing a new request.${dispatch.external ? " Another request may incur another charge; check the provider records." : ""}`
      : dispatch.state === "response_received"
        ? `A response was recorded. This does not by itself confirm a usable result${dispatch.external ? " or a final charge" : ""}.`
        : dispatch.state === "dispatching"
          ? "The request was handed to the provider. Cancelling locally cannot guarantee that the provider stops processing it."
          : "No request dispatch has been recorded.";
    if (dispatch.counts) {
      const counts = dispatch.counts;
      explanation = `External requests: ${counts.response_received || 0} responses recorded · ${counts.dispatching || 0} in progress · ${counts.outcome_unknown || 0} outcomes unknown · ${counts.not_started || 0} not sent. ${explanation}`;
    }
    return { label: labels[dispatch.state] || statusName(dispatch.state), explanation, unknown: dispatch.state === "outcome_unknown" };
  }
  function canContinue(detail, preview) {
    return Boolean(detail?.recovery?.can_check && !active(detail.job) && detail.job.kind === "extract" &&
      !detail.dispatch?.external && preview?.source_job_id === detail.job.id && preview.available === true &&
      preview.mode === "continue_extraction" && typeof preview.fingerprint === "string" && preview.fingerprint && preview.remaining_count > 0);
  }
  function findApprovedRequest(records, previewId) {
    if (!previewId || !Array.isArray(records)) return null;
    return records.find((record) => record.config?.consent?.preview_id === previewId && (record.job_id || record.job?.id)) || null;
  }
  function retryableBatchFrames(detail) {
    if (!detail || detail.counts?.queued || detail.counts?.running) return [];
    return (detail.frames || []).filter((frame) => ["failed", "cancelled", "interrupted"].includes(frame.status) && !(frame.suggestions_created > 0));
  }
  return { active, kindNames, statusName, history, dispatchPresentation, canContinue, findApprovedRequest, retryableBatchFrames };
});
