"use strict";

// Display and launch validation only. Timings are taken from the fresh saved-frame run.
((root) => {
  function configuration({ name, lane_index, device, repeats, policy, cadence_fps }) {
    if (typeof name !== "string" || !name.trim() || name.trim().length > 160) throw new Error("Enter a run name of 1–160 characters.");
    if (lane_index !== 0 && lane_index !== 1) throw new Error("Choose one frozen tracker lane.");
    if (!["cpu", "cuda"].includes(device)) throw new Error("Choose CPU or CUDA.");
    if (!Number.isInteger(repeats) || repeats < 1 || repeats > 5) throw new Error("Use 1–5 complete repetitions.");
    if (!["offline_all", "simulated_latest"].includes(policy)) throw new Error("Choose a supported scheduling policy.");
    if (policy === "offline_all" && cadence_fps !== null) throw new Error("Offline runs do not declare an arrival cadence.");
    if (policy === "simulated_latest" && (typeof cadence_fps !== "number" || !Number.isFinite(cadence_fps) || cadence_fps < 0.1 || cadence_fps > 240)) throw new Error("Simulated arrival cadence must be 0.1–240 FPS.");
    return { name: name.trim(), lane_index, device, repeats, policy, cadence_fps };
  }
  function milliseconds(value) { return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)} ms` : "Unavailable"; }
  function fps(value) { return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)} FPS` : "Undefined"; }
  function memory(value) { return typeof value === "number" && Number.isFinite(value) ? `${(value / 1048576).toFixed(2)} MiB` : "Unavailable"; }
  function policyLabel(policy) { return policy === "simulated_latest" ? "Simulated latest available frame" : policy === "offline_all" ? "Offline · every available frame" : "Unknown policy"; }
  function active(job) { return ["queued", "running", "cancelling"].includes(job?.status); }
  function matchesContext(record, context) {
    if (!context || !record || record.comparison_id !== context.comparison_id || record.sequence_id !== context.sequence_id) return false;
    return !record.report || (record.report.source?.comparison_id === context.comparison_id && record.report.source?.sequence_id === context.sequence_id);
  }
  function laneIndex(record) { return record.report?.request?.lane_index ?? record.job?.params?.config?.lane_index ?? record.job?.result?.request?.lane_index ?? null; }
  function parseImport(raw) {
    if (typeof raw !== "string") throw new Error("Choose a completed cost report JSON file.");
    const report = JSON.parse(raw);
    if (!report || typeof report !== "object" || Array.isArray(report) || report.schema !== "iris-tracking-cost-v1" || report.complete !== true) throw new Error("Choose a completed iris-tracking-cost-v1 JSON report.");
    // Frozen reports distinguish integer and floating JSON values in their
    // canonical checksums. Parsing is for preview only: preserve source tokens
    // (120.0, 0.0, etc.) when wrapping the validated text for the import API.
    return { report, body: '{"report":' + raw + '}' };
  }
  const stages = Object.freeze([
    ["pipeline_ms", "Whole saved-frame pipeline", "outer"],
    ["verify_decode_ms", "Verify and decode source image", "outer"],
    ["detector_call_ms", "Detector call", "outer"],
    ["filter_ms", "Filter measured detections", "outer"],
    ["tracking_image_ms", "Prepare tracker image", "outer"],
    ["tracker_call_ms", "Tracker call", "outer"],
    ["detector_preprocess_ms", "Detector preprocessing", "nested"],
    ["detector_inference_ms", "Detector inference", "nested"],
    ["detector_postprocess_ms", "Detector postprocessing", "nested"],
    ["detector_crop_ms", "Detector crops", "nested"],
    ["detector_merge_ms", "Detector merge", "nested"],
    ["tracker_gmc_ms", "Tracker camera-motion compensation", "nested"],
    ["tracker_association_ms", "Tracker association", "nested"],
    ["tracker_adapter_ms", "Tracker adapter", "nested"],
  ]);
  const exported = Object.freeze({ configuration, milliseconds, fps, memory, policyLabel, active, matchesContext, laneIndex, parseImport, stages });
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
  else root.IRISTrackingCostTools = exported;
})(typeof window !== "undefined" ? window : globalThis);
