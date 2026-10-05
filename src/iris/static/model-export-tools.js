"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISModelExportTools = tools;
})(typeof window === "undefined" ? globalThis : window, () => {
  const MAX_MEASUREMENT_BYTES = 8 * 1024 * 1024;
  function selectionValid(options) {
    return Boolean(options?.name?.trim() && options.trained_model_id && options.evaluation_id &&
      ["cpu", "cuda"].includes(options.target_device || "cpu") &&
      Array.isArray(options.frame_ids) && options.frame_ids.length >= 1 && options.frame_ids.length <= 8 &&
      options.frame_ids.every((id) => typeof id === "string" && id) && new Set(options.frame_ids).size === options.frame_ids.length);
  }
  function selectionKey(options) {
    return JSON.stringify([options.trained_model_id, options.evaluation_id, options.name, options.frame_ids, options.target_device || "cpu"]);
  }
  function findExport(rows, requestId) {
    return requestId && Array.isArray(rows) ? rows.find((row) => (row.request_id || row.config?.request_id) === requestId) || null : null;
  }
  function findMeasurement(rows, fingerprint) {
    return fingerprint && Array.isArray(rows) ? rows.find((row) => row.fingerprint === fingerprint) || null : null;
  }
  function fileProblem(file) {
    if (!file) return "Choose the measurement JSON written by the exported runner.";
    if (!Number.isSafeInteger(file.size) || file.size < 1) return "The measurement file is empty or its size is unavailable.";
    if (file.size > MAX_MEASUREMENT_BYTES) return "The measurement file must be at most 8 MiB.";
    return null;
  }
  function measurementPresentation(summary) {
    const parity = summary?.parity_passed === true ? "Exact parity passed" : summary?.parity_passed === false ? "Exact parity failed" : "Parity not verified";
    const kind = summary?.declaration || summary?.evidence_kind || summary?.provenance?.evidence_kind;
    const evidence = kind === "simulation" ? "Simulated measurements · no real model performance measured" :
      "Imported runner measurements · execution declared by the author, not independently verified";
    return { parity, evidence };
  }
  return { MAX_MEASUREMENT_BYTES, selectionValid, selectionKey, findExport, findMeasurement, fileProblem, measurementPresentation };
});
