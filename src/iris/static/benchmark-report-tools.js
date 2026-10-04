"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISBenchmarkReportTools = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const number = (value) => finite(value) ? value.toLocaleString("en-US", { maximumFractionDigits: 3 }) : "N/A";
  const percentage = (value) => finite(value) ? `${(value * 100).toFixed(1)}%` : "N/A";
  const approach = (value) => ({ multimodal: "A · Astra", segmentation: "B · SAM 3", combined: "C · Astra + SAM 3", local_detector: "Local detector control" })[value] || value || "Unknown approach";
  const evidence = (value) => ({ not_declared: "Evidence origin not declared", simulation: "Simulated evidence · does not measure real model performance", real_data: "Real data declared by the author · not independently verified" })[value] || "Evidence origin unavailable";
  const validComparison = (value, benchmarkId, role) => Boolean(value && value.protocol === "iris-benchmark-comparison-v1" &&
    value.benchmark?.id === benchmarkId && ["tuning", "evaluation"].includes(value.role) && (!role || value.role === role) &&
    Array.isArray(value.configs) && Array.isArray(value.reference?.frames));
  const validReport = (value, benchmarkId) => Boolean(value?.id && value.benchmark_id === benchmarkId && value.snapshot_sha256 &&
    validComparison(value.snapshot?.comparison, benchmarkId));
  function metricRange(value, format = number) {
    if (!value || !finite(value.count) || value.count < 1 || !finite(value.min) || !finite(value.max)) return "Unavailable";
    if (value.count === 1) return `${format(value.min)} · n=1`;
    return `${format(value.min)}–${format(value.max)} · mean ${format(value.mean)} · n=${number(value.count)}`;
  }
  function repeatability(value) {
    if (!value?.measured || value.complete_count < 2)
      return `Stability unavailable · ${number(value?.complete_count)} complete trial(s); at least two are needed.`;
    return `${number(value.complete_count)} complete trials · ${number(value.distinct_geometry_count)} distinct box sets${value.identical_geometry ? " · identical recorded geometry" : ""}. Repeated runs use the same images; they are not independent datasets.`;
  }
  function frameFor(comparison, frameId) {
    return comparison?.reference?.frames?.find((frame) => frame.frame_id === frameId) || null;
  }
  function trialFrame(config, trialId, frameId) {
    const trial = config?.trials?.find((item) => item.id === trialId);
    return { trial: trial || null, frame: trial?.frames?.find((item) => item.frame_id === frameId) || null };
  }
  function quality(value) {
    const summary = value?.complete === true ? value.metrics?.summary : null;
    return summary ? {
      errors: `${number(summary.fp)} extra / ${number(summary.fn)} missed`,
      conflicts: number(summary.class_conflicts),
      precisionRecall: `${percentage(summary.precision)} / ${percentage(summary.recall)}`,
      iou: percentage(summary.matched_iou_mean),
    } : { errors: "Incomplete · not scored", conflicts: "N/A", precisionRecall: "N/A", iou: "N/A" };
  }
  return { number, percentage, approach, evidence, validComparison, validReport, metricRange, repeatability, frameFor, trialFrame, quality };
});
