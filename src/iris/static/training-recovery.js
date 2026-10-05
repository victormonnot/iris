"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISTrainingRecovery = tools;
})(typeof window === "undefined" ? globalThis : window, () => {
  const LOSS_PAGE_SIZE = 100;

  function checkpointMinimum(steps) {
    return Number.isSafeInteger(steps) && steps > 0 ? Math.max(1, Math.ceil(steps / 200)) : 1;
  }

  function previewMatches(preview, payload) {
    return Boolean(preview?.fingerprint && preview.request_id &&
      preview.config?.scope === payload.scope && preview.scope?.id === payload.scope &&
      preview.dataset?.id === payload.dataset_id && preview.parent?.id === payload.parent_model_id &&
      preview.workload?.steps === payload.steps && preview.workload?.device === "cpu" &&
      preview.config?.checkpoint_interval === payload.checkpoint_interval &&
      preview.config?.learning_rate === payload.learning_rate && preview.config?.seed === payload.seed);
  }

  function findRequest(rows, requestId) {
    return requestId && Array.isArray(rows) ? rows.find((row) => row.config?.request_id === requestId) || null : null;
  }

  function findResume(rows, sourceId) {
    return sourceId && Array.isArray(rows) ? rows.find((row) => row.config?.resume_from?.training_id === sourceId) || null : null;
  }

  function resumeMatches(preview, detail) {
    const recovery = detail?.recovery;
    return Boolean(recovery?.can_resume && preview?.fingerprint &&
      preview.source_training_id === detail.id &&
      preview.checkpoint_id === recovery.checkpoint_id &&
      preview.checkpoint_step === recovery.checkpoint_step &&
      preview.target_steps === detail.config?.steps &&
      preview.remaining_steps === preview.target_steps - preview.checkpoint_step &&
      preview.remaining_steps >= 0 && preview.recorded_steps === recovery.recorded_steps &&
      preview.recomputed_steps === recovery.recomputed_steps);
  }

  function recoveryKey(detail) {
    return JSON.stringify([detail?.id, detail?.job?.status, detail?.config?.steps, detail?.recovery]);
  }

  function lossPage(history, page = 0) {
    const values = Array.isArray(history) ? history : [];
    const pages = Math.max(1, Math.ceil(values.length / LOSS_PAGE_SIZE));
    const index = Math.min(pages - 1, Math.max(0, Number.isSafeInteger(page) ? page : 0));
    const end = Math.max(0, values.length - index * LOSS_PAGE_SIZE);
    const start = Math.max(0, end - LOSS_PAGE_SIZE);
    return { items: values.slice(start, end), page: index, pages, start, end, total: values.length };
  }

  function observedDuration(history, targetSteps) {
    if (!Array.isArray(history) || history.length < 3 || !Number.isSafeInteger(targetSteps)) return null;
    const recent = history.slice(-21);
    const latest = recent.at(-1);
    let seconds = 0;
    let steps = 0;
    for (let index = 1; index < recent.length; index += 1) {
      const previous = recent[index - 1];
      const current = recent[index];
      const delta = current.elapsed_seconds - previous.elapsed_seconds;
      if (!Number.isFinite(current.elapsed_seconds) || !Number.isFinite(previous.elapsed_seconds) ||
          !Number.isSafeInteger(current.step) || !Number.isSafeInteger(previous.step) ||
          current.step !== previous.step + 1 || delta <= 0 || !Number.isFinite(delta)) return null;
      seconds += delta;
      steps += 1;
    }
    if (steps < 2 || latest.step > targetSteps) return null;
    return { secondsPerStep: seconds / steps, remainingSeconds: Math.max(0, targetSteps - latest.step) * seconds / steps, observedSteps: steps };
  }

  function durationText(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "Unavailable";
    if (seconds < 60) return `${Math.ceil(seconds)} s`;
    if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
    return `${(seconds / 3600).toFixed(1)} h`;
  }

  return { LOSS_PAGE_SIZE, checkpointMinimum, previewMatches, findRequest, findResume,
    resumeMatches, recoveryKey, lossPage, observedDuration, durationText };
});
