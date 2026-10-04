"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISBenchmarkTools = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const canonical = (value) => JSON.stringify(value, function (key, item) {
    return item && typeof item === "object" && !Array.isArray(item)
      ? Object.fromEntries(Object.keys(item).sort().map((name) => [name, item[name]])) : item;
  });
  function duration(milliseconds) {
    if (typeof milliseconds !== "number" || !Number.isFinite(milliseconds) || milliseconds < 0) return "Unmeasured";
    const seconds = milliseconds / 1000;
    return seconds < 60 ? `${seconds.toFixed(1)} s` : `${Math.floor(seconds / 60)} min ${(seconds % 60).toFixed(1)} s`;
  }
  const ownsRunningTimer = (timer, token) => timer?.state === "running" && Boolean(token) && timer.owner_token === token;
  const canSaveCorrection = (timer, token, active) => timer?.state !== "running" || active && ownsRunningTimer(timer, token);
  const currentTrial = (trial, benchmarkId, trialId) => Boolean(trial && benchmarkId && trialId && trial.benchmark_id === benchmarkId && trial.id === trialId);
  function correctionMatches(record, payload, previousRevision) {
    return record?.revision > previousRevision && record.status === payload.status &&
      canonical(record.boxes) === canonical(payload.boxes) && record.reviewer === payload.reviewer && record.notes === payload.notes;
  }
  function boxAfterDrag(original, kind, start, point, width, height) {
    const [x1, y1, x2, y2] = original;
    if (kind === "move") {
      const dx = Math.max(-x1, Math.min(width - x2, point[0] - start[0]));
      const dy = Math.max(-y1, Math.min(height - y2, point[1] - start[1]));
      return [x1 + dx, y1 + dy, x2 + dx, y2 + dy];
    }
    return [
      kind.includes("w") ? Math.max(0, Math.min(point[0], x2 - 1)) : x1,
      kind.includes("n") ? Math.max(0, Math.min(point[1], y2 - 1)) : y1,
      kind.includes("e") ? Math.min(width, Math.max(point[0], x1 + 1)) : x2,
      kind.includes("s") ? Math.min(height, Math.max(point[1], y1 + 1)) : y2,
    ];
  }
  function referenceSelection(groups, roles, chosen) {
    const frame_ids = [], selectedRoles = Object.create(null), counts = { tuning: 0, evaluation: 0 };
    for (const group of groups) {
      const role = roles.get(group.scene_group);
      if (!Object.hasOwn(counts, role)) continue;
      const ids = group.frames.filter((frame) => chosen.has(frame.id || frame.frame_id)).map((frame) => frame.id || frame.frame_id);
      if (!ids.length) continue;
      selectedRoles[group.scene_group] = role;
      frame_ids.push(...ids);
      counts[role] += ids.length;
    }
    return { frame_ids, roles: selectedRoles, counts, valid: counts.tuning > 0 && counts.evaluation > 0 && counts.tuning <= 25 && counts.evaluation <= 25 };
  }
  return { canonical, duration, ownsRunningTimer, canSaveCorrection, currentTrial, correctionMatches, boxAfterDrag, referenceSelection };
});
