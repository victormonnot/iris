"use strict";

(() => {
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const canonical = (value) => JSON.stringify(value, function (_key, item) {
    return item && typeof item === "object" && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]])) : item;
  });
  function text(value, label, maximum = 160) {
    if (typeof value !== "string" || !value.trim() || value.trim().length > maximum || /[\u0000-\u001f]/.test(value)) throw new Error(`${label} is required (${maximum} characters maximum).`);
    return value.trim();
  }
  function sourceKey(source) { return canonical(source); }
  function sourceLabel(row) { return `${row.name} · ${row.sequence_name || row.source.sequence_id} · ${row.profile.algorithm === "bytetrack" ? "ByteTrack" : "BoT-SORT"} · ${row.source.kind === "study" ? "study profile" : "comparison lane"}${row.available === false ? " · unavailable" : ""}`; }
  function configuration(value, source) {
    if (!source) throw new Error("Choose an exact saved comparison lane or study profile.");
    if (source.available === false) throw new Error(source.reason || "This source cannot be packaged.");
    const descriptor = source.source;
    if (!descriptor || !["comparison", "study"].includes(descriptor.kind) || !/^[a-f0-9]{64}$/.test(descriptor.profile_sha256)) throw new Error("Choose a saved source with a frozen tracker profile.");
    text(descriptor.job_id, "Source task", 128); text(descriptor.sequence_id, "Source sequence", 128);
    if (!["cpu", "cuda"].includes(value.target_device)) throw new Error("Choose CPU or NVIDIA CUDA as the intended detector target.");
    const selection = value.selection_id || null;
    if (selection !== null && !(source.selections || []).some((row) => row.id === selection)) throw new Error("Choose a completed continuity policy from this exact source profile.");
    return { name: text(value.name, "Bundle name"), source: clone(descriptor), selection_id: selection, target_device: value.target_device };
  }
  function bytes(value) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "Size unavailable";
    return value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(1)} MiB` : value >= 1024 ? `${(value / 1024).toFixed(1)} KiB` : `${value} bytes`;
  }
  function sourceSummary(row) {
    if (!row) return "Choose a completed source to inspect its detector and tracking settings.";
    if (row.available === false) return row.reason || "This saved source is unavailable for packaging.";
    const detector = row.detector || {}, profile = row.profile;
    const classes = profile.class_ids.map((id) => `${detector.classes?.find((item) => item.id === id)?.name || "class"} (${id})`).join(", ");
    return `${detector.architecture || "Frozen detector"} · ${classes} · saved score floor ${detector.min_score ?? "recorded in preview"}. ${profile.algorithm === "bytetrack" ? "ByteTrack" : "BoT-SORT"}: low ${profile.low_threshold}, high ${profile.high_threshold}, ${profile.buffer_updates} available-update memory. ${profile.gmc_method === "sparseOptFlow" ? "Camera compensation requires original BGR frames." : "Camera compensation is disabled."} Tracker runs on CPU; appearance re-identification is disabled.`;
  }
  function policySummary(policy) {
    if (!policy) return "No selected-object policy. The bundle contains detector and tracker settings only.";
    return `Experimental guarded geometry · minimum score ${policy.min_score} · overlap ${policy.min_iou} · center distance ${policy.max_center_distance} box diagonals · area ratio ${policy.max_area_ratio} · expiry ${policy.max_lost_updates} available updates${policy.max_lost_seconds === null ? " (no seconds limit)" : ` or ${policy.max_lost_seconds} source seconds`} · ${policy.recovery_confirmation_updates} consecutive recovery confirmations. Geometric recovery does not prove physical identity.`;
  }
  function active(job) { return ["queued", "running", "cancelling"].includes(job?.status); }
  const api = Object.freeze({ clone, canonical, sourceKey, sourceLabel, configuration, bytes, sourceSummary, policySummary, active });
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.IRISPipelineBundleTools = api;
})();
