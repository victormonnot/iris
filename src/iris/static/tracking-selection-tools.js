"use strict";

(() => {
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const canonical = (value) => JSON.stringify(value, function (_key, item) {
    return item && typeof item === "object" && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]])) : item;
  });
  const defaults = Object.freeze({ schema: "iris-selection-policy-v1", min_score: 0.3, min_iou: 0.05, max_center_distance: 1, max_area_ratio: 3, max_lost_seconds: 1, max_lost_updates: 15, recovery_confirmation_updates: 2 });
  function number(value, label, minimum, maximum, integer = false) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isInteger(value))) throw new Error(`${label} must be ${integer ? "an integer" : "a number"} between ${minimum} and ${maximum}.`);
    return value;
  }
  function text(value, label, maximum = 160) {
    if (typeof value !== "string" || !value.trim() || value.trim().length > maximum) throw new Error(`${label} is required (${maximum} characters maximum).`);
    return value.trim();
  }
  function sourceKey(source) { return canonical(source); }
  function sourceLabel(row) { return `${row.name} · ${row.sequence_name || row.source.sequence_id} · ${row.profile.algorithm === "bytetrack" ? "ByteTrack" : "BoT-SORT"} · ${row.source.kind === "study" ? "saved study profile" : "saved comparison lane"}`; }
  function framePosition(sequence, frameID) { return sequence?.manifest?.frames.findIndex((frame) => frame.frame_id === frameID) ?? -1; }
  function anchor(source, selection) {
    if (!source || !selection) return null;
    const frame = source.replay.passes[0].frames.find((item) => item.frame_id === selection.frame_id);
    return frame?.observations.find((row) => row.detection_index === selection.detection_index) || null;
  }
  function configuration(value, source) {
    if (!source) throw new Error("Choose a saved source first.");
    if (!value.selection) throw new Error("Click a confirmed observed box to choose the initial object.");
    const observed = anchor(source, value.selection), policy = clone(value.policy);
    if (policy.schema !== defaults.schema || Object.keys(policy).some((key) => !Object.hasOwn(defaults, key))) throw new Error("Use the documented selection rules.");
    if (!observed?.confirmed) throw new Error("The initial selection must be a confirmed measured observation.");
    number(policy.min_score, "Minimum detector score", 0, 1);
    number(policy.min_iou, "Minimum overlap", 0, 1);
    number(policy.max_center_distance, "Maximum center distance", 0, 10);
    number(policy.max_area_ratio, "Maximum area ratio", 1, 100);
    if (policy.max_lost_seconds !== null) number(policy.max_lost_seconds, "Maximum lost source seconds", 0.001, 60);
    number(policy.max_lost_updates, "Maximum lost updates", 1, 1000, true);
    number(policy.recovery_confirmation_updates, "Recovery confirmations", 2, 10, true);
    if (observed.score < policy.min_score) throw new Error("The initial observation is below the chosen minimum score.");
    if (value.release_frame_id !== null && framePosition(source.sequence, value.release_frame_id) <= framePosition(source.sequence, value.selection.frame_id)) throw new Error("Release must occur on an available frame after the initial selection.");
    const evaluation = value.evaluation === null ? null : clone(value.evaluation);
    if (evaluation) {
      text(evaluation.reference_id, "Saved reference revision"); text(evaluation.identity_id, "Human reference identity");
      const ids = source.replay.profile.class_ids.map(String);
      if (!evaluation.class_mapping || Object.keys(evaluation.class_mapping).length !== ids.length || ids.some((id) => !Object.hasOwn(evaluation.class_mapping, id))) throw new Error("Map every source class to a reference class or explicitly ignore it.");
      for (const id of ids) if (evaluation.class_mapping[id] !== null) {
        text(evaluation.class_mapping[id], "Reference class");
        if (!source.sequence.manifest.taxonomy.classes.some((item) => item.id === evaluation.class_mapping[id])) throw new Error("Choose a class from this frozen reference taxonomy.");
      }
      if (!ids.some((id) => evaluation.class_mapping[id] !== null)) throw new Error("Map at least one source class for assessment.");
      number(evaluation.iou_threshold, "Assessment overlap", 1e-12, 1);
    }
    return { name: text(value.name, "Scenario name"), source: clone(source.source), selection: clone(value.selection), release_frame_id: value.release_frame_id, policy, evaluation, max_seconds: number(value.max_seconds, "Wall-time limit", 1, 120) };
  }
  function stateLabel(value) { return ({ idle: "Before selection", observed: "Observed", lost: "Lost", recovering: "Confirming recovery", ambiguous: "Ambiguous", recovered: "Recovered", expired: "Expired", released: "Released" })[value] || value || "Unavailable"; }
  function reasonLabel(value) {
    return ({ before_selection: "Initial selection has not happened yet", explicit_selection: "Initial measured object selected", initial_selection: "Initial measured object selected", explicit_release: "Released by the saved scenario", selection_released: "Selection remains released", selection_expired: "Selection remains expired", seconds_limit: "The source-time limit was reached", update_limit: "The available-update limit was reached", no_compatible_observation: "No compatible measured observation", multiple_compatible_observations: "Several measured objects are plausible", continued_observation: "The measured observation continues", same_track_returned: "The selected native ID returned", unique_candidate_confirmed: "One plausible candidate passed consecutive confirmation", candidate_needs_confirmation: "A plausible candidate needs more confirmation", no_reference: "No human reference was selected", source_replay_semantic_mismatch: "Repeated source tracker outputs differ", no_evaluable_frames: "No eligible human-reviewed frame supports assessment" })[value] || (value ? value.replaceAll("_", " ") : "No additional reason");
  }
  function duration(value) { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "Unavailable"; }
  function clockLabel(clock) {
    if (!clock || clock.basis === "unknown") return "Source time is unknown. Expiry uses available updates; durations cannot establish real camera latency.";
    const label = clock.basis === "nominal_fps" ? `Estimated source time from nominal ${clock.fps} FPS` : "Declared source timestamps";
    return `${label}. ${clock.provenance || "No clock provenance was declared."}`;
  }
  function active(job) { return ["queued", "running", "cancelling"].includes(job?.status); }
  const api = Object.freeze({ clone, canonical, defaults, sourceKey, sourceLabel, framePosition, anchor, configuration, stateLabel, reasonLabel, duration, clockLabel, active });
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.IRISTrackingSelectionTools = api;
})();
