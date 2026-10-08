"use strict";

// Form validation and presentation. The server owns evaluation and provenance.
((root) => {
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const canonical = (value) => JSON.stringify(value, (_key, item) => item && typeof item === "object" && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]])) : item);
  const active = (job) => ["queued", "running", "cancelling"].includes(job?.status);
  const number = (value, label, minimum, maximum, integer = false) => {
    if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isInteger(value))) throw new Error(`${label} must be ${integer ? "an integer " : ""}between ${minimum} and ${maximum}.`);
    return value;
  };
  function profile(value) {
    const result = clone(value);
    if (!["bytetrack", "botsort"].includes(result.algorithm)) throw new Error("Choose ByteTrack or BoT-SORT.");
    for (const name of ["high_threshold", "low_threshold", "new_track_threshold", "match_threshold"]) number(result[name], name.replaceAll("_", " "), 0, 1);
    number(result.buffer_updates, "Lost buffer", 0, 10000, true);
    if (result.algorithm === "bytetrack") {
      if (result.low_threshold !== 0.1 || result.high_threshold <= 0.1 || result.high_threshold > 0.9 || Math.abs(result.new_track_threshold - result.high_threshold - 0.1) > 1e-12 || result.gmc_method !== "none") throw new Error("ByteTrack uses low 0.1, high greater than 0.1 and at most 0.9, birth high + 0.1, and no camera compensation.");
      result.new_track_threshold = result.high_threshold + 0.1;
    } else if (!(result.low_threshold < result.high_threshold && result.high_threshold <= result.new_track_threshold)) throw new Error("BoT-SORT needs low < high ≤ birth.");
    if (!["none", "sparseOptFlow"].includes(result.gmc_method) || typeof result.fuse_score !== "boolean" || result.with_reid !== false) throw new Error("Choose a supported camera compensation and score-fusion setting. Learned ReID is unavailable.");
    return result;
  }
  function editProfile(source, key, value) {
    const result = clone(source); result[key] = value;
    if (result.algorithm === "bytetrack") {
      result.low_threshold = 0.1; result.new_track_threshold = result.high_threshold + 0.1; result.gmc_method = "none";
    }
    return result;
  }
  function configuration(value) {
    const result = clone(value);
    if (typeof result.name !== "string" || !result.name.trim() || result.name.trim().length > 160) throw new Error("Enter a study name of 1–160 characters.");
    result.name = result.name.trim();
    if (!result.dataset_id) throw new Error("Choose a frozen temporal dataset.");
    if (!result.sources?.length || result.sources.length > 4 || result.sources.some((source) => !source.comparison_id)) throw new Error("Choose a completed comparison for every development and validation sequence (1–4 total).");
    if (!result.baseline?.profile) throw new Error("Choose a baseline lane from the first source comparison.");
    if (!result.candidates?.length || result.candidates.length > 7) throw new Error("Include 1–7 candidate profiles.");
    const profiles = [result.baseline, ...result.candidates];
    for (const item of profiles) {
      if (typeof item.name !== "string" || !item.name.trim() || item.name.trim().length > 80) throw new Error("Every profile needs a name of 1–80 characters.");
      item.name = item.name.trim(); item.profile = profile(item.profile);
      if (canonical(item.profile.class_ids) !== canonical(result.baseline.profile.class_ids)) throw new Error("All profiles must use the same native detector classes.");
    }
    if (new Set(profiles.map((item) => canonical(item.profile))).size !== profiles.length) throw new Error("Every included candidate must differ from the baseline and the other candidates.");
    if (new Set(profiles.map((item) => item.name)).size !== profiles.length) throw new Error("Baseline and candidate names must be distinct.");
    number(result.iou_threshold, "Matching IoU", 1e-12, 1);
    number(result.repeats, "Repetitions", 1, 3, true);
    number(result.max_updates, "Tracker update budget", 1, 20000, true);
    number(result.max_seconds, "Wall time limit", 1, 600);
    const keys = Object.keys(result.class_mapping || {}), ids = result.baseline.profile.class_ids.map(String);
    if (keys.length !== ids.length || ids.some((id) => !keys.includes(id) || result.class_mapping[id] === "")) throw new Error("Explicitly map or ignore every native class.");
    if (!keys.some((key) => result.class_mapping[key] !== null)) throw new Error("Map at least one native class.");
    return result;
  }
  function datasetRequest(name, rows) {
    if (typeof name !== "string" || !name.trim() || name.trim().length > 160) throw new Error("Enter a dataset name of 1–160 characters.");
    const entries = rows.filter((row) => row.selected).map(({ sequence_id, split, reference_id }) => ({ sequence_id, split, reference_id: reference_id || null }));
    if (!entries.length) throw new Error("Select at least one sequence for this frozen dataset.");
    if (!entries.some((entry) => entry.split !== "test")) throw new Error("Include a development or validation sequence. Test sequences stay reserved.");
    if (entries.some((entry) => !["train", "val", "test"].includes(entry.split) || (entry.split !== "test" && !entry.reference_id))) throw new Error("Pin a human reference revision for every development and validation sequence.");
    return { name: name.trim(), entries };
  }
  const splitLabel = (split) => ({ train: "Development", val: "Validation · tuning", test: "Reserved test · never evaluated" })[split] || split;
  const statusLabel = (status) => ({ gain: "Quality gain observed", tradeoff: "Trade-off observed", regression: "Quality regression", no_gain: "No quality gain", insufficient: "Insufficient evidence", unstable: "Repeatability differs", baseline: "Baseline", development_only: "Development evidence only" })[status] || (status ? status.replaceAll("_", " ") : "Unavailable");
  const percentage = (value) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Unavailable";
  const milliseconds = (value) => typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)} ms` : "Unavailable";
  const exported = Object.freeze({ clone, canonical, active, profile, editProfile, configuration, datasetRequest, splitLabel, statusLabel, percentage, milliseconds });
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
  else root.IRISTrackingStudyTools = exported;
})(typeof window !== "undefined" ? window : globalThis);
