"use strict";

// Presentation and explicit request validation only; scoring is a frozen server protocol.
((root) => {
  const canonical = (value) => JSON.stringify(value, function (_key, item) {
    return item && typeof item === "object" && !Array.isArray(item)
      ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]])) : item;
  });
  function percentage(value) { return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Not defined"; }
  function reason(value) {
    return ({ no_evaluated_frames: "No fully reviewed, scorable frames.", incomplete_reference_coverage: "IDF1 requires a fully dense, scorable human reference across the whole source clip.", no_ground_truth_identity_detections: "No ground-truth identity detections in the evaluated scope." })[value] || (value ? value.replaceAll("_", " ") : "Unavailable for this report.");
  }
  function nativeClasses(report) { return [...new Set((report?.lanes || []).flatMap((lane) => lane.report.profile.class_ids))].sort((a, b) => a - b); }
  function suggestions(detector, taxonomy, classIDs) {
    const contract = detector?.class_contract, known = {};
    if (contract?.taxonomy_id === "coco-2017-v1") {
      for (const item of taxonomy.classes) if (Number.isInteger(item.coco_id)) known[String(item.coco_id)] = item.id;
    } else if (contract?.taxonomy && canonical(contract.taxonomy) === canonical(taxonomy)) {
      for (const [label, id] of Object.entries(contract.output_class_mapping || {})) if (taxonomy.classes.some((item) => item.id === label)) known[String(id)] = label;
    }
    return Object.fromEntries(classIDs.filter((id) => Object.hasOwn(known, String(id))).map((id) => [String(id), known[String(id)]]));
  }
  function configuration(referenceID, classIDs, mapping, threshold, taxonomy) {
    if (!referenceID) throw new Error("Choose an explicit saved reference revision.");
    if (typeof threshold !== "number" || !Number.isFinite(threshold) || threshold <= 0 || threshold > 1) throw new Error("IoU threshold must be greater than 0 and no greater than 1.");
    const ids = classIDs.map(String), keys = Object.keys(mapping);
    if (keys.length !== ids.length || ids.some((id) => !Object.hasOwn(mapping, id))) throw new Error("Explicitly map or ignore every native detector class.");
    if (keys.some((id) => mapping[id] !== null && !taxonomy.classes.some((item) => item.id === mapping[id]))) throw new Error("Choose a frozen taxonomy class or explicitly ignore each native class.");
    if (!keys.some((id) => mapping[id] !== null)) throw new Error("Map at least one detector class to the frozen taxonomy.");
    return { reference_id: referenceID, class_mapping: { ...mapping }, iou_threshold: threshold };
  }
  function matchesContext(record, context) {
    return Boolean(context && record && record.comparison_id === context.comparison_id && record.sequence_id === context.sequence_id && record.report?.source?.comparison_id === context.comparison_id && record.report?.source?.sequence_id === context.sequence_id && record.reference_id === record.report.source.reference_id);
  }
  function referenceLabel(record) { return `Revision ${record.revision} · ${record.summary?.human_complete_frames || 0}/${record.summary?.available_frames || 0} complete human frames · ${record.payload?.provenance?.author || "legacy review provenance"}`; }
  function eventText(event) {
    const human = event.reference_identity || "unknown identity", track = event.track_id ?? "—";
    if (event.kind === "identity_switch") return `${human}: tracker ID ${event.previous_track_id} → ${track}`;
    if (event.kind === "fragment") return `${human}: matched again as tracker ID ${track} after a missed observation`;
    if (event.kind === "identity_transfer") return `Tracker ID ${track}: ${event.previous_reference_identity} → ${human}`;
    return `${event.kind?.replaceAll("_", " ") || "Event"}: ${human} · tracker ID ${track}`;
  }
  const exported = Object.freeze({ percentage, reason, nativeClasses, suggestions, configuration, matchesContext, referenceLabel, eventText });
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
  else root.IRISTrackingQualityTools = exported;
})(typeof window !== "undefined" ? window : globalThis);
