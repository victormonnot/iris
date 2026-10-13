"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISDemoAnnotations = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const clone = (value) => structuredClone(value);
  const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const length = (value) => [...value].length;

  function fail(message, status = 422) {
    const error = new Error(message);
    error.status = status;
    throw error;
  }

  function frames(seed) {
    const rows = seed.responses?.[`/api/sessions/${seed.session_id}/frames`];
    if (!Array.isArray(rows)) fail("The demo image records are unavailable.", 409);
    return rows;
  }

  function sourceDocument(seed, frameId) {
    if (typeof frameId !== "string" || !frames(seed).some((frame) => frame.id === frameId))
      fail("Annotation image not found.", 404);
    const source = seed.responses[`/api/frames/${frameId}/annotation`];
    if (!object(source)) fail("The demo annotation source is unavailable.", 409);
    return source;
  }

  function snapshots(state, frameId) {
    if (!object(state.annotations)) fail("The saved demo annotations are unavailable.", 409);
    const saved = own(state.annotations, frameId) ? state.annotations[frameId] : null;
    if (saved == null) return [];
    if (!object(saved) || !Array.isArray(saved.revisions))
      fail("The saved demo annotation history is unavailable.", 409);
    return saved.revisions;
  }

  function coordinates(box, frame) {
    if (!Array.isArray(box) || box.length !== 4 ||
        !box.every((value) => typeof value === "number" && Number.isFinite(value)))
      fail("Each box must contain four finite numeric xyxy pixel coordinates");
    const [x1, y1, x2, y2] = box;
    if (!(0 <= x1 && x1 < x2 && x2 <= frame.width && 0 <= y1 && y1 < y2 && y2 <= frame.height))
      fail("Boxes must have positive area and stay inside the original image");
    return [...box];
  }

  // Match iris.annotations: source and review state are derived, never accepted
  // from the request. The demo's immutable seed contains imported suggestions.
  function normalizeBoxes(boxes, source) {
    if (!Array.isArray(boxes) || boxes.length > 500)
      fail("Annotations support at most 500 boxes per image");
    const classes = new Set(source.taxonomy.classes.map((item) => item.id));
    const suggestions = new Map(source.suggestions.map((item) => [item.id, item]));
    const identifiers = new Set();
    const referenced = new Set();
    return boxes.map((box) => {
      if (!object(box)) fail("Each annotation box must be an object");
      const { id, label } = box;
      if (typeof id !== "string" || !id.trim() || length(id) > 128 || identifiers.has(id))
        fail("Each annotation box requires a distinct nonempty ID (128 max)");
      identifiers.add(id);
      if (typeof label !== "string" || !classes.has(label))
        fail("Annotation label must belong to this image's class version");
      const geometry = coordinates(box.box, source.frame);
      const suggestionId = box.suggestion_id ?? null;
      if (suggestionId !== null && (typeof suggestionId !== "string" || !suggestions.has(suggestionId)))
        fail("A referenced suggestion must belong to this image");
      if (referenced.has(suggestionId)) fail("A suggestion cannot be used by multiple annotation boxes");
      let reviewState = "manual";
      let origin = { kind: "manual" };
      if (suggestionId !== null) {
        referenced.add(suggestionId);
        const suggestion = suggestions.get(suggestionId);
        reviewState = label === suggestion.label && geometry.every((value, index) => value === suggestion.box[index])
          ? "accepted" : "corrected";
        origin = { kind: suggestion.kind, suggestion_id: suggestionId, metadata: clone(suggestion.metadata) };
      }
      return { id, label, box: geometry, suggestion_id: suggestionId, review_state: reviewState, source: origin };
    });
  }

  function validateDecisions(decisions, boxes, source, previous) {
    const suggestions = new Set(source.suggestions.map((item) => item.id));
    if (!object(decisions) || Object.entries(decisions).some(([id, value]) =>
      !suggestions.has(id) || !["accepted", "corrected", "rejected"].includes(value)))
      fail("Review decisions must reference this image's suggestions");
    if (previous && Object.keys(previous.decisions).some((id) => !own(decisions, id)))
      fail("Keep previous suggestion decisions in the new annotation snapshot");
    const references = new Map(boxes.filter((box) => box.suggestion_id !== null).map((box) => [box.suggestion_id, box]));
    for (const [id, box] of references)
      if (!own(decisions, id) || decisions[id] !== box.review_state)
        fail("Each suggested box requires its matching accepted/corrected decision");
    for (const [id, decision] of Object.entries(decisions)) {
      if (decision === "rejected" && references.has(id))
        fail("A rejected suggestion cannot remain in the annotation boxes");
      if (["accepted", "corrected"].includes(decision) && !references.has(id))
        fail("An accepted or corrected suggestion must have exactly one box");
    }
    if (previous) {
      const oldBoxes = new Map(previous.boxes.map((box) => [box.id, box]));
      for (const box of boxes) {
        const old = oldBoxes.get(box.id);
        if (old && old.suggestion_id !== box.suggestion_id)
          fail("An existing box cannot change its annotation origin");
      }
    }
  }

  function createState(seed) {
    return { annotations: {}, selection: Object.fromEntries(frames(seed).map((frame) => [frame.id, frame.selected])) };
  }

  function revisionRecord(source, state, snapshot) {
    return {
      id: `demo-${state.epoch}-${source.frame.id}-${snapshot.revision}`,
      frame_id: source.frame.id,
      revision: snapshot.revision,
      status: snapshot.status,
      taxonomy_id: source.taxonomy_id,
      taxonomy: clone(source.taxonomy),
      frame_sha256: source.frame_sha256,
      boxes: normalizeBoxes(snapshot.boxes, source),
      decisions: clone(snapshot.decisions),
      reviewer: snapshot.reviewer,
      notes: snapshot.notes,
      created_at: snapshot.created_at,
    };
  }

  function document(seed, state, frameId) {
    const source = sourceDocument(seed, frameId);
    const result = clone(source);
    const saved = snapshots(state, frameId);
    const latest = saved.at(-1);
    if (latest) {
      const record = revisionRecord(source, state, latest);
      for (const key of ["revision", "status", "boxes", "decisions", "reviewer", "notes"])
        result[key] = record[key];
      result.history = [...saved].reverse().map((item) => ({
        id: `demo-${state.epoch}-${frameId}-${item.revision}`,
        revision: item.revision, status: item.status,
        taxonomy_id: source.taxonomy_id, taxonomy: clone(source.taxonomy),
        box_count: item.boxes.length, reviewer: item.reviewer, notes: item.notes, created_at: item.created_at,
      }));
    }
    result.frame.selected = state.selection?.[frameId] ?? source.frame.selected;
    result.suggestions = result.suggestions.map((item) => ({ ...item, state: result.decisions[item.id] || "pending" }));
    result.demo_epoch = state.epoch;
    return result;
  }

  function revision(seed, state, frameId, number) {
    const source = sourceDocument(seed, frameId);
    const saved = snapshots(state, frameId).find((item) => item.revision === number);
    if (!Number.isSafeInteger(number) || number < 1 || !saved)
      fail("Annotation revision not found", 404);
    return revisionRecord(source, state, saved);
  }

  function save(seed, state, frameId, payload) {
    const source = sourceDocument(seed, frameId);
    if (!object(payload)) fail("Annotation payload must be an object");
    if (typeof state.epoch !== "string" || !state.epoch || payload.demo_epoch !== state.epoch)
      fail("The demo was reset or updated. Reload the image before saving.", 409);
    const saved = snapshots(state, frameId);
    const previous = saved.at(-1);
    const actual = previous?.revision || 0;
    if (!Number.isSafeInteger(payload.expected_revision) || payload.expected_revision < 0)
      fail("Expected revision must be a nonnegative integer");
    if (payload.expected_revision !== actual)
      fail(`Annotation changed: expected revision ${payload.expected_revision}, found ${actual}. Reload the image before saving.`, 409);
    if (payload.taxonomy_id != null && payload.taxonomy_id !== source.taxonomy_id)
      fail("The annotation class version changed. Reload the image before saving.", 409);
    const { status = "draft", reviewer = "", notes = "" } = payload;
    if (!["draft", "validated"].includes(status)) fail("Annotation status must be draft or validated");
    if (typeof reviewer !== "string" || length(reviewer.trim()) > 120)
      fail("Reviewer must contain at most 120 characters");
    if (typeof notes !== "string" || length(notes) > 4000)
      fail("Annotation notes must contain at most 4000 characters");
    if (status === "validated" && !reviewer.trim()) fail("Human validation requires a reviewer name");
    const boxes = normalizeBoxes(payload.boxes, source);
    validateDecisions(payload.decisions, boxes, source, previous);
    if (status === "validated" && source.suggestions.some((item) => !own(payload.decisions, item.id)))
      fail("Resolve every pending suggestion before validating this image");
    const snapshot = {
      revision: actual + 1, status,
      boxes: boxes.map(({ id, label, box, suggestion_id }) => ({ id, label, box, suggestion_id })),
      decisions: clone(payload.decisions), reviewer: reviewer.trim(), notes,
      created_at: new Date().toISOString(),
    };
    // A storage transaction persists this update only after this whole operation succeeds.
    state.annotations[frameId] = { revisions: [...saved, snapshot] };
    return document(seed, state, frameId);
  }

  return { createState, document, revision, save };
});
