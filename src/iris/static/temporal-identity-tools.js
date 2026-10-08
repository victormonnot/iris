"use strict";

// Local reference edits never infer source positions or carry forward human review.
((root) => {
  const objectFields = ["identity_id", "label", "box", "visibility", "certainty"];
  const visibilityValues = new Set(["visible", "occluded", "out_of_view", "unknown"]);
  const certaintyValues = new Set(["certain", "uncertain"]);

  function clone(value) {
    if (Array.isArray(value)) return value.map(clone);
    if (value !== null && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, clone(item)]));
    }
    return value;
  }
  function text(value, description, maximum = 128) {
    if (typeof value !== "string" || !value.trim() || value.length > maximum || /[\u0000-\u001f]/.test(value)) {
      throw new Error(`${description} must contain bounded, nonempty text.`);
    }
    return value;
  }
  function index(value) {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error("Source frame index must be a nonnegative integer.");
    return value;
  }
  function reset(frame) {
    frame.coverage = "unreviewed";
    frame.review = { status: "unreviewed", reviewer: "" };
  }
  function frameAt(payload, frameIndex, create = false) {
    index(frameIndex);
    let frame = payload.frames.find((item) => item.frame_index === frameIndex);
    if (!frame && create) {
      frame = { frame_index: frameIndex, objects: [] };
      reset(frame);
      payload.frames.push(frame);
      payload.frames.sort((a, b) => a.frame_index - b.frame_index);
    }
    if (!frame) throw new Error("The selected frame has no reference annotation.");
    return frame;
  }
  function identityAt(payload, identifier) {
    const identity = payload.identities.find((item) => item.id === identifier);
    if (!identity) throw new Error("Select a declared reference identity.");
    return identity;
  }
  function objectAt(frame, objectIndex) {
    if (!Number.isInteger(objectIndex) || objectIndex < 0 || objectIndex >= frame.objects.length) {
      throw new Error("Select an existing reference object.");
    }
    return frame.objects[objectIndex];
  }
  function objectContent(object) {
    return objectFields.map((field) => object[field]);
  }
  function frameContent(frame) {
    return frame ? JSON.stringify(frame.objects.map(objectContent)) : null;
  }
  function checkUnique(frame) {
    const seen = new Set();
    for (const object of frame.objects) {
      if (object.identity_id === null) continue;
      if (seen.has(object.identity_id)) throw new Error("An identity cannot occur twice in the same frame.");
      seen.add(object.identity_id);
    }
  }
  function normalizedObject(payload, object) {
    if (!object || typeof object !== "object" || Array.isArray(object) ||
        Object.keys(object).length !== objectFields.length || objectFields.some((field) => !Object.hasOwn(object, field))) {
      throw new Error("Reference objects must contain exactly their supported fields.");
    }
    const result = clone(object);
    text(result.label, "Object class", 64);
    if (result.identity_id !== null && identityAt(payload, result.identity_id).label !== result.label) {
      throw new Error("Object class must match its reference identity.");
    }
    if (!visibilityValues.has(result.visibility)) throw new Error("Select a supported object visibility.");
    if (!certaintyValues.has(result.certainty)) throw new Error("Select certain or uncertain identity evidence.");
    if (result.identity_id === null || result.visibility === "unknown") result.certainty = "uncertain";
    if (result.visibility === "unknown" || result.visibility === "out_of_view") result.box = null;
    if (result.visibility === "visible" && result.box === null) throw new Error("A visible object needs an observed box.");
    if (result.box !== null) {
      const box = result.box;
      if (!Array.isArray(box) || box.length !== 4 || box.some((value) => typeof value !== "number" || !Number.isFinite(value) || value < 0) ||
          box[0] >= box[2] || box[1] >= box[3]) {
        throw new Error("An observed box needs four finite xyxy coordinates with positive area.");
      }
    }
    return result;
  }
  function blank(sequenceRecord) {
    const manifest = sequenceRecord.manifest;
    if (!manifest || !Array.isArray(manifest.frames) || typeof sequenceRecord.manifest_sha256 !== "string") {
      throw new Error("Select a saved temporal sequence with its manifest checksum.");
    }
    return {
      schema: "iris-temporal-reference-v2",
      sequence_id: manifest.id,
      sequence_sha256: sequenceRecord.manifest_sha256,
      taxonomy_id: manifest.taxonomy.id,
      identities: [],
      frames: manifest.frames.map((frame) => ({
        frame_index: frame.frame_index, coverage: "unreviewed",
        review: { status: "unreviewed", reviewer: "" }, objects: [],
      })),
      notes: "",
      provenance: { author: "", origin: null },
    };
  }
  function ensureFrame(payload, frameIndex) {
    const result = clone(payload);
    frameAt(result, frameIndex, true);
    return result;
  }
  function editObject(payload, frameIndex, objectIndex, patch) {
    if (!patch || typeof patch !== "object" || Array.isArray(patch) || Object.keys(patch).some((field) => !objectFields.includes(field))) {
      throw new Error("Object edits may change only supported reference fields.");
    }
    const result = clone(payload), frame = frameAt(result, frameIndex);
    const previous = objectAt(frame, objectIndex), edited = { ...previous, ...clone(patch) };
    if (Object.hasOwn(patch, "identity_id") && edited.identity_id !== null && !Object.hasOwn(patch, "label")) {
      edited.label = identityAt(result, edited.identity_id).label;
    }
    frame.objects[objectIndex] = normalizedObject(result, edited);
    checkUnique(frame);
    if (JSON.stringify(objectContent(previous)) !== JSON.stringify(objectContent(frame.objects[objectIndex]))) reset(frame);
    return result;
  }
  function addObject(payload, frameIndex, object) {
    const result = clone(payload), frame = frameAt(result, frameIndex, true);
    frame.objects.push(normalizedObject(result, object));
    checkUnique(frame);
    reset(frame);
    return result;
  }
  function removeObject(payload, frameIndex, objectIndex) {
    const result = clone(payload), frame = frameAt(result, frameIndex);
    objectAt(frame, objectIndex);
    frame.objects.splice(objectIndex, 1);
    reset(frame);
    return result;
  }
  function newIdentity(payload, label, identifier) {
    text(label, "Identity class", 64);
    text(identifier, "Reference identity ID");
    if (payload.identities.some((identity) => identity.id === identifier)) throw new Error("Choose a new, distinct reference identity ID.");
    const result = clone(payload);
    result.identities.push({ id: identifier, label });
    return result;
  }
  function removeIdentity(payload, identifier) {
    identityAt(payload, identifier);
    if (payload.frames.some((frame) => frame.objects.some((object) => object.identity_id === identifier))) {
      throw new Error("Reassign or remove this identity's objects before removing the identity.");
    }
    const result = clone(payload);
    result.identities = result.identities.filter((identity) => identity.id !== identifier);
    return result;
  }
  function updateIdentity(payload, identifier, label) {
    text(label, "Identity class", 64);
    const result = clone(payload), identity = identityAt(result, identifier);
    if (identity.label === label) return result;
    identity.label = label;
    for (const frame of result.frames) {
      const objects = frame.objects.filter((object) => object.identity_id === identifier);
      if (objects.length) {
        objects.forEach((object) => { object.label = label; });
        reset(frame);
      }
    }
    return result;
  }
  function assignObject(payload, frameIndex, objectIndex, identifier) {
    return editObject(payload, frameIndex, objectIndex, { identity_id: identifier });
  }
  function splitIdentity(payload, identifier, boundary, newIdentifier) {
    index(boundary);
    const identity = identityAt(payload, identifier);
    const occupied = payload.frames.filter((frame) => frame.objects.some((object) => object.identity_id === identifier));
    if (!occupied.some((frame) => frame.frame_index < boundary) || !occupied.some((frame) => frame.frame_index >= boundary)) {
      throw new Error("A split needs observations both before and from the selected source frame.");
    }
    const result = newIdentity(payload, identity.label, newIdentifier);
    for (const frame of result.frames) {
      if (frame.frame_index < boundary) continue;
      const objects = frame.objects.filter((object) => object.identity_id === identifier);
      if (objects.length) {
        objects.forEach((object) => { object.identity_id = newIdentifier; });
        reset(frame);
      }
    }
    return result;
  }
  function mergeIdentities(payload, keepIdentifier, removeIdentifier) {
    if (keepIdentifier === removeIdentifier) throw new Error("Select two distinct reference identities to merge.");
    const keep = identityAt(payload, keepIdentifier), remove = identityAt(payload, removeIdentifier);
    if (keep.label !== remove.label) throw new Error("Only identities with the same class can be merged.");
    for (const frame of payload.frames) {
      if (frame.objects.some((object) => object.identity_id === keepIdentifier) &&
          frame.objects.some((object) => object.identity_id === removeIdentifier)) {
        throw new Error(`Both identities occur in source frame ${frame.frame_index}; resolve the overlap before merging.`);
      }
    }
    const result = clone(payload);
    result.identities = result.identities.filter((identity) => identity.id !== removeIdentifier);
    for (const frame of result.frames) {
      const objects = frame.objects.filter((object) => object.identity_id === removeIdentifier);
      if (objects.length) {
        objects.forEach((object) => { object.identity_id = keepIdentifier; });
        reset(frame);
      }
    }
    return result;
  }
  function confirmFrame(payload, frameIndex, reviewer, coverage) {
    text(reviewer, "Human reviewer", 200);
    if (coverage !== "complete" && coverage !== "partial") throw new Error("Choose complete or partial human review.");
    const result = clone(payload), frame = frameAt(result, frameIndex, true);
    frame.objects = frame.objects.map((object) => normalizedObject(result, object));
    checkUnique(frame);
    if (coverage === "complete") {
      if (frame.objects.some((object) => object.identity_id === null || object.visibility === "unknown")) {
        throw new Error("Complete review needs known identities and visibility for every object.");
      }
      // This function is an explicit human confirmation action, never part of an edit or seed.
      frame.objects.forEach((object) => { object.certainty = "certain"; });
    }
    frame.coverage = coverage;
    frame.review = { status: "human_reviewed", reviewer };
    return result;
  }
  function changedFrames(before, after) {
    const previous = new Map(before.frames.map((frame) => [frame.frame_index, frame]));
    const current = new Map(after.frames.map((frame) => [frame.frame_index, frame]));
    return [...new Set([...previous.keys(), ...current.keys()])]
      .filter((frameIndex) => frameContent(previous.get(frameIndex)) !== frameContent(current.get(frameIndex)))
      .sort((a, b) => a - b);
  }
  function summary(payload, manifest) {
    const source = new Set(manifest.frames.map((frame) => frame.frame_index));
    const frames = payload.frames.filter((frame) => source.has(frame.frame_index));
    const human = frames.filter((frame) => frame.review.status === "human_reviewed");
    const assistant = frames.filter((frame) => frame.review.status === "assistant_reviewed");
    const humanComplete = human.filter((frame) => frame.coverage === "complete");
    const objects = frames.flatMap((frame) => frame.objects);
    const available = manifest.frames.length, clipCount = manifest.clip.end_frame - manifest.clip.start_frame + 1;
    return {
      available_frames: available,
      annotated_frames: human.length + assistant.length,
      omitted_frames: available - frames.length,
      human_reviewed_frames: human.length,
      assistant_reviewed_frames: assistant.length,
      complete_frames: frames.filter((frame) => frame.coverage === "complete").length,
      human_complete_frames: humanComplete.length,
      partial_frames: frames.filter((frame) => frame.coverage === "partial").length,
      unreviewed_frames: available - human.length - assistant.length,
      identity_count: payload.identities.length,
      object_count: objects.length,
      uncertain_objects: objects.filter((object) => object.certainty === "uncertain").length,
      dense_human_reference: available === clipCount && humanComplete.length === available,
    };
  }

  const exported = Object.freeze({ clone, blank, ensureFrame, editObject, addObject, removeObject,
    newIdentity, removeIdentity, updateIdentity, assignObject, splitIdentity, mergeIdentities,
    confirmFrame, changedFrames, summary });
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
  else root.IRISTemporalIdentityTools = exported;
})(typeof window !== "undefined" ? window : globalThis);
