"use strict";

// Presentation helpers do not associate objects, interpolate positions or score identities.
((root) => {
  function selectedSource(frames, assets) {
    const selected = (frames || []).filter((frame) => frame.selected);
    if (selected.length < 2) return { error: "Select at least two saved video frames in Data intake." };
    const asset = (assets || []).find((item) => item.id === selected[0].asset_id);
    if (!asset || asset.kind !== "video" || selected.some((frame) => frame.asset_id !== asset.id)) {
      return { error: "Select frames from exactly one video in the active session." };
    }
    if (selected.some((frame) => !Number.isInteger(frame.frame_index))) return { error: "Every selected frame needs a saved source frame index." };
    const ordered = [...selected].sort((a, b) => a.frame_index - b.frame_index);
    if (new Set(ordered.map((frame) => frame.frame_index)).size !== ordered.length) return { error: "Select only one saved image at each source frame index." };
    return { asset_id: asset.id, frames: ordered, name: asset.filename || asset.id,
      gap_count: ordered.at(-1).frame_index - ordered[0].frame_index + 1 - ordered.length };
  }
  function clockLabel(clock) {
    if (clock?.basis === "nominal_fps") return `Nominal source time · ${clock.fps} FPS · estimated, not capture timestamps`;
    if (clock?.basis === "provided") return "Provided source timestamps · caller-declared timing";
    return "Unknown source time · playback uses a display cadence of 2 analyzed frames/s";
  }
  function timeLabel(frame) {
    return typeof frame?.timestamp_seconds === "number" && Number.isFinite(frame.timestamp_seconds)
      ? `${frame.timestamp_seconds.toFixed(3)} s` : "Time unknown";
  }
  function playbackDelay(current, next, clock, speed = 1) {
    if (!next) return null;
    if (clock?.basis === "unknown" || typeof current?.timestamp_seconds !== "number" || typeof next.timestamp_seconds !== "number") return 500 / speed;
    return Math.max(1, (next.timestamp_seconds - current.timestamp_seconds) * 1000 / speed);
  }
  function gapBefore(frames, index) {
    return index > 0 ? Math.max(0, frames[index].frame_index - frames[index - 1].frame_index - 1) : 0;
  }
  function trails(frames, position, maximum = 40) {
    const segments = [], previous = new Map();
    for (let index = Math.max(0, position - maximum + 1); index <= position; index++) {
      const frame = frames[index];
      if (!frame) continue;
      const active = new Set();
      for (const observation of frame.observations || []) {
        const id = observation.track_id, box = observation.box;
        const point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
        let segment = previous.get(id);
        if (!segment || segment.position !== index - 1 || segment.source + 1 !== frame.frame_index) {
          segment = { track_id: id, points: [] }; segments.push(segment);
        }
        segment.points.push(point); segment.position = index; segment.source = frame.frame_index;
        previous.set(id, segment); active.add(id);
      }
      for (const id of previous.keys()) if (!active.has(id)) previous.delete(id);
    }
    return segments.filter((segment) => segment.points.length > 1).map(({ track_id, points }) => ({ track_id, points }));
  }
  function events(frames) {
    const result = [], lastSeen = new Map();
    let previous = new Map();
    frames.forEach((frame, position) => {
      const observations = frame.observations || [];
      const current = new Map(observations.map((item) => [item.track_id, item]));
      const gap = gapBefore(frames, position);
      if (gap) result.push({ position, kind: "source_gap", text: `${gap} missing source frame${gap === 1 ? "" : "s"} before frame ${frame.frame_index}; no tracker updates or visibility evidence.` });
      for (const [id] of previous) {
        if (!current.has(id)) result.push({ position, kind: "disappearance", track_id: id, text: `ID ${id}: no detector observation at frame ${frame.frame_index}. Visibility and cause are unknown.` });
      }
      for (const [id, observation] of current) {
        if (lastSeen.has(id) && !previous.has(id)) {
          result.push({ position, kind: "return", track_id: id, text: `ID ${id}: observation returns after ${position - lastSeen.get(id) - 1} analyzed updates without an observation. Same tracker ID; real identity is unverified.` });
        } else if (!lastSeen.has(id)) {
          result.push({ position, kind: "new_id", track_id: id, text: `ID ${id}: first observation${observation.confirmed ? "" : " (unconfirmed)"}. A new ID does not establish a new real object or an identity switch.` });
        }
        lastSeen.set(id, position);
      }
      let overlap = false;
      for (let a = 0; a < observations.length && !overlap; a++) for (let b = a + 1; b < observations.length; b++) {
        const x = observations[a].box, y = observations[b].box;
        if (Math.min(x[2], y[2]) > Math.max(x[0], y[0]) && Math.min(x[3], y[3]) > Math.max(x[1], y[1])) { overlap = true; break; }
      }
      if (overlap) result.push({ position, kind: "overlap", text: "Overlapping observed boxes: inspect possible crossing or occlusion. Geometry alone does not prove an identity error." });
      previous = current;
    });
    return result;
  }
  const exported = Object.freeze({ selectedSource, clockLabel, timeLabel, playbackDelay, gapBefore, trails, events });
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
  else root.IRISTrackingTools = exported;
})(typeof window !== "undefined" ? window : globalThis);
