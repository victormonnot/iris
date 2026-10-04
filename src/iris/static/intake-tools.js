"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISIntakeTools = tools;
})(typeof window === "undefined" ? globalThis : window, () => {
  function filterFrames(frames, assets, insights, filters) {
    const sources = new Map(assets.map((asset) => [asset.id, asset]));
    const query = (filters.search || "").trim().toLocaleLowerCase();
    const filtered = frames.filter((frame) => {
      const info = insights.get(frame.id);
      if (filters.selected && !frame.selected) return false;
      if (filters.source && frame.asset_id !== filters.source) return false;
      if (query && !`${sources.get(frame.asset_id)?.filename || ""} ${frame.id} ${frame.frame_index ?? ""} ${frame.timestamp_seconds ?? ""}`.toLocaleLowerCase().includes(query)) return false;
      const review = filters.review || "all";
      if (review === "unreviewed" && info?.review_status !== "unannotated") return false;
      if (review === "draft" && info?.review_status !== "draft") return false;
      if (review === "pending" && info?.review_status !== "pending_suggestions") return false;
      if (review === "validated_positive" && info?.positive !== true) return false;
      if (review === "validated_negative" && info?.negative !== true) return false;
      const signal = filters.signal || "all";
      if (signal === "low_confidence" && !(info?.low_confidence_count > 0)) return false;
      if (signal === "no_detections" && info?.no_target_predictions !== true) return false;
      if (signal === "exact_duplicates" && !info?.exact_duplicate_ids?.length) return false;
      if (signal === "similar" && !info?.similar_frame_ids?.length) return false;
      return true;
    });
    if (filters.source) filtered.sort((a, b) => (a.timestamp_seconds ?? Infinity) - (b.timestamp_seconds ?? Infinity) || (a.frame_index ?? 0) - (b.frame_index ?? 0) || String(a.id).localeCompare(String(b.id)));
    return filtered;
  }

  function selectionPayload(frames, selected) {
    const targets = frames.filter((frame) => Boolean(frame.selected) !== selected);
    return { frame_ids: targets.map((frame) => frame.id), selected,
      expected_selection: Object.fromEntries(targets.map((frame) => [frame.id, Boolean(frame.selected)])) };
  }

  function createUploadQueue({ upload, onChange = () => {} }) {
    const queue = { entries: [], busy: false, sessionId: null };
    const changed = () => onChange(queue);
    async function run() {
      if (queue.busy) return;
      queue.busy = true;
      changed();
      try {
        for (const entry of queue.entries) {
          if (entry.status !== "pending") continue;
          entry.status = "uploading";
          entry.loaded = null;
          entry.total = null;
          entry.error = null;
          changed();
          try {
            const result = await upload(entry.file, queue.sessionId, (progress) => {
              if (entry.status !== "uploading" && entry.status !== "processing") return;
              entry.loaded = progress.loaded ?? entry.loaded;
              entry.total = progress.total ?? entry.total;
              if (progress.processing) entry.status = "processing";
              changed();
            });
            entry.status = result.import_status === "existing" ? "existing" : "succeeded";
            entry.assetId = result.id;
            entry.file = null;
          } catch (error) {
            entry.status = "failed";
            entry.error = error.message || "Import failed.";
          }
          changed();
        }
      } finally {
        queue.busy = false;
        changed();
      }
      return queue;
    }
    return {
      state: queue,
      start(files, sessionId) {
        if (queue.busy || !sessionId || !files.length) return Promise.resolve(queue);
        queue.sessionId = sessionId;
        queue.entries = [...files].map((file, index) => ({ id: index, name: file.name, size: file.size, file, status: "pending", loaded: null, total: null, error: null }));
        return run();
      },
      retryFailed() {
        if (queue.busy) return Promise.resolve(queue);
        for (const entry of queue.entries) if (entry.status === "failed") entry.status = "pending";
        return run();
      },
      cancelRemaining() {
        for (const entry of queue.entries) if (entry.status === "pending") {
          entry.status = "cancelled";
          entry.file = null;
        }
        changed();
      },
    };
  }
  return { filterFrames, selectionPayload, createUploadQueue };
});
