"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IrisAnnotationTools = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const MIN_ZOOM = 1 / 64;
  const MAX_ZOOM = 64;
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  function finite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value))
      throw new TypeError(`${name} must be a finite number.`);
    return value;
  }

  function positive(value, name) {
    finite(value, name);
    if (value <= 0) throw new RangeError(`${name} must be positive.`);
    return value;
  }

  function dimensions(width, height) {
    positive(width, "Image width");
    positive(height, "Image height");
  }

  function rectangle(view) {
    if (!view || typeof view !== "object")
      throw new TypeError("Viewport must be a rectangle.");
    finite(view.x, "Viewport x");
    finite(view.y, "Viewport y");
    positive(view.width, "Viewport width");
    positive(view.height, "Viewport height");
  }

  function origin(value, extent, imageExtent) {
    return extent > imageExtent
      ? (imageExtent - extent) / 2
      : clamp(value, 0, imageExtent - extent);
  }

  function fitViewport(width, height) {
    dimensions(width, height);
    return { x: 0, y: 0, width, height };
  }

  // Every view has the original image aspect ratio. A larger view centers the
  // image in blank padding; a smaller view cannot be panned beyond its edges.
  function clampViewport(view, width, height) {
    dimensions(width, height);
    rectangle(view);
    const scale = Math.max(view.width / width, view.height / height);
    const nextWidth = positive(width * scale, "Viewport width");
    const nextHeight = positive(height * scale, "Viewport height");
    return {
      x: origin(view.x + (view.width - nextWidth) / 2, nextWidth, width),
      y: origin(view.y + (view.height - nextHeight) / 2, nextHeight, height),
      width: nextWidth,
      height: nextHeight,
    };
  }

  function zoomViewport(view, factor, anchor, width, height, limits = {}) {
    const current = clampViewport(view, width, height);
    positive(factor, "Zoom factor");
    if (!anchor || typeof anchor !== "object")
      throw new TypeError("Zoom anchor must be a point.");
    finite(anchor.x, "Anchor x");
    finite(anchor.y, "Anchor y");
    const minZoom = positive(limits.minZoom ?? MIN_ZOOM, "Minimum zoom");
    const maxZoom = positive(limits.maxZoom ?? MAX_ZOOM, "Maximum zoom");
    if (minZoom > maxZoom)
      throw new RangeError("Minimum zoom cannot exceed maximum zoom.");
    const zoom = clamp((width / current.width) * factor, minZoom, maxZoom);
    const nextWidth = positive(width / zoom, "Viewport width");
    const nextHeight = positive(height / zoom, "Viewport height");
    return clampViewport(
      {
        x: anchor.x - ((anchor.x - current.x) / current.width) * nextWidth,
        y: anchor.y - ((anchor.y - current.y) / current.height) * nextHeight,
        width: nextWidth,
        height: nextHeight,
      },
      width,
      height,
    );
  }

  // Deltas are in original image pixels; callers convert screen deltas with
  // the SVG transform so dragging remains consistent at every zoom level.
  function panViewport(view, dx, dy, width, height) {
    const current = clampViewport(view, width, height);
    finite(dx, "Pan x");
    finite(dy, "Pan y");
    return {
      ...current,
      x: origin(current.x + dx, current.width, width),
      y: origin(current.y + dy, current.height, height),
    };
  }

  function focusViewport(box, width, height, padding = 1.5) {
    dimensions(width, height);
    positive(padding, "Focus padding");
    if (padding < 1) throw new RangeError("Focus padding must be at least 1.");
    let rect;
    if (Array.isArray(box)) {
      if (box.length !== 4) throw new TypeError("Box must contain four coordinates.");
      box.forEach((value) => finite(value, "Box coordinate"));
      rect = { x: box[0], y: box[1], width: box[2] - box[0], height: box[3] - box[1] };
    } else {
      rect = box;
    }
    rectangle(rect);
    const x1 = clamp(rect.x, 0, width);
    const y1 = clamp(rect.y, 0, height);
    const x2 = clamp(rect.x + rect.width, 0, width);
    const y2 = clamp(rect.y + rect.height, 0, height);
    if (x2 <= x1 || y2 <= y1) throw new RangeError("Box must overlap the image.");
    const scale = Math.min(
      1,
      Math.max(1 / MAX_ZOOM, ((x2 - x1) / width) * padding, ((y2 - y1) / height) * padding),
    );
    return clampViewport(
      {
        x: x1 + (x2 - x1) / 2 - (width * scale) / 2,
        y: y1 + (y2 - y1) / 2 - (height * scale) / 2,
        width: width * scale,
        height: height * scale,
      },
      width,
      height,
    );
  }

  // SVG preserveAspectRatio="xMidYMid meet" may leave letterboxing. The
  // smaller CSS/image ratio determines the exact 1 image pixel : 1 CSS pixel
  // view, including when a small image was enlarged by the fit view.
  function oneToOneViewport(view, width, height, canvasWidth, canvasHeight) {
    const current = clampViewport(view, width, height);
    positive(canvasWidth, "Canvas width");
    positive(canvasHeight, "Canvas height");
    const scale = Math.min(canvasWidth / width, canvasHeight / height);
    const nextWidth = positive(width * scale, "Viewport width");
    const nextHeight = positive(height * scale, "Viewport height");
    return clampViewport(
      {
        x: current.x + (current.width - nextWidth) / 2,
        y: current.y + (current.height - nextHeight) / 2,
        width: nextWidth,
        height: nextHeight,
      },
      width,
      height,
    );
  }

  // Snapshots contain JSON data. Sorting object keys makes equivalence
  // independent of insertion order while preserving array (box) order.
  function snapshotKey(snapshot) {
    const ancestors = new Set();
    function normalize(value) {
      if (value === null || typeof value === "string" || typeof value === "boolean")
        return value;
      if (typeof value === "number") return finite(value, "Snapshot number");
      if (typeof value !== "object")
        throw new TypeError("History snapshots must contain only JSON data.");
      if (ancestors.has(value)) throw new TypeError("History snapshots cannot be circular.");
      if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype)
        throw new TypeError("History snapshots must contain only JSON data.");
      ancestors.add(value);
      const result = Array.isArray(value)
        ? value.map(normalize)
        : Object.fromEntries(Object.keys(value).sort().map((key) => [key, normalize(value[key])]));
      ancestors.delete(value);
      return result;
    }
    return JSON.stringify(normalize(snapshot));
  }

  class SnapshotHistory {
    constructor({ limit = 100 } = {}) {
      if (!Number.isInteger(limit) || limit < 1 || limit > 100)
        throw new RangeError("History limit must be an integer between 1 and 100.");
      this.limit = limit;
      this.entries = [];
      this.index = -1;
      this.mergeKey = null;
    }

    reset(snapshot) {
      const key = snapshotKey(snapshot);
      this.entries = [{ value: structuredClone(snapshot), key }];
      this.index = 0;
      this.breakMerge();
    }

    breakMerge() {
      this.mergeKey = null;
    }

    commit(snapshot, mergeKey = null) {
      if (mergeKey !== null && typeof mergeKey !== "string")
        throw new TypeError("History merge key must be a string or null.");
      const key = snapshotKey(snapshot);
      if (this.index < 0) {
        this.reset(snapshot);
        return false;
      }
      if (key === this.entries[this.index].key) return false;
      const entry = { value: structuredClone(snapshot), key };
      const atEnd = this.index === this.entries.length - 1;
      if (mergeKey !== null && mergeKey === this.mergeKey && atEnd && this.index > 0) {
        if (key === this.entries[this.index - 1].key) {
          this.entries.pop();
          this.index -= 1;
          this.breakMerge();
          return true;
        }
        this.entries[this.index] = entry;
      } else {
        this.entries.splice(this.index + 1);
        this.entries.push(entry);
        // The baseline is retained in addition to `limit` undoable changes.
        if (this.entries.length > this.limit + 1) this.entries.shift();
        this.index = this.entries.length - 1;
      }
      this.mergeKey = mergeKey;
      return true;
    }

    get canUndo() {
      return this.index > 0;
    }

    get canRedo() {
      return this.index >= 0 && this.index < this.entries.length - 1;
    }

    undo() {
      this.breakMerge();
      if (!this.canUndo) return null;
      return structuredClone(this.entries[--this.index].value);
    }

    redo() {
      this.breakMerge();
      if (!this.canRedo) return null;
      return structuredClone(this.entries[++this.index].value);
    }
  }

  return {
    fitViewport,
    clampViewport,
    zoomViewport,
    panViewport,
    focusViewport,
    oneToOneViewport,
    SnapshotHistory,
  };
});
