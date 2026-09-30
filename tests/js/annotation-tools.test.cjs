"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const {
  fitViewport,
  clampViewport,
  zoomViewport,
  panViewport,
  focusViewport,
  oneToOneViewport,
  SnapshotHistory,
} = require("../../src/iris/static/annotation-tools.js");

function near(actual, expected, epsilon = 1e-9) {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} differs from ${expected}`);
}

function validView(view, width, height) {
  for (const value of Object.values(view)) assert.ok(Number.isFinite(value));
  assert.ok(view.width > 0 && view.height > 0);
  near(view.width / view.height, width / height);
  if (view.width <= width) {
    assert.ok(view.x >= -1e-9 && view.x + view.width <= width + 1e-9);
  } else {
    near(view.x, (width - view.width) / 2);
  }
  if (view.height <= height) {
    assert.ok(view.y >= -1e-9 && view.y + view.height <= height + 1e-9);
  } else {
    near(view.y, (height - view.height) / 2);
  }
}

test("the browser export works without a CommonJS module", () => {
  const window = {};
  vm.runInNewContext(
    readFileSync(join(__dirname, "../../src/iris/static/annotation-tools.js"), "utf8"),
    { window, globalThis: window, structuredClone },
  );
  assert.equal(typeof window.IrisAnnotationTools.zoomViewport, "function");
  assert.equal(typeof window.IrisAnnotationTools.SnapshotHistory, "function");
});

test("fit preserves original coordinates for a non-square image", () => {
  assert.deepEqual(fitViewport(1600, 900), { x: 0, y: 0, width: 1600, height: 900 });
});

test("clamping keeps a zoomed viewport inside every edge", () => {
  const corners = [
    [-100, -100, 0, 0],
    [2000, -100, 1200, 0],
    [-100, 2000, 0, 675],
    [2000, 2000, 1200, 675],
  ];
  for (const [x, y, expectedX, expectedY] of corners) {
    const view = clampViewport({ x, y, width: 400, height: 225 }, 1600, 900);
    assert.deepEqual(view, { x: expectedX, y: expectedY, width: 400, height: 225 });
  }
});

test("a viewport larger than the image stays centered in blank padding", () => {
  assert.deepEqual(
    clampViewport({ x: 900, y: -100, width: 1600, height: 1200 }, 800, 600),
    { x: -400, y: -300, width: 1600, height: 1200 },
  );
});

test("clamping restores image aspect without cropping the supplied view", () => {
  const view = clampViewport({ x: 600, y: 300, width: 400, height: 300 }, 1600, 900);
  validView(view, 1600, 900);
  near(view.width, 1600 / 3);
  near(view.height, 300);
  near(view.x + view.width / 2, 800);
  near(view.y + view.height / 2, 450);
});

test("zoom preserves the original image coordinate under the cursor", () => {
  const current = { x: 200, y: 100, width: 800, height: 450 };
  const anchor = { x: 400, y: 235 };
  const result = zoomViewport(current, 2, anchor, 1600, 900);
  near((anchor.x - result.x) / result.width, (anchor.x - current.x) / current.width);
  near((anchor.y - result.y) / result.height, (anchor.y - current.y) / current.height);
  assert.deepEqual(result, { x: 300, y: 167.5, width: 400, height: 225 });
});

test("zooming at each image corner leaves that corner visible", () => {
  for (const x of [0, 1600]) {
    for (const y of [0, 900]) {
      const view = zoomViewport(fitViewport(1600, 900), 64, { x, y }, 1600, 900);
      validView(view, 1600, 900);
      near(x === 0 ? view.x : view.x + view.width, x);
      near(y === 0 ? view.y : view.y + view.height, y);
    }
  }
});

test("zoom is bounded even when multiplication overflows", () => {
  const fit = fitViewport(640, 512);
  const anchor = { x: 320, y: 256 };
  const closest = zoomViewport(fit, Number.MAX_VALUE, anchor, 640, 512);
  near(closest.width, 10);
  const furthest = zoomViewport(fit, Number.MIN_VALUE, anchor, 640, 512);
  near(furthest.width, 640 * 64);
  validView(closest, 640, 512);
  validView(furthest, 640, 512);
});

test("custom zoom bounds keep an exact 1:1 view reachable for a tiny image", () => {
  const fit = fitViewport(2, 1);
  const one = oneToOneViewport(fit, 2, 1, 1200, 600);
  const result = zoomViewport(one, 1, { x: 1, y: 0.5 }, 2, 1, {
    minZoom: 1 / 600,
    maxZoom: 64,
  });
  assert.deepEqual(result, one);
  const zoomed = zoomViewport(one, 2, { x: 1, y: 0.5 }, 2, 1, {
    minZoom: 1 / 600,
    maxZoom: 64,
  });
  near(zoomed.width, one.width / 2);
});

test("pan adds image-coordinate deltas and clamps at the image edge", () => {
  const current = { x: 100, y: 100, width: 400, height: 225 };
  assert.deepEqual(panViewport(current, 50, -30, 1600, 900), {
    x: 150, y: 70, width: 400, height: 225,
  });
  assert.deepEqual(panViewport(current, -500, 2000, 1600, 900), {
    x: 0, y: 675, width: 400, height: 225,
  });
});

test("panning an image smaller than its viewport cannot move it off center", () => {
  const current = { x: -400, y: -300, width: 1600, height: 1200 };
  assert.deepEqual(panViewport(current, 150, -200, 800, 600), current);
});

test("focus fits a tall annotation into a wide image aspect", () => {
  const view = focusViewport([700, 200, 730, 600], 1600, 900);
  validView(view, 1600, 900);
  near(view.height, 600);
  near(view.x + view.width / 2, 715);
  near(view.y + view.height / 2, 400);
  assert.ok(view.x <= 700 && view.x + view.width >= 730);
  assert.ok(view.y <= 200 && view.y + view.height >= 600);
});

test("focus handles edge boxes, clips outside coordinates, and limits tiny boxes", () => {
  const corner = focusViewport([1590, 895, 1600, 900], 1600, 900);
  validView(corner, 1600, 900);
  near(corner.x + corner.width, 1600);
  near(corner.y + corner.height, 900);
  near(corner.width, 25);
  const clipped = focusViewport({ x: -20, y: -20, width: 40, height: 40 }, 1600, 900);
  validView(clipped, 1600, 900);
  assert.equal(clipped.x, 0);
  assert.equal(clipped.y, 0);
  assert.deepEqual(focusViewport([0, 0, 1600, 900], 1600, 900), fitViewport(1600, 900));
});

for (const [label, width, height, canvasWidth, canvasHeight] of [
  ["wide image letterboxed vertically", 4000, 1000, 800, 600],
  ["tall image letterboxed horizontally", 1000, 4000, 800, 600],
  ["small image enlarged by fit", 64, 32, 1024, 800],
  ["tiny image beyond default minimum zoom", 2, 1, 1200, 600],
  ["large image beyond default maximum zoom", 100000, 50000, 500, 500],
]) {
  test(`1:1 means one CSS pixel per image pixel: ${label}`, () => {
    const result = oneToOneViewport(fitViewport(width, height), width, height, canvasWidth, canvasHeight);
    validView(result, width, height);
    near(Math.min(canvasWidth / result.width, canvasHeight / result.height), 1);
  });
}

test("1:1 keeps the current visible center where the image boundary allows", () => {
  const current = { x: 1000, y: 800, width: 1000, height: 500 };
  const result = oneToOneViewport(current, 4000, 2000, 800, 600);
  near(result.x + result.width / 2, 1500);
  near(result.y + result.height / 2, 1050);
});

test("viewport helpers never mutate their input view or annotation", () => {
  const current = Object.freeze({ x: 100, y: 100, width: 400, height: 225 });
  const anchor = Object.freeze({ x: 200, y: 200 });
  clampViewport(current, 1600, 900);
  zoomViewport(current, 2, anchor, 1600, 900);
  panViewport(current, 10, 20, 1600, 900);
  oneToOneViewport(current, 1600, 900, 800, 600);
  focusViewport(Object.freeze([100, 100, 120, 130]), 1600, 900);
  assert.deepEqual(current, { x: 100, y: 100, width: 400, height: 225 });
});

test("a deterministic sequence of zooms and pans stays finite and inside bounds", () => {
  let view = fitViewport(1920, 1080);
  for (let index = 0; index < 500; index += 1) {
    const anchor = {
      x: view.x + view.width * ((index % 13) / 12),
      y: view.y + view.height * ((index % 7) / 6),
    };
    view = zoomViewport(view, index % 11 === 0 ? 0.25 : 1.1, anchor, 1920, 1080);
    view = panViewport(view, (index % 17) - 8, (index % 19) - 9, 1920, 1080);
    validView(view, 1920, 1080);
  }
});

test("invalid dimensions, rectangles, anchors, factors, and limits are rejected", () => {
  const fit = fitViewport(640, 512);
  for (const value of [0, -1, NaN, Infinity, "640", null]) {
    assert.throws(() => fitViewport(value, 512));
    assert.throws(() => fitViewport(640, value));
    assert.throws(() => zoomViewport(fit, value, { x: 0, y: 0 }, 640, 512));
    assert.throws(() => oneToOneViewport(fit, 640, 512, value, 600));
  }
  assert.throws(() => clampViewport({ ...fit, x: NaN }, 640, 512));
  assert.throws(() => clampViewport({ ...fit, width: -1 }, 640, 512));
  assert.throws(() => zoomViewport(fit, 2, { x: 0, y: Infinity }, 640, 512));
  assert.throws(() => zoomViewport(fit, 2, null, 640, 512));
  assert.throws(() => zoomViewport(fit, 2, { x: 0, y: 0 }, 640, 512, { minZoom: 4, maxZoom: 2 }));
  assert.throws(() => panViewport(fit, NaN, 0, 640, 512));
  assert.throws(() => focusViewport([0, 0, 0, 10], 640, 512));
  assert.throws(() => focusViewport([700, 0, 710, 10], 640, 512));
  assert.throws(() => focusViewport([0, 0, 10, 10], 640, 512, 0.5));
  assert.throws(() => focusViewport([0, 0, 10], 640, 512));
});

const snapshot = (notes = "", boxes = []) => ({ boxes, decisions: {}, notes, reviewer: "A" });

test("history preserves a baseline and returns each undo/redo state", () => {
  const history = new SnapshotHistory();
  assert.equal(history.canUndo, false);
  assert.equal(history.canRedo, false);
  assert.equal(history.undo(), null);
  history.reset(snapshot());
  history.commit(snapshot("First"));
  history.commit(snapshot("Second"));
  assert.equal(history.canUndo, true);
  assert.deepEqual(history.undo(), snapshot("First"));
  assert.deepEqual(history.undo(), snapshot());
  assert.equal(history.undo(), null);
  assert.equal(history.canRedo, true);
  assert.deepEqual(history.redo(), snapshot("First"));
  assert.deepEqual(history.redo(), snapshot("Second"));
  assert.equal(history.redo(), null);
});

test("history clones input states and returned nested box states", () => {
  const history = new SnapshotHistory();
  const initial = snapshot("", [{ id: "a", box: [1, 2, 3, 4] }]);
  history.reset(initial);
  initial.boxes[0].box[0] = 999;
  const changed = snapshot("Change", [{ id: "a", box: [5, 6, 7, 8] }]);
  history.commit(changed);
  changed.boxes[0].box[0] = 999;
  const undo = history.undo();
  assert.equal(undo.boxes[0].box[0], 1);
  undo.boxes[0].box[0] = 500;
  const redo = history.redo();
  assert.equal(redo.boxes[0].box[0], 5);
  redo.boxes[0].box[0] = 500;
  assert.equal(history.undo().boxes[0].box[0], 1);
  assert.equal(history.redo().boxes[0].box[0], 5);
});

test("a no-op commit after undo preserves redo, including reordered object keys", () => {
  const history = new SnapshotHistory();
  history.reset(snapshot());
  history.commit(snapshot("Saved"));
  history.undo();
  assert.equal(history.commit({ reviewer: "A", notes: "", decisions: {}, boxes: [] }), false);
  assert.equal(history.canRedo, true);
  assert.deepEqual(history.redo(), snapshot("Saved"));
});

test("a meaningful edit after undo replaces the redo branch", () => {
  const history = new SnapshotHistory();
  history.reset(snapshot());
  history.commit(snapshot("A"));
  history.commit(snapshot("B"));
  history.undo();
  assert.equal(history.commit(snapshot("C")), true);
  assert.equal(history.canRedo, false);
  assert.deepEqual(history.undo(), snapshot("A"));
  assert.deepEqual(history.redo(), snapshot("C"));
});

test("explicit text merge keys coalesce only contiguous edits", () => {
  const history = new SnapshotHistory();
  history.reset(snapshot());
  history.commit(snapshot("H"), "notes");
  history.commit(snapshot("Hi"), "notes");
  history.commit(snapshot("Hi!"), "notes");
  assert.deepEqual(history.undo(), snapshot());
  assert.deepEqual(history.redo(), snapshot("Hi!"));
  history.commit({ ...snapshot("Hi!"), reviewer: "B" }, "reviewer");
  history.commit({ ...snapshot("Bye"), reviewer: "B" }, "notes");
  assert.equal(history.undo().notes, "Hi!");
  assert.equal(history.undo().reviewer, "A");
});

test("undo and redo break a text merge even when the key is unchanged", () => {
  const history = new SnapshotHistory();
  history.reset(snapshot());
  history.commit(snapshot("A"), "notes");
  history.undo();
  history.redo();
  history.commit(snapshot("AB"), "notes");
  assert.deepEqual(history.undo(), snapshot("A"));
  assert.deepEqual(history.undo(), snapshot());
});

test("breaking a merge separates edits to the same field", () => {
  const history = new SnapshotHistory();
  history.reset(snapshot());
  history.commit(snapshot("A"), "notes");
  history.breakMerge();
  history.commit(snapshot("AB"), "notes");
  assert.deepEqual(history.undo(), snapshot("A"));
});

test("a coalesced edit returning to its starting state adds no empty undo step", () => {
  const history = new SnapshotHistory();
  history.reset(snapshot());
  history.commit(snapshot("A"), "notes");
  history.commit(snapshot(), "notes");
  assert.equal(history.canUndo, false);
  assert.equal(history.canRedo, false);
  history.commit(snapshot("B"), "notes");
  assert.deepEqual(history.undo(), snapshot());
});

test("a non-text edit stops text coalescing and keeps its own undo step", () => {
  const history = new SnapshotHistory();
  history.reset(snapshot());
  history.commit(snapshot("A"), "notes");
  const withBox = snapshot("A", [{ id: "box", box: [1, 2, 3, 4] }]);
  history.commit(withBox);
  history.commit({ ...withBox, notes: "AB" }, "notes");
  assert.deepEqual(history.undo(), withBox);
  assert.deepEqual(history.undo(), snapshot("A"));
});

test("the history limit drops oldest entries while preserving the current state", () => {
  const history = new SnapshotHistory({ limit: 3 });
  history.reset(snapshot("0"));
  for (let index = 1; index <= 6; index += 1) history.commit(snapshot(String(index)));
  assert.deepEqual(history.undo(), snapshot("5"));
  assert.deepEqual(history.undo(), snapshot("4"));
  assert.deepEqual(history.undo(), snapshot("3"));
  assert.equal(history.undo(), null);
  assert.deepEqual(history.redo(), snapshot("4"));
  assert.deepEqual(history.redo(), snapshot("5"));
  assert.deepEqual(history.redo(), snapshot("6"));
  assert.equal(history.redo(), null);
});

test("the default history retains one hundred undoable changes plus their baseline", () => {
  const history = new SnapshotHistory();
  history.reset(snapshot("0"));
  for (let index = 1; index <= 150; index += 1) history.commit(snapshot(String(index)));
  for (let index = 149; index >= 50; index -= 1)
    assert.deepEqual(history.undo(), snapshot(String(index)));
  assert.equal(history.undo(), null);
  assert.equal(history.canRedo, true);
});

test("reset discards both previous frame history and text merge state", () => {
  const history = new SnapshotHistory();
  history.reset(snapshot("A"));
  history.commit(snapshot("AB"), "notes");
  history.undo();
  history.reset(snapshot("New frame"));
  assert.equal(history.canUndo, false);
  assert.equal(history.canRedo, false);
  history.commit(snapshot("New frame!"), "notes");
  assert.deepEqual(history.undo(), snapshot("New frame"));
});

test("history rejects unsupported snapshots, invalid limits, and merge keys", () => {
  for (const limit of [0, -1, 101, 3.5, "10", NaN])
    assert.throws(() => new SnapshotHistory({ limit }));
  const history = new SnapshotHistory();
  history.reset(snapshot());
  for (const invalid of [undefined, { f() {} }, new Date(), { value: NaN }])
    assert.throws(() => history.commit(invalid));
  const circular = {};
  circular.self = circular;
  assert.throws(() => history.commit(circular));
  assert.throws(() => history.commit(snapshot("A"), 1));
  assert.equal(history.canUndo, false);
});
