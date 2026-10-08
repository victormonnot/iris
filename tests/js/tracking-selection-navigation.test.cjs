"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), vm = require("node:vm");
const tools = require("../../src/iris/static/tracking-selection-tools.js"), quality = require("../../src/iris/static/tracking-quality-tools.js");
const settle = () => new Promise(setImmediate);
const defer = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const descriptor = (id) => ({ kind: "comparison", job_id: id, sequence_id: `sequence-${id}`, profile_sha256: "a".repeat(64) });
function source(id = "source") {
  const frames = [0, 1, 2].map((index) => ({ frame_id: `${id}-${index}`, frame_index: index, timestamp_seconds: index / 10, width: 80, height: 60 }));
  return { source: descriptor(id), sequence: { id: `sequence-${id}`, name: `Sequence ${id}`, manifest: { frames, taxonomy: { id: "taxonomy", classes: [{ id: "person", coco_id: 1 }] } } }, references: [{ id: "reference", revision: 1, summary: { human_complete_frames: 3, available_frames: 3 } }], context: { current_dataset_memberships: [] }, replay: { profile: { algorithm: "bytetrack", class_ids: [1] }, cache: { config: { detector: { classes: [{ id: 1, name: "person" }], class_contract: { taxonomy_id: "coco-2017-v1" } } } }, passes: [{ frames: frames.map((frame) => ({ ...frame, input_size: [80, 60], observations: [{ detection_index: 0, track_id: 2, label: "person", box: [2, 3, 20, 40], score: .9, confirmed: true }] })) }] } };
}
function record(id, status = "succeeded") {
  const savedSource = source(id), request = { name: `Scenario ${id}`, source: descriptor(id), selection: { frame_id: `${id}-0`, detection_index: 0 }, release_frame_id: null, policy: { ...tools.defaults }, evaluation: null, max_seconds: 60 };
  const lane = (id, name) => ({ id, name, frames: savedSource.replay.passes[0].frames.map((frame) => ({ ...frame, state: "observed", reason: "continued", selected: frame.observations[0], pending: null, age: { updates: 0, seconds: 0 }, candidates: [], source_gap: false })), summary: { selected_frames: 3, recovery_events: 0, track_id_changes: 0, unobserved_frames: 0 }, quality: { status: "unavailable", reason: "No reference selected", coverage: { evaluated_frames: 0, active_frames: 3 }, counts: {}, recoveries: {}, durations: {} } });
  return { id, name: `Scenario ${id}`, job: { id, status, progress: 1, params: { request } }, report: status === "succeeded" ? { complete: true, request, repeatability: { status: "observed_match" }, lanes: [lane("track_id_only", "Track ID only"), lane("guarded_geometry", "Guarded geometry")], limitations: ["Geometric recovery is not proof of identity."], source_binding: {}, reference: null } : null };
}
function fixture(api, { query = "", autoImages = true } = {}) {
  const images = [];
  class Element extends EventTarget {
    constructor(tag = "div") { super(); this.tagName = tag; this.value = ""; this.children = []; this.dataset = {}; this.attributes = {}; this.hidden = false; this.disabled = false; this.style = {}; this.classList = { add() {}, remove() {} }; this.naturalWidth = 80; this.naturalHeight = 60; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    querySelectorAll(selector) { return this.children.filter((child) => child instanceof Element).flatMap((child) => [...(selector.split(",").includes(child.tagName) ? [child] : []), ...child.querySelectorAll(selector)]); }
    setAttribute(key, value) { this.attributes[key] = value; }
    remove() {}
    scrollIntoView() { this.scrolled = true; }
    set src(value) { this._src = value; images.push(this); if (autoImages) queueMicrotask(() => this.onload?.()); }
    get src() { return this._src; }
  }
  class Option extends Element { constructor(label, value) { super("option"); this.textContent = label; this.value = value; } }
  const elements = new Map(), $ = (key) => { if (!elements.has(key)) elements.set(key, new Element()); return elements.get(key); };
  Object.entries({ name: "Scenario", ...tools.defaults, seconds: "60", iou: ".5", lane: "guarded_geometry" }).forEach(([name, value]) => { $(`#selection-${name}`).value = String(value); });
  const window = new EventTarget(); window.location = { href: `http://localhost/?project=default${query}` }; window.IRISTrackingSelectionTools = tools; window.IRISTrackingQualityTools = quality;
  window.IRISNavigation = { open(name) { window.dispatchEvent(new CustomEvent("iris:workspace", { detail: { name } })); return true; } };
  const timers = new Map(); let timerID = 0;
  const node = (tag, className, text) => { const result = new Element(tag); result.className = className; result.textContent = text; return result; };
  vm.runInNewContext(fs.readFileSync(require.resolve("../../src/iris/static/tracking-selection.js"), "utf8"), {
    window, $, node, api, URL, Option, CustomEvent, state: { projectId: "default" }, projectURL: (path) => `${path}?project=default`,
    document: { createElementNS: (_ns, tag) => new Element(tag) },
    setTimeout(callback) { const id = ++timerID; timers.set(id, callback); return id; }, clearTimeout(id) { timers.delete(id); },
  });
  return { window, $, images, timers, selectSource(id) { $("#selection-source").value = tools.sourceKey(descriptor(id)); $("#selection-source").dispatchEvent(new Event("change")); }, runTimer() { assert.equal(timers.size, 1); const [id, callback] = [...timers][0]; timers.delete(id); return callback(); } };
}
function common(path, rows = ["source"], history = []) {
  if (path.endsWith("tracking-selection-sources")) return { sources: rows.map((id) => ({ source: descriptor(id), name: id, sequence_name: id, profile: { algorithm: "bytetrack" }, frame_count: 3 })) };
  if (path.endsWith("tracking-selection-status")) return { default_policy: { ...tools.defaults } };
  if (path.endsWith("tracking-selections")) return history;
  return null;
}

test("deep-linked saved scenarios read source evidence without queuing work", async () => {
  const calls = [], saved = record("saved");
  const { window, $ } = fixture(async (path, options) => { calls.push({ path, options }); return common(path, ["saved"], [saved]) || (path.endsWith("tracking-selection-source") ? source(JSON.parse(options.body).job_id) : saved); }, { query: "&tracking_selection=saved" });
  window.dispatchEvent(new Event("iris:project-initialized")); await settle();
  assert.equal($("#selection-result").hidden, false); assert.equal($("#selection-result-title").textContent, "Scenario saved");
  assert.match($("#selection-result-warning").textContent, /not a replay of an application/);
  assert.equal($("#selection-download").href, "/api/temporal/tracking-selections/saved/report?project=default");
  assert.ok(calls.every(({ path, options }) => !options?.method || path.endsWith("tracking-selection-source")));
});
test("late source responses cannot replace the newer selected source", async () => {
  const old = defer();
  const { window, $, selectSource } = fixture((path, options) => Promise.resolve(common(path, ["old", "current"]) || (JSON.parse(options.body).job_id === "old" ? old.promise : source("current"))));
  window.IRISNavigation.open("tracking"); await settle(); selectSource("old"); selectSource("current"); await settle(); old.resolve(source("old")); await settle();
  assert.match($("#selection-source-summary").textContent, /Sequence current/); assert.doesNotMatch($("#selection-source-summary").textContent, /Sequence old/);
});
test("late saved scenario opens and failures do not overwrite a newer explicit selection", async () => {
  const old = defer();
  const { window, $ } = fixture((path, options) => Promise.resolve(common(path, ["current"]) || (path.endsWith("tracking-selection-source") ? source(JSON.parse(options.body).job_id) : path.endsWith("/old") ? old.promise : record("current"))));
  window.IRISNavigation.open("tracking"); await settle(); window.IRISTrackingSelection.open("old"); window.IRISTrackingSelection.open("current"); await settle(); old.reject(new Error("Old scenario unavailable")); await settle();
  assert.equal($("#selection-result-title").textContent, "Scenario current"); assert.equal($("#selection-error").hidden, true);
});
test("source images must pass verification before mouse or keyboard selection", async () => {
  const calls = [], f = fixture(async (path, options) => { calls.push({ path, options }); return common(path) || source(); }, { autoImages: false });
  f.window.IRISNavigation.open("tracking"); await settle(); f.selectSource("source"); await settle();
  assert.equal(f.$("#selection-observations").children[0].disabled, true);
  f.images[0].naturalWidth = 79; f.images[0].onload(); assert.equal(f.$("#selection-observations").children[0].disabled, true);
  assert.match(f.$("#selection-image-status").textContent, /verification failed/);
  f.selectSource("source"); await settle(); f.images.at(-1).onload();
  f.$("#selection-observations").children[0].dispatchEvent(new Event("click"));
  assert.match(f.$("#selection-anchor").textContent, /source frame 0/); assert.equal(f.$("#selection-preview").disabled, false);
  assert.ok(calls.every(({ path, options }) => !options?.method || path.endsWith("tracking-selection-source")));
});
test("initial box and later release alter only the draft until explicit preview and run", async () => {
  const calls = [], f = fixture(async (path, options) => { calls.push({ path, options }); if (path.endsWith("/preview")) return { request: JSON.parse(options.body), fingerprint: "a".repeat(64) }; if (path.endsWith("tracking-selections") && options?.method === "POST") return record("new", "queued"); return common(path) || source(); });
  f.window.IRISNavigation.open("tracking"); await settle(); f.selectSource("source"); await settle(); f.$("#selection-observations").children[0].dispatchEvent(new Event("click"));
  f.$("#selection-next").dispatchEvent(new Event("click")); await settle(); f.$("#selection-set-release").dispatchEvent(new Event("click"));
  assert.match(f.$("#selection-release").textContent, /frame 1/); assert.ok(calls.every(({ path }) => !path.endsWith("/preview")));
  f.$("#selection-preview").dispatchEvent(new Event("click")); await settle(); assert.equal(f.$("#selection-run").disabled, false);
  const preview = JSON.parse(calls.find(({ path }) => path.endsWith("/preview")).options.body); assert.deepEqual(preview.selection, { frame_id: "source-0", detection_index: 0 }); assert.equal(preview.release_frame_id, "source-1");
  f.$("#selection-run").dispatchEvent(new Event("click")); await settle();
  assert.equal(calls.filter(({ path, options }) => path.endsWith("tracking-selections") && options?.method === "POST").length, 1);
});
test("late images and project-generation results are discarded", async () => {
  const f = fixture(async (path) => common(path) || source(), { autoImages: false });
  f.window.IRISNavigation.open("tracking"); await settle(); f.selectSource("source"); await settle();
  const image = f.images.at(-1); f.window.dispatchEvent(new Event("iris:project-initialized")); await settle(); image.onload();
  assert.equal(f.$("#selection-viewer").hidden, true); assert.equal(f.$("#selection-stage").children.length, 0);
});
test("returning after a background job finishes opens its completed evidence", async () => {
  let finished = false; const running = record("active", "running"), saved = record("active");
  const f = fixture(async (path, options) => common(path, ["active"], [finished ? saved : running]) || (path.endsWith("tracking-selection-source") ? source(JSON.parse(options.body).job_id) : finished ? saved : running));
  f.window.IRISNavigation.open("tracking"); await settle(); f.window.IRISTrackingSelection.open("active"); await settle();
  f.window.IRISNavigation.open("intake"); finished = true; f.window.IRISNavigation.open("tracking"); await settle();
  assert.equal(f.$("#selection-result").hidden, false); assert.equal(f.$("#selection-result-title").textContent, "Scenario active");
});

test("explicit report and timeline navigation scroll to the viewer while scrubbing does not", async () => {
  const f = fixture(async (path, options) => common(path, ["saved"], [record("saved")]) || (path.endsWith("tracking-selection-source") ? source(JSON.parse(options.body).job_id) : record("saved")));
  f.window.IRISNavigation.open("tracking"); await settle(); f.window.IRISTrackingSelection.open("saved"); await settle();
  assert.equal(f.$("#selection-viewer").scrolled, true);
  f.$("#selection-viewer").scrolled = false;
  f.$("#selection-next").dispatchEvent(new Event("click")); await settle();
  assert.equal(f.$("#selection-viewer").scrolled, false);
  f.$("#selection-timeline").children[0].dispatchEvent(new Event("click")); await settle();
  assert.equal(f.$("#selection-viewer").scrolled, true);
});
test("a pending preview cannot apply to a different scenario opened from Tasks", async () => {
  const preview = defer(), f = fixture((path, options) => Promise.resolve(path.endsWith("/preview") ? preview.promise : common(path) || (path.endsWith("tracking-selection-source") ? source(JSON.parse(options.body).job_id) : record("saved"))));
  f.window.IRISNavigation.open("tracking"); await settle(); f.selectSource("source"); await settle();
  f.$("#selection-observations").children[0].dispatchEvent(new Event("click"));
  f.$("#selection-preview").dispatchEvent(new Event("click")); await settle();
  f.window.IRISTrackingSelection.open("saved"); await settle();
  preview.resolve({ request: record("source").report.request, fingerprint: "a".repeat(64) }); await settle();
  assert.equal(f.$("#selection-result-title").textContent, "Scenario saved");
  assert.equal(f.$("#selection-preview-result").hidden, true);
  assert.equal(f.$("#selection-run").disabled, true);
});
