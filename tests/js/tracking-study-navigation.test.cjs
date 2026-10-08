"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), vm = require("node:vm");
const tools = require("../../src/iris/static/tracking-study-tools.js");
const quality = require("../../src/iris/static/tracking-quality-tools.js");
const settle = () => new Promise(setImmediate);
const defer = () => { let resolve, reject; const promise = new Promise((done, fail) => { resolve = done; reject = fail; }); return { promise, resolve, reject }; };
function fixture(api, query = "") {
  class Element extends EventTarget {
    constructor(tag = "div") { super(); this.tagName = tag; this.value = ""; this.children = []; this.dataset = {}; this.attributes = {}; this.hidden = false; }
    get options() { return this.children; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    querySelectorAll() { return []; }
    setAttribute(key, value) { this.attributes[key] = value; }
    removeAttribute(key) { delete this.attributes[key]; }
    scrollIntoView() { this.scrolled = true; }
  }
  class Option extends Element { constructor(label, value) { super("option"); this.textContent = label; this.value = value; } }
  const elements = new Map(), $ = (selector) => { if (!elements.has(selector)) elements.set(selector, new Element()); return elements.get(selector); };
  for (const [name, value] of Object.entries({ name: "Study", iou: ".5", repeats: "2", updates: "5000", seconds: "120" })) $(`#study-${name}`).value = value;
  const window = new EventTarget(); window.location = { href: `http://localhost/?project=default${query}` };
  window.IRISTrackingStudyTools = tools; window.IRISTrackingQualityTools = quality;
  window.IRISNavigation = { open(name) { window.dispatchEvent(new CustomEvent("iris:workspace", { detail: { name } })); return true; } };
  const timers = new Map(); let timerID = 0;
  const node = (tag, className, text) => { const item = new Element(tag); item.className = className; item.textContent = text; return item; };
  vm.runInNewContext(fs.readFileSync(require.resolve("../../src/iris/static/tracking-study.js"), "utf8"), {
    window, $, node, api, URL, Option, CustomEvent, state: { projectId: "default" }, projectURL: (path) => `${path}?project=default`,
    document: { createTextNode: (text) => text },
    setTimeout(callback) { const id = ++timerID; timers.set(id, callback); return id; }, clearTimeout(id) { timers.delete(id); },
  });
  return { window, $, timers, runTimer() { assert.equal(timers.size, 1); const [id, callback] = [...timers][0]; timers.delete(id); return callback(); } };
}
const record = (id, status = "running") => ({ id, name: `Study ${id}`, dataset_id: "dataset", job: { id, status, message: "saved" }, report: status === "succeeded" ? {
  complete: true, schema: "iris-tracking-study-v1", request: {}, dataset: {}, sources: [],
  summary: { decision: { applied: false, reason: "development_only" }, splits: { train: null, val: null }, source_results: [{ sequence_id: "source", split: "train", reference_id: "reference", coverage: { reference_origin: { comparison_id: "seed" }, dense: true, available_frames: 8, evaluated_frames: 8 } }], limitations: ["No independent test evidence."] },
} : null });
function common(path) {
  if (path.endsWith("tracking-study-sources")) return { datasets: [], sequences: [] };
  if (path.endsWith("tracking-study-status")) return { runtime: { available: true } };
  return null;
}

test("opening a deep-linked study loads saved evidence without preview, suggestions or job POSTs", async () => {
  const calls = [], saved = record("saved", "succeeded");
  const { window, $ } = fixture(async (path, options) => { calls.push({ path, options }); return common(path) || (path.endsWith("/saved") ? saved : [saved]); }, "&tracking_study=saved");
  window.dispatchEvent(new Event("iris:project-initialized")); await settle();
  assert.equal($("#study-result").hidden, false);
  assert.equal($("#study-result-title").textContent, "Study saved");
  assert.match($("#study-decision").textContent, /Baseline retained.*No profile was applied/);
  assert.equal($("#study-origin").hidden, false);
  assert.match($("#study-origin").textContent, /tracker proposals/);
  assert.equal($("#study-download").href, "/api/temporal/tracking-studies/saved/report?project=default");
  assert.ok(calls.every((call) => !call.options?.method || call.options.method === "GET"));
  assert.ok(calls.every((call) => !/preview|suggestions/.test(call.path)));
});

test("a late study result cannot replace a newer explicit history selection", async () => {
  const old = defer(), current = defer();
  const { window, $ } = fixture((path) => Promise.resolve(common(path) || (path.endsWith("/old") ? old.promise : path.endsWith("/current") ? current.promise : [])));
  window.IRISNavigation.open("tracking"); await settle();
  window.IRISTrackingStudy.open("old"); window.IRISTrackingStudy.open("current");
  current.resolve(record("current", "succeeded")); await settle();
  old.resolve(record("old", "succeeded")); await settle();
  assert.equal($("#study-result-title").textContent, "Study current");
  assert.equal($("#study-history").value, "current");
});

test("a late failed open cannot put an error on a newer valid study", async () => {
  const old = defer();
  const { window, $ } = fixture((path) => Promise.resolve(common(path) || (path.endsWith("/old") ? old.promise : path.endsWith("/current") ? record("current", "succeeded") : [])));
  window.IRISNavigation.open("tracking"); await settle();
  window.IRISTrackingStudy.open("old"); window.IRISTrackingStudy.open("current"); await settle();
  old.reject(new Error("Abandoned saved study unavailable")); await settle();
  assert.equal($("#study-result-title").textContent, "Study current");
  assert.equal($("#study-error").hidden, true);
});

test("abandoned study polling neither blocks a new workspace visit nor releases its active request", async () => {
  const abandoned = defer(), latest = defer(); let histories = 0;
  const saved = record("active");
  const { window, $, timers, runTimer } = fixture((path) => {
    const result = common(path); if (result) return Promise.resolve(result);
    assert.ok(path.endsWith("/tracking-studies")); histories++;
    if (histories === 2) return abandoned.promise;
    if (histories === 4) return latest.promise;
    return Promise.resolve([saved]);
  });
  window.IRISNavigation.open("tracking"); await settle();
  const firstPoll = runTimer(); await settle(); assert.equal(histories, 2);
  window.IRISNavigation.open("intake"); window.IRISNavigation.open("tracking"); await settle();
  const secondPoll = runTimer(); await settle(); assert.equal(histories, 4); assert.equal(timers.size, 0);
  abandoned.resolve([record("abandoned")]); await firstPoll; await settle();
  assert.equal(timers.size, 0); assert.equal($("#study-history").children.some((option) => option.value === "abandoned"), false);
  latest.resolve([saved]); await secondPoll; await settle(); assert.equal(timers.size, 1);
});
