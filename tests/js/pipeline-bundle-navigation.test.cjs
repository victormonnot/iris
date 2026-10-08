"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), vm = require("node:vm");
const tools = require("../../src/iris/static/pipeline-bundle-tools.js");
const settle = () => new Promise(setImmediate);
const defer = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const descriptor = (id) => ({ kind: "comparison", job_id: id, sequence_id: `sequence-${id}`, profile_sha256: "a".repeat(64) });
const policy = { min_score: .3, min_iou: .05, max_center_distance: 1, max_area_ratio: 3, max_lost_seconds: 1, max_lost_updates: 15, recovery_confirmation_updates: 2 };
function source(id = "source") { return { source: descriptor(id), name: id, sequence_name: `Sequence ${id}`, detector: { architecture: "ssdlite320_mobilenet_v3_large", device: "cpu", classes: [{ id: 1, name: "person" }], min_score: .001 }, profile: { algorithm: "bytetrack", class_ids: [1], low_threshold: .1, high_threshold: .5, buffer_updates: 30, gmc_method: "none" }, selections: [{ id: `policy-${id}`, name: `Policy ${id}`, policy }] }; }
function request(id) { return { name: `Bundle ${id}`, source: descriptor(id), selection_id: null, target_device: "cpu" }; }
function manifest(id) { const row = source(id); return { format: "iris-pipeline-bundle-v1", detector: { config: row.detector, checkpoint: { size: 1024, sha256: "b".repeat(64) }, target_device: "cpu", output_mapping: { entries: [{ internal_index: 1, output_id: 1, label: "person" }] } }, tracker: { profile: row.profile }, selection: null }; }
function record(id, status = "succeeded") { return { id, name: `Bundle ${id}`, job: { id, status, progress: 1, params: { request: request(id) } }, bundle: status === "succeeded" ? { manifest: manifest(id), archive_bytes: 4096 } : null }; }
function preview(id) { return { request: request(id), manifest: manifest(id), fingerprint: "c".repeat(64), limitations: ["Experimental; pipeline not run."] }; }
function fixture(api, { query = "" } = {}) {
  class Element extends EventTarget {
    constructor(tag = "div") { super(); this.tagName = tag; this.value = ""; this.children = []; this.dataset = {}; this.attributes = {}; this.hidden = false; this.disabled = false; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    querySelectorAll(selector) { return this.children.filter((child) => child instanceof Element).flatMap((child) => [...(selector.split(",").includes(child.tagName) ? [child] : []), ...child.querySelectorAll(selector)]); }
    setAttribute(key, value) { this.attributes[key] = value; }
    scrollIntoView() { this.scrolled = true; }
  }
  class Option extends Element { constructor(label, value) { super("option"); this.textContent = label; this.value = value; } }
  const elements = new Map(), $ = (key) => { if (!elements.has(key)) elements.set(key, new Element()); return elements.get(key); };
  $("#bundle-name").value = "Bundle source"; $("#bundle-target").value = "cpu";
  const window = new EventTarget(); window.location = { href: `http://localhost/?project=default${query}` }; window.IRISPipelineBundleTools = tools;
  window.IRISNavigation = { open(name) { window.dispatchEvent(new CustomEvent("iris:workspace", { detail: { name } })); return true; } };
  const timers = new Map(); let timerID = 0;
  const node = (tag, className, text) => { const result = new Element(tag); result.className = className; result.textContent = text; return result; };
  vm.runInNewContext(fs.readFileSync(require.resolve("../../src/iris/static/pipeline-bundle.js"), "utf8"), {
    window, $, node, api, URL, Option, CustomEvent, state: { projectId: "default" }, projectURL: (path) => `${path}?project=default`,
    setTimeout(callback) { const id = ++timerID; timers.set(id, callback); return id; }, clearTimeout(id) { timers.delete(id); },
  });
  return { window, $, timers, selectSource(id) { $("#bundle-source").value = tools.sourceKey(descriptor(id)); $("#bundle-source").dispatchEvent(new Event("change")); }, click(id) { $(`#bundle-${id}`).dispatchEvent(new Event("click")); }, runTimer() { assert.equal(timers.size, 1); const [id, callback] = [...timers][0]; timers.delete(id); return callback(); } };
}
function common(path, rows = ["source"], history = []) {
  if (path.endsWith("pipeline-bundle-sources")) return { sources: rows.map(source) };
  if (path.endsWith("pipeline-bundle-status")) return {};
  if (path.endsWith("pipeline-bundles")) return history;
  return null;
}
test("deep-linked bundle opens via reads only and exposes project-scoped downloads", async () => {
  const calls = [], f = fixture(async (path, options) => { calls.push({ path, options }); return common(path, ["saved"], [record("saved")]) || record("saved"); }, { query: "&pipeline_bundle=saved" });
  f.window.dispatchEvent(new Event("iris:project-initialized")); await settle();
  assert.equal(f.$("#bundle-result").hidden, false); assert.equal(f.$("#bundle-result-title").textContent, "Bundle saved");
  assert.match(f.$("#bundle-result-scope").textContent, /Historical v1 package.*inspection only/);
  assert.equal(f.$("#bundle-runtime-help").hidden, true);
  assert.equal(f.$("#bundle-download").href, "/api/temporal/pipeline-bundles/saved/download?project=default");
  assert.equal(f.$("#bundle-download-manifest").href, "/api/temporal/pipeline-bundles/saved/manifest?project=default");
  assert.ok(calls.every(({ options }) => !options?.method)); assert.equal(f.$("#bundle-run").disabled, true);
});
test("only preview then explicit package creates a job; edits invalidate the preview", async () => {
  const calls = [], f = fixture(async (path, options) => { calls.push({ path, options }); if (path.endsWith("/preview")) return preview("source"); if (path.endsWith("pipeline-bundles") && options?.method === "POST") return record("new", "queued"); return common(path); });
  f.window.IRISNavigation.open("tracking"); await settle(); f.selectSource("source");
  assert.equal(f.$("#bundle-preview").disabled, false); assert.equal(f.$("#bundle-run").disabled, true);
  f.click("preview"); await settle(); assert.equal(f.$("#bundle-run").disabled, false);
  f.$("#bundle-target").value = "cuda"; f.$("#bundle-target").dispatchEvent(new Event("change")); assert.equal(f.$("#bundle-run").disabled, true);
  f.click("preview"); await settle(); f.click("run"); await settle();
  const posts = calls.filter(({ path, options }) => path.endsWith("pipeline-bundles") && options?.method === "POST");
  assert.equal(posts.length, 1); assert.equal(JSON.parse(posts[0].options.body).expected_fingerprint, "c".repeat(64));
  assert.equal(f.$("#bundle-job").hidden, false); assert.equal(f.$("#bundle-result").hidden, true);
});
test("changing source clears a previously selected policy from a different profile", async () => {
  const f = fixture(async (path) => common(path, ["first", "second"])); f.window.IRISNavigation.open("tracking"); await settle(); f.selectSource("first");
  f.$("#bundle-selection").value = "policy-first"; f.$("#bundle-selection").dispatchEvent(new Event("change")); assert.match(f.$("#bundle-policy-summary").textContent, /Experimental guarded geometry/);
  f.selectSource("second"); assert.equal(f.$("#bundle-selection").value, ""); assert.match(f.$("#bundle-policy-summary").textContent, /No selected-object policy/);
});
test("late saved-record responses and errors cannot replace a newer Tasks open", async () => {
  const old = defer(), f = fixture(async (path) => common(path, ["current"]) || (path.endsWith("/old") ? old.promise : record("current")));
  f.window.IRISNavigation.open("tracking"); await settle(); f.window.IRISPipelineBundle.open("old"); f.window.dispatchEvent(new CustomEvent("iris:pipeline-bundle-open", { detail: { bundle_id: "current" } })); await settle();
  old.reject(new Error("Old response")); await settle(); assert.equal(f.$("#bundle-result-title").textContent, "Bundle current"); assert.equal(f.$("#bundle-error").hidden, true);
});
test("pending preview is discarded when a saved bundle is opened from Tasks", async () => {
  const pending = defer(), f = fixture(async (path) => path.endsWith("/preview") ? pending.promise : common(path) || record("saved"));
  f.window.IRISNavigation.open("tracking"); await settle(); f.selectSource("source"); f.click("preview"); await settle();
  f.window.IRISPipelineBundle.open("saved"); await settle(); pending.resolve(preview("source")); await settle();
  assert.equal(f.$("#bundle-result-title").textContent, "Bundle saved"); assert.equal(f.$("#bundle-preview-result").hidden, true); assert.equal(f.$("#bundle-run").disabled, true);
});
test("project and workspace generation discard in-flight catalogue and record data", async () => {
  const pending = defer(); let ready = false;
  const f = fixture(async (path) => path.endsWith("pipeline-bundle-sources") && !ready ? pending.promise : common(path, ["new"]));
  f.window.IRISNavigation.open("tracking"); ready = true; f.window.dispatchEvent(new Event("iris:project-initialized")); await settle(); pending.resolve({ sources: [source("old")] }); await settle();
  assert.equal(f.$("#bundle-source").children[1].textContent.includes("new"), true); assert.equal(f.$("#bundle-source").children.some((item) => item.textContent.includes("old")), false);
});
test("background completion becomes downloadable on return without requeue or auto-scroll", async () => {
  let finished = false; const calls = [], f = fixture(async (path, options) => { calls.push({ path, options }); return common(path, ["active"], [record("active", finished ? "succeeded" : "running")]) || record("active", finished ? "succeeded" : "running"); });
  f.window.IRISNavigation.open("tracking"); await settle(); f.window.IRISPipelineBundle.open("active"); await settle();
  f.window.IRISNavigation.open("intake"); finished = true; f.window.IRISNavigation.open("tracking"); await settle();
  assert.equal(f.$("#bundle-result").hidden, false); assert.notEqual(f.$("#bundle-result").scrolled, true); assert.ok(calls.every(({ options }) => !options?.method));
});
test("explicit cancellation uses only the active bundle job", async () => {
  const calls = [], f = fixture(async (path, options) => { calls.push({ path, options }); return common(path, ["active"], [record("active", "running")]) || record("active", "running"); });
  f.window.IRISNavigation.open("tracking"); await settle(); f.window.IRISPipelineBundle.open("active"); await settle(); f.click("cancel"); await settle();
  assert.equal(calls.filter(({ path, options }) => path === "/api/jobs/active/cancel" && options.method === "POST").length, 1);
  assert.equal(f.$("#bundle-cancel").disabled, true); assert.equal(f.$("#bundle-result").hidden, true);
});

test("a cold deep link restores source and matching policy when the catalogue arrives last", async () => {
  const catalogue = defer(), saved = record("saved"); saved.job.params.request.selection_id = "policy-saved";
  const f = fixture(async (path) => path.endsWith("pipeline-bundle-sources") ? catalogue.promise : common(path, ["saved"], [saved]) || saved, { query: "&pipeline_bundle=saved" });
  f.window.dispatchEvent(new Event("iris:project-initialized")); await settle(); assert.equal(f.$("#bundle-result").hidden, false);
  catalogue.resolve({ sources: [source("saved")] }); await settle();
  assert.equal(f.$("#bundle-source").value, tools.sourceKey(descriptor("saved"))); assert.equal(f.$("#bundle-selection").value, "policy-saved");
  assert.equal(f.$("#bundle-preview").disabled, false); assert.match(f.$("#bundle-policy-summary").textContent, /Experimental guarded geometry/);
});
test("a pending catalogue refresh restores the latest opened bundle instead of its captured predecessor", async () => {
  let delayed = false; const catalogue = defer();
  const f = fixture(async (path) => path.endsWith("pipeline-bundle-sources") && delayed ? catalogue.promise : common(path, ["old", "current"], [record("old"), record("current")]) || record(path.split("/").at(-1)));
  f.window.IRISNavigation.open("tracking"); await settle(); f.window.IRISPipelineBundle.open("old"); await settle();
  delayed = true; f.click("refresh"); f.window.IRISPipelineBundle.open("current"); await settle(); catalogue.resolve({ sources: [source("old"), source("current")] }); await settle();
  assert.equal(f.$("#bundle-source").value, tools.sourceKey(descriptor("current"))); assert.equal(f.$("#bundle-result-title").textContent, "Bundle current"); assert.equal(f.$("#bundle-name").value, "Bundle current");
});

test("v2 saved bundles expose integration help without claiming saved runtime qualification", async () => {
  const saved = record("v2"), calls = []; saved.bundle.manifest.format = "iris-pipeline-bundle-v2";
  const f = fixture(async (path, options) => { calls.push({ path, options }); return common(path, ["v2"], [saved]) || saved; }, { query: "&pipeline_bundle=v2" });
  f.window.dispatchEvent(new Event("iris:project-initialized")); await settle();
  assert.equal(f.$("#bundle-runtime-help").hidden, false);
  assert.match(f.$("#bundle-result-scope").textContent, /standalone runtime included.*no attached execution or quality results/);
  assert.ok(calls.every(({ options }) => !options?.method));
});
