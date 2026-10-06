"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const vm = require("node:vm");

const settle = () => new Promise((resolve) => setImmediate(resolve));
const report = (id) => ({
  id, title: `Report ${id}`, objective: "", conclusion: "", revision: 1,
  created_at: "2026-01-01T00:00:00Z", images: [],
  snapshot: {
    dataset: { name: "Reviewed release" }, lanes: [],
    evaluation: { id: "evaluation", name: "Saved evaluation", split: "val", config: {} },
  },
});

function environment() {
  const document = new EventTarget();
  document.activeElement = null;
  class Element extends EventTarget {
    constructor(tag = "div", className = "", text = "") {
      super();
      this.tagName = tag;
      this.className = className;
      this.textContent = text;
      this.value = "";
      this.hidden = false;
      this.open = false;
      this.dataset = {};
      this.children = [];
      this.attributes = new Map();
    }
    setAttribute(name, value) { this.attributes.set(name, value); }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    querySelectorAll() { return []; }
    closest(selector) { return selector === "[data-experiment-id]" && this.dataset.experimentId ? this : null; }
    focus() { document.activeElement = this; }
    click() { this.dispatchEvent(new Event("click")); }
    reportValidity() { return true; }
    reset() {}
    showModal() { this.open = true; }
    close() {
      this.open = false;
      queueMicrotask(() => this.dispatchEvent(new Event("close")));
    }
  }
  const nodes = new Map();
  const $ = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, new Element());
    return nodes.get(selector);
  };
  const window = new EventTarget();
  window.IRISDatasetTools = { taxonomyOf: () => ({ classes: [] }) };
  window.IRISTaxonomyTools = {};
  const requests = [];
  vm.runInNewContext(readFileSync(require.resolve("../../src/iris/static/experiments.js"), "utf8"), {
    $, document, window, URL, clearTimeout, Option: Element, notify() {},
    node: (...args) => new Element(...args),
    api: (path, { method = "GET" } = {}) => new Promise((resolve, reject) => {
      requests.push({ path, method, resolve, reject, taken: false });
    }),
  });
  const field = (name) => $(`#experiments-${name}`);
  const take = (path, method = "GET") => {
    const request = requests.find((item) => !item.taken && item.path === path && item.method === method);
    assert.ok(request, `Expected ${method} ${path}`);
    request.taken = true;
    return request;
  };
  const workspace = (name) => window.dispatchEvent(new CustomEvent("iris:workspace", { detail: { name } }));
  const row = (id) => field("list").children.find((item) => item.dataset.experimentId === id);
  return {
    field, take, requests, workspace, row,
    activeId: () => field("list").children.find((item) => item.attributes.get("aria-current") === "true")?.dataset.experimentId,
    async open(ids = ["a", "b"]) {
      workspace("experiments");
      take("/api/experiments").resolve(ids.map(report));
      await settle();
      take(`/api/experiments/${ids[0]}`).resolve(report(ids[0]));
      await settle();
      assert.equal(field("detail-status").textContent, "");
      assert.equal(field("report-title").textContent, report(ids[0]).title);
    },
    async create(id) {
      field("new").click();
      take("/api/evaluations").resolve([{ id: "evaluation", job: { status: "succeeded" } }]);
      await settle();
      field("evaluation").value = "evaluation";
      field("evaluation").dispatchEvent(new Event("change"));
      take("/api/evaluations/evaluation/experiment-preview").resolve({
        snapshot: report(id).snapshot, available_examples: [], available_measurements: [], source_fingerprint: "saved-evidence",
      });
      await settle();
      field("title").value = report(id).title;
      field("compose-form").dispatchEvent(new Event("submit", { cancelable: true }));
      take("/api/experiments", "POST").resolve(report(id));
      await settle();
      return take("/api/experiments");
    },
  };
}

test("a selection during tab-return refresh survives the delayed list and keeps its pending detail", async () => {
  const env = environment();
  await env.open();
  env.workspace("training");
  env.workspace("experiments");
  const refresh = env.take("/api/experiments");
  env.row("b").click();
  const detail = env.take("/api/experiments/b");
  refresh.resolve([report("a"), report("b")]);
  await settle();
  assert.equal(env.activeId(), "b");
  assert.equal(env.requests.filter((item) => item.path === "/api/experiments/b").length, 1);
  assert.equal(env.requests.filter((item) => item.path === "/api/experiments/a").length, 1);
  detail.resolve(report("b"));
  await settle();
  assert.equal(env.field("report-title").textContent, "Report b");
  assert.equal(env.field("detail-status").textContent, "");
  assert.equal(env.field("refresh").disabled, false);
});

test("a report opened before list completion stays visible without restarting its detail request", async () => {
  const env = environment();
  await env.open();
  env.field("refresh").click();
  const refresh = env.take("/api/experiments");
  env.row("b").click();
  env.take("/api/experiments/b").resolve(report("b"));
  await settle();
  refresh.resolve([report("a"), report("b")]);
  await settle();
  assert.equal(env.activeId(), "b");
  assert.equal(env.field("detail").hidden, false);
  assert.equal(env.field("report-title").textContent, "Report b");
  assert.equal(env.requests.filter((item) => item.path === "/api/experiments/b").length, 1);
});

test("a newly saved report is preferred when no newer selection is made", async () => {
  const env = environment();
  await env.open();
  const refresh = await env.create("created");
  refresh.resolve([report("a"), report("created"), report("b")]);
  await settle();
  env.take("/api/experiments/created").resolve(report("created"));
  await settle();
  assert.equal(env.activeId(), "created");
  assert.equal(env.field("report-title").textContent, "Report created");
});

test("clicking even the previously active report overrides a save's delayed preferred selection", async () => {
  const env = environment();
  await env.open();
  const refresh = await env.create("created");
  env.row("b").click();
  const staleDetail = env.take("/api/experiments/b");
  env.row("a").click();
  env.take("/api/experiments/a").resolve(report("a"));
  await settle();
  refresh.resolve([report("created"), report("a"), report("b")]);
  await settle();
  staleDetail.resolve(report("b"));
  await settle();
  assert.equal(env.activeId(), "a");
  assert.equal(env.field("report-title").textContent, "Report a");
  assert.ok(!env.requests.some((item) => item.path === "/api/experiments/created"));
});

test("saving notes refreshes the edited report when the list has a different first row", async () => {
  const env = environment();
  await env.open();
  env.field("edit").click();
  env.field("edit-form").dispatchEvent(new Event("submit", { cancelable: true }));
  env.take("/api/experiments/a", "PATCH").resolve(report("a"));
  await settle();
  env.take("/api/experiments").resolve([report("b"), report("a")]);
  await settle();
  env.take("/api/experiments/a").resolve({ ...report("a"), revision: 2 });
  await settle();
  assert.equal(env.activeId(), "a");
  assert.equal(env.field("revision").textContent, "Notes revision 2");
});

test("a selected report missing from the refreshed list falls back and ignores its stale detail", async () => {
  const env = environment();
  await env.open();
  env.field("refresh").click();
  const refresh = env.take("/api/experiments");
  env.row("b").click();
  const staleDetail = env.take("/api/experiments/b");
  refresh.resolve([report("c")]);
  await settle();
  env.take("/api/experiments/c").resolve(report("c"));
  await settle();
  staleDetail.resolve(report("b"));
  await settle();
  assert.equal(env.activeId(), "c");
  assert.equal(env.field("report-title").textContent, "Report c");
});

test("an empty refreshed list invalidates a detail still in flight", async () => {
  const env = environment();
  await env.open();
  env.field("refresh").click();
  const refresh = env.take("/api/experiments");
  env.row("b").click();
  const staleDetail = env.take("/api/experiments/b");
  refresh.resolve([]);
  await settle();
  staleDetail.resolve(report("b"));
  await settle();
  assert.equal(env.activeId(), undefined);
  assert.equal(env.field("detail").hidden, true);
  assert.equal(env.field("empty").hidden, false);
});

test("a response from an earlier workspace visit cannot replace the newer list or selection", async () => {
  const env = environment();
  await env.open();
  env.field("refresh").click();
  const staleRefresh = env.take("/api/experiments");
  env.workspace("training");
  env.workspace("experiments");
  const refresh = env.take("/api/experiments");
  env.row("b").click();
  env.take("/api/experiments/b").resolve(report("b"));
  await settle();
  refresh.resolve([report("b")]);
  await settle();
  staleRefresh.resolve([report("a")]);
  await settle();
  assert.equal(env.activeId(), "b");
  assert.equal(env.field("report-title").textContent, "Report b");
  assert.equal(env.field("list").children.length, 1);
});
