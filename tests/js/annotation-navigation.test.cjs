"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const vm = require("node:vm");

const settle = () => new Promise((resolve) => setImmediate(resolve));

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
      this.dataset = {};
      this.children = [];
      this.classList = { toggle() {} };
    }
    setAttribute() {}
    setCustomValidity() {}
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    contains(item) { return this === item || this.children.some((child) => child.contains?.(item)); }
    querySelector() { return new Element(); }
    querySelectorAll(selector) {
      return this.children.flatMap((child) => [
        ...(selector === child.tagName ? [child] : []),
        ...(child.querySelectorAll?.(selector) || []),
      ]);
    }
    click() { this.dispatchEvent(new Event("click")); }
  }
  const nodes = new Map();
  const $ = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, new Element());
    return nodes.get(selector);
  };
  document.body = new Element();
  document.createTextNode = (text) => text;
  document.querySelectorAll = (selector) => selector === "#review-queue-list button"
    ? $("#review-queue-list").querySelectorAll("button") : [];
  const window = new EventTarget();
  window.document = document;
  window.IrisAnnotationTools = require("../../src/iris/static/annotation-tools.js");
  window.IRISTaxonomyTools = require("../../src/iris/static/taxonomy-tools.js");
  window.IRISPreannotationTools = require("../../src/iris/static/preannotation-tools.js");
  const frames = ["a", "b"].map((id) => ({ id, selected: true, review_status: "unannotated" }));
  const state = { sessionId: "session", frames, jobs: [] };
  const requests = [];
  vm.runInNewContext(readFileSync(require.resolve("../../src/iris/static/annotation.js"), "utf8"), {
    $, document, window, state, CustomEvent, URLSearchParams,
    Option: Element, ResizeObserver: class { observe() {} },
    clearTimeout() {}, setTimeout() {}, structuredClone,
    runtimePreference: (key) => key, irisRuntime: { mode: "local" },
    sourceFor() {}, timestamp: String, projectURL: (path) => path,
    node: (...args) => new Element(...args),
    api: (path) => new Promise((resolve, reject) => requests.push({ path, resolve, reject })),
  });
  const take = (path) => {
    const request = requests.find((item) => !item.taken && item.path === path);
    assert.ok(request, `Expected request for ${path}`);
    request.taken = true;
    return request;
  };
  const openFrame = (id) => window.dispatchEvent(new CustomEvent("iris:annotation-open-frame", {
    detail: { frame_id: id },
  }));
  return {
    $, requests, take, openFrame,
    row: (id) => $("#review-queue-list").querySelectorAll("button").find((item) => item.dataset.frameId === id),
    async open() {
      window.dispatchEvent(new CustomEvent("iris:workspace", { detail: { name: "annotation" } }));
      $("#review-queue-refresh").click();
      take("/api/sessions/session/comparisons").resolve([]);
      await settle();
      take("/api/sessions/session/review-queue?confidence_threshold=0.5&iou_threshold=0.5").resolve({
        frames, counts: { total: frames.length, validated: 0 },
      });
      await settle();
      return take("/api/frames/a/annotation");
    },
  };
}

test("a failed annotation can be retried by selecting the same review queue frame", async () => {
  const env = environment();
  const first = await env.open();
  first.reject(new Error("Connection lost"));
  await settle();
  assert.equal(env.$("#annotation-error").textContent, "Connection lost");
  assert.equal(env.row("a").disabled, false);

  env.row("a").click();
  const retry = env.take("/api/frames/a/annotation");
  assert.equal(env.$("#annotation-error").hidden, true);
  env.row("a").click();
  assert.equal(env.requests.filter((item) => item.path === retry.path).length, 2);
  retry.reject(new Error("Still unavailable"));
  await settle();
  assert.equal(env.$("#annotation-error").textContent, "Still unavailable");
});

test("frame handoffs retry failed annotations without duplicating an in-flight load", async () => {
  const env = environment();
  const first = await env.open();
  env.openFrame("a");
  assert.equal(env.requests.filter((item) => item.path === first.path).length, 1);
  first.reject(new Error("Connection lost"));
  await settle();

  env.openFrame("a");
  const retry = env.take("/api/frames/a/annotation");
  env.openFrame("a");
  assert.equal(env.requests.filter((item) => item.path === first.path).length, 2);
  retry.reject(new Error("Still unavailable"));
  await settle();
  assert.equal(env.$("#annotation-error").textContent, "Still unavailable");
});

test("a late failure from another frame does not block retrying the current frame", async () => {
  const env = environment();
  const first = await env.open();
  env.openFrame("b");
  const second = env.take("/api/frames/b/annotation");
  first.reject(new Error("Old failure"));
  await settle();
  assert.equal(env.$("#annotation-error").hidden, true);
  second.reject(new Error("Current failure"));
  await settle();
  assert.equal(env.$("#annotation-error").textContent, "Current failure");

  env.row("b").click();
  env.take("/api/frames/b/annotation").reject(new Error("Retry failure"));
  await settle();
  assert.equal(env.$("#annotation-error").textContent, "Retry failure");
});
