"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const tools = require("../../src/iris/static/trackingcost-tools.js");

test("an abandoned cost poll cannot block a new context or release its in-flight poll", async () => {
  class Element extends EventTarget {
    constructor(value = "") { super(); this.value = value; this.children = []; this.files = []; }
    get options() { return this.children; }
    replaceChildren(...children) { this.children = children; }
    append(...children) { this.children.push(...children); }
    setAttribute() {}
    removeAttribute() {}
    scrollIntoView() {}
  }
  class Option { constructor(text, value) { this.textContent = text; this.value = value; } }
  const elements = new Map();
  const $ = (selector) => { if (!elements.has(selector)) elements.set(selector, new Element()); return elements.get(selector); };
  for (const [name, value] of Object.entries({ name: "Measurement", lane: "0", device: "cpu", repeats: "2", policy: "offline_all", cadence: "30", "history-lane": "all" })) $(`#cost-${name}`).value = value;
  $("#cost-lane").append(new Option("Lane 1", "0"), new Option("Lane 2", "1"));
  $("#cost-history-lane").append(new Option("Both lanes", "all"), new Option("Lane 1", "0"), new Option("Lane 2", "1"));
  const window = new EventTarget();
  window.location = { href: "http://localhost/?project=default" };
  window.IRISTrackingCostTools = tools;
  const timers = new Map(); let timerID = 0;
  const defer = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
  const oldPoll = defer(), newPoll = defer(), calls = { a: 0, b: 0 };
  const record = (id) => ({ id: `job-${id}`, name: `Run ${id}`, comparison_id: id, sequence_id: `sequence-${id}`, report: null, origin: "local_worker", job: { status: "running", params: { config: { lane_index: 0 } } } });
  const api = (path) => {
    const id = path.match(/tracking-comparisons\/([ab])\/cost-runs$/)?.[1];
    assert.ok(id, `Unexpected API operation: ${path}`);
    calls[id]++;
    if (calls[id] === 1) return Promise.resolve([record(id)]);
    return id === "a" ? oldPoll.promise : newPoll.promise;
  };
  vm.runInNewContext(fs.readFileSync(require.resolve("../../src/iris/static/trackingcost.js"), "utf8"), {
    window, $, api, URL, Option, CustomEvent, state: { projectId: "default" },
    node: () => new Element(),
    setTimeout: (callback) => { const id = ++timerID; timers.set(id, callback); return id; },
    clearTimeout: (id) => timers.delete(id),
  });
  const settle = () => new Promise(setImmediate);
  const enter = (id) => window.dispatchEvent(new CustomEvent("iris:tracking-quality-context", { detail: {
    sequence: { id: `sequence-${id}`, name: `Sequence ${id}`, manifest: { frames: [{ frame_index: 0 }] } },
    comparison: { id, sequence_id: `sequence-${id}`, name: `Comparison ${id}` },
    report: { lanes: [{ name: "ByteTrack" }, { name: "BoT-SORT" }] },
  } }));
  const runTimer = () => { assert.equal(timers.size, 1); const [id, callback] = [...timers.entries()][0]; timers.delete(id); return callback(); };
  window.dispatchEvent(new CustomEvent("iris:workspace", { detail: { name: "tracking" } }));
  enter("a"); await settle();
  const abandoned = runTimer(); await settle();
  assert.equal(calls.a, 2, "old context poll is now waiting indefinitely");
  enter("b"); await settle();
  const current = runTimer(); await settle();
  assert.equal(calls.b, 2, "new context starts polling without waiting for the old request");
  assert.equal(timers.size, 0, "new context has one request in flight, no retry timer");
  oldPoll.resolve([record("a")]); await abandoned; await settle();
  assert.equal(timers.size, 0, "late old completion must not release or reschedule the active new poll");
  assert.match($("#cost-context").textContent, /Comparison b/);
  assert.equal($("#cost-history").children.some((option) => option.value === "job-a"), false);
  newPoll.resolve([record("b")]); await current; await settle();
  assert.equal(timers.size, 1, "only the current context schedules its next poll after completion");
  assert.equal($("#cost-history").children.some((option) => option.value === "job-b"), true);
});
