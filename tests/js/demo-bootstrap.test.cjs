"use strict";

const assert = require("node:assert/strict");
const { createHash, webcrypto } = require("node:crypto");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const vm = require("node:vm");

const source = readFileSync(require.resolve("../../src/iris/static/demo/bootstrap.js"), "utf8");

function boot({ href, seed, sample, fixture = { example: 1 }, storageError } = {}) {
  const bytes = JSON.stringify(fixture);
  const hash = createHash("sha256").update(bytes).digest("hex");
  const requests = [], openings = [], channels = [], events = [];
  const window = {
    location: { href },
    history: { replaceState(_state, _title, address) { window.location.href = String(address); } },
    dispatchEvent(event) { events.push(event); },
    IRISDemoAnnotations: { createState(value) { return { example: value }; } },
    IRISDemoStorage: {
      async create(options) {
        openings.push(options);
        if (storageError) throw storageError;
        return { status: { mode: "persistent" } };
      },
    },
    IRISDemoAdapter: {
      create({ load, onChange }) { return { ready: Promise.resolve().then(load), onChange }; },
    },
  };
  vm.runInNewContext(source, {
    window,
    document: { currentScript: { dataset: { ...(seed === undefined ? {} : { seed }), sample } } },
    URL, TextDecoder, crypto: webcrypto,
    CustomEvent: class {
      constructor(type, { detail } = {}) { this.type = type; this.detail = detail; }
    },
    Event: class { constructor(type) { this.type = type; } },
    BroadcastChannel: class {
      constructor(name) { this.name = name; this.messages = []; channels.push(this); }
      addEventListener(type, listener) { this[type] = listener; }
      postMessage(message) { this.messages.push(message); }
    },
    async fetch(url, options) {
      requests.push({ url: String(url), options });
      if (url.pathname.endsWith("/fixture.json")) return new Response(bytes);
      assert.ok(url.pathname.endsWith("/provenance.json"), "Only the public seed is fetched");
      return Response.json({ files: { "fixture.json": { sha256: hash } } });
    },
  });
  return { adapter: window.IRISRuntimeAdapter, window, requests, openings, channels, events, hash };
}

test("preview and versioned root bundles retain the existing browser workspace", async () => {
  for (const [href, seed] of [
    ["https://iris.example/", "/seed/"],
    ["https://iris.example/index.html", "assets/first/seed/"],
    ["https://iris.example/?project=web-demo#main", "assets/second/seed/"],
  ]) {
    const result = boot({ href, seed });
    await result.adapter.ready;
    assert.equal(result.openings[0].name, "iris-web-demo:/seed/");
    assert.equal(result.channels[0].name, result.openings[0].name);
    assert.equal(result.openings[0].seedVersion, result.hash);
    assert.equal(new URL(result.window.location.href).pathname, new URL(href).pathname);
    assert.equal(new URL(result.window.location.href).searchParams.get("project"), "web-demo");
  }
});

test("subpath bundles fetch their own assets while storage and notifications survive asset updates", async () => {
  const first = boot({ href: "https://iris.example/tools/iris/", seed: "assets/first/seed/" });
  const next = boot({ href: "https://iris.example/tools/iris/index.html?project=web-demo", seed: "assets/next/seed/" });
  const loaded = await Promise.all([first.adapter.ready, next.adapter.ready]);
  assert.equal(first.openings[0].name, "iris-web-demo:/tools/iris/seed/");
  assert.equal(next.openings[0].name, first.openings[0].name);
  assert.equal(next.openings[0].seedVersion, first.openings[0].seedVersion);
  for (const [result, version, data] of [[first, "first", loaded[0]], [next, "next", loaded[1]]]) {
    const base = `https://iris.example/tools/iris/assets/${version}/seed/`;
    assert.equal(data.baseURL, base);
    assert.deepEqual(result.requests.map(({ url }) => url), [base + "fixture.json", base + "provenance.json"]);
    for (const { options } of result.requests) {
      assert.equal(options.mode, "same-origin");
      assert.equal(options.redirect, "error");
    }
    assert.equal(result.channels[0].name, first.openings[0].name);
    result.adapter.onChange({ kind: "save" });
    assert.deepEqual(result.channels[0].messages, [{ kind: "save" }]);
    result.channels[0].message();
    assert.equal(result.events.at(-1).type, "iris:demo-change");
  }
  const other = boot({ href: "https://iris.example/other/", seed: "assets/next/seed/" });
  await other.adapter.ready;
  assert.notEqual(other.openings[0].name, first.openings[0].name);
  assert.notEqual(other.channels[0].name, first.channels[0].name);
});

test("changed seed bytes reach version validation in the same workspace without an automatic reset", async () => {
  const first = boot({ href: "https://iris.example/demo/", seed: "assets/old/seed/" });
  await first.adapter.ready;
  let resets = 0;
  const error = Object.assign(new Error("Saved demo version differs"), {
    name: "SeedMismatchError", reset() { resets++; },
  });
  const next = boot({ href: "https://iris.example/demo/", seed: "assets/new/seed/", fixture: { example: 2 }, storageError: error });
  await assert.rejects(next.adapter.ready, (value) => value === error);
  assert.equal(next.openings[0].name, first.openings[0].name);
  assert.notEqual(next.openings[0].seedVersion, first.openings[0].seedVersion);
  assert.equal(resets, 0);
  assert.equal(next.events.at(-1).type, "iris:demo-status");
  assert.equal(next.events.at(-1).detail.mode, "error");
  assert.equal(next.events.at(-1).detail.message, error.message);
});

test("a new sample opens separately while updates to that sample retain its saved workspace", async () => {
  const shared = { href: "https://iris.example/demo/", seed: "assets/old/seed/" };
  const previous = boot({ ...shared, sample: "fpv-yolox-v1" });
  const fpv = boot({ ...shared, sample: "fpv-yolox-person-v1" });
  const updated = boot({ ...shared, seed: "assets/new/seed/", sample: "fpv-yolox-person-v1" });
  await Promise.all([previous.adapter.ready, fpv.adapter.ready, updated.adapter.ready]);
  assert.notEqual(previous.openings[0].name, fpv.openings[0].name);
  assert.notEqual(previous.channels[0].name, fpv.channels[0].name);
  assert.equal(fpv.openings[0].name, "iris-web-demo:/demo/seed/:fpv-yolox-person-v1");
  assert.equal(updated.openings[0].name, fpv.openings[0].name);
  assert.equal(updated.channels[0].name, fpv.channels[0].name);
});
