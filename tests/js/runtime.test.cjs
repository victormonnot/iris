"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { create: createScope } = require("../../src/iris/static/project-scope.js");
const { create, createLocalAdapter } = require("../../src/iris/static/runtime.js");

function scope() {
  return createScope("http://localhost:8000/?project=inventory");
}

test("a distribution requiring an adapter cannot fall back to local transport after a script failure", (t) => {
  t.mock.method(globalThis, "fetch", () => assert.fail("No local API request is allowed"));
  for (const adapter of [undefined, null, createLocalAdapter()])
    assert.throws(() => create({ projectScope: scope(), adapter, requiredMode: "demo" }), /required Iris runtime/);
  const adapter = { mode: "demo", request() {}, resourceURL() {}, upload() {} };
  assert.equal(create({ projectScope: scope(), adapter, requiredMode: "demo" }).mode, "demo");
  assert.equal(create({ projectScope: scope() }).mode, "local");
});

function xhrHarness() {
  const requests = [];
  class Request extends EventTarget {
    constructor() {
      super();
      this.upload = new EventTarget();
      this.status = 0;
      this.statusText = "";
      this.responseText = "";
      this.headers = new Headers();
      requests.push(this);
    }
    open(method, path) { this.method = method; this.path = path; }
    send(body) { this.body = body; this.sent = true; }
    setRequestHeader(name, value) { this.headers.append(name, value); }
    abort() { this.aborted = true; this.dispatchEvent(new Event("abort")); }
    getAllResponseHeaders() { return this.responseHeaders || ""; }
    complete(status = 200, body = "", headers = "") {
      this.status = status;
      this.statusText = status === 200 ? "OK" : "";
      this.responseText = body;
      this.response = body;
      this.responseHeaders = headers;
      this.dispatchEvent(new Event("load"));
    }
  }
  const runtime = create({ projectScope: scope(), adapter: createLocalAdapter({ XMLHttpRequest: Request }) });
  return { runtime, requests, Request };
}

test("local requests scope resources without changing fetch options or responses", async () => {
  const calls = [];
  const response = new Response("archive", { status: 201, headers: { "Content-Type": "application/zip" } });
  const adapter = createLocalAdapter({ fetch: (...args) => { calls.push(args); return Promise.resolve(response); } });
  const runtime = create({ projectScope: scope(), adapter });
  const options = {
    method: "POST", headers: new Headers({ Authorization: "Bearer example" }),
    body: "payload", signal: new AbortController().signal, mode: "same-origin", redirect: "error",
  };
  assert.equal(await runtime.request("/api/experiments/a/export?include_images=true&project_id=wrong", options), response);
  assert.deepEqual(calls, [["/api/experiments/a/export?include_images=true&project_id=inventory", options]]);
  assert.equal(calls[0][1], options);
  assert.equal(runtime.mode, "local");
  assert.deepEqual(runtime.capabilities, {});
  assert.ok(Object.isFrozen(runtime));
  assert.ok(Object.isFrozen(runtime.capabilities));
  assert.equal(runtime.resourceURL("/api/frames/a/image?version=2"), "/api/frames/a/image?version=2&project_id=inventory");
  assert.equal(runtime.resourceURL("/api/workspace/operations/a/archive"), "/api/workspace/operations/a/archive");
});

test("request leaves HTTP failures, network errors and aborts for the caller", async () => {
  const response = new Response('{"detail":"Invalid source"}', { status: 422 });
  const failure = new TypeError("Disconnected");
  const abort = new DOMException("Cancelled", "AbortError");
  for (const outcome of [response, failure, abort]) {
    const adapter = createLocalAdapter({ fetch: async () => {
      if (outcome instanceof Error) throw outcome;
      return outcome;
    } });
    const runtime = create({ projectScope: scope(), adapter });
    if (outcome instanceof Error) await assert.rejects(runtime.request("/api/jobs"), (error) => error === outcome);
    else assert.equal(await runtime.request("/api/jobs"), outcome);
  }
});

test("custom adapters own requests, assets and uploads without browser networking", async (t) => {
  t.mock.method(globalThis, "fetch", () => { throw new Error("Unexpected network request"); });
  const previousRequest = Object.getOwnPropertyDescriptor(globalThis, "XMLHttpRequest");
  globalThis.XMLHttpRequest = class { constructor() { throw new Error("Unexpected upload"); } };
  t.after(() => {
    if (previousRequest) Object.defineProperty(globalThis, "XMLHttpRequest", previousRequest);
    else delete globalThis.XMLHttpRequest;
  });
  const calls = [];
  const response = new Response('{"name":"Example"}', { headers: { "Content-Type": "application/json" } });
  const operation = { promise: Promise.resolve(response), abort() {} };
  const adapter = {
    mode: "demo", capabilities: { annotation: true, training: false },
    request(path, options) { calls.push(["request", path, options]); return Promise.resolve(response); },
    resourceURL(path) { calls.push(["resource", path]); return "blob:example"; },
    upload(path, options) { calls.push(["upload", path, options]); return operation; },
  };
  const runtime = create({ projectScope: scope(), adapter });
  const options = { method: "PATCH", body: "{}" };
  assert.equal(await runtime.request("/api/frames/a/annotations", options), response);
  assert.equal(runtime.resourceURL("/api/frames/a/image"), "blob:example");
  assert.equal(runtime.upload("/api/sessions/a/assets", options), operation);
  assert.deepEqual(calls, [
    ["request", "/api/frames/a/annotations?project_id=inventory", options],
    ["resource", "/api/frames/a/image?project_id=inventory"],
    ["upload", "/api/sessions/a/assets?project_id=inventory", options],
  ]);
  adapter.capabilities.training = true;
  assert.deepEqual(runtime.capabilities, { annotation: true, training: false });
  assert.equal(runtime.mode, "demo");
});

test("every transport entry point rejects external and non-API URLs before delegation", () => {
  const unexpected = () => assert.fail("Invalid resource reached adapter");
  const runtime = create({ projectScope: scope(), adapter: {
    mode: "demo", request: unexpected, resourceURL: unexpected, upload: unexpected,
  } });
  for (const path of ["https://example.org/api/jobs", "//example.org/api/jobs", "/static/image.jpg"])
    for (const name of ["request", "resourceURL", "upload"])
      assert.throws(() => runtime[name](path), /local IRIS server/);
});

test("missing or partial adapters fail before any local fallback can occur", () => {
  assert.throws(() => create(), /project scope/);
  for (const adapter of [null, {}, { mode: "demo" }, { mode: "", request() {}, resourceURL() {}, upload() {} }])
    assert.throws(() => create({ projectScope: scope(), adapter }), /adapter must define/);
  const complete = { mode: "demo", request() {}, resourceURL() {}, upload() {} };
  for (const name of ["request", "resourceURL", "upload"])
    assert.throws(() => create({ projectScope: scope(), adapter: { ...complete, [name]: undefined } }), /adapter must define/);
});

test("local upload preserves multipart body, scope, progress, HTTP status and headers", async () => {
  const { runtime, requests } = xhrHarness();
  const body = new FormData();
  body.append("file", new Blob(["contents"]), "source.txt");
  const progress = [];
  let uploaded = 0;
  const operation = runtime.upload("/api/sessions/a/assets", {
    body, headers: { "X-Request-ID": "upload-1" },
    onProgress: (event) => progress.push(event), onUploaded: () => uploaded++,
  });
  assert.ok(operation.promise instanceof Promise);
  assert.equal(typeof operation.abort, "function");
  const request = requests[0];
  assert.equal(request.path, "/api/sessions/a/assets?project_id=inventory");
  assert.equal(request.method, "POST");
  assert.equal(request.body, body);
  assert.equal(request.headers.has("Content-Type"), false);
  assert.equal(request.headers.get("X-Request-ID"), "upload-1");
  for (const lengthComputable of [true, false]) {
    const event = new Event("progress");
    Object.assign(event, { loaded: 4, total: 8, lengthComputable });
    request.upload.dispatchEvent(event);
  }
  request.upload.dispatchEvent(new Event("load"));
  assert.deepEqual(progress, [
    { loaded: 4, total: 8, lengthComputable: true },
    { loaded: 4, total: 8, lengthComputable: false },
  ]);
  assert.equal(uploaded, 1);
  request.complete(409, '{"detail":"Already exists"}', "Content-Type: application/json\r\nX-Value: one:two\r\n");
  const response = await operation.promise;
  assert.equal(response.status, 409);
  assert.equal(response.ok, false);
  assert.equal(response.headers.get("content-type"), "application/json");
  assert.equal(response.headers.get("x-value"), "one:two");
  assert.deepEqual(await response.json(), { detail: "Already exists" });
  operation.abort();
  assert.equal(request.aborted, undefined);
});

test("uploads support binary and empty responses", async () => {
  const { runtime, requests } = xhrHarness();
  const binary = runtime.upload("/api/workspace/restore", { responseType: "arraybuffer" });
  requests[0].complete(200, new Uint8Array([0, 255, 13]).buffer, "Content-Type: application/octet-stream\r\n");
  assert.deepEqual(new Uint8Array(await (await binary.promise).arrayBuffer()), new Uint8Array([0, 255, 13]));
  for (const status of [204, 205, 304]) {
    const empty = runtime.upload("/api/workspace/restore");
    requests.at(-1).complete(status);
    const response = await empty.promise;
    assert.equal(response.status, status);
    assert.equal(await response.text(), "");
  }
});

test("uploads reject transport failures without interpreting response JSON", async () => {
  for (const [event, name] of [["error", "TypeError"], ["timeout", "TimeoutError"], ["abort", "AbortError"]]) {
    const { runtime, requests } = xhrHarness();
    const operation = runtime.upload("/api/workspace/restore");
    const rejected = assert.rejects(operation.promise, { name });
    requests[0].dispatchEvent(new Event(event));
    await rejected;
  }
  const { runtime, requests } = xhrHarness();
  const operation = runtime.upload("/api/workspace/restore");
  requests[0].complete(0);
  await assert.rejects(operation.promise, { name: "TypeError" });
});

test("uploads can be aborted explicitly or through a signal, with no late progress", async () => {
  for (const viaSignal of [false, true]) {
    const { runtime, requests } = xhrHarness();
    const controller = new AbortController();
    let progress = 0;
    const operation = runtime.upload("/api/workspace/restore", {
      signal: controller.signal, onProgress: () => progress++, onUploaded: () => progress++,
    });
    const rejected = assert.rejects(operation.promise, { name: "AbortError" });
    if (viaSignal) controller.abort(); else operation.abort();
    await rejected;
    assert.equal(requests[0].aborted, true);
    requests[0].upload.dispatchEvent(new Event("progress"));
    requests[0].upload.dispatchEvent(new Event("load"));
    requests[0].complete(200, "late");
    assert.equal(progress, 0);
  }
});

test("an already aborted signal prevents sending or allocating an upload", async () => {
  const { runtime, requests } = xhrHarness();
  const controller = new AbortController();
  controller.abort();
  const operation = runtime.upload("/api/workspace/restore", { signal: controller.signal });
  await assert.rejects(operation.promise, { name: "AbortError" });
  assert.equal(requests.length, 0);
});

test("upload setup errors reject the operation promise", async () => {
  const failure = new Error("Cannot open upload");
  class BrokenRequest { constructor() { throw failure; } }
  const runtime = create({ projectScope: scope(), adapter: createLocalAdapter({ XMLHttpRequest: BrokenRequest }) });
  const operation = runtime.upload("/api/workspace/restore");
  await assert.rejects(operation.promise, (error) => error === failure);
  assert.doesNotThrow(operation.abort);
});
