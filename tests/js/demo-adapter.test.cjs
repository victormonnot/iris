"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { create, capabilities } = require("../../src/iris/static/demo/adapter.js");
const storage = require("../../src/iris/static/demo/storage.js");
const annotations = require("../../src/iris/static/demo/annotations.js");
const fixture = require("../../examples/web-demo/seed/fixture.json");
const provenance = require("../../examples/web-demo/seed/provenance.json");

const FIRST = "frame-6723";
const SECOND = "frame-32941";
const SESSION = "/api/sessions/street-scenes";
const scope = (path) => `${path}${path.includes("?") ? "&" : "?"}project_id=web-demo`;
const body = (method, value) => ({ method, body: JSON.stringify(value) });

async function setup(t, changes = {}) {
  const seed = structuredClone(fixture);
  const store = await storage.create({ indexedDB: null, seedVersion: "test-seed", initialState: annotations.createState(seed) });
  const events = [];
  const adapter = create({
    load: async () => ({ seed, store, provenance, baseURL: "https://iris.example/demo/seed/", ...changes }),
    onChange: (event) => events.push(event),
  });
  t.after(() => adapter.close());
  await adapter.ready;
  return { adapter, store, seed, events };
}

async function read(adapter, path, expected = 200, options) {
  const response = await adapter.request(scope(path), options);
  assert.equal(response.status, expected, `${path}: ${await response.clone().text()}`);
  assert.match(response.headers.get("content-type"), /application\/json/);
  return response.json();
}

function savePayload(document, changes = {}) {
  return {
    demo_epoch: document.demo_epoch, expected_revision: document.revision,
    taxonomy_id: document.taxonomy_id, boxes: [], decisions: {},
    status: "draft", reviewer: "", notes: "", ...changes,
  };
}

function manual(id = "manual") {
  return { id, label: "person", box: [1, 2, 11, 22], suggestion_id: null };
}

test("demo requests never fall through to fetch, XHR, upload, or unavailable native operations", async (t) => {
  t.mock.method(globalThis, "fetch", () => assert.fail("The demo must not fetch native API routes"));
  const xhr = Object.getOwnPropertyDescriptor(globalThis, "XMLHttpRequest");
  Object.defineProperty(globalThis, "XMLHttpRequest", {
    configurable: true, value: class { constructor() { assert.fail("The demo must not create XHR"); } },
  });
  t.after(() => {
    if (xhr) Object.defineProperty(globalThis, "XMLHttpRequest", xhr);
    else delete globalThis.XMLHttpRequest;
  });
  const { adapter, store } = await setup(t);
  assert.equal(adapter.mode, "demo");
  assert.equal(adapter.capabilities, capabilities);
  assert.equal(adapter.capabilities.training, false);
  assert.equal(adapter.capabilities.annotation, true);
  const initial = await store.read();
  await read(adapter, "/api/projects");
  for (const [method, path] of [
    ["POST", `${SESSION}/assets`], ["POST", `${SESSION}/comparisons`],
    ["POST", "/api/training-runs"], ["POST", "/api/evaluations"],
    ["POST", "/api/projects"], ["DELETE", `${SESSION}`],
    ["POST", `/api/frames/${FIRST}/assistance`], ["POST", "/api/workspace/backups"],
    ["GET", "/api/private-route"],
  ]) {
    const error = await read(adapter, path, 501, body(method, {}));
    assert.match(error.detail, /local Iris/);
  }
  const upload = adapter.upload(scope(`${SESSION}/assets`), { body: new FormData() });
  assert.equal(typeof upload.abort, "function");
  upload.abort();
  await assert.rejects(upload.promise, { status: 501 });
  assert.deepEqual(await store.read(), initial);
});

test("scope, query and resource restrictions fail closed", async (t) => {
  const { adapter } = await setup(t);
  for (const [path, status] of [
    ["https://other.example/api/sessions?project_id=web-demo", 400],
    ["//other.example/api/sessions?project_id=web-demo", 400],
    ["/static/app.js", 400], ["/api/../static/app.js?project_id=web-demo", 400],
    ["/api/frames\\frame-6723?project_id=web-demo", 400],
    ["/api/sessions", 404], ["/api/sessions?project_id=default", 404],
    ["/api/projects?project_id=other", 404],
    ["/api/sessions?project_id=web-demo&project_id=web-demo", 422],
    ["/api/sessions?project_id=web-demo&unknown=true", 422],
    ["/api/sessions?project_id=web-demo&comparison_id=saved", 422],
    ["/api/sessions?project_id=web-demo&confidence_threshold=0.5", 422],
    ["/api/sessions?project_id=web-demo#fragment", 400],
  ]) {
    const response = await adapter.request(path);
    assert.equal(response.status, status, path);
  }
  const resource = `/api/frames/${FIRST}/image`;
  assert.equal(adapter.resourceURL(scope(resource)), "https://iris.example/demo/seed/media/6723.png");
  assert.equal(adapter.resourceURL(scope(`/api/assets/asset-32941/media`)), "https://iris.example/demo/seed/media/32941.jpg");
  assert.throws(() => adapter.resourceURL(resource), { status: 404 });
  assert.throws(() => adapter.resourceURL(scope("/api/workspace/archives/private")), { status: 501 });
  assert.throws(() => adapter.resourceURL(scope(`${resource}?comparison_id=saved`)), { status: 422 });
  assert.throws(() => adapter.resourceURL("https://other.example/image.png"), { status: 400 });
});

test("initialization failures remain visible without a local fallback and reset requires explicit action", async (t) => {
  t.mock.method(globalThis, "fetch", () => assert.fail("No fallback fetch"));
  const error = new Error("Demo files unavailable");
  let resets = 0;
  let closed = 0;
  error.reset = async () => { resets++; return { close() { closed++; } }; };
  const adapter = create({ load: async () => { throw error; } });
  await assert.rejects(adapter.ready, (value) => value === error);
  await assert.rejects(adapter.request(scope("/api/sessions")), (value) => value === error);
  await assert.rejects(adapter.exportAnnotations(), (value) => value === error);
  assert.equal(adapter.error, error);
  assert.equal(resets, 0);
  assert.throws(() => adapter.resourceURL(scope(`/api/frames/${FIRST}/image`)), { status: 503 });
  await adapter.reset();
  assert.equal(resets, 1);
  assert.equal(closed, 1);
});

test("invalid seed media locations cannot escape the declared public media directory", async (t) => {
  for (const target of ["https://other.example/image.jpg", "../private.jpg", "media/../private.jpg", "media/..", "media/."]) {
    const seed = structuredClone(fixture);
    seed.resources[`/api/frames/${FIRST}/image`] = target;
    const store = await storage.create({ indexedDB: null, seedVersion: 1, initialState: annotations.createState(seed) });
    t.after(() => store.close());
    const adapter = create({ load: async () => ({ seed, store, provenance, baseURL: "https://iris.example/demo/seed/" }) });
    await assert.rejects(adapter.ready, { status: 503 }, target);
    assert.throws(() => adapter.resourceURL(scope("/api/assets/asset-32941/media")), { status: 503 });
  }
});

test("loaded endpoints expose browser capabilities and recorded model identities without runtime claims", async (t) => {
  const { adapter } = await setup(t);
  const system = await read(adapter, "/api/system");
  assert.equal(system.version, provenance.iris_version);
  assert.match(system.data_dir, /tab.*demo/);
  assert.deepEqual(system.capabilities, capabilities);
  assert.deepEqual(await read(adapter, "/api/jobs"), []);
  assert.deepEqual(await read(adapter, "/api/annotation-providers"), { providers: [], default_provider: null, default_model: null });
  const models = await read(adapter, "/api/models");
  assert.equal(models.length, 2);
  assert.ok(models.every((model) => model.status === "unavailable" && model.runtime_load_verified === false));
  assert.deepEqual(models.map((model) => model.weight_sha256), fixture.responses["/api/models"].map((model) => model.weight_sha256));
});

test("annotation writes, revision reads and API conflicts use the same current storage snapshot", async (t) => {
  const { adapter, events } = await setup(t);
  const path = `/api/frames/${FIRST}/annotation`;
  const initial = await read(adapter, path);
  const input = savePayload(initial, { boxes: [manual()], notes: "Saved browser draft" });
  const saved = await read(adapter, path, 200, body("PUT", input));
  assert.equal(saved.revision, 1);
  assert.equal(saved.boxes[0].source.kind, "manual");
  assert.deepEqual(await read(adapter, path), saved);
  const revision = await read(adapter, `${path}/revisions/1`);
  assert.equal(revision.notes, "Saved browser draft");
  assert.deepEqual(revision.boxes, saved.boxes);
  await read(adapter, path, 409, body("PUT", input));
  await read(adapter, path, 422, { method: "PUT", body: "not-json" });
  await read(adapter, path, 422, { method: "PUT", body: "[]" });
  await read(adapter, `${path}/revisions/2`, 404);
  await read(adapter, "/api/frames/missing/annotation", 404);
  assert.deepEqual(events, [{ kind: "annotation", frameId: FIRST }]);
  assert.equal((await read(adapter, `/api/frames/${SECOND}/annotation`)).revision, 0);
});

test("concurrent saves serialize so a stale revision cannot overwrite the winning edit", async (t) => {
  const { adapter } = await setup(t);
  const path = `/api/frames/${FIRST}/annotation`;
  const initial = await read(adapter, path);
  const responses = await Promise.all([
    adapter.request(scope(path), body("PUT", savePayload(initial, { notes: "first" }))),
    adapter.request(scope(path), body("PUT", savePayload(initial, { notes: "second" }))),
  ]);
  assert.deepEqual(responses.map((response) => response.status), [200, 409]);
  const saved = await read(adapter, path);
  assert.equal(saved.revision, 1);
  assert.equal(saved.notes, "first");
});

test("selection mutations are atomic and reject stale tab state, foreign frames and reset epochs", async (t) => {
  const { adapter, store, events } = await setup(t);
  const initial = await store.read();
  const patch = { demo_epoch: initial.epoch, selected: false, expected_selected: true };
  const changed = await read(adapter, `/api/frames/${FIRST}`, 200, body("PATCH", patch));
  assert.equal(changed.selected, false);
  await read(adapter, `/api/frames/${FIRST}`, 409, body("PATCH", patch));
  const stateBeforeBulk = await store.read();
  const bulk = {
    demo_epoch: initial.epoch, selected: true, frame_ids: [FIRST, SECOND],
    expected_selection: { [FIRST]: false, [SECOND]: false },
  };
  await read(adapter, `${SESSION}/selection`, 409, body("POST", bulk));
  assert.deepEqual(await store.read(), stateBeforeBulk);
  await read(adapter, `${SESSION}/selection`, 404, body("POST", {
    ...bulk, frame_ids: [FIRST, "missing"], expected_selection: { [FIRST]: false, missing: true },
  }));
  assert.deepEqual(await store.read(), stateBeforeBulk);
  await read(adapter, `${SESSION}/selection`, 422, body("POST", { ...bulk, frame_ids: [FIRST, FIRST] }));
  const selected = await read(adapter, `${SESSION}/selection`, 200, body("POST", {
    ...bulk, expected_selection: { [FIRST]: false, [SECOND]: true },
  }));
  assert.equal(selected.changed_count, 1);
  assert.ok(selected.frames.every((frame) => frame.selected));
  assert.deepEqual(events, [{ kind: "selection" }, { kind: "selection" }]);
  await adapter.reset();
  await read(adapter, `/api/frames/${FIRST}`, 409, body("PATCH", patch));
});

test("review counts and annotation insights follow saved decisions and selected images", async (t) => {
  const { adapter, store } = await setup(t);
  const path = `/api/frames/${FIRST}/annotation`;
  const initial = await read(adapter, path);
  await read(adapter, path, 200, body("PUT", savePayload(initial, {
    status: "validated", reviewer: "Visitor", decisions: Object.fromEntries(initial.suggestions.map((item) => [item.id, "rejected"])),
  })));
  const insights = await read(adapter, `${SESSION}/selection-insights`);
  assert.equal(insights.summary.review_counts.validated, 1);
  assert.equal(insights.summary.review_counts.pending_suggestions, 1);
  assert.equal(insights.summary.negative_frames, 1);
  const first = insights.frames.find((frame) => frame.frame_id === FIRST);
  assert.equal(first.pending_count, 0);
  assert.equal(first.revision, 1);
  assert.equal(first.negative, true);
  const queue = await read(adapter, `${SESSION}/review-queue?confidence_threshold=0.25&iou_threshold=0.7`);
  assert.deepEqual(queue.counts, { total: 2, needs_review: 1, unannotated: 0, draft: 0, pending: 1, validated: 1 });
  assert.equal(queue.config.confidence_threshold, 0.25);
  assert.equal(queue.config.iou_threshold, 0.7);
  assert.ok(queue.frames.every((frame) => frame.hints.prediction_status === "unavailable"));
  const state = await store.read();
  await read(adapter, `/api/frames/${FIRST}`, 200, body("PATCH", {
    demo_epoch: state.epoch, expected_selected: true, selected: false,
  }));
  const reduced = await read(adapter, `${SESSION}/review-queue`);
  assert.deepEqual(reduced.frames.map((frame) => frame.id), [SECOND]);
  assert.equal(reduced.counts.total, 1);
  assert.equal(reduced.counts.validated, 0);
});

test("review thresholds are bounded and unsupported comparison ordering is explicit", async (t) => {
  const { adapter } = await setup(t);
  for (const query of ["confidence_threshold=NaN", "confidence_threshold=", "confidence_threshold=1.1", "iou_threshold=0", "iou_threshold=-1", "iou_threshold=Infinity", "confidence_threshold=0.1&confidence_threshold=0.2"])
    await read(adapter, `${SESSION}/review-queue?${query}`, 422);
  await read(adapter, `${SESSION}/review-queue?comparison_id=${fixture.comparison_id}`, 501);
});

test("recorded comparison responses stay immutable after selection and annotation edits", async (t) => {
  const { adapter, store } = await setup(t);
  const path = `/api/comparisons/${fixture.comparison_id}`;
  const original = await read(adapter, path);
  original.predictions[0].detections[0].score = -1;
  assert.deepEqual(await read(adapter, path), fixture.responses[path]);
  const doc = await read(adapter, `/api/frames/${FIRST}/annotation`);
  await read(adapter, `/api/frames/${FIRST}/annotation`, 200, body("PUT", savePayload(doc, { boxes: [manual()] })));
  const state = await store.read();
  await read(adapter, `/api/frames/${FIRST}`, 200, body("PATCH", {
    demo_epoch: state.epoch, selected: false, expected_selected: true,
  }));
  assert.deepEqual(await read(adapter, path), fixture.responses[path]);
  assert.deepEqual(await read(adapter, `${SESSION}/comparisons`), fixture.responses[`${SESSION}/comparisons`]);
});

test("export uses current saved labels and attribution; reset restores seed without changing recordings", async (t) => {
  const { adapter, events } = await setup(t);
  const path = `/api/frames/${FIRST}/annotation`;
  const original = await read(adapter, path);
  assert.deepEqual((await adapter.exportAnnotations()).annotations, []);
  const saved = await read(adapter, path, 200, body("PUT", savePayload(original, { boxes: [manual()] })));
  const exported = await adapter.exportAnnotations();
  assert.equal(exported.annotations.length, 1);
  assert.deepEqual(exported.annotations[0].bbox, [1, 2, 10, 20]);
  assert.equal(exported.iris.reviews[0].status, "draft");
  assert.deepEqual(exported.iris.source_provenance, provenance.source);
  await adapter.reset();
  const reset = await read(adapter, path);
  assert.equal(reset.revision, 0);
  assert.notEqual(reset.demo_epoch, saved.demo_epoch);
  assert.deepEqual(reset.boxes, []);
  assert.ok(reset.suggestions.every((item) => item.state === "pending"));
  assert.ok((await read(adapter, `${SESSION}/frames`)).every((frame) => frame.selected));
  assert.deepEqual((await adapter.exportAnnotations()).annotations, []);
  await read(adapter, path, 409, body("PUT", savePayload(saved, { expected_revision: 0 })));
  assert.deepEqual(events, [{ kind: "annotation", frameId: FIRST }, { kind: "reset" }]);
});

test("aborted requests remain AbortErrors and cannot mutate the browser snapshot", async (t) => {
  const { adapter, store, events } = await setup(t);
  const initial = await store.read();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(adapter.request(scope(`/api/frames/${FIRST}`), {
    ...body("PATCH", { demo_epoch: initial.epoch, expected_selected: true, selected: false }),
    signal: controller.signal,
  }), { name: "AbortError" });
  assert.deepEqual(await store.read(), initial);
  assert.deepEqual(events, []);
});

test("storage write failure reports an error without an invented successful save or fallback", async (t) => {
  t.mock.method(globalThis, "fetch", () => assert.fail("A storage failure must not send the edit to a server"));
  const store = await storage.create({ indexedDB: null, seedVersion: 1, initialState: annotations.createState(fixture) });
  const events = [];
  const unavailable = {
    ...store,
    update: async () => { throw new DOMException("Browser storage quota exceeded", "QuotaExceededError"); },
  };
  const adapter = create({
    load: async () => ({ seed: fixture, store: unavailable, provenance, baseURL: "https://iris.example/demo/seed/" }),
    onChange: (event) => events.push(event),
  });
  t.after(() => adapter.close());
  await adapter.ready;
  const path = `/api/frames/${FIRST}/annotation`;
  const before = await read(adapter, path);
  const failure = await read(adapter, path, 503, body("PUT", savePayload(before, { boxes: [manual()] })));
  assert.match(failure.detail, /quota/);
  assert.deepEqual(await read(adapter, path), before);
  assert.deepEqual(events, []);
});
