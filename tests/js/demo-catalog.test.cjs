"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const catalog = require("../../src/iris/static/demo/catalog.js");

function setup() {
  return catalog.create({
    project_id: "sample", session_id: "sample-session",
    responses: {
      "/api/projects/sample/taxonomies": {
        current_taxonomy_id: "classes-a",
        versions: [
          { id: "classes-a", classes: [{ id: "person", name: "Person" }] },
          { id: "classes-b", classes: [{ id: "vehicle", name: "Vehicle" }] },
        ],
      },
      "/api/models": [{ id: "recorded-detector", name: "Recorded detector", status: "ready" }],
    },
  });
}

test("empty catalogue does not fabricate saved work or match unknown resources", () => {
  const view = setup();
  for (const path of ["/api/datasets", "/api/trainings", "/api/evaluations", "/api/experiments",
    "/api/benchmarks", "/api/temporal/sequences", "/api/model-exports",
    "/api/sessions/sample-session/preannotations", "/api/sessions/sample-session/dinox-batches"])
    assert.deepEqual(view.response(path), []);
  for (const path of ["/api/trainings/unknown", "/api/sessions/other/preannotations",
    "/api/temporal/pipeline-bundles/preview", "/api/datasets/plan", "/api/unknown"])
    assert.equal(view.response(path), undefined);
});

test("catalogue never claims an executable model, provider or device", () => {
  const view = setup();
  const local = view.response("/api/preannotation-providers").providers[0];
  assert.equal(local.available, false);
  assert.equal(local.models[0].status, "unavailable");
  assert.equal(local.models[0].runtime_load_verified, false);
  for (const device of view.response("/api/training/devices").devices) {
    assert.equal(device.available, false);
    assert.match(device.reason, /demo mode/);
  }
  const benchmark = view.response("/api/benchmark-providers");
  for (const provider of [benchmark.multimodal, benchmark.segmentation, benchmark.combined,
    view.response("/api/dinox/provider"), view.response("/api/temporal/tracking-status"),
    view.response("/api/temporal/tracking-study-status").runtime]) {
    assert.equal(provider.available, false);
    assert.equal(provider.status, "unavailable");
    assert.match(provider.reason, /demo mode/);
  }
  assert.equal(view.response("/api/dinox/provider").key_source, null);
});

test("class selection is scoped and response mutation cannot alter the catalogue", () => {
  const view = setup();
  const path = "/api/dataset-candidates";
  assert.deepEqual(view.queryParameters(path), ["taxonomy_id"]);
  assert.deepEqual(view.queryParameters("/api/trainings"), []);
  const response = view.response(path);
  response.taxonomy.classes[0].name = "Changed";
  response.groups.push({ id: "invented" });
  assert.equal(view.response(path).taxonomy.classes[0].name, "Person");
  assert.deepEqual(view.response(path).groups, []);
  assert.equal(view.response(path, new URLSearchParams("taxonomy_id=classes-b")).taxonomy.id, "classes-b");
  assert.throws(() => view.response(path, new URLSearchParams("taxonomy_id=foreign")), { status: 404 });
  const providers = view.response("/api/preannotation-providers");
  providers.providers[0].models[0].status = "ready";
  assert.equal(view.response("/api/preannotation-providers").providers[0].models[0].status, "unavailable");
});
