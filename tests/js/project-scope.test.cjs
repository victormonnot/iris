"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { create } = require("../../src/iris/static/project-scope.js");

function storage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
}

test("separate open projects keep their requests and session recall isolated", () => {
  const saved = storage({ "iris.session": "old-session" });
  const first = create("http://localhost:8000/?project=default", saved);
  const second = create("http://localhost:8000/?project=inventory", saved);
  assert.equal(first.recalledSession(), "old-session");
  assert.equal(second.recalledSession(), null);
  first.rememberSession("argos-session");
  second.rememberSession("inventory-session");
  second.remember();
  assert.equal(first.recalledSession(), "argos-session");
  assert.equal(second.recalledSession(), "inventory-session");
  assert.equal(first.url("/api/sessions"), "/api/sessions?project_id=default");
  assert.equal(second.url("/api/sessions"), "/api/sessions?project_id=inventory");
  assert.equal(create("http://localhost:8000/", saved).id, "inventory");
});

test("project navigation leaves the current scope and remembered project unchanged", () => {
  const saved = storage({ "iris.project": "default" });
  const scope = create("http://localhost:8000/?project=default#details", saved);
  const target = scope.location("workshop");
  assert.equal(target, "/?project=workshop");
  // A cancelled browser beforeunload prompt continues on the original page.
  assert.equal(scope.id, "default");
  assert.equal(saved.getItem("iris.project"), "default");
  assert.equal(scope.url("/api/jobs"), "/api/jobs?project_id=default");
  const nextPage = create(`http://localhost:8000${target}`, saved);
  nextPage.remember();
  assert.equal(saved.getItem("iris.project"), "workshop");
  assert.equal(nextPage.recalledSession(), null);
});

test("detail, media, exports and previews are scoped with their existing parameters preserved", () => {
  const scope = create("http://localhost:8000/?project=inventory", storage());
  for (const path of [
    "/api/models", "/api/model-references", "/api/frames/frame/image",
    "/api/assets/video/media", "/api/datasets/dataset/export/coco",
    "/api/experiments/report/export?include_images=true&expected_revision=2",
    "/api/assistance-previews/preview/images/image", "/api/jobs/job/log",
    "/api/dataset-imports/import/images/image", "/api/video-reviews/review/images/image",
  ]) {
    const url = new URL(scope.url(path), "http://localhost:8000");
    assert.equal(url.searchParams.get("project_id"), "inventory");
    for (const [key, value] of new URL(path, url).searchParams)
      assert.equal(url.searchParams.get(key), value);
  }
  assert.equal(scope.url("/api/frames/frame/image?project_id=default"),
    "/api/frames/frame/image?project_id=inventory");
});

test("workspace backups, project catalog, providers and system remain global", () => {
  const scope = create("http://localhost:8000/?project=inventory", storage());
  for (const path of ["/api/system", "/api/projects", "/api/projects/default",
    "/api/annotation-providers", "/api/workspace/backup-preview",
    "/api/workspace/operations/id/archive"])
    assert.equal(scope.url(path), path);
});

test("project selection works without optional browser storage", () => {
  const blocked = {
    getItem() { throw new Error("blocked"); },
    setItem() { throw new Error("blocked"); },
  };
  const scope = create("http://localhost:8000/", blocked);
  assert.equal(scope.id, "default");
  assert.doesNotThrow(() => scope.remember());
  assert.doesNotThrow(() => scope.rememberSession("session"));
  assert.equal(scope.recalledSession(), null);
  assert.equal(create("http://localhost:8000/?project=inventory", blocked).id, "inventory");
});

test("remote and non-API URLs cannot be mistaken for scoped project resources", () => {
  const scope = create("http://localhost:8000/?project=inventory", storage());
  for (const url of ["https://example.org/api/frames/a/image", "//example.org/api/jobs", "/static/logo.svg"])
    assert.throws(() => scope.url(url), /local IRIS server/);
});
