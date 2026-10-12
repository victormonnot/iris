"use strict";

((root, factory) => {
  const domain = typeof module === "object" && module.exports
    ? require("./annotations.js") : root.IRISDemoAnnotations;
  const tools = factory(domain);
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISDemoAdapter = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, (annotations) => {
  const LOCAL_ONLY = "This action requires a local Iris installation. The demo supports sample selection, annotation and recorded comparisons.";
  const copy = (value) => structuredClone(value);
  const fail = (status, message) => { throw Object.assign(new Error(message), { status }); };
  const json = (value, status = 200) => new Response(JSON.stringify(value), {
    status, headers: { "Content-Type": "application/json" },
  });
  const capabilities = Object.freeze({
    projects: true, frame_selection: true, selection_insights: true, annotation: true,
    review_queue: true, comparison_replay: true, annotation_export: true,
    media_import: false, inference: false, training: false, evaluation: false,
    assisted_annotation: false, workspace_backup: false,
  });

  function create({ load, onChange = () => {} }) {
    let seed, store, provenance, base, resources, initializationError;
    let ready = Promise.resolve().then(load).then((loaded) => {
      ({ seed, store, provenance } = loaded);
      if (seed.format !== "iris-web-demo-v1" || seed.fixture_version !== 1)
        fail(503, "This example format is not supported. Update or reload the demo.");
      base = new URL(loaded.baseURL);
      if (!/^https?:$/.test(base.protocol) || !base.pathname.endsWith("/"))
        fail(503, "Invalid demo asset location.");
      const mappedResources = new Map();
      for (const [path, target] of Object.entries(seed.resources)) {
        if (!/^media\/[A-Za-z0-9_-][A-Za-z0-9_.-]*\.(jpg|png)$/.test(target)) fail(503, "Invalid demo media path.");
        mappedResources.set(path, new URL(target, base).href);
      }
      resources = mappedResources;
      return loaded;
    }).catch((error) => { initializationError = error; throw error; });
    // The shared UI may not call request until a later task; keep startup errors observable.
    ready.catch(() => {});

    function route(path, allowed = []) {
      if (typeof path !== "string" || !path.startsWith("/api/") || path.includes("\\"))
        fail(400, "Only logical Iris API paths are accepted by the demo.");
      const url = new URL(path, base);
      if (url.origin !== base.origin || !url.pathname.startsWith("/api/") || url.hash)
        fail(400, "Invalid demo request path.");
      for (const key of new Set(url.searchParams.keys())) {
        if (url.searchParams.getAll(key).length !== 1 || !["project_id", ...allowed].includes(key))
          fail(422, "Unsupported or repeated demo query parameter.");
      }
      const global = /^\/api\/(system|projects|annotation-providers)(\/|$)/.test(url.pathname);
      const scope = url.searchParams.get("project_id");
      if ((!global && scope !== seed.project_id) || (scope && scope !== seed.project_id))
        fail(404, "This project is not part of the demo.");
      return url;
    }

    const frameList = () => seed.responses[`/api/sessions/${seed.session_id}/frames`];
    function frames(state) {
      return frameList().map((frame) => ({
        ...copy(frame), selected: state.selection[frame.id], demo_epoch: state.epoch,
      }));
    }
    function progress(document) {
      const pending = document.suggestions.filter((item) => item.state === "pending").length;
      const validated = document.status === "validated" && !pending;
      return {
        annotation_status: document.status, revision: document.revision,
        pending_count: pending, box_count: document.boxes.length,
        review_status: pending ? "pending_suggestions" : document.status,
        class_counts: Object.fromEntries(document.taxonomy.classes.map((category) => [
          category.id, document.boxes.filter((box) => box.label === category.id).length,
        ])),
        negative: validated ? !document.boxes.length : null,
        positive: validated ? Boolean(document.boxes.length) : null,
      };
    }
    function insights(state) {
      const value = copy(seed.responses[`/api/sessions/${seed.session_id}/selection-insights`]);
      value.frames = value.frames.map((frame) => ({
        ...frame, ...progress(annotations.document(seed, state, frame.frame_id)),
      }));
      value.summary.review_counts = Object.fromEntries(
        ["unannotated", "draft", "validated", "pending_suggestions"].map((status) => [
          status, value.frames.filter((frame) => frame.review_status === status).length,
        ]),
      );
      value.summary.positive_frames = value.frames.filter((frame) => frame.positive).length;
      value.summary.negative_frames = value.frames.filter((frame) => frame.negative).length;
      return value;
    }
    function reviewQueue(state, query) {
      if (query.get("comparison_id")) fail(501, "Comparison-based review ordering requires local Iris. Inspect the recorded outputs in Model comparison.");
      for (const key of ["confidence_threshold", "iou_threshold"]) {
        if (!query.has(key)) continue;
        const text = query.get(key), number = Number(text);
        if (!text.trim() || !Number.isFinite(number) || number < 0 || number > 1 || (key === "iou_threshold" && number === 0))
          fail(422, "Invalid review threshold.");
      }
      const value = copy(seed.responses[`/api/sessions/${seed.session_id}/review-queue`]);
      value.frames = value.frames.filter((frame) => state.selection[frame.id]).map((frame) => {
        const document = annotations.document(seed, state, frame.id);
        const info = progress(document);
        return {
          ...frame, ...info, reviewer: document.reviewer,
          review_status: info.pending_count ? "pending" : document.status,
          // Recorded detector outputs stay available, but live review hints are not recomputed.
          hints: { ...frame.hints, low_confidence_count: 0, uncertain_count: 0,
            possible_omission: false, reasons: [], prediction_id: null,
            prediction_status: "unavailable", prediction_reason: "Inspect saved outputs in Model comparison.",
            sources_inspected: 0, sources_truncated: false },
        };
      });
      for (const key of ["confidence_threshold", "iou_threshold"])
        if (query.has(key)) value.config[key] = Number(query.get(key));
      value.counts = { total: value.frames.length, needs_review: 0,
        unannotated: 0, draft: 0, pending: 0, validated: 0 };
      for (const frame of value.frames) {
        value.counts[frame.review_status]++;
        if (frame.review_status !== "validated") value.counts.needs_review++;
      }
      return value;
    }
    function payload(options) {
      let value;
      try { value = JSON.parse(options.body); } catch { fail(422, "Expected a JSON request body."); }
      if (!value || typeof value !== "object" || Array.isArray(value)) fail(422, "Expected an object.");
      return value;
    }
    function epoch(state, body) {
      if (body.demo_epoch !== state.epoch)
        fail(409, "The demo was reset or updated in another tab. Reload before saving.");
    }
    function selection(state, body, ids, expected) {
      epoch(state, body);
      if (typeof body.selected !== "boolean" || !Array.isArray(ids) || !ids.length || new Set(ids).size !== ids.length)
        fail(422, "Select one or more distinct example frames.");
      for (const id of ids) {
        if (!Object.hasOwn(state.selection, id)) fail(404, "Frame is not part of this demo.");
        if (typeof expected?.[id] !== "boolean" || expected[id] !== state.selection[id])
          fail(409, "Selection changed in another tab. Refresh and try again.");
      }
      let changed = 0;
      for (const id of ids) {
        if (state.selection[id] !== body.selected) changed++;
        state.selection[id] = body.selected;
      }
      return changed;
    }

    async function request(path, options = {}) {
      try {
        options.signal?.throwIfAborted();
        await ready;
        options.signal?.throwIfAborted();
        const isReviewQueue = typeof path === "string" && path.split("?")[0] === `/api/sessions/${seed.session_id}/review-queue`;
        const parsed = route(path, isReviewQueue ? ["comparison_id", "confidence_threshold", "iou_threshold"] : []);
        const key = parsed.pathname, method = options.method || "GET";
        if (method === "GET") {
          if (key === "/api/system") return json({version: provenance.iris_version,
            data_dir: store.status.mode === "memory" ? "This tab only · temporary demo" : "This browser · demo data",
            capabilities});
          if (key === "/api/annotation-providers") return json({providers: [], default_provider: null, default_model: null});
          if (key === "/api/jobs") return json([]);
          if (key === "/api/models") return json(seed.responses[key].map((model) => ({
            ...model, status: "unavailable", reason: "Recorded example. Install Iris locally to run models.",
            runtime_load_verified: false,
          })));
          const state = await store.read();
          options.signal?.throwIfAborted();
          if (key === `/api/sessions/${seed.session_id}/frames`) return json(frames(state));
          if (key === `/api/sessions/${seed.session_id}/selection-insights`) return json(insights(state));
          if (key === `/api/sessions/${seed.session_id}/review-queue`) return json(reviewQueue(state, parsed.searchParams));
          const annotation = key.match(/^\/api\/frames\/([^/]+)\/annotation$/);
          if (annotation) return json(annotations.document(seed, state, annotation[1]));
          const revision = key.match(/^\/api\/frames\/([^/]+)\/annotation\/revisions\/([1-9]\d*)$/);
          if (revision) return json(annotations.revision(seed, state, revision[1], Number(revision[2])));
          if (Object.hasOwn(seed.responses, key)) return json(seed.responses[key]);
          fail(501, LOCAL_ONLY);
        }
        const annotation = key.match(/^\/api\/frames\/([^/]+)\/annotation$/);
        const oneFrame = key.match(/^\/api\/frames\/([^/]+)$/);
        const bulk = key === `/api/sessions/${seed.session_id}/selection`;
        if (method === "PUT" && annotation) {
          const body = payload(options);
          const updated = await store.update((state) => {
            options.signal?.throwIfAborted();
            annotations.save(seed, state, annotation[1], body);
          });
          onChange({ kind: "annotation", frameId: annotation[1] });
          return json(annotations.document(seed, updated, annotation[1]));
        }
        if ((method === "PATCH" && oneFrame) || (method === "POST" && bulk)) {
          const body = payload(options);
          const ids = oneFrame ? [oneFrame[1]] : body.frame_ids;
          const expected = oneFrame ? { [oneFrame[1]]: body.expected_selected } : body.expected_selection;
          let changed;
          const updated = await store.update((state) => {
            options.signal?.throwIfAborted();
            changed = selection(state, body, ids, expected);
          });
          const result = frames(updated).filter((frame) => ids.includes(frame.id));
          onChange({ kind: "selection" });
          return json(oneFrame ? result[0] : { frames: result, changed_count: changed });
        }
        fail(501, LOCAL_ONLY);
      } catch (error) {
        if (error.name === "AbortError" || !seed || !store) throw error;
        return json({ detail: error.message || "The demo could not save this change." }, error.status || 503);
      }
    }

    return {
      mode: "demo", capabilities,
      get ready() { return ready; },
      get status() { return store?.status || { mode: "loading" }; },
      get error() { return initializationError; },
      request,
      resourceURL(path) {
        if (!resources) fail(503, "The example media is still loading.");
        const key = route(path).pathname;
        if (!resources.has(key)) fail(501, LOCAL_ONLY);
        return resources.get(key);
      },
      upload() {
        const promise = Promise.reject(Object.assign(new Error(LOCAL_ONLY), { status: 501 }));
        return { promise, abort() {} };
      },
      async reset() {
        if (initializationError?.reset) {
          const reopened = await initializationError.reset();
          reopened.close();
          onChange({ kind: "reset" });
          // The caller reloads after an explicit reset; bootstrap owns reopening the bundle.
          return;
        }
        await ready;
        await store.reset();
        onChange({ kind: "reset" });
      },
      async exportAnnotations() {
        await ready;
        return annotations.exportCOCO(seed, await store.read(), { provenance });
      },
      close() { store?.close(); },
    };
  }
  return { create, capabilities, LOCAL_ONLY };
});
