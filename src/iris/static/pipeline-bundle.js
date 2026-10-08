"use strict";

(() => {
  const tools = window.IRISPipelineBundleTools, field = (name) => $(`#bundle-${name}`), safe = encodeURIComponent;
  const requestedID = new URL(window.location.href).searchParams.get("pipeline_bundle");
  const view = { visible: false, loaded: false, generation: 0, recordRequest: 0, mutationRequest: 0, formVersion: 0,
    loading: false, recordLoading: false, busy: false, pollBusy: false, timer: null, sources: [], history: [], source: null, preview: null, record: null };
  function error(value) { field("error").textContent = value?.message || value || ""; field("error").hidden = !value; }
  function options(select, rows, label, placeholder, selected = "") {
    select.replaceChildren(new Option(placeholder, ""));
    rows.forEach((row) => select.append(new Option(label(row), row.id)));
    select.value = rows.some((row) => row.id === selected) ? selected : "";
  }
  function configuration() { return tools.configuration({ name: field("name").value, target_device: field("target").value, selection_id: field("selection").value || null }, view.source); }
  function update() {
    const active = tools.active(view.record?.job), blocked = view.loading || view.recordLoading || view.busy || active;
    for (const control of field("prepare").querySelectorAll("input,select,button")) control.disabled = blocked;
    field("selection").disabled = blocked || !(view.source?.selections || []).length;
    field("refresh").disabled = view.loading || view.busy;
    field("history").disabled = view.loading || !view.history.length;
    field("cancel").disabled = view.busy || !active || view.record?.job.status === "cancelling";
    let valid = false, hint = "Review the exact settings and preview the files before packaging.";
    try { configuration(); valid = true; } catch (failure) { hint = failure.message; }
    if (view.history.some((row) => tools.active(row.job))) { valid = false; hint = "A bundle is being prepared. Inspect or cancel it in history before creating another."; }
    field("prepare-status").textContent = view.loading || view.recordLoading ? "Loading saved sources and bundles…" : view.busy ? "Processing the explicit request…" : hint;
    field("preview").disabled = blocked || !valid;
    field("run").disabled = blocked || !valid || !view.preview;
    $("#pipeline-bundle-panel").setAttribute("aria-busy", String(blocked));
  }
  function clearPreview() { view.formVersion++; view.preview = null; field("preview-result").hidden = true; }
  function clearResult() { field("result").hidden = true; field("manifest").textContent = ""; }
  function renderHistory() { options(field("history"), view.history, (row) => `${row.name} · ${row.job.status}`, "Choose a saved bundle", view.record?.id); }
  function renderSource() {
    field("source-summary").textContent = tools.sourceSummary(view.source);
    const selected = field("selection").value;
    options(field("selection"), view.source?.selections || [], (row) => `${row.name} · guarded geometry`, "No selected-object policy", selected);
    renderPolicy();
  }
  function renderPolicy() { field("policy-summary").textContent = tools.policySummary(view.source?.selections?.find((row) => row.id === field("selection").value)?.policy); }
  function invalidate() {
    clearPreview(); view.recordRequest++; view.record = null; field("job").hidden = true; clearResult(); renderHistory(); renderPolicy(); update();
  }
  function reset() {
    clearTimeout(view.timer); view.timer = null; view.generation++; view.recordRequest++; view.mutationRequest++;
    view.loaded = false; view.loading = false; view.recordLoading = false; view.busy = false; view.pollBusy = false;
    view.sources = []; view.history = []; view.source = null; view.record = null; clearPreview(); clearResult();
    field("job").hidden = true; options(field("source"), [], () => "", "Choose a saved source"); renderHistory(); renderSource(); error(null); update();
  }
  async function refresh() {
    if (view.loading || view.busy) return;
    const generation = view.generation, token = view.recordRequest, recordID = view.record?.id;
    view.loading = true; clearPreview(); error(null); update();
    try {
      const [catalogue, status, history] = await Promise.all([api("/api/temporal/pipeline-bundle-sources"), api("/api/temporal/pipeline-bundle-status"), api("/api/temporal/pipeline-bundles")]);
      if (generation !== view.generation || !view.visible) return;
      view.sources = catalogue.sources; view.history = history; view.loaded = true;
      const savedRequest = view.record?.job.params?.request;
      const key = savedRequest ? tools.sourceKey(savedRequest.source) : view.source ? tools.sourceKey(view.source.source) : field("source").value;
      options(field("source"), view.sources.map((row) => ({ ...row, id: tools.sourceKey(row.source) })), tools.sourceLabel, "Choose a saved source", key);
      view.source = view.sources.find((row) => tools.sourceKey(row.source) === field("source").value) || null;
      renderSource(); if (savedRequest) { field("selection").value = savedRequest.selection_id || ""; renderPolicy(); } renderHistory();
      field("readiness").textContent = `${view.sources.filter((row) => row.available !== false).length} saved profiles available for packaging.${status.reason ? ` ${status.reason}` : ""}`;
      const changed = recordID && token === view.recordRequest && history.find((row) => row.id === recordID && row.job.status !== view.record?.job.status);
      if (changed) await loadRecord(recordID, false);
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.loading = false; update(); schedulePoll(); } }
  }
  function renderJob() {
    const job = view.record?.job; field("job").hidden = !job; if (!job) return;
    field("job-status").textContent = [view.record.name, job.status, job.message, job.error].filter(Boolean).join(" · ");
    field("progress").hidden = !tools.active(job); field("progress").value = (job.progress || 0) * 100;
    field("cancel").hidden = !tools.active(job);
  }
  function renderManifest(manifest, element) {
    const detector = manifest.detector, profile = manifest.tracker.profile;
    const entries = detector.output_mapping.entries.filter((row) => profile.class_ids.includes(row.output_id));
    element.replaceChildren(
      node("p", "", `${detector.config.architecture} · ${tools.bytes(detector.checkpoint.size)} unchanged checkpoint · intended detector target: ${detector.target_device === "cuda" ? "NVIDIA CUDA" : "CPU"}.`),
      node("p", "", `${profile.algorithm === "bytetrack" ? "ByteTrack" : "BoT-SORT"} on CPU · ${entries.map((row) => `${row.label} (${row.output_id})`).join(", ")} · score floor ${detector.config.min_score}.`),
      node("p", "", profile.gmc_method === "sparseOptFlow" ? "Camera compensation requires original BGR frames, one estimate per available update." : "Camera compensation disabled. Memory advances once per available analyzed frame; skipped frames do not become empty detections."),
      node("p", "", tools.policySummary(manifest.selection?.policy)),
      node("p", "", `Recorded detector device: ${detector.config.device}. Intended target: ${detector.target_device === "cuda" ? "NVIDIA CUDA" : "CPU"}.`),
    );
  }
  function renderResult(record) {
    const manifest = record.bundle?.manifest;
    if (record.job.status !== "succeeded" || !manifest) { clearResult(); return; }
    field("result").hidden = false; field("result-title").textContent = record.name;
    renderManifest(manifest, field("result-summary"));
    field("result-scope").textContent = "Experimental · package integrity verified · target execution not tested.";
    field("manifest").textContent = JSON.stringify(manifest, null, 2);
    field("download").href = projectURL(`/api/temporal/pipeline-bundles/${safe(record.id)}/download`);
    field("download-manifest").href = projectURL(`/api/temporal/pipeline-bundles/${safe(record.id)}/manifest`);
    const link = new URL(window.location.href); link.search = ""; link.searchParams.set("project", state.projectId); link.searchParams.set("pipeline_bundle", record.id); field("link").href = link.href;
  }
  function acceptRecord(record, scroll = false, restore = false) {
    const index = view.history.findIndex((row) => row.id === record.id);
    if (index < 0) view.history.unshift(record); else view.history[index] = record;
    view.record = record;
    if (restore) {
      const request = record.job.params?.request;
      if (request) {
        field("name").value = request.name; field("target").value = request.target_device;
        view.source = view.sources.find((row) => tools.sourceKey(row.source) === tools.sourceKey(request.source)) || null;
        field("source").value = tools.sourceKey(request.source); renderSource();
        field("selection").value = request.selection_id || ""; renderPolicy();
      }
    }
    renderHistory(); renderJob(); renderResult(record); update(); schedulePoll();
    if (scroll) (field("result").hidden ? field("job") : field("result")).scrollIntoView({ block: "start" });
  }
  async function loadRecord(id, scroll = false) {
    const generation = view.generation, request = ++view.recordRequest;
    view.mutationRequest++; view.busy = false; clearPreview(); clearResult(); view.record = null; field("job").hidden = true; error(null);
    view.recordLoading = Boolean(id); update(); if (!id) return;
    try {
      const record = await api(`/api/temporal/pipeline-bundles/${safe(id)}`);
      if (generation !== view.generation || request !== view.recordRequest || !view.visible) return;
      acceptRecord(record, scroll, true);
    } catch (failure) { if (generation === view.generation && request === view.recordRequest) error(failure); }
    finally { if (generation === view.generation && request === view.recordRequest) { view.recordLoading = false; update(); } }
  }
  async function mutation(action) {
    if (view.busy || view.loading || view.recordLoading) return;
    const generation = view.generation, token = ++view.mutationRequest, record = view.recordRequest, version = view.formVersion;
    const current = () => generation === view.generation && token === view.mutationRequest && record === view.recordRequest && version === view.formVersion && view.visible;
    view.busy = true; error(null); update();
    try { await action(current); }
    catch (failure) { if (current()) error(failure); }
    finally { if (generation === view.generation && token === view.mutationRequest) { view.busy = false; update(); schedulePoll(); } }
  }
  function schedulePoll() { clearTimeout(view.timer); view.timer = null; if (view.visible && view.history.some((row) => tools.active(row.job))) view.timer = setTimeout(poll, 1800); }
  async function poll() {
    view.timer = null; if (!view.visible || view.pollBusy || view.busy || view.loading || view.recordLoading) { schedulePoll(); return; }
    const generation = view.generation, token = view.recordRequest, id = view.record?.id; view.pollBusy = true;
    try {
      const [history, record] = await Promise.all([api("/api/temporal/pipeline-bundles"), id ? api(`/api/temporal/pipeline-bundles/${safe(id)}`) : Promise.resolve(null)]);
      if (generation !== view.generation || !view.visible) return;
      view.history = history; renderHistory();
      if (record && token === view.recordRequest && id === view.record?.id) acceptRecord(record);
    } catch (failure) { if (generation === view.generation && token === view.recordRequest) error(failure); }
    finally { if (generation === view.generation) { view.pollBusy = false; update(); schedulePoll(); } }
  }
  function open(id) { if (!id || window.IRISNavigation.open("tracking") === false) return false; loadRecord(id, true); return true; }
  field("source").addEventListener("change", () => { view.source = view.sources.find((row) => tools.sourceKey(row.source) === field("source").value) || null; field("selection").value = ""; renderSource(); error(null); invalidate(); });
  field("selection").addEventListener("change", invalidate); field("target").addEventListener("change", invalidate); field("name").addEventListener("input", invalidate);
  field("history").addEventListener("change", () => loadRecord(field("history").value, true)); field("refresh").addEventListener("click", refresh);
  field("preview").addEventListener("click", () => {
    let request; try { request = configuration(); } catch (failure) { error(failure); return; }
    mutation(async (current) => {
      const preview = await api("/api/temporal/pipeline-bundles/preview", { method: "POST", body: JSON.stringify(request) });
      if (!current()) return; view.preview = preview; field("preview-result").hidden = false;
      renderManifest(preview.manifest, field("preview-summary"));
      field("limits").replaceChildren(...(preview.limitations || []).map((value) => node("li", "", value)));
      field("preview-manifest").textContent = JSON.stringify(preview.manifest, null, 2);
    });
  });
  field("run").addEventListener("click", () => {
    if (!view.preview || view.history.some((row) => tools.active(row.job))) return;
    const payload = { ...view.preview.request, expected_fingerprint: view.preview.fingerprint };
    mutation(async (current) => {
      const record = await api("/api/temporal/pipeline-bundles", { method: "POST", body: JSON.stringify(payload) });
      if (!current()) return; clearPreview(); acceptRecord(record, true); if (typeof refreshJobs === "function") refreshJobs().catch(() => {});
    });
  });
  field("cancel").addEventListener("click", () => mutation(async (current) => {
    const record = view.record; if (!tools.active(record?.job)) return;
    await api(`/api/jobs/${safe(record.job.id)}/cancel`, { method: "POST" });
    if (!current() || view.record !== record) return; record.job.status = "cancelling"; renderJob();
  }));
  window.addEventListener("iris:pipeline-bundle-open", (event) => open(event.detail.bundle_id));
  window.addEventListener("iris:workspace", (event) => {
    view.visible = event.detail.name === "tracking";
    if (view.visible) { if (!view.loaded) refresh(); else schedulePoll(); }
    else { clearTimeout(view.timer); view.timer = null; view.generation++; view.recordRequest++; view.mutationRequest++; view.loading = false; view.recordLoading = false; view.busy = false; view.pollBusy = false; view.loaded = false; clearPreview(); }
  });
  window.addEventListener("iris:project-initialized", () => { reset(); if (requestedID) open(requestedID); else if (view.visible) refresh(); });
  window.IRISPipelineBundle = Object.freeze({ open }); update();
})();
