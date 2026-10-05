"use strict";

(() => {
  const tools = window.IRISModelExportTools;
  const field = (name) => $(`#model-export-${name}`);
  const safe = encodeURIComponent;
  const view = { visible: false, models: [], rows: [], selected: null, detail: null,
    frames: new Set(), preview: null, generation: 0, candidateRequest: 0,
    historyRequest: 0, detailRequest: 0, loading: false, loadingHistory: false,
    loadingDetail: false, busy: null, measurement: null, measurementGeneration: 0,
    jobStatuses: new Map() };
  const options = () => ({ trained_model_id: field("model").value, evaluation_id: field("evaluation").value,
    frame_ids: selectedEvaluation()?.frames.filter((frame) => view.frames.has(frame.frame_id)).map((frame) => frame.frame_id) || [],
    name: field("name").value.trim(), target_device: field("target-device").value });
  const deviceLabel = (device) => device === "cpu" || !device ? "CPU" : `NVIDIA GPU (${device})`;
  const selectedModel = () => view.models.find((item) => item.id === field("model").value);
  const selectedEvaluation = () => selectedModel()?.evaluations?.find((item) => item.id === field("evaluation").value);
  const path = (id = view.selected) => `/api/model-exports/${safe(id)}`;
  const date = (value) => value ? new Date(value).toLocaleString() : "Date unavailable";
  function error(name, value) {
    field(name).textContent = value?.message || value || "";
    field(name).hidden = !value;
  }
  function invalidatePreview() {
    view.generation++; view.preview = null; field("preview-result").hidden = true; update();
  }
  function invalidateMeasurement(clearFile = false) {
    view.measurementGeneration++; view.measurement = null;
    field("measurement-review").hidden = true;
    if (clearFile) field("measurement-file").value = "";
    error("measurement-error", null); update();
  }
  function update() {
    const blocked = Boolean(view.busy || view.loading);
    field("refresh").disabled = blocked || view.loadingHistory;
    field("model").disabled = blocked || !view.models.length;
    field("evaluation").disabled = blocked || !selectedModel()?.eligible || !selectedModel()?.evaluations?.length;
    field("name").disabled = blocked;
    field("target-device").disabled = blocked;
    for (const input of field("frames").querySelectorAll("input")) input.disabled = blocked || (!input.checked && view.frames.size >= 8);
    field("frame-count").textContent = `${view.frames.size} image${view.frames.size === 1 ? "" : "s"} selected · maximum 8`;
    field("preview").disabled = blocked || !selectedModel()?.eligible || !tools.selectionValid(options());
    field("preview").textContent = view.busy === "preview" ? "Checking saved export inputs…" : "Preview export package";
    field("create").disabled = blocked || view.loadingHistory || !view.preview || view.preview.key !== tools.selectionKey(options());
    field("create").textContent = view.busy === "create" ? "Queuing export…" : "Create export package";
    field("history").disabled = Boolean(view.busy || view.loadingHistory || !view.rows.length);
    const importBlocked = Boolean(view.busy || view.loadingHistory || view.loadingDetail || !view.detail?.ready);
    field("measurement-file").disabled = importBlocked;
    field("measurement-preview").disabled = importBlocked || Boolean(tools.fileProblem(field("measurement-file").files?.[0]));
    field("measurement-preview").textContent = view.busy === "measurement-preview" ? "Checking file…" : "Check measurement file";
    field("measurement-save").disabled = importBlocked || !view.measurement || view.measurement.exportId !== view.selected;
    field("measurement-save").textContent = view.busy === "measurement-save" ? "Saving measurements…" : "Save measurements in IRIS";
  }
  function renderFrames() {
    const frames = selectedEvaluation()?.frames || [];
    view.frames = new Set([...view.frames].filter((id) => frames.some((frame) => frame.frame_id === id)));
    field("frames").replaceChildren();
    for (const frame of frames) {
      const label = node("label", "model-export-frame");
      const input = node("input"); input.type = "checkbox"; input.value = frame.frame_id; input.checked = view.frames.has(frame.frame_id);
      const title = `${frame.frame_id} · ${frame.width} × ${frame.height}`;
      input.setAttribute("aria-label", `Include parity image ${title}`);
      input.addEventListener("change", () => {
        if (input.checked) view.frames.add(frame.frame_id); else view.frames.delete(frame.frame_id);
        invalidatePreview();
      });
      label.append(input);
      if (frame.image_url) {
        const image = node("img"); image.src = projectURL(frame.image_url); image.alt = ""; image.loading = "lazy";
        image.addEventListener("error", () => { image.hidden = true; }); label.append(image);
      }
      label.append(node("span", "", title)); field("frames").append(label);
    }
    if (!frames.length) field("frames").append(node("p", "field-hint", "Choose an eligible checkpoint and a completed full-image evaluation to select parity images."));
    update();
  }
  function renderEvaluations(previous = null) {
    const model = selectedModel(), evaluations = model?.evaluations || [];
    field("evaluation").replaceChildren();
    if (!evaluations.length) field("evaluation").append(new Option("No eligible saved evaluation", ""));
    for (const item of evaluations) field("evaluation").append(new Option(`${item.name || item.id} · ${deviceLabel(item.device)} · ${item.frames?.length || 0} images`, item.id));
    if (evaluations.some((item) => item.id === previous)) field("evaluation").value = previous;
    field("model-reason").textContent = model?.reason || (model?.eligible ? "Uses saved native outputs from this exact checkpoint. Tiled evaluations are not eligible. Training, reference evaluation and destination devices are independent." : "Supported trained Faster R-CNN and SSDLite checkpoints can use a native PyTorch export profile.");
    renderFrames();
  }
  async function refreshCandidates() {
    if (view.busy || view.loading) return;
    const request = ++view.candidateRequest;
    const modelId = field("model").value, evaluationId = field("evaluation").value;
    view.loading = true; invalidatePreview(); error("error", null);
    field("candidates-status").textContent = "Reading trained checkpoints and saved CPU / CUDA evaluations…";
    try {
      const result = await api("/api/model-exports/candidates");
      if (request !== view.candidateRequest) return;
      view.models = result.models || []; field("model").replaceChildren();
      if (!view.models.length) field("model").append(new Option("No trained checkpoint available", ""));
      for (const model of view.models) field("model").append(new Option(`${model.name || model.id}${model.eligible ? "" : " · not ready"}`, model.id));
      if (view.models.some((item) => item.id === modelId)) field("model").value = modelId;
      else if (view.models.some((item) => item.eligible)) field("model").value = view.models.find((item) => item.eligible).id;
      field("candidates-status").textContent = view.models.length
        ? `${view.models.filter((item) => item.eligible).length} of ${view.models.length} trained checkpoints ready for export preparation.`
        : "No trained checkpoint in this project yet. Complete training and a full-image evaluation before exporting. Existing model downloads alone are not export candidates.";
      renderEvaluations(evaluationId);
    } catch (failure) {
      if (request === view.candidateRequest) { error("error", failure); field("candidates-status").textContent = "Export candidates could not be read. Refresh to try again."; }
    } finally { if (request === view.candidateRequest) { view.loading = false; update(); } }
  }
  function renderHistory() {
    field("history").replaceChildren();
    if (!view.rows.length) field("history").append(new Option("No export packages yet", ""));
    for (const row of view.rows) field("history").append(new Option(`${row.name} · ${row.ready ? "package ready" : row.job?.status || "not ready"} · ${date(row.created_at)}`, row.id));
    if (view.rows.some((row) => row.id === view.selected)) field("history").value = view.selected;
    field("history-empty").hidden = Boolean(view.rows.length);
    update();
  }
  async function refreshHistory() {
    if (view.loadingHistory) return;
    const request = ++view.historyRequest; view.loadingHistory = true; update();
    try {
      const rows = await api("/api/model-exports");
      if (request !== view.historyRequest) return;
      view.rows = rows; error("history-error", null);
      if (!rows.some((row) => row.id === view.selected)) {
        view.selected = rows[0]?.id || null; view.detail = null; invalidateMeasurement(true);
      }
      renderHistory();
      if (view.selected) await loadDetail(view.selected);
      else field("detail").hidden = true;
    } catch (failure) { if (request === view.historyRequest) error("history-error", failure); }
    finally { if (request === view.historyRequest) { view.loadingHistory = false; update(); } }
  }
  function renderMeasurementSummary(container, summary) {
    const presentation = tools.measurementPresentation(summary);
    container.replaceChildren(node("p", "field-hint", presentation.parity), node("p", "field-hint", presentation.evidence));
    if (Number.isInteger(summary?.sample_count)) container.append(node("p", "field-hint", `${summary.frames} images · ${summary.repeats} repeats · ${summary.sample_count} measured predictions`));
    if (summary?.parity_passed === false) container.append(node("p", "field-hint", "One or more outputs differ from the frozen IRIS outputs. Inspect mismatches before using the model in another application."));
    const milliseconds = (value) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("en-US", { maximumFractionDigits: 3 }) : "Unavailable";
    if (summary?.timing_ms) {
      const wrap = node("div", "dataset-class-table-wrap"), table = node("table", "dataset-class-table");
      table.append(node("caption", "", "Declared target-machine timings · milliseconds"));
      const head = node("thead"), headings = node("tr");
      for (const title of ["Stage", "Median", "Min–max"]) { const cell = node("th", "", title); cell.scope = "col"; headings.append(cell); }
      head.append(headings); table.append(head); const body = node("tbody");
      for (const [title, values] of [
        ["Prediction total", summary.timing_ms.total_ms], ["Preprocessing", summary.timing_ms.preprocess_ms],
        ["Model inference", summary.timing_ms.inference_ms], ["Postprocessing", summary.timing_ms.postprocess_ms],
        ["Image decoding (separate)", summary.decode_ms],
      ]) {
        const row = node("tr"), label = node("th", "", title); label.scope = "row";
        row.append(label, node("td", "", milliseconds(values?.median)), node("td", "", `${milliseconds(values?.min)}–${milliseconds(values?.max)}`)); body.append(row);
      }
      table.append(body); wrap.append(table); container.append(wrap);
      container.append(node("p", "field-hint", `Model load: ${milliseconds(summary.load_ms)} ms · excluded warm-up: ${milliseconds(summary.warmup_ms)} ms. Prediction totals exclude image decoding, package verification, loading and warm-up.`));
    }
  }
  function renderDetail(detail) {
    field("detail").hidden = false;
    field("detail-name").textContent = detail.name;
    const contract = detail.manifest || detail.config;
    field("detail-devices").textContent = `Destination: ${deviceLabel(contract?.profile?.device)} · saved reference: ${deviceLabel(contract?.source?.reference_device)}. Training hardware does not restrict this destination.`;
    field("detail-status").textContent = detail.ready ? "Package ready" : (detail.job?.status || "Not ready").replaceAll("_", " ");
    field("detail-status").className = `job-status ${detail.ready ? "succeeded" : detail.job?.status || ""}`;
    field("detail-message").textContent = detail.job?.message || "";
    error("detail-error", detail.job?.error || null);
    field("job").hidden = !detail.job?.id;
    field("ready").hidden = !detail.ready;
    field("download").hidden = !detail.ready;
    if (detail.ready) { field("download").href = projectURL(`${path(detail.id)}/download`); field("download").download = `iris-model-${detail.id}.zip`; }
    else field("download").removeAttribute("href");
    field("manifest").textContent = JSON.stringify({ manifest_sha256: detail.manifest_sha256, archive_sha256: detail.archive_sha256, manifest: detail.manifest }, null, 2);
    const measurements = detail.measurements || [];
    field("validation-status").textContent = measurements.length
      ? `${measurements.length} target-machine report${measurements.length === 1 ? "" : "s"} imported. Each report is checked against this package; imported measurements are not independent proof of execution.`
      : "Target-machine validation has not been recorded. Creating or downloading a package does not verify that it reproduces IRIS outputs.";
    field("measurements").replaceChildren();
    if (!measurements.length) field("measurements").append(node("p", "field-hint", "No imported measurements for this package."));
    for (const measurement of measurements) {
      const item = node("article", "model-export-measurement");
      item.append(node("h5", "", `Imported ${date(measurement.created_at)}`));
      const summary = node("div"); renderMeasurementSummary(summary, measurement.summary); item.append(summary);
      const record = node("details", "dataset-provenance"); record.append(node("summary", "", "Checked parity, timings and declared environment"), node("pre", "", JSON.stringify(measurement.summary, null, 2)));
      const full = node("button", "text-button", "Read full imported record"); full.type = "button";
      full.addEventListener("click", async () => {
        full.disabled = true;
        try {
          const result = await api(`/api/model-export-measurements/${safe(measurement.id)}`);
          record.querySelector("pre").textContent = JSON.stringify(result, null, 2); record.open = true; full.hidden = true;
        } catch (failure) { notify(failure.message, true); } finally { full.disabled = false; }
      });
      item.append(record, full); field("measurements").append(item);
    }
    update();
  }
  async function loadDetail(id) {
    const request = ++view.detailRequest; view.loadingDetail = true; update();
    try {
      const detail = await api(path(id));
      if (request !== view.detailRequest || id !== view.selected) return;
      view.detail = detail; error("history-error", null); renderDetail(detail);
    } catch (failure) {
      if (request === view.detailRequest && id === view.selected) { view.detail = null; field("detail").hidden = true; error("history-error", failure); }
    } finally { if (request === view.detailRequest) { view.loadingDetail = false; update(); } }
  }
  async function preview(event) {
    event.preventDefault(); update(); if (field("preview").disabled || !field("form").reportValidity()) return;
    const payload = options(), key = tools.selectionKey(payload), generation = ++view.generation;
    view.busy = "preview"; view.preview = null; field("preview-result").hidden = true; error("error", null); update();
    try {
      const result = await api("/api/model-exports/preview", { method: "POST", body: JSON.stringify(payload) });
      if (generation !== view.generation || key !== tools.selectionKey(options())) return;
      if (result.plan?.profile?.device !== payload.target_device)
        throw new Error("The export preview does not match the selected target device. Preview the package again.");
      view.preview = { ...result, key, options: payload };
      field("preview-summary").textContent = `${payload.name} · ${selectedModel()?.name || payload.trained_model_id} · ${payload.frame_ids.length} explicitly selected parity images. Full checkpoint weights and saved native predictions are copied with the standalone ${deviceLabel(payload.target_device)} runner. Reference evaluation: ${deviceLabel(result.plan?.source?.reference_device || selectedEvaluation()?.device)}. No model execution is started.`;
      field("preview-warnings").replaceChildren(...(result.warnings || []).map((text) => node("p", "field-hint", text)));
      const model = result.plan?.model, runtime = result.plan?.profile?.runtime;
      if (model?.class_contract?.class_mapping) field("preview-warnings").prepend(node("p", "field-hint", `Frozen class mapping: ${JSON.stringify(model.class_contract.class_mapping)}.`));
      if (runtime) field("preview-warnings").prepend(node("p", "field-hint", `Runtime: PyTorch ${runtime.torch}, Torchvision ${runtime.torchvision}, Pillow ${runtime.pillow} · ${deviceLabel(result.plan?.profile?.device)} float32.`));
      if (model?.sha256) field("preview-warnings").append(node("p", "field-hint", `Checkpoint SHA-256: ${model.sha256}`));
      field("preview-contract").textContent = JSON.stringify(result.plan, null, 2); field("preview-result").hidden = false;
    } catch (failure) { if (generation === view.generation) error("error", failure); }
    finally { view.busy = null; update(); }
  }
  async function createExport() {
    update(); if (field("create").disabled) return;
    const approved = view.preview; view.busy = "create"; error("error", null); update(); let saved;
    try {
      saved = await api("/api/model-exports", { method: "POST", body: JSON.stringify({ ...approved.options, request_id: approved.request_id, expected_fingerprint: approved.fingerprint }) });
    } catch (failure) {
      try { saved = tools.findExport(await api("/api/model-exports"), approved.request_id); } catch { /* Keep the original failure when reconciliation is unavailable. */ }
      if (!saved) error("error", `${failure.message} Refresh export history before preparing another package. The creation request was not repeated automatically.`);
    } finally { view.busy = null; invalidatePreview(); }
    if (saved) {
      view.selected = saved.id; invalidateMeasurement(true); await refreshHistory();
      notify(`Export “${saved.name}” is recorded. Follow packaging progress or cancel in Processing jobs.`);
      refreshJobs().catch((failure) => notify(failure.message, true));
    }
  }
  async function previewMeasurement() {
    update(); if (field("measurement-preview").disabled) return;
    const file = field("measurement-file").files[0], exportId = view.selected;
    const generation = ++view.measurementGeneration; view.measurement = null; view.busy = "measurement-preview";
    field("measurement-review").hidden = true; error("measurement-error", null); update();
    try {
      const problem = tools.fileProblem(file); if (problem) throw new Error(problem);
      const body = await file.text();
      if (new TextEncoder().encode(body).byteLength > tools.MAX_MEASUREMENT_BYTES) throw new Error("The measurement file must be at most 8 MiB.");
      try { JSON.parse(body); } catch { throw new Error("The selected file is not valid JSON."); }
      const result = await api(`${path(exportId)}/measurements/preview`, { method: "POST", body });
      if (generation !== view.measurementGeneration || exportId !== view.selected) return;
      view.measurement = { ...result, exportId, body };
      renderMeasurementSummary(field("measurement-summary"), result.summary);
      field("measurement-record").textContent = JSON.stringify(result.summary, null, 2); field("measurement-review").hidden = false;
    } catch (failure) { if (generation === view.measurementGeneration) error("measurement-error", failure); }
    finally { view.busy = null; update(); }
  }
  async function saveMeasurement() {
    update(); if (field("measurement-save").disabled) return;
    const approved = view.measurement; view.busy = "measurement-save"; error("measurement-error", null); update(); let saved;
    try {
      saved = await api(`${path(approved.exportId)}/measurements?expected_fingerprint=${safe(approved.fingerprint)}`, { method: "POST", body: approved.body });
    } catch (failure) {
      try { saved = tools.findMeasurement((await api(path(approved.exportId))).measurements, approved.fingerprint); } catch { /* Reconciliation never repeats a write. */ }
      if (!saved) error("measurement-error", `${failure.message} Refresh this package and inspect its saved measurements before trying again. No save request was repeated automatically.`);
    } finally {
      view.busy = null; view.measurement = null; field("measurement-review").hidden = true; update();
    }
    if (saved) {
      invalidateMeasurement(true); await loadDetail(approved.exportId);
      notify("Target-machine measurements saved. IRIS checked parity and timing consistency; execution remains declared by the imported report.");
    }
  }
  field("form").addEventListener("submit", preview);
  field("create").addEventListener("click", createExport);
  field("name").addEventListener("input", invalidatePreview);
  field("target-device").addEventListener("change", invalidatePreview);
  field("model").addEventListener("change", () => { view.frames.clear(); invalidatePreview(); renderEvaluations(); });
  field("evaluation").addEventListener("change", () => { view.frames.clear(); invalidatePreview(); renderFrames(); });
  field("history").addEventListener("change", () => {
    view.selected = field("history").value; view.detail = null; field("detail").hidden = true; invalidateMeasurement(true);
    if (view.selected) loadDetail(view.selected);
  });
  field("refresh").addEventListener("click", () => { refreshCandidates(); refreshHistory(); });
  field("measurement-file").addEventListener("change", () => {
    invalidateMeasurement(); const file = field("measurement-file").files?.[0];
    if (file) error("measurement-error", tools.fileProblem(file)); update();
  });
  field("measurement-preview").addEventListener("click", previewMeasurement);
  field("measurement-save").addEventListener("click", saveMeasurement);
  field("job").addEventListener("click", () => {
    if (view.detail?.job?.id) window.dispatchEvent(new CustomEvent("iris:job-open", { detail: { job_id: view.detail.job.id } }));
  });
  window.addEventListener("iris:workspace", (event) => {
    view.visible = event.detail.name === "training";
    if (view.visible) { refreshCandidates(); refreshHistory(); }
    else { invalidatePreview(); invalidateMeasurement(); }
  });
  window.addEventListener("iris:jobs", () => {
    let changed = false, candidatesChanged = false;
    for (const job of state.jobs) {
      if (!["model_export", "train", "evaluate"].includes(job.kind)) continue;
      const previous = view.jobStatuses.get(job.id); view.jobStatuses.set(job.id, job.status);
      if (job.kind === "model_export" && (previous !== job.status || isActive(job))) changed = true;
      if (job.kind !== "model_export" && previous !== job.status && job.status === "succeeded") candidatesChanged = true;
    }
    if (view.visible && !view.busy) { if (changed) refreshHistory(); if (candidatesChanged) refreshCandidates(); }
  });
  window.addEventListener("pagehide", () => { invalidatePreview(); invalidateMeasurement(true); });
  update();
})();
