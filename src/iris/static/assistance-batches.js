"use strict";

(() => {
  const panel = $("#assistance-batches");
  const field = (name) => $(`#assistance-batch-${name}`);
  const batch = {
    active: false, sessionId: null, selected: new Set(), frameKey: "",
    models: [], comparisons: [], history: [], currentId: null, detail: null,
    preview: null, generation: 0, catalogRequest: 0, comparisonRequest: 0,
    historyRequest: 0, detailRequest: 0, catalogLoading: false,
    historyLoading: false, comparisonLoading: false, operation: null, operationToken: 0,
    cancelling: false, cancelToken: 0, loaded: false, comparisonJobs: "",
    retryPreview: null, retryGeneration: 0,
  };
  const sessionURL = (suffix) =>
    `/api/sessions/${encodeURIComponent(batch.sessionId)}/${suffix}`;
  const shown = () => batch.active && panel.open && batch.sessionId;
  const selectedFrames = () => state.frames.filter((frame) => frame.selected);
  const activeBatch = (detail) =>
    Boolean(detail && (detail.counts.queued || detail.counts.running));
  const model = () => batch.models.find((entry) => entry.id === field("model").value);
  const ready = () => model()?.status === "ready";
  const plural = (count, word) =>
    `${count} ${word}${count === 1 ? "" : word.endsWith("box") ? "es" : "s"}`;
  const statusLabel = (status) => String(status || "unknown").replaceAll("_", " ");

  function showError(name, failure) {
    field(name).textContent = failure?.message || failure || "";
    field(name).hidden = !failure;
  }

  function labelFor(frameId, source) {
    const frame = state.frames.find((entry) => entry.id === frameId);
    const filename = source || frame?.source_filename || (frame && sourceFor(frame)?.original_name) ||
      (frame && sourceFor(frame)?.filename) || frameId;
    return frame?.timestamp_seconds == null
      ? filename : `${filename} · ${timestamp(frame.timestamp_seconds)}`;
  }

  function payload() {
    const comparison = field("source").value === "comparison";
    const detector = field("detector").selectedOptions[0];
    return {
      frame_ids: selectedFrames().filter((frame) => batch.selected.has(frame.id))
        .map((frame) => frame.id),
      source: field("source").value,
      comparison_id: comparison ? field("comparison").value : null,
      detector_model_id: comparison ? detector?.dataset.modelId || null : null,
      detector_variant: comparison ? detector?.dataset.variant || null : null,
      model: field("model").value,
      threshold: Number(field("threshold").value),
      instructions: field("instructions").value.trim(),
    };
  }

  function payloadKey() {
    return JSON.stringify(payload());
  }

  function invalidate() {
    batch.generation++;
    batch.preview = null;
    field("preview-result").hidden = true;
    field("preview-frames").replaceChildren();
    showError("error", null);
    updateControls();
  }

  function updateControls() {
    const busy = Boolean(batch.operation);
    const comparison = field("source").value === "comparison";
    const sourceReady = !comparison ||
      Boolean(field("comparison").value && field("detector").value);
    const count = batch.selected.size;
    field("selection-count").textContent = `${count} chosen · maximum 25`;
    field("select-all").disabled = busy || selectedFrames().length > 25 || !selectedFrames().length;
    field("select-all").title = selectedFrames().length > 25
      ? "Choose up to 25 frames individually." : "Choose all selected frames in this session.";
    field("clear").disabled = busy || !count;
    for (const input of field("frames").querySelectorAll("input"))
      input.disabled = busy || (!input.checked && count >= 25);
    for (const name of ["name", "source", "threshold", "instructions"])
      field(name).disabled = busy;
    field("model").disabled = busy || batch.catalogLoading || !batch.models.length;
    field("refresh-models").disabled = busy || batch.catalogLoading;
    field("comparison").disabled = busy || batch.comparisonLoading || !batch.comparisons.length;
    field("detector").disabled = busy || batch.comparisonLoading || !field("detector").value;
    field("comparison-settings").hidden = !comparison;
    field("preview").disabled = busy || !batch.sessionId || !count || count > 25 ||
      !ready() || !sourceReady || batch.catalogLoading || batch.comparisonLoading;
    field("preview").textContent = batch.operation === "preview"
      ? "Checking frames…" : "Check eligible frames";
    const preview = batch.preview;
    field("start").disabled = busy || !ready() || !preview?.eligible_count ||
      preview.key !== payloadKey() || !field("name").value.trim();
    field("start").textContent = batch.operation === "start"
      ? "Starting batch…" : preview?.eligible_count
        ? `Start ${plural(preview.eligible_count, "frame")}` : "Start local review";
    field("refresh-history").disabled = batch.historyLoading;
    field("history").disabled = busy || !batch.history.length;
    field("retry-preview").disabled = busy || batch.cancelling || !window.IRISJobTools.retryableBatchFrames(batch.detail).length;
    field("retry-preview").textContent = batch.operation === "retry-preview" ? "Checking unfinished frames…" : "Check unfinished frames";
    field("retry-name").disabled = busy;
    field("retry-confirm").disabled = busy || batch.cancelling || !batch.retryPreview?.eligible_count || !field("retry-name").value.trim();
    field("retry-confirm").textContent = batch.operation === "retry" ? "Creating linked batch…" : "Start checked frames in a new batch";
    field("cancel").disabled = batch.cancelling || Boolean(batch.detail?.cancel_requested);
  }

  function renderFrames() {
    const frames = selectedFrames();
    const ids = new Set(frames.map((frame) => frame.id));
    for (const id of batch.selected) if (!ids.has(id)) batch.selected.delete(id);
    field("frames").replaceChildren();
    if (!frames.length)
      field("frames").append(node("p", "field-hint", "Select frames in Data intake to prepare a batch."));
    for (const frame of frames) {
      const label = node("label", "assistance-batch-frame");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = frame.id;
      input.checked = batch.selected.has(frame.id);
      input.addEventListener("change", () => {
        if (input.checked) batch.selected.add(frame.id);
        else batch.selected.delete(frame.id);
        invalidate();
      });
      label.append(input, node("span", "", labelFor(frame.id)));
      field("frames").append(label);
    }
    updateControls();
  }

  function renderProvider() {
    const entry = model();
    field("provider-status").textContent = batch.catalogLoading
      ? "Checking local models…" : !entry ? "No local vision model is available."
        : entry.status === "ready" ? `Ready locally · ${entry.id}`
          : `Unavailable · ${entry.reason || statusLabel(entry.status)}. Refresh after starting Ollama or installing the model.`;
    updateControls();
  }

  async function loadModels() {
    if (batch.catalogLoading || batch.operation) return;
    const request = ++batch.catalogRequest;
    const previous = field("model").value;
    batch.catalogLoading = true;
    invalidate();
    renderProvider();
    try {
      const catalog = await api("/api/annotation-providers");
      if (request !== batch.catalogRequest) return;
      batch.models = (catalog.providers || []).find(
        (provider) => provider.id === "ollama" && provider.local === true,
      )?.models || [];
      field("model").replaceChildren();
      for (const entry of batch.models)
        field("model").append(new Option(entry.label || entry.id, entry.id));
      if (!batch.models.length) field("model").append(new Option("No local models", ""));
      const preferred = previous || catalog.default_model;
      if (batch.models.some((entry) => entry.id === preferred)) field("model").value = preferred;
    } catch (failure) {
      if (request !== batch.catalogRequest) return;
      batch.models = [];
      field("model").replaceChildren(new Option("Local models unavailable", ""));
      showError("error", failure);
    } finally {
      if (request === batch.catalogRequest) {
        batch.catalogLoading = false;
        renderProvider();
      }
    }
  }

  function renderDetectors() {
    const previous = field("detector").value;
    const comparison = batch.comparisons.find((entry) => entry.id === field("comparison").value);
    const lanes = comparison?.lanes || (comparison?.model_ids || []).map((model_id) => ({ model_id, variant: "full" }));
    field("detector").replaceChildren();
    for (const lane of lanes) {
      const option = new Option(`${lane.model_id} · ${lane.variant === "tiled" ? "Tiled" : "Full image"}`, JSON.stringify([lane.model_id, lane.variant]));
      option.dataset.modelId = lane.model_id;
      option.dataset.variant = lane.variant;
      field("detector").append(option);
    }
    if (!lanes.length)
      field("detector").append(new Option("Choose a comparison", ""));
    if ([...field("detector").options].some((option) => option.value === previous)) field("detector").value = previous;
    updateControls();
  }

  async function loadComparisons() {
    if (!batch.sessionId || batch.comparisonLoading) return;
    const request = ++batch.comparisonRequest;
    const sessionId = batch.sessionId;
    batch.comparisonLoading = true;
    updateControls();
    try {
      const history = await api(sessionURL("comparisons"));
      if (request !== batch.comparisonRequest || sessionId !== batch.sessionId) return;
      const previousKey = payloadKey();
      const previous = field("comparison").value;
      batch.comparisons = history.filter((entry) => entry.job?.status === "succeeded");
      field("comparison").replaceChildren();
      for (const entry of batch.comparisons)
        field("comparison").append(new Option(entry.name, entry.id));
      if (!batch.comparisons.length)
        field("comparison").append(new Option("No successful comparisons", ""));
      if (batch.comparisons.some((entry) => entry.id === previous)) field("comparison").value = previous;
      renderDetectors();
      if (previousKey !== payloadKey()) invalidate();
    } catch (failure) {
      if (request === batch.comparisonRequest && sessionId === batch.sessionId)
        showError("error", failure);
    } finally {
      if (request === batch.comparisonRequest) {
        batch.comparisonLoading = false;
        updateControls();
      }
    }
  }

  function guardFrames(ids) {
    return window.dispatchEvent(new CustomEvent("iris:before-assistance-batch", {
      cancelable: true, detail: { frame_ids: ids },
    }));
  }

  function validForm() {
    return ["name", "threshold", "instructions"].every((name) => field(name).reportValidity());
  }

  function renderPreview() {
    const preview = batch.preview;
    field("preview-result").hidden = !preview;
    field("preview-frames").replaceChildren();
    if (!preview) return;
    field("preview-summary").textContent =
      `${plural(preview.eligible_count, "eligible frame")} · ${preview.excluded_count} skipped · ${plural(preview.candidate_count, "candidate box")}. No inference has run.`;
    for (const frame of preview.frames) {
      const row = node("li");
      row.append(node("strong", "", labelFor(frame.frame_id, frame.source_filename)));
      row.append(node("p", "muted", frame.eligible
        ? `Eligible · ${plural(frame.candidate_count, "candidate box")} · saved revision ${frame.base_revision}`
        : `Skipped · ${frame.reason || "This frame is not eligible."}`));
      field("preview-frames").append(row);
    }
  }

  async function previewBatch() {
    if (field("preview").disabled || !validForm()) return;
    const settings = payload();
    if (!guardFrames(settings.frame_ids)) return;
    invalidate();
    const generation = batch.generation;
    const sessionId = batch.sessionId;
    const key = payloadKey();
    const operationToken = ++batch.operationToken;
    batch.operation = "preview";
    updateControls();
    try {
      const preview = await api(sessionURL("assistance-batches/preview"), {
        method: "POST", body: JSON.stringify(settings),
      });
      if (sessionId !== batch.sessionId || generation !== batch.generation || key !== payloadKey()) return;
      batch.preview = { ...preview, key };
      renderPreview();
    } catch (failure) {
      if (sessionId === batch.sessionId && generation === batch.generation) showError("error", failure);
    } finally {
      if (operationToken === batch.operationToken) {
        batch.operation = null;
        updateControls();
      }
    }
  }

  async function startBatch() {
    if (field("start").disabled || !validForm()) return;
    const settings = payload();
    if (!guardFrames(settings.frame_ids)) return;
    const sessionId = batch.sessionId;
    const preview = batch.preview;
    const operationToken = ++batch.operationToken;
    batch.historyRequest++;
    batch.detailRequest++;
    batch.historyLoading = false;
    batch.operation = "start";
    showError("error", null);
    updateControls();
    try {
      const detail = await api(sessionURL("assistance-batches"), {
        method: "POST",
        body: JSON.stringify({ ...settings, name: field("name").value.trim(), expected_fingerprint: preview.fingerprint }),
      });
      if (sessionId !== batch.sessionId || operationToken !== batch.operationToken) return;
      batch.currentId = detail.id;
      batch.detail = detail;
      batch.history = [detail, ...batch.history.filter((entry) => entry.id !== detail.id)];
      invalidate();
      renderHistory();
      renderDetail();
      notify("Local review queued. Proposals require human review and validation.");
      refreshJobs().catch((failure) => showError("history-error", failure));
    } catch (failure) {
      if (sessionId !== batch.sessionId || operationToken !== batch.operationToken) return;
      // Never repeat a start automatically: a lost response can hide a queued batch.
      invalidate();
      showError("error", `${failure.message} Refresh batch history before starting again.`);
      loadHistory();
    } finally {
      if (operationToken === batch.operationToken) {
        batch.operation = null;
        updateControls();
        loadHistory();
      }
    }
  }

  function clearRetryPreview() {
    batch.retryGeneration++;
    batch.retryPreview = null;
    field("retry-result").hidden = true;
    field("retry-frames").replaceChildren();
    showError("retry-error", null);
  }

  async function previewRetry() {
    const source = batch.detail;
    const candidates = window.IRISJobTools.retryableBatchFrames(source);
    if (field("retry-preview").disabled || !candidates.length || !guardFrames(candidates.map((frame) => frame.frame_id))) return;
    clearRetryPreview();
    const generation = batch.retryGeneration;
    const sessionId = batch.sessionId;
    const operationToken = ++batch.operationToken;
    batch.operation = "retry-preview";
    updateControls();
    try {
      const preview = await api(`/api/assistance-batches/${encodeURIComponent(source.id)}/retry-preview`, { method: "POST", body: "{}" });
      if (sessionId !== batch.sessionId || generation !== batch.retryGeneration || source.id !== batch.currentId) return;
      if (preview.retry_of !== source.id) throw new Error("The retry preview does not match this saved batch. Refresh its history.");
      batch.retryPreview = preview;
      field("retry-name").value = `${source.name.slice(0, 140)} · retry`;
      field("retry-result").hidden = false;
      field("retry-summary").textContent = `${preview.eligible_count} frames eligible for a new batch · ${preview.retained_count ?? 0} previous frame results retained. ${preview.reason || "Current saved reviews were checked; no inference has run."}`;
      for (const frame of preview.frames || []) {
        const row = node("li");
        row.append(node("strong", "", labelFor(frame.frame_id, frame.source_filename)), node("p", "field-hint", frame.eligible ? `Eligible · ${plural(frame.candidate_count, "candidate box")} · current saved revision ${frame.base_revision}` : `Excluded · ${frame.reason || "No longer eligible"}`));
        field("retry-frames").append(row);
      }
    } catch (failure) {
      if (sessionId === batch.sessionId && generation === batch.retryGeneration) showError("retry-error", failure);
    } finally {
      if (operationToken === batch.operationToken) { batch.operation = null; updateControls(); }
    }
  }

  async function confirmRetry() {
    const preview = batch.retryPreview;
    const source = batch.detail;
    if (field("retry-confirm").disabled || !preview || !field("retry-name").reportValidity()) return;
    const frames = (preview.frames || []).filter((frame) => frame.eligible).map((frame) => frame.frame_id);
    if (!guardFrames(frames)) return;
    const sessionId = batch.sessionId;
    const operationToken = ++batch.operationToken;
    batch.operation = "retry";
    batch.historyRequest++;
    batch.detailRequest++;
    batch.historyLoading = false;
    updateControls();
    showError("retry-error", null);
    try {
      const detail = await api(`/api/assistance-batches/${encodeURIComponent(source.id)}/retry`, {
        method: "POST", body: JSON.stringify({ name: field("retry-name").value.trim(), expected_fingerprint: preview.fingerprint }),
      });
      if (sessionId !== batch.sessionId || operationToken !== batch.operationToken) return;
      clearRetryPreview();
      batch.currentId = detail.id;
      batch.detail = detail;
      batch.history = [detail, ...batch.history.filter((entry) => entry.id !== detail.id)];
      renderHistory();
      renderDetail();
      notify("A new linked local batch was queued for the checked frames. Previous results and proposals were retained.");
      await refreshJobs();
    } catch (failure) {
      if (sessionId !== batch.sessionId || operationToken !== batch.operationToken) return;
      clearRetryPreview();
      showError("retry-error", `${failure.message} Refresh batch history before preparing another retry. No request was repeated automatically.`);
    } finally {
      if (operationToken === batch.operationToken) {
        batch.operation = null;
        updateControls();
        loadHistory();
      }
    }
  }

  function renderHistory() {
    const select = field("history");
    select.replaceChildren();
    for (const detail of batch.history)
      select.append(new Option(`${detail.name} · ${statusLabel(detail.status)} · ${new Date(detail.created_at).toLocaleString()}`, detail.id));
    if (!batch.history.length) select.append(new Option("No batches in this session", ""));
    select.value = batch.currentId || "";
    updateControls();
  }

  function renderDetail() {
    const detail = batch.detail;
    field("detail").hidden = !detail;
    field("detail-frames").replaceChildren();
    field("excluded-frames").replaceChildren();
    if (!detail) return;
    field("detail-summary").textContent =
      `${statusLabel(detail.status)} · ${detail.finished_count}/${detail.counts.total} frames finished · ${plural(detail.suggestions_created, "proposal")} created`;
    field("progress").max = 1;
    field("progress").value = Math.min(1, Math.max(0, detail.progress || 0));
    const config = detail.config || {};
    field("state-counts").textContent = Object.entries(detail.counts).filter(([key, count]) => key !== "total" && count > 0).map(([key, count]) => `${count} ${statusLabel(key)}`).join(" · ");
    field("lineage").textContent = detail.retry_of || config.retry_of ? `New batch linked to ${detail.retry_of || config.retry_of}. Earlier results were retained.` : "";
    field("retry-preview").hidden = !window.IRISJobTools.retryableBatchFrames(detail).length;
    const excluded = config.excluded?.length || 0;
    field("excluded").hidden = !excluded;
    field("excluded-summary").textContent = `${plural(excluded, "frame")} skipped at preparation`;
    for (const frame of config.excluded || []) {
      const row = node("li");
      row.append(node("strong", "", labelFor(frame.frame_id, frame.source_filename)),
        node("p", "muted", frame.reason || "This frame was not eligible."));
      field("excluded-frames").append(row);
    }
    field("detail-config").textContent =
      `${config.model || "Local model"} · ${config.source === "comparison" ? `saved comparison · ${config.detector_model_id} · ${config.detector_variant === "tiled" ? "Tiled" : "Full image"}` : "saved annotations"}${excluded ? ` · ${excluded} skipped at preparation` : ""}. Results remain proposals until you review them.`;
    field("cancel").hidden = !activeBatch(detail);
    field("cancel-hint").hidden = !detail.cancel_requested;
    field("cancel-hint").textContent = "Cancellation requested. An active model request may need to finish before it stops; already saved proposals are kept.";
    for (const frame of detail.frames) {
      const row = node("li");
      const heading = node("div", "assistance-batch-result-heading");
      heading.append(node("span", "", labelFor(frame.frame_id)));
      const button = node("button", "text-button", "Review frame");
      button.type = "button";
      button.disabled = !state.frames.some((entry) => entry.id === frame.frame_id && entry.selected);
      button.addEventListener("click", () => window.dispatchEvent(new CustomEvent(
        "iris:annotation-open-frame", { detail: { frame_id: frame.frame_id } },
      )));
      heading.append(button);
      if (frame.job_id) {
        const job = node("button", "text-button", "Job details");
        job.type = "button";
        job.addEventListener("click", () => window.dispatchEvent(new CustomEvent("iris:job-open", { detail: { job_id: frame.job_id } })));
        heading.append(job);
      }
      row.append(heading, node("p", "muted", `${statusLabel(frame.status)} · ${plural(frame.suggestions_created || 0, "proposal")}${frame.message ? ` · ${frame.message}` : ""}`));
      if (["queued", "running"].includes(frame.status)) {
        const progress = document.createElement("progress");
        progress.max = 1;
        progress.value = Math.min(1, Math.max(0, frame.progress || 0));
        progress.setAttribute("aria-label", `Progress for ${labelFor(frame.frame_id)}`);
        row.append(progress);
      }
      if (frame.error) row.append(node("p", "inline-error", frame.error));
      field("detail-frames").append(row);
    }
    updateControls();
  }

  async function loadHistory() {
    if (!batch.sessionId || batch.historyLoading || batch.operation || batch.cancelling) return;
    const request = ++batch.historyRequest;
    batch.detailRequest++;
    const sessionId = batch.sessionId;
    batch.historyLoading = true;
    updateControls();
    try {
      const history = await api(sessionURL("assistance-batches"));
      if (sessionId !== batch.sessionId || request !== batch.historyRequest) return;
      batch.history = history;
      if (!history.some((detail) => detail.id === batch.currentId)) batch.currentId = history[0]?.id || null;
      batch.detail = history.find((detail) => detail.id === batch.currentId) || null;
      renderHistory();
      renderDetail();
      showError("history-error", null);
    } catch (failure) {
      if (sessionId === batch.sessionId && request === batch.historyRequest) showError("history-error", failure);
    } finally {
      if (request === batch.historyRequest) {
        batch.historyLoading = false;
        updateControls();
      }
    }
  }

  async function loadDetail(id) {
    const request = ++batch.detailRequest;
    const sessionId = batch.sessionId;
    try {
      const detail = await api(`/api/assistance-batches/${encodeURIComponent(id)}`);
      if (sessionId !== batch.sessionId || request !== batch.detailRequest || id !== batch.currentId) return;
      batch.detail = detail;
      renderDetail();
      showError("history-error", null);
    } catch (failure) {
      if (sessionId === batch.sessionId && request === batch.detailRequest) showError("history-error", failure);
    }
  }

  async function cancelBatch() {
    const detail = batch.detail;
    if (!activeBatch(detail) || detail.cancel_requested || batch.cancelling) return;
    const sessionId = batch.sessionId;
    const request = ++batch.detailRequest;
    batch.historyRequest++;
    batch.historyLoading = false;
    const cancelToken = ++batch.cancelToken;
    batch.cancelling = true;
    updateControls();
    try {
      const result = await api(`/api/assistance-batches/${encodeURIComponent(detail.id)}/cancel`, {
        method: "POST", body: "{}",
      });
      if (sessionId !== batch.sessionId || request !== batch.detailRequest || batch.currentId !== result.id) return;
      batch.detail = result;
      renderDetail();
      showError("history-error", null);
      refreshJobs().catch((failure) => showError("history-error", failure));
    } catch (failure) {
      if (sessionId === batch.sessionId && request === batch.detailRequest) showError("history-error", failure);
    } finally {
      if (cancelToken === batch.cancelToken) {
        batch.cancelling = false;
        updateControls();
        loadHistory();
      }
    }
  }

  function openPanel() {
    if (!shown()) return;
    renderFrames();
    if (!batch.loaded) {
      batch.loaded = true;
      loadModels();
      loadComparisons();
    }
    loadHistory();
  }

  field("select-all").addEventListener("click", () => {
    if (field("select-all").disabled) return;
    batch.selected = new Set(selectedFrames().map((frame) => frame.id));
    invalidate();
    renderFrames();
  });
  field("clear").addEventListener("click", () => {
    batch.selected.clear();
    invalidate();
    renderFrames();
  });
  for (const name of ["model", "source", "comparison", "detector", "threshold", "instructions", "name"])
    field(name).addEventListener(["threshold", "instructions", "name"].includes(name) ? "input" : "change", () => {
      invalidate();
      if (name === "model") renderProvider();
      if (name === "comparison") renderDetectors();
      if (name === "source" && field("source").value === "comparison") loadComparisons();
    });
  field("refresh-models").addEventListener("click", loadModels);
  field("preview").addEventListener("click", previewBatch);
  field("start").addEventListener("click", startBatch);
  field("refresh-history").addEventListener("click", () => {
    loadHistory();
    loadComparisons();
  });
  field("cancel").addEventListener("click", cancelBatch);
  field("retry-preview").addEventListener("click", previewRetry);
  field("retry-confirm").addEventListener("click", confirmRetry);
  field("retry-name").addEventListener("input", updateControls);
  field("history").addEventListener("change", () => {
    clearRetryPreview();
    batch.historyRequest++;
    batch.historyLoading = false;
    batch.currentId = field("history").value || null;
    batch.detail = batch.history.find((detail) => detail.id === batch.currentId) || null;
    renderDetail();
    if (batch.currentId) loadDetail(batch.currentId);
  });
  panel.addEventListener("toggle", () => {
    if (!panel.open) { invalidate(); clearRetryPreview(); }
    else openPanel();
  });
  window.addEventListener("iris:workspace", (event) => {
    batch.active = event.detail.name === "annotation";
    if (!batch.active) { invalidate(); clearRetryPreview(); }
    openPanel();
  });
  window.addEventListener("iris:session", () => {
    if (batch.sessionId === state.sessionId) return;
    batch.sessionId = state.sessionId;
    clearRetryPreview();
    batch.selected.clear();
    batch.frameKey = "";
    batch.comparisonRequest++;
    batch.historyRequest++;
    batch.detailRequest++;
    batch.catalogRequest++;
    batch.operation = null;
    batch.operationToken++;
    batch.cancelling = false;
    batch.cancelToken++;
    batch.loaded = false;
    batch.comparisonJobs = "";
    batch.comparisonLoading = false;
    batch.catalogLoading = false;
    batch.historyLoading = false;
    batch.comparisons = [];
    batch.history = [];
    batch.currentId = null;
    batch.detail = null;
    field("comparison").replaceChildren(new Option("No successful comparisons", ""));
    renderDetectors();
    invalidate();
    showError("history-error", null);
    renderFrames();
    renderHistory();
    renderDetail();
    openPanel();
  });
  window.addEventListener("iris:frames", () => {
    if (batch.sessionId !== state.sessionId) return;
    const key = JSON.stringify(selectedFrames().map((frame) => frame.id));
    if (key !== batch.frameKey) {
      batch.frameKey = key;
      invalidate();
      renderFrames();
    }
  });
  window.addEventListener("iris:jobs", () => {
    if (!shown()) return;
    loadHistory();
    const key = JSON.stringify(state.jobs.filter((job) => job.kind === "infer")
      .map((job) => [job.id, job.status]));
    if (key !== batch.comparisonJobs) {
      batch.comparisonJobs = key;
      loadComparisons();
    }
  });
  window.addEventListener("iris:assistance-batch-open", (event) => {
    if (!event.detail?.batch_id || !batch.active) return;
    clearRetryPreview();
    batch.currentId = event.detail.batch_id;
    panel.open = true;
    openPanel();
    loadDetail(batch.currentId);
    field("history").scrollIntoView({ block: "nearest" });
  });
  updateControls();
})();
