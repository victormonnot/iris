"use strict";

(() => {
  const panel = $("#preannotations");
  const field = (name) => $(`#preannotation-${name}`);
  const tools = window.IRISPreannotationTools;
  const view = {
    active: false, sessionId: state.sessionId, selected: new Set(), context: null,
    models: [], catalogLoading: false, catalogRequest: 0, loaded: false,
    preview: null, generation: 0, operation: null, operationToken: 0,
    history: [], currentId: null, detail: null, historyLoading: false,
    historyRequest: 0, detailRequest: 0, detailLoading: false,
    framesKey: "", contextKey: "", jobsKey: "",
  };
  const selectedFrames = () => state.frames.filter((frame) => frame.selected);
  const shown = () => view.active && panel.open && view.sessionId;
  const sessionURL = () => `/api/sessions/${encodeURIComponent(view.sessionId)}/preannotations`;
  const model = () => view.models.find((entry) => entry.id === field("model").value);
  const ready = () => model()?.status === "ready";
  const frameLabel = (id, filename) => {
    const frame = state.frames.find((entry) => entry.id === id);
    const source = filename || (frame && sourceFor(frame)?.filename) || frame?.source_filename || id;
    return frame?.timestamp_seconds == null ? source : `${source} · ${timestamp(frame.timestamp_seconds)}`;
  };
  const emitUpdate = () => window.dispatchEvent(new CustomEvent("iris:preannotations-updated"));

  function error(name, failure) {
    field(name).textContent = failure?.message || failure || "";
    field(name).hidden = !failure;
  }

  function payload() {
    const available = new Set(selectedFrames().map((frame) => frame.id));
    const current = view.context?.session_id === view.sessionId && view.context.frame_id;
    const tiled = field("mode").value === "tiled";
    return {
      frame_ids: field("scope").value === "current"
        ? current && available.has(current) ? [current] : []
        : selectedFrames().filter((frame) => view.selected.has(frame.id)).map((frame) => frame.id),
      model_id: field("model").value,
      threshold: Number(field("threshold").value),
      inference_mode: field("mode").value,
      tile_size: tiled ? Number(field("tile-size").value) : 640,
      overlap: tiled ? Number(field("overlap").value) : 0.2,
      device: field("device").value,
    };
  }

  const payloadKey = () => JSON.stringify(payload());
  const contextBlocked = () => payload().frame_ids.includes(view.context?.frame_id) &&
    (view.context.dirty || view.context.busy);

  function invalidate() {
    view.generation++;
    view.preview = null;
    field("preview-result").hidden = true;
    error("error", null);
    updateControls();
  }

  function updateControls() {
    const busy = Boolean(view.operation);
    const batch = field("scope").value === "batch";
    const options = payload();
    const count = view.selected.size;
    field("batch").hidden = !batch;
    field("current").hidden = batch;
    field("current").textContent = options.frame_ids.length && !batch
      ? `${frameLabel(options.frame_ids[0])}${contextBlocked() ? " · save or discard edits and finish the current action first" : ""}`
      : "Open a selected frame in the review queue first.";
    field("selection-count").textContent = `${count} chosen · maximum 25`;
    field("select-all").disabled = busy || !selectedFrames().length || selectedFrames().length > 25;
    field("select-all").title = selectedFrames().length > 25 ? "Choose up to 25 frames individually." : "Choose every selected frame in this session.";
    field("clear").disabled = busy || !count;
    for (const input of field("frames").querySelectorAll("input")) input.disabled = busy || (!input.checked && count >= 25);
    for (const name of ["scope", "name", "threshold", "mode", "device", "tile-size", "overlap"]) field(name).disabled = busy;
    field("tiled").hidden = options.inference_mode !== "tiled";
    field("model").disabled = busy || view.catalogLoading || !view.models.length;
    field("refresh-models").disabled = busy || view.catalogLoading;
    field("preview").disabled = busy || view.catalogLoading || !view.sessionId || !ready() ||
      !options.frame_ids.length || options.frame_ids.length > 25 || contextBlocked();
    field("preview").textContent = view.operation === "preview" ? "Checking saved inputs…" : "Preview proposals run";
    field("start").disabled = busy || !ready() || contextBlocked() || !view.preview?.eligible_count ||
      view.preview.key !== payloadKey() || !field("name").value.trim();
    field("start").textContent = view.operation === "start" ? "Creating proposal run…" : "Create proposals";
    field("refresh-history").disabled = view.historyLoading || busy;
    field("history").disabled = busy || !view.history.length;
    field("job").disabled = !view.detail?.job_id && !view.detail?.job?.id;
    field("raw").disabled = !view.detail || view.detailLoading || !Object.hasOwn(view.detail, "predictions");
  }

  function renderFrames() {
    const ids = new Set(selectedFrames().map((frame) => frame.id));
    for (const id of view.selected) if (!ids.has(id)) view.selected.delete(id);
    field("frames").replaceChildren();
    if (!ids.size) field("frames").append(node("p", "field-hint", "Select frames in Data intake to prepare a batch."));
    for (const frame of selectedFrames()) {
      const label = node("label", "assistance-batch-frame");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = frame.id;
      input.checked = view.selected.has(frame.id);
      input.addEventListener("change", () => {
        if (input.checked) view.selected.add(frame.id);
        else view.selected.delete(frame.id);
        invalidate();
      });
      label.append(input, node("span", "", frameLabel(frame.id)));
      field("frames").append(label);
    }
    updateControls();
  }

  function renderModel() {
    const entry = model();
    field("model-status").textContent = view.catalogLoading ? "Checking available detectors…"
      : !entry ? "No detector is ready. The manual editor remains fully available."
        : entry.status !== "ready" ? `${entry.reason || entry.status}. No download is started here.`
          : entry.origin === "trained"
            ? `Ready locally · ${window.IRISTaxonomyTools.versionLabel(entry.taxonomy)}. Preview checks the exact saved definitions on each frame.`
            : "Ready locally · only classes with explicit mappings to this detector can be proposed. Preview lists any unsupported classes.";
    updateControls();
  }

  async function loadModels() {
    if (view.catalogLoading || view.operation || !view.sessionId) return;
    const request = ++view.catalogRequest;
    const old = field("model").value;
    view.catalogLoading = true;
    invalidate();
    renderModel();
    try {
      const catalog = await api("/api/preannotation-providers");
      if (request !== view.catalogRequest) return;
      view.models = (catalog.providers || []).find((provider) => provider.id === "local_detector")?.models || [];
      field("model").replaceChildren();
      for (const entry of view.models) {
        const option = new Option(`${entry.name || entry.label || entry.id}${entry.status === "ready" ? "" : " · unavailable"}`, entry.id);
        option.disabled = entry.status !== "ready";
        option.title = entry.reason || entry.description || "";
        field("model").append(option);
      }
      const preferred = view.models.find((entry) => entry.id === old && entry.status === "ready") || view.models.find((entry) => entry.status === "ready");
      if (preferred) field("model").value = preferred.id;
      else { field("model").prepend(new Option("No ready detector", "")); field("model").value = ""; }
      field("capabilities").replaceChildren();
      for (const provider of catalog.providers || []) {
        const capability = provider.capabilities || {};
        const text = capability.creates_boxes
          ? "Generates new boxes. Matching trained classes or explicit official mappings; compatibility is checked per frame."
          : "Reviews existing candidate boxes only; does not discover new objects. Current review supports the original person / car definitions.";
        field("capabilities").append(node("p", "field-hint", `${provider.label || provider.id} · ${capability.execution === "external" ? "external, separate explicit consent" : "local"}. ${text}`));
      }
    } catch (failure) {
      if (request !== view.catalogRequest) return;
      view.models = [];
      field("model").replaceChildren(new Option("Detectors unavailable", ""));
      error("error", failure);
    } finally {
      if (request === view.catalogRequest) { view.catalogLoading = false; renderModel(); }
    }
  }

  function guardFrames(ids) {
    return window.dispatchEvent(new CustomEvent("iris:before-assistance-batch", {
      cancelable: true, detail: { frame_ids: ids },
    }));
  }

  function settingsValid() {
    const fields = ["threshold", ...(field("mode").value === "tiled" ? ["tile-size", "overlap"] : [])];
    return fields.every((name) => {
      const input = field(name);
      if (!input.value.trim()) { input.setCustomValidity("Enter a number."); input.reportValidity(); return false; }
      input.setCustomValidity("");
      return input.reportValidity();
    });
  }

  function renderPreview() {
    const preview = view.preview;
    field("preview-result").hidden = !preview;
    if (!preview) return;
    field("preview-summary").textContent = `${preview.eligible_count} eligible · ${preview.excluded_count} excluded. Only eligible frames will run; no inference has started.`;
    const options = payload();
    const passes = Number.isInteger(preview.work?.total_forward_passes) ? ` · ${preview.work.total_forward_passes} detector passes including warm-up` : "";
    field("preview-work").textContent = `${model()?.name || options.model_id} · ${options.inference_mode === "tiled" ? `${options.tile_size}px tiles, overlap ${options.overlap}` : "full image"} · ${options.device.toUpperCase()} · proposal score ≥ ${options.threshold}${passes}. Saved revisions, image hashes and class coverage are checked again when creating the run.`;
    field("preview-frames").replaceChildren();
    for (const frame of preview.frames || []) {
      const row = node("li", frame.eligible ? "eligible" : "excluded");
      row.append(node("strong", "", frameLabel(frame.frame_id)), node("p", "field-hint", frame.eligible ? `Eligible · saved revision ${frame.base_revision}` : `Excluded · ${frame.reason || "not eligible"}`));
      if (frame.coverage) {
        const supported = frame.coverage.supported_class_ids || [];
        const unsupported = frame.coverage.unsupported_class_ids || [];
        row.append(node("p", "field-hint", `Supported class IDs: ${supported.join(", ") || "none"}.${unsupported.length ? ` Not covered: ${unsupported.join(", ")}. Inspect these classes manually.` : " All frame classes are covered; detections can still miss targets."}`));
      }
      field("preview-frames").append(row);
    }
    field("preview-warnings").replaceChildren();
    for (const warning of preview.warnings || []) field("preview-warnings").append(node("li", "", warning));
    updateControls();
  }

  async function previewRun() {
    if (field("preview").disabled || !settingsValid()) return;
    const options = payload();
    if (!guardFrames(options.frame_ids)) return;
    invalidate();
    const generation = view.generation;
    const key = payloadKey();
    const session = view.sessionId;
    const token = ++view.operationToken;
    view.operation = "preview";
    updateControls();
    try {
      const result = await api(`${sessionURL()}/preview`, { method: "POST", body: JSON.stringify(options) });
      if (session !== view.sessionId || generation !== view.generation || key !== payloadKey()) return;
      view.preview = { ...result, key };
      renderPreview();
    } catch (failure) {
      if (session === view.sessionId && generation === view.generation) error("error", failure);
    } finally {
      if (token === view.operationToken) { view.operation = null; updateControls(); }
    }
  }

  function retainRun(detail) {
    view.currentId = detail.id;
    view.detail = detail;
    view.historyRequest++;
    view.historyLoading = false;
    view.detailRequest++;
    view.detailLoading = false;
    renderDetail();
    loadHistory();
    refreshJobs().catch((failure) => error("history-error", failure));
    emitUpdate();
  }

  async function startRun() {
    if (field("start").disabled || !view.preview || !settingsValid()) return;
    const options = payload();
    if (!guardFrames(options.frame_ids)) return;
    const fingerprint = view.preview.fingerprint;
    const session = view.sessionId;
    const path = sessionURL();
    const token = ++view.operationToken;
    view.operation = "start";
    error("error", null);
    updateControls();
    try {
      const detail = await api(path, { method: "POST", body: JSON.stringify({ ...options, name: field("name").value.trim(), expected_fingerprint: fingerprint }) });
      if (session !== view.sessionId) return;
      invalidate();
      retainRun(detail);
      notify("Local proposal run queued. Review every proposal and the whole image before human validation.");
    } catch (failure) {
      if (session !== view.sessionId) return;
      let receipt = null;
      try { receipt = tools.findReceipt(await api(path), fingerprint); } catch { /* Keep the original error; never resend a POST. */ }
      if (session !== view.sessionId) return;
      invalidate();
      if (receipt) {
        retainRun(receipt);
        loadDetail(receipt.id);
        notify("The proposal run was already recorded. Its saved history is open; no second request was sent.");
      } else error("error", `${failure.message} Check Saved proposal runs and Project jobs before preparing another run. No request was repeated automatically.`);
    } finally {
      if (token === view.operationToken) { view.operation = null; updateControls(); }
    }
  }

  function renderHistory() {
    const selector = field("history");
    selector.replaceChildren(new Option("Choose a saved proposal run", ""));
    for (const entry of view.history) selector.append(new Option(`${entry.name} · ${entry.job?.status || "saved"} · ${new Date(entry.created_at).toLocaleString()}`, entry.id));
    if (!view.history.length) selector.replaceChildren(new Option("No proposal runs in this session", ""));
    selector.value = view.currentId || "";
    updateControls();
  }

  async function loadHistory() {
    if (!view.sessionId || view.historyLoading) return;
    const request = ++view.historyRequest;
    const session = view.sessionId;
    view.historyLoading = true;
    error("history-error", null);
    updateControls();
    try {
      const records = await api(sessionURL());
      if (request !== view.historyRequest || session !== view.sessionId) return;
      view.history = records;
      if (!view.currentId && records.length) view.currentId = records[0].id;
      renderHistory();
      if (view.currentId) loadDetail(view.currentId);
    } catch (failure) {
      if (request === view.historyRequest) error("history-error", failure);
    } finally {
      if (request === view.historyRequest) { view.historyLoading = false; updateControls(); }
    }
  }

  async function loadDetail(id) {
    if (!id || view.detailLoading) return;
    const request = ++view.detailRequest;
    const session = view.sessionId;
    view.detailLoading = true;
    updateControls();
    try {
      const detail = await api(`/api/preannotations/${encodeURIComponent(id)}`);
      if (request !== view.detailRequest || session !== view.sessionId || id !== view.currentId) return;
      const changed = JSON.stringify(view.detail?.frames) !== JSON.stringify(detail.frames);
      view.detail = detail;
      renderDetail();
      if (changed) emitUpdate();
    } catch (failure) {
      if (request === view.detailRequest) error("history-error", failure);
    } finally {
      if (request === view.detailRequest) { view.detailLoading = false; updateControls(); }
    }
  }

  function renderDetail() {
    const detail = view.detail;
    field("detail").hidden = !detail;
    if (!detail) return;
    const frames = detail.frames || [];
    const proposals = frames.reduce((total, frame) => total + (frame.proposal_count || 0), 0);
    field("detail-summary").textContent = `${detail.name} · ${detail.job?.status || "saved"} · ${proposals} saved proposals across ${frames.length} frames`;
    field("progress").value = Number.isFinite(detail.job?.progress) ? detail.job.progress : 0;
    const config = detail.config?.preannotation || {};
    const settings = config.settings || {};
    field("detail-config").textContent = `${config.source?.name || detail.model_ids?.join(", ") || "Local detector"} · ${settings.inference_mode || "full"}${settings.inference_mode === "tiled" ? ` (${settings.tile_size}px, overlap ${settings.overlap})` : ""}${config.threshold != null ? ` · proposal score ≥ ${config.threshold}` : ""} · every frame remains subject to human inspection. Empty outputs never validate negatives; partial results remain available.`;
    const list = field("detail-frames");
    const focusedId = list.contains(document.activeElement) ? document.activeElement.dataset.frameId : null;
    list.replaceChildren();
    for (const frame of frames) {
      const row = node("li", "preannotation-frame-result");
      row.append(node("strong", "", frameLabel(frame.frame_id, frame.source_filename)),
        node("p", "field-hint", `${tools.frameStates[frame.state] || frame.state || "Saved"} · ${frame.proposal_count || 0} proposals${frame.filtered_count ? ` · ${frame.filtered_count} below threshold` : ""}${frame.unmapped_count ? ` · ${frame.unmapped_count} outside mapped classes` : ""}`));
      if (frame.reason) row.append(node("p", "field-hint", frame.reason));
      const button = node("button", "text-button", "Review frame");
      button.type = "button";
      button.dataset.frameId = frame.frame_id;
      button.disabled = !selectedFrames().some((entry) => entry.id === frame.frame_id);
      if (button.disabled) button.title = "Select this frame in Data intake to open it in the review queue.";
      button.addEventListener("click", () => window.dispatchEvent(new CustomEvent("iris:annotation-open-frame", { detail: { frame_id: frame.frame_id } })));
      row.append(button);
      list.append(row);
    }
    for (const frame of config.excluded || []) {
      const row = node("li", "preannotation-frame-result excluded");
      row.append(node("strong", "", frameLabel(frame.frame_id, frame.source_filename)), node("p", "field-hint", `Excluded before launch · ${frame.reason || "not eligible"}`));
      list.append(row);
    }
    if (focusedId) [...list.querySelectorAll("button")].find((button) => button.dataset.frameId === focusedId)?.focus({ preventScroll: true });
    updateControls();
  }

  function showRaw() {
    if (!view.detail || field("raw").disabled) return;
    const dialog = node("dialog", "annotation-record-dialog");
    const heading = node("h2", "", "Saved proposal run · read only");
    heading.id = "preannotation-record-title";
    dialog.setAttribute("aria-labelledby", heading.id);
    const close = node("button", "button button-secondary", "Close");
    close.type = "button";
    close.addEventListener("click", () => dialog.close());
    dialog.append(heading, node("p", "field-hint", "Frozen inputs, raw predictions, class mappings and run provenance are shown as saved. These are detector outputs, not human labels. Human corrections are retained in the frame's saved revisions."), node("pre", "", JSON.stringify(view.detail, null, 2)), close);
    dialog.addEventListener("close", () => dialog.remove());
    document.body.append(dialog);
    dialog.showModal();
  }

  function openPanel() {
    if (!shown()) return;
    renderFrames();
    window.dispatchEvent(new CustomEvent("iris:annotation-context-request"));
    if (!view.loaded) { view.loaded = true; loadModels(); }
    loadHistory();
  }

  for (const name of ["scope", "model", "threshold", "mode", "device", "tile-size", "overlap"])
    field(name).addEventListener(["threshold", "tile-size", "overlap"].includes(name) ? "input" : "change", () => {
      field(name).setCustomValidity("");
      invalidate();
      if (name === "model") renderModel();
    });
  field("name").addEventListener("input", updateControls);
  field("select-all").addEventListener("click", () => {
    if (field("select-all").disabled) return;
    view.selected = new Set(selectedFrames().map((frame) => frame.id));
    invalidate(); renderFrames();
  });
  field("clear").addEventListener("click", () => { view.selected.clear(); invalidate(); renderFrames(); });
  field("preview").addEventListener("click", previewRun);
  field("start").addEventListener("click", startRun);
  field("refresh-models").addEventListener("click", loadModels);
  field("refresh-history").addEventListener("click", loadHistory);
  field("raw").addEventListener("click", showRaw);
  field("job").addEventListener("click", () => window.dispatchEvent(new CustomEvent("iris:job-open", { detail: { job_id: view.detail?.job_id || view.detail?.job?.id } })));
  field("history").addEventListener("change", () => {
    view.currentId = field("history").value || null;
    view.detailRequest++;
    view.detailLoading = false;
    view.detail = null;
    renderDetail();
    loadDetail(view.currentId);
  });
  panel.addEventListener("toggle", () => { if (panel.open) openPanel(); else invalidate(); });
  window.addEventListener("iris:workspace", (event) => {
    view.active = event.detail.name === "annotation";
    if (!view.active) invalidate();
    openPanel();
  });
  window.addEventListener("iris:annotation-context", (event) => {
    const previous = view.context;
    view.context = event.detail;
    const key = JSON.stringify(event.detail);
    if (key !== view.contextKey) {
      view.contextKey = key;
      if (field("scope").value === "current" ||
          view.selected.has(previous?.frame_id) || view.selected.has(event.detail?.frame_id)) invalidate();
      else updateControls();
    }
  });
  window.addEventListener("iris:before-session", (event) => {
    if (!view.operation) return;
    event.preventDefault();
    notify("Wait for the proposal preview or creation request to finish before changing sessions.", true);
  });
  window.addEventListener("beforeunload", (event) => {
    if (view.operation === "start") { event.preventDefault(); event.returnValue = ""; }
  });
  window.addEventListener("iris:session", () => {
    if (view.sessionId === state.sessionId) return;
    view.sessionId = state.sessionId;
    view.context = null;
    view.contextKey = "";
    view.selected.clear();
    view.framesKey = "";
    view.catalogRequest++; view.historyRequest++; view.detailRequest++; view.operationToken++;
    view.catalogLoading = false; view.historyLoading = false; view.detailLoading = false;
    view.operation = null; view.loaded = false; view.history = []; view.currentId = null; view.detail = null;
    view.jobsKey = "";
    invalidate(); renderFrames(); renderHistory(); renderDetail();
    error("history-error", null);
    openPanel();
  });
  window.addEventListener("iris:frames", () => {
    const key = JSON.stringify(selectedFrames().map((frame) => frame.id));
    if (key === view.framesKey) return;
    view.framesKey = key;
    invalidate(); renderFrames(); renderDetail();
  });
  window.addEventListener("iris:jobs", () => {
    if (!shown()) return;
    const key = JSON.stringify(state.jobs.filter((job) => job.kind === "infer").map((job) => [job.id, job.status, job.progress]));
    if (key !== view.jobsKey) { view.jobsKey = key; loadHistory(); }
  });
  updateControls();
})();
