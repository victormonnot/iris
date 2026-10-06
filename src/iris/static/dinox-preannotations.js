"use strict";

(() => {
  const panel = $("#dinox-preannotations");
  const field = (name) => $(`#dinox-${name}`);
  const tools = window.IRISDINOXTools;
  const view = {
    active: false, projectId: state.projectId, sessionId: state.sessionId, context: null, contextKey: "",
    selected: new Set(), framesKey: "", generation: 0, preview: null, operation: null, operationToken: 0,
    provider: null, providerLoading: false, providerRequest: 0, history: [], currentId: null, detail: null,
    historyLoading: false, historyRequest: 0, detailLoading: false, detailRequest: 0, jobsKey: "",
  };
  const selectedFrames = () => state.frames.filter((frame) => frame.selected);
  const shown = () => view.active && panel.open && view.sessionId;
  const sessionURL = () => `/api/sessions/${encodeURIComponent(view.sessionId)}/dinox-batches`;
  const scope = () => `${view.projectId}/${view.sessionId}`;
  const ready = () => view.provider?.status === "ready";
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
    const frames = selectedFrames();
    const current = view.context?.session_id === view.sessionId && view.context.frame_id;
    const options = {
      frame_ids: field("scope").value === "current"
        ? current && frames.some((frame) => frame.id === current) ? [current] : []
        : frames.filter((frame) => view.selected.has(frame.id)).map((frame) => frame.id),
      threshold: Number(field("threshold").value),
    };
    const prompts = tools.parsePrompts(field("prompts").value);
    if (prompts) options.class_prompts = prompts;
    return options;
  }
  function payloadKey() {
    try { return JSON.stringify(payload()); } catch { return null; }
  }
  function frameIds() {
    const current = view.context?.session_id === view.sessionId && view.context.frame_id;
    return selectedFrames().filter((frame) => field("scope").value === "current" ? frame.id === current : view.selected.has(frame.id)).map((frame) => frame.id);
  }
  const contextBlocked = () => frameIds().includes(view.context?.frame_id) && (view.context.dirty || view.context.busy);
  function invalidate() {
    view.generation++;
    view.preview = null;
    field("preview-result").hidden = true;
    field("consent").checked = false;
    error("error", null);
    updateControls();
  }
  function updateControls() {
    const busy = Boolean(view.operation);
    const batch = field("scope").value === "batch";
    const ids = frameIds();
    const count = view.selected.size;
    field("batch").hidden = !batch;
    field("current").hidden = batch;
    field("current").textContent = !batch && ids.length
      ? `${frameLabel(ids[0])}${contextBlocked() ? " · save or discard edits and finish the current action first" : ""}`
      : "Open a selected image in the review queue first.";
    field("selection-count").textContent = `${count} chosen · maximum 25`;
    field("select-all").disabled = busy || !selectedFrames().length || selectedFrames().length > 25;
    field("select-all").title = selectedFrames().length > 25 ? "Choose up to 25 images individually." : "Choose all selected images.";
    field("clear").disabled = busy || !count;
    for (const input of field("frames").querySelectorAll("input")) input.disabled = busy || (!input.checked && count >= 25);
    for (const name of ["scope", "name", "threshold", "prompts", "consent", "budget", "key"]) field(name).disabled = busy;
    field("save-key").disabled = busy || view.providerLoading || !field("key").value.trim();
    field("delete-key").disabled = busy || view.providerLoading || !view.provider;
    field("refresh-provider").disabled = busy || view.providerLoading;
    field("preview").disabled = busy || view.providerLoading || !view.sessionId || !ids.length || ids.length > 25 || contextBlocked();
    field("preview").textContent = view.operation === "preview" ? "Checking saved images and cost…" : "2 · Preview images and cost";
    field("start").disabled = busy || !tools.canProcess(view.preview, view.provider) || contextBlocked() || !field("name").value.trim() ||
      !tools.approved(view.preview, { key: payloadKey(), consent: field("consent").checked, budget: field("budget").value });
    field("start").textContent = view.operation === "start" ? "Creating DINO-X batch…"
      : view.preview?.request_count === 0 && view.preview?.poll_count === 0 ? "3 · Reuse saved proposals" : "3 · Create cloud proposals";
    field("refresh-history").disabled = busy || view.historyLoading;
    field("history").disabled = busy || !view.history.length;
    field("job").disabled = !view.detail?.job_id && !view.detail?.job?.id;
    field("raw").disabled = !view.detail || view.detailLoading;
    field("prepare").disabled = busy || !view.detail || view.detailLoading || tools.active(view.detail?.job);
  }
  function renderFrames() {
    const frames = selectedFrames();
    const available = new Set(frames.map((frame) => frame.id));
    for (const id of view.selected) if (!available.has(id)) view.selected.delete(id);
    field("frames").replaceChildren();
    if (!frames.length) field("frames").append(node("p", "field-hint", "Select images in Data intake to prepare a batch."));
    for (const frame of frames) {
      const label = node("label", "assistance-batch-frame");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = view.selected.has(frame.id);
      input.addEventListener("change", () => {
        if (input.checked) view.selected.add(frame.id); else view.selected.delete(frame.id);
        invalidate();
      });
      label.append(input, node("span", "", frameLabel(frame.id)));
      field("frames").append(label);
    }
    updateControls();
  }
  function renderProvider() {
    const provider = view.provider;
    field("provider-status").textContent = view.providerLoading ? "Checking DINO-X configuration…"
      : !provider ? "DINO-X status unavailable. Refresh to check the local configuration."
        : `${ready() ? "Ready" : provider.status === "missing_key" ? "Key required for cloud requests" : "Cloud credential unavailable"} · ${provider.model || "DINO-X-1.0"} · ${tools.money(provider.price?.amount_per_request)} per new request${provider.reason ? ` · ${provider.reason}` : ""}`;
    field("key-source").textContent = provider?.key_source === "environment"
      ? "Using the server environment key. It takes precedence over a saved key; change it in the server environment. Key values are never returned here."
      : provider?.key_source === "file"
        ? "A key is saved on this local server. Saving replaces it; removing it disables new requests unless an environment key is available."
        : "Save a key on the local server to enable new cloud requests. Saved results can be reused without a key. This page never reads back or retains the key.";
    updateControls();
  }
  async function loadProvider() {
    if (view.providerLoading || view.operation) return;
    const request = ++view.providerRequest;
    const current = scope();
    view.providerLoading = true;
    invalidate(); renderProvider();
    try {
      const provider = await api("/api/dinox/provider");
      if (request === view.providerRequest && current === scope()) view.provider = provider;
    } catch (failure) {
      if (request === view.providerRequest && current === scope()) { view.provider = null; error("key-error", failure); }
    } finally {
      if (request === view.providerRequest) { view.providerLoading = false; renderProvider(); }
    }
  }
  async function changeKey(remove = false) {
    if (view.operation || field(remove ? "delete-key" : "save-key").disabled) return;
    const key = field("key").value.trim();
    field("key").value = "";
    const current = scope();
    const token = ++view.operationToken;
    view.operation = "key";
    invalidate(); error("key-error", null);
    try {
      const provider = await api("/api/dinox/key", remove ? { method: "DELETE" } : { method: "PUT", body: JSON.stringify({ key }) });
      if (current !== scope()) return;
      view.provider = provider;
      renderProvider();
      notify(remove ? "Saved DINO-X key removed." : "DINO-X key saved on the local server.");
    } catch {
      if (current === scope()) error("key-error", "Could not update the DINO-X key. Refresh the provider status and try again; the input has been cleared.");
    } finally {
      field("key").value = "";
      if (token === view.operationToken) { view.operation = null; updateControls(); }
    }
  }
  function validSettings() {
    const threshold = field("threshold");
    threshold.setCustomValidity(threshold.value.trim() ? "" : "Enter a proposal score.");
    if (!threshold.reportValidity()) return false;
    try { payload(); return true; } catch (failure) { error("error", failure); return false; }
  }
  const guardFrames = (ids) => window.dispatchEvent(new CustomEvent("iris:before-assistance-batch", { cancelable: true, detail: { frame_ids: ids } }));
  function renderPreview() {
    const preview = view.preview;
    field("preview-result").hidden = !preview;
    if (!preview) return;
    field("preview-summary").textContent = `${preview.eligible_count} eligible · ${preview.excluded_count} excluded · ${preview.request_count} new requests · ${preview.reuse_count} saved results · ${preview.poll_count} already submitted. No cloud request has started.`;
    field("preview-cost").textContent = `Estimated new cost: ${tools.money(preview.estimate?.total)}`;
    field("consent-label").textContent = preview.request_count === 0 && preview.poll_count === 0
      ? "I approve reusing these saved DINO-X results as proposals. No image will be sent."
      : "I approve external DINO-X processing for these eligible images and class prompts, with the estimated cost shown above.";
    const prompts = preview.provider_config?.class_prompts || preview.config?.class_prompts || {};
    field("preview-settings").textContent = `${preview.provider?.model || "DINO-X-1.0"} · score ≥ ${preview.config?.threshold ?? payload().threshold} · ${tools.money(preview.estimate?.amount_per_request)} per new request. Class prompts: ${Object.entries(prompts).map(([id, prompt]) => `${id}: ${prompt}`).join("; ") || "saved taxonomy defaults"}.`;
    field("preview-frames").replaceChildren();
    for (const frame of preview.frames || []) {
      const row = node("li", frame.eligible ? "eligible" : "excluded");
      row.append(node("strong", "", frameLabel(frame.frame_id, frame.source_filename)), node("p", "field-hint", `${tools.actions[frame.action] || frame.action || "Excluded"}${frame.reason ? ` · ${frame.reason}` : ""}${frame.proposal_count != null ? ` · ${frame.proposal_count} saved proposals` : ""}`));
      field("preview-frames").append(row);
    }
    field("preview-warnings").replaceChildren(...(preview.warnings || []).map((warning) => node("li", "", warning)));
    field("budget").placeholder = Number.isFinite(preview.estimate?.total) ? preview.estimate.total.toFixed(2) : "Enter a spending limit";
    updateControls();
  }
  async function previewRun() {
    if (field("preview").disabled || !validSettings()) return;
    const options = payload();
    if (!guardFrames(options.frame_ids)) return;
    invalidate();
    const generation = view.generation;
    const key = payloadKey();
    const current = scope();
    const token = ++view.operationToken;
    view.operation = "preview";
    updateControls();
    try {
      const result = await api(`${sessionURL()}/preview`, { method: "POST", body: JSON.stringify(options) });
      if (current !== scope() || generation !== view.generation || key !== payloadKey()) return;
      view.preview = { ...result, key };
      if (result.provider) { view.provider = result.provider; renderProvider(); }
      renderPreview();
    } catch (failure) {
      if (current === scope() && generation === view.generation) error("error", failure);
    } finally {
      if (token === view.operationToken) { view.operation = null; updateControls(); }
    }
  }
  function retainBatch(detail) {
    view.currentId = detail.id;
    view.detail = detail;
    view.historyRequest++; view.detailRequest++;
    view.historyLoading = false; view.detailLoading = false;
    renderDetail();
    loadHistory();
    refreshJobs().catch((failure) => error("history-error", failure));
    emitUpdate();
  }
  async function startRun() {
    if (field("start").disabled || !view.preview || !validSettings() || !field("budget").reportValidity()) return;
    const options = payload();
    if (!guardFrames(options.frame_ids)) return;
    const fingerprint = view.preview.fingerprint;
    const current = scope();
    const path = sessionURL();
    const token = ++view.operationToken;
    view.operation = "start";
    error("error", null); updateControls();
    try {
      const detail = await api(path, { method: "POST", body: JSON.stringify({ ...options, name: field("name").value.trim(), expected_fingerprint: fingerprint, approve_external: true, max_cost_cny: Number(field("budget").value) }) });
      if (current !== scope()) return;
      invalidate(); retainBatch(detail);
      notify("DINO-X batch queued. Review every proposal and inspect each whole image before validating.");
    } catch (failure) {
      if (current !== scope()) return;
      let receipt = null;
      try { receipt = tools.findReceipt(await api(path), fingerprint); } catch { /* Reconcile the saved receipt without repeating a paid request. */ }
      if (current !== scope()) return;
      invalidate();
      if (receipt) {
        retainBatch(receipt);
        notify("The DINO-X batch was already recorded. Its saved history is open; no second request was sent.");
      } else error("error", `${failure.message} Check saved batches and Project jobs before preparing another batch. No request was repeated automatically.`);
    } finally {
      if (token === view.operationToken) { view.operation = null; updateControls(); }
    }
  }
  function renderHistory() {
    const selector = field("history");
    selector.replaceChildren(new Option(view.history.length ? "Choose a saved DINO-X batch" : "No DINO-X batches in this session", ""));
    for (const entry of view.history) selector.append(new Option(`${entry.name} · ${entry.job?.status || "saved"}${entry.created_at ? ` · ${new Date(entry.created_at).toLocaleString()}` : ""}`, entry.id));
    selector.value = view.currentId || "";
    updateControls();
  }
  async function loadHistory() {
    if (!view.sessionId || view.historyLoading) return;
    const request = ++view.historyRequest;
    const current = scope();
    view.historyLoading = true;
    error("history-error", null); updateControls();
    try {
      const records = await api(sessionURL());
      if (request !== view.historyRequest || current !== scope()) return;
      view.history = records;
      if (!view.currentId && records.length) view.currentId = records[0].id;
      renderHistory();
      if (view.currentId) loadDetail(view.currentId);
    } catch (failure) {
      if (request === view.historyRequest && current === scope()) error("history-error", failure);
    } finally {
      if (request === view.historyRequest) { view.historyLoading = false; updateControls(); }
    }
  }
  async function loadDetail(id) {
    if (!id || view.detailLoading) return;
    const request = ++view.detailRequest;
    const current = scope();
    view.detailLoading = true; updateControls();
    try {
      const detail = await api(`/api/dinox-batches/${encodeURIComponent(id)}`);
      if (request !== view.detailRequest || current !== scope() || id !== view.currentId) return;
      const changed = JSON.stringify(view.detail?.frames) !== JSON.stringify(detail.frames);
      view.detail = detail;
      renderDetail();
      if (changed) emitUpdate();
    } catch (failure) {
      if (request === view.detailRequest && current === scope()) error("history-error", failure);
    } finally {
      if (request === view.detailRequest) { view.detailLoading = false; updateControls(); }
    }
  }
  function renderDetail() {
    const detail = view.detail;
    field("detail").hidden = !detail;
    if (!detail) { updateControls(); return; }
    const frames = detail.frames || [];
    field("detail-summary").textContent = `${detail.name} · ${detail.job?.status || "saved"} · ${detail.counts?.ready || 0}/${detail.counts?.total ?? frames.length} images ready · ${detail.counts?.issues || 0} issues · ${detail.counts?.proposals ?? frames.reduce((sum, frame) => sum + (frame.proposal_count || 0), 0)} proposals`;
    field("progress").value = Number.isFinite(detail.job?.progress) ? detail.job.progress : 0;
    field("detail-cost").textContent = `Estimated new cost: ${tools.money(detail.cost?.estimated_cny)} · ${detail.cost?.new_requests ?? 0} new requests${detail.cost?.unknown_outcome_count ? ` · ${detail.cost.unknown_outcome_count} unknown delivery outcomes; check provider records before any new request` : ""}. Provider billing determines the final charge.`;
    field("detail-config").textContent = `DINO-X-1.0 · score ≥ ${detail.config?.settings?.threshold ?? "unknown"}. Proposals and empty outputs still require human inspection. Partial results remain available.`;
    const list = field("detail-frames");
    const focusedId = list.contains(document.activeElement) ? document.activeElement.dataset.frameId : null;
    list.replaceChildren();
    for (const frame of frames) {
      const row = node("li", "dinox-frame-result");
      row.append(node("strong", "", frameLabel(frame.frame_id, frame.source_filename)), node("p", "field-hint", `${tools.frameStates[frame.state] || frame.state || "Saved"} · ${frame.proposal_count || 0} proposals${frame.reused ? " · saved result reused" : ""}`));
      if (frame.reason) row.append(node("p", "field-hint", frame.reason));
      if (frame.request_id) row.append(node("p", "field-hint monospace", `Saved request: ${frame.request_id}`));
      const button = node("button", "text-button", "Review image");
      button.type = "button"; button.dataset.frameId = frame.frame_id;
      button.disabled = !selectedFrames().some((entry) => entry.id === frame.frame_id);
      if (button.disabled) button.title = "Select this image in Data intake to open it in the review queue.";
      button.addEventListener("click", () => window.dispatchEvent(new CustomEvent("iris:annotation-open-frame", { detail: { frame_id: frame.frame_id } })));
      row.append(button); list.append(row);
    }
    for (const frame of detail.config?.excluded || []) {
      const row = node("li", "dinox-frame-result excluded");
      row.append(node("strong", "", frameLabel(frame.frame_id, frame.source_filename)), node("p", "field-hint", `Excluded before launch · ${frame.reason || "not eligible"}`));
      list.append(row);
    }
    if (focusedId) [...list.querySelectorAll("button")].find((button) => button.dataset.frameId === focusedId)?.focus({ preventScroll: true });
    updateControls();
  }
  function prepareRemaining() {
    if (field("prepare").disabled) return;
    const detail = view.detail;
    const available = new Set(selectedFrames().map((frame) => frame.id));
    const ids = detail.frame_ids || (detail.frames || []).map((frame) => frame.frame_id);
    view.selected = new Set(ids.filter((id) => available.has(id)).slice(0, 25));
    field("scope").value = "batch";
    field("threshold").value = detail.config?.settings?.threshold ?? 0.25;
    field("prompts").value = detail.config?.settings?.class_prompts ? JSON.stringify(detail.config.settings.class_prompts, null, 2) : "";
    field("name").value = `${detail.name} · remaining`.slice(0, 160);
    field("budget").value = "";
    invalidate(); renderFrames();
    field("preview").scrollIntoView({ block: "center" });
    field("preview").focus({ preventScroll: true });
    notify(`Saved settings restored. Preview again to check remaining requests, reuse and cost.${ids.length !== view.selected.size ? " Some images are no longer selected; select them in Data intake to include them." : ""}`);
  }
  function showRaw() {
    if (field("raw").disabled) return;
    const dialog = node("dialog", "annotation-record-dialog");
    const heading = node("h2", "", "Saved DINO-X batch · read only");
    heading.id = "dinox-record-title"; dialog.setAttribute("aria-labelledby", heading.id);
    const close = node("button", "button button-secondary", "Close");
    close.type = "button"; close.addEventListener("click", () => dialog.close());
    dialog.append(heading, node("p", "field-hint", "Saved inputs, provider request IDs, raw outputs and proposal provenance. Credentials are omitted. Human decisions are retained in the image's saved annotation revisions."), node("pre", "", JSON.stringify(view.detail, null, 2)), close);
    dialog.addEventListener("close", () => dialog.remove());
    document.body.append(dialog); dialog.showModal();
  }
  function openPanel() {
    if (!shown()) return;
    renderFrames();
    window.dispatchEvent(new CustomEvent("iris:annotation-context-request"));
    if (!view.provider) loadProvider();
    loadHistory();
  }
  for (const name of ["scope", "threshold", "prompts"]) field(name).addEventListener(name === "scope" ? "change" : "input", () => { field(name).setCustomValidity(""); invalidate(); });
  for (const name of ["name", "budget", "key"]) field(name).addEventListener("input", updateControls);
  field("consent").addEventListener("change", updateControls);
  field("key-form").addEventListener("submit", (event) => { event.preventDefault(); changeKey(); });
  field("delete-key").addEventListener("click", () => changeKey(true));
  field("refresh-provider").addEventListener("click", loadProvider);
  field("select-all").addEventListener("click", () => { if (field("select-all").disabled) return; view.selected = new Set(selectedFrames().map((frame) => frame.id)); invalidate(); renderFrames(); });
  field("clear").addEventListener("click", () => { view.selected.clear(); invalidate(); renderFrames(); });
  field("preview").addEventListener("click", previewRun);
  field("start").addEventListener("click", startRun);
  field("refresh-history").addEventListener("click", loadHistory);
  field("prepare").addEventListener("click", prepareRemaining);
  field("raw").addEventListener("click", showRaw);
  field("job").addEventListener("click", () => window.dispatchEvent(new CustomEvent("iris:job-open", { detail: { job_id: view.detail?.job_id || view.detail?.job?.id } })));
  field("history").addEventListener("change", () => {
    view.currentId = field("history").value || null;
    view.detailRequest++; view.detailLoading = false; view.detail = null;
    renderDetail(); loadDetail(view.currentId);
  });
  panel.addEventListener("toggle", () => { if (panel.open) openPanel(); else { field("key").value = ""; invalidate(); } });
  window.addEventListener("iris:workspace", (event) => {
    view.active = event.detail.name === "annotation";
    if (!view.active) { field("key").value = ""; invalidate(); }
    openPanel();
  });
  window.addEventListener("iris:annotation-context", (event) => {
    const previous = view.context;
    view.context = event.detail;
    const key = JSON.stringify(event.detail);
    if (key === view.contextKey) return;
    view.contextKey = key;
    if (field("scope").value === "current" || view.selected.has(previous?.frame_id) || view.selected.has(event.detail?.frame_id)) invalidate(); else updateControls();
  });
  window.addEventListener("iris:before-session", (event) => {
    if (!view.operation) return;
    event.preventDefault(); notify("Wait for the DINO-X request to finish before changing sessions.", true);
  });
  window.addEventListener("beforeunload", (event) => { field("key").value = ""; if (view.operation === "start") { event.preventDefault(); event.returnValue = ""; } });
  function resetScope() {
    if (view.sessionId === state.sessionId && view.projectId === state.projectId) return;
    view.sessionId = state.sessionId; view.projectId = state.projectId;
    view.context = null; view.contextKey = ""; view.selected.clear(); view.framesKey = "";
    view.providerRequest++; view.historyRequest++; view.detailRequest++; view.operationToken++;
    view.providerLoading = false; view.historyLoading = false; view.detailLoading = false;
    view.operation = null; view.provider = null; view.history = []; view.currentId = null; view.detail = null; view.jobsKey = "";
    field("key").value = ""; field("budget").value = ""; field("prompts").value = "";
    invalidate(); renderProvider(); renderFrames(); renderHistory(); renderDetail();
    error("key-error", null); error("history-error", null); openPanel();
  }
  window.addEventListener("iris:session", resetScope);
  window.addEventListener("iris:project-ready", resetScope);
  window.addEventListener("iris:frames", () => {
    const key = JSON.stringify(selectedFrames().map((frame) => frame.id));
    if (key === view.framesKey) return;
    view.framesKey = key; invalidate(); renderFrames(); renderDetail();
  });
  window.addEventListener("iris:jobs", () => {
    if (!shown()) return;
    const key = JSON.stringify(state.jobs.filter((job) => job.kind === "dinox").map((job) => [job.id, job.status, job.progress]));
    if (key !== view.jobsKey) { view.jobsKey = key; loadHistory(); }
  });
  window.addEventListener("iris:dinox-batch-open", (event) => {
    if (!event.detail?.batch_id || !view.sessionId) return;
    view.currentId = event.detail.batch_id; view.detailRequest++; view.detailLoading = false; view.detail = null;
    panel.open = true;
    renderDetail(); openPanel(); loadDetail(view.currentId);
    panel.scrollIntoView({ block: "start" });
  });
  updateControls();
})();
