"use strict";

(() => {
  const svgNS = "http://www.w3.org/2000/svg";
  const editor = {
    sessionId: null,
    frameId: null,
    document: null,
    frames: [],
    boxes: [],
    decisions: {},
    dirty: false,
    busy: false,
    loading: false,
    request: 0,
    selected: null,
    tool: "select",
    drag: null,
    provider: null,
    catalog: null,
    catalogLoading: false,
    catalogRequest: 0,
    preview: null,
    previewGeneration: 0,
    previewTimer: null,
    jobStatuses: new Map(),
    remoteUpdate: false,
    refreshAfterRequest: false,
  };
  const clone = (value) => structuredClone(value);
  const url = (suffix = "annotation") =>
    `/api/frames/${encodeURIComponent(editor.frameId)}/${suffix}`;
  const selectedBox = () =>
    editor.boxes.find((box) => box.id === editor.selected);
  const pending = () =>
    (editor.document?.suggestions || []).filter(
      (proposal) =>
        !(
          editor.decisions[proposal.id] ||
          (proposal.state !== "pending" && proposal.state)
        ),
    );
  const decisionFor = (proposal) =>
    editor.decisions[proposal.id] || proposal.state || "pending";
  const proposalOrigin = (proposal) =>
    proposal.kind === "multimodal" ? "Model review"
      : proposal.kind === "imported" ? "Imported annotation" : "Detector";
  const assistActive = () =>
    state.jobs.some(
      (job) =>
        job.kind === "assist" &&
        job.params?.frame_id === editor.frameId &&
        isActive(job),
    );

  function error(message, conflict = false) {
    $("#annotation-error").textContent = message || "";
    $("#annotation-error").hidden = !message;
    if (conflict) $("#annotation-conflict").hidden = false;
  }

  function reportFailure(failure) {
    const recovery = failure.message.includes("unchanged saved annotation")
      ? " Your edits have been kept. Reload the saved frame, reject the outdated proposal, save your draft, then request a fresh review."
      : "";
    error(failure.message + recovery, failure.status === 409);
  }

  function discardAllowed() {
    if (editor.busy) {
      notify(
        "Wait for the annotation request to finish before changing frames.",
        true,
      );
      return false;
    }
    return (
      !editor.dirty ||
      window.confirm("Discard unsaved annotation changes for this frame?")
    );
  }

  function changed() {
    invalidatePreview();
    editor.dirty = true;
    updateStatus();
  }

  function updateBoxDecision(box) {
    if (!box.suggestion_id) return;
    const proposal = editor.document.suggestions.find(
      (item) => item.id === box.suggestion_id,
    );
    if (!proposal) return;
    editor.decisions[proposal.id] =
      box.label === proposal.label &&
      box.box.every((value, index) => value === proposal.box[index])
        ? "accepted"
        : "corrected";
  }

  function finishRequest() {
    editor.busy = false;
    updateStatus();
    renderProposals();
    if (editor.refreshAfterRequest) {
      editor.refreshAfterRequest = false;
      refreshCurrent();
    }
  }

  function updateStatus() {
    const document = editor.document;
    if (!document) return;
    const status = $("#annotation-status");
    const count = pending().length;
    status.className = `annotation-status ${editor.dirty || (count && document.status === "validated") ? "dirty" : document.status}`;
    status.textContent = editor.dirty
      ? `Unsaved draft · based on revision ${document.revision}`
      : count && document.status === "validated"
        ? `Revision ${document.revision} validated · new proposals pending`
        : `${document.status === "validated" ? "Validated" : document.status === "draft" ? "Draft" : "Unannotated"} · revision ${document.revision}`;
    $("#annotation-review-summary").textContent =
      `${editor.boxes.length} label${editor.boxes.length === 1 ? "" : "s"} · ${count} pending proposal${count === 1 ? "" : "s"}`;
    const blocked = editor.busy || editor.loading;
    $("#annotation-save").disabled =
      blocked || (!editor.dirty && document.status === "draft");
    $("#annotation-validate").disabled =
      blocked || count > 0 || !$("#annotation-reviewer").value.trim();
    $("#annotation-save-hint").textContent = count
      ? "Accept or reject every pending proposal before validation."
      : !$("#annotation-reviewer").value.trim()
        ? "Enter your name or initials to validate this frame."
        : editor.remoteUpdate
          ? "New results are available. Save or discard your local changes, then reload the latest revision."
          : editor.dirty
            ? "Changes are local until you save. Saving edits creates a new revision."
            : "Saved revisions remain available below.";
    $("#annotation-import").disabled =
      blocked || editor.dirty || !$("#annotation-prediction").value;
    $("#annotation-assist").disabled =
      blocked ||
      editor.dirty ||
      assistActive() ||
      editor.catalogLoading ||
      !providerReady() ||
      ($("#annotation-assist-source").value === "prediction" &&
        !$("#annotation-prediction").value);
    $("#annotation-proposal-hint").textContent = editor.dirty
      ? "Save your draft before importing proposals or requesting model review."
      : "Only people and cars from this frame are imported. Importing does not validate any labels.";
    const position = editor.frames.findIndex(
      (frame) => frame.id === editor.frameId,
    );
    $("#annotation-previous").disabled = blocked || position <= 0;
    $("#annotation-next").disabled =
      blocked || position < 0 || position >= editor.frames.length - 1;
    $("#annotation-frame").disabled = blocked;
    $("#annotation-refresh").disabled = blocked;
    $("#annotation-canvas").classList.toggle("drawing", editor.tool === "draw");
    $("#annotation-canvas").classList.toggle("busy", blocked);
    for (const name of ["draw", "select"]) {
      $(`#annotation-${name}-tool`).classList.toggle(
        "active",
        editor.tool === name,
      );
      $(`#annotation-${name}-tool`).setAttribute(
        "aria-pressed",
        String(editor.tool === name),
      );
      $(`#annotation-${name}-tool`).disabled = blocked;
    }
    for (const selector of [
      "#annotation-add-box",
      "#annotation-remove-box",
      "#annotation-apply-coordinates",
      "#annotation-box-class",
      "#annotation-reviewer",
      "#annotation-notes",
      "#annotation-assist-source",
      "#annotation-instructions",
      "#annotation-threshold",
    ]) {
      $(selector).disabled = blocked;
    }
    $("#annotation-prediction").disabled =
      blocked || !$("#annotation-prediction").value;
    for (const selector of [
      "#annotation-execution",
      "#annotation-provider",
      "#annotation-model",
      "#annotation-provider-refresh",
    ]) {
      $(selector).disabled = blocked || editor.catalogLoading;
    }
    $("#annotation-assist").textContent =
      editor.provider && !editor.provider.local
        ? "Preview API request"
        : "Request local review";
    updatePreviewStatus();
    $("#annotation-assist-message").textContent = assistActive()
      ? "Model review is running. Follow progress or cancel in Processing jobs."
      : editor.remoteUpdate
        ? "New results are waiting; your unsaved edits have been preserved."
        : "Model proposals always require your decision. They never validate a frame.";
  }

  function frameCaption(frame, index) {
    const source = sourceFor(frame);
    return `${index + 1} / ${editor.frames.length} · ${source?.filename || frame.id.slice(0, 8)}${frame.timestamp_seconds != null ? ` · ${timestamp(frame.timestamp_seconds)}` : ""}${frame.selected ? "" : " · no longer selected"}`;
  }

  function syncFrames() {
    if (editor.sessionId !== state.sessionId) return;
    editor.frames = state.frames.filter((frame) => frame.selected);
    if (
      (editor.dirty || editor.busy || editor.drag) &&
      editor.document &&
      !editor.frames.some((frame) => frame.id === editor.frameId)
    ) {
      editor.frames.push({ ...editor.document.frame, selected: false });
    }
    const selector = $("#annotation-frame");
    selector.replaceChildren();
    editor.frames.forEach((frame, index) =>
      selector.append(new Option(frameCaption(frame, index), frame.id)),
    );
    if (editor.frames.some((frame) => frame.id === editor.frameId)) {
      selector.value = editor.frameId;
      updateStatus();
      return;
    }
    if (editor.frames.length) loadFrame(editor.frames[0].id);
    else {
      editor.frameId = null;
      editor.document = null;
      editor.request++;
      $("#annotation-editor").hidden = true;
      $("#annotation-empty").hidden = false;
    }
  }

  async function loadFrame(id) {
    if (!id) return;
    invalidatePreview();
    const request = ++editor.request;
    editor.frameId = id;
    editor.loading = true;
    editor.document = null;
    editor.dirty = false;
    $("#annotation-editor").hidden = true;
    $("#annotation-empty").hidden = true;
    error(null);
    $("#annotation-conflict").hidden = true;
    try {
      const result = await api(url());
      if (request !== editor.request) return;
      applyDocument(result);
    } catch (failure) {
      if (request === editor.request) reportFailure(failure);
    } finally {
      if (request === editor.request) {
        editor.loading = false;
        updateStatus();
      }
    }
  }

  async function navigate(id) {
    if (!id || id === editor.frameId) return;
    if (!discardAllowed()) {
      $("#annotation-frame").value = editor.frameId;
      return;
    }
    await loadFrame(id);
  }

  function applyDocument(document) {
    invalidatePreview();
    editor.document = document;
    editor.boxes = clone(document.boxes || []);
    editor.decisions = clone(document.decisions || {});
    editor.selected = null;
    editor.dirty = false;
    editor.remoteUpdate = false;
    editor.drag = null;
    let reviewer = document.reviewer || "";
    if (!reviewer) {
      try {
        reviewer = localStorage.getItem("iris.reviewer") || "";
      } catch {
        /* Optional storage. */
      }
    }
    $("#annotation-reviewer").value = reviewer;
    $("#annotation-notes").value = document.notes || "";
    $("#annotation-conflict").hidden = true;
    $("#annotation-editor").hidden = false;
    $("#annotation-empty").hidden = true;
    const { frame } = document;
    const source = sourceFor(frame);
    $("#annotation-source").textContent =
      `${source?.filename || frame.asset_id} · ${frame.width} × ${frame.height}${frame.timestamp_seconds != null ? ` · ${timestamp(frame.timestamp_seconds)}` : ""}`;
    $("#annotation-frame").value = editor.frameId;
    const canvas = $("#annotation-canvas");
    canvas.setAttribute("viewBox", `0 0 ${frame.width} ${frame.height}`);
    const image = $("#annotation-image");
    image.setAttribute(
      "href",
      `/api/frames/${encodeURIComponent(frame.id)}/image`,
    );
    image.setAttribute("width", frame.width);
    image.setAttribute("height", frame.height);
    $("#annotation-taxonomy").replaceChildren();
    for (const category of document.taxonomy.classes) {
      const entry = node("p");
      entry.append(
        node("strong", "", `${category.name}. `),
        documentText(category.definition),
      );
      $("#annotation-taxonomy").append(entry);
    }
    renderSources();
    renderBoxes();
    renderProposals();
    renderHistory();
    updateStatus();
  }

  function documentText(text) {
    return window.document.createTextNode(text || "");
  }

  function renderSources() {
    const selector = $("#annotation-prediction");
    const previous = selector.value;
    selector.replaceChildren();
    for (const prediction of editor.document.prediction_sources || []) {
      selector.append(
        new Option(
          `${prediction.comparison_name} · ${prediction.model_id} · ${prediction.detection_count} detections`,
          prediction.id,
        ),
      );
    }
    if (!selector.options.length)
      selector.append(new Option("No saved detections for this frame", ""));
    else if ([...selector.options].some((option) => option.value === previous))
      selector.value = previous;
    selector.disabled = !selector.value;
  }

  function svg(tag, attributes) {
    const element = document.createElementNS(svgNS, tag);
    for (const [name, value] of Object.entries(attributes))
      element.setAttribute(name, value);
    return element;
  }

  function boxShape(box, className, caption, interactive = false) {
    const [left, top, right, bottom] = box.box;
    const group = svg("g", { class: className });
    const rectangle = svg("rect", {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
      "vector-effect": "non-scaling-stroke",
    });
    if (interactive) rectangle.dataset.boxId = box.id;
    group.append(rectangle);
    const scale =
      editor.document.frame.width /
      Math.max($("#annotation-canvas").getBoundingClientRect().width, 200);
    const text = svg("text", {
      x: left + 3 * scale,
      y: Math.max(14 * scale, top - 5 * scale),
      "font-size": 12 * scale,
      "paint-order": "stroke",
      "stroke-width": 3 * scale,
    });
    text.textContent = caption;
    group.append(text);
    return group;
  }

  function paintCanvas() {
    if (!editor.document) return;
    const layer = $("#annotation-box-layer");
    layer.replaceChildren();
    editor.boxes.forEach((box, index) => {
      const selected = box.id === editor.selected;
      const group = boxShape(
        box,
        `annotation-box ${selected ? "selected" : ""}`,
        `${index + 1} · ${box.label}`,
        true,
      );
      if (selected && editor.tool === "select") {
        const [x1, y1, x2, y2] = box.box;
        const size =
          (9 * editor.document.frame.width) /
          Math.max($("#annotation-canvas").getBoundingClientRect().width, 200);
        for (const [corner, x, y] of [
          ["nw", x1, y1],
          ["ne", x2, y1],
          ["sw", x1, y2],
          ["se", x2, y2],
        ]) {
          const handle = svg("rect", {
            x: x - size / 2,
            y: y - size / 2,
            width: size,
            height: size,
            class: `annotation-handle ${corner}`,
            "vector-effect": "non-scaling-stroke",
          });
          handle.dataset.boxId = box.id;
          handle.dataset.corner = corner;
          group.append(handle);
        }
      }
      layer.append(group);
    });
    const proposals = $("#annotation-proposal-layer");
    proposals.replaceChildren();
    if ($("#annotation-show-proposals").checked) {
      for (const proposal of pending())
        proposals.append(
          boxShape(proposal, "annotation-proposal-box", `? ${proposal.label}`),
        );
    }
  }

  function renderBoxes() {
    paintCanvas();
    const list = $("#annotation-boxes");
    list.replaceChildren();
    $("#annotation-box-count").textContent = editor.boxes.length;
    if (!editor.boxes.length)
      list.append(
        node(
          "p",
          "field-hint",
          "No labels yet. Draw a box, accept a proposal, or validate an empty frame after inspection.",
        ),
      );
    editor.boxes.forEach((box, index) => {
      const button = node(
        "button",
        `annotation-box-item${box.id === editor.selected ? " selected" : ""}`,
      );
      button.type = "button";
      button.setAttribute("aria-pressed", String(box.id === editor.selected));
      const proposal = editor.document.suggestions.find(
        (item) => item.id === box.suggestion_id,
      );
      const origin = proposal
        ? `${proposalOrigin(proposal)} · human ${decisionFor(proposal)}`
        : "Manual label";
      button.append(
        node("strong", "", `${index + 1} · ${box.label}`),
        node("span", "small muted", origin),
      );
      button.addEventListener("click", () => {
        editor.selected = box.id;
        editor.tool = "select";
        renderBoxes();
        updateStatus();
      });
      list.append(button);
    });
    renderProperties();
  }

  function renderProperties() {
    const box = selectedBox();
    $("#annotation-box-properties").hidden = !box;
    if (!box) return;
    $("#annotation-box-class").value = box.label;
    ["x1", "y1", "x2", "y2"].forEach((coordinate, index) => {
      const input = $(`#annotation-${coordinate}`);
      input.value = Number(box.box[index].toFixed(1));
      input.max =
        index % 2 ? editor.document.frame.height : editor.document.frame.width;
    });
  }

  function renderProposals() {
    const list = $("#annotation-proposals");
    list.replaceChildren();
    const suggestions = editor.document.suggestions || [];
    if (!suggestions.length)
      list.append(
        node(
          "p",
          "field-hint",
          "No proposals for this frame. Import a saved detector output or draw your labels directly.",
        ),
      );
    for (const proposal of suggestions) {
      const decision = decisionFor(proposal);
      const item = node("article", `annotation-proposal ${decision}`);
      const heading = node("div", "annotation-proposal-heading");
      heading.append(
        node("strong", "", proposal.label),
        node(
          "span",
          "annotation-proposal-origin",
          proposal.kind === "multimodal"
            ? "Multimodal proposal"
            : proposal.kind === "imported"
              ? "Imported annotation"
              : "Detector proposal",
        ),
        node(
          "span",
          "small",
          decision === "pending"
            ? "Awaiting human review"
            : `Human: ${decision}${editor.dirty ? " · unsaved" : ""}`,
        ),
      );
      item.append(heading);
      const recommendation = proposal.metadata?.recommendation;
      if (recommendation)
        item.append(
          node(
            "p",
            `annotation-recommendation ${recommendation}`,
            `Model recommendation: ${recommendation}`,
          ),
        );
      if (proposal.metadata?.reason)
        item.append(node("p", "small", proposal.metadata.reason));
      if (proposal.metadata?.target_box_id)
        item.append(
          node(
            "p",
            "field-hint",
            "Review of an existing label. Accepting updates that label and replaces any prior proposal decision; earlier decisions remain in saved revisions. Rejecting this proposal leaves the label unchanged.",
          ),
        );
      if (
        proposal.metadata?.target_box_id &&
        !editor.boxes.some((box) => box.id === proposal.metadata.target_box_id)
      )
        item.append(
          node(
            "p",
            "field-hint",
            "The reviewed label was removed. Reject this proposal and request a new review after saving your labels.",
          ),
        );
      if (proposal.metadata?.score != null)
        item.append(
          node(
            "p",
            "field-hint",
            `Detector confidence: ${Number(proposal.metadata.score).toFixed(3)}`,
          ),
        );
      const actions = node("div", "annotation-proposal-actions");
      if (decision === "pending" || decision === "rejected") {
        const accept = node(
          "button",
          "button button-secondary",
          recommendation === "reject"
            ? proposal.metadata?.target_box_id
              ? "Keep box anyway"
              : "Add box anyway"
            : "Accept proposal",
        );
        accept.type = "button";
        accept.disabled =
          editor.busy ||
          Boolean(
            proposal.metadata?.target_box_id &&
              !editor.boxes.some(
                (box) => box.id === proposal.metadata.target_box_id,
              ),
          );
        accept.addEventListener("click", () => acceptProposal(proposal));
        const reject = node("button", "text-button", "Reject proposal");
        reject.type = "button";
        reject.disabled = editor.busy;
        reject.addEventListener("click", () => {
          if (editor.busy) return;
          editor.decisions[proposal.id] = "rejected";
          changed();
          renderProposals();
          paintCanvas();
        });
        actions.append(accept);
        if (decision === "pending") actions.append(reject);
      } else {
        const undo = node(
          "button",
          "text-button",
          "Reject proposal and remove its label",
        );
        undo.type = "button";
        undo.disabled = editor.busy;
        undo.addEventListener("click", () => {
          if (editor.busy) return;
          if (
            editor.boxes.some((box) => box.suggestion_id === proposal.id) &&
            !window.confirm(
              "Reject this proposal and remove its current label?",
            )
          )
            return;
          editor.boxes = editor.boxes.filter(
            (box) => box.suggestion_id !== proposal.id,
          );
          editor.decisions[proposal.id] = "rejected";
          changed();
          renderBoxes();
          renderProposals();
        });
        actions.append(undo);
      }
      const provenance = node("details", "annotation-proposal-provenance");
      provenance.append(
        node("summary", "", "Proposal provenance"),
        node(
          "pre",
          "",
          JSON.stringify(
            {
              id: proposal.id,
              kind: proposal.kind,
              box: proposal.box,
              metadata: proposal.metadata,
            },
            null,
            2,
          ),
        ),
      );
      item.append(actions, provenance);
      list.append(item);
    }
  }

  function acceptProposal(proposal) {
    if (editor.busy) return;
    const existing = editor.boxes.find(
      (box) => box.id === proposal.metadata?.target_box_id,
    );
    if (existing) {
      if (existing.suggestion_id && existing.suggestion_id !== proposal.id)
        editor.decisions[existing.suggestion_id] = "rejected";
      existing.label = proposal.label;
      existing.box = [...proposal.box];
      existing.suggestion_id = proposal.id;
      editor.selected = existing.id;
    } else {
      const box = {
        id: crypto.randomUUID(),
        label: proposal.label,
        box: [...proposal.box],
        suggestion_id: proposal.id,
      };
      editor.boxes.push(box);
      editor.selected = box.id;
    }
    editor.decisions[proposal.id] = "accepted";
    changed();
    renderBoxes();
    renderProposals();
  }

  function removeSelected() {
    const box = selectedBox();
    if (!box || editor.busy) return;
    if (box.suggestion_id) editor.decisions[box.suggestion_id] = "rejected";
    editor.boxes = editor.boxes.filter((item) => item.id !== box.id);
    editor.selected = null;
    changed();
    renderBoxes();
    renderProposals();
  }

  async function save(status) {
    if (!editor.document || editor.busy) return;
    editor.boxes.forEach(updateBoxDecision);
    error(null);
    editor.busy = true;
    updateStatus();
    renderProposals();
    try {
      const result = await api(url(), {
        method: "PUT",
        body: JSON.stringify({
          expected_revision: editor.document.revision,
          boxes: editor.boxes.map(({ id, label, box, suggestion_id }) => ({
            id,
            label,
            box,
            suggestion_id: suggestion_id || null,
          })),
          decisions: editor.decisions,
          status,
          reviewer: $("#annotation-reviewer").value.trim(),
          notes: $("#annotation-notes").value.trim(),
        }),
      });
      try {
        localStorage.setItem("iris.reviewer", result.reviewer || "");
      } catch {
        /* Optional storage. */
      }
      applyDocument(result);
      notify(
        status === "validated"
          ? `Frame validated · revision ${result.revision}.`
          : `Draft saved · revision ${result.revision}.`,
      );
    } catch (failure) {
      reportFailure(failure);
    } finally {
      finishRequest();
    }
  }

  function threshold() {
    const input = $("#annotation-threshold");
    if (!input.reportValidity() || input.value === "")
      throw new Error("Enter a confidence threshold between 0 and 1.");
    return Number(input.value);
  }

  async function importProposals() {
    if (editor.dirty || editor.busy) return;
    editor.busy = true;
    updateStatus();
    error(null);
    try {
      const result = await api(url("suggestions"), {
        method: "POST",
        body: JSON.stringify({
          prediction_id: $("#annotation-prediction").value,
          threshold: threshold(),
          expected_revision: editor.document.revision,
        }),
      });
      applyDocument(result);
      notify(
        "Detector proposals imported. Review each proposal before validation.",
      );
    } catch (failure) {
      reportFailure(failure);
    } finally {
      finishRequest();
    }
  }

  function providerReady() {
    if (!editor.provider) return false;
    return editor.provider.local
      ? editor.provider.status === "ready"
      : ["configured", "ready"].includes(editor.provider.status);
  }

  function invalidatePreview() {
    editor.previewGeneration++;
    editor.preview = null;
    clearTimeout(editor.previewTimer);
    editor.previewTimer = null;
    $("#annotation-api-preview").hidden = true;
    $("#annotation-api-consent").checked = false;
    $("#annotation-api-images").replaceChildren();
  }

  function selectedProvider() {
    return editor.catalog?.providers.find(
      (provider) => provider.id === $("#annotation-provider").value,
    );
  }

  function renderProviderChoices(preferredProvider, preferredModel) {
    const local = $("#annotation-execution").value === "local";
    const providers = (editor.catalog?.providers || []).filter(
      (provider) => provider.local === local,
    );
    const selector = $("#annotation-provider");
    selector.replaceChildren();
    for (const provider of providers)
      selector.append(new Option(provider.name, provider.id));
    if (providers.some((provider) => provider.id === preferredProvider))
      selector.value = preferredProvider;
    renderModelChoices(preferredModel);
  }

  function renderModelChoices(preferredModel) {
    const provider = selectedProvider();
    const selector = $("#annotation-model");
    selector.replaceChildren();
    for (const model of provider?.models || [])
      selector.append(new Option(model.label || model.id, model.id));
    if (provider?.models.some((model) => model.id === preferredModel))
      selector.value = preferredModel;
    renderProviderStatus();
  }

  function renderProviderStatus() {
    const provider = selectedProvider();
    const model = provider?.models.find(
      (entry) => entry.id === $("#annotation-model").value,
    );
    editor.provider = model
      ? { ...provider, ...model, id: provider.id, model: model.id }
      : null;
    const selected = editor.provider;
    const remote = $("#annotation-execution").value === "api";
    const ready = providerReady();
    $("#annotation-provider-status").textContent = !selected
      ? "No models available for this execution mode."
      : ready
        ? `${remote ? "API key configured · connection not tested" : "Ready"} · ${model.label || model.id}${selected.endpoint ? ` · ${selected.endpoint}` : ""}`
        : `Unavailable · ${selected.reason || selected.status || "Model not configured"}. Manual annotation remains available.`;
    $("#annotation-provider-setup").hidden = remote || ready;
    $("#annotation-api-setup").hidden = !remote || ready;
    $("#annotation-api-notice").hidden = !remote;
    $("#annotation-local-command").textContent =
      `ollama pull ${remote ? "qwen3-vl:4b-instruct" : model?.id || "qwen3-vl:4b-instruct"}`;
    $("#annotation-assist").textContent = remote
      ? "Preview API request"
      : "Request local review";
    updateStatus();
  }

  async function refreshProvider() {
    if (editor.busy) return;
    const request = ++editor.catalogRequest;
    const previousProvider = $("#annotation-provider").value;
    const previousModel = $("#annotation-model").value;
    invalidatePreview();
    editor.catalogLoading = true;
    $("#annotation-provider-refresh").disabled = true;
    updateStatus();
    try {
      const catalog = await api("/api/annotation-providers");
      if (request !== editor.catalogRequest) return;
      editor.catalog = catalog;
      if (!previousProvider) {
        const initial = catalog.providers.find(
          (provider) => provider.id === catalog.default_provider,
        );
        $("#annotation-execution").value = initial?.local === false ? "api" : "local";
      }
      renderProviderChoices(
        previousProvider || catalog.default_provider,
        previousModel || catalog.default_model,
      );
    } catch (failure) {
      if (request !== editor.catalogRequest) return;
      editor.provider = null;
      $("#annotation-provider-status").textContent =
        `Unavailable · ${failure.message}`;
    } finally {
      if (request === editor.catalogRequest) {
        editor.catalogLoading = false;
        $("#annotation-provider-refresh").disabled = false;
        $("#annotation-provider").disabled = !editor.catalog;
        $("#annotation-model").disabled = !editor.catalog;
        updateStatus();
      }
    }
  }

  function assistancePayload() {
    return {
      expected_revision: editor.document.revision,
      prediction_id:
        $("#annotation-assist-source").value === "prediction"
          ? $("#annotation-prediction").value
          : null,
      threshold: threshold(),
      instructions: $("#annotation-instructions").value.trim(),
      provider: $("#annotation-provider").value,
      model: $("#annotation-model").value,
    };
  }

  function formatCost(value, currency = "USD") {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 10,
    }).format(value);
  }

  function updatePreviewStatus() {
    const preview = editor.preview;
    if (!preview) return;
    const expired = Date.now() >= Date.parse(preview.expires_at);
    const images = [...$("#annotation-api-images").querySelectorAll("img")];
    const imagesReady = images.length > 0 && images.every(
      (image) => image.complete && image.naturalWidth > 0,
    );
    $("#annotation-api-expiry").textContent = expired
      ? "This preview has expired. Prepare a new preview before sending."
      : !imagesReady
        ? "Waiting for all preview images to load. If an image fails to load, discard this preview and try again."
        : `Approval expires at ${new Date(preview.expires_at).toLocaleTimeString()}. Changing any review settings discards this preview.`;
    $("#annotation-api-confirm").disabled =
      editor.busy || editor.dirty || expired || !imagesReady ||
      !$("#annotation-api-consent").checked || assistActive();
    $("#annotation-api-consent").disabled = editor.busy || expired;
    $("#annotation-api-cancel").disabled = editor.busy;
  }

  function renderPreview(preview, payload, frameId) {
    const amount = preview.cost?.upper_bound_usd;
    if (
      !preview.id || preview.frame_id !== frameId ||
      preview.provider !== payload.provider || preview.model !== payload.model ||
      typeof amount !== "number" || !Number.isFinite(amount) || amount < 0 ||
      (preview.cost.currency && preview.cost.currency !== "USD") ||
      !Number.isFinite(Date.parse(preview.expires_at)) ||
      !preview.images?.length
    ) throw new Error("The API preview is incomplete. No images were sent.");
    editor.preview = { ...preview, payload: clone(payload), frameId };
    $("#annotation-api-preview").hidden = false;
    $("#annotation-api-consent").checked = false;
    $("#annotation-api-destination").textContent =
      `${selectedProvider()?.name || preview.provider} · ${preview.model} · ${preview.endpoint}`;
    $("#annotation-api-data-notice").textContent =
      `${preview.deployment_scope ? `Deployment scope: ${preview.deployment_scope}. ` : ""}${preview.data_notice || "Your images and review focus will leave this computer and be processed by the selected provider."}`;
    $("#annotation-api-preview-details").textContent =
      `${preview.images.length} outbound images · ${preview.candidate_count} candidate${preview.candidate_count === 1 ? "" : "s"}. These exact images and your review focus will be sent only after approval.`;
    const gallery = $("#annotation-api-images");
    gallery.replaceChildren();
    for (const [index, item] of preview.images.entries()) {
      const source = new URL(item.url, window.location.href);
      if (source.origin !== window.location.origin)
        throw new Error("The preview image must be stored by IRIS. No images were sent.");
      const figure = node("figure");
      const image = node("img");
      image.alt = item.label || (item.kind === "frame" ? "Full frame sent to the provider" : `Candidate crop ${index}`);
      image.addEventListener("load", updatePreviewStatus);
      image.addEventListener("error", updatePreviewStatus);
      image.src = source.href;
      figure.append(image, node("figcaption", "field-hint", image.alt));
      gallery.append(figure);
    }
    const cost = formatCost(amount, preview.cost.currency || "USD");
    $("#annotation-api-cost").textContent =
      `Maximum approved charge: ${cost}. ${preview.cost.estimate_label || "Conservative estimate for this single request."}`;
    $("#annotation-api-cost-basis").textContent = preview.cost.basis || "";
    const pricing = $("#annotation-api-pricing");
    pricing.hidden = true;
    if (preview.cost.pricing_source) {
      try {
        const link = new URL(preview.cost.pricing_source);
        if (link.protocol === "https:") {
          pricing.href = link.href;
          pricing.hidden = false;
        }
      } catch { /* Pricing is optional; the amount is required. */ }
    }
    $("#annotation-api-consent-label").textContent =
      `I approve sending these images to ${selectedProvider()?.name || preview.provider} and a charge of up to ${cost} for this review.`;
    clearTimeout(editor.previewTimer);
    editor.previewTimer = setTimeout(
      updatePreviewStatus,
      Math.max(0, Math.min(Date.parse(preview.expires_at) - Date.now() + 20, 2147483647)),
    );
    updatePreviewStatus();
    $("#annotation-api-preview").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function requestAssistance() {
    if (editor.dirty || editor.busy || !providerReady()) return;
    invalidatePreview();
    const generation = editor.previewGeneration;
    const frameId = editor.frameId;
    const remote = !editor.provider.local;
    editor.busy = true;
    updateStatus();
    error(null);
    try {
      const payload = assistancePayload();
      const result = await api(url(remote ? "assist/preview" : "assist"), {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (generation !== editor.previewGeneration || frameId !== editor.frameId) return;
      if (remote) {
        renderPreview(result, payload, frameId);
      } else {
        notify("Local multimodal review queued. Your labels remain subject to human validation.");
        await refreshJobs();
      }
    } catch (failure) {
      invalidatePreview();
      reportFailure(failure);
    } finally {
      finishRequest();
    }
  }

  async function confirmApiReview() {
    const preview = editor.preview;
    updatePreviewStatus();
    if (!preview || $("#annotation-api-confirm").disabled) return;
    if (
      preview.frameId !== editor.frameId ||
      JSON.stringify(preview.payload) !== JSON.stringify(assistancePayload())
    ) {
      invalidatePreview();
      error("The review settings changed. Prepare a new preview before sending.");
      return;
    }
    editor.busy = true;
    updateStatus();
    error(null);
    try {
      await api(url("assist"), {
        method: "POST",
        body: JSON.stringify({
          ...preview.payload,
          preview_id: preview.id,
          allow_external: true,
          max_cost_usd: preview.cost.upper_bound_usd,
        }),
      });
      invalidatePreview();
      notify("API review queued with your approved images and cost limit. Your labels remain subject to human validation.");
      await refreshJobs();
    } catch (failure) {
      invalidatePreview();
      reportFailure(failure);
    } finally {
      finishRequest();
    }
  }

  function renderHistory() {
    const document = editor.document;
    $("#annotation-provenance").textContent =
      `Frame SHA-256: ${document.frame_sha256 || document.frame.sha256} · Taxonomy: ${document.taxonomy.id}`;
    const imported = sourceFor(document.frame)?.metadata?.dataset_import;
    const source = $("#annotation-import-provenance");
    source.replaceChildren();
    source.hidden = !imported;
    if (imported) {
      source.append(
        node("h3", "", "Imported dataset source"),
        node("p", "field-hint", "Source annotations are proposals. Review the entire image, including any missing target objects, before validating."),
        node("pre", "", JSON.stringify(imported, null, 2)),
      );
    }
    const list = $("#annotation-history");
    list.replaceChildren();
    if (!document.history?.length)
      list.append(node("p", "field-hint", "No saved annotation revisions."));
    for (const revision of document.history || []) {
      const button = node(
        "button",
        "annotation-history-row",
        `Revision ${revision.revision} · ${revision.status} · ${revision.reviewer || "No reviewer"} · ${new Date(revision.created_at).toLocaleString()}`,
      );
      button.type = "button";
      button.addEventListener("click", () =>
        inspectRevision(revision.revision),
      );
      list.append(button);
    }
    const assistance = node(
      "button",
      "text-button",
      "View model review records",
    );
    assistance.type = "button";
    assistance.addEventListener("click", async () => {
      try {
        showRecord("Model review records", await api(url("assistance")));
      } catch (failure) {
        reportFailure(failure);
      }
    });
    list.append(assistance);
  }

  async function inspectRevision(revision) {
    try {
      showRecord(
        `Saved revision ${revision} · read only`,
        await api(url(`annotation/revisions/${revision}`)),
      );
    } catch (failure) {
      reportFailure(failure);
    }
  }

  function showRecord(title, record) {
    const dialog = node("dialog", "annotation-record-dialog");
    const heading = node("h2", "", title);
    heading.id = "annotation-record-title";
    dialog.setAttribute("aria-labelledby", heading.id);
    const close = node("button", "button button-secondary", "Close");
    close.type = "button";
    close.addEventListener("click", () => dialog.close());
    dialog.append(
      heading,
      node(
        "p",
        "field-hint",
        "Stored provenance is shown exactly as recorded. Viewing a record does not replace your current work.",
      ),
      node("pre", "", JSON.stringify(record, null, 2)),
      close,
    );
    dialog.addEventListener("close", () => dialog.remove());
    document.body.append(dialog);
    dialog.showModal();
  }

  function canvasPoint(event) {
    const canvas = $("#annotation-canvas");
    const matrix = canvas.getScreenCTM();
    if (!matrix) return null;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      matrix.inverse(),
    );
    return [
      Math.max(0, Math.min(editor.document.frame.width, point.x)),
      Math.max(0, Math.min(editor.document.frame.height, point.y)),
    ];
  }

  function pointerDown(event) {
    if (!editor.document || editor.busy || editor.loading || event.button !== 0)
      return;
    const point = canvasPoint(event);
    if (!point) return;
    invalidatePreview();
    event.preventDefault();
    const canvas = $("#annotation-canvas");
    canvas.focus({ preventScroll: true });
    const id = event.target.dataset.boxId;
    if (editor.tool === "draw") {
      editor.selected = null;
      editor.drag = {
        kind: "draw",
        start: point,
        current: point,
        pointer: event.pointerId,
      };
    } else if (id) {
      editor.selected = id;
      editor.drag = {
        kind: event.target.dataset.corner || "move",
        start: point,
        original: [...selectedBox().box],
        pointer: event.pointerId,
      };
    } else {
      editor.selected = null;
      renderBoxes();
      return;
    }
    canvas.setPointerCapture(event.pointerId);
    renderBoxes();
  }

  function pointerMove(event) {
    const drag = editor.drag;
    if (!drag || drag.pointer !== event.pointerId) return;
    const point = canvasPoint(event);
    drag.current = point;
    if (drag.kind === "draw") {
      const box = [
        Math.min(drag.start[0], point[0]),
        Math.min(drag.start[1], point[1]),
        Math.max(drag.start[0], point[0]),
        Math.max(drag.start[1], point[1]),
      ];
      $("#annotation-drawing-layer").replaceChildren(
        boxShape(
          { box },
          "annotation-box selected",
          $("#annotation-class").value,
        ),
      );
      return;
    }
    const box = selectedBox();
    if (!box) return;
    const [x1, y1, x2, y2] = drag.original;
    const { width, height } = editor.document.frame;
    if (drag.kind === "move") {
      const dx = Math.max(-x1, Math.min(width - x2, point[0] - drag.start[0]));
      const dy = Math.max(-y1, Math.min(height - y2, point[1] - drag.start[1]));
      box.box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy];
    } else {
      const left = drag.kind.includes("w") ? Math.min(point[0], x2 - 1) : x1;
      const right = drag.kind.includes("e") ? Math.max(point[0], x1 + 1) : x2;
      const top = drag.kind.includes("n") ? Math.min(point[1], y2 - 1) : y1;
      const bottom = drag.kind.includes("s") ? Math.max(point[1], y1 + 1) : y2;
      box.box = [left, top, right, bottom];
    }
    paintCanvas();
    renderProperties();
  }

  function pointerEnd(event, cancel = false) {
    const drag = editor.drag;
    if (!drag || (event?.pointerId != null && drag.pointer !== event.pointerId))
      return;
    const canvas = $("#annotation-canvas");
    if (canvas.hasPointerCapture(drag.pointer))
      canvas.releasePointerCapture(drag.pointer);
    $("#annotation-drawing-layer").replaceChildren();
    if (drag.kind === "draw" && !cancel) {
      const point = drag.current || drag.start;
      const coords = [
        Math.min(drag.start[0], point[0]),
        Math.min(drag.start[1], point[1]),
        Math.max(drag.start[0], point[0]),
        Math.max(drag.start[1], point[1]),
      ];
      if (coords[2] - coords[0] >= 1 && coords[3] - coords[1] >= 1) {
        const box = {
          id: crypto.randomUUID(),
          label: $("#annotation-class").value,
          box: coords,
          suggestion_id: null,
        };
        editor.boxes.push(box);
        editor.selected = box.id;
        editor.tool = "select";
        changed();
      }
    } else if (drag.kind !== "draw") {
      const box = selectedBox();
      if (box && cancel) box.box = drag.original;
      else if (
        box &&
        JSON.stringify(box.box) !== JSON.stringify(drag.original)
      ) {
        updateBoxDecision(box);
        changed();
      }
    }
    editor.drag = null;
    renderBoxes();
    renderProposals();
    updateStatus();
  }

  function applyCoordinates() {
    const box = selectedBox();
    if (!box || editor.busy) return;
    const inputs = ["x1", "y1", "x2", "y2"].map((coordinate) =>
      $(`#annotation-${coordinate}`),
    );
    if (inputs.some((input) => input.value === "" || !input.reportValidity()))
      return;
    const values = inputs.map((input) => Number(input.value));
    if (values[2] <= values[0] || values[3] <= values[1]) {
      error(
        "Right must be greater than left, and bottom must be greater than top.",
      );
      return;
    }
    box.box = values;
    updateBoxDecision(box);
    error(null);
    changed();
    renderBoxes();
    renderProposals();
  }

  async function refreshCurrent() {
    if (!editor.frameId || !editor.document) return;
    if (editor.dirty || editor.busy || editor.drag) {
      if (editor.busy) editor.refreshAfterRequest = true;
      editor.remoteUpdate = true;
      updateStatus();
      return;
    }
    const frameId = editor.frameId;
    const request = editor.request;
    try {
      const document = await api(url());
      if (
        frameId === editor.frameId &&
        request === editor.request &&
        !editor.dirty &&
        !editor.busy &&
        !editor.drag
      )
        applyDocument(document);
    } catch (failure) {
      if (frameId === editor.frameId) reportFailure(failure);
    }
  }

  $("#annotation-frame").addEventListener("change", (event) =>
    navigate(event.target.value),
  );
  for (const [selector, offset] of [
    ["#annotation-previous", -1],
    ["#annotation-next", 1],
  ]) {
    $(selector).addEventListener("click", () =>
      navigate(
        editor.frames[
          editor.frames.findIndex((frame) => frame.id === editor.frameId) +
            offset
        ]?.id,
      ),
    );
  }
  for (const name of ["select", "draw"])
    $(`#annotation-${name}-tool`).addEventListener("click", () => {
      editor.tool = name;
      pointerEnd(null, true);
      paintCanvas();
      updateStatus();
    });
  $("#annotation-save").addEventListener("click", () => save("draft"));
  $("#annotation-validate").addEventListener("click", () => save("validated"));
  $("#annotation-reviewer").addEventListener("input", changed);
  $("#annotation-notes").addEventListener("input", changed);
  $("#annotation-show-proposals").addEventListener("change", paintCanvas);
  for (const selector of [
    "#annotation-prediction",
    "#annotation-assist-source",
    "#annotation-threshold",
    "#annotation-instructions",
  ]) {
    $(selector).addEventListener("input", () => {
      invalidatePreview();
      updateStatus();
    });
  }
  $("#annotation-execution").addEventListener("change", () => {
    invalidatePreview();
    renderProviderChoices();
  });
  $("#annotation-provider").addEventListener("change", () => {
    invalidatePreview();
    renderModelChoices();
  });
  $("#annotation-model").addEventListener("change", () => {
    invalidatePreview();
    renderProviderStatus();
  });
  $("#annotation-api-consent").addEventListener("change", updatePreviewStatus);
  $("#annotation-api-confirm").addEventListener("click", confirmApiReview);
  $("#annotation-api-cancel").addEventListener("click", invalidatePreview);
  $("#annotation-import").addEventListener("click", importProposals);
  $("#annotation-assist").addEventListener("click", requestAssistance);
  $("#annotation-provider-refresh").addEventListener("click", refreshProvider);
  $("#annotation-reload").addEventListener("click", () => {
    if (discardAllowed()) loadFrame(editor.frameId);
  });
  $("#annotation-refresh").addEventListener("click", () => {
    if (discardAllowed()) loadFrame(editor.frameId);
  });
  $("#annotation-remove-box").addEventListener("click", removeSelected);
  $("#annotation-apply-coordinates").addEventListener(
    "click",
    applyCoordinates,
  );
  $("#annotation-box-class").addEventListener("change", (event) => {
    const box = selectedBox();
    if (!box || editor.busy) return;
    box.label = event.target.value;
    updateBoxDecision(box);
    changed();
    renderBoxes();
    renderProposals();
  });
  $("#annotation-add-box").addEventListener("click", () => {
    if (!editor.document || editor.busy) return;
    const { width, height } = editor.document.frame;
    const box = {
      id: crypto.randomUUID(),
      label: $("#annotation-class").value,
      box: [width / 4, height / 4, (width * 3) / 4, (height * 3) / 4],
      suggestion_id: null,
    };
    editor.boxes.push(box);
    editor.selected = box.id;
    editor.tool = "select";
    changed();
    renderBoxes();
    $("#annotation-x1").focus();
  });
  const canvas = $("#annotation-canvas");
  canvas.addEventListener("pointerdown", pointerDown);
  canvas.addEventListener("pointermove", pointerMove);
  canvas.addEventListener("pointerup", (event) => pointerEnd(event));
  canvas.addEventListener("pointercancel", (event) => pointerEnd(event, true));
  canvas.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      pointerEnd(null, true);
      editor.tool = "select";
      updateStatus();
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      removeSelected();
    }
  });
  window.addEventListener("beforeunload", (event) => {
    if (editor.dirty || editor.busy) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  window.addEventListener("iris:before-session", (event) => {
    if (!discardAllowed()) event.preventDefault();
  });
  window.addEventListener("iris:session", () => {
    if (editor.sessionId === state.sessionId) return;
    invalidatePreview();
    editor.sessionId = state.sessionId;
    editor.frameId = null;
    editor.document = null;
    editor.dirty = false;
    editor.request++;
    syncFrames();
  });
  window.addEventListener("iris:frames", syncFrames);
  window.addEventListener("iris:workspace", (event) => {
    if (event.detail.name === "annotation") {
      syncFrames();
      paintCanvas();
      refreshProvider();
    }
  });
  window.addEventListener("iris:jobs", () => {
    let completed = false;
    for (const job of state.jobs) {
      const old = editor.jobStatuses.get(job.id);
      editor.jobStatuses.set(job.id, job.status);
      if (
        old !== job.status &&
        !isActive(job) &&
        (job.kind === "infer" ||
          (job.kind === "assist" && job.params?.frame_id === editor.frameId))
      )
        completed = true;
    }
    if (completed) refreshCurrent();
    updateStatus();
  });
  window.addEventListener("resize", paintCanvas);
  editor.sessionId = state.sessionId;
  syncFrames();
  refreshProvider();
})();
