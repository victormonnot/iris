"use strict";

(() => {
  const view = {
    visible: false,
    datasets: [],
    models: [],
    chosen: new Set(),
    touched: false,
    history: [],
    activeId: null,
    detail: null,
    frameId: null,
    references: null,
    busy: false,
    auditBusy: false,
    referenceBusy: false,
    catalogRequest: 0,
    historyRequest: 0,
    detailRequest: 0,
    jobStatuses: new Map(),
  };
  const percent = (value) =>
    Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "N/A";
  const count = (value) => (Number.isFinite(value) ? String(value) : "N/A");
  const safe = (value) => encodeURIComponent(value);
  const modelName = (id) =>
    view.detail?.config?.model_names?.[id] ||
    view.models.find((model) => model.id === id)?.name ||
    id;
  const frameId = (frame) => frame.frame_id || frame.id;
  const frames = () => view.detail?.frames || [];
  const split = () => view.detail?.split || "val";
  const config = () => view.detail?.config || {};
  function error(selector, value) {
    const target = $(selector);
    target.textContent = value?.message || "";
    target.hidden = !value;
  }
  function complete() {
    const detail = view.detail;
    return (
      detail?.job?.status === "succeeded" &&
      detail.models?.length === detail.model_ids.length &&
      detail.models.every(
        (model) => model.metrics?.summary?.frame_count === frames().length,
      )
    );
  }
  function updateLaunch() {
    const dataset = view.datasets.find(
      (item) => item.id === $("#evaluation-dataset").value,
    );
    const size = dataset?.summary?.split_counts?.val || 0;
    $("#evaluation-dataset-context").textContent = dataset
      ? `${size} validation frames · ${dataset.summary.split_counts.test || 0} reserved test frames · frozen labels and scene groups`
      : "Freeze a release with validation data in Dataset & training first.";
    $("#evaluation-selection").textContent =
      `${size} validation frames · ${view.chosen.size} models · CPU`;
    $("#evaluation-start").disabled =
      view.busy || !size || !view.chosen.size || view.chosen.size > 2;
    $("#evaluation-start").textContent = view.busy
      ? "Queuing evaluation…"
      : "Evaluate on validation →";
  }
  function renderModels() {
    const container = $("#evaluation-models");
    container.replaceChildren();
    for (const model of view.models) {
      const ready = model.status === "ready";
      const card = node(
        "label",
        `model-card${ready ? " ready" : " unavailable"}`,
      );
      const input = node("input");
      input.type = "checkbox";
      input.checked = view.chosen.has(model.id);
      input.disabled = !ready;
      input.setAttribute("aria-label", `Evaluate ${model.name}`);
      input.addEventListener("change", () => {
        view.touched = true;
        if (input.checked) view.chosen.add(model.id);
        else view.chosen.delete(model.id);
        updateLaunch();
      });
      const content = node("span", "model-card-content");
      content.append(node("strong", "model-card-title", model.name));
      content.append(
        node(
          "span",
          "model-description",
          ready
            ? "Ready · local verified weights"
            : model.reason || "Local dependencies are unavailable",
        ),
      );
      card.append(input, content);
      container.append(card);
    }
    if (!view.models.length)
      container.append(
        node(
          "p",
          "field-hint",
          "No models available. Check Model comparison for local setup.",
        ),
      );
  }
  async function refreshCatalogs() {
    const request = ++view.catalogRequest;
    try {
      const [datasets, models] = await Promise.all([
        api("/api/datasets"),
        api("/api/models"),
      ]);
      if (request !== view.catalogRequest) return;
      const previous = $("#evaluation-dataset").value;
      view.datasets = datasets;
      view.models = models;
      const select = $("#evaluation-dataset");
      select.replaceChildren();
      for (const dataset of datasets)
        select.append(new Option(dataset.name, dataset.id));
      if (!datasets.length)
        select.append(new Option("Freeze a dataset first", ""));
      if (datasets.some((item) => item.id === previous))
        select.value = previous;
      select.disabled = !datasets.length;
      view.chosen = new Set(
        models
          .filter(
            (model) =>
              model.status === "ready" &&
              (!view.touched || view.chosen.has(model.id)),
          )
          .slice(0, 2)
          .map((model) => model.id),
      );
      renderModels();
      updateLaunch();
      error("#evaluation-error", null);
    } catch (failure) {
      if (request !== view.catalogRequest) return;
      view.datasets = [];
      view.models = [];
      view.chosen.clear();
      $("#evaluation-dataset").replaceChildren(
        new Option("Catalog unavailable", ""),
      );
      $("#evaluation-dataset").disabled = true;
      renderModels();
      updateLaunch();
      error("#evaluation-error", failure);
    }
  }
  async function refreshHistory() {
    const request = ++view.historyRequest;
    try {
      const history = await api("/api/evaluations");
      if (request !== view.historyRequest) return;
      view.history = history;
      if (!history.some((item) => item.id === view.activeId))
        view.activeId = history[0]?.id || null;
      const select = $("#evaluation-history");
      select.replaceChildren();
      for (const item of history)
        select.append(
          new Option(
            `${item.name} · ${item.split === "test" ? "Test audit" : "Validation"} · ${item.job?.status || "Unknown status"}`,
            item.id,
          ),
        );
      if (!history.length) select.append(new Option("No evaluations yet", ""));
      select.value = view.activeId || "";
      select.disabled = !history.length;
      $("#evaluation-history-count").textContent = String(history.length);
      $("#evaluation-empty").hidden = Boolean(history.length);
      if (!history.length) $("#evaluation-detail").hidden = true;
      error("#evaluation-history-error", null);
      if (view.activeId) await loadDetail(view.activeId);
    } catch (failure) {
      if (request === view.historyRequest)
        error("#evaluation-history-error", failure);
    }
  }
  async function loadDetail(id) {
    const request = ++view.detailRequest;
    try {
      const detail = await api(`/api/evaluations/${safe(id)}`);
      if (request !== view.detailRequest || view.activeId !== id) return;
      error("#evaluation-history-error", null);
      if (JSON.stringify(detail) === JSON.stringify(view.detail)) return;
      const changed = view.detail?.id !== id;
      view.detail = detail;
      if (changed) {
        view.frameId = null;
        $("#evaluation-errors-only").checked = false;
        $("#evaluation-audit-confirm").checked = false;
        $("#evaluation-reference-notes").value = "";
        error("#evaluation-audit-error", null);
        error("#evaluation-reference-error", null);
      }
      renderDetail(changed);
    } catch (failure) {
      if (request === view.detailRequest)
        error("#evaluation-history-error", failure);
    }
  }
  function table(caption, headings, rows) {
    const result = node("table", "evaluation-table");
    result.append(node("caption", "sr-only", caption));
    const head = node("thead");
    const header = node("tr");
    for (const text of headings) {
      const cell = node("th", "", text);
      cell.scope = "col";
      header.append(cell);
    }
    head.append(header);
    const body = node("tbody");
    for (const values of rows) {
      const row = node("tr");
      values.forEach((value, index) => {
        const cell = node(index ? "td" : "th", "", String(value));
        if (!index) cell.scope = "row";
        row.append(cell);
      });
      body.append(row);
    }
    result.append(head, body);
    return result;
  }
  function timing(modelId, key) {
    const values = (view.detail.predictions || [])
      .filter((item) => item.model_id === modelId)
      .map((item) => item.timing?.[key])
      .filter(Number.isFinite);
    return values.length
      ? `${(values.reduce((a, b) => a + b, 0) / values.length).toFixed(1)} ms`
      : "N/A";
  }
  function renderMetrics() {
    const summary = $("#evaluation-metrics"),
      classes = $("#evaluation-class-metrics");
    summary.replaceChildren();
    classes.replaceChildren();
    $("#evaluation-delta").textContent = "";
    if (!complete()) return;
    const models = view.detail.model_ids.map((id) =>
      view.detail.models.find((model) => model.model_id === id),
    );
    summary.append(
      table(
        "Detection quality by model",
        [
          "Model",
          "mAP .50–.95",
          "AP50",
          "AP75",
          "Precision",
          "Recall",
          "TP / FP / FN",
          `Mean ${(config().device || "cpu").toUpperCase()} forward`,
          "Mean total",
        ],
        models.map((model) => {
          const s = model.metrics.summary;
          return [
            modelName(model.model_id),
            percent(s.map),
            percent(s.map50),
            percent(s.map75),
            percent(s.precision),
            percent(s.recall),
            `${count(s.tp)} / ${count(s.fp)} / ${count(s.fn)}`,
            timing(model.model_id, "inference_ms"),
            timing(model.model_id, "total_ms"),
          ];
        }),
      ),
    );
    const rows = [];
    for (const model of models)
      for (const item of model.metrics.per_class || [])
        rows.push([
          modelName(model.model_id),
          item.label,
          count(item.support),
          percent(item.ap),
          percent(item.ap50),
          percent(item.ap75),
          percent(item.precision),
          percent(item.recall),
          `${count(item.tp)} / ${count(item.fp)} / ${count(item.fn)}`,
        ]);
    classes.append(
      table(
        "Quality and labeled support per class",
        [
          "Model",
          "Class",
          "Labeled boxes",
          "AP .50–.95",
          "AP50",
          "AP75",
          "Precision",
          "Recall",
          "TP / FP / FN",
        ],
        rows,
      ),
    );
    if (models.length === 2) {
      const first = models[0].metrics.summary,
        second = models[1].metrics.summary;
      const deltas = [
        ["mAP", "map"],
        ["precision", "precision"],
        ["recall", "recall"],
      ].map(([label, key]) => {
        const difference = second[key] - first[key];
        return `${label}: ${Number.isFinite(first[key]) && Number.isFinite(second[key]) ? `${difference >= 0 ? "+" : ""}${(difference * 100).toFixed(1)} percentage points` : "N/A"}`;
      });
      $("#evaluation-delta").textContent =
        `${modelName(models[1].model_id)} minus ${modelName(models[0].model_id)} on these same frozen frames — ${deltas.join(" · ")}.`;
    }
  }
  function errorsFor(modelId, id) {
    return view.detail.models
      .find((model) => model.model_id === modelId)
      ?.metrics?.frames?.find((frame) => frame.frame_id === id);
  }
  function availableFrames() {
    return frames().filter(
      (frame) =>
        !$("#evaluation-errors-only").checked ||
        view.detail.model_ids.some((id) => {
          const errors = errorsFor(id, frameId(frame));
          return errors && (errors.fp > 0 || errors.fn > 0);
        }),
    );
  }
  function populateFrames() {
    const options = availableFrames();
    if (!options.some((frame) => frameId(frame) === view.frameId))
      view.frameId = options.length ? frameId(options[0]) : null;
    const select = $("#evaluation-frame");
    select.replaceChildren();
    options.forEach((frame, index) =>
      select.append(
        new Option(
          `${index + 1}. ${frame.scene_group || "Scene"} · ${(frame.boxes || []).length} human labels · ${frameId(frame).slice(0, 8)}`,
          frameId(frame),
        ),
      ),
    );
    if (!options.length)
      select.append(new Option("No frames match this filter", ""));
    select.disabled = !options.length;
    select.value = view.frameId || "";
    renderFrame();
  }
  function svgNode(tag, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes))
      element.setAttribute(key, String(value));
    return element;
  }
  function overlayBox(svg, box, caption, color, dashed, width, height) {
    const [x1, y1, x2, y2] = box;
    const title = svgNode("title");
    title.textContent = caption;
    const rect = svgNode("rect", {
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1,
      fill: "none",
      stroke: color,
      "stroke-width": 2,
      "stroke-dasharray": dashed ? "6 4" : "none",
      "vector-effect": "non-scaling-stroke",
    });
    rect.append(title);
    svg.append(rect);
    const font = Math.max(9, Math.min(width, height) * 0.027);
    const text = svgNode("text", {
      x: Math.max(
        2,
        Math.min(x1 + 2, width - caption.length * font * 0.56 - 2),
      ),
      y: Math.max(font + 2, Math.min(height - 2, dashed ? y2 - 3 : y1 - 4)),
      fill: color,
      stroke: "#152422",
      "stroke-width": font / 5,
      "paint-order": "stroke",
      "font-size": font,
      "font-weight": 600,
    });
    text.textContent = caption;
    svg.append(text);
  }
  function renderFrame() {
    const container = $("#evaluation-canvases");
    container.replaceChildren();
    const options = availableFrames();
    const position = options.findIndex(
      (frame) => frameId(frame) === view.frameId,
    );
    $("#evaluation-previous").disabled = position <= 0;
    $("#evaluation-next").disabled =
      position < 0 || position >= options.length - 1;
    const frame = options[position];
    $("#evaluation-frame-context").textContent = frame
      ? `${frame.scene_group || "Scene"} · ${frame.width} × ${frame.height} · ${(frame.boxes || []).length ? `${frame.boxes.length} human-validated boxes` : "Validated negative frame: no labeled objects"}`
      : "No frozen frame matches this filter.";
    container.classList.toggle(
      "single-model",
      view.detail.model_ids.length === 1,
    );
    if (!frame) return;
    for (const modelId of view.detail.model_ids) {
      const prediction = (view.detail.predictions || []).find(
        (item) => item.frame_id === view.frameId && item.model_id === modelId,
      );
      const errors = errorsFor(modelId, view.frameId);
      const card = node("article", "prediction-card");
      const heading = node("div", "prediction-heading");
      heading.append(node("h3", "", modelName(modelId)));
      const shown = (prediction?.detections || [])
        .map((detection, index) => ({ ...detection, index }))
        .filter(
          (detection) =>
            ["person", "car"].includes(detection.label) &&
            detection.score >= config().confidence_threshold,
        );
      heading.append(
        node(
          "span",
          "prediction-count",
          prediction ? `${shown.length} detections` : "Not processed",
        ),
      );
      const visual = node("div", "prediction-visual");
      const svg = svgNode("svg", {
        viewBox: `0 0 ${frame.width} ${frame.height}`,
        role: "img",
        "aria-label": `${modelName(modelId)}, ${shown.length} detections, ${(frame.boxes || []).length} human labels`,
      });
      svg.append(
        svgNode("image", {
          href:
            frame.image_url ||
            `/api/datasets/${safe(view.detail.dataset_id)}/frames/${safe(view.frameId)}/image`,
          width: frame.width,
          height: frame.height,
          preserveAspectRatio: "none",
        }),
      );
      const missed = new Set(errors?.false_negatives || []);
      const falsePositives = new Set(errors?.false_positives || []);
      (frame.boxes || []).forEach((box, index) =>
        overlayBox(
          svg,
          box.box,
          `${missed.has(index) ? "Missed" : "Label"}: ${box.label}`,
          missed.has(index) ? "#ff8888" : "#80c5ff",
          true,
          frame.width,
          frame.height,
        ),
      );
      for (const detection of shown)
        overlayBox(
          svg,
          detection.box,
          `${errors ? (falsePositives.has(detection.index) ? "FP " : "TP ") : ""}${detection.label} ${Math.round(detection.score * 100)}%`,
          errors
            ? falsePositives.has(detection.index)
              ? "#ffd078"
              : "#96e8a4"
            : "#ffd078",
          false,
          frame.width,
          frame.height,
        );
      visual.append(svg);
      const context = node("div", "evaluation-frame-errors");
      context.append(
        node(
          "p",
          "field-hint",
          errors
            ? `${errors.tp} matched · ${errors.fp} false positives · ${errors.fn} missed labels`
            : prediction
              ? "Raw predictions available; complete frame error metrics are not available yet."
              : "This model has not processed this frame.",
        ),
      );
      if (errors && (errors.fp || errors.fn)) {
        const descriptions = [];
        for (const index of errors.false_positives || []) {
          const detection = prediction?.detections[index];
          if (detection)
            descriptions.push(
              `FP: ${detection.label} (${percent(detection.score)} confidence)`,
            );
        }
        for (const index of errors.false_negatives || []) {
          const box = frame.boxes?.[index];
          if (box)
            descriptions.push(`Missed: ${box.label}, label ${index + 1}`);
        }
        context.append(node("p", "field-hint", descriptions.join(" · ")));
      }
      card.append(heading, visual, context);
      container.append(card);
    }
  }
  function renderActions(changed) {
    const finished = complete();
    $("#evaluation-reference-form").hidden = !finished || split() !== "val";
    const select = $("#evaluation-reference-model");
    const previous = changed ? "" : select.value;
    select.replaceChildren(new Option("Choose a model", ""));
    for (const id of view.detail.model_ids)
      select.append(new Option(modelName(id), id));
    select.value = previous;
    $("#evaluation-reference-save").disabled =
      view.referenceBusy || !view.references;
    const dataset = view.datasets.find(
      (item) => item.id === view.detail.dataset_id,
    );
    $("#evaluation-test-audit").hidden =
      !finished ||
      split() !== "val" ||
      !(dataset?.summary?.split_counts?.test > 0);
    $("#evaluation-audit-start").disabled =
      view.auditBusy || !$("#evaluation-audit-confirm").checked;
  }
  function renderDetail(changed) {
    const detail = view.detail,
      job = detail.job;
    $("#evaluation-detail").hidden = false;
    $("#evaluation-detail-name").textContent = detail.name;
    $("#evaluation-detail-context").textContent =
      `${config().dataset_name || detail.dataset_id} · ${split() === "test" ? "Final test audit" : "Validation"} · ${frames().length} frozen frames · ${(config().device || "cpu").toUpperCase()} · ${new Date(detail.created_at).toLocaleString()}`;
    $("#evaluation-detail-status").textContent =
      job?.status || "Unknown status";
    $("#evaluation-detail-status").className =
      `job-status ${job?.status || ""}`;
    $("#evaluation-detail-message").textContent = job?.message || "";
    error("#evaluation-detail-error", job?.error ? new Error(job.error) : null);
    $("#evaluation-completeness").textContent = complete()
      ? split() === "test"
        ? "Completed test audit. Report this result; choose reference models from validation evidence."
        : "Completed validation evaluation. Metrics cover every frozen frame for each model."
      : "Incomplete evaluation. Saved predictions remain inspectable; no complete quality comparison is available.";
    const warnings = $("#evaluation-warnings");
    warnings.replaceChildren();
    for (const warning of new Set([
      ...(config().warnings || []),
      ...(detail.models || []).flatMap(
        (model) => model.metrics?.warnings || [],
      ),
    ]))
      warnings.append(node("p", "field-hint", String(warning)));
    renderMetrics();
    $("#evaluation-inspection-thresholds").textContent =
      `Saved confidence ≥ ${config().confidence_threshold} · matching IoU ≥ ${config().iou_threshold} · person / car only`;
    $("#evaluation-errors-only").disabled = !(detail.models || []).some(
      (model) => model.metrics?.frames?.length,
    );
    populateFrames();
    $("#evaluation-provenance").textContent = JSON.stringify(
      {
        dataset_id: detail.dataset_id,
        split: detail.split,
        config: detail.config,
        models: detail.models.map((model) => ({
          model_id: model.model_id,
          metadata: model.metadata,
          protocol: model.metrics?.protocol,
        })),
        prediction_count: detail.predictions.length,
        frame_timings: detail.predictions.map((prediction) => ({
          model_id: prediction.model_id,
          frame_id: prediction.frame_id,
          timing: prediction.timing,
        })),
      },
      null,
      2,
    );
    renderActions(changed);
  }
  function referenceContent(reference, current) {
    const entry = node("article", "evaluation-reference-entry");
    entry.append(
      node(
        current ? "h3" : "h4",
        "",
        reference.metadata?.model_name || reference.model_id,
      ),
    );
    entry.append(
      node(
        "p",
        "field-hint",
        `${reference.reviewer} · ${new Date(reference.created_at).toLocaleString()} · ${reference.metadata?.dataset_name || "Validation evaluation"}`,
      ),
    );
    entry.append(node("p", "small", reference.notes));
    const button = node("button", "text-button", "Open validation evidence →");
    button.type = "button";
    button.addEventListener("click", async () => {
      view.activeId = reference.evaluation_id;
      view.detail = null;
      $("#evaluation-detail").hidden = true;
      await refreshHistory();
      $("#evaluation-history-title").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
    entry.append(button);
    return entry;
  }
  async function refreshReferences() {
    try {
      const references = await api("/api/model-references");
      view.references = references;
      const current = $("#evaluation-reference-current");
      current.replaceChildren();
      current.append(
        references.current
          ? referenceContent(references.current, true)
          : node(
              "p",
              "field-hint",
              "No reference has been chosen. Select a model manually from a completed validation run.",
            ),
      );
      const history = $("#evaluation-reference-history");
      history.replaceChildren();
      for (const reference of references.history || [])
        history.append(referenceContent(reference, false));
      if (!(references.history || []).length)
        history.append(node("p", "field-hint", "No recorded decisions yet."));
      error("#evaluation-reference-history-error", null);
    } catch (failure) {
      view.references = null;
      error("#evaluation-reference-history-error", failure);
    }
    if (view.detail) renderActions(false);
  }
  async function refresh() {
    $("#evaluation-refresh").disabled = true;
    await refreshCatalogs();
    await Promise.all([refreshHistory(), refreshReferences()]);
    $("#evaluation-refresh").disabled = false;
  }
  async function submit(payload) {
    const detail = await api("/api/evaluations", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    view.activeId = detail.id;
    await refreshHistory();
    await refreshJobs();
    notify(
      `Evaluation “${detail.name}” queued. Follow progress or cancel in Processing jobs.`,
    );
  }
  $("#evaluation-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if ($("#evaluation-start").disabled) return;
    const payload = {
      name: $("#evaluation-name").value.trim(),
      dataset_id: $("#evaluation-dataset").value,
      split: "val",
      model_ids: [...view.chosen],
      confidence_threshold: Number($("#evaluation-confidence").value),
      iou_threshold: Number($("#evaluation-iou").value),
      device: "cpu",
      validation_evaluation_id: null,
    };
    if (!payload.name) return $("#evaluation-name").focus();
    view.busy = true;
    updateLaunch();
    error("#evaluation-error", null);
    try {
      await submit(payload);
      $("#evaluation-name").value = "";
    } catch (failure) {
      error("#evaluation-error", failure);
    } finally {
      view.busy = false;
      updateLaunch();
    }
  });
  $("#evaluation-audit-start").addEventListener("click", async () => {
    if (
      $("#evaluation-audit-start").disabled ||
      !complete() ||
      split() !== "val"
    )
      return;
    const detail = view.detail;
    view.auditBusy = true;
    renderActions(false);
    error("#evaluation-audit-error", null);
    try {
      await submit({
        name: `${detail.name.slice(0, 147)} · test audit`,
        dataset_id: detail.dataset_id,
        split: "test",
        model_ids: detail.model_ids,
        confidence_threshold: detail.config.confidence_threshold,
        iou_threshold: detail.config.iou_threshold,
        device: detail.config.device || "cpu",
        validation_evaluation_id: detail.id,
      });
    } catch (failure) {
      error("#evaluation-audit-error", failure);
    } finally {
      view.auditBusy = false;
      if (view.detail) renderActions(false);
    }
  });
  $("#evaluation-reference-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (
      view.referenceBusy ||
      !view.references ||
      !complete() ||
      split() !== "val"
    )
      return;
    const payload = {
      evaluation_id: view.detail.id,
      model_id: $("#evaluation-reference-model").value,
      reviewer: $("#evaluation-reference-reviewer").value.trim(),
      notes: $("#evaluation-reference-notes").value.trim(),
      expected_previous_id: view.references.current?.id || null,
    };
    if (!payload.notes) return $("#evaluation-reference-notes").focus();
    if (!payload.reviewer) return $("#evaluation-reference-reviewer").focus();
    if (!payload.model_id) return $("#evaluation-reference-model").focus();
    view.referenceBusy = true;
    renderActions(false);
    error("#evaluation-reference-error", null);
    try {
      await api("/api/model-references", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      await refreshReferences();
      $("#evaluation-reference-notes").value = "";
      notify("Reference model decision saved with its validation evidence.");
    } catch (failure) {
      error("#evaluation-reference-error", failure);
      if (failure.status === 409) await refreshReferences();
    } finally {
      view.referenceBusy = false;
      if (view.detail) renderActions(false);
    }
  });
  $("#evaluation-refresh").addEventListener("click", refresh);
  $("#evaluation-dataset").addEventListener("change", updateLaunch);
  $("#evaluation-history").addEventListener("change", (event) => {
    view.activeId = event.target.value;
    view.detail = null;
    $("#evaluation-detail").hidden = true;
    loadDetail(view.activeId);
  });
  $("#evaluation-frame").addEventListener("change", (event) => {
    view.frameId = event.target.value;
    renderFrame();
  });
  $("#evaluation-errors-only").addEventListener("change", populateFrames);
  $("#evaluation-audit-confirm").addEventListener("change", () =>
    renderActions(false),
  );
  for (const [selector, step] of [
    ["#evaluation-previous", -1],
    ["#evaluation-next", 1],
  ])
    $(selector).addEventListener("click", () => {
      const options = availableFrames();
      const position = options.findIndex(
        (frame) => frameId(frame) === view.frameId,
      );
      const next = options[position + step];
      if (next) {
        view.frameId = frameId(next);
        $("#evaluation-frame").value = view.frameId;
        renderFrame();
      }
    });
  window.addEventListener("iris:workspace", (event) => {
    view.visible = event.detail.name === "evaluation";
    if (view.visible) refresh();
  });
  window.addEventListener("iris:jobs", () => {
    let changed = false;
    for (const job of state.jobs) {
      if (job.kind !== "evaluate") continue;
      if (view.jobStatuses.get(job.id) !== job.status || isActive(job))
        changed = true;
      view.jobStatuses.set(job.id, job.status);
    }
    if (changed && view.visible) refreshHistory();
  });
})();
