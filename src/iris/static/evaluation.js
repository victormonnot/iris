"use strict";

(() => {
  const datasetTools = window.IRISDatasetTools;
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
    analysisRequest: 0,
    analysisController: null,
    analysis: null,
    previewKey: null,
    previewRequest: 0,
    previewPending: false,
    preview: null,
    previewError: null,
    auditPreviewKey: null,
    auditPreviewRequest: 0,
    auditPreviewPending: false,
    auditPreview: null,
    auditPreviewError: null,
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
  const modeName = (variant) => variant === "tiled" ? "Tiled" : "Full image";
  function lanes() {
    const detail = view.detail;
    if (!detail) return [];
    return detail.lanes || detail.model_ids.map((id) => ({
      model_id: id,
      variant: "full",
      evaluation_model_id: detail.models?.find((model) => model.model_id === id)?.id || null,
    }));
  }
  function laneKey(lane) {
    return lane.evaluation_model_id || (view.detail.lanes
      ? JSON.stringify([lane.model_id, lane.variant]) : lane.model_id);
  }
  function laneName(lane) {
    return `${modelName(lane.model_id)} · ${modeName(lane.variant)}`;
  }
  function modelFor(lane) {
    return (view.detail.models || []).find((model) => lane.evaluation_model_id
      ? model.id === lane.evaluation_model_id
      : model.model_id === lane.model_id && (model.variant || "full") === lane.variant);
  }
  function belongsToLane(prediction, lane) {
    const id = lane.evaluation_model_id || modelFor(lane)?.id;
    if (id) return prediction.evaluation_model_id === id;
    return !view.detail.lanes && prediction.model_id === lane.model_id;
  }
  function inferenceFields(inference = { mode: "full" }) {
    return {
      inference_mode: inference.mode,
      tile_size: inference.tiling?.tile_size ?? 640,
      overlap: inference.tiling?.overlap ?? 0.2,
    };
  }
  function pipelineDescription(inference = { mode: "full" }, variant = inference.mode) {
    const label = variant === "paired" ? "Full image vs tiled" : modeName(variant);
    if (variant === "full" || !variant) return label;
    const settings = inference.tiling || inference;
    return `${label} · ${settings.tile_size} px tiles · ${Math.round(settings.overlap * 100)}% overlap`;
  }
  function auditPayload(detail) {
    return {
      name: `${detail.name.slice(0, 147)} · test audit`,
      dataset_id: detail.dataset_id,
      split: "test",
      model_ids: detail.model_ids,
      confidence_threshold: detail.config.confidence_threshold,
      iou_threshold: detail.config.iou_threshold,
      device: detail.config.device || "cpu",
      validation_evaluation_id: detail.id,
      ...inferenceFields(detail.config.inference),
    };
  }
  async function requestAuditPreview(payload, key) {
    const request = ++view.auditPreviewRequest;
    view.auditPreviewKey = key;
    view.auditPreviewPending = true;
    view.auditPreview = null;
    view.auditPreviewError = null;
    try {
      const result = await api("/api/evaluations/preview", {
        method: "POST", body: JSON.stringify(payload),
      });
      if (request !== view.auditPreviewRequest || view.detail?.id !== payload.validation_evaluation_id) return;
      view.auditPreview = result;
    } catch (failure) {
      if (request !== view.auditPreviewRequest || view.detail?.id !== payload.validation_evaluation_id) return;
      view.auditPreviewError = failure;
    } finally {
      if (request === view.auditPreviewRequest && view.detail?.id === payload.validation_evaluation_id) {
        view.auditPreviewPending = false;
        renderActions(false);
      }
    }
  }
  function evaluationPayload(preview = false) {
    const mode = $("#evaluation-inference-mode").value;
    return {
      name: preview ? "Evaluation preview" : $("#evaluation-name").value.trim(),
      dataset_id: $("#evaluation-dataset").value,
      split: "val",
      model_ids: [...view.chosen],
      confidence_threshold: Number($("#evaluation-confidence").value),
      iou_threshold: Number($("#evaluation-iou").value),
      device: "cpu",
      validation_evaluation_id: null,
      inference_mode: mode,
      tile_size: mode === "full" ? 640 : Number($("#evaluation-tile-size").value),
      overlap: mode === "full" ? 0.2 : Number($("#evaluation-tile-overlap").value),
    };
  }
  async function requestPreview(payload, key) {
    const request = ++view.previewRequest;
    view.previewKey = key;
    view.previewPending = true;
    view.preview = null;
    view.previewError = null;
    try {
      const preview = await api("/api/evaluations/preview", {
        method: "POST", body: JSON.stringify(payload),
      });
      if (request !== view.previewRequest) return;
      view.preview = preview;
    } catch (failure) {
      if (request !== view.previewRequest) return;
      view.previewError = failure;
    } finally {
      if (request === view.previewRequest) {
        view.previewPending = false;
        updateLaunch();
      }
    }
  }
  function error(selector, value) {
    const target = $(selector);
    target.textContent = value?.message || "";
    target.hidden = !value;
  }
  function complete() {
    const detail = view.detail;
    return (
      detail?.job?.status === "succeeded" &&
      detail.models?.length === lanes().length &&
      lanes().every(
        (lane) => modelFor(lane)?.metrics?.summary?.frame_count === frames().length,
      )
    );
  }
  function updateLaunch() {
    const dataset = view.datasets.find(
      (item) => item.id === $("#evaluation-dataset").value,
    );
    const size = dataset?.summary?.split_counts?.val || 0;
    const mode = $("#evaluation-inference-mode").value;
    const tiled = mode !== "full";
    const payload = evaluationPayload(true);
    const validTiles = !tiled || (
      Number.isInteger(payload.tile_size) && payload.tile_size >= 128 && payload.tile_size <= 2048 &&
      $("#evaluation-tile-overlap").value !== "" && Number.isFinite(payload.overlap) && payload.overlap >= 0 && payload.overlap <= 0.5
    );
    const validThresholds = $("#evaluation-confidence").value !== "" &&
      $("#evaluation-iou").value !== "" &&
      Number.isFinite(payload.confidence_threshold) && payload.confidence_threshold >= 0 && payload.confidence_threshold <= 1 &&
      Number.isFinite(payload.iou_threshold) && payload.iou_threshold >= 0.01 && payload.iou_threshold <= 1;
    const valid = datasetTools.mlSupported(dataset) && size > 0 && view.chosen.size > 0 && view.chosen.size <= (mode === "paired" ? 1 : 2) && validTiles && validThresholds;
    $("#evaluation-tiling-fields").hidden = !tiled;
    $("#evaluation-tile-size").disabled = !tiled;
    $("#evaluation-tile-overlap").disabled = !tiled;
    $("#evaluation-mode-hint").textContent = mode === "paired"
      ? "Choose one model. Evaluate the same checkpoint on full images and overlapping crops against the same frozen human labels."
      : tiled
        ? "Evaluate merged crop detections in the original image. Tile size and overlap are saved with the quality metrics."
        : "Evaluate each selected model on the whole image.";
    $("#evaluation-dataset-context").textContent = dataset
      ? `${size} validation frames · ${dataset.summary.split_counts.test || 0} reserved test frames · frozen labels and scene groups`
      : "Freeze a release with validation data in Dataset & training first.";
    $("#evaluation-selection").textContent =
      `${size} validation frames · ${view.chosen.size} models · ${mode === "paired" ? "Full image vs tiled" : modeName(mode)} · CPU`;
    if (valid) {
      const key = JSON.stringify(payload);
      if (key !== view.previewKey) requestPreview(payload, key);
    } else if (view.previewKey !== null) {
      ++view.previewRequest;
      view.previewKey = null;
      view.previewPending = false;
      view.preview = null;
      view.previewError = null;
    }
    const plan = $("#evaluation-plan");
    plan.classList.toggle("inline-error", Boolean(view.previewError));
    if (view.previewPending) plan.textContent = "Checking the number of model passes…";
    else if (view.previewError) plan.textContent = view.previewError.message;
    else if (view.chosen.size > (mode === "paired" ? 1 : 2))
      plan.textContent = mode === "paired" ? "Choose one model for full image vs tiled." : "Choose at most two models.";
    else if (!validTiles) plan.textContent = "Choose a whole tile size from 128 to 2048 pixels and an overlap from 0 to 0.5.";
    else if (!validThresholds) plan.textContent = "Choose confidence from 0 to 1 and matching IoU from 0.01 to 1.";
    else if (view.preview) {
      const preview = view.preview;
      const counts = (preview.tiles || []).map((item) => item.tile_count);
      const tileHint = tiled && counts.length
        ? ` · ${Math.min(...counts) === Math.max(...counts) ? counts[0] : `${Math.min(...counts)}–${Math.max(...counts)}`} tiles per image` : "";
      plan.textContent = `${preview.forward_passes} model passes + ${preview.warmup_passes} warm-up passes${tileHint}. Limit: ${tiled ? `${preview.limits.max_tiles_per_frame} tiles per image, ` : ""}${preview.limits.max_forward_passes} passes per evaluation. All processing stays local.`;
    } else plan.textContent = "";
    $("#evaluation-start").disabled =
      view.busy || !valid || view.previewPending || !view.preview || Boolean(view.previewError);
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
    ++view.previewRequest;
    view.previewKey = null;
    view.previewPending = false;
    view.preview = null;
    view.previewError = null;
    $("#evaluation-start").disabled = true;
    try {
      const [datasets, models] = await Promise.all([
        api("/api/datasets"),
        api("/api/models"),
      ]);
      if (request !== view.catalogRequest) return;
      const previous = $("#evaluation-dataset").value;
      view.datasets = datasets;
      const supported = datasets.filter(datasetTools.mlSupported);
      view.models = models;
      const select = $("#evaluation-dataset");
      select.replaceChildren();
      for (const dataset of supported)
        select.append(new Option(dataset.name, dataset.id));
      if (!supported.length)
        select.append(new Option(datasets.length ? "No compatible evaluation release" : "Freeze a dataset first", ""));
      if (supported.some((item) => item.id === previous))
        select.value = previous;
      select.disabled = !supported.length;
      $("#evaluation-dataset-limitation").textContent = datasets.length > supported.length
        ? `${datasets.length - supported.length} custom-class release${datasets.length - supported.length === 1 ? " is" : "s are"} available for inspection and COCO export in Dataset & training. Evaluation currently requires the original person / car definitions.`
        : "Evaluation currently supports releases using the original person / car definitions.";
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
      $("#evaluation-dataset-limitation").textContent = "Dataset compatibility could not be checked. Refresh to try again.";
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
      if (!history.length) {
        ++view.detailRequest;
        view.detail = null;
        view.frameId = null;
        resetAnalysis("No evaluation selected.");
        $("#evaluation-detail").hidden = true;
      }
      error("#evaluation-history-error", null);
      if (view.activeId) await loadDetail(view.activeId);
    } catch (failure) {
      if (request === view.historyRequest)
        error("#evaluation-history-error", failure);
    }
  }
  async function loadDetail(id) {
    const request = ++view.detailRequest;
    resetAnalysis("Loading saved error analysis…");
    try {
      const detail = await api(`/api/evaluations/${safe(id)}`);
      if (request !== view.detailRequest || view.activeId !== id) return;
      error("#evaluation-history-error", null);
      if (JSON.stringify(detail) === JSON.stringify(view.detail)) {
        await loadAnalysis(id, request);
        return;
      }
      const changed = view.detail?.id !== id;
      view.detail = detail;
      if (changed) {
        view.frameId = null;
        $("#evaluation-errors-only").checked = false;
        $("#evaluation-analysis-class").value = "all";
        $("#evaluation-analysis-filter").value = "all";
        $("#evaluation-analysis-sort").value = "source";
        $("#evaluation-audit-confirm").checked = false;
        $("#evaluation-reference-notes").value = "";
        error("#evaluation-audit-error", null);
        error("#evaluation-reference-error", null);
      }
      renderDetail(changed);
      await loadAnalysis(id, request);
    } catch (failure) {
      if (request === view.detailRequest) {
        resetAnalysis(
          "Saved error analysis is unavailable. Refresh to try again.",
        );
        error("#evaluation-history-error", failure);
      }
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
  function timing(lane, key) {
    const values = (view.detail.predictions || [])
      .filter((item) => belongsToLane(item, lane))
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
    const runLanes = lanes();
    const models = runLanes.map(modelFor);
    summary.append(
      table(
        "Detection quality by model and inference mode",
        [
          "Model / mode",
          "mAP .50–.95",
          "AP50",
          "AP75",
          "Precision",
          "Recall",
          "TP / FP / FN",
          `Mean ${(config().device || "cpu").toUpperCase()} forward`,
          "Mean total",
        ],
        models.map((model, index) => {
          const s = model.metrics.summary;
          const lane = runLanes[index];
          return [
            laneName(lane),
            percent(s.map),
            percent(s.map50),
            percent(s.map75),
            percent(s.precision),
            percent(s.recall),
            `${count(s.tp)} / ${count(s.fp)} / ${count(s.fn)}`,
            timing(lane, "inference_ms"),
            timing(lane, "total_ms"),
          ];
        }),
      ),
    );
    const rows = [];
    for (const [index, model] of models.entries())
      for (const item of model.metrics.per_class || [])
        rows.push([
          laneName(runLanes[index]),
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
          "Model / mode",
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
        `${laneName(runLanes[1])} minus ${laneName(runLanes[0])} on these same frozen frames — ${deltas.join(" · ")}.`;
    }
  }
  function errorsFor(lane, id) {
    return modelFor(lane)
      ?.metrics?.frames?.find((frame) => frame.frame_id === id);
  }
  function normalizeAnalysis(result) {
    if (result.protocol !== "iris-error-analysis-v2") return result;
    const counts = (scopes) => Object.fromEntries(Object.entries(scopes).map(
      ([scope, stats]) => [scope, { ...stats, models: stats.runs }],
    ));
    return {
      ...result,
      models: result.runs,
      comparison: result.comparison ? {
        ...result.comparison,
        baseline_model_id: result.comparison.baseline_run_id,
        candidate_model_id: result.comparison.candidate_run_id,
      } : null,
      summary: counts(result.summary),
      frames: result.frames.map((frame) => ({ ...frame, counts: counts(frame.counts) })),
    };
  }
  function analysisClass() {
    return view.analysis ? $("#evaluation-analysis-class").value : "all";
  }
  function includesClass(label) {
    return analysisClass() === "all" || label === analysisClass();
  }
  function resetAnalysis(message) {
    ++view.analysisRequest;
    view.analysisController?.abort();
    view.analysisController = null;
    view.analysis = null;
    $("#evaluation-analysis-status").textContent = message;
    $("#evaluation-analysis-summary").replaceChildren();
    $("#evaluation-analysis-table").replaceChildren();
    $("#evaluation-analysis-context").textContent = "";
    $("#evaluation-analysis-count").textContent = "";
    $("#evaluation-analysis-warnings").replaceChildren();
    $("#evaluation-analysis-method").hidden = true;
    error("#evaluation-analysis-error", null);
    for (const selector of ["class", "filter", "sort"])
      $(`#evaluation-analysis-${selector}`).disabled = true;
  }
  async function loadAnalysis(id, detailRequest) {
    if (!complete()) {
      resetAnalysis(
        "Analysis unavailable for an incomplete evaluation. Saved predictions remain inspectable below.",
      );
      return;
    }
    const request = ++view.analysisRequest;
    const controller = new AbortController();
    view.analysisController = controller;
    const current = () =>
      request === view.analysisRequest &&
      detailRequest === view.detailRequest &&
      id === view.activeId;
    try {
      const result = await api(`/api/evaluations/${safe(id)}/analysis`, {
        signal: controller.signal,
      });
      if (!current()) return;
      view.analysis = normalizeAnalysis(result);
      $("#evaluation-analysis-status").textContent = "";
      for (const selector of ["class", "filter", "sort"])
        $(`#evaluation-analysis-${selector}`).disabled = false;
      const paired = Boolean(result.comparison);
      for (const option of document.querySelectorAll(
        "#evaluation-analysis [data-paired]",
      ))
        option.disabled = !paired;
      if (!paired) {
        if (
          ["new_misses", "recovered", "more_fp"].includes(
            $("#evaluation-analysis-filter").value,
          )
        )
          $("#evaluation-analysis-filter").value = "all";
        if ($("#evaluation-analysis-sort").value === "new_misses")
          $("#evaluation-analysis-sort").value = "source";
      }
      if ($("#evaluation-errors-only").checked)
        $("#evaluation-analysis-filter").value = "errors";
      renderAnalysis();
    } catch (failure) {
      if (!current()) return;
      $("#evaluation-analysis-status").textContent =
        "Saved error analysis is unavailable. No error totals are assumed. Refresh to try again.";
      error("#evaluation-analysis-error", failure);
      populateFrames();
    } finally {
      if (current()) view.analysisController = null;
    }
  }
  function analysisFrames() {
    if (!view.analysis) return [];
    const scope = analysisClass();
    const filter = $("#evaluation-analysis-filter").value;
    const order = $("#evaluation-analysis-sort").value;
    const maximum = (stats, key) =>
      Math.max(...Object.values(stats.models).map((item) => item[key]));
    return view.analysis.frames
      .filter((frame) => {
        const stats = frame.counts[scope];
        if (filter === "errors")
          return maximum(stats, "fn") + maximum(stats, "fp") > 0;
        if (filter === "misses") return maximum(stats, "fn") > 0;
        if (filter === "fp") return maximum(stats, "fp") > 0;
        if (filter === "new_misses") return stats.changes?.new_misses > 0;
        if (filter === "recovered") return stats.changes?.recovered > 0;
        if (filter === "more_fp") return stats.changes?.fp_delta > 0;
        return true;
      })
      .sort((left, right) => {
        const value = (frame) => {
          const stats = frame.counts[scope];
          if (order === "misses") return maximum(stats, "fn");
          if (order === "fp") return maximum(stats, "fp");
          if (order === "new_misses") return stats.changes?.new_misses || 0;
          return 0;
        };
        return value(right) - value(left) || left.position - right.position;
      });
  }
  function availableFrames() {
    if (view.analysis) {
      const byId = new Map(frames().map((frame) => [frameId(frame), frame]));
      return analysisFrames()
        .map((row) => byId.get(row.frame_id))
        .filter(Boolean);
    }
    return frames().filter(
      (frame) =>
        !$("#evaluation-errors-only").checked ||
        lanes().some((lane) => {
          const errors = errorsFor(lane, frameId(frame));
          return errors && (errors.fp > 0 || errors.fn > 0);
        }),
    );
  }
  function renderAnalysis() {
    const analysis = view.analysis;
    if (!analysis) return;
    const scope = analysisClass();
    const stats = analysis.summary[scope];
    const signed = (value) => `${value > 0 ? "+" : ""}${value}`;
    const names = new Map(
      analysis.models.map((model) => [model.id, model.name]),
    );
    const baseline = analysis.comparison?.baseline_model_id;
    const candidate = analysis.comparison?.candidate_model_id;
    const label = (model) =>
      `${baseline === model.id ? "Baseline: " : candidate === model.id ? "Candidate: " : ""}${model.name}`;
    $("#evaluation-analysis-context").textContent =
      `Saved confidence ≥ ${analysis.confidence_threshold} · matching IoU ≥ ${analysis.iou_threshold}. ` +
      (analysis.comparison
        ? `Baseline: ${names.get(baseline)} → candidate: ${names.get(candidate)} (saved run order). `
        : "One model; paired changes are unavailable. ") +
      (analysis.split === "test"
        ? "Test audit: reporting only. Use validation for model selection."
        : "Uses saved matches and predictions; no new inference or AP calculation.");
    const summary = $("#evaluation-analysis-summary");
    summary.replaceChildren(
      node(
        "strong",
        "",
        `Whole ${analysis.split === "test" ? "test" : "validation"} split · ${scope === "all" ? "all classes" : scope} · ${stats.frame_count} frames · ${stats.ground_truth_count} labeled objects`,
      ),
    );
    for (const model of analysis.models) {
      const totals = stats.models[model.id];
      summary.append(
        node(
          "p",
          "field-hint",
          `${label(model)}: ${totals.tp} matched · ${totals.fp} false positives · ${totals.fn} missed · ${totals.error_frames} frames with errors`,
        ),
      );
    }
    if (stats.changes)
      summary.append(
        node(
          "p",
          "field-hint",
          `Candidate changes: ${stats.changes.new_misses} new misses · ${stats.changes.recovered} recovered labels · false-positive count change: ${signed(stats.changes.fp_delta)}`,
        ),
      );
    $("#evaluation-analysis-method").hidden = false;
    const warnings = $("#evaluation-analysis-warnings");
    warnings.replaceChildren();
    for (const warning of analysis.warnings || [])
      warnings.append(node("p", "field-hint", warning));
    const rows = analysisFrames();
    $("#evaluation-analysis-count").textContent =
      `${rows.length} of ${analysis.frames.length} frames shown. Filters affect this table and image navigation; whole-split totals stay unchanged.`;
    const headings = [
      "Frozen image / scene",
      ...analysis.models.map(
        (model) =>
          `${baseline === model.id ? "Baseline" : candidate === model.id ? "Candidate" : "Model"} · TP / FP / FN`,
      ),
    ];
    if (analysis.comparison)
      headings.push("New misses", "Recovered", "FP change");
    const result = table("Saved errors by frozen image", headings, []);
    analysis.models.forEach((model, index) => {
      result.querySelectorAll("thead th")[index + 1].title =
        `${label(model)}: matched detections / false positives / missed labeled objects`;
    });
    const body = result.querySelector("tbody");
    for (const frame of rows) {
      const row = node("tr");
      row.dataset.frameId = frame.frame_id;
      const context = node("th");
      context.scope = "row";
      const open = node(
        "button",
        "text-button",
        frame.source_filename || frame.frame_id,
      );
      open.type = "button";
      open.addEventListener("click", () => {
        view.frameId = frame.frame_id;
        $("#evaluation-frame").value = view.frameId;
        renderFrame();
        $("#evaluation-inspection-title").scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
        $("#evaluation-inspection-title").focus({ preventScroll: true });
      });
      context.append(
        open,
        node("span", "field-hint", frame.scene_group || "Scene"),
      );
      row.append(context);
      const counts = frame.counts[scope];
      for (const model of analysis.models) {
        const item = counts.models[model.id];
        row.append(node("td", "", `${item.tp} / ${item.fp} / ${item.fn}`));
      }
      if (counts.changes)
        row.append(
          node("td", "", String(counts.changes.new_misses)),
          node("td", "", String(counts.changes.recovered)),
          node("td", "", signed(counts.changes.fp_delta)),
        );
      body.append(row);
    }
    const container = $("#evaluation-analysis-table");
    container.replaceChildren(
      rows.length
        ? result
        : node("p", "field-hint", "No frames match these filters."),
    );
    populateFrames();
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
    for (const row of document.querySelectorAll(
      "#evaluation-analysis-table tbody tr",
    )) {
      const active = row.dataset.frameId === view.frameId;
      row.classList.toggle("active", active);
      if (active)
        row.querySelector("button").setAttribute("aria-current", "true");
      else row.querySelector("button").removeAttribute("aria-current");
    }
    const labels = (frame?.boxes || []).filter((box) =>
      includesClass(box.label),
    );
    $("#evaluation-frame-context").textContent = frame
      ? `${frame.scene_group || "Scene"} · ${frame.width} × ${frame.height} · ${(frame.boxes || []).length ? `${frame.boxes.length} human-validated boxes` : "Validated negative frame: no labeled objects"}`
      : "No frozen frame matches this filter.";
    if (frame && analysisClass() !== "all")
      $("#evaluation-frame-context").textContent =
        `${frame.scene_group || "Scene"} · ${frame.width} × ${frame.height} · ${labels.length} ${analysisClass()} human-validated boxes (class filter)`;
    container.classList.toggle(
      "single-model",
      lanes().length === 1,
    );
    if (!frame) return;
    for (const lane of lanes()) {
      const prediction = (view.detail.predictions || []).find(
        (item) => item.frame_id === view.frameId && belongsToLane(item, lane),
      );
      const errors = errorsFor(lane, view.frameId);
      const card = node("article", "prediction-card");
      const heading = node("div", "prediction-heading");
      heading.append(node("h3", "", laneName(lane)));
      const shown = (prediction?.detections || [])
        .map((detection, index) => ({ ...detection, index }))
        .filter(
          (detection) =>
            ["person", "car"].includes(detection.label) &&
            includesClass(detection.label) &&
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
        "aria-label": `${laneName(lane)}, ${shown.length} detections, ${labels.length} human labels`,
      });
      svg.append(
        svgNode("image", {
          href: projectURL(
            frame.image_url ||
            `/api/datasets/${safe(view.detail.dataset_id)}/frames/${safe(view.frameId)}/image`),
          width: frame.width,
          height: frame.height,
          preserveAspectRatio: "none",
        }),
      );
      const missed = new Set(errors?.false_negatives || []);
      const falsePositives = new Set(errors?.false_positives || []);
      (frame.boxes || []).forEach((box, index) => {
        if (!includesClass(box.label)) return;
        overlayBox(
          svg,
          box.box,
          `${missed.has(index) ? "Missed" : "Label"}: ${box.label}`,
          missed.has(index) ? "#ff8888" : "#80c5ff",
          true,
          frame.width,
          frame.height,
        );
      });
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
      const scoped =
        view.analysis?.frames.find((item) => item.frame_id === view.frameId)
          ?.counts[analysisClass()].models[view.analysis?.protocol === "iris-error-analysis-v2" ? laneKey(lane) : lane.model_id] || errors;
      context.append(
        node(
          "p",
          "field-hint",
          errors
            ? `${scoped.tp} matched · ${scoped.fp} false positives · ${scoped.fn} missed labels${analysisClass() === "all" ? "" : ` · ${analysisClass()} only`}`
            : prediction
              ? "Raw predictions available; complete frame error metrics are not available yet."
              : "This model and inference mode have not processed this frame.",
        ),
      );
      if (errors && (scoped.fp || scoped.fn)) {
        const descriptions = [];
        for (const index of errors.false_positives || []) {
          const detection = prediction?.detections[index];
          if (detection && includesClass(detection.label))
            descriptions.push(
              `FP: ${detection.label} (${percent(detection.score)} confidence)`,
            );
        }
        for (const index of errors.false_negatives || []) {
          const box = frame.boxes?.[index];
          if (box && includesClass(box.label))
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
    $("#evaluation-save-experiment").hidden = !finished;
    $("#evaluation-save-experiment").disabled = !finished;
    $("#evaluation-reference-form").hidden = !finished || split() !== "val";
    const select = $("#evaluation-reference-model");
    const previous = changed ? "" : select.value;
    select.replaceChildren(new Option("Choose a model and inference mode", ""));
    for (const lane of lanes())
      select.append(new Option(laneName(lane), laneKey(lane)));
    select.value = previous;
    $("#evaluation-reference-save").disabled =
      view.referenceBusy || !view.references;
    const dataset = view.datasets.find(
      (item) => item.id === view.detail.dataset_id,
    );
    const auditAvailable = finished && split() === "val" && dataset?.summary?.split_counts?.test > 0;
    $("#evaluation-test-audit").hidden = !auditAvailable;
    if (auditAvailable) {
      const payload = auditPayload(view.detail);
      const key = JSON.stringify(payload);
      if (key !== view.auditPreviewKey) requestAuditPreview(payload, key);
    } else if (view.auditPreviewKey !== null) {
      ++view.auditPreviewRequest;
      view.auditPreviewKey = null;
      view.auditPreview = null;
      view.auditPreviewPending = false;
      view.auditPreviewError = null;
    }
    const plan = $("#evaluation-audit-plan");
    plan.classList.toggle("inline-error", Boolean(view.auditPreviewError));
    plan.textContent = view.auditPreviewPending ? "Checking model passes on the reserved test split…"
      : view.auditPreviewError ? view.auditPreviewError.message
        : view.auditPreview ? `${view.auditPreview.frames_total} reserved test frames · ${view.auditPreview.forward_passes} model passes + ${view.auditPreview.warmup_passes} warm-up passes · ${pipelineDescription(config().inference)}. The saved settings will be copied unchanged.` : "";
    $("#evaluation-audit-start").disabled =
      view.auditBusy || !$("#evaluation-audit-confirm").checked || view.auditPreviewPending || !view.auditPreview || Boolean(view.auditPreviewError);
  }
  function renderDetail(changed) {
    const detail = view.detail,
      job = detail.job;
    $("#evaluation-detail").hidden = false;
    $("#evaluation-detail-name").textContent = detail.name;
    $("#evaluation-detail-context").textContent =
      `${config().dataset_name || detail.dataset_id} · ${split() === "test" ? "Final test audit" : "Validation"} · ${frames().length} frozen frames · ${pipelineDescription(config().inference)} · ${(config().device || "cpu").toUpperCase()} · ${new Date(detail.created_at).toLocaleString()}`;
    $("#evaluation-detail-status").textContent =
      job?.status || "Unknown status";
    $("#evaluation-detail-status").className =
      `job-status ${job?.status || ""}`;
    $("#evaluation-detail-message").textContent = job?.message || "";
    error("#evaluation-detail-error", job?.error ? new Error(job.error) : null);
    $("#evaluation-completeness").textContent = complete()
      ? split() === "test"
        ? "Completed test audit. Report this result; choose reference models from validation evidence."
        : "Completed validation evaluation. Metrics cover every frozen frame for each model and inference mode."
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
          evaluation_model_id: model.id,
          model_id: model.model_id,
          variant: model.variant || "full",
          metadata: model.metadata,
          protocol: model.metrics?.protocol,
        })),
        prediction_count: detail.predictions.length,
        frame_timings: detail.predictions.map((prediction) => ({
          evaluation_model_id: prediction.evaluation_model_id,
          model_id: prediction.model_id,
          frame_id: prediction.frame_id,
          timing: prediction.timing,
          tiles: prediction.metadata?.tiles,
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
        `${reference.metadata?.model_name || reference.model_id} · ${modeName(reference.metadata?.variant || "full")}`,
      ),
    );
    if (reference.metadata?.variant === "tiled")
      entry.append(node("p", "field-hint", pipelineDescription(reference.metadata.inference, "tiled")));
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
    ++view.auditPreviewRequest;
    view.auditPreviewKey = null;
    view.auditPreview = null;
    view.auditPreviewPending = false;
    view.auditPreviewError = null;
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
    const payload = evaluationPayload();
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
      await submit(auditPayload(detail));
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
    const lane = lanes().find((item) => laneKey(item) === $("#evaluation-reference-model").value);
    if (!lane) return $("#evaluation-reference-model").focus();
    const payload = {
      evaluation_id: view.detail.id,
      model_id: lane.model_id,
      variant: lane.variant,
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
  $("#evaluation-inference-mode").addEventListener("change", updateLaunch);
  for (const selector of ["tile-size", "tile-overlap", "confidence", "iou"])
    $(`#evaluation-${selector}`).addEventListener("input", updateLaunch);
  $("#evaluation-save-experiment").addEventListener("click", () => {
    if (!complete()) return;
    window.dispatchEvent(new CustomEvent("iris:experiment-create", {
      detail: { evaluation_id: view.detail.id },
    }));
  });
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
  $("#evaluation-errors-only").addEventListener("change", () => {
    if (view.analysis) {
      $("#evaluation-analysis-filter").value = $("#evaluation-errors-only")
        .checked
        ? "errors"
        : "all";
      renderAnalysis();
    } else populateFrames();
  });
  for (const selector of ["class", "filter", "sort"])
    $(`#evaluation-analysis-${selector}`).addEventListener("change", () => {
      $("#evaluation-errors-only").checked =
        $("#evaluation-analysis-filter").value === "errors";
      renderAnalysis();
    });
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
