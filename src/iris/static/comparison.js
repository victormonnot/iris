"use strict";

// The intake workspace owns sessions, frame selection and the shared job queue.
// Comparisons keep their own saved frame snapshot and never change that selection.
(() => {
  const comparison = {
    models: [],
    chosenModels: new Set(),
    choicesTouched: false,
    catalogRequest: 0,
    history: [],
    sessionId: null,
    activeId: null,
    detail: null,
    position: 0,
    historyRequest: 0,
    detailRequest: 0,
    submitting: false,
    previewKey: null,
    previewRequest: 0,
    previewPending: false,
    preview: null,
    previewError: null,
  };
  const svgNamespace = "http://www.w3.org/2000/svg";
  const replay = window.IRISComparisonReplay.create($("#comparison-replay"), {
    urlFor: projectURL,
    onSelectFrame(frameId) {
      const position = comparison.detail?.frame_ids.indexOf(frameId) ?? -1;
      if (position < 0 || position === comparison.position) return;
      comparison.position = position;
      renderFrame();
    },
  });

  function showError(selector, error) {
    const element = $(selector);
    element.textContent = error?.message || "";
    element.hidden = !error;
  }

  function setWorkspace(name) {
    if (!window.dispatchEvent(new CustomEvent("iris:before-workspace", { cancelable: true, detail: { name } }))) return;
    const workspaces = {
      intake: [
        "01",
        "Data intake",
        "From sources to useful frames.",
        "Import source footage, sample frames and select the data worth keeping.",
      ],
      comparison: [
        "02",
        "Model comparison",
        "See what your models see.",
        "Run detectors on a shared selection, inspect their differences and keep a reproducible baseline.",
      ],
      annotation: [
        "03",
        "Annotation",
        "Turn observations into labels.",
        "Review proposals, correct bounding boxes and validate each frame before it becomes training data.",
      ],
      training: [
        "04",
        "Dataset & training",
        "Build on what you have learned.",
        "Freeze validated data, train a local detector and bring its checkpoint back into comparison.",
      ],
    };
    workspaces.evaluation = [
      "05",
      "Quality evaluation",
      "Measure gains and regressions.",
      "Evaluate frozen labels, inspect detection errors and choose a reference model with evidence.",
    ];
    workspaces.experiments = [
      "06",
      "Experiments",
      "Keep the evidence together.",
      "Turn a completed evaluation into a clear experiment record, add your conclusions and share a self-contained report.",
    ];
    workspaces.benchmark = [
      "07", "Preannotation benchmark", "Measure the work behind the labels.",
      "Freeze an independent human reference, separate tuning from evaluation and measure corrections to candidate proposals.",
    ];
    const info = workspaces[name];
    if (!info) return;
    for (const workspace of Object.keys(workspaces)) {
      $(`#${workspace}-workspace`).hidden = workspace !== name;
      const button = $(`#workspace-${workspace}`);
      button.classList.toggle("active", workspace === name);
      button.setAttribute("aria-pressed", String(workspace === name));
    }
    $("#workspace-step").replaceChildren(
      node("span", "step-marker", info[0]),
      document.createTextNode(info[1]),
    );
    $("#workspace-title").textContent = info[2];
    $("#workspace-description").textContent = info[3];
    window.dispatchEvent(
      new CustomEvent("iris:workspace", { detail: { name } }),
    );
  }

  function modelName(id) {
    return comparison.models.find((model) => model.id === id)?.name || id;
  }

  function modelClasses(id) {
    const model = comparison.models.find((item) => item.id === id);
    const taxonomy = comparison.detail?.config?.model_class_contracts?.[id]?.taxonomy || model?.taxonomy;
    return taxonomy?.classes?.map((item) => ({ id: item.id, name: item.name })) ||
      (model?.classes || []).map((item) => ({ id: item.name, name: item.display_name || item.name }));
  }

  function className(modelId, label) {
    return modelClasses(modelId).find((item) => item.id === label)?.name || label;
  }

  function selectedFrames() {
    return state.frames.filter((frame) => frame.selected);
  }

  function inferenceMode() {
    return $("#comparison-inference-mode").value;
  }

  function comparisonPayload(preview = false) {
    return {
      name: preview
        ? "Comparison preview"
        : $("#comparison-name").value.trim() ||
          `Comparison ${comparison.history.length + 1}`,
      frame_ids: selectedFrames().map((frame) => frame.id),
      model_ids: [...comparison.chosenModels],
      device: $("#comparison-device").value,
      inference_mode: inferenceMode(),
      tile_size: inferenceMode() === "full" ? 640 : Number($("#comparison-tile-size").value),
      overlap: inferenceMode() === "full" ? 0.2 : Number($("#comparison-tile-overlap").value),
    };
  }

  function modeName(variant) {
    return variant === "tiled" ? "Tiled" : "Full image";
  }

  function lanes() {
    const detail = comparison.detail;
    return detail.lanes || detail.model_ids.map((modelId) => ({
      model_id: modelId,
      variant: "full",
      run_id: detail.runs.find((run) => run.model_id === modelId)?.id || null,
    }));
  }

  function laneName(lane) {
    return `${modelName(lane.model_id)} · ${modeName(lane.variant)}`;
  }

  function belongsToLane(prediction, lane) {
    if (lane.run_id) return prediction.run_id === lane.run_id;
    const run = comparison.detail.runs.find((item) =>
      item.model_id === lane.model_id &&
      (item.variant || "full") === lane.variant,
    );
    if (run) return prediction.run_id === run.id;
    return !comparison.detail.lanes && prediction.model_id === lane.model_id;
  }

  async function requestPreview(payload, key) {
    const request = ++comparison.previewRequest;
    const sessionId = state.sessionId;
    comparison.previewKey = key;
    comparison.previewPending = true;
    comparison.preview = null;
    comparison.previewError = null;
    try {
      const preview = await api(
        `/api/sessions/${encodeURIComponent(sessionId)}/comparisons/preview`,
        { method: "POST", body: JSON.stringify(payload) },
      );
      if (request !== comparison.previewRequest || sessionId !== state.sessionId)
        return;
      comparison.preview = preview;
    } catch (error) {
      if (request !== comparison.previewRequest || sessionId !== state.sessionId)
        return;
      comparison.previewError = error;
    } finally {
      if (request === comparison.previewRequest && sessionId === state.sessionId) {
        comparison.previewPending = false;
        updateLaunch();
      }
    }
  }

  function updateLaunch() {
    const count = selectedFrames().length;
    const saving = state.bulkSelecting || state.pendingSelections.size > 0;
    const modelCount = comparison.chosenModels.size;
    const mode = inferenceMode();
    const tiled = mode !== "full";
    const tileSize = Number($("#comparison-tile-size").value);
    const overlapText = $("#comparison-tile-overlap").value;
    const overlap = Number(overlapText);
    const validTiles = !tiled || (
      Number.isInteger(tileSize) && tileSize >= 128 && tileSize <= 2048 &&
      overlapText !== "" && Number.isFinite(overlap) && overlap >= 0 && overlap <= 0.5
    );
    $("#comparison-tiling-fields").hidden = !tiled;
    $("#comparison-tile-size").disabled = !tiled;
    $("#comparison-tile-overlap").disabled = !tiled;
    $("#comparison-mode-hint").textContent = mode === "paired"
      ? "Choose one model to compare the same checkpoint on the whole image and on overlapping crops. Boxes return to the original image, with duplicates suppressed per class."
      : tiled
        ? "Run each model on overlapping crops, then merge boxes in the original image. This uses the pixels already captured; it does not add detail."
        : "Run each selected model once on the whole image.";
    $("#comparison-selection").textContent =
      `${count} selected frame${count === 1 ? "" : "s"} · ${modelCount} model${modelCount === 1 ? "" : "s"}`;
    let hint =
      "The selected frames are frozen for this run. Inference stays on this machine.";
    if (saving) hint = "Saving your frame selection…";
    else if (!count) hint = "Select 1–100 frames in Data intake to begin.";
    else if (count > 100)
      hint =
        "Select at most 100 frames for a comparison. Your current selection is too large.";
    else if (!modelCount) hint = "Choose at least one ready model above.";
    else if (modelCount > (mode === "paired" ? 1 : 2))
      hint = mode === "paired" ? "Choose one model for full image vs tiled." : "Choose at most two models.";
    else if (!validTiles)
      hint = "Choose a whole tile size from 128 to 2048 pixels and an overlap from 0 to 0.5.";
    else if ($("#comparison-device").value === "cuda")
      hint =
        "CUDA availability has not been verified. A compatible GPU and CUDA-enabled PyTorch are required.";
    $("#comparison-selection-hint").textContent = hint;
    const valid = Boolean(state.sessionId) && !saving && count > 0 && count <= 100 &&
      modelCount > 0 && modelCount <= (mode === "paired" ? 1 : 2) && validTiles;
    if (valid) {
      const payload = comparisonPayload(true);
      const key = JSON.stringify([state.sessionId, payload]);
      if (key !== comparison.previewKey) requestPreview(payload, key);
    } else if (comparison.previewKey !== null) {
      comparison.previewRequest++;
      comparison.previewKey = null;
      comparison.previewPending = false;
      comparison.preview = null;
      comparison.previewError = null;
    }
    const plan = $("#comparison-plan");
    plan.classList.toggle("inline-error", Boolean(comparison.previewError));
    if (comparison.previewPending) plan.textContent = "Checking the number of model passes…";
    else if (comparison.previewError) plan.textContent = comparison.previewError.message;
    else if (comparison.preview) {
      const preview = comparison.preview;
      const tileCounts = (preview.tiles || []).map((item) => item.tile_count);
      const low = Math.min(...tileCounts);
      const high = Math.max(...tileCounts);
      const tileHint = tiled && tileCounts.length
        ? ` · ${low === high ? low : `${low}–${high}`} tiles per image`
        : "";
      const tileLimit = tiled ? `${preview.limits.max_tiles_per_frame} tiles per image, ` : "";
      plan.textContent = `${preview.forward_passes} model passes + ${preview.warmup_passes} warm-up passes${tileHint}. Limit: ${tileLimit}${preview.limits.max_forward_passes} passes per comparison. All processing stays local.`;
    } else plan.textContent = "";
    $("#run-comparison").disabled =
      comparison.submitting || !valid || comparison.previewPending ||
      !comparison.preview || Boolean(comparison.previewError);
  }

  function renderModels() {
    const catalog = $("#model-catalog");
    catalog.replaceChildren();
    for (const model of comparison.models) {
      const ready = model.status === "ready";
      const card = node(
        "label",
        `model-card${ready ? " ready" : " unavailable"}`,
      );
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.value = model.id;
      checkbox.checked = comparison.chosenModels.has(model.id);
      checkbox.disabled = !ready;
      checkbox.setAttribute("aria-label", `Use ${model.name}`);
      checkbox.addEventListener("change", () => {
        comparison.choicesTouched = true;
        if (checkbox.checked) {
          if (inferenceMode() === "paired") comparison.chosenModels.clear();
          comparison.chosenModels.add(model.id);
        }
        else comparison.chosenModels.delete(model.id);
        for (const input of catalog.querySelectorAll("input"))
          input.checked = comparison.chosenModels.has(input.value);
        updateLaunch();
      });
      const content = node("span", "model-card-content");
      const title = node("span", "model-card-title");
      title.append(
        node("strong", "", model.name),
        node(
          "span",
          `model-availability${ready ? " ready" : ""}`,
          ready ? "Ready" : "Setup required",
        ),
      );
      content.append(
        title,
        node("span", "model-description", model.architecture),
      );
      content.append(
        node(
          "span",
          "model-description",
          ready
            ? `${model.classes.length} classes · verified local weights · runtime checked at launch`
            : model.reason || "Local model dependencies are unavailable.",
        ),
      );
      if (!ready && model.download_bytes)
        content.append(
          node(
            "span",
            "model-description",
            `Weights: ${formatBytes(model.download_bytes)}`,
          ),
        );
      card.append(checkbox, content);
      catalog.append(card);
    }
    if (!comparison.models.length)
      catalog.append(
        node("p", "muted small", "No models are available in the catalog."),
      );
    $("#model-setup").hidden = !comparison.models.some(
      (model) => model.status !== "ready",
    );
    $("#model-setup-commands").textContent =
      "uv sync --locked --extra ml\nuv run --extra ml iris models download --all";
    updateLaunch();
  }

  async function refreshModels() {
    const request = ++comparison.catalogRequest;
    $("#refresh-models").disabled = true;
    showError("#model-error", null);
    try {
      const models = await api("/api/models");
      if (request !== comparison.catalogRequest) return;
      comparison.models = models;
      const ready = models.filter((model) => model.status === "ready");
      comparison.chosenModels = new Set(
        ready
          .filter(
            (model) =>
              !comparison.choicesTouched ||
              comparison.chosenModels.has(model.id),
          )
          .slice(0, inferenceMode() === "paired" ? 1 : 2)
          .map((model) => model.id),
      );
      comparison.previewRequest++;
      comparison.previewKey = null;
      comparison.previewPending = false;
      comparison.preview = null;
      comparison.previewError = null;
      renderModels();
    } catch (error) {
      if (request !== comparison.catalogRequest) return;
      comparison.models = [];
      comparison.chosenModels.clear();
      renderModels();
      showError("#model-error", error);
    } finally {
      if (request === comparison.catalogRequest)
        $("#refresh-models").disabled = false;
    }
  }

  function renderHistory() {
    const select = $("#comparison-history");
    select.replaceChildren();
    $("#comparison-history-count").textContent = comparison.history.length;
    select.disabled = !comparison.history.length;
    if (!comparison.history.length) {
      replay.reset();
      select.append(new Option("No comparisons yet", ""));
      $("#comparison-empty").hidden = false;
      $("#comparison-detail").hidden = true;
      return;
    }
    for (const item of comparison.history) {
      const status = item.job?.status || "Unknown status";
      const created = new Date(item.created_at).toLocaleString();
      select.append(
        new Option(`${item.name} · ${status} · ${created}`, item.id),
      );
    }
    select.value = comparison.activeId || "";
  }

  async function refreshHistory() {
    const sessionId = state.sessionId;
    if (!sessionId) return;
    const request = ++comparison.historyRequest;
    try {
      const history = await api(
        `/api/sessions/${encodeURIComponent(sessionId)}/comparisons`,
      );
      if (
        state.sessionId !== sessionId ||
        request !== comparison.historyRequest
      )
        return;
      comparison.history = history;
      if (!history.some((item) => item.id === comparison.activeId)) {
        comparison.activeId = history[0]?.id || null;
        comparison.detail = null;
        comparison.position = 0;
        replay.reset();
      }
      renderHistory();
      showError("#comparison-history-error", null);
      if (comparison.activeId) await loadDetail(comparison.activeId);
    } catch (error) {
      if (
        state.sessionId === sessionId &&
        request === comparison.historyRequest
      )
        showError("#comparison-history-error", error);
    }
  }

  async function loadDetail(id) {
    const request = ++comparison.detailRequest;
    const sessionId = state.sessionId;
    try {
      const detail = await api(`/api/comparisons/${encodeURIComponent(id)}`);
      if (
        request !== comparison.detailRequest ||
        comparison.activeId !== id ||
        state.sessionId !== sessionId
      )
        return;
      showError("#comparison-history-error", null);
      if (JSON.stringify(detail) === JSON.stringify(comparison.detail)) return;
      const isNew = comparison.detail?.id !== id;
      comparison.detail = detail;
      if (isNew) comparison.position = 0;
      renderDetail(isNew);
    } catch (error) {
      if (request === comparison.detailRequest && state.sessionId === sessionId)
        showError("#comparison-history-error", error);
    }
  }

  function populateClasses(reset) {
    const select = $("#comparison-class");
    const previous = reset ? "" : select.value;
    const labels = new Map();
    const addLabel = (id, name) => {
      if (!labels.has(id)) labels.set(id, new Set());
      labels.get(id).add(name);
    };
    for (const prediction of comparison.detail.predictions)
      for (const detection of prediction.detections)
        addLabel(detection.label, className(prediction.model_id, detection.label));
    for (const modelId of comparison.detail.model_ids)
      for (const label of modelClasses(modelId)) addLabel(label.id, label.name);
    select.replaceChildren(new Option("All classes", ""));
    for (const [label, names] of [...labels].sort(([left], [right]) => left.localeCompare(right)))
      select.append(new Option([...names].join(" / "), label));
    select.value = labels.has(previous) ? previous : "";
  }

  function renderDetail(resetFilters) {
    const detail = comparison.detail;
    $("#comparison-empty").hidden = true;
    $("#comparison-detail").hidden = false;
    $("#comparison-run-name").textContent = detail.name;
    const mode = detail.config.inference?.mode || "full";
    const modeLabel = mode === "paired" ? "Full image vs tiled" : modeName(mode);
    $("#comparison-run-context").textContent =
      `${detail.frame_ids.length} saved frames · ${modeLabel} · ${detail.config.device.toUpperCase()} · ${new Date(detail.created_at).toLocaleString()}`;
    const job = detail.job;
    const badge = $("#comparison-run-status");
    badge.className = `job-status ${job?.status || ""}`;
    badge.textContent = job?.status || "Unknown status";
    $("#comparison-run-message").textContent = job?.message || "";
    showError(
      "#comparison-run-error",
      job?.error ? new Error(job.error) : null,
    );
    populateClasses(resetFilters);
    renderSignals();
    renderFrame();
    replay.update(detail, detail.frame_ids[comparison.position]);
    if ($("#comparison-workspace").hidden || document.hidden) replay.deactivate();
    renderProvenance();
  }

  function displayedDetections(prediction) {
    const confidence = Number($("#comparison-confidence").value);
    const label = $("#comparison-class").value;
    return (prediction?.detections || []).filter(
      (detection) =>
        detection.score >= confidence && (!label || detection.label === label),
    );
  }

  function predictionFor(frameId, lane) {
    return comparison.detail.predictions.find(
      (prediction) =>
        prediction.frame_id === frameId && belongsToLane(prediction, lane),
    );
  }

  function renderSignals() {
    const detail = comparison.detail;
    const container = $("#comparison-signals");
    container.replaceChildren();
    const runLanes = lanes();
    for (const lane of runLanes) {
      const predictions = detail.predictions.filter(
        (item) => belongsToLane(item, lane),
      );
      const count = predictions.reduce(
        (sum, prediction) => sum + displayedDetections(prediction).length,
        0,
      );
      const signal = node("div", "comparison-signal");
      signal.append(
        node("span", "", laneName(lane)),
        node("strong", "", String(count)),
        node(
          "span",
          "",
          `displayed detections · ${predictions.length}/${detail.frame_ids.length} frames processed`,
        ),
      );
      container.append(signal);
    }
    if (runLanes.length === 2) {
      let pairs = 0;
      let differences = 0;
      for (const frameId of detail.frame_ids) {
        const left = predictionFor(frameId, runLanes[0]);
        const right = predictionFor(frameId, runLanes[1]);
        if (!left || !right) continue;
        pairs++;
        if (
          displayedDetections(left).length !== displayedDetections(right).length
        )
          differences++;
      }
      const signal = node("div", "comparison-signal");
      signal.append(
        node("span", "", "Detection count differences"),
        node("strong", "", pairs ? `${differences}/${pairs}` : "—"),
        node(
          "span",
          "",
          "jointly processed frames · counts only, no box matching",
        ),
      );
      container.append(signal);
    }
  }

  function svgNode(tag, attributes = {}) {
    const element = document.createElementNS(svgNamespace, tag);
    for (const [name, value] of Object.entries(attributes))
      element.setAttribute(name, String(value));
    return element;
  }

  function predictionVisual(frame, detections, color, modelId) {
    const svg = svgNode("svg", {
      viewBox: `0 0 ${frame.width} ${frame.height}`,
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
      "aria-label": `${frame.source_filename || "Frame"}, ${detections.length} displayed detections`,
    });
    svg.append(
      svgNode("image", {
        href: projectURL(`/api/frames/${encodeURIComponent(frame.id)}/image`),
        width: frame.width,
        height: frame.height,
        preserveAspectRatio: "none",
      }),
    );
    const fontSize = Math.max(10, Math.min(frame.width, frame.height) * 0.029);
    for (const detection of detections) {
      const [x1, y1, x2, y2] = detection.box;
      const group = svgNode("g");
      const title = svgNode("title");
      title.textContent = `${className(modelId, detection.label)} · ${(detection.score * 100).toFixed(1)}%`;
      group.append(
        title,
        svgNode("rect", {
          x: x1,
          y: y1,
          width: Math.max(0, x2 - x1),
          height: Math.max(0, y2 - y1),
          fill: "none",
          stroke: color,
          "stroke-width": 2,
          "vector-effect": "non-scaling-stroke",
        }),
      );
      const caption = `${className(modelId, detection.label)} ${(detection.score * 100).toFixed(0)}%`;
      const text = svgNode("text", {
        x: Math.max(
          2,
          Math.min(x1 + 3, frame.width - caption.length * fontSize * 0.6 - 3),
        ),
        y: Math.max(fontSize + 2, y1 - 4),
        fill: color,
        stroke: "#10221d",
        "stroke-width": fontSize / 6,
        "stroke-linejoin": "round",
        "paint-order": "stroke",
        "font-size": fontSize,
        "font-family": "ui-sans-serif, sans-serif",
        "font-weight": 600,
      });
      text.textContent = caption;
      group.append(text);
      svg.append(group);
    }
    return svg;
  }

  function formatMilliseconds(value) {
    return Number.isFinite(value) ? `${value.toFixed(1)} ms` : "Unavailable";
  }

  function renderFrame() {
    const detail = comparison.detail;
    comparison.position = Math.max(
      0,
      Math.min(comparison.position, detail.frame_ids.length - 1),
    );
    const frameId = detail.frame_ids[comparison.position];
    const frame = detail.frames.find((item) => item.id === frameId);
    const replaySample = detail.replay?.sources.flatMap((source) => source.samples)
      .find((sample) => sample.frame_id === frameId);
    const imageAvailable = Boolean(frame) && replaySample?.image_available !== false;
    const sourcePosition = replaySample
      ? Number.isFinite(replaySample.timestamp_seconds)
        ? `≈ ${timestamp(replaySample.timestamp_seconds)}`
        : "Video timestamp unavailable"
      : timestamp(frame?.timestamp_seconds);
    $("#comparison-position").textContent =
      `Frame ${comparison.position + 1} / ${detail.frame_ids.length}`;
    $("#comparison-previous").disabled = comparison.position === 0;
    $("#comparison-next").disabled =
      comparison.position >= detail.frame_ids.length - 1;
    $("#comparison-frame-source").textContent = frame
      ? `${frame.source_filename || frame.asset_id} · ${sourcePosition} · ${frame.width} × ${frame.height}`
      : "Frame source unavailable";
    const container = $("#comparison-canvases");
    container.replaceChildren();
    const runLanes = lanes();
    container.classList.toggle("single-model", runLanes.length === 1);
    runLanes.forEach((lane, index) => {
      const prediction = predictionFor(frameId, lane);
      const detections = displayedDetections(prediction);
      const card = node("article", "prediction-card");
      const heading = node("div", "prediction-heading");
      heading.append(
        node("h3", "", laneName(lane)),
        node(
          "span",
          "prediction-count",
          prediction ? `${detections.length} detections` : "Not processed",
        ),
      );
      const visual = node("div", "prediction-visual");
      if (imageAvailable)
        visual.append(
          predictionVisual(
            frame,
            detections,
            index === 0 ? "#b1ee88" : "#80d4ff",
            lane.model_id,
          ),
        );
      if (!imageAvailable || !prediction || !detections.length)
        visual.append(
          node(
            "span",
            "prediction-empty",
            !imageAvailable
              ? "Saved image unavailable. Prediction data is retained below."
              : !prediction
              ? "Not processed"
              : `No detections at this display threshold${$("#comparison-class").value ? " for this class" : ""}`,
          ),
        );
      card.append(heading, visual);
      if (prediction) {
        const timing = node("div", "prediction-timing");
        timing.append(
          node(
            "span",
            "",
            `${lane.variant === "tiled" ? "Summed model forwards" : "Model forward"} ${formatMilliseconds(prediction.timing?.inference_ms)}`,
          ),
          node(
            "span",
            "",
            `Total ${formatMilliseconds(prediction.timing?.total_ms)}`,
          ),
        );
        if (lane.variant === "tiled")
          timing.append(node("span", "", `${prediction.timing?.tile_count ?? "—"} tiles · ${prediction.timing?.forward_passes ?? "—"} model passes`));
        const details = node("details", "prediction-data");
        details.append(node("summary", "", "Detections and timing details"));
        const timingNote = node(
          "p",
          "field-hint",
          lane.variant === "tiled"
            ? "Model forward is summed across all tiles. Total includes image decoding, cropping, tensor preparation, model forwards, output serialization and merging. The warm-up is excluded. These are local wall-clock timings, not an accuracy score."
            : "Model forward includes internal resize, normalization, suppression and coordinate restoration. Total includes decoding, tensor preparation and output serialization. The warm-up is excluded. These are local wall-clock timings, not an accuracy score.",
        );
        const timingList = node("dl", "timing-list");
        const timingEntries = [
          ["Decode", "decode_ms"],
          ...(lane.variant === "tiled" ? [["Crop tiles", "crop_ms"]] : []),
          ["Prepare tensor / device", "preprocess_ms"],
          [lane.variant === "tiled" ? "Summed model forwards" : "Model forward", "inference_ms"],
          ["Transfer / serialize output", "postprocess_ms"],
          ...(lane.variant === "tiled" ? [["Merge / suppress duplicates", "merge_ms"]] : []),
          ["Total", "total_ms"],
        ];
        for (const [label, key] of timingEntries) {
          const entry = node("div");
          entry.append(
            node("dt", "", label),
            node("dd", "", formatMilliseconds(prediction.timing?.[key])),
          );
          timingList.append(entry);
        }
        details.append(timingNote, timingList);
        if (detections.length) {
          const list = node("ul", "detection-list");
          for (const detection of detections)
            list.append(
              node(
                "li",
                "",
                `${className(lane.model_id, detection.label)} · ${(detection.score * 100).toFixed(1)}% · box [${detection.box.map((value) => Math.round(value)).join(", ")}] px`,
              ),
            );
          details.append(list);
        }
        card.append(timing);
        if (lane.variant === "tiled" && prediction.timing?.truncated_detection_count > 0)
          card.append(node(
            "p",
            "prediction-note",
            `The saved-output limit kept ${prediction.timing.kept_detection_count} boxes after duplicate suppression; ${prediction.timing.truncated_detection_count} lower-scored boxes were discarded.`,
          ));
        card.append(details);
      } else {
        card.append(
          node(
            "p",
            "prediction-note",
            "No saved prediction for this run and frame.",
          ),
        );
      }
      container.append(card);
    });
  }

  function renderProvenance() {
    const container = $("#comparison-provenance");
    container.replaceChildren();
    for (const run of comparison.detail.runs) {
      const item = node("div", "run-metadata");
      item.append(
        node("h3", "", laneName(run)),
        node("pre", "", JSON.stringify(run.metadata, null, 2)),
      );
      container.append(item);
    }
    if (!comparison.detail.runs.length)
      container.append(
        node(
          "p",
          "muted small",
          "Model provenance will appear when processing starts.",
        ),
      );
  }

  function sessionChanged() {
    if (comparison.sessionId === state.sessionId) return;
    replay.reset();
    comparison.sessionId = state.sessionId;
    comparison.history = [];
    comparison.activeId = null;
    comparison.detail = null;
    comparison.position = 0;
    comparison.detailRequest++;
    $("#comparison-name").value = "";
    $("#comparison-detail").hidden = true;
    showError("#comparison-error", null);
    showError("#comparison-history-error", null);
    renderHistory();
    updateLaunch();
    refreshHistory();
  }

  $("#workspace-intake").addEventListener("click", () =>
    setWorkspace("intake"),
  );
  $("#workspace-comparison").addEventListener("click", () =>
    setWorkspace("comparison"),
  );
  $("#workspace-annotation").addEventListener("click", () =>
    setWorkspace("annotation"),
  );
  $("#workspace-training").addEventListener("click", () =>
    setWorkspace("training"),
  );
  $("#workspace-evaluation").addEventListener("click", () =>
    setWorkspace("evaluation"),
  );
  $("#workspace-experiments").addEventListener("click", () =>
    setWorkspace("experiments"),
  );
  $("#workspace-benchmark").addEventListener("click", () => setWorkspace("benchmark"));
  $("#refresh-models").addEventListener("click", refreshModels);
  $("#comparison-device").addEventListener("change", updateLaunch);
  $("#comparison-inference-mode").addEventListener("change", () => {
    if (inferenceMode() === "paired" && comparison.chosenModels.size > 1) {
      comparison.chosenModels = new Set([...comparison.chosenModels].slice(0, 1));
      notify("Full image vs tiled uses one checkpoint. The first selected model has been kept.");
    }
    renderModels();
  });
  for (const selector of ["#comparison-tile-size", "#comparison-tile-overlap"])
    $(selector).addEventListener("input", updateLaunch);
  $("#comparison-history").addEventListener("change", (event) => {
    replay.reset();
    comparison.activeId = event.target.value;
    comparison.detail = null;
    $("#comparison-detail").hidden = true;
    loadDetail(comparison.activeId);
  });
  for (const [selector, step] of [
    ["#comparison-previous", -1],
    ["#comparison-next", 1],
  ]) {
    $(selector).addEventListener("click", () => {
      comparison.position += step;
      renderFrame();
      replay.selectFrame(comparison.detail.frame_ids[comparison.position], { seek: true });
    });
  }
  for (const selector of ["#comparison-confidence", "#comparison-class"]) {
    $(selector).addEventListener("input", () => {
      $("#comparison-confidence-value").textContent = Number(
        $("#comparison-confidence").value,
      ).toFixed(2);
      if (!comparison.detail) return;
      renderSignals();
      renderFrame();
    });
  }
  $("#comparison-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if ($("#run-comparison").disabled) return;
    const sessionId = state.sessionId;
    const payload = comparisonPayload();
    comparison.submitting = true;
    updateLaunch();
    showError("#comparison-error", null);
    try {
      const result = await api(
        `/api/sessions/${encodeURIComponent(sessionId)}/comparisons`,
        {
          method: "POST",
          body: JSON.stringify(payload),
        },
      );
      if (state.sessionId === sessionId) {
        replay.reset();
        comparison.activeId = result.id;
        comparison.detail = null;
        $("#comparison-name").value = "";
        await refreshHistory();
      }
      notify(
        `Comparison queued for ${payload.frame_ids.length} frames. Follow its progress in Processing jobs.`,
      );
      await refreshJobs();
    } catch (error) {
      if (state.sessionId === sessionId) showError("#comparison-error", error);
      else notify(error.message, true);
    } finally {
      comparison.submitting = false;
      updateLaunch();
    }
  });
  window.addEventListener("iris:session", sessionChanged);
  window.addEventListener("iris:workspace", (event) => {
    if (event.detail?.name !== "comparison") replay.deactivate();
    else if (comparison.detail)
      replay.update(comparison.detail, comparison.detail.frame_ids[comparison.position]);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) replay.deactivate();
    else if (!$("#comparison-workspace").hidden && comparison.detail)
      replay.update(comparison.detail, comparison.detail.frame_ids[comparison.position]);
  });
  window.addEventListener("pagehide", () => replay.deactivate());
  window.addEventListener("iris:frames", updateLaunch);
  window.addEventListener("iris:jobs", () => {
    if (state.sessionId) refreshHistory();
  });
  window.addEventListener("iris:models", async (event) => {
    if (event.detail?.model_ids) {
      comparison.choicesTouched = true;
      comparison.chosenModels = new Set(event.detail.model_ids.slice(0, inferenceMode() === "paired" ? 1 : 2));
    }
    await refreshModels();
    if (event.detail?.openComparison) setWorkspace("comparison");
  });
  refreshModels();
  sessionChanged();
})();
