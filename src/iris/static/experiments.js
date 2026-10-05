"use strict";

(() => {
  const datasetTools = window.IRISDatasetTools;
  const taxonomyTools = window.IRISTaxonomyTools;
  const reportTaxonomy = (snapshot) => datasetTools.taxonomyOf(
    snapshot.evaluation?.config?.taxonomy ? snapshot.evaluation : snapshot.dataset,
  );
  const aggregateCounts = (snapshot, counts) => counts?.[datasetTools.aggregateFilter(snapshot.error_analysis)];
  const field = (name) => $(`#experiments-${name}`);
  const library = {
    visible: false,
    rows: [],
    activeId: null,
    detail: null,
    listRequest: 0,
    detailRequest: 0,
    listLoading: false,
    exportRequest: 0,
    exportController: null,
    urls: new Map(),
  };
  const compose = {
    context: 0,
    request: 0,
    preview: null,
    selected: new Set(),
    measurements: new Set(),
    limit: 12,
    busy: false,
    loading: false,
    dirty: false,
    autoTitle: "",
  };
  const edit = {
    context: 0,
    id: null,
    revision: null,
    busy: false,
    dirty: false,
    conflict: false,
  };
  const svgNS = "http://www.w3.org/2000/svg";
  const safe = encodeURIComponent;
  const ratio = (value) =>
    Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "N/A";
  const number = (value) =>
    Number.isFinite(value) ? value.toLocaleString() : "N/A";
  const milliseconds = (value) =>
    Number.isFinite(value) ? `${value.toFixed(1)} ms` : "Not recorded";
  const measurementHardware = (measurement) =>
    measurement.environment?.cuda?.name ||
    measurement.environment?.processor ||
    measurement.environment?.machine ||
    "Hardware not recorded";
  const changeLabel = (value) => ({
    improved: "Improved", regressed: "Regressed", mixed: "Mixed changes",
    unchanged: "Unchanged", single: "Single pipeline",
  })[value] || "Change unavailable";
  const mode = (value) => (value === "tiled" ? "Tiled" : "Full image");
  const date = (value) =>
    new Date(value).toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  const split = (value) => (value === "test" ? "Test audit" : "Validation");
  const depth = (value) =>
    ({
      prediction_head_only: "Light adaptation",
      partial_backbone: "Partial adaptation",
      full_model: "Full adaptation",
    })[value || "prediction_head_only"] || value;
  const laneTitle = (lane) => {
    const name = lane.name || lane.model_id;
    const suffix = ` · ${mode(lane.variant)}`;
    return name.endsWith(suffix) ? name : `${name}${suffix}`;
  };
  const composeCurrent = (context) =>
    field("compose-dialog").open && context === compose.context;
  const editCurrent = (context) =>
    field("edit-dialog").open && context === edit.context;

  function error(name, failure) {
    field(name).textContent = failure?.message || failure || "";
    field(name).hidden = !failure;
  }

  function localURL(value) {
    const url = new URL(value, location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith("/api/"))
      throw new Error("This image is not available from the local workspace.");
    return projectURL(url.href);
  }

  function orderedLanes(snapshot) {
    const comparison = snapshot.error_analysis?.comparison;
    if (!comparison) return snapshot.lanes;
    return [comparison.baseline_run_id, comparison.candidate_run_id]
      .map((id) => snapshot.lanes.find((lane) => lane.id === id))
      .filter(Boolean);
  }

  function updateCompose() {
    const busy = compose.busy || compose.loading;
    for (const input of field("compose-form").querySelectorAll(
      "input, select, textarea",
    ))
      input.disabled = busy;
    for (const input of field("example-picker").querySelectorAll("input"))
      input.disabled = busy || (!input.checked && compose.selected.size >= 6);
    for (const input of field("measurement-picker").querySelectorAll("input"))
      input.disabled = busy || (!input.checked && compose.measurements.size >= 4);
    field("create").disabled =
      busy || !compose.preview || compose.selected.size > 6 || compose.measurements.size > 4;
    field("create").textContent = compose.busy
      ? "Saving report…"
      : "Save experiment →";
    field("more-examples").disabled = busy;
    field("preview-refresh").disabled = busy;
    field("suggest-examples").disabled = busy || compose.selected.size >= 6 ||
      !compose.preview?.snapshot.insights?.suggested_examples?.some(
        (item) => !compose.selected.has(item.frame_id),
      );
    field("clear-examples").disabled = busy || !compose.selected.size;
    field("picked-count").textContent = `${compose.selected.size} / 6`;
    field("measurement-count").textContent = `${compose.measurements.size} / 4`;
  }

  function renderList() {
    const query = field("search").value.trim().toLocaleLowerCase();
    const visible = library.rows.filter((item) =>
      [
        item.title,
        item.objective,
        item.dataset?.name,
        item.evaluation?.name,
        ...(item.lanes || []).map((lane) => lane.name),
      ]
        .filter(Boolean)
        .join(" ")
        .toLocaleLowerCase()
        .includes(query),
    );
    const focusedId = document.activeElement?.closest("[data-experiment-id]")
      ?.dataset.experimentId;
    field("list").replaceChildren();
    field("count").textContent = `${visible.length}`;
    field("no-matches").hidden = Boolean(visible.length);
    field("empty").hidden = Boolean(library.rows.length) || library.listLoading;
    field("layout").hidden = !library.rows.length;
    for (const item of visible) {
      const button = node(
        "button",
        `experiments-list-item${item.id === library.activeId ? " active" : ""}`,
      );
      button.type = "button";
      button.dataset.experimentId = item.id;
      button.setAttribute("aria-current", String(item.id === library.activeId));
      button.append(
        node(
          "span",
          "experiments-list-meta",
          `${split(item.evaluation?.split)} · ${date(item.created_at)}`,
        ),
      );
      button.append(node("strong", "", item.title));
      button.append(
        node(
          "span",
          "experiments-list-dataset",
          item.dataset?.name || "Frozen dataset",
        ),
      );
      button.append(
        node(
          "span",
          "experiments-list-foot",
          `${item.lanes?.length || 0} pipeline(s) · ${item.example_count || 0} examples`,
        ),
      );
      button.addEventListener("click", () => selectReport(item.id, true));
      field("list").append(button);
      if (focusedId === item.id) button.focus({ preventScroll: true });
    }
  }

  async function refreshList(preferredId = library.activeId) {
    const request = ++library.listRequest;
    library.listLoading = true;
    field("refresh").disabled = true;
    field("status").textContent = "Loading saved experiments…";
    error("error", null);
    try {
      const rows = await api("/api/experiments");
      if (request !== library.listRequest || !library.visible) return;
      library.rows = rows;
      library.activeId = rows.some((item) => item.id === preferredId)
        ? preferredId
        : rows[0]?.id || null;
      field("status").textContent =
        `${rows.length} saved experiment${rows.length === 1 ? "" : "s"} · across the current project`;
      if (library.activeId) await selectReport(library.activeId);
      else {
        ++library.detailRequest;
        library.detail = null;
        field("detail").hidden = true;
      }
    } catch (failure) {
      if (request === library.listRequest && library.visible) {
        field("status").textContent =
          "Could not refresh saved reports. Try Refresh again.";
        error("error", failure);
      }
    } finally {
      if (request === library.listRequest) {
        library.listLoading = false;
        field("refresh").disabled = false;
        renderList();
      }
    }
  }

  function releaseURL(url) {
    clearTimeout(library.urls.get(url));
    library.urls.delete(url);
    URL.revokeObjectURL(url);
  }

  function resetExport(message = "") {
    ++library.exportRequest;
    library.exportController?.abort();
    library.exportController = null;
    field("export").disabled = !library.detail;
    field("export").textContent = "Download report ↓";
    field("export-cancel").hidden = true;
    field("export-images").disabled = !library.detail?.images?.length;
    field("export-status").textContent = message;
    error("export-error", null);
  }

  async function selectReport(id, focus = false) {
    const request = ++library.detailRequest;
    library.activeId = id;
    library.detail = null;
    resetExport();
    field("detail").hidden = true;
    field("detail-status").textContent = "Opening the frozen report…";
    renderList();
    try {
      const detail = await api(`/api/experiments/${safe(id)}`);
      if (
        request !== library.detailRequest ||
        id !== library.activeId ||
        !library.visible
      )
        return;
      library.detail = detail;
      renderReport();
      field("detail-status").textContent = "";
      if (focus) field("report-title").focus({ preventScroll: true });
    } catch (failure) {
      if (request === library.detailRequest && library.visible) {
        field("detail-status").textContent =
          "This report could not be opened. Select it again to retry.";
        error("error", failure);
      }
    }
  }

  function renderLineage(snapshot, lanes) {
    const container = field("lineage");
    container.replaceChildren();
    const training = lanes.map((lane) =>
      lane.training
        ? `${depth(lane.training.config?.scope)} · ${lane.training.history_summary?.steps_completed ?? lane.training.config?.steps ?? "?"} steps`
        : lane.training_status === "pretrained"
          ? "Official pretrained weights"
          : "Training history unavailable",
    );
    const stages = [
      [
        "01",
        "Dataset",
        snapshot.dataset.name,
        `${snapshot.dataset.summary?.frame_count ?? "?"} frozen images`,
      ],
      [
        "02",
        "Training",
        [...new Set(training)].join(" / "),
        "Recorded checkpoint lineage",
      ],
      [
        "03",
        "Checkpoint",
        lanes.map((lane) => lane.name || lane.model_id).join(" / "),
        lanes.map((lane) => mode(lane.variant)).join(" / "),
      ],
      [
        "04",
        "Evaluation",
        snapshot.evaluation.name,
        `${split(snapshot.evaluation.split)} · saved results`,
      ],
    ];
    for (const [step, label, value, hint] of stages) {
      const card = node("div", "experiments-stage");
      card.append(
        node("span", "experiments-stage-number", step),
        node("span", "experiments-stage-label", label),
      );
      card.append(
        node("strong", "", value),
        node("span", "experiments-caption", hint),
      );
      container.append(card);
    }
  }

  function renderMetrics(snapshot, lanes) {
    field("lane-key").replaceChildren();
    lanes.forEach((lane, index) => {
      const label = node("p", `experiments-lane experiments-lane-${index}`);
      label.append(
        node(
          "span",
          "",
          lanes.length === 1
            ? "Evaluated pipeline"
            : index
              ? "Candidate"
              : "Baseline",
        ),
      );
      label.append(node("strong", "", laneTitle(lane)));
      field("lane-key").append(label);
    });
    const candidate = lanes.at(-1)?.metrics?.summary || {};
    const baseline = lanes[0]?.metrics?.summary || {};
    field("metrics").replaceChildren();
    for (const [key, label] of [
      ["map", "mAP .50–.95"],
      ["map50", "AP50"],
      ["map75", "AP75"],
      ["precision", "Precision"],
      ["recall", "Recall"],
      ["f1", "F1"],
    ]) {
      const card = node("div", "experiments-metric");
      card.append(
        node("span", "experiments-metric-label", label),
        node("strong", "", ratio(candidate[key])),
      );
      if (lanes.length > 1) {
        const difference =
          Number.isFinite(candidate[key]) && Number.isFinite(baseline[key])
            ? `${candidate[key] - baseline[key] >= 0 ? "+" : ""}${((candidate[key] - baseline[key]) * 100).toFixed(1)} pp`
            : "N/A";
        card.append(
          node("span", "experiments-metric-delta", `${difference} vs baseline`),
        );
        card.append(
          node(
            "span",
            "experiments-caption",
            `Baseline ${ratio(baseline[key])}`,
          ),
        );
      } else
        card.append(
          node("span", "experiments-caption", "Single evaluated pipeline"),
        );
      field("metrics").append(card);
    }
    const count =
      candidate.frame_count ??
      aggregateCounts(snapshot, snapshot.error_analysis?.summary)?.frame_count;
    field("frame-count").textContent = `${number(count)} evaluated frames`;
    const config = snapshot.evaluation.config;
    field("metric-context").textContent =
      `Precision and recall use confidence ≥ ${config.confidence_threshold} and IoU ≥ ${config.iou_threshold}. AP uses its saved COCO protocol.` +
      (lanes.length > 1
        ? " Deltas are candidate minus baseline on the same frozen frames, in percentage points; they are not a decision about which model to use."
        : " This report contains one evaluated pipeline.");
    const table = node("table");
    table.append(
      node(
        "caption",
        "sr-only",
        "Detection quality by class and evaluated pipeline",
      ),
    );
    const head = node("thead"),
      heading = node("tr"),
      body = node("tbody");
    for (const label of [
      "Class / pipeline",
      "Labeled",
      "AP",
      "Precision",
      "Recall",
      "TP / FP / FN",
    ]) {
      const cell = node("th", "", label);
      cell.scope = "col";
      heading.append(cell);
    }
    head.append(heading);
    for (const category of reportTaxonomy(snapshot).classes) {
      const label = category.id;
      lanes.forEach((lane, index) => {
        const metric = lane.metrics?.per_class?.find(
          (item) => item.label === label,
        );
        if (!metric) return;
        const row = node("tr");
        const name = node(
          "th",
          "",
          `${category.name} · ${lanes.length === 1 ? "evaluated" : index ? "candidate" : "baseline"}`,
        );
        name.scope = "row";
        row.append(name);
        for (const value of [
          number(metric.support),
          ratio(metric.ap),
          ratio(metric.precision),
          ratio(metric.recall),
          `${number(metric.tp)} / ${number(metric.fp)} / ${number(metric.fn)}`,
        ])
          row.append(node("td", "", value));
        body.append(row);
      });
    }
    table.append(head, body);
    field("class-table").replaceChildren(table);
    field("timing").textContent =
      lanes
        .map(
          (lane, index) =>
            `${lanes.length === 1 ? "Evaluated pipeline" : index ? "Candidate" : "Baseline"} IRIS mean total: ${Number.isFinite(lane.timing?.mean_total_ms) ? `${lane.timing.mean_total_ms.toFixed(1)} ms / image` : "not recorded"}`,
        )
        .join(" · ") +
      ". " + (snapshot.insights?.timing?.scope ||
        "IRIS evaluation timing includes image decoding. The saved device and timing protocol are recorded below.") +
      (snapshot.insights?.timing?.comparable === false
        ? ` No direct timing comparison: ${(snapshot.insights.timing.reasons || []).join("; ") || "the saved protocols are not comparable"}.`
        : " Timing alone does not establish deployment performance.");
    field("warnings").replaceChildren();
    const warnings = new Set([
      ...(config.warnings || []),
      ...(snapshot.error_analysis?.warnings || []),
      ...lanes.flatMap((lane) => lane.metrics?.warnings || []),
      ...(snapshot.evaluation.split === "test"
        ? [
            "Final test audit: this report records the result. Use validation evidence for model selection and tuning.",
          ]
        : []),
    ]);
    for (const warning of warnings)
      field("warnings").append(node("p", "field-hint", warning));
  }

  function renderScenes(snapshot, lanes) {
    const insights = snapshot.insights;
    field("scenes").replaceChildren();
    field("sampling").replaceChildren();
    if (!insights) {
      field("scene-context").textContent =
        "Scene comparisons and sampled-video context were not captured in this older report. Its saved results remain unchanged.";
      return;
    }
    const changes = Object.values(insights.frame_changes || {});
    field("scene-context").textContent = lanes.length > 1
      ? ["regressed", "improved", "mixed", "unchanged"].map(
          (kind) => `${changes.filter((value) => value === kind).length} ${changeLabel(kind).toLowerCase()}`,
        ).join(" · ") + ". Image changes use matched labels and false positives at the saved thresholds; they are separate from dataset AP. Negative images have no evaluated ground-truth objects."
      : "Saved per-scene counts for this single pipeline. Negative images have no evaluated ground-truth objects.";
    if (insights.scenes?.length) {
      const table = node("table"), head = node("thead"), heading = node("tr"), body = node("tbody");
      table.append(node("caption", "sr-only", "Saved detection errors by scene and pipeline"));
      for (const title of ["Scene / pipeline", "Images", "Negative", "TP / FP / FN", "Recovered / new misses", "FP change"]) {
        const cell = node("th", "", title);
        cell.scope = "col";
        heading.append(cell);
      }
      head.append(heading);
      for (const scene of insights.scenes) {
        const counts = aggregateCounts(snapshot, scene.counts);
        lanes.forEach((lane, index) => {
          const row = node("tr");
          const title = node("th", "", `${scene.scene_group} · ${lanes.length === 1 ? "evaluated" : index ? "candidate" : "baseline"}`);
          title.scope = "row";
          row.append(title);
          const run = counts?.runs?.[lane.id];
          for (const value of [
            number(scene.frame_count), number(scene.negative_frame_count),
            run ? `${number(run.tp)} / ${number(run.fp)} / ${number(run.fn)}` : "Not recorded",
            index && counts?.changes ? `${number(counts.changes.recovered)} / ${number(counts.changes.new_misses)}` : "—",
            index && Number.isFinite(counts?.changes?.fp_delta)
              ? `${counts.changes.fp_delta >= 0 ? "+" : ""}${number(counts.changes.fp_delta)}` : "—",
          ]) row.append(node("td", "", value));
          body.append(row);
        });
      }
      table.append(head, body);
      field("scenes").append(table);
    }
    const sampling = insights.sampling;
    if (!sampling) {
      field("sampling").append(node("p", "field-hint", "Video sampling context was not captured in this report."));
      return;
    }
    field("sampling").append(node("h4", "", "Image and sampled-video coverage"));
    field("sampling").append(node("p", "field-hint",
      `${number(sampling.still_image_count)} still images · ${number(sampling.unknown_source_count)} images with unknown source type. ${sampling.warning || "Video results cover sampled images, not continuous inference, tracking or video throughput."}`));
    for (const source of sampling.video_sources || []) {
      const range = Number.isFinite(source.first_timestamp_seconds) && Number.isFinite(source.last_timestamp_seconds)
        ? `${timestamp(source.first_timestamp_seconds)}–${timestamp(source.last_timestamp_seconds)} (approximate)`
        : "Timestamp range unavailable";
      field("sampling").append(node("p", "experiments-caption",
        `${source.filename || source.source_id} · ${number(source.frame_count)} sampled frames · ${number(source.timestamps_available)} saved timestamps · ${range}`));
    }
  }

  function measurementHeading(measurement) {
    return `${measurement.name || measurement.id} · ${(measurement.environment?.device || measurement.profile?.device || "Unknown device").toUpperCase()} · ${measurementHardware(measurement)}`;
  }

  function renderTargets(snapshot, lanes) {
    field("targets").replaceChildren();
    field("target-limitations").replaceChildren();
    const deployment = snapshot.deployments;
    const measurements = deployment?.measurements || [];
    field("target-count").textContent = `${measurements.length}`;
    field("target-context").textContent = !deployment
      ? "Target measurements were not captured in this older report. Nothing has been inferred from the current workspace."
      : !measurements.length
        ? "No target measurements were selected when this report was saved. Evaluation quality above does not establish deployment speed."
        : "Imported measurements describe the declared target and environment; IRIS has not verified their execution. Runner totals exclude image decoding, which is reported separately. IRIS evaluation totals include decoding. These timings do not establish a winner across different targets.";
    for (const measurement of measurements) {
      const card = node("article", "experiments-target");
      card.append(node("h4", "", measurementHeading(measurement)));
      const lane = lanes.find((item) => item.id === measurement.lane_id);
      card.append(node("p", "experiments-caption",
        `${lane ? laneTitle(lane) : measurement.model_id} · ${measurement.profile?.precision || "Precision not recorded"} · batch ${measurement.profile?.batch_size ?? "?"} · imported ${date(measurement.created_at)}`));
      card.append(node("p", "experiments-caption", measurement.declaration === "simulation"
        ? "Declared simulation — not measured hardware performance."
        : "Declared external execution — execution and hardware are not independently verified."));
      const summary = measurement.summary || {};
      const mismatches = Array.isArray(summary.mismatched_samples) ? summary.mismatched_samples.length : null;
      card.append(node("p", `experiments-parity${summary.parity_passed ? "" : " failed"}`,
        summary.parity_passed
          ? "Exact parity passed against the saved reference."
          : `Exact parity failed · ${number(mismatches)} mismatched sample${mismatches === 1 ? "" : "s"}. Timing does not establish equivalent quality.`));
      const table = node("table"), head = node("thead"), heading = node("tr"), body = node("tbody");
      table.append(node("caption", "sr-only", `Imported timing for ${measurement.name || measurement.id}`));
      for (const label of ["Runner stage", "Minimum", "Median", "Maximum"]) {
        const cell = node("th", "", label);
        cell.scope = "col";
        heading.append(cell);
      }
      head.append(heading);
      for (const [label, stats] of [
        ["Preprocessing", summary.timing_ms?.preprocess_ms],
        ["Inference", summary.timing_ms?.inference_ms],
        ["Postprocessing", summary.timing_ms?.postprocess_ms],
        ["Total (excludes decode)", summary.timing_ms?.total_ms],
        ["Image decode (separate)", summary.decode_ms],
      ]) {
        const row = node("tr"), title = node("th", "", label);
        title.scope = "row";
        row.append(title);
        for (const key of ["min", "median", "max"]) row.append(node("td", "", milliseconds(stats?.[key])));
        body.append(row);
      }
      table.append(head, body);
      const wrap = node("div", "experiments-table-wrap");
      wrap.append(table);
      card.append(wrap);
      card.append(node("p", "experiments-caption",
        `${number(summary.frames)} reference images × ${number(summary.repeats)} repeats · ${number(summary.sample_count)} samples. Model load ${milliseconds(summary.load_ms)}; warm-up ${milliseconds(summary.warmup_ms)} (excluded from repeated samples).`));
      card.append(node("p", "experiments-caption",
        `Reference evaluation: ${measurement.source?.evaluation_id || "Not recorded"} · reference device: ${measurement.source?.reference_device || "Not recorded"}. Exact parity covers the packaged reference images only.`));
      field("targets").append(card);
    }
    for (const limitation of deployment?.limitations || [])
      field("target-limitations").append(node("p", "field-hint", limitation));
  }

  function overlay(svg, box, caption, color, dashed, width, height) {
    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite))
      return;
    const [x1, y1, x2, y2] = box;
    const rect = document.createElementNS(svgNS, "rect");
    for (const [key, value] of Object.entries({
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1,
      fill: "none",
      stroke: color,
      "stroke-width": 2,
      "stroke-dasharray": dashed ? "6 4" : "none",
      "vector-effect": "non-scaling-stroke",
    }))
      rect.setAttribute(key, value);
    const title = document.createElementNS(svgNS, "title");
    title.textContent = caption;
    rect.append(title);
    const text = document.createElementNS(svgNS, "text");
    const font = width / 32;
    const padding = Math.min(width, height) * 0.006;
    for (const [key, value] of Object.entries({
      x: Math.max(
        padding,
        Math.min(x1 + padding, width - caption.length * font * 0.55),
      ),
      y: Math.max(
        font + padding,
        Math.min(height - padding, dashed ? y2 - padding : y1 - padding),
      ),
      fill: color,
      stroke: "#152422",
      "stroke-width": font / 5,
      "paint-order": "stroke",
      "font-size": font,
      "font-weight": 600,
    }))
      text.setAttribute(key, value);
    text.textContent = caption;
    svg.append(rect, text);
  }

  function renderExamples(record, lanes) {
    const taxonomy = reportTaxonomy(record.snapshot);
    const examples = record.snapshot.examples || [];
    const container = field("examples");
    container.replaceChildren();
    field("example-count").textContent = `${examples.length}`;
    field("no-examples").hidden = Boolean(examples.length);
    for (const [index, example] of examples.entries()) {
      const article = node("article", "experiments-example");
      const heading = node("div", "experiments-example-heading");
      heading.append(
        node("span", "experiments-example-number", `${index + 1}`),
        node("h4", "", example.source?.filename || example.frame_id),
        node(
          "span",
          "experiments-caption",
          example.source?.timestamp_seconds == null
            ? example.scene_group
            : `${timestamp(example.source.timestamp_seconds)} · ${example.scene_group}`,
        ),
      );
      article.append(heading);
      const panels = node("div", "experiments-example-panels");
      const image = record.images.find(
        (item) => item.frame_id === example.frame_id,
      );
      for (const [laneIndex, lane] of lanes.entries()) {
        const saved = example.lanes.find((item) => item.run_id === lane.id);
        const panel = node("div", "experiments-example-panel");
        panel.append(
          node(
            "p",
            "experiments-example-lane",
            `${lanes.length === 1 ? "Evaluated pipeline" : laneIndex ? "Candidate" : "Baseline"} · ${mode(lane.variant)}`,
          ),
        );
        const visual = node("div", "experiments-visual");
        visual.style.aspectRatio = `${example.width} / ${example.height}`;
        const img = node("img");
        img.alt = `${example.source?.filename || "Selected example"} with reviewed labels and ${laneTitle(lane)} predictions`;
        img.loading = "lazy";
        const failed = () => {
          visual.classList.add("unavailable");
          if (!visual.querySelector(".experiments-image-error"))
            visual.append(
              node(
                "span",
                "experiments-image-error",
                "Example image unavailable. Saved metrics remain unchanged.",
              ),
            );
        };
        img.addEventListener("error", failed, { once: true });
        try {
          img.src = localURL(image?.url);
        } catch {
          failed();
        }
        const svg = document.createElementNS(svgNS, "svg");
        svg.setAttribute("viewBox", `0 0 ${example.width} ${example.height}`);
        svg.setAttribute("aria-hidden", "true");
        for (const label of example.ground_truth || [])
          overlay(
            svg,
            label.box,
            `GT ${datasetTools.className(taxonomy, label.label)}`,
            "#ffffff",
            true,
            example.width,
            example.height,
          );
        const threshold =
          record.snapshot.evaluation.config.confidence_threshold;
        for (const detection of datasetTools.displayedDetections(taxonomy, saved?.detections, threshold)) {
          overlay(
            svg,
            detection.box,
            `${datasetTools.className(taxonomy, detection.label)} ${ratio(detection.score)}`,
            taxonomyTools.classColor(taxonomy, detection.label),
            false,
            example.width,
            example.height,
          );
        }
        visual.append(img, svg);
        panel.append(visual);
        const counts = aggregateCounts(record.snapshot, example.counts)?.runs?.[lane.id] || saved?.errors;
        panel.append(
          node(
            "p",
            "experiments-example-counts",
            counts
              ? `${number(counts.tp)} matched · ${number(counts.fp)} false positives · ${number(counts.fn)} missed labels`
              : "Saved per-frame error counts are unavailable.",
          ),
        );
        panels.append(panel);
      }
      article.append(panels);
      const credit = example.source?.attribution;
      if (credit)
        article.append(
          node(
            "p",
            "experiments-caption experiments-image-credit",
            [credit.attribution, credit.license_name, credit.source_url]
              .filter(Boolean)
              .join(" · "),
          ),
        );
      container.append(article);
    }
  }

  function renderReport() {
    const record = library.detail,
      snapshot = record.snapshot;
    const lanes = orderedLanes(snapshot);
    field("detail").hidden = false;
    field("report-title").textContent = record.title;
    field("report-date").textContent = `Saved ${date(record.created_at)}`;
    field("report-split").textContent = split(snapshot.evaluation.split);
    field("report-context").textContent =
      `${snapshot.dataset.name} / ${snapshot.evaluation.name}`;
    field("revision").textContent = `Notes revision ${record.revision}`;
    field("objective").textContent =
      record.objective ||
      "No objective recorded yet. Use Edit notes to add the question behind this experiment.";
    field("conclusion").textContent =
      record.conclusion ||
      "No conclusion recorded yet. Add your interpretation after reviewing the evidence below.";
    renderLineage(snapshot, lanes);
    renderMetrics(snapshot, lanes);
    renderScenes(snapshot, lanes);
    renderTargets(snapshot, lanes);
    renderExamples(record, lanes);
    field("reference-note").textContent =
      "Reference decisions below are a historical snapshot. Creating or editing this report does not promote a model or change today's reference.";
    field("provenance").textContent = JSON.stringify(
      {
        report_id: record.id,
        snapshot_sha256: record.snapshot_sha256,
        captured_at: snapshot.captured_at,
        evaluation: snapshot.evaluation,
        dataset: snapshot.dataset,
        lanes: snapshot.lanes,
        error_analysis: snapshot.error_analysis,
        insights: snapshot.insights,
        deployments: snapshot.deployments,
        reference_decisions: snapshot.reference_decisions,
      },
      null,
      2,
    );
    field("export-images").checked = false;
    resetExport();
  }

  function renderPicker() {
    const query = field("example-filter").value.trim().toLocaleLowerCase();
    const change = field("example-change").value;
    const scene = field("example-scene").value;
    const time = field("example-time").value;
    const candidates = (compose.preview?.available_examples || []).filter(
      (item) => {
        const hasTime = Number.isFinite(item.timestamp_seconds);
        const counts = aggregateCounts(compose.preview.snapshot, item.counts);
        return `${item.source_filename} ${item.scene_group} ${hasTime ? timestamp(item.timestamp_seconds) : ""}`
          .toLocaleLowerCase()
          .includes(query) &&
          (!scene || item.scene_group === scene) &&
          (!time || (time === "timestamped" ? hasTime : !hasTime)) &&
          (!change || (change === "negative" ? counts?.ground_truth_count === 0 :
            compose.preview.snapshot.insights?.frame_changes?.[item.frame_id] === change));
      },
    );
    const focusedId =
      document.activeElement?.closest("[data-frame-id]")?.dataset.frameId;
    field("example-picker").replaceChildren();
    for (const item of candidates.slice(0, compose.limit)) {
      const label = node("label", "experiments-picker-card");
      label.dataset.frameId = item.frame_id;
      const input = node("input");
      input.type = "checkbox";
      input.checked = compose.selected.has(item.frame_id);
      input.setAttribute(
        "aria-label",
        `Include ${item.source_filename} ${item.timestamp_seconds == null ? item.frame_id : timestamp(item.timestamp_seconds)}`,
      );
      input.addEventListener("change", () => {
        if (input.checked) compose.selected.add(item.frame_id);
        else compose.selected.delete(item.frame_id);
        compose.dirty = true;
        label.classList.toggle("selected", input.checked);
        updateCompose();
      });
      const img = node("img");
      img.alt = "";
      img.loading = "lazy";
      try {
        img.src = localURL(item.image_url);
      } catch {
        /* Selection remains possible by frame identity. */
      }
      label.classList.toggle("selected", input.checked);
      label.append(
        img,
        input,
        node("strong", "", item.source_filename || item.frame_id),
      );
      label.append(
        node(
          "span",
          "experiments-caption",
          `${item.timestamp_seconds == null ? "Timestamp unavailable" : `${timestamp(item.timestamp_seconds)} (approx.)`} · ${item.scene_group}`,
        ),
      );
      const frameChange = compose.preview.snapshot.insights?.frame_changes?.[item.frame_id];
      if (frameChange) label.append(node("span", `experiments-change experiments-change-${frameChange}`, changeLabel(frameChange)));
      const suggested = compose.preview.snapshot.insights?.suggested_examples?.find((entry) => entry.frame_id === item.frame_id);
      if (suggested) label.append(node("span", "experiments-caption", `Suggested: ${suggested.reason}`));
      const counts = orderedLanes(compose.preview.snapshot)
        .map((lane) => aggregateCounts(compose.preview.snapshot, item.counts)?.runs?.[lane.id])
        .filter(Boolean);
      label.append(
        node(
          "span",
          "experiments-picker-counts",
          counts
            .map(
              (entry, index) =>
                `${counts.length > 1 ? (index ? "C" : "B") : ""} ${entry.fp} FP / ${entry.fn} missed`,
            )
            .join(" · "),
        ),
      );
      const changes = aggregateCounts(compose.preview.snapshot, item.counts)?.changes;
      if (changes)
        label.append(
          node(
            "span",
            "experiments-picker-counts",
            `${changes.recovered} recovered · ${changes.new_misses} new misses · ${changes.fp_delta >= 0 ? "+" : ""}${changes.fp_delta} FP`,
          ),
        );
      field("example-picker").append(label);
      if (focusedId === item.frame_id) input.focus({ preventScroll: true });
    }
    field("picker-empty").hidden = Boolean(candidates.length);
    field("more-examples").hidden = candidates.length <= compose.limit;
    field("more-examples").textContent =
      `Show more images (${Math.min(compose.limit, candidates.length)} / ${candidates.length})`;
    updateCompose();
  }

  function renderMeasurementPicker() {
    const measurements = compose.preview?.available_measurements || [];
    field("measurement-picker").replaceChildren();
    field("measurement-empty").hidden = Boolean(measurements.length);
    for (const measurement of measurements) {
      const label = node("label", "experiments-measurement-choice");
      const input = node("input");
      input.type = "checkbox";
      input.checked = compose.measurements.has(measurement.id);
      input.dataset.measurementId = measurement.id;
      input.setAttribute("aria-label", `Include ${measurementHeading(measurement)}`);
      input.addEventListener("change", () => {
        if (input.checked) compose.measurements.add(measurement.id);
        else compose.measurements.delete(measurement.id);
        compose.dirty = true;
        updateCompose();
      });
      const info = node("span");
      info.append(node("strong", "", measurementHeading(measurement)));
      info.append(node("span", "experiments-caption", measurement.declaration === "simulation"
        ? "Declared simulation — not measured hardware performance"
        : "Declared external execution — not independently verified"));
      info.append(node("span", `experiments-parity${measurement.summary?.parity_passed ? "" : " failed"}`,
        measurement.summary?.parity_passed ? "Exact parity passed" : "Exact parity failed — retained as failed evidence"));
      info.append(node("span", "experiments-caption",
        `${number(measurement.summary?.frames)} reference images · ${number(measurement.summary?.repeats)} repeats · median runner total ${milliseconds(measurement.summary?.timing_ms?.total_ms?.median)} (excludes decode) · imported ${date(measurement.created_at)}`));
      label.append(input, info);
      field("measurement-picker").append(label);
    }
  }

  function configurePickerFilters() {
    const previous = field("example-scene").value;
    const scenes = [...new Set((compose.preview?.available_examples || []).map((item) => item.scene_group))].sort();
    field("example-scene").replaceChildren(new Option("All scenes", ""));
    for (const scene of scenes) field("example-scene").append(new Option(scene, scene));
    if (scenes.includes(previous)) field("example-scene").value = previous;
    const suggestions = compose.preview?.snapshot.insights?.suggested_examples || [];
    field("suggestions-context").textContent = suggestions.length
      ? "Suggestions include saved successes and failures. Add them explicitly, then review the selection; examples do not represent the whole dataset."
      : "No example suggestions are available for this evaluation. You can choose images individually.";
  }

  async function loadPreview(preserveSelection = false) {
    const id = field("evaluation").value;
    const request = ++compose.request,
      context = compose.context;
    compose.preview = null;
    if (!preserveSelection) {
      compose.selected.clear();
      compose.measurements.clear();
      for (const name of ["example-filter", "example-change", "example-scene", "example-time"]) field(name).value = "";
    }
    compose.limit = 12;
    compose.loading = Boolean(id);
    field("compose-fields").hidden = true;
    field("preview-refresh").hidden = true;
    field("evaluation-status").textContent = id
      ? "Checking the completed evaluation…"
      : "Choose a completed evaluation to begin.";
    error("compose-error", null);
    updateCompose();
    if (!id) return;
    try {
      const preview = await api(
        `/api/evaluations/${safe(id)}/experiment-preview`,
      );
      if (!composeCurrent(context) || request !== compose.request) return;
      compose.preview = preview;
      compose.selected = new Set([...compose.selected].filter((id) => preview.available_examples.some((item) => item.frame_id === id)));
      compose.measurements = new Set([...compose.measurements].filter((id) => preview.available_measurements?.some((item) => item.id === id)));
      field("compose-fields").hidden = false;
      if (!field("title").value || field("title").value === compose.autoTitle)
        field("title").value = preview.snapshot.evaluation.name;
      compose.autoTitle = preview.snapshot.evaluation.name;
      const snapshot = preview.snapshot;
      field("evaluation-status").textContent =
        `${snapshot.dataset.name} · ${split(snapshot.evaluation.split)} · ${preview.available_examples.length} evaluated frames · ${snapshot.lanes.length} pipeline(s).${snapshot.evaluation.split === "test" ? " Reporting only: use validation evidence for model selection." : ""}`;
      configurePickerFilters();
      renderMeasurementPicker();
      renderPicker();
    } catch (failure) {
      if (composeCurrent(context) && request === compose.request) {
        field("evaluation-status").textContent =
          "Preview unavailable. Choose the evaluation again or reopen this form to retry.";
        field("preview-refresh").hidden = false;
        error("compose-error", failure);
      }
    } finally {
      if (composeCurrent(context) && request === compose.request) {
        compose.loading = false;
        updateCompose();
      }
    }
  }

  async function openCompose(evaluationId = null) {
    if (compose.busy || edit.busy || field("edit-dialog").open) return;
    if (field("compose-dialog").open) return;
    compose.context++;
    compose.preview = null;
    compose.selected.clear();
    compose.measurements.clear();
    compose.dirty = false;
    compose.autoTitle = "";
    compose.loading = true;
    field("compose-form").reset();
    field("compose-fields").hidden = true;
    field("preview-refresh").hidden = true;
    field("evaluation").replaceChildren(
      new Option("Loading completed evaluations…", ""),
    );
    field("evaluation-status").textContent = "Loading saved evaluations…";
    error("compose-error", null);
    field("compose-dialog").showModal();
    field("compose-dialog").scrollTop = 0;
    updateCompose();
    const context = compose.context;
    try {
      const rows = await api("/api/evaluations");
      if (!composeCurrent(context)) return;
      const complete = rows.filter((item) => item.job?.status === "succeeded");
      field("evaluation").replaceChildren(
        new Option(
          complete.length
            ? "Choose a completed evaluation…"
            : "No completed evaluations yet",
          "",
        ),
      );
      for (const item of complete)
        field("evaluation").append(
          new Option(
            `${item.name} · ${split(item.split)} · ${date(item.created_at)}`,
            item.id,
          ),
        );
      compose.loading = false;
      field("evaluation-status").textContent = complete.length
        ? "Choose the result you want to document. No computation will be rerun."
        : "Complete a quality evaluation first, then create its report here.";
      updateCompose();
      if (evaluationId && complete.some((item) => item.id === evaluationId)) {
        field("evaluation").value = evaluationId;
        await loadPreview();
      }
    } catch (failure) {
      if (composeCurrent(context)) {
        compose.loading = false;
        error("compose-error", failure);
        field("evaluation-status").textContent =
          "The evaluation list could not be loaded. Close and reopen this form to retry.";
        updateCompose();
      }
    }
  }

  async function createReport(event) {
    event.preventDefault();
    if (
      compose.busy ||
      compose.loading ||
      !compose.preview ||
      !field("compose-form").reportValidity()
    )
      return;
    const payload = {
      evaluation_id: compose.preview.snapshot.evaluation.id,
      title: field("title").value.trim(),
      objective: field("objective-input").value.trim(),
      conclusion: field("conclusion-input").value.trim(),
      example_frame_ids: [...compose.selected],
      measurement_ids: [...compose.measurements],
      expected_source_fingerprint: compose.preview.source_fingerprint,
    };
    if (!payload.title) return field("title").focus();
    const context = compose.context;
    compose.busy = true;
    error("compose-error", null);
    updateCompose();
    try {
      const record = await api("/api/experiments", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (!composeCurrent(context)) return;
      compose.dirty = false;
      compose.busy = false;
      field("compose-dialog").close();
      await refreshList(record.id);
      if (library.activeId === record.id)
        field("report-title").focus({ preventScroll: true });
      notify(
        `Experiment “${record.title}” saved with frozen evaluation results.`,
      );
    } catch (failure) {
      if (composeCurrent(context)) {
        error("compose-error", failure);
        if (failure.status === 409) {
          field("preview-refresh").hidden = false;
          field("evaluation-status").textContent =
            "Saved evidence has changed. Your notes and selections are preserved. Refresh the evidence and review it before saving again.";
          compose.preview = null;
        }
      }
    } finally {
      if (context === compose.context) {
        compose.busy = false;
        updateCompose();
      }
    }
  }

  function populateEditor(record) {
    edit.id = record.id;
    edit.revision = record.revision;
    edit.dirty = false;
    edit.conflict = false;
    field("edit-name").value = record.title;
    field("edit-objective").value = record.objective;
    field("edit-conclusion").value = record.conclusion;
    field("edit-reload").hidden = true;
    error("edit-error", null);
  }

  function updateEditor() {
    for (const input of field("edit-form").querySelectorAll(
      "input, textarea, button",
    ))
      input.disabled = edit.busy;
    field("edit-save").disabled = edit.busy || edit.conflict;
    field("edit-save").textContent = edit.busy ? "Saving notes…" : "Save notes";
  }

  async function saveNotes(event) {
    event.preventDefault();
    if (edit.busy || edit.conflict || !field("edit-form").reportValidity())
      return;
    const payload = {
      expected_revision: edit.revision,
      title: field("edit-name").value.trim(),
      objective: field("edit-objective").value.trim(),
      conclusion: field("edit-conclusion").value.trim(),
    };
    if (!payload.title) return field("edit-name").focus();
    const context = edit.context;
    edit.busy = true;
    updateEditor();
    error("edit-error", null);
    try {
      const record = await api(`/api/experiments/${safe(edit.id)}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
      });
      if (!editCurrent(context)) return;
      edit.dirty = false;
      edit.busy = false;
      field("edit-dialog").close();
      await refreshList(record.id);
      notify(
        "Experiment notes saved. Frozen results and examples are unchanged.",
      );
    } catch (failure) {
      if (!editCurrent(context)) return;
      if (failure.status === 409) {
        edit.conflict = true;
        field("edit-reload").hidden = false;
        error(
          "edit-error",
          "These notes changed in another view. Your draft is preserved here. Copy anything you want to keep, then reload the latest notes before saving.",
        );
      } else error("edit-error", failure);
    } finally {
      if (context === edit.context) {
        edit.busy = false;
        updateEditor();
      }
    }
  }

  async function exportReport() {
    const detail = library.detail;
    if (!detail || library.exportController) return;
    const request = ++library.exportRequest;
    const controller = new AbortController();
    library.exportController = controller;
    const current = () =>
      request === library.exportRequest &&
      library.visible &&
      library.detail?.id === detail.id &&
      library.detail?.revision === detail.revision;
    const include = field("export-images").checked;
    field("export").disabled = true;
    field("export-images").disabled = true;
    field("export-cancel").hidden = false;
    field("export").textContent = "Preparing report…";
    field("export-status").textContent = include
      ? "Embedding the selected example images locally…"
      : "Preparing metrics and notes without example images…";
    error("export-error", null);
    try {
      const response = await fetch(
        projectURL(`/api/experiments/${safe(detail.id)}/export?include_images=${include}&expected_revision=${detail.revision}`),
        {
          signal: controller.signal,
          mode: "same-origin",
          redirect: "error",
        },
      );
      if (!current()) return;
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(
          body?.detail || `Report export failed (${response.status}).`,
        );
      }
      if (
        response.headers.get("content-type")?.split(";")[0].trim() !==
        "text/html"
      )
        throw new Error(
          "The server did not return an HTML report. Refresh this report before retrying.",
        );
      const blob = await response.blob();
      if (!current()) return;
      if (!blob.size || blob.size > 16 * 1024 * 1024)
        throw new Error("The report is empty or exceeds the download limit.");
      const url = URL.createObjectURL(blob);
      library.urls.set(
        url,
        setTimeout(() => releaseURL(url), 60_000),
      );
      const link = node("a");
      link.href = url;
      link.download = `iris-experiment-${detail.id}-r${detail.revision}.html`;
      document.body.append(link);
      link.click();
      link.remove();
      field("export-status").textContent =
        `Download started · revision ${detail.revision} · ${include ? "selected images included" : "no images included"}.`;
    } catch (failure) {
      if (current() && failure.name !== "AbortError") {
        field("export-status").textContent = "";
        error(
          "export-error",
          failure instanceof TypeError
            ? "Cannot reach IRIS. Try the download again when the local server is available."
            : failure,
        );
      }
    } finally {
      if (current()) {
        library.exportController = null;
        field("export").disabled = false;
        field("export").textContent = "Download report ↓";
        field("export-images").disabled = !detail.images.length;
        field("export-cancel").hidden = true;
      }
    }
  }

  function mayClose(which) {
    const state = which === "compose" ? compose : edit;
    if (state.busy) {
      error(
        `${which}-error`,
        "Wait for the save request to finish before closing.",
      );
      return false;
    }
    return (
      !state.dirty ||
      window.confirm("Discard the unsaved report notes and selections?")
    );
  }

  for (const name of ["new", "empty-create"])
    field(name).addEventListener("click", () => openCompose());
  field("refresh").addEventListener("click", () => refreshList());
  field("search").addEventListener("input", renderList);
  field("evaluation").addEventListener("change", () => loadPreview());
  field("preview-refresh").addEventListener("click", () => loadPreview(true));
  field("example-filter").addEventListener("input", () => {
    compose.limit = 12;
    renderPicker();
  });
  for (const name of ["example-change", "example-scene", "example-time"])
    field(name).addEventListener("change", () => {
      compose.limit = 12;
      renderPicker();
    });
  field("suggest-examples").addEventListener("click", () => {
    if (compose.busy || compose.loading || !compose.preview) return;
    let added = 0;
    for (const suggestion of compose.preview.snapshot.insights?.suggested_examples || []) {
      if (compose.selected.size >= 6) break;
      if (!compose.selected.has(suggestion.frame_id) &&
          compose.preview.available_examples.some((item) => item.frame_id === suggestion.frame_id)) {
        compose.selected.add(suggestion.frame_id);
        added++;
      }
    }
    if (added) compose.dirty = true;
    field("suggestions-context").textContent =
      `${added} suggested image${added === 1 ? "" : "s"} added. Existing selections were kept. Review successes and failures before saving; active filters may hide selected images.`;
    renderPicker();
  });
  field("clear-examples").addEventListener("click", () => {
    if (compose.busy || compose.loading) return;
    compose.selected.clear();
    compose.dirty = true;
    renderPicker();
    field("suggestions-context").textContent = "Example selection cleared. Nothing is selected automatically.";
  });
  field("more-examples").addEventListener("click", () => {
    compose.limit += 12;
    renderPicker();
  });
  field("compose-form").addEventListener("submit", createReport);
  for (const name of ["title", "objective-input", "conclusion-input"])
    field(name).addEventListener("input", () => {
      compose.dirty = true;
    });
  for (const which of ["compose", "edit"]) {
    field(`${which}-close`).addEventListener("click", () => {
      if (mayClose(which)) field(`${which}-dialog`).close();
    });
    field(`${which}-dialog`).addEventListener("cancel", (event) => {
      if (!mayClose(which)) event.preventDefault();
    });
    field(`${which}-dialog`).addEventListener("close", () => {
      // A close event may arrive after the same dialog has already reopened.
      if (field(`${which}-dialog`).open) return;
      const state = which === "compose" ? compose : edit;
      state.context++;
      state.dirty = false;
      if (which === "compose") {
        compose.request++;
        compose.loading = false;
      }
    });
  }
  field("edit").addEventListener("click", () => {
    if (!library.detail) return;
    edit.context++;
    populateEditor(library.detail);
    updateEditor();
    field("edit-dialog").showModal();
    field("edit-name").focus();
  });
  field("edit-form").addEventListener("input", () => {
    edit.dirty = true;
  });
  field("edit-form").addEventListener("submit", saveNotes);
  field("edit-reload").addEventListener("click", async () => {
    if (edit.busy || !mayClose("edit")) return;
    const context = edit.context;
    edit.busy = true;
    updateEditor();
    try {
      const record = await api(`/api/experiments/${safe(edit.id)}`);
      if (editCurrent(context)) populateEditor(record);
    } catch (failure) {
      if (editCurrent(context)) error("edit-error", failure);
    } finally {
      if (editCurrent(context)) {
        edit.busy = false;
        updateEditor();
      }
    }
  });
  field("export").addEventListener("click", exportReport);
  field("export-cancel").addEventListener("click", () =>
    resetExport("Download cancelled. No report was downloaded."),
  );
  window.addEventListener("iris:experiment-create", (event) => {
    field("compose-dialog").open || $("#workspace-experiments").click();
    if (library.visible) openCompose(event.detail?.evaluation_id);
  });
  document.addEventListener(
    "click",
    (event) => {
      const tab = event.target.closest?.(".workspace-tab");
      if (!tab || tab.id === "workspace-experiments") return;
      for (const which of ["compose", "edit"])
        if (field(`${which}-dialog`).open) {
          if (!mayClose(which)) {
            event.preventDefault();
            event.stopImmediatePropagation();
            return;
          }
          field(`${which}-dialog`).close();
        }
    },
    true,
  );
  window.addEventListener("iris:before-session", (event) => {
    for (const which of ["compose", "edit"])
      if (field(`${which}-dialog`).open) {
        if (!mayClose(which)) {
          event.preventDefault();
          return;
        }
        field(`${which}-dialog`).close();
      }
  });
  window.addEventListener("iris:workspace", (event) => {
    library.visible = event.detail.name === "experiments";
    if (library.visible) refreshList();
    else {
      library.listRequest++;
      library.detailRequest++;
      library.listLoading = false;
      resetExport();
    }
  });
  window.addEventListener("beforeunload", (event) => {
    if (compose.dirty || edit.dirty || compose.busy || edit.busy) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  window.addEventListener("pagehide", () => {
    resetExport();
    for (const url of library.urls.keys()) releaseURL(url);
  });
})();
