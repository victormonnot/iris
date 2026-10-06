"use strict";

window.IRISBenchmarkReport = (() => {
  const tools = window.IRISBenchmarkReportTools;
  const common = window.IRISBenchmarkTools;
  const external = window.IRISBenchmarkExternalTools;
  const taxonomy = window.IRISTaxonomyTools;
  const field = (name) => $(`#benchmark-report-${name}`);
  const safe = encodeURIComponent;
  const roleName = (role) => role === "evaluation" ? "Evaluation" : "Tuning";
  const date = (value) => value ? new Date(value).toLocaleString() : "Not recorded";
  const svgNS = "http://www.w3.org/2000/svg";

  function create({ onBusy, onOpenTrial, showRecord }) {
    const view = { id: null, status: null, comparison: null, preview: null, saved: null,
      rows: [], busy: false, blocked: false, loading: false, request: 0, generation: 0, reportRequest: 0 };
    const path = () => `/api/benchmarks/${safe(view.id)}`;
    const options = () => ({ role: field("role").value, title: field("title").value.trim(),
      objective: field("objective").value.trim(), conclusion: field("conclusion").value.trim(), evidence_kind: field("evidence").value });
    function error(failure) { field("error").textContent = failure?.message || failure || ""; field("error").hidden = !failure; }
    function invalidate() {
      view.generation++; view.preview = null; field("preview-result").hidden = true;
      field("preview-comparison").replaceChildren(); update();
    }
    function update() {
      const blocked = view.busy || view.blocked || !view.id;
      for (const name of ["role", "refresh", "title", "objective", "conclusion", "evidence", "history"]) field(name).disabled = blocked;
      field("refresh").disabled ||= view.loading;
      field("preview").disabled = blocked || view.loading || !view.comparison || !options().title || !options().objective;
      const counts = view.preview?.snapshot.comparison.coverage;
      const terminal = view.preview?.snapshot.comparison.configs.some((config) => config.trials.some((trial) => ["succeeded", "failed", "cancelled", "interrupted"].includes(trial.status)));
      field("save").disabled = blocked || !view.preview || view.preview.key !== common.canonical(options()) || !terminal || counts?.active_trial_count > 0;
      field("preview").textContent = view.busy === "preview" ? "Preparing report…" : "Preview immutable report";
      field("save").textContent = view.busy === "save" ? "Saving report…" : "Save immutable report";
      field("history").disabled ||= !view.rows.length;
      field("status").textContent = view.loading ? "Reading saved benchmark evidence…" : view.comparison
        ? `${roleName(view.comparison.role)} only · ${tools.number(view.comparison.coverage.config_count)} configurations · ${tools.number(view.comparison.coverage.trial_count)} trials · ${tools.number(view.comparison.coverage.complete_trial_count)} complete measurements` : "Choose a saved benchmark to compare its evidence.";
    }
    function table(headers, rows) {
      const wrap = node("div", "benchmark-table-wrap");
      wrap.tabIndex = 0; wrap.setAttribute("role", "region");
      wrap.setAttribute("aria-label", `${headers[0]} comparison, scroll horizontally for all measurements`);
      const result = node("table", "benchmark-results-table"), head = node("thead", ""), tr = node("tr", "");
      for (const text of headers) { const th = node("th", "", text); th.scope = "col"; tr.append(th); }
      head.append(tr); result.append(head); const body = node("tbody", "");
      for (const cells of rows) { const row = node("tr", ""); for (const text of cells) row.append(node("td", "", text)); body.append(row); }
      result.append(body); wrap.append(result); return wrap;
    }
    function latency(value) {
      if (!value) return "Unmeasured";
      return `${common.duration(value.total_ms)} · ${tools.number(value.measured_count)}/${tools.number(value.planned_count)} images timed${value.model_load_ms != null ? ` · model load ${common.duration(value.model_load_ms)} ${value.model_load_included ? "included" : "separate"}` : ""}`;
    }
    function corrections(value) {
      if (!value) return "Unmeasured";
      return `${tools.number(value.reviewed_count)}/${tools.number(value.planned_count)} reviews complete · ${common.duration(value.recorded_review_ms)} recorded · ${tools.number(value.fully_timed_count)}/${tools.number(value.timed_count)} timed records without known interruption${value.complete ? "" : " · incomplete review coverage"}`;
    }
    function trialRows(config) {
      return config.trials.map((trial, index) => {
        const quality = tools.quality(trial.quality);
        return [`${index + 1} · ${trial.id}\n${date(trial.created_at)}`, `${trial.status} · ${tools.number(trial.coverage.ready)}/${tools.number(trial.coverage.planned)} ready\n${tools.number(trial.coverage.failed)} failed · ${tools.number(trial.coverage.missing)} missing${trial.error ? `\n${trial.error}` : ""}`,
          quality.errors, quality.conflicts, quality.precisionRecall, quality.iou,
          `${latency(trial.latency)}\nScope: ${trial.latency?.includes || "Not recorded"}\n${trial.latency?.note || ""}`,
          config.approach === "recorded_proposals" ? "No provider call during import. Historical source receipts are retained with image evidence; missing amounts remain unknown."
            : trial.cost?.external ? `${external.costPresentation(trial.cost)}\n${config.approach === "combined" ? "External API usage only; local SAM compute cost is unmeasured.\n" : ""}${trial.cost.note || ""}` : "Local monetary cost unmeasured",
          `${corrections(trial.corrections)}\n${trial.corrections?.note || ""}`];
      });
    }
    function svgElement(tag, attributes) {
      const item = document.createElementNS(svgNS, tag);
      for (const [name, value] of Object.entries(attributes || {})) item.setAttribute(name, value);
      return item;
    }
    function imagePanel(reference, boxes, classes, label) {
      const svg = svgElement("svg", { viewBox: `0 0 ${reference.width} ${reference.height}`, role: "img", "aria-label": label });
      const image = svgElement("image", { href: projectURL(reference.image_url), width: reference.width, height: reference.height });
      const wrapper = node("div", "benchmark-report-image");
      image.addEventListener("error", () => { svg.hidden = true; wrapper.replaceChildren(node("p", "inline-error", "The frozen source image could not be loaded. Saved coordinates remain inspectable.")); });
      svg.append(image);
      const colors = ["#00856a", "#9b5be0", "#d64c19", "#2373bd", "#ab2577", "#9c7200"];
      for (const box of boxes || []) {
        if (!Array.isArray(box.box) || box.box.length !== 4 || !box.box.every(Number.isFinite)) continue;
        const [x1, y1, x2, y2] = box.box, index = classes.findIndex((item) => item.id === box.label);
        const color = colors[(Math.max(0, index)) % colors.length], group = svgElement("g");
        const title = svgElement("title"); title.textContent = `${classes[index]?.name || box.label} · ${box.box.join(", ")}`;
        const rect = svgElement("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, fill: "none", stroke: color, "stroke-width": 2, "vector-effect": "non-scaling-stroke" });
        const fontSize = Math.max(2, reference.width / 22);
        const text = svgElement("text", { x: x1 + fontSize / 6, y: Math.max(fontSize, y1 - fontSize / 5), fill: color, "font-size": fontSize, "paint-order": "stroke", stroke: "white", "stroke-width": fontSize / 10 });
        text.textContent = classes[index]?.name || box.label; group.append(title, rect, text); svg.append(group);
      }
      wrapper.append(svg); return wrapper;
    }
    function frameDetails(container, comparison, selection) {
      container.replaceChildren(); const reference = tools.frameFor(comparison, selection);
      if (!reference) return;
      const classes = comparison.benchmark.taxonomy.classes;
      const referenceCard = node("article", "benchmark-report-frame");
      referenceCard.append(node("h4", "", "Independent reference"), imagePanel(reference, reference.boxes, classes, "Frozen reference boxes"),
        node("p", "field-hint", `${reference.boxes.length} reviewed objects · ${reference.width} × ${reference.height} pixels · ${reference.scene_group}`)); container.append(referenceCard);
      for (const config of comparison.configs) {
        const card = node("article", "benchmark-report-frame"), select = node("select", ""), content = node("div", "");
        card.append(node("h4", "", `${tools.approach(config.approach)} · ${config.name}`));
        select.setAttribute("aria-label", `Trial to inspect for ${config.name}`);
        for (const [index, trial] of config.trials.entries()) select.append(new Option(`Trial ${index + 1} · ${trial.status} · ${date(trial.created_at)}`, trial.id));
        if (!config.trials.length) { select.append(new Option("No trial for this role", "")); select.disabled = true; }
        function render() {
          content.replaceChildren(); const { trial, frame } = tools.trialFrame(config, select.value, reference.frame_id);
          if (!frame) { content.append(node("p", "field-hint", "No saved output for this image. Quality is unavailable.")); return; }
          content.append(node("p", "field-hint", `Output: ${frame.state}${frame.error ? ` · ${frame.error}` : ""}`));
          if (Array.isArray(frame.proposals)) content.append(imagePanel(reference, frame.proposals, classes, `${config.name} saved candidate boxes`), node("p", "field-hint", `${frame.proposals.length} saved proposals · ${common.duration(frame.elapsed_ms)} processing`));
          else content.append(node("p", "field-hint", "Proposals unavailable. Missing or failed output is not an empty prediction."));
          const score = frame.quality;
          if (score) content.append(node("p", "field-hint", `${tools.number(score.fp)} extra · ${tools.number(score.fn)} missed · ${tools.number(score.class_conflicts)} class conflicts · ${tools.number(score.tp)} matched objects`));
          const correction = frame.correction;
          content.append(node("p", "field-hint", correction ? `Human correction: ${correction.status} · revision ${correction.revision} · ${correction.reviewer || "reviewer unrecorded"} · ${common.duration(correction.timing?.elapsed_ms)} recorded` : "No saved human correction"));
          const inspect = node("button", "text-button", "Inspect saved image evidence"); inspect.type = "button";
          inspect.addEventListener("click", () => showRecord("Comparison image evidence", { reference, config_id: config.id, trial_id: trial.id, frame })); content.append(inspect);
          const open = node("button", "text-button", "Open saved trial"); open.type = "button";
          open.addEventListener("click", () => onOpenTrial(trial.id)); content.append(open);
        }
        select.addEventListener("change", render); card.append(select, content); render(); container.append(card);
      }
    }
    function renderComparison(container, comparison) {
      container.replaceChildren();
      container.append(node("p", "benchmark-summary", `${roleName(comparison.role)} · ${comparison.reference.image_count} frozen images · ${comparison.reference.scene_count} scenes · ${comparison.reference.object_count} reference objects`));
      const warnings = node("ul", "field-hint"); for (const warning of comparison.warnings || []) warnings.append(node("li", "", warning)); container.append(warnings);
      if (!comparison.configs.length) container.append(node("p", "field-hint", "No frozen candidate configuration. No measurement is available."));
      for (const config of comparison.configs) {
        const article = node("article", "benchmark-report-config");
        article.append(node("h4", "", `${tools.approach(config.approach)} · ${config.name}`), node("p", "field-hint", `${config.model_id} · ${config.trials.length} saved trials for this role`),
          node("p", "field-hint", tools.repeatability(config.repeatability)));
        if (config.approach === "recorded_proposals") article.append(node("p", "field-hint", "Imported saved proposals. Processing time covers local evidence validation only, not the source model or provider. Reimporting identical evidence does not measure model repeatability."));
        if (config.repeatability.mixed_runtime_identity || config.repeatability.mixed_returned_models) article.append(node("p", "field-hint", "Runtime or returned-model identities differ between trials. Inspect the evidence before attributing variation to the configuration."));
        if (config.repeatability.note) article.append(node("p", "field-hint", config.repeatability.note));
        const metrics = config.repeatability.metrics;
        if (config.repeatability.measured && metrics) article.append(node("p", "field-hint", `Across complete trials — precision ${tools.metricRange(metrics.precision, tools.percentage)}; recall ${tools.metricRange(metrics.recall, tools.percentage)}; matched IoU ${tools.metricRange(metrics.matched_iou_mean, tools.percentage)}.`));
        if (config.trials.length) article.append(table(["Trial", "Status / image coverage", "Extra / missed", "Class conflicts", "Precision / recall", "Matched IoU", "Processing time and scope", "Usage cost estimate", "Human correction"], trialRows(config)));
        else article.append(node("p", "field-hint", "Not tested in this role · quality, latency, cost and correction time unavailable."));
        const details = node("details", "benchmark-report-details"); details.append(node("summary", "", "Class results and saved settings"));
        for (const trial of config.trials) {
          if (!trial.quality.complete) continue;
          details.append(node("p", "field-hint", `Trial ${trial.id} · ${date(trial.created_at)}`));
          const perClass = trial.quality.metrics?.per_class || {};
          details.append(table(["Class", "Matched", "Extra", "Missed"], Object.entries(perClass).map(([id, metric]) => [taxonomy.className(comparison.benchmark.taxonomy, id), tools.number(metric.tp), tools.number(metric.fp), tools.number(metric.fn)])));
        }
        const inspect = node("button", "text-button", "Inspect configuration, identities and repeats"); inspect.type = "button";
        inspect.addEventListener("click", () => showRecord("Frozen comparison configuration and trial evidence", config)); details.append(inspect); article.append(details); container.append(article);
      }
      if (comparison.reference.frames.length) {
        const section = node("section", "benchmark-report-images"), label = node("label", "", "Compare the same frozen image"), selector = node("select", ""), grid = node("div", "benchmark-report-image-grid");
        selector.setAttribute("aria-label", "Frozen image for comparison");
        for (const frame of comparison.reference.frames) selector.append(new Option(`${frame.source_filename || frame.frame_id} · ${frame.scene_group}`, frame.frame_id));
        label.append(selector); section.append(node("h4", "", "Image evidence"), label,
          node("p", "field-hint", "Every configuration uses the same reference image. Choose any saved repetition; the first listed trial is displayed initially, with no best-run selection. Viewing the reference can influence a later human correction; reviewer independence is not verified."), grid);
        selector.addEventListener("change", () => frameDetails(grid, comparison, selector.value)); frameDetails(grid, comparison, selector.value); container.append(section);
      }
    }
    async function load() {
      if (!view.id || view.busy) return;
      const id = view.id, role = field("role").value, request = ++view.request;
      view.loading = true; update(); error(null);
      try {
        const [comparison, rows] = await Promise.all([api(`${path()}/comparison?role=${safe(role)}`), api(`${path()}/reports`)]);
        if (request !== view.request || id !== view.id || role !== field("role").value) return;
        if (!tools.validComparison(comparison, id, role)) throw new Error("The comparison does not belong to this benchmark and role.");
        if (common.canonical(view.comparison) !== common.canonical(comparison)) invalidate();
        view.comparison = comparison; view.rows = rows.filter((row) => row.benchmark_id === id);
        renderComparison(field("comparison"), comparison); renderHistory();
      } catch (failure) { if (request === view.request) error(failure); }
      finally { if (request === view.request) { view.loading = false; update(); } }
    }
    function renderHistory() {
      field("history").replaceChildren(new Option("Choose an immutable report", ""));
      for (const row of view.rows) field("history").append(new Option(`${row.title} · ${roleName(row.role)} · ${date(row.created_at)}`, row.id));
      field("history").value = view.rows.some((row) => row.id === view.saved?.id) ? view.saved.id : "";
    }
    function renderSaved() {
      const report = view.saved; field("saved").hidden = !report;
      field("saved-comparison").replaceChildren();
      if (!report) {
        for (const name of ["saved-title", "saved-context", "saved-objective", "saved-conclusion", "saved-hash"]) field(name).textContent = "";
        for (const format of ["json", "html"]) field(`export-${format}`).removeAttribute("href");
        return;
      }
      field("saved-title").textContent = report.snapshot.title;
      field("saved-context").textContent = `Immutable report · ${roleName(report.snapshot.comparison.role)} · ${date(report.created_at)} · ${tools.evidence(report.snapshot.evidence_kind)}`;
      field("saved-objective").textContent = report.snapshot.objective;
      field("saved-conclusion").textContent = report.snapshot.conclusion || "No conclusion recorded.";
      field("saved-hash").textContent = `Snapshot SHA-256: ${report.snapshot_sha256}`;
      for (const format of ["json", "html"]) field(`export-${format}`).href = projectURL(`/api/benchmark-reports/${safe(report.id)}/export.${format}`);
      renderComparison(field("saved-comparison"), report.snapshot.comparison);
    }
    async function openReport(id) {
      const benchmarkId = view.id, request = ++view.reportRequest;
      view.saved = null; renderSaved(); error(null);
      if (!id) return;
      try {
        const report = await api(`/api/benchmark-reports/${safe(id)}`);
        if (request !== view.reportRequest || benchmarkId !== view.id) return;
        if (!tools.validReport(report, benchmarkId)) throw new Error("This report does not belong to the selected benchmark.");
        view.saved = report; renderSaved();
      } catch (failure) { if (request === view.reportRequest) error(failure); }
    }
    async function preview() {
      if (field("preview").disabled) return;
      invalidate(); const generation = view.generation, id = view.id, data = options(), key = common.canonical(data);
      view.busy = "preview"; onBusy(); update(); error(null);
      try {
        const result = await api(`${path()}/reports/preview`, { method: "POST", body: JSON.stringify(data) });
        if (generation !== view.generation || id !== view.id || key !== common.canonical(options())) return;
        if (result.benchmark_id !== id || !result.fingerprint || !tools.validComparison(result.snapshot?.comparison, id, data.role)) throw new Error("Report preview does not match the selected reference and role.");
        view.preview = { ...result, key }; field("preview-result").hidden = false;
        field("preview-context").textContent = `${result.snapshot.title} · ${roleName(data.role)} · ${tools.evidence(result.snapshot.evidence_kind)}`;
        field("preview-objective").textContent = result.snapshot.objective;
        field("preview-conclusion").textContent = result.snapshot.conclusion || "No conclusion recorded.";
        field("preview-note").textContent = "All configurations and all saved trials in this role are included. Saving requires at least one finished trial and no active trial in this role. Later results and correction edits require a new preview; a saved report never changes.";
        renderComparison(field("preview-comparison"), result.snapshot.comparison);
      } catch (failure) { if (generation === view.generation) error(failure); }
      finally { view.busy = false; onBusy(); update(); }
    }
    async function save() {
      if (field("save").disabled) return;
      const id = view.id, preview = view.preview;
      let saved = false;
      view.busy = "save"; onBusy(); update(); error(null);
      try {
        let report;
        try { report = await api(`${path()}/reports`, { method: "POST", body: JSON.stringify({ ...options(), expected_fingerprint: preview.fingerprint }) }); }
        catch (failure) {
          if (failure.status && failure.status < 500) throw failure;
          const rows = await api(`${path()}/reports`);
          const receipt = rows.find((row) => row.benchmark_id === id && row.snapshot_sha256 === preview.fingerprint);
          if (!receipt) throw new Error("Save outcome is uncertain. Refresh saved reports before preparing another report. No save was repeated.");
          report = await api(`/api/benchmark-reports/${safe(receipt.id)}`);
        }
        if (id !== view.id) return;
        if (!tools.validReport(report, id) || report.snapshot_sha256 !== preview.fingerprint) throw new Error("The saved report does not match the preview.");
        view.saved = report; invalidate(); renderSaved(); saved = true; notify("Immutable benchmark report saved locally.");
      } catch (failure) { invalidate(); error(failure); }
      finally { view.busy = false; onBusy(); update(); }
      if (saved) await load();
    }
    function setContext(detail) {
      const id = detail?.id || null, changed = view.id !== id;
      if (changed) {
        view.request++; view.reportRequest++; view.loading = false; view.id = id; view.status = null; view.comparison = null; view.saved = null; view.rows = [];
        field("comparison").replaceChildren(); field("title").value = detail ? `${detail.name} · comparison`.slice(0, 160) : "";
        field("objective").value = ""; field("conclusion").value = ""; field("evidence").value = "not_declared";
        invalidate(); renderSaved(); renderHistory(); error(null);
      }
      if (changed || view.status !== detail?.status) {
        field("role").value = detail?.status === "locked" ? "evaluation" : "tuning";
        view.status = detail?.status || null; invalidate();
      }
      if (id) load(); else update();
    }
    field("refresh").addEventListener("click", load);
    field("role").addEventListener("change", () => { view.comparison = null; field("comparison").replaceChildren(); invalidate(); load(); });
    for (const name of ["title", "objective", "conclusion", "evidence"]) field(name).addEventListener(name === "evidence" ? "change" : "input", invalidate);
    field("preview").addEventListener("click", preview); field("save").addEventListener("click", save);
    field("history").addEventListener("change", () => openReport(field("history").value));
    field("inspect").addEventListener("click", () => { if (view.saved) showRecord("Immutable benchmark report snapshot", view.saved); });
    for (const name of ["iris:before-workspace", "iris:before-session"]) window.addEventListener(name, (event) => {
      if (view.busy) { event.preventDefault(); notify("Wait for the report request to finish before changing context.", true); }
    });
    window.addEventListener("iris:workspace", (event) => { if (event.detail.name !== "benchmark") invalidate(); });
    window.addEventListener("iris:session", invalidate);
    window.addEventListener("beforeunload", (event) => { if (view.busy) { event.preventDefault(); event.returnValue = ""; } });
    update();
    return { setContext, busy: () => Boolean(view.busy), setBlocked: (blocked) => { view.blocked = blocked; update(); } };
  }
  return { create };
})();
