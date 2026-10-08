"use strict";

(() => {
  const tools = window.IRISTrackingQualityTools, safe = encodeURIComponent;
  const field = (name) => $(`#quality-${name}`);
  const requestedReport = new URL(window.location.href).searchParams.get("tracking_quality");
  const view = { visible: false, generation: 0, recordRequest: 0, openRequest: 0, loading: false, busy: false,
    sequence: null, comparison: null, tracking: null, references: [], history: [], record: null,
    pending: null, memory: new Map(), status: null };
  const context = () => view.comparison && { comparison_id: view.comparison.id, sequence_id: view.sequence.id };
  const reference = () => view.references.find((item) => item.id === field("reference").value);
  function error(failure) { field("error").textContent = failure?.message || failure || ""; field("error").hidden = !failure; }
  function options(select, rows, label, placeholder, selected = "") {
    select.replaceChildren(new Option(placeholder, ""));
    rows.forEach((row) => select.append(new Option(label(row), row.id)));
    if (rows.some((row) => row.id === selected)) select.value = selected;
  }
  function mapping() {
    return Object.fromEntries([...field("mapping").querySelectorAll("select")].map((select) => [select.dataset.nativeId, select.value === "__ignore__" ? null : select.value]));
  }
  function config() {
    return tools.configuration(field("reference").value, tools.nativeClasses(view.tracking), mapping(), Number(field("iou").value), view.sequence.manifest.taxonomy);
  }
  function update() {
    const blocked = view.loading || view.busy, ready = Boolean(view.comparison && view.tracking), selected = reference();
    field("refresh").disabled = !ready || blocked;
    field("reference").disabled = !ready || blocked || !view.references.length;
    field("iou").disabled = !ready || blocked;
    field("history").disabled = !ready || blocked || !view.history.length;
    field("open-reference").disabled = !view.sequence || blocked;
    field("open-reference").textContent = selected ? `Open reference revision ${selected.revision}` : view.record ? "Open report reference revision" : "Open temporal identities";
    field("show-frame").disabled = !view.record || blocked || !field("frame").value;
    field("frame").disabled = blocked || !view.record;
    field("mapping").querySelectorAll("select").forEach((select) => { select.disabled = blocked; });
    let valid = false, hint = "Choose a completed comparison above.";
    if (ready) {
      try { config(); valid = true; hint = "Ready to calculate from saved evidence. Press Calculate to create a new immutable report; no detector or tracker is run."; }
      catch (failure) { hint = failure.message; }
    }
    if (ready && view.status && view.sequence.manifest.frames.length > view.status.limits.max_frames) { valid = false; hint = `This sequence exceeds the ${view.status.limits.max_frames} frame calculation limit. Saved reports remain readable.`; }
    field("calculate").disabled = !ready || blocked || !valid;
    field("config-status").textContent = view.busy ? "Calculating from the pinned reference and saved observations…" : view.loading ? "Loading saved references and report history…" : hint;
    field("reference-summary").textContent = selected ? `${tools.referenceLabel(selected)}. ${selected.summary.dense_human_reference ? "Dense human reference." : "Sparse reference: only fully reviewed, scorable frames will be evaluated; IDF1 may be unavailable."} ${selected.payload.provenance?.origin ? "Reference began with assisted tracking proposals; review does not establish independence." : "Manual or legacy reference provenance."}` : view.references.length ? "Choose a saved revision explicitly. Later identity edits create a new revision and never update an existing quality report." : ready ? "No saved temporal reference. Open Identities, review frames and save a reference revision before calculating." : "";
    $("#tracking-quality-panel").setAttribute("aria-busy", String(blocked));
  }
  function remember() {
    if (view.comparison) view.memory.set(view.comparison.id, { reference_id: field("reference").value, report_id: view.record?.id || "" });
  }
  function clearContext() {
    remember(); view.generation++; view.recordRequest++; view.loading = false; view.busy = false;
    view.sequence = null; view.comparison = null; view.tracking = null; view.references = []; view.history = []; view.record = null;
    options(field("reference"), [], () => "", "Choose a completed comparison first");
    options(field("history"), [], () => "", "No saved report selected");
    field("mapping").replaceChildren(); field("result").hidden = true;
    field("context").textContent = "Choose a completed tracking comparison to inspect or calculate its quality reports.";
    error(null); update();
  }
  function renderMapping() {
    field("mapping").replaceChildren();
    const detector = view.tracking?.lanes[0]?.report.cache.config.detector;
    if (!detector) return;
    const ids = tools.nativeClasses(view.tracking), taxonomy = view.sequence.manifest.taxonomy;
    const suggested = tools.suggestions(detector, taxonomy, ids);
    for (const id of ids) {
      const native = detector.classes.find((item) => item.id === id), row = node("div"), label = node("label", "", `${native?.name || "Native class"} · detector ${id}`), select = node("select");
      select.id = `quality-map-${id}`; select.dataset.nativeId = id; label.htmlFor = select.id;
      select.append(new Option("Choose a taxonomy class or explicitly ignore", ""), new Option("Ignore this detector class", "__ignore__"));
      taxonomy.classes.forEach((item) => select.append(new Option(`${item.name || item.id} · ${item.id}`, item.id)));
      if (Object.hasOwn(suggested, String(id))) select.value = suggested[id];
      select.addEventListener("change", update); row.append(label, select); field("mapping").append(row);
    }
  }
  function renderHistory(selected = "") {
    options(field("history"), view.history, (record) => `Reference r${record.report.source.reference_revision} · IoU ${record.config.iou_threshold} · ${record.report.coverage.evaluated_frames}/${record.report.coverage.available_frames} frames · ${record.created_at} · ${record.id.slice(0, 8)}`, view.history.length ? "Choose a saved report" : "No saved quality reports", selected);
  }
  async function loadContext(detail) {
    if (!detail?.report || detail.comparison.sequence_id !== detail.sequence.id) return clearContext();
    clearContext(); const generation = view.generation;
    view.sequence = detail.sequence; view.comparison = detail.comparison; view.tracking = detail.report; view.loading = true;
    const remembered = view.memory.get(view.comparison.id) || {};
    const explicitOpen = view.pending?.comparison_id === view.comparison.id;
    const requested = explicitOpen ? view.pending : remembered;
    view.pending = null;
    field("context").textContent = `${view.comparison.name} · ${view.sequence.name} · metrics use the saved lane outputs independently of the replay display.`;
    renderMapping(); update();
    try {
      const [references, history] = await Promise.all([api(`/api/temporal/sequences/${safe(view.sequence.id)}/references`), api(`/api/temporal/tracking-comparisons/${safe(view.comparison.id)}/quality-reports`)]);
      if (generation !== view.generation || !view.visible) return;
      view.references = references; view.history = history.filter((record) => tools.matchesContext(record, context()));
      options(field("reference"), references, tools.referenceLabel, references.length ? "Choose an explicit saved reference revision" : "No saved temporal references", requested.reference_id);
      renderHistory(requested.report_id);
      if (requested.reference_id && !reference()) error("The requested reference revision is unavailable for this sequence. Choose another saved revision explicitly.");
      if (requested.report_id) await loadRecord(requested.report_id, generation, explicitOpen);
      else if (explicitOpen) $("#tracking-quality-panel").scrollIntoView({ block: "start" });
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.loading = false; update(); } }
  }
  async function loadRecord(id, generation = view.generation, scroll = false) {
    const request = ++view.recordRequest;
    view.record = null; field("result").hidden = true; update();
    if (!id) return;
    try {
      const record = await api(`/api/temporal/tracking-quality-reports/${safe(id)}`);
      if (generation !== view.generation || request !== view.recordRequest || !view.visible) return;
      if (!tools.matchesContext(record, context())) throw new Error("Saved quality report does not match the selected comparison and frozen sequence.");
      view.record = record; field("history").value = record.id; renderRecord(); remember();
      if (scroll) field("result").scrollIntoView({ block: "start" });
    } catch (failure) { if (generation === view.generation && request === view.recordRequest) error(failure); }
    update();
  }
  function metricTable(report) {
    const table = node("table", "quality-metrics-table"), caption = node("caption", "", "Saved metrics · each lane compared with the same pinned reference"), head = node("thead"), row = node("tr");
    row.append(node("th", "", "Metric"));
    report.lanes.forEach((lane) => { const title = node("th", "", lane.name); title.scope = "col"; row.append(title); });
    head.append(row); table.append(caption, head); const body = node("tbody");
    const metrics = [["Precision", (lane) => tools.percentage(lane.precision), (lane) => lane.precision === null ? "No predicted observations in the evaluated scope; precision is undefined." : "TP / (TP + FP)"],
      ["Recall", (lane) => tools.percentage(lane.recall), (lane) => lane.recall === null ? "No ground-truth boxes in the evaluated scope; recall is undefined." : "TP / (TP + FN)"],
      ["True positives", (lane) => lane.counts.true_positives], ["False positives", (lane) => lane.counts.false_positives], ["False negatives", (lane) => lane.counts.false_negatives],
      ["Reference boxes", (lane) => lane.counts.ground_truth], ["Confirmed observations", (lane) => lane.counts.observations],
      ["Identity switches", (lane) => lane.counts.identity_switches], ["Fragments", (lane) => lane.counts.fragments], ["Identity transfers", (lane) => lane.counts.identity_transfers], ["Class confusions", (lane) => lane.counts.class_confusions],
      ["IDF1", (lane) => lane.identity.available ? tools.percentage(lane.identity.idf1) : "Unavailable", (lane) => lane.identity.available ? "Global identity assignment over the fully dense evaluated clip." : tools.reason(lane.identity.reason)],
      ["ID true positives", (lane) => lane.identity.idtp], ["ID false positives", (lane) => lane.identity.idfp], ["ID false negatives", (lane) => lane.identity.idfn]];
    for (const [label, value, note] of metrics) {
      const tr = node("tr"), name = node("th", "", label); name.scope = "row"; tr.append(name);
      for (const lane of report.lanes) { const td = node("td", "", String(value(lane) ?? "Unavailable")); if (note) td.append(node("small", "", note(lane))); tr.append(td); }
      body.append(tr);
    }
    table.append(body); field("metrics").replaceChildren(table);
  }
  function renderRecord() {
    const record = view.record, report = record.report, coverage = report.coverage;
    field("result").hidden = false;
    field("result-title").textContent = `Saved report · reference revision ${report.source.reference_revision}`;
    field("result-scope").textContent = `Reference ${report.source.reference_id} · IoU ≥ ${report.protocol.iou_threshold} · saved ${record.created_at}. Form selections apply only to a new calculation; this report remains pinned to its saved inputs.`;
    field("coverage").textContent = `${coverage.dense ? "Dense evaluated reference" : "Sparse / partial evaluated coverage"} · ${coverage.evaluated_frames}/${coverage.available_frames} available frames evaluated · ${coverage.source_frames} source frames in clip · ${coverage.excluded_frames} available frames excluded · ${coverage.evaluated_transitions} contiguous evaluated transitions · scope: ${coverage.scope_labels.join(", ") || "none"}`;
    field("origin").hidden = !coverage.reference_origin;
    field("origin").textContent = coverage.reference_origin ? `This reference was seeded from saved tracking comparison ${coverage.reference_origin.comparison_id}, lane ${Number(coverage.reference_origin.lane_index) + 1}. Human review is recorded, but assistance does not establish an independent reference.` : "";
    const noGT = report.lanes.every((lane) => lane.counts.ground_truth === 0);
    field("empty").hidden = coverage.evaluated_frames > 0 && !noGT;
    field("empty").textContent = !coverage.evaluated_frames ? "No frames meet the complete human review and scorability rules. Detection rates and IDF1 are unavailable; inspect the excluded reasons and review a new reference revision." : "No ground-truth boxes occur in the evaluated scope. Human-reviewed empty frames can still expose false positives. Recall and identity scores are undefined.";
    const link = new URL(window.location.href); link.searchParams.set("project", state.projectId); link.searchParams.set("tracking_quality", record.id); link.searchParams.delete("temporal_sequence"); link.searchParams.delete("temporal_comparison"); link.searchParams.delete("temporal_lane"); link.searchParams.delete("tracking_comparison");
    field("permalink").href = `${link.pathname}${link.search}`;
    metricTable(report); field("diagnostics").replaceChildren();
    for (const lane of report.lanes) { const article = node("article"); article.append(node("h4", "", lane.name), node("p", "", `${lane.counts.excluded_predictions} predictions excluded · ${lane.counts.excluded_unconfirmed} unconfirmed observations excluded · ${lane.counts.excluded_unassigned} unassigned detections excluded. Scope: all available frames within the selected classes.`)); field("diagnostics").append(article); }
    field("excluded").replaceChildren();
    if (!coverage.excluded.length) field("excluded").append(node("li", "", "No available frames excluded. Source gaps, when present, remain outside the available frame list."));
    for (const excluded of coverage.excluded) field("excluded").append(node("li", "", `Source ${excluded.frame_index}: ${excluded.reasons.map((reason) => reason.replaceAll("_", " ")).join("; ")}`));
    field("limitations").replaceChildren(); for (const limitation of report.limitations) field("limitations").append(node("li", "", limitation));
    field("provenance").textContent = JSON.stringify({ report_id: record.id, report_sha256: record.report_sha256, schema: report.schema, protocol: report.protocol, source: report.source, reference_origin: coverage.reference_origin }, null, 2);
    const frames = report.lanes[0]?.frames || [];
    options(field("frame"), frames.map((frame) => ({ ...frame, id: String(frame.frame_index) })), (frame) => `Source ${frame.frame_index} · ${frame.evaluated ? "evaluated" : "excluded"}`, frames.length ? "Choose a source frame" : "No frame evidence", frames.length ? String(frames[0].frame_index) : "");
    renderFrame(); update();
  }
  function renderFrame() {
    field("frame-evidence").replaceChildren();
    if (!view.record || field("frame").value === "") return update();
    const index = Number(field("frame").value);
    for (const lane of view.record.report.lanes) {
      const current = lane.frames.find((frame) => frame.frame_index === index), article = node("article"); article.append(node("h4", "", lane.name));
      if (!current) { article.append(node("p", "", "Frame evidence unavailable.")); field("frame-evidence").append(article); continue; }
      article.append(node("p", "", current.evaluated ? `${current.matches.length} matches · ${current.false_positives.length} false positives · ${current.false_negatives.length} false negatives` : `Excluded: ${current.reasons.map((reason) => reason.replaceAll("_", " ")).join("; ")}`));
      const list = node("ul");
      for (const event of current.events) list.append(node("li", "", tools.eventText(event)));
      for (const confusion of current.class_confusions) list.append(node("li", "", `Class confusion: ${confusion.reference_identity} (${confusion.reference_label}) / tracker ${confusion.track_id} (${confusion.observed_label}), IoU ${confusion.iou.toFixed(3)}`));
      for (const match of current.matches) list.append(node("li", "", `Match ${match.reference_identity} ↔ tracker ${match.track_id}, IoU ${match.iou.toFixed(3)}`));
      for (const id of current.false_negatives) list.append(node("li", "", `Missed reference identity: ${id}`));
      for (const id of current.false_positives) list.append(node("li", "", `False-positive tracker ID: ${id}`));
      if (!list.children.length && current.evaluated) list.append(node("li", "", "No boxed observations or identity events on this evaluated frame."));
      article.append(list); field("frame-evidence").append(article);
    }
    update();
  }
  async function calculate() {
    if (view.busy || view.loading || !view.comparison) return;
    let payload; try { payload = config(); } catch (failure) { error(failure); return; }
    const generation = view.generation, comparisonID = view.comparison.id;
    view.recordRequest++;
    view.busy = true; error(null); update();
    try {
      const record = await api(`/api/temporal/tracking-comparisons/${safe(comparisonID)}/quality-reports`, { method: "POST", body: JSON.stringify(payload) });
      if (generation !== view.generation || !view.visible) return;
      if (!tools.matchesContext(record, context())) throw new Error("Calculated quality report does not match the selected saved evidence.");
      view.history.unshift(record); view.record = record; renderHistory(record.id); renderRecord(); remember();
      field("result").scrollIntoView({ block: "start" });
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.busy = false; update(); } }
  }
  function openContext(request) {
    view.pending = request;
    const accepted = window.IRISTracking.open(request.comparison_id);
    if (accepted === false) view.pending = null;
  }
  async function openReport(id) {
    const request = ++view.openRequest, generation = view.generation;
    try {
      const record = await api(`/api/temporal/tracking-quality-reports/${safe(id)}`);
      if (request !== view.openRequest || generation !== view.generation) return;
      openContext({ comparison_id: record.comparison_id, reference_id: record.reference_id, report_id: record.id });
    } catch (failure) { if (request === view.openRequest && generation === view.generation) { window.IRISNavigation.open("tracking"); error(failure); } }
  }
  field("calculate").addEventListener("click", calculate);
  field("reference").addEventListener("change", () => { remember(); update(); });
  field("iou").addEventListener("input", update);
  field("history").addEventListener("change", () => { error(null); loadRecord(field("history").value, view.generation, true); });
  field("refresh").addEventListener("click", () => { if (view.comparison && !view.busy) loadContext({ sequence: view.sequence, comparison: view.comparison, report: view.tracking }); });
  field("frame").addEventListener("change", renderFrame);
  field("show-frame").addEventListener("click", () => { if (view.record && field("frame").value !== "") window.dispatchEvent(new CustomEvent("iris:tracking-show-source-frame", { detail: { comparison_id: view.record.comparison_id, frame_index: Number(field("frame").value) } })); });
  field("open-reference").addEventListener("click", () => { if (view.sequence) window.dispatchEvent(new CustomEvent("iris:temporal-identities-open", { detail: { sequence_id: view.sequence.id, reference_id: field("reference").value || view.record?.reference_id } })); });
  window.addEventListener("iris:tracking-quality-context", (event) => { if (event.detail) loadContext(event.detail); else clearContext(); });
  window.addEventListener("iris:tracking-quality-open", (event) => { if (event.detail.report_id) openReport(event.detail.report_id); else openContext(event.detail); });
  window.addEventListener("iris:workspace", (event) => { view.visible = event.detail.name === "tracking"; if (!view.visible) { view.openRequest++; clearContext(); } });
  window.addEventListener("iris:project-initialized", () => {
    api("/api/temporal/tracking-quality-status").then((status) => {
      view.status = status;
      field("runtime-limits").textContent = `Protocol ${status.protocol.name} · bounded calculation: ${status.limits.max_frames} frames, ${status.limits.max_frame_objects} combined reference/observation boxes per frame, ${status.limits.max_identity_count} identities, ${status.limits.max_assignment_work.toLocaleString()} assignment work units. Limits fail explicitly without sampling.`;
      update();
    }).catch(() => { field("runtime-limits").textContent = "Calculation limits are unavailable. The server still validates the frozen inputs and its work limits before saving."; });
    if (requestedReport) openReport(requestedReport);
  });
  window.IRISTrackingQuality = Object.freeze({ open: openReport });
  update();
})();
