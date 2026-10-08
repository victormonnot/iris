"use strict";

(() => {
  const tools = window.IRISTrackingCostTools, safe = encodeURIComponent, field = (name) => $(`#cost-${name}`);
  const requestedRun = new URL(window.location.href).searchParams.get("tracking_cost");
  const maxImportBytes = 24 * 1024 * 1024;
  const view = { visible: false, generation: 0, request: 0, openRequest: 0, fileRequest: 0,
    loading: false, busy: false, pollBusy: false, timer: null, sequence: null, comparison: null,
    tracking: null, history: [], record: null, renderedID: null, imported: null, pending: null, memory: new Map() };
  const context = () => view.comparison && { comparison_id: view.comparison.id, sequence_id: view.sequence.id };
  const hasActive = () => view.history.some((record) => tools.active(record.job));
  function error(failure) { field("error").textContent = failure?.message || failure || ""; field("error").hidden = !failure; }
  function config() {
    return tools.configuration({ name: field("name").value, lane_index: Number(field("lane").value), device: field("device").value, repeats: Number(field("repeats").value), policy: field("policy").value, cadence_fps: field("policy").value === "simulated_latest" ? Number(field("cadence").value) : null });
  }
  function update() {
    const ready = Boolean(view.comparison && view.tracking), blocked = view.loading || view.busy, simulated = field("policy").value === "simulated_latest";
    for (const name of ["name", "lane", "device", "repeats", "policy", "cadence", "file", "history-lane"]) field(name).disabled = !ready || blocked;
    field("refresh").disabled = !ready || blocked;
    field("history").disabled = !ready || blocked || !view.history.length;
    field("cadence-field").hidden = !simulated;
    field("policy-hint").textContent = simulated ? "Virtual single worker: arrivals follow source frame index / your declared FPS. Superseded available arrivals are dropped, while missing source frames remain separate source gaps. The simulation uses measured service time and does not sleep or claim real camera latency." : "Every available saved frame is processed in source order. No arrival cadence is assumed and no available frames are intentionally dropped.";
    let valid = false, hint = "Choose a completed tracking comparison above.";
    if (ready) {
      try { config(); valid = true; hint = "Ready for an explicit fresh local measurement. Existing detector-cache timings are not added to produce this report."; }
      catch (failure) { hint = failure.message; }
      if (view.sequence.manifest.frames.length > 500) { valid = false; hint = "This sequence exceeds the 500 available frame measurement limit. Saved reports remain readable."; }
      if (hasActive()) { valid = false; hint = "A cost measurement is active for this comparison. Select it in history to inspect or cancel it before starting another."; }
    }
    field("config-status").textContent = view.loading ? "Loading saved cost runs…" : view.busy ? "Saving the explicit request…" : hint;
    field("run").disabled = !ready || blocked || !valid;
    field("import").disabled = !ready || blocked || !view.imported;
    field("cancel").disabled = blocked || !tools.active(view.record?.job) || view.record?.job.status === "cancelling";
    field("pass").disabled = blocked || !view.record?.report;
    $("#tracking-cost-panel").setAttribute("aria-busy", String(blocked));
  }
  function options(select, rows, label, placeholder, selected = "") {
    select.replaceChildren(new Option(placeholder, ""));
    rows.forEach((row) => select.append(new Option(label(row), row.id)));
    if (rows.some((row) => row.id === selected)) select.value = selected;
  }
  function remember() { if (view.comparison && view.record) view.memory.set(view.comparison.id, view.record.id); }
  function clearContext() {
    remember(); clearTimeout(view.timer); view.timer = null; view.generation++; view.request++; view.fileRequest++;
    view.loading = false; view.busy = false; view.pollBusy = false; view.sequence = null; view.comparison = null; view.tracking = null;
    view.history = []; view.record = null; view.renderedID = null; view.imported = null;
    field("file").value = ""; field("file-summary").textContent = "Choose a report file, then press Import explicitly.";
    field("job").hidden = true; field("result").hidden = true;
    options(field("history"), [], () => "", "No cost run selected");
    field("context").textContent = "Choose a completed tracking comparison to prepare a cost measurement or read its saved runs.";
    error(null); update();
  }
  function renderHistory(selected = view.record?.id) {
    const lane = field("history-lane").value;
    const history = view.history.filter((record) => lane === "all" || String(tools.laneIndex(record)) === lane);
    options(field("history"), history, (record) => `${record.name} · lane ${(tools.laneIndex(record) ?? 0) + 1} · ${record.job?.status || "saved"} · ${record.origin === "imported_declaration" ? "imported declaration" : "local worker"}`, history.length ? "Choose a saved cost run" : "No runs in this lane", selected);
  }
  async function loadContext(detail) {
    if (!detail?.report || detail.comparison.sequence_id !== detail.sequence.id) return clearContext();
    clearContext(); const generation = view.generation;
    view.sequence = detail.sequence; view.comparison = detail.comparison; view.tracking = detail.report; view.loading = true;
    const explicit = view.pending?.comparison_id === view.comparison.id, selected = explicit ? view.pending.run_id : view.memory.get(view.comparison.id);
    view.pending = null;
    field("context").textContent = `${view.comparison.name} · ${view.sequence.name} · ${view.sequence.manifest.frames.length} available saved frames. Cost runs preserve this comparison's frozen detector recipe and selected tracker profile.`;
    for (const name of ["lane", "history-lane"]) [...field(name).options].forEach((option) => { if (option.value !== "all") option.textContent = detail.report.lanes[Number(option.value)]?.name || `Lane ${Number(option.value) + 1}`; });
    update();
    try {
      const history = await api(`/api/temporal/tracking-comparisons/${safe(view.comparison.id)}/cost-runs`);
      if (generation !== view.generation || !view.visible) return;
      view.history = history.filter((record) => tools.matchesContext(record, context()));
      field("history-lane").value = "all"; renderHistory(selected);
      if (selected) await loadRecord(selected, generation, explicit);
      else if (explicit) $("#tracking-cost-panel").scrollIntoView({ block: "start" });
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.loading = false; update(); schedulePoll(); } }
  }
  function renderJob() {
    const record = view.record, job = record?.job; field("job").hidden = !job;
    if (!job) return;
    field("job-status").textContent = [record.name, job.status?.replaceAll("_", " "), job.message, job.error].filter(Boolean).join(" · ");
    field("progress").hidden = !tools.active(job);
    if (typeof job.progress === "number") field("progress").value = job.progress <= 1 ? job.progress * 100 : job.progress;
    else field("progress").removeAttribute("value");
    field("cancel").hidden = !tools.active(job) || record.origin === "imported_declaration";
  }
  function acceptRecord(record, scroll = false) {
    if (!tools.matchesContext(record, context())) throw new Error("Cost run does not match the selected comparison and frozen sequence.");
    const existing = view.history.findIndex((row) => row.id === record.id);
    if (existing < 0) view.history.unshift(record); else view.history[existing] = record;
    view.record = record; renderHistory(record.id); renderJob(); remember();
    if (record.job?.status === "succeeded" && record.report?.complete === true) {
      if (view.renderedID !== record.id) { view.renderedID = record.id; renderReport(); }
      if (scroll) field("result").scrollIntoView({ block: "start" });
    } else {
      view.renderedID = null; field("result").hidden = true;
      if (scroll) field("job").scrollIntoView({ block: "start" });
    }
    update();
  }
  async function loadRecord(id, generation = view.generation, scroll = false) {
    const request = ++view.request; view.record = null; view.renderedID = null;
    field("result").hidden = true; field("job").hidden = true; update();
    if (!id) return;
    try {
      const record = await api(`/api/temporal/tracking-cost-runs/${safe(id)}`);
      if (generation !== view.generation || request !== view.request || !view.visible) return;
      acceptRecord(record, scroll);
    } catch (failure) { if (generation === view.generation && request === view.request) error(failure); }
    update(); schedulePoll();
  }
  function card(container, label, value, explanation) {
    const row = node("dl", "cost-card"), amount = node("dd", "", value); if (explanation) amount.append(node("small", "", explanation));
    row.append(node("dt", "", label), amount); container.append(row);
  }
  function table(container, caption, headings, rows) {
    const result = node("table", "cost-table"), head = node("thead"), header = node("tr"), body = node("tbody");
    result.append(node("caption", "", caption));
    for (const heading of headings) { const th = node("th", "", heading); th.scope = "col"; header.append(th); } head.append(header);
    for (const row of rows) {
      const tr = node("tr", row.nested ? "cost-nested" : "");
      row.values.forEach((value, index) => { const cell = node(index ? "td" : "th", "", String(value)); if (!index) cell.scope = "row"; tr.append(cell); }); body.append(tr);
    }
    result.append(head, body); container.replaceChildren(result);
  }
  function renderReport() {
    const record = view.record, report = record.report, summary = report.summary, simulated = report.request.policy === "simulated_latest";
    field("result").hidden = false; field("result-title").textContent = record.name;
    field("result-scope").textContent = `${report.source.lane_name} · ${tools.policyLabel(report.request.policy)} · ${report.request.device.toUpperCase()} · ${report.request.repeats} repetition(s)${simulated ? ` · declared ${report.request.cadence_fps} FPS arrivals` : ""}. Saved run ${record.id}; changing preparation settings applies only to a future measurement.`;
    field("origin").hidden = record.origin !== "imported_declaration";
    field("origin").textContent = "Imported declaration: hardware, timing and execution metadata come from the report file. IRIS validates its structure and frozen source binding; it does not authenticate execution on the declared hardware.";
    field("coverage").textContent = `${summary.processed_frames} processed frames across ${summary.repeats} repetition(s) · ${summary.available_frames_per_pass} available frames per repetition · ${summary.dropped_frames} superseded available frames dropped · ${summary.source_gap_frames_per_pass} missing source frames per repetition. Source gaps are not scheduling drops.`;
    field("cards").replaceChildren();
    card(field("cards"), "Whole saved-frame pipeline · median", tools.milliseconds(summary.stages_ms.pipeline_ms.median), `p95 ${tools.milliseconds(summary.stages_ms.pipeline_ms.p95)} · ${summary.sample_count} measured samples`);
    card(field("cards"), "Measured service throughput", tools.fps(summary.service_fps), "Processed frames / total measured pipeline service time; excludes setup and warmup.");
    card(field("cards"), "Measured repetition wall time", tools.milliseconds(summary.wall_ms), "Sum of repetition wall times. Setup and warmup are recorded separately.");
    if (simulated) {
      card(field("cards"), "Simulated arrival-to-finish · median", tools.milliseconds(summary.simulated_latency_ms?.median), `p95 ${tools.milliseconds(summary.simulated_latency_ms?.p95)} · virtual scheduling delay + measured service; not camera latency`);
      card(field("cards"), "Simulated output cadence", tools.fps(summary.simulated_output_fps), "Completion cadence of processed available frames on the declared virtual arrival clock.");
    }
    const stageRows = tools.stages.map(([key, label, kind]) => {
      const stats = summary.stages_ms[key];
      return { nested: kind === "nested", values: [label, kind === "nested" ? "Nested detail" : key === "pipeline_ms" ? "Outer pipeline" : "Outer stage", stats?.count ?? 0, tools.milliseconds(stats?.median), tools.milliseconds(stats?.p95), tools.milliseconds(stats?.mean), tools.milliseconds(stats?.min), tools.milliseconds(stats?.max)] };
    });
    table(field("stages"), "Fresh run timing distribution · milliseconds", ["Stage", "Scope", "Samples", "Median", "p95", "Mean", "Minimum", "Maximum"], stageRows);
    field("memory").replaceChildren();
    card(field("memory"), "Process RSS · sampled peak", tools.memory(summary.memory.rss_sampled_peak_bytes), "Boundary samples; short peaks between samples may be missed.");
    card(field("memory"), "Process lifetime RSS high-water", tools.memory(summary.memory.process_lifetime_peak_bytes), "Includes process setup and warmup, unlike measured-frame boundary samples.");
    card(field("memory"), "CUDA allocated peak", tools.memory(summary.memory.cuda_allocated_peak_bytes), report.request.device === "cuda" ? "PyTorch allocator on the actual selected CUDA device." : "CPU run: no CUDA allocator measurement.");
    card(field("memory"), "CUDA reserved peak", tools.memory(summary.memory.cuda_reserved_peak_bytes), report.request.device === "cuda" ? "Reserved allocator pool can retain warmup cache." : "CPU run: no CUDA allocator measurement.");
    options(field("pass"), report.passes.map((pass) => ({ ...pass, id: String(pass.pass_index) })), (pass) => `Repetition ${pass.pass_index + 1} · ${pass.frames.length} processed · ${pass.dropped_frame_indices.length} dropped`, "Choose a repetition", String(report.passes[0]?.pass_index ?? ""));
    field("limitations").replaceChildren(); for (const limitation of report.limitations) field("limitations").append(node("li", "", limitation));
    field("provenance").textContent = JSON.stringify({ origin: record.origin, schema: report.schema, request: report.request, source: report.source, execution: report.execution, profile: report.profile, detector_config: report.detector_config, protocol: report.protocol }, null, 2);
    const link = new URL(window.location.href); link.search = ""; link.searchParams.set("project", state.projectId); link.searchParams.set("tracking_cost", record.id);
    field("permalink").href = `${link.pathname}${link.search}`; field("download").href = projectURL(`/api/temporal/tracking-cost-runs/${safe(record.id)}/report`);
    renderPass();
  }
  function renderPass() {
    const report = view.record?.report, pass = report?.passes.find((row) => String(row.pass_index) === field("pass").value);
    if (!pass) { field("timeline").replaceChildren(); field("pass-summary").textContent = ""; field("drops").textContent = ""; return; }
    field("pass-summary").textContent = `Repetition ${pass.pass_index + 1} · ${tools.milliseconds(pass.wall_ms)} wall time · tracker reset before this repetition. ${report.request.policy === "simulated_latest" ? "Schedule times use a virtual clock in milliseconds." : "Offline policy has no arrival, queue-delay or simulated-latency clock."}`;
    table(field("timeline"), "Processed source frames · no synthetic updates for missing or dropped frames", ["Source frame", "Pipeline service", "Arrival", "Start", "Finish", "Queue delay", "Simulated latency", "Detections", "Observed tracks"], pass.frames.map((frame) => ({ values: [frame.frame_index, tools.milliseconds(frame.timing.pipeline_ms), frame.schedule ? tools.milliseconds(frame.schedule.arrival_ms) : "Not simulated", frame.schedule ? tools.milliseconds(frame.schedule.start_ms) : "Not simulated", frame.schedule ? tools.milliseconds(frame.schedule.finish_ms) : "Not simulated", frame.schedule ? tools.milliseconds(frame.schedule.queue_delay_ms) : "Not simulated", frame.schedule ? tools.milliseconds(frame.schedule.latency_ms) : "Not simulated", frame.work.detection_count, frame.work.observation_count] })));
    field("drops").textContent = pass.dropped_frame_indices.length ? `Superseded available frame indices: ${pass.dropped_frame_indices.join(", ")}. Missing source frame gaps remain separate.` : "No available frames were dropped in this repetition. Missing source gaps are still outside the processed frame list.";
  }
  async function mutation(action) {
    if (!view.comparison || view.loading || view.busy) return;
    const generation = view.generation; view.busy = true; view.request++; error(null); update();
    try { await action(generation); } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.busy = false; update(); schedulePoll(); } }
  }
  function schedulePoll() {
    clearTimeout(view.timer); view.timer = null;
    if (view.visible && view.comparison && (hasActive() || tools.active(view.record?.job))) view.timer = setTimeout(poll, 1800);
  }
  async function poll() {
    if (!view.visible || !view.comparison || view.busy || view.loading || view.pollBusy) return schedulePoll();
    const generation = view.generation, id = view.record?.id, request = view.request; view.pollBusy = true;
    try {
      const [history, detail] = await Promise.all([api(`/api/temporal/tracking-comparisons/${safe(view.comparison.id)}/cost-runs`), id ? api(`/api/temporal/tracking-cost-runs/${safe(id)}`) : Promise.resolve(null)]);
      if (generation !== view.generation || !view.visible) return;
      view.history = history.filter((record) => tools.matchesContext(record, context())); renderHistory();
      if (detail && request === view.request && id === view.record?.id) acceptRecord(detail);
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.pollBusy = false; update(); schedulePoll(); } }
  }
  function openContext(request) { view.pending = request; if (window.IRISTracking.open(request.comparison_id) === false) view.pending = null; }
  async function openRun(id) {
    const request = ++view.openRequest, generation = view.generation;
    try {
      const record = await api(`/api/temporal/tracking-cost-runs/${safe(id)}`);
      if (request !== view.openRequest || generation !== view.generation) return;
      openContext({ comparison_id: record.comparison_id, run_id: record.id });
    } catch (failure) { if (request === view.openRequest && generation === view.generation) { window.IRISNavigation.open("tracking"); error(failure); } }
  }
  field("run").addEventListener("click", () => {
    let payload; try { payload = config(); } catch (failure) { error(failure); return; }
    if (hasActive()) return;
    mutation(async (generation) => {
      const record = await api(`/api/temporal/tracking-comparisons/${safe(view.comparison.id)}/cost-runs`, { method: "POST", body: JSON.stringify(payload) });
      if (generation !== view.generation || !view.visible) return;
      field("history-lane").value = "all"; acceptRecord(record, true);
      if (typeof refreshJobs === "function") refreshJobs().catch(() => {});
    });
  });
  field("cancel").addEventListener("click", () => mutation(async (generation) => {
    if (!tools.active(view.record?.job)) return;
    await api(`/api/jobs/${safe(view.record.job.id)}/cancel`, { method: "POST" });
    if (generation !== view.generation || !view.visible) return;
    view.record.job.status = "cancelling"; renderJob();
  }));
  field("file").addEventListener("change", async () => {
    const generation = view.generation, request = ++view.fileRequest, file = field("file").files[0]; view.imported = null; error(null); update();
    field("file-summary").textContent = file ? "Reading report JSON…" : "Choose a report file, then press Import explicitly.";
    if (!file) return;
    try {
      if (file.size > maxImportBytes) throw new Error("Report exceeds the 24 MiB import limit.");
      const imported = tools.parseImport(await file.text()), report = imported.report;
      if (generation !== view.generation || request !== view.fileRequest || !view.visible) return;
      if (!tools.matchesContext({ comparison_id: report.source?.comparison_id, sequence_id: report.source?.sequence_id, report }, context())) throw new Error("Imported report belongs to a different comparison or frozen sequence.");
      field("file-summary").textContent = `${file.name} · ${report.source.lane_name} · ${tools.policyLabel(report.request.policy)} · ${report.request.device.toUpperCase()} declared. Press Import to validate and save this external declaration.`;
      view.imported = imported;
    } catch (failure) { if (generation === view.generation && request === view.fileRequest) { error(failure); field("file-summary").textContent = "File not ready to import. Choose a valid completed report for this comparison."; } }
    update();
  });
  field("import").addEventListener("click", () => mutation(async (generation) => {
    if (!view.imported) return;
    const record = await api(`/api/temporal/tracking-comparisons/${safe(view.comparison.id)}/cost-runs/import`, { method: "POST", body: view.imported.body });
    if (generation !== view.generation || !view.visible) return;
    view.imported = null; field("file").value = ""; field("file-summary").textContent = "Imported as an execution and hardware declaration.";
    field("history-lane").value = "all"; acceptRecord(record, true); if (typeof refreshJobs === "function") refreshJobs().catch(() => {});
  }));
  for (const name of ["name", "lane", "device", "repeats", "policy", "cadence"]) field(name).addEventListener("input", update);
  field("history-lane").addEventListener("change", () => { view.request++; view.record = null; view.renderedID = null; field("job").hidden = true; field("result").hidden = true; renderHistory(); update(); });
  field("history").addEventListener("change", () => { error(null); loadRecord(field("history").value, view.generation, true); });
  field("refresh").addEventListener("click", () => { if (view.comparison && !view.busy) loadContext({ sequence: view.sequence, comparison: view.comparison, report: view.tracking }); });
  field("pass").addEventListener("change", renderPass);
  window.addEventListener("iris:tracking-quality-context", (event) => { if (event.detail) loadContext(event.detail); else clearContext(); });
  window.addEventListener("iris:tracking-cost-open", (event) => { if (event.detail.comparison_id) openContext(event.detail); else if (event.detail.run_id) openRun(event.detail.run_id); });
  window.addEventListener("iris:workspace", (event) => { view.visible = event.detail.name === "tracking"; if (!view.visible) { view.openRequest++; clearContext(); } });
  window.addEventListener("iris:project-initialized", () => { if (requestedRun) openRun(requestedRun); });
  window.IRISTrackingCost = Object.freeze({ open: openRun });
  update();
})();
