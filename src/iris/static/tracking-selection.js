"use strict";

(() => {
  const tools = window.IRISTrackingSelectionTools, qualityTools = window.IRISTrackingQualityTools;
  const field = (name) => $(`#selection-${name}`), safe = encodeURIComponent;
  const requestedID = new URL(window.location.href).searchParams.get("tracking_selection");
  const policyKeys = Object.keys(tools.defaults).filter((key) => key !== "schema");
  const view = { visible: false, loaded: false, generation: 0, sourceRequest: 0, recordRequest: 0, referenceRequest: 0, imageRequest: 0, formVersion: 0,
    loading: false, sourceLoading: false, busy: false, pollBusy: false, timer: null, sources: [], history: [], source: null, reference: null, status: null,
    selection: null, release: null, preview: null, record: null, report: null, position: 0, imagesReady: false, image: null, svg: null, renderedID: null };
  function error(value) { field("error").textContent = value?.message || value || ""; field("error").hidden = !value; }
  function options(select, rows, label, placeholder, selected = "") {
    select.replaceChildren(new Option(placeholder, ""));
    for (const row of rows) select.append(new Option(label(row), row.id));
    select.value = rows.some((row) => row.id === selected) ? selected : "";
  }
  function currentFrame() { return view.source?.replay.passes[0].frames[view.position]; }
  function selectedLane() { return view.report?.lanes.find((lane) => lane.id === field("lane").value) || view.report?.lanes[1]; }
  function laneFrame() { return selectedLane()?.frames[view.position]; }
  function policy() {
    return Object.fromEntries([["schema", tools.defaults.schema], ...policyKeys.map((key) => [key, key === "max_lost_seconds" && field(key).value.trim() === "" ? null : field(key).value.trim() === "" ? NaN : Number(field(key).value)])]);
  }
  function mapping() { return Object.fromEntries([...field("mapping").querySelectorAll("select")].map((select) => [select.dataset.nativeId, select.value === "__ignore__" ? null : select.value])); }
  function configuration() {
    return tools.configuration({ name: field("name").value, selection: view.selection, release_frame_id: view.release, policy: policy(), max_seconds: Number(field("seconds").value),
      evaluation: field("reference").value ? { reference_id: field("reference").value, identity_id: field("identity").value, class_mapping: mapping(), iou_threshold: Number(field("iou").value) } : null }, view.source);
  }
  function update() {
    const active = tools.active(view.record?.job), blocked = view.loading || view.sourceLoading || view.busy || active;
    for (const control of field("prepare").querySelectorAll("input,select,button")) control.disabled = blocked;
    field("source").disabled = blocked; field("refresh").disabled = blocked;
    field("history").disabled = view.loading || view.busy || !view.history.length;
    field("identity").disabled = blocked || !view.reference;
    field("iou").disabled = blocked || !view.reference;
    field("previous").disabled = !view.source || view.position === 0 || view.sourceLoading;
    field("next").disabled = !view.source || view.position >= view.source.sequence.manifest.frames.length - 1 || view.sourceLoading;
    field("scrub").disabled = !view.source || view.sourceLoading;
    field("clear-anchor").disabled = blocked || !view.selection;
    field("set-release").disabled = blocked || !view.selection || view.position <= tools.framePosition(view.source?.sequence, view.selection.frame_id);
    field("clear-release").disabled = blocked || !view.release;
    field("cancel").disabled = view.busy || !active || view.record?.job.status === "cancelling";
    let valid = false, hint;
    try { configuration(); valid = true; hint = "Inspect the initial object, release and rules, then preview this exact scenario."; }
    catch (failure) { hint = failure.message; }
    if (view.history.some((row) => tools.active(row.job))) { valid = false; hint = "A continuity scenario is active. Inspect or cancel it in history before starting another."; }
    field("prepare-status").textContent = view.loading || view.sourceLoading ? "Loading frozen source evidence…" : view.busy ? "Processing the explicit request…" : hint;
    field("preview").disabled = blocked || !valid; field("run").disabled = blocked || !valid || !view.preview;
    $("#tracking-selection-panel").setAttribute("aria-busy", String(blocked));
  }
  function clearResult() {
    view.report = null; view.renderedID = null; field("result").hidden = true; field("lane-field").hidden = true;
    field("timeline").replaceChildren(); field("candidates").textContent = "";
  }
  function invalidate() { view.formVersion++; view.preview = null; field("preview-result").hidden = true; clearResult(); renderSelection(); draw(); update(); }
  function clearSource() {
    view.source = null; view.reference = null; view.selection = null; view.release = null; view.imagesReady = false; view.imageRequest++; view.referenceRequest++;
    view.image = null; view.svg = null; field("stage").replaceChildren(); field("observations").replaceChildren(); field("viewer").hidden = true;
    field("source-summary").textContent = ""; field("clock").textContent = ""; field("mapping").replaceChildren(); options(field("reference"), [], () => "", "No quality assessment"); options(field("identity"), [], () => "", "Choose explicitly from the reference");
    field("reference-summary").textContent = "Without a reference, the report describes behavior only."; clearResult();
  }
  function reset() {
    clearTimeout(view.timer); view.timer = null; view.generation++; view.recordRequest++; view.sourceRequest++;
    view.loaded = false; view.loading = false; view.sourceLoading = false; view.busy = false; view.pollBusy = false;
    view.sources = []; view.history = []; view.record = null; view.preview = null; clearSource();
    field("job").hidden = true; field("preview-result").hidden = true;
    options(field("source"), [], () => "", "Choose a saved source"); options(field("history"), [], () => "", "No saved scenarios"); error(null); update();
  }
  function renderHistory() { options(field("history"), view.history, (row) => `${row.name} · ${row.job.status}`, "Choose a saved scenario", view.record?.id); }
  async function refresh() {
    if (view.loading || view.busy || view.sourceLoading) return;
    const generation = view.generation, recordToken = view.recordRequest, recordID = view.record?.id; view.loading = true; error(null); update();
    try {
      const [catalogue, status, history] = await Promise.all([api("/api/temporal/tracking-selection-sources"), api("/api/temporal/tracking-selection-status"), api("/api/temporal/tracking-selections")]);
      if (generation !== view.generation || !view.visible) return;
      view.sources = catalogue.sources; view.status = status; view.history = history; view.loaded = true;
      options(field("source"), view.sources.map((row) => ({ ...row, id: tools.sourceKey(row.source) })), tools.sourceLabel, "Choose a saved source", view.source ? tools.sourceKey(view.source.source) : "");
      renderHistory();
      const changedRecord = recordID && recordToken === view.recordRequest && history.find((row) => row.id === recordID && row.job.status !== view.record?.job.status);
      if (changedRecord) await loadRecord(recordID);
      if (generation !== view.generation || !view.visible) return;
      field("readiness").textContent = `${view.sources.length} frozen source lane(s) or profile(s). These scenarios reuse measured observations from the first saved replay; no detector, tracker or cloud call is launched.`;
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.loading = false; update(); schedulePoll(); } }
  }
  async function loadSource(descriptor, restore = null, recordToken = null) {
    const generation = view.generation, request = ++view.sourceRequest;
    view.formVersion++; view.sourceLoading = true; clearSource(); view.preview = null; field("preview-result").hidden = true; update();
    try {
      const source = await api("/api/temporal/tracking-selection-source", { method: "POST", body: JSON.stringify(descriptor) });
      if (generation !== view.generation || request !== view.sourceRequest || !view.visible || (recordToken !== null && recordToken !== view.recordRequest)) return false;
      view.source = source; view.position = 0;
      field("source").value = tools.sourceKey(source.source);
      field("viewer").hidden = false; field("scrub").max = source.sequence.manifest.frames.length - 1;
      const profile = source.replay.profile, role = source.source.kind === "study" ? "Study evidence remains tuning evidence." : "Using this sequence does not establish an independent test.";
      const memberships = (source.context?.current_dataset_memberships || []).map((item) => `${item.name}: ${item.split}`).join("; ");
      field("source-summary").textContent = `${source.sequence.name} · ${source.sequence.manifest.frames.length} available frames · ${profile.algorithm} · saved first pass. ${role}${memberships ? ` Dataset roles: ${memberships}.` : " No frozen dataset role is assigned."}`;
      field("clock").textContent = tools.clockLabel(source.sequence.manifest.clock);
      options(field("reference"), source.references || [], qualityTools.referenceLabel, "No quality assessment");
      if (restore) {
        view.selection = tools.clone(restore.selection); view.release = restore.release_frame_id;
        field("name").value = restore.name; field("seconds").value = restore.max_seconds;
        policyKeys.forEach((key) => { field(key).value = restore.policy[key] === null ? "" : String(restore.policy[key]); });
        if (restore.evaluation) {
          field("reference").value = restore.evaluation.reference_id;
          await loadReference(restore.evaluation.reference_id, restore.evaluation, false);
          if (generation !== view.generation || request !== view.sourceRequest || (recordToken !== null && recordToken !== view.recordRequest)) return false;
        }
        view.position = Math.max(0, tools.framePosition(source.sequence, restore.selection.frame_id));
      }
      renderSelection(); showFrame(view.position); return true;
    } catch (failure) { if (generation === view.generation && request === view.sourceRequest && (recordToken === null || recordToken === view.recordRequest)) error(failure); return false; }
    finally { if (generation === view.generation && request === view.sourceRequest) { view.sourceLoading = false; update(); } }
  }
  async function loadReference(id, restore = null, edit = true) {
    const generation = view.generation, request = ++view.referenceRequest, source = view.source;
    view.reference = null; field("mapping").replaceChildren(); options(field("identity"), [], () => "", "Choose explicitly from the reference");
    if (edit) invalidate();
    if (!id || !source) { field("reference-summary").textContent = "Without a reference, the report describes behavior only."; update(); return; }
    try {
      const reference = await api(`/api/temporal/references/${safe(id)}`);
      if (generation !== view.generation || request !== view.referenceRequest || view.source !== source || !view.visible) return;
      if (reference.sequence_id !== source.sequence.id) throw new Error("The reference belongs to another source sequence.");
      view.reference = reference;
      options(field("identity"), reference.payload.identities, (item) => `${item.id} · ${item.label}`, "Choose explicitly from the reference", restore?.identity_id);
      const taxonomy = source.sequence.manifest.taxonomy, detector = source.replay.cache.config.detector;
      const suggested = qualityTools.suggestions(detector, taxonomy, source.replay.profile.class_ids);
      for (const id of source.replay.profile.class_ids) {
        const row = node("div"), label = node("label", "", `${detector.classes.find((item) => item.id === id)?.name || "Native class"} · detector ${id}`), select = node("select");
        select.id = `selection-map-${id}`; select.dataset.nativeId = String(id); label.htmlFor = select.id;
        select.append(new Option("Choose reference class or ignore", ""), new Option("Ignore this detector class", "__ignore__"));
        taxonomy.classes.forEach((item) => select.append(new Option(`${item.name || item.id} · ${item.id}`, item.id)));
        const mapping = restore?.class_mapping || suggested;
        if (Object.hasOwn(mapping, String(id))) select.value = mapping[id] === null ? "__ignore__" : mapping[id];
        select.addEventListener("change", invalidate); row.append(label, select); field("mapping").append(row);
      }
      if (restore) field("iou").value = restore.iou_threshold;
      field("reference-summary").textContent = `${qualityTools.referenceLabel(reference)}. Choose the human identity explicitly. The initial observed box must uniquely match it. Reference identity is used for assessment only, never for recovery.`;
    } catch (failure) { if (generation === view.generation && request === view.referenceRequest && source === view.source) error(failure); }
    finally { if (generation === view.generation && request === view.referenceRequest) update(); }
  }
  function renderSelection() {
    const observed = tools.anchor(view.source, view.selection), position = tools.framePosition(view.source?.sequence, view.selection?.frame_id);
    const frame = view.source?.sequence.manifest.frames[position];
    field("anchor").textContent = observed && frame ? `Initial object: source frame ${frame.frame_index}, ${observed.label}, detection ${observed.detection_index}, native ID ${observed.track_id}. The logical selection is independent of later numeric IDs.` : "No initial object selected.";
    const released = view.source?.sequence.manifest.frames.find((item) => item.frame_id === view.release);
    field("release").textContent = released ? `Explicit release at source frame ${released.frame_index}.` : "No explicit release. Timeout can still expire the scenario.";
  }
  function choose(row) {
    if (!view.imagesReady || view.loading || view.sourceLoading || view.busy || tools.active(view.record?.job) || !row.confirmed) return;
    view.selection = { frame_id: currentFrame().frame_id, detection_index: row.detection_index };
    if (view.release && tools.framePosition(view.source.sequence, view.release) <= view.position) view.release = null;
    invalidate();
  }
  function svgNode(tag, attrs = {}, text) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function draw() {
    view.svg?.remove(); view.svg = null; field("observations").replaceChildren();
    const frame = currentFrame(); if (!frame) return;
    const result = laneFrame();
    field("state").textContent = result ? `${tools.stateLabel(result.state)} · ${tools.reasonLabel(result.reason)} · ${result.selected ? `accepted native ID ${result.selected.track_id}` : "no accepted measured observation"}${result.pending ? ` · confirming ID ${result.pending.track_id}: ${result.pending.observations}/${result.pending.required}` : ""}${result.age.updates !== null ? ` · last accepted ${result.age.updates} available update(s) ago` : ""}${result.age.seconds !== null ? ` · ${result.age.seconds.toFixed(3)} source seconds` : " · source age unavailable"}.` : "Prepare a scenario to inspect its continuity. Selecting a box changes the draft only.";
    field("candidates").textContent = result ? JSON.stringify(result.candidates, null, 2) : "Candidate checks become available after an explicit run.";
    const [width, height] = frame.input_size, svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Measured observations and selected object" });
    for (const row of frame.observations) {
      const initial = view.selection?.frame_id === frame.frame_id && view.selection.detection_index === row.detection_index;
      const accepted = result?.selected?.detection_index === row.detection_index;
      const group = svgNode("g", { "data-detection": row.detection_index }), [x1, y1, x2, y2] = row.box;
      group.append(svgNode("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: `tracking-box${initial ? " tracking-selected" : ""}${accepted ? " selection-accepted" : ""}${row.confirmed ? "" : " selection-ineligible"}` }));
      group.append(svgNode("text", { x: x1 + 2, y: Math.max(height / 30, y1 - 4), "font-size": Math.max(10, width / 52) }, `${accepted ? "Selected · " : ""}ID ${row.track_id} · ${row.label}`));
      group.append(svgNode("title", {}, `Measured detection ${row.detection_index}, score ${row.score.toFixed(3)}${row.confirmed ? "; click to set the initial selection" : "; unconfirmed, cannot select"}`));
      group.addEventListener("click", () => choose(row)); svg.append(group);
      const button = node("button", "tracking-evidence-button", `ID ${row.track_id} · ${row.label} · detection ${row.detection_index} · score ${row.score.toFixed(3)}${row.confirmed ? "" : " · unconfirmed"}`);
      button.type = "button"; button.disabled = !row.confirmed || !view.imagesReady || view.busy || view.sourceLoading || tools.active(view.record?.job); button.setAttribute("aria-pressed", String(initial)); button.addEventListener("click", () => choose(row)); field("observations").append(button);
    }
    if (!frame.observations.length) field("observations").append(node("p", "field-hint", "No measured tracking observation on this available frame. Predictions are not selectable."));
    if (view.imagesReady) { field("stage").append(svg); view.svg = svg; }
    renderTimeline();
  }
  function showFrame(position) {
    if (!view.source || !view.visible) return;
    const frames = view.source.sequence.manifest.frames;
    view.position = Math.max(0, Math.min(position, frames.length - 1)); view.imagesReady = false; view.image = null; view.svg = null;
    const frame = frames[view.position], source = view.source, generation = view.generation, request = ++view.imageRequest;
    field("scrub").value = String(view.position); field("position").textContent = `${view.position + 1}/${frames.length} · source frame ${frame.frame_index}${frame.timestamp_seconds === null ? " · source time unknown" : ` · ${frame.timestamp_seconds.toFixed(3)} s`}`;
    const gap = view.position > 0 ? frame.frame_index - frames[view.position - 1].frame_index - 1 : 0;
    field("gap").hidden = gap <= 0; field("gap").textContent = `${gap} source frames are missing before this image. Their content and selection state are unknown; no synthetic updates are added.`;
    field("stage").replaceChildren(); field("stage").classList.remove("tracking-stage-loaded"); field("stage").style.aspectRatio = `${frame.width} / ${frame.height}`;
    field("image-status").textContent = "Loading and verifying the saved source image…";
    const image = node("img"); image.alt = `Source frame ${frame.frame_index}`;
    image.onload = () => {
      if (generation !== view.generation || request !== view.imageRequest || source !== view.source || !view.visible) return;
      if (image.naturalWidth !== frame.width || image.naturalHeight !== frame.height) { image.onerror(); return; }
      view.imagesReady = true; view.image = image; field("stage").classList.add("tracking-stage-loaded"); field("stage").append(image);
      field("image-status").textContent = "Verified source pixels; boxes use the original image coordinates."; draw(); update();
    };
    image.onerror = () => {
      if (generation !== view.generation || request !== view.imageRequest || source !== view.source || !view.visible) return;
      view.imagesReady = false; field("stage").replaceChildren(); field("image-status").textContent = "Source image unavailable or verification failed. Overlays and initial selection are disabled; refresh the source to retry."; draw(); update();
    };
    image.src = projectURL(`/api/temporal/sequences/${safe(source.sequence.id)}/frames/${safe(frame.frame_id)}/image`);
    draw(); update();
  }
  function table(container, headings, rows) {
    container.replaceChildren(); const wrapper = node("div", "selection-table-scroll"), table = node("table", "selection-table"), head = node("thead"), tr = node("tr"), body = node("tbody");
    headings.forEach((label) => { const cell = node("th", "", label); cell.scope = "col"; tr.append(cell); }); head.append(tr);
    rows.forEach((row) => { const line = node("tr"); row.forEach((value, index) => { const cell = node(index ? "td" : "th", "", String(value ?? "Unavailable")); if (!index) cell.scope = "row"; line.append(cell); }); body.append(line); });
    table.append(head, body); wrapper.append(table); container.append(wrapper);
  }
  function renderTimeline() {
    field("timeline").replaceChildren(); const lane = selectedLane(); if (!lane) return;
    for (const [position, frame] of lane.frames.entries()) {
      const button = node("button", "", `Source ${frame.frame_index} · ${tools.stateLabel(frame.state)}${frame.selected ? ` · accepted ID ${frame.selected.track_id}` : ""}${frame.source_gap ? " · preceding source gap" : ""}`);
      button.type = "button"; button.setAttribute("aria-current", String(position === view.position)); button.addEventListener("click", () => { showFrame(position); field("viewer").scrollIntoView({ block: "start" }); }); field("timeline").append(button);
    }
  }
  function renderReport(record) {
    const report = record.report; view.report = report; view.renderedID = record.id;
    field("result").hidden = false; field("lane-field").hidden = false; field("result-title").textContent = record.name;
    field("result-scope").textContent = `Saved scenario ${record.id} · first saved replay pass · one logical object. Editing the draft hides this saved result; history preserves it unchanged.`;
    field("result-warning").textContent = report.repeatability.status === "observed_mismatch" ? "The source tracker repetitions differ. Behavior can be inspected, but identity-quality claims are unavailable." : "Track ID only is a diagnostic comparison, not a replay of an application’s complete policy. Guarded recovery can still select a similar-looking object; no physical identity is guaranteed.";
    table(field("behavior"), ["Diagnostic policy", "Accepted frames", "Recoveries", "Native ID changes", "Unobserved active frames"], report.lanes.map((lane) => [lane.name, lane.summary.selected_frames, lane.summary.recovery_events, lane.summary.track_id_changes, lane.summary.unobserved_frames]));
    const evaluated = report.lanes.some((lane) => lane.quality.status === "available");
    field("quality-note").textContent = evaluated ? `Assessment uses explicitly chosen human identity ${report.request.evaluation.identity_id}. Only supported human-reviewed frames count; absent, unknown and ambiguous cases remain distinct. Source gaps contribute no duration.` : report.lanes.map((lane) => `${lane.name}: ${tools.reasonLabel(lane.quality.reason) || "No eligible human reference assessment"}`).join(" · ");
    const qualityRows = [
      ["Assessed / active frames", (q) => `${q.coverage.evaluated_frames} / ${q.coverage.active_frames}`],
      ["Correct target", (q) => q.counts.correct_target], ["Wrong other identity", (q) => q.counts.wrong_other_identity],
      ["Selected box unmatched", (q) => q.counts.unmatched_selected_box], ["Reference ambiguous", (q) => q.counts.reference_ambiguous],
      ["Visible target · no accepted observation", (q) => q.counts.abstained_visible], ["Target absent · no accepted observation", (q) => q.counts.abstained_absent],
      ["Target agreement", (q) => qualityTools.percentage(q.rates.target_agreement)], ["Wrong identity among selected", (q) => qualityTools.percentage(q.rates.wrong_identity_among_selected)],
      ["Recovery correct / wrong / unavailable", (q) => `${q.recoveries.correct} / ${q.recoveries.wrong} / ${q.recoveries.unavailable}`],
      ["Supported source seconds", (q) => tools.duration(q.durations.evaluated_seconds)], ["Unobserved source seconds", (q) => tools.duration(q.durations.unobserved_seconds)],
      ["Wrong-identity source seconds", (q) => tools.duration(q.durations.wrong_identity_seconds)],
    ];
    table(field("quality"), ["Assessment metric", ...report.lanes.map((lane) => lane.name)], qualityRows.map(([label, value]) => [label, ...report.lanes.map((lane) => lane.quality.status === "available" ? value(lane.quality) : null)]));
    const assisted = report.reference?.origin || report.lanes.find((lane) => lane.quality.coverage.reference_origin)?.quality.coverage.reference_origin;
    field("origin").hidden = !assisted;
    field("origin").textContent = assisted ? "This human reference started from tracker proposals. Human review preserves its assisted origin; these results do not establish an independent identity benchmark." : "";
    field("limitations").replaceChildren(...report.limitations.map((value) => node("li", "", value)));
    field("provenance").textContent = JSON.stringify({ source_binding: report.source_binding, request: report.request, reference: report.reference, repeatability: report.repeatability, quality: report.lanes.map((lane) => ({ policy: lane.id, quality: lane.quality })) }, null, 2);
    field("download").href = projectURL(`/api/temporal/tracking-selections/${safe(record.id)}/report`);
    const link = new URL(window.location.href); link.search = ""; link.searchParams.set("project", state.projectId); link.searchParams.set("tracking_selection", record.id); field("link").href = link.href;
    draw(); update();
  }
  function renderJob() {
    const job = view.record?.job; field("job").hidden = !job; if (!job) return;
    field("job-status").textContent = [view.record.name, job.status, job.message, job.error].filter(Boolean).join(" · ");
    field("progress").hidden = !tools.active(job); field("progress").value = (job.progress || 0) * 100; field("cancel").hidden = !tools.active(job);
  }
  async function acceptRecord(record, scroll = false, token = view.recordRequest) {
    const generation = view.generation, index = view.history.findIndex((row) => row.id === record.id);
    if (index < 0) view.history.unshift(record); else view.history[index] = record;
    view.record = record; renderHistory(); renderJob();
    if (record.job.status === "succeeded" && record.report?.complete) {
      if (view.renderedID !== record.id) {
        const ready = await loadSource(record.report.request.source, record.report.request, token);
        if (!ready || generation !== view.generation || token !== view.recordRequest || !view.visible) return;
        renderReport(record);
      }
      if (scroll) field("viewer").scrollIntoView({ block: "start" });
    } else { clearResult(); if (scroll) field("job").scrollIntoView({ block: "start" }); }
    update(); schedulePoll();
  }
  async function loadRecord(id, scroll = false) {
    const generation = view.generation, request = ++view.recordRequest;
    view.formVersion++; view.preview = null; field("preview-result").hidden = true;
    view.sourceRequest++; view.sourceLoading = Boolean(id); view.record = null; clearSource(); field("job").hidden = true; error(null); update();
    if (!id) return;
    try {
      const record = await api(`/api/temporal/tracking-selections/${safe(id)}`);
      if (generation !== view.generation || request !== view.recordRequest || !view.visible) return;
      await acceptRecord(record, scroll, request);
    } catch (failure) { if (generation === view.generation && request === view.recordRequest) error(failure); }
    finally { if (generation === view.generation && request === view.recordRequest) { view.sourceLoading = false; update(); } }
  }
  async function mutation(action) {
    if (view.busy || view.loading || view.sourceLoading) return;
    const generation = view.generation; view.busy = true; error(null); update();
    try { await action(generation); }
    catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.busy = false; update(); draw(); schedulePoll(); } }
  }
  function schedulePoll() { clearTimeout(view.timer); view.timer = null; if (view.visible && view.history.some((row) => tools.active(row.job))) view.timer = setTimeout(poll, 1800); }
  async function poll() {
    view.timer = null; if (!view.visible || view.pollBusy || view.busy || view.sourceLoading) { schedulePoll(); return; }
    const generation = view.generation, request = view.recordRequest, id = view.record?.id; view.pollBusy = true;
    try {
      const [history, record] = await Promise.all([api("/api/temporal/tracking-selections"), id ? api(`/api/temporal/tracking-selections/${safe(id)}`) : Promise.resolve(null)]);
      if (generation !== view.generation || !view.visible) return;
      view.history = history; renderHistory();
      if (record && request === view.recordRequest && id === view.record?.id) await acceptRecord(record, false, request);
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.pollBusy = false; update(); schedulePoll(); } }
  }
  function open(id) { if (!id || window.IRISNavigation.open("tracking") === false) return false; loadRecord(id, true); return true; }
  field("source").addEventListener("change", () => { view.recordRequest++; view.record = null; field("job").hidden = true; error(null); const row = view.sources.find((item) => tools.sourceKey(item.source) === field("source").value); if (row) loadSource(row.source); else { view.sourceRequest++; clearSource(); update(); } });
  field("history").addEventListener("change", () => loadRecord(field("history").value, true));
  field("refresh").addEventListener("click", refresh);
  field("reference").addEventListener("change", () => loadReference(field("reference").value));
  for (const name of [...policyKeys, "name", "seconds", "identity", "iou"]) field(name).addEventListener("input", invalidate);
  field("clear-anchor").addEventListener("click", () => { view.selection = null; view.release = null; invalidate(); });
  field("set-release").addEventListener("click", () => { if (view.selection && view.position > tools.framePosition(view.source.sequence, view.selection.frame_id)) { view.release = currentFrame().frame_id; invalidate(); } });
  field("clear-release").addEventListener("click", () => { view.release = null; invalidate(); });
  field("previous").addEventListener("click", () => showFrame(view.position - 1)); field("next").addEventListener("click", () => showFrame(view.position + 1));
  field("scrub").addEventListener("input", () => showFrame(Number(field("scrub").value))); field("lane").addEventListener("change", draw);
  field("preview").addEventListener("click", () => {
    let payload; try { payload = configuration(); } catch (failure) { error(failure); return; }
    const version = view.formVersion;
    mutation(async (generation) => { const preview = await api("/api/temporal/tracking-selections/preview", { method: "POST", body: JSON.stringify(payload) }); if (generation !== view.generation || version !== view.formVersion || !view.visible) return; view.preview = preview; field("preview-result").hidden = false; field("preview-summary").textContent = `Two diagnostic policies · ${view.source.sequence.manifest.frames.length} available source frames · initial detection ${preview.request.selection.detection_index} · ${preview.request.release_frame_id ? "explicit release included" : "no explicit release"} · ${preview.request.evaluation ? "pinned human identity assessment" : "behavior only"} · ${preview.request.max_seconds} s wall limit.`; });
  });
  field("run").addEventListener("click", () => {
    if (!view.preview || view.history.some((row) => tools.active(row.job))) return;
    const payload = { ...view.preview.request, expected_fingerprint: view.preview.fingerprint };
    mutation(async (generation) => { const record = await api("/api/temporal/tracking-selections", { method: "POST", body: JSON.stringify(payload) }); if (generation !== view.generation || !view.visible) return; view.recordRequest++; view.preview = null; field("preview-result").hidden = true; await acceptRecord(record, true); if (typeof refreshJobs === "function") refreshJobs().catch(() => {}); });
  });
  field("cancel").addEventListener("click", () => mutation(async (generation) => { const record = view.record; if (!tools.active(record?.job)) return; await api(`/api/jobs/${safe(record.job.id)}/cancel`, { method: "POST" }); if (generation !== view.generation || view.record !== record || !view.visible) return; record.job.status = "cancelling"; renderJob(); }));
  window.addEventListener("iris:tracking-selection-open", (event) => open(event.detail.selection_id));
  window.addEventListener("iris:workspace", (event) => {
    view.visible = event.detail.name === "tracking";
    if (view.visible) { if (view.source && !view.imagesReady) showFrame(view.position); if (!view.loaded) refresh(); else schedulePoll(); }
    else { clearTimeout(view.timer); view.timer = null; view.generation++; view.recordRequest++; view.sourceRequest++; view.referenceRequest++; view.imageRequest++; view.loading = false; view.sourceLoading = false; view.busy = false; view.pollBusy = false; view.loaded = false; }
  });
  window.addEventListener("iris:project-initialized", () => { reset(); if (requestedID) open(requestedID); else if (view.visible) refresh(); });
  window.IRISTrackingSelection = Object.freeze({ open }); update();
})();
