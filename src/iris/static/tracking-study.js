"use strict";

(() => {
  const tools = window.IRISTrackingStudyTools, quality = window.IRISTrackingQualityTools;
  const safe = encodeURIComponent, field = (name) => $(`#study-${name}`);
  const requestedStudy = new URL(window.location.href).searchParams.get("tracking_study");
  const view = { visible: false, generation: 0, formVersion: 0, recordRequest: 0, openRequest: 0,
    loading: false, busy: false, pollBusy: false, timer: null, loaded: false,
    catalogue: { datasets: [], sequences: [] }, status: null, history: [], record: null,
    sourceChoices: new Map(), baseline: null, candidates: [], preview: null, renderedID: null };
  const dataset = () => view.catalogue.datasets.find((item) => item.id === field("dataset").value);
  const entries = () => dataset()?.entries || [];
  const sequence = (id) => view.catalogue.sequences.find((item) => item.id === id);
  const selectedComparison = (id) => sequence(id)?.comparisons.find((item) => item.id === view.sourceChoices.get(id));
  const activeEntries = () => entries().filter((item) => item.split !== "test");
  const hasActive = () => view.history.some((item) => tools.active(item.job));
  function error(failure) { field("error").textContent = failure?.message || failure || ""; field("error").hidden = !failure; }
  function options(select, rows, label, placeholder, selected = "") {
    select.replaceChildren(new Option(placeholder, ""));
    for (const row of rows) select.append(new Option(label(row), row.id));
    if (rows.some((row) => row.id === selected)) select.value = selected;
  }
  function invalidate() { view.formVersion++; view.preview = null; field("preview-result").hidden = true; update(); }
  function mapping() {
    return Object.fromEntries([...field("mapping").querySelectorAll("select")].map((select) => [select.dataset.nativeId, select.value === "__ignore__" ? null : select.value]));
  }
  function configuration() {
    return tools.configuration({ name: field("name").value, dataset_id: field("dataset").value,
      sources: activeEntries().map((entry) => ({ sequence_id: entry.sequence_id, comparison_id: view.sourceChoices.get(entry.sequence_id) || "" })),
      baseline: view.baseline, candidates: view.candidates.filter((item) => item.included).map(({ name, profile }) => ({ name, profile })),
      class_mapping: mapping(), iou_threshold: Number(field("iou").value), repeats: Number(field("repeats").value), max_updates: Number(field("updates").value), max_seconds: Number(field("seconds").value) });
  }
  function update() {
    const blocked = view.loading || view.busy;
    for (const control of field("prepare").querySelectorAll("input,select,button")) control.disabled = blocked;
    for (const row of field("candidates").querySelectorAll("[data-derived=true]")) row.disabled = true;
    field("refresh").disabled = blocked;
    field("history").disabled = blocked || !view.history.length;
    field("suggest").disabled = blocked || !view.baseline;
    field("create-dataset").disabled = blocked || !view.catalogue.sequences.length;
    field("cancel").disabled = blocked || !tools.active(view.record?.job) || view.record.job.status === "cancelling";
    let valid = false, hint;
    try { configuration(); valid = true; hint = "Review the source roles and candidate settings, then preview the exact work budget."; }
    catch (failure) { hint = failure.message; }
    if (hasActive()) { valid = false; hint = "A study is active in this project. Inspect or cancel it in history before starting another."; }
    if (view.status?.runtime?.available === false) { valid = false; hint = "Tracker runtime is unavailable. Saved studies remain readable; inspect runtime details above."; }
    field("prepare-status").textContent = view.loading ? "Loading source catalogue…" : view.busy ? "Processing the explicit request…" : hint;
    field("preview").disabled = blocked || !valid;
    field("run").disabled = blocked || !valid || !view.preview;
    $("#tracking-study-panel").setAttribute("aria-busy", String(blocked));
  }
  function reset() {
    clearTimeout(view.timer); view.timer = null; view.generation++; view.recordRequest++; view.openRequest++;
    view.loading = false; view.busy = false; view.pollBusy = false; view.loaded = false;
    view.catalogue = { datasets: [], sequences: [] }; view.history = []; view.record = null; view.baseline = null; view.candidates = []; view.preview = null; view.renderedID = null;
    view.sourceChoices.clear(); field("job").hidden = true; field("result").hidden = true; field("preview-result").hidden = true;
    for (const name of ["sources", "candidates", "mapping", "dataset-entries"]) field(name).replaceChildren();
    for (const name of ["dataset", "history", "baseline"]) options(field(name), [], () => "", "Choose saved evidence");
    field("dataset-summary").textContent = ""; field("baseline-summary").textContent = ""; error(null); update();
  }
  function renderDatasetCreation() {
    const previous = new Map([...field("dataset-entries").querySelectorAll("[data-sequence]")].map((row) => [row.dataset.sequence, {
      selected: row.querySelector("input").checked, split: row.querySelector("[data-role]").value, reference: row.querySelector("[data-reference]").value,
    }]));
    field("dataset-entries").replaceChildren();
    for (const source of view.catalogue.sequences) {
      const prior = previous.get(source.id), row = node("div", "study-source"); row.dataset.sequence = source.id;
      const selectLabel = node("label", "study-select"), check = node("input"); check.type = "checkbox"; check.checked = Boolean(prior?.selected);
      selectLabel.append(check, document.createTextNode(`${source.name} · ${source.frame_count} available frames`)); row.append(selectLabel);
      const grid = node("div", "tracking-fields"), split = node("select"), reference = node("select");
      split.dataset.role = "true"; split.id = `study-role-${source.id}`;
      for (const value of ["train", "val", "test"]) split.append(new Option(tools.splitLabel(value), value));
      split.value = prior?.split || "train";
      reference.dataset.reference = "true"; reference.id = `study-reference-${source.id}`;
      options(reference, source.references || [], quality.referenceLabel, "Choose a saved human reference", prior?.reference);
      for (const [text, control] of [["Sequence role", split], ["Pinned reference revision", reference]]) { const group = node("div"), label = node("label", "", text); label.htmlFor = control.id; group.append(label, control); grid.append(group); }
      row.append(grid); field("dataset-entries").append(row);
    }
    if (!view.catalogue.sequences.length) field("dataset-entries").append(node("p", "field-hint", "No frozen sequences yet. Freeze source frames above, then save a reviewed identity reference."));
  }
  function renderSources() {
    field("sources").replaceChildren(); const selected = dataset();
    field("dataset-summary").textContent = selected ? `${selected.name} · ${entries().length} frozen sequence(s) · references and data roles stay pinned.` : "Choose an existing frozen dataset or create one below.";
    for (const entry of entries()) {
      const source = sequence(entry.sequence_id), row = node("article", "study-source"), reserved = entry.split === "test";
      row.append(node("h4", "", `${source?.name || entry.sequence_id} · ${tools.splitLabel(entry.split)}`));
      const reference = source?.references.find((item) => item.id === entry.reference_id);
      row.append(node("p", "field-hint", reserved ? "Reserved metadata only: this study never loads this sequence into a tracker or calculates its quality." : `${source?.frame_count ?? "Unknown"} available frames · ${reference ? quality.referenceLabel(reference) : "No pinned reference available"}`));
      if (!reserved) {
        const label = node("label", "", "Completed source comparison"), select = node("select"); select.id = `study-comparison-${entry.sequence_id}`; label.htmlFor = select.id;
        options(select, source?.comparisons || [], (item) => item.name, "Choose a completed comparison", view.sourceChoices.get(entry.sequence_id));
        select.addEventListener("change", () => { view.sourceChoices.set(entry.sequence_id, select.value); renderBaseline(); invalidate(); });
        row.append(label, select);
      }
      field("sources").append(row);
    }
    renderBaseline();
  }
  function renderBaseline() {
    const first = activeEntries()[0], comparison = first && selectedComparison(first.sequence_id), previous = field("baseline").value;
    const rows = (comparison?.profiles || []).map((profile, index) => ({ id: String(index), profile }));
    options(field("baseline"), rows, (row) => `${row.profile.algorithm === "bytetrack" ? "ByteTrack" : "BoT-SORT"} · lane ${Number(row.id) + 1}`, "Choose a frozen baseline lane", rows.some((row) => row.id === previous) ? previous : rows[0]?.id);
    selectBaseline();
  }
  function selectBaseline() {
    const first = activeEntries()[0], comparison = first && selectedComparison(first.sequence_id), selected = field("baseline").value;
    const profile = selected !== "" ? comparison?.profiles[Number(selected)] : null;
    const next = profile ? { name: `${profile.algorithm === "bytetrack" ? "ByteTrack" : "BoT-SORT"} baseline`, profile: tools.clone(profile) } : null;
    if (tools.canonical(next) !== tools.canonical(view.baseline)) { view.candidates = []; field("candidates").replaceChildren(); }
    view.baseline = next;
    field("baseline-summary").textContent = profile ? `Frozen baseline: high ${profile.high_threshold}, low ${profile.low_threshold}, birth ${profile.new_track_threshold}, association ${profile.match_threshold}, lost buffer ${profile.buffer_updates} updates, score fusion ${profile.fuse_score ? "on" : "off"}, camera motion ${profile.gmc_method}. Classes ${profile.class_ids.join(", ")}. No learned ReID. One update per available frame.` : "Select a completed source comparison to reuse one of its saved profiles.";
    renderMapping();
  }
  function renderMapping() {
    const previous = mapping(); field("mapping").replaceChildren();
    if (!view.baseline) return;
    const first = activeEntries()[0], source = sequence(first.sequence_id), comparison = selectedComparison(first.sequence_id), taxonomy = source.taxonomy;
    if (!taxonomy?.classes) return;
    const suggested = quality.suggestions(comparison.detector, taxonomy, view.baseline.profile.class_ids);
    for (const id of view.baseline.profile.class_ids) {
      const name = comparison.detector?.classes.find((item) => item.id === id)?.name || "Native class";
      const row = node("div"), label = node("label", "", `${name} · detector ${id}`), select = node("select"); select.id = `study-map-${id}`; select.dataset.nativeId = String(id); label.htmlFor = select.id;
      select.append(new Option("Choose a taxonomy class or ignore", ""), new Option("Ignore this detector class", "__ignore__"));
      for (const item of taxonomy.classes) select.append(new Option(`${item.name || item.id} · ${item.id}`, item.id));
      if (Object.hasOwn(previous, String(id))) select.value = previous[id] === null ? "__ignore__" : previous[id];
      else if (Object.hasOwn(suggested, String(id))) select.value = suggested[id];
      select.addEventListener("change", invalidate); row.append(label, select); field("mapping").append(row);
    }
  }
  function candidateControl(card, index, key, labelText, kind, values = null) {
    const candidate = view.candidates[index], group = node("div"), label = node("label", "", labelText), control = node(kind === "select" ? "select" : "input");
    control.id = `study-candidate-${index}-${key}`; label.htmlFor = control.id;
    if (kind === "select") for (const [value, label] of values) control.append(new Option(label, value));
    else { control.type = kind; if (kind === "number") { control.min = "0"; control.max = key === "buffer_updates" ? "10000" : "1"; control.step = key === "buffer_updates" ? "1" : "any"; } }
    control.value = String(key === "name" ? candidate.name : candidate.profile[key]);
    if (key === "name") control.maxLength = 80;
    if (candidate.profile.algorithm === "bytetrack" && ["low_threshold", "new_track_threshold", "gmc_method"].includes(key)) { control.dataset.derived = "true"; control.disabled = true; }
    control.addEventListener(kind === "select" ? "change" : "input", () => {
      if (key === "name") candidate.name = control.value;
      else {
        const value = kind === "number" ? Number(control.value) : key === "fuse_score" ? control.value === "true" : control.value;
        candidate.profile = tools.editProfile(candidate.profile, key, value);
        if (key === "algorithm") renderCandidates();
        else if (key === "high_threshold" && candidate.profile.algorithm === "bytetrack") $(`#study-candidate-${index}-new_track_threshold`).value = String(candidate.profile.new_track_threshold);
      }
      invalidate();
    });
    group.append(label, control); card.append(group);
  }
  function renderCandidates() {
    field("candidates").replaceChildren();
    view.candidates.forEach((candidate, index) => {
      const card = node("article", "study-profile"), heading = node("div", "study-profile-name"), check = node("input"), label = node("label", "", `Include candidate ${index + 1}`);
      card.setAttribute("aria-disabled", String(!candidate.included)); check.type = "checkbox"; check.id = `study-include-${index}`; check.checked = candidate.included; label.htmlFor = check.id;
      check.addEventListener("change", () => { candidate.included = check.checked; card.setAttribute("aria-disabled", String(!candidate.included)); invalidate(); }); heading.append(check, label); card.append(heading);
      const grid = node("div", "tracking-fields");
      candidateControl(grid, index, "name", "Candidate name", "text");
      candidateControl(grid, index, "algorithm", "Tracker", "select", [["bytetrack", "ByteTrack"], ["botsort", "BoT-SORT"]]);
      for (const [key, label] of [["high_threshold", "High confidence threshold"], ["low_threshold", "Low confidence threshold"], ["new_track_threshold", "New-track threshold"], ["match_threshold", "Association threshold"], ["buffer_updates", "Lost buffer · analyzed updates"]]) candidateControl(grid, index, key, label, "number");
      candidateControl(grid, index, "fuse_score", "Score fusion", "select", [["true", "On"], ["false", "Off"]]);
      candidateControl(grid, index, "gmc_method", "Camera compensation", "select", [["none", "None"], ["sparseOptFlow", "Sparse optical flow"]]);
      card.append(grid, node("p", "field-hint", `Classes ${candidate.profile.class_ids.join(", ")} · seed ${candidate.profile.seed} · ${candidate.profile.opencv_threads} OpenCV thread(s) · GMC downscale ${candidate.profile.gmc_downscale} · no learned ReID. ByteTrack fixes low at 0.1 and birth at high + 0.1.`));
      field("candidates").append(card);
    }); update();
  }
  function renderHistory() { options(field("history"), view.history, (record) => `${record.name} · ${record.job?.status || "saved"}`, view.history.length ? "Choose a saved study" : "No saved studies", view.record?.id); }
  async function refresh() {
    if (view.loading || view.busy) return;
    const generation = view.generation, selectedDataset = field("dataset").value;
    view.loading = true; error(null); update();
    try {
      const [catalogue, status, history] = await Promise.all([api("/api/temporal/tracking-study-sources"), api("/api/temporal/tracking-study-status"), api("/api/temporal/tracking-studies")]);
      if (generation !== view.generation || !view.visible) return;
      view.catalogue = catalogue; view.status = status; view.history = history; view.loaded = true;
      options(field("dataset"), catalogue.datasets, (item) => item.name, catalogue.datasets.length ? "Choose a frozen temporal dataset" : "Create a frozen temporal dataset below", selectedDataset);
      renderDatasetCreation(); renderSources(); renderHistory();
      const runtime = status.runtime;
      field("readiness").textContent = `${catalogue.datasets.length} frozen dataset(s) · ${catalogue.sequences.length} source sequence(s). ${runtime?.available === false ? runtime.installation || "Tracker runtime unavailable." : "Bounded local tracker replays use cached detections."} Maximum 4 active sequences, 500 frames each and 7 candidates plus the baseline.`;
      invalidate();
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.loading = false; update(); schedulePoll(); } }
  }
  async function mutation(action) {
    if (view.busy || view.loading) return;
    const generation = view.generation; view.busy = true; error(null); update();
    try { await action(generation); }
    catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.busy = false; update(); schedulePoll(); } }
  }
  function renderJob() {
    const job = view.record?.job; field("job").hidden = !job;
    if (!job) return;
    field("job-status").textContent = [view.record.name, job.status?.replaceAll("_", " "), job.message, job.error].filter(Boolean).join(" · ");
    field("progress").hidden = !tools.active(job);
    if (typeof job.progress === "number") field("progress").value = job.progress <= 1 ? job.progress * 100 : job.progress;
    else field("progress").removeAttribute("value");
    field("cancel").hidden = !tools.active(job);
  }
  function acceptRecord(record, scroll = false) {
    const index = view.history.findIndex((item) => item.id === record.id);
    if (index < 0) view.history.unshift(record); else view.history[index] = record;
    view.record = record; renderHistory(); renderJob();
    if (record.job?.status === "succeeded" && record.report?.complete === true) {
      if (view.renderedID !== record.id) { renderReport(record); view.renderedID = record.id; }
      if (scroll) field("result").scrollIntoView({ block: "start" });
    } else { view.renderedID = null; field("result").hidden = true; if (scroll) field("job").scrollIntoView({ block: "start" }); }
    update();
  }
  async function loadRecord(id, scroll = false) {
    const generation = view.generation, request = ++view.recordRequest;
    view.record = null; view.renderedID = null; field("result").hidden = true; field("job").hidden = true; update();
    if (!id) return;
    try {
      const record = await api(`/api/temporal/tracking-studies/${safe(id)}`);
      if (generation !== view.generation || request !== view.recordRequest || !view.visible) return;
      acceptRecord(record, scroll);
    } catch (failure) { if (generation === view.generation && request === view.recordRequest) error(failure); }
    update(); schedulePoll();
  }
  function table(container, caption, headings, rows) {
    const wrapper = node("div", "study-table-scroll"), result = node("table", "study-table"), head = node("thead"), header = node("tr"), body = node("tbody");
    wrapper.tabIndex = 0; wrapper.setAttribute("role", "region"); wrapper.setAttribute("aria-label", caption); result.append(node("caption", "", caption));
    headings.forEach((heading) => { const cell = node("th", "", heading); cell.scope = "col"; header.append(cell); }); head.append(header);
    rows.forEach((row) => { const tr = node("tr"); row.forEach((value, index) => { const cell = node(index ? "td" : "th", "", String(value ?? "Unavailable")); if (!index) cell.scope = "row"; tr.append(cell); }); body.append(tr); });
    result.append(head, body); wrapper.append(result); container.append(wrapper);
  }
  function renderReport(record) {
    const report = record.report, summary = report.summary;
    field("result").hidden = false; field("result-title").textContent = record.name;
    field("result-scope").textContent = `Saved study ${record.id} · dataset ${record.dataset_id}. Preparation changes apply only to a future study. Repeated timings do not increase the reference sample size.`;
    const reasons = { development_only: "This dataset has no validation take; development results cannot establish improvement on another take.", manual_review_required: "The observed gains need a manual decision and independent testing.", no_verified_gain: "The validation evidence does not establish a gain over the baseline." };
    field("decision").textContent = `Baseline retained. No profile was applied. ${reasons[summary.decision?.reason] || "Review the per-split evidence before deciding on another profile."}`;
    const train = summary.splits?.train, val = summary.splits?.val;
    field("coverage").textContent = `${train?.sequence_count || 0} development sequence(s), ${train?.evaluated_frames || 0}/${train?.available_frames || 0} available frames evaluated · ${val?.sequence_count || 0} validation sequence(s), ${val?.evaluated_frames || 0}/${val?.available_frames || 0} frames evaluated. Reserved test data was not evaluated.${!val?.sequence_count ? " Development-only evidence cannot establish improvement on another take." : " Validation also informs tuning; it is not an independent test."}`;
    const assisted = summary.source_results?.some((source) => source.coverage?.reference_origin);
    field("origin").hidden = !assisted;
    field("origin").textContent = "At least one reference began with tracker proposals. Human review is recorded, but this does not establish an independent reference.";
    field("results").replaceChildren(); field("timing").replaceChildren();
    for (const split of ["train", "val"]) {
      const evidence = summary.splits?.[split];
      if (!evidence?.profiles?.length) { field("results").append(node("p", "field-hint", `${tools.splitLabel(split)}: no evaluated sequences in this role.`)); continue; }
      table(field("results"), `${tools.splitLabel(split)} · ${evidence.sequence_count} source sequence(s)`, ["Profile", "Evidence", "Precision", "Recall", "IDF1", "False positives", "Missed boxes", "ID switches", "Fragments", "Tracker median", "Tracker p95", "Median vs baseline"], evidence.profiles.map((row, index) => {
        const conclusion = summary.comparisons?.find((item) => item.profile_index === (row.profile_index ?? index));
        const status = conclusion?.by_split?.[split];
        const delta = conclusion?.cost_delta_by_split?.[split]?.tracker_median_ms;
        return [row.name || conclusion?.name || `Profile ${index + 1}`, index === 0 ? "Baseline" : tools.statusLabel(typeof status === "string" ? status : status?.status), tools.percentage(row.precision), tools.percentage(row.recall), tools.percentage(row.identity?.idf1), row.counts?.false_positives, row.counts?.false_negatives, row.counts?.identity_switches, row.counts?.fragments, tools.milliseconds(row.timing?.tracker_ms?.median), tools.milliseconds(row.timing?.tracker_ms?.p95), index === 0 ? "Baseline" : typeof delta === "number" ? `${delta > 0 ? "+" : ""}${tools.milliseconds(delta)}` : "Unavailable"];
      }));
      table(field("timing"), `${tools.splitLabel(split)} · timing scopes`, ["Profile", "Tracker samples", "GMC median", "Association median", "Source read median", "Setup median", "Replay wall median", "Repeatability"], evidence.profiles.map((row) => [row.name, row.timing?.tracker_ms?.count, tools.milliseconds(row.timing?.gmc_ms?.median), tools.milliseconds(row.timing?.association_ms?.median), tools.milliseconds(row.timing?.image_read_ms?.median), tools.milliseconds(row.timing?.setup_ms?.median), tools.milliseconds(row.timing?.replay_wall_ms?.median), typeof row.repeatability === "string" ? row.repeatability.replaceAll("_", " ") : row.repeatability === true ? "Repeated outputs match" : row.repeatability?.status?.replaceAll("_", " ") || "Inspect saved evidence"]));
    }
    field("source-evidence").replaceChildren();
    const sources = summary.source_results || [];
    for (const source of sources) {
      const row = node("article", "study-source"), coverage = source.coverage || source.quality?.coverage;
      row.append(node("h4", "", `${sequence(source.sequence_id)?.name || source.sequence_id} · ${tools.splitLabel(source.split)}`));
      if (coverage) {
        row.append(node("p", "field-hint", `${coverage.evaluated_frames ?? "Unknown"}/${coverage.available_frames ?? "Unknown"} available frames evaluated · ${coverage.dense ? "dense" : "partial or sparse"} reference coverage.`));
        if (coverage.reference_origin) row.append(node("p", "tracking-warning", "This human reference began with tracker proposals. Human review does not establish an independent reference."));
      }
      field("source-evidence").append(row);
    }
    field("limitations").replaceChildren();
    for (const limitation of summary.limitations || report.limitations || []) field("limitations").append(node("li", "", limitation));
    field("provenance").textContent = JSON.stringify({ schema: report.schema, request: report.request, fingerprint: report.fingerprint, dataset: report.dataset, decision: summary.decision, sources: report.sources, source_results: sources, comparisons: summary.comparisons }, null, 2);
    const link = new URL(window.location.href); link.search = ""; link.searchParams.set("project", state.projectId); link.searchParams.set("tracking_study", record.id);
    field("permalink").href = `${link.pathname}${link.search}`; field("download").href = projectURL(`/api/temporal/tracking-studies/${safe(record.id)}/report`);
  }
  function schedulePoll() {
    clearTimeout(view.timer); view.timer = null;
    if (view.visible && hasActive() && !view.pollBusy && !view.busy && !view.loading) view.timer = setTimeout(poll, 1200);
  }
  async function poll() {
    if (!view.visible || view.pollBusy || view.busy || view.loading) return schedulePoll();
    const generation = view.generation, request = view.recordRequest, selected = view.record?.id;
    view.pollBusy = true;
    try {
      const [history, record] = await Promise.all([api("/api/temporal/tracking-studies"), selected ? api(`/api/temporal/tracking-studies/${safe(selected)}`) : Promise.resolve(null)]);
      if (generation !== view.generation || !view.visible) return;
      view.history = history; renderHistory();
      if (record && request === view.recordRequest && selected === view.record?.id) acceptRecord(record);
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.pollBusy = false; update(); schedulePoll(); } }
  }
  function openStudy(id) {
    if (!id || window.IRISNavigation.open("tracking") === false) return false;
    loadRecord(id, true); return true;
  }
  field("create-dataset").addEventListener("click", () => {
    let payload;
    try { payload = tools.datasetRequest(field("dataset-name").value, [...field("dataset-entries").querySelectorAll("[data-sequence]")].map((row) => ({ sequence_id: row.dataset.sequence, selected: row.querySelector("input").checked, split: row.querySelector("[data-role]").value, reference_id: row.querySelector("[data-reference]").value }))); }
    catch (failure) { error(failure); return; }
    mutation(async (generation) => {
      const created = await api("/api/temporal/datasets", { method: "POST", body: JSON.stringify(payload) });
      if (generation !== view.generation || !view.visible) return;
      view.catalogue.datasets.unshift({ ...created, entries: created.entries || created.manifest.entries });
      options(field("dataset"), view.catalogue.datasets, (item) => item.name, "Choose a frozen temporal dataset", created.id);
      view.sourceChoices.clear(); renderSources(); invalidate(); field("dataset-create").open = false;
      field("dataset-summary").textContent = `${created.name} saved. Choose each source comparison below to prepare the study.`;
    });
  });
  field("dataset").addEventListener("change", () => { view.sourceChoices.clear(); view.recordRequest++; view.record = null; field("result").hidden = true; field("job").hidden = true; renderSources(); invalidate(); });
  field("baseline").addEventListener("change", () => { selectBaseline(); invalidate(); });
  field("suggest").addEventListener("click", () => mutation(async (generation) => {
    if (!view.baseline) return;
    const version = view.formVersion;
    const response = await api("/api/temporal/tracking-studies/suggestions", { method: "POST", body: JSON.stringify({ baseline_profile: view.baseline.profile }) });
    if (generation !== view.generation || version !== view.formVersion || !view.visible) return;
    view.candidates = response.candidates.map((item) => ({ ...tools.clone(item), included: true })); renderCandidates(); invalidate();
  }));
  field("preview").addEventListener("click", () => {
    let payload; try { payload = configuration(); } catch (failure) { error(failure); return; }
    mutation(async (generation) => {
      const version = view.formVersion;
      const preview = await api("/api/temporal/tracking-studies/preview", { method: "POST", body: JSON.stringify(payload) });
      if (generation !== view.generation || version !== view.formVersion || !view.visible) return;
      view.preview = preview; field("preview-result").hidden = false;
      const budget = preview.budget;
      field("preview-summary").textContent = `${preview.request.candidates.length + 1} profiles × ${preview.request.repeats} repetition(s) · ${budget.required_updates ?? budget.total_updates ?? budget.tracker_updates ?? "See budget"} tracker updates · ${preview.request.max_seconds} s wall limit.`;
      field("preview-coverage").textContent = `${preview.coverage.active_sequence_count} active sequence(s) · ${preview.coverage.reserved_test_count} reserved test sequence(s), never replayed. ${preview.coverage.development_only ? "Development-only study: the baseline remains current and gains cannot establish performance on a new take." : "Development and validation stay separate; validation informs tuning."}`;
    });
  });
  field("run").addEventListener("click", () => {
    if (!view.preview || hasActive()) return;
    const payload = { ...view.preview.request, expected_fingerprint: view.preview.fingerprint };
    mutation(async (generation) => {
      const record = await api("/api/temporal/tracking-studies", { method: "POST", body: JSON.stringify(payload) });
      if (generation !== view.generation || !view.visible) return;
      view.recordRequest++; acceptRecord(record, true); view.preview = null; field("preview-result").hidden = true;
      if (typeof refreshJobs === "function") refreshJobs().catch(() => {});
    });
  });
  field("cancel").addEventListener("click", () => mutation(async (generation) => {
    const record = view.record;
    if (!tools.active(record?.job)) return;
    await api(`/api/jobs/${safe(record.job.id)}/cancel`, { method: "POST" });
    if (generation !== view.generation || record !== view.record || !view.visible) return;
    record.job.status = "cancelling"; renderJob();
  }));
  for (const name of ["name", "iou", "repeats", "updates", "seconds"]) field(name).addEventListener("input", invalidate);
  field("history").addEventListener("change", () => { error(null); loadRecord(field("history").value, true); });
  field("refresh").addEventListener("click", refresh);
  window.addEventListener("iris:tracking-study-open", (event) => openStudy(event.detail.study_id));
  window.addEventListener("iris:workspace", (event) => {
    view.visible = event.detail.name === "tracking";
    if (view.visible) { if (!view.loaded) refresh(); else schedulePoll(); }
    else { clearTimeout(view.timer); view.timer = null; view.generation++; view.recordRequest++; view.busy = false; view.loading = false; view.pollBusy = false; view.loaded = false; }
  });
  window.addEventListener("iris:project-initialized", () => { reset(); if (requestedStudy) openStudy(requestedStudy); else if (view.visible) refresh(); });
  window.IRISTrackingStudy = Object.freeze({ open: openStudy }); update();
})();
