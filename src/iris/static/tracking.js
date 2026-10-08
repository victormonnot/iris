"use strict";

(() => {
  const tools = window.IRISTrackingTools;
  const field = (name) => $(`#tracking-${name}`);
  const safe = encodeURIComponent;
  const activeJob = (job) => ["queued", "running", "cancelling"].includes(job?.status);
  const view = { visible: false, generation: 0, imageGeneration: 0, refreshRequest: 0, busy: false, loading: false,
    sequences: [], models: [], readiness: null, sequence: null, caches: [], cache: null,
    histories: [], comparison: null, report: null, position: 0, lanes: [], events: [],
    playing: false, timer: null, polling: null, pollBusy: false, imagesReady: false,
    requestedComparison: new URL(window.location.href).searchParams.get("tracking_comparison"), requestedSequence: null };
  function error(value) { field("error").textContent = value?.message || value || ""; field("error").hidden = !value; }
  function stopPlayback() { clearTimeout(view.timer); view.timer = null; view.playing = false; field("play").textContent = "Play"; field("play").disabled = !view.report || !view.imagesReady || view.report.sequence.frames.length < 2; }
  function invalidate() { stopPlayback(); view.generation++; view.imageGeneration++; view.imagesReady = false; return view.generation; }
  function resetReport() {
    stopPlayback(); view.imageGeneration++; view.report = null; view.imagesReady = false;
    view.lanes = []; view.events = []; field("lanes").replaceChildren(); field("events").replaceChildren(); field("viewer").hidden = true;
    window.dispatchEvent(new CustomEvent("iris:tracking-quality-context", { detail: null }));
  }
  function options(select, rows, label, placeholder, selected) {
    select.replaceChildren();
    if (!rows.length) select.append(new Option(placeholder, ""));
    for (const row of rows) select.append(new Option(label(row), row.id));
    if (rows.some((row) => row.id === selected)) select.value = selected;
  }
  function sourceSelection() {
    if (!state.sessionId) return { error: "Open a session and select saved video frames in Data intake." };
    if (state.collectionLoading || state.pendingSelections.size || state.bulkSelecting) return { error: "Wait for the session's saved selection to finish updating." };
    if (state.collectionError) return { error: state.collectionError };
    return tools.selectedSource(state.frames, state.assets);
  }
  function classes() { return [...field("classes").querySelectorAll("input:checked")].map((input) => Number(input.value)).sort((a, b) => a - b); }
  function update() {
    const blocked = view.busy || view.loading;
    const selection = sourceSelection();
    field("selection").textContent = selection.error || `${selection.frames.length} saved frames · ${selection.name} · ${selection.gap_count} omitted source frames will become unknown gaps.`;
    field("create-sequence").disabled = blocked || Boolean(selection.error) || !field("sequence-name").value.trim();
    field("refresh").disabled = blocked;
    field("review-identities").disabled = blocked || !view.sequence;
    for (const name of ["sequence", "cache", "history"]) field(name).disabled = blocked || !field(name).value;
    field("run-cache").disabled = blocked || !view.sequence || !field("model").value || !field("cache-name").value.trim();
    const count = view.sequence?.manifest?.frames?.length || 0;
    const floor = view.cache?.config?.detector?.min_score;
    const reason = !view.cache ? "Choose or calculate a saved detector cache." : view.cache.coverage.state !== "complete" ? "A complete detector cache is required. Partial results never become empty observations." : count > (view.readiness?.max_frames || 500) ? "This sequence exceeds the 500 available frame limit. Freeze a shorter selection." : floor > 0.1 ? "This cache's score floor exceeds the default tracker low threshold (0.1). Calculate another cache at floor 0.001." : !classes().length ? "Select at least one native detector class." : !view.readiness?.available ? "Tracker runtime is unavailable. Saved comparisons remain readable." : "Ready: both lanes will consume the same complete cache and selected classes.";
    field("launch-hint").textContent = reason;
    field("run").disabled = blocked || !view.cache || view.cache.coverage.state !== "complete" || !count || count > (view.readiness?.max_frames || 500) || floor > 0.1 || !classes().length || !view.readiness?.available || !field("name").value.trim();
    field("previous").disabled = !view.report || view.position === 0;
    field("next").disabled = !view.report || view.position >= view.report.sequence.frames.length - 1;
    field("play").disabled = !view.report || view.report.sequence.frames.length < 2 || (!view.imagesReady && !view.playing);
    $("#tracking-workspace").setAttribute("aria-busy", String(blocked));
  }
  function renderSequence() {
    const manifest = view.sequence?.manifest;
    field("sequence-summary").textContent = manifest ? `${manifest.frames.length} available frames · source frames ${manifest.clip.start_frame}–${manifest.clip.end_frame} · ${view.sequence.name}` : "Freeze a video sequence from selected saved frames, or select an existing sequence.";
    field("clock").textContent = manifest ? `${tools.clockLabel(manifest.clock)}. ${manifest.clock.provenance}` : "";
    field("gaps").hidden = !manifest?.gaps?.length;
    field("gaps").textContent = manifest?.gaps?.length ? `Source gaps: ${manifest.gaps.map((gap) => `${gap.start_frame}–${gap.end_frame} (${gap.reason})`).join("; ")}. Missing frames have no observations or tracker updates.` : "";
    update();
  }
  function renderClasses() {
    field("classes").replaceChildren();
    for (const item of view.cache?.config?.detector?.classes || []) {
      const label = node("label"), input = node("input"); input.type = "checkbox"; input.value = item.id; input.checked = true;
      input.addEventListener("change", update); label.append(input, document.createTextNode(`${item.name || item.label} · native ${item.id}`)); field("classes").append(label);
    }
    if (!field("classes").children.length) field("classes").append(node("p", "field-hint", "Choose a cache to read its frozen class definitions."));
  }
  function renderJob(job, kind) {
    const prefix = kind === "cache" ? "cache-" : "";
    field(`${prefix}job`).hidden = !job;
    if (!job) return;
    const status = [job.status?.replaceAll("_", " "), job.message, job.error].filter(Boolean).join(" · ");
    field(`${prefix}job-status`).textContent = status;
    const progress = field(`${prefix}progress`);
    if (typeof job.progress === "number") progress.value = job.progress <= 1 ? job.progress * 100 : job.progress;
    else progress.removeAttribute("value");
    progress.hidden = !activeJob(job);
    const cancel = field(kind === "cache" ? "cancel-cache" : "cancel");
    cancel.hidden = !activeJob(job); cancel.disabled = view.busy || job.status === "cancelling";
  }
  function renderCache() {
    const cache = view.cache, coverage = cache?.coverage;
    field("cache-summary").textContent = cache ? `${coverage.state} cache · ${coverage.completed_count}/${coverage.total_count} available frames saved · ${cache.config.detector.device.toUpperCase()} · ${cache.config.detector.inference.mode} image · score floor ${cache.config.detector.min_score}. ${coverage.state === "partial" ? "Use Task activity to inspect explicit continuation, or calculate a new cache." : ""}` : "No detector cache selected.";
    renderJob(cache?.attempts?.at(-1), "cache"); update();
  }
  function renderHistory() {
    options(field("history"), view.histories, (row) => `${row.name} · ${row.job?.status || "saved"}`, "No saved comparisons", view.comparison?.id);
    renderJob(view.comparison?.job); update();
  }
  async function loadComparison(id, generation = view.generation) {
    resetReport(); view.comparison = view.histories.find((row) => row.id === id) || null; renderHistory();
    let detail;
    try { detail = await api(`/api/temporal/tracking-comparisons/${safe(id)}`); }
    catch (failure) { if (generation === view.generation && view.visible) error(failure); return; }
    if (generation !== view.generation || !view.visible) return;
    if (detail.cache_id !== view.cache?.id || detail.sequence_id !== view.sequence?.id) return error("Saved comparison does not match the selected source and cache.");
    view.comparison = detail; field("history").value = detail.id; renderJob(detail.job);
    if (detail.report) renderReport(detail.report);
    update();
  }
  async function selectCache(id, preferredComparison = null) {
    const generation = invalidate(); resetReport(); view.comparison = null; view.histories = []; view.cache = view.caches.find((cache) => cache.id === id) || null;
    error(null); renderClasses(); renderCache(); renderHistory();
    if (!view.cache) return;
    let rows;
    try { rows = await api(`/api/temporal/detection-caches/${safe(id)}/tracking-comparisons`); }
    catch (failure) { if (generation === view.generation && view.visible) error(failure); return; }
    if (generation !== view.generation || !view.visible) return;
    view.histories = rows; renderHistory();
    const selected = rows.find((row) => row.id === preferredComparison) || rows[0];
    if (selected) await loadComparison(selected.id, generation);
  }
  async function selectSequence(id, preferredCache = null, preferredComparison = null) {
    const generation = invalidate(); resetReport(); view.sequence = view.sequences.find((item) => item.id === id) || null;
    error(null);
    view.caches = []; view.cache = null; view.comparison = null; view.histories = [];
    options(field("cache"), [], () => "", "No detector caches", null); renderClasses(); renderSequence(); renderCache(); renderHistory();
    if (!view.sequence) return;
    field("sequence").value = id;
    let rows;
    try { rows = await api(`/api/temporal/sequences/${safe(id)}/detection-caches`); }
    catch (failure) { if (generation === view.generation && view.visible) error(failure); return; }
    if (generation !== view.generation || !view.visible) return;
    view.caches = rows;
    const selected = rows.find((row) => row.id === preferredCache) || rows.find((row) => row.coverage.state === "complete") || rows[0];
    options(field("cache"), rows, (row) => `${row.name} · ${row.coverage.state} · ${row.coverage.completed_count}/${row.coverage.total_count}`, "No detector caches", selected?.id);
    await selectCache(selected?.id, preferredComparison);
  }
  async function refresh() {
    if (!view.visible || view.busy) return;
    const generation = invalidate(), request = ++view.refreshRequest; resetReport(); view.loading = true; error(null); update();
    const requestedComparison = view.requestedComparison, requestedSequence = view.requestedSequence;
    view.requestedComparison = null; view.requestedSequence = null;
    const oldSequence = requestedSequence || view.sequence?.id, oldCache = view.cache?.id, oldComparison = view.comparison?.id;
    try {
      const results = await Promise.allSettled([api("/api/temporal/sequences"), api("/api/models"), api("/api/temporal/tracking-status")]);
      if (generation !== view.generation || !view.visible) return;
      if (results[0].status === "rejected") throw results[0].reason;
      view.sequences = results[0].value;
      view.models = results[1].status === "fulfilled" ? results[1].value : [];
      view.readiness = results[2].status === "fulfilled" ? results[2].value : null;
      const missing = Object.entries(view.readiness?.packages || {}).filter(([, item]) => !item.ready).map(([name, item]) => `${name} ${item.required}`);
      field("readiness").textContent = view.readiness?.available ? "Pinned tracker packages available. Execution is checked when the job runs." : results[2].status === "rejected" ? `Readiness unavailable: ${results[2].reason.message}` : `${view.readiness?.installation || "Install the optional tracking extra in the IRIS server environment"}. Required: ${missing.join(", ")}.`;
      const ready = view.models.filter((model) => model.status === "ready");
      options(field("model"), ready, (model) => model.name || model.id, results[1].status === "rejected" ? "Model catalog unavailable · refresh to retry" : "No ready local detector", field("model").value);
      let selected = view.sequences.find((row) => row.id === oldSequence) || view.sequences[0], cache = oldCache, comparison = oldComparison;
      if (requestedComparison) {
        const detail = await api(`/api/temporal/tracking-comparisons/${safe(requestedComparison)}`);
        if (generation !== view.generation || !view.visible) return;
        selected = view.sequences.find((row) => row.id === detail.sequence_id); cache = detail.cache_id; comparison = detail.id;
        if (!selected) throw new Error("This comparison's sequence is unavailable in the current project.");
      }
      options(field("sequence"), view.sequences, (row) => `${row.name} · ${row.manifest.frames.length} frames`, "No saved sequences", selected?.id);
      await selectSequence(selected?.id, cache, comparison);
    } catch (failure) { if (request === view.refreshRequest && generation === view.generation && view.visible) error(failure); }
    finally { if (request === view.refreshRequest) { view.loading = false; update(); } }
  }
  async function mutation(action) {
    if (view.busy || view.loading) return;
    view.busy = true; const generation = view.generation; error(null); update();
    try { await action(generation); } catch (failure) { if (generation === view.generation) error(failure); }
    finally { view.busy = false; update(); if (view.visible) schedulePoll(); }
  }
  async function poll() {
    if (!view.visible || view.pollBusy || view.busy || view.loading) return schedulePoll();
    const generation = view.generation, cache = view.cache, comparison = view.comparison;
    view.pollBusy = true;
    try {
      const requests = [];
      if (cache && activeJob(cache.attempts?.at(-1))) requests.push(["cache", api(`/api/temporal/detection-caches/${safe(cache.id)}`)]);
      if (comparison && activeJob(comparison.job)) requests.push(["comparison", api(`/api/temporal/tracking-comparisons/${safe(comparison.id)}`)]);
      const results = await Promise.allSettled(requests.map(([, promise]) => promise));
      if (generation !== view.generation || !view.visible) return;
      results.forEach((result, index) => {
        if (result.status === "rejected") { error(result.reason); return; }
        if (requests[index][0] === "cache") {
          view.cache = result.value; view.caches = view.caches.map((row) => row.id === view.cache.id ? view.cache : row);
          options(field("cache"), view.caches, (row) => `${row.name} · ${row.coverage.state} · ${row.coverage.completed_count}/${row.coverage.total_count}`, "No detector caches", view.cache.id); renderCache();
        } else {
          view.comparison = result.value; view.histories = view.histories.map((row) => row.id === view.comparison.id ? { ...view.comparison, report: null } : row); renderHistory();
          if (result.value.report && !view.report) renderReport(result.value.report);
        }
      });
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { view.pollBusy = false; schedulePoll(); }
  }
  function schedulePoll() { clearTimeout(view.polling); if (view.visible) view.polling = setTimeout(poll, 1800); }
  function svgNode(tag, attrs = {}, text) {
    const item = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, value);
    if (text !== undefined) item.textContent = text;
    return item;
  }
  function renderReport(report) {
    resetReport(); view.report = report; view.position = 0;
    field("viewer").hidden = false; field("report-title").textContent = view.comparison.name;
    field("scrub").max = report.sequence.frames.length - 1; field("scrub").value = "0";
    view.lanes = report.lanes.map((lane, index) => {
      const article = node("article", "tracking-lane"), header = node("header"), title = node("h3", "", lane.name);
      const review = node("button", "text-button", "Review identities"); review.type = "button";
      review.addEventListener("click", () => window.dispatchEvent(new CustomEvent("iris:temporal-identities-open", { detail: { sequence_id: view.sequence.id, comparison_id: view.comparison.id, lane_index: index } })));
      header.append(title, node("p", "", `${lane.report.profile.algorithm} · IDs local to this lane · GMC ${lane.report.profile.gmc_method}`), review);
      const stage = node("div", "tracking-stage"), summary = node("p", "tracking-lane-summary"), evidence = node("div", "tracking-lane-evidence");
      stage.setAttribute("aria-label", `${lane.name} source frame and tracking overlays`); article.append(header, stage, summary, evidence); field("lanes").append(article);
      const frames = lane.report.passes[0].frames;
      return { ...lane, index, frames, stage, summary, evidence, selected: null, svg: null, image: null };
    });
    view.events = view.lanes.flatMap((lane) => tools.events(lane.frames).map((event) => ({ ...event, lane: lane.index }))).sort((a, b) => a.position - b.position || a.lane - b.lane);
    field("provenance").textContent = JSON.stringify({ sequence_id: view.sequence.id, cache_id: report.cache_id, clock: report.sequence.clock, gaps: report.sequence.gaps,
      limitations: report.limitations, lanes: report.lanes.map((lane) => ({ name: lane.name, profile: lane.report.profile, profile_sha256: lane.report.profile_sha256,
        cache_fingerprint: lane.report.cache.fingerprint, result_sha256: lane.report.cache.result_sha256,
        repeatability: lane.report.repeatability, timing_scope: lane.report.timing_scope, replay_timing: lane.report.passes[0].timing, runtime: lane.report.passes[0].metadata })) }, null, 2);
    renderEvents(); showFrame(0);
    window.dispatchEvent(new CustomEvent("iris:tracking-quality-context", { detail: { sequence: view.sequence, comparison: view.comparison, report } }));
  }
  function renderEvents() {
    const filter = field("event-filter").value;
    const events = view.events.filter((event) => filter === "all" || event.kind === filter);
    field("events").replaceChildren();
    if (!events.length) field("events").append(node("p", "field-hint", "No matching recorded observation events. This does not establish correct identities."));
    for (const event of events) {
      const lane = view.lanes[event.lane], frame = lane.frames[event.position];
      const button = node("button", "tracking-event"); button.type = "button";
      button.setAttribute("aria-current", String(event.position === view.position));
      button.append(node("strong", "", `${lane.name} · source ${frame.frame_index} · ${tools.timeLabel(frame)}: `), document.createTextNode(event.text));
      button.addEventListener("click", () => { stopPlayback(); lane.selected = event.track_id ?? null; showFrame(event.position); }); field("events").append(button);
    }
  }
  function drawLane(lane) {
    lane.svg?.remove(); lane.svg = null; lane.evidence.replaceChildren();
    const frame = lane.frames[view.position], source = view.report.sequence.frames[view.position];
    if (!frame || frame.frame_id !== source.frame_id) return;
    const [width, height] = frame.input_size;
    const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, "aria-label": `${lane.name} observed and predicted boxes`, role: "img" });
    const fontSize = Math.max(10, width / 52);
    if (field("show-trails").checked) for (const trail of tools.trails(lane.frames, view.position)) {
      svg.append(svgNode("polyline", { points: trail.points.map((point) => point.join(",")).join(" "), class: `tracking-trail${trail.track_id === lane.selected ? " tracking-selected" : ""}` }));
    }
    function box(row, kind) {
      const [x1, y1, x2, y2] = row.box;
      const group = svgNode("g", row.track_id !== undefined ? { "data-track-id": row.track_id } : {});
      const selected = row.track_id !== undefined && row.track_id === lane.selected;
      group.append(svgNode("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: `tracking-box tracking-${kind}${selected ? " tracking-selected" : ""}` }));
      const label = kind === "prediction" ? `ID ${row.track_id} · predicted` : kind === "unassigned" ? `${row.label} · unassigned` : `ID ${row.track_id} · ${row.confirmed ? "observed" : "unconfirmed observation"}`;
      group.append(svgNode("text", { x: Math.max(0, x1 + 2), y: Math.max(fontSize + 2, y1 - 4), "font-size": fontSize }, label));
      group.append(svgNode("title", {}, kind === "prediction" ? `${label}; no fresh score; age ${row.age_updates} analyzed updates${row.age_seconds === null ? "; source time unknown" : `; ${row.age_seconds.toFixed(3)} source seconds`}` : `${label}; ${row.label}; score ${row.score.toFixed(3)}`));
      if (row.track_id !== undefined) group.addEventListener("click", () => { lane.selected = row.track_id; drawLane(lane); });
      svg.append(group);
    }
    const predictions = frame.predictions || [], unassigned = frame.unassigned || [], observations = frame.observations || [];
    if (field("show-predictions").checked) predictions.forEach((row) => box(row, "prediction"));
    const threshold = Number(field("display-score").value);
    if (field("show-unassigned").checked) unassigned.filter((row) => row.score >= threshold).forEach((row) => box(row, "unassigned"));
    observations.forEach((row) => box(row, "observation"));
    if (view.imagesReady) { lane.stage.append(svg); lane.svg = svg; }
    lane.summary.textContent = `${observations.length} observed (${observations.filter((row) => !row.confirmed).length} unconfirmed) · ${predictions.length} predicted · ${unassigned.length} unassigned · analyzed update ${frame.update_index}`;
    for (const row of observations) {
      const button = node("button", "tracking-evidence-button", `ID ${row.track_id} · ${row.label} · ${row.confirmed ? "observed" : "unconfirmed observation"} · detector score ${row.score.toFixed(3)} · detection ${row.detection_index}`);
      button.type = "button"; button.setAttribute("aria-pressed", String(lane.selected === row.track_id)); button.addEventListener("click", () => { lane.selected = row.track_id; drawLane(lane); }); lane.evidence.append(button);
    }
    for (const row of predictions) lane.evidence.append(node("p", "", `ID ${row.track_id} · prediction only · last observed at source ${row.last_observed_frame_index} · age ${row.age_updates} analyzed updates${row.age_seconds === null ? " · source seconds unknown" : ` · ${row.age_seconds.toFixed(3)} source seconds`} · no current detector score.`));
    for (const row of unassigned) lane.evidence.append(node("p", "", `Unassigned detection ${row.detection_index} · ${row.label} · score ${row.score.toFixed(3)} · ${row.reason.replaceAll("_", " ")}. Association requires inspection.`));
    if (!observations.length && !predictions.length && !unassigned.length) lane.evidence.append(node("p", "", "No saved detector observation or tracker prediction on this analyzed frame. This is not proof that the scene was empty."));
    lane.evidence.append(node("p", "field-hint", `GMC: ${frame.gmc.status || frame.gmc.method || "see saved evidence"}`));
  }
  function showFrame(position) {
    if (!view.report || !view.visible) return;
    clearTimeout(view.timer); view.timer = null;
    const generation = view.generation, imageGeneration = ++view.imageGeneration;
    view.position = Math.max(0, Math.min(position, view.report.sequence.frames.length - 1)); view.imagesReady = false;
    const frame = view.report.sequence.frames[view.position];
    field("scrub").value = view.position;
    field("position").textContent = `${view.position + 1}/${view.report.sequence.frames.length} · source ${frame.frame_index} · ${tools.timeLabel(frame)}`;
    const gap = tools.gapBefore(view.report.sequence.frames, view.position);
    field("frame-gap").hidden = !gap; field("frame-gap").textContent = gap ? `${gap} missing source frames before this image. No interpolation or synthetic tracker updates. The source-time jump remains part of playback.` : "";
    field("image-status").textContent = "Loading and verifying this source frame…";
    const url = projectURL(`/api/temporal/sequences/${safe(view.sequence.id)}/frames/${safe(frame.frame_id)}/image`);
    const loaded = view.lanes.map((lane) => {
      lane.stage.classList.remove("tracking-stage-loaded"); lane.stage.replaceChildren(); lane.svg = null; drawLane(lane);
      // Reserve the frozen image geometry while verification is in flight so a
      // deep-linked quality result below the replay does not jump after loading.
      lane.stage.style.aspectRatio = `${frame.width} / ${frame.height}`;
      lane.stage.style.minHeight = "0";
      const image = node("img"); lane.image = image; image.alt = `Source frame ${frame.frame_index}, ${lane.name}`;
      // Keep new images detached until both lanes load. Old pixels disappear immediately.
      return new Promise((resolve, reject) => {
        image.onload = () => image.naturalWidth === frame.width && image.naturalHeight === frame.height ? resolve(image) : reject(new Error("Saved source image dimensions do not match this sequence."));
        image.onerror = () => reject(new Error("Source frame unavailable or failed verification. No overlays are shown. Refresh to retry."));
        image.src = url;
      });
    });
    Promise.all(loaded).then((images) => {
      if (generation !== view.generation || imageGeneration !== view.imageGeneration || !view.visible) return;
      if (view.lanes.some((lane) => lane.frames[view.position]?.frame_id !== frame.frame_id)) throw new Error("Tracker output does not match this source frame.");
      view.imagesReady = true;
      images.forEach((image, index) => { view.lanes[index].stage.classList.add("tracking-stage-loaded"); view.lanes[index].stage.append(image); drawLane(view.lanes[index]); });
      field("image-status").textContent = "Both lanes show the same verified source image. Boxes use original pixel coordinates.";
      update(); schedulePlayback();
    }).catch((failure) => {
      if (generation !== view.generation || imageGeneration !== view.imageGeneration || !view.visible) return;
      view.imagesReady = false; stopPlayback();
      view.lanes.forEach((lane) => { lane.stage.classList.remove("tracking-stage-loaded"); lane.stage.replaceChildren(); lane.svg = null; drawLane(lane); lane.summary.textContent += " · image unavailable, overlays suppressed"; });
      field("image-status").textContent = failure.message;
      update();
    });
    renderEvents(); update();
  }
  function schedulePlayback() {
    clearTimeout(view.timer);
    if (!view.playing || !view.imagesReady || !view.visible) return;
    const frames = view.report.sequence.frames, next = frames[view.position + 1];
    if (!next) return stopPlayback();
    const delay = tools.playbackDelay(frames[view.position], next, view.report.sequence.clock, Number(field("speed").value));
    view.timer = setTimeout(() => showFrame(view.position + 1), Math.min(delay, 2147483647));
  }
  field("refresh").addEventListener("click", refresh);
  field("review-identities").addEventListener("click", () => window.dispatchEvent(new CustomEvent("iris:temporal-identities-open", { detail: { sequence_id: view.sequence.id } })));
  field("sequence").addEventListener("change", () => selectSequence(field("sequence").value).catch(error));
  field("cache").addEventListener("change", () => selectCache(field("cache").value).catch(error));
  field("history").addEventListener("change", () => { invalidate(); error(null); loadComparison(field("history").value).catch(error); });
  for (const name of ["sequence-name", "cache-name", "name", "model"]) field(name).addEventListener("input", update);
  field("mode").addEventListener("change", () => { field("tiles").hidden = field("mode").value !== "tiled"; });
  field("create-sequence").addEventListener("click", () => mutation(async (generation) => {
    const selected = sourceSelection(); if (selected.error) throw new Error(selected.error);
    const payload = { name: field("sequence-name").value.trim(), asset_id: selected.asset_id, frame_ids: selected.frames.map((frame) => frame.id) };
    if (field("take-group").value.trim()) payload.take_group = field("take-group").value.trim();
    const result = await api("/api/temporal/sequences", { method: "POST", body: JSON.stringify(payload) });
    if (generation !== view.generation || !view.visible) return;
    view.sequences.unshift(result); options(field("sequence"), view.sequences, (row) => `${row.name} · ${row.manifest.frames.length} frames`, "No saved sequences", result.id); await selectSequence(result.id);
  }));
  field("run-cache").addEventListener("click", () => mutation(async (generation) => {
    const payload = { name: field("cache-name").value.trim(), model_id: field("model").value, device: field("device").value, inference_mode: field("mode").value, min_score: 0.001, tile_size: Number(field("tile-size").value), overlap: Number(field("overlap").value) };
    if (!Number.isInteger(payload.tile_size) || payload.tile_size < 128 || payload.tile_size > 2048 || !Number.isFinite(payload.overlap) || payload.overlap < 0 || payload.overlap > 0.5) throw new Error("Tile size must be an integer from 128–2048 and overlap from 0–0.5.");
    const result = await api(`/api/temporal/sequences/${safe(view.sequence.id)}/detection-caches`, { method: "POST", body: JSON.stringify(payload) });
    if (generation !== view.generation || !view.visible) return;
    await selectSequence(view.sequence.id, result.id); if (result.reused) field("cache-summary").textContent += " Existing matching cache reused; no duplicate detector calculation was queued.";
  }));
  field("run").addEventListener("click", () => mutation(async (generation) => {
    const detail = await api(`/api/temporal/detection-caches/${safe(view.cache.id)}/tracking-comparisons`, { method: "POST", body: JSON.stringify({ name: field("name").value.trim(), class_ids: classes(), gmc_method: field("gmc").value }) });
    if (generation !== view.generation || !view.visible) return;
    resetReport(); view.comparison = detail; view.histories.unshift({ ...detail, report: null }); renderHistory();
    if (detail.report) renderReport(detail.report);
    if (typeof refreshJobs === "function") refreshJobs().catch(() => {});
  }));
  for (const [name, getJob] of [["cancel", () => view.comparison?.job], ["cancel-cache", () => view.cache?.attempts?.at(-1)]]) {
    field(name).addEventListener("click", () => mutation(async (generation) => {
      const job = getJob(); if (!activeJob(job)) return;
      await api(`/api/jobs/${safe(job.id)}/cancel`, { method: "POST" });
      if (generation === view.generation) { job.status = "cancelling"; renderCache(); renderJob(view.comparison?.job); }
    }));
  }
  field("scrub").addEventListener("input", () => { stopPlayback(); showFrame(Number(field("scrub").value)); });
  field("previous").addEventListener("click", () => { stopPlayback(); showFrame(view.position - 1); });
  field("next").addEventListener("click", () => { stopPlayback(); showFrame(view.position + 1); });
  field("play").addEventListener("click", () => {
    if (view.playing) return stopPlayback();
    view.playing = true; field("play").textContent = "Pause";
    if (view.position >= view.report.sequence.frames.length - 1) showFrame(0); else schedulePlayback();
  });
  field("speed").addEventListener("change", schedulePlayback);
  field("event-filter").addEventListener("change", renderEvents);
  for (const name of ["show-predictions", "show-trails", "show-unassigned", "display-score"]) field(name).addEventListener("input", () => view.lanes.forEach(drawLane));
  window.addEventListener("iris:frames", update);
  window.addEventListener("iris:session", () => { stopPlayback(); update(); });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); });
  window.addEventListener("iris:workspace", (event) => {
    view.visible = event.detail.name === "tracking";
    invalidate(); clearTimeout(view.polling);
    if (!view.visible) { view.refreshRequest++; view.loading = false; }
    if (view.visible) { refresh(); schedulePoll(); }
  });
  function openComparison(id) {
    view.requestedComparison = id;
    if (view.visible) { refresh(); return true; }
    return window.IRISNavigation.open("tracking");
  }
  window.addEventListener("iris:tracking-comparison-open", (event) => openComparison(event.detail.comparison_id));
  window.addEventListener("iris:tracking-sequence-open", (event) => {
    view.requestedSequence = event.detail.sequence_id;
    if (view.visible) refresh(); else window.IRISNavigation.open("tracking");
  });
  window.addEventListener("iris:tracking-show-source-frame", (event) => {
    if (!view.visible || event.detail.comparison_id !== view.comparison?.id) return;
    const position = view.report?.sequence.frames.findIndex((frame) => frame.frame_index === event.detail.frame_index) ?? -1;
    if (position < 0) return;
    stopPlayback(); showFrame(position); field("viewer").scrollIntoView({ block: "start", behavior: "smooth" });
  });
  window.addEventListener("iris:project-initialized", () => { if (view.requestedComparison) window.IRISNavigation.open("tracking"); });
  window.IRISTracking = Object.freeze({ open: openComparison });
  update();
})();
