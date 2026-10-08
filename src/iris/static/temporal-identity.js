"use strict";

// This workspace edits immutable temporal references, independently of image annotations.
(() => {
  const tools = window.IRISTemporalIdentityTools, playback = window.IRISTrackingTools;
  const field = (name) => $(`#identity-${name}`), safe = encodeURIComponent;
  const params = new URL(window.location.href).searchParams;
  const view = { visible: false, generation: 0, imageGeneration: 0, comparisonGeneration: 0,
    busy: false, loading: false, sequences: [], sequence: null, references: [], latest: null,
    draft: null, baseline: "", pending: {}, undo: [], redo: [], readonly: false, recordID: null,
    position: 0, selected: null, imagesReady: false, svg: null, drawing: false, gesture: null,
    playing: false, timer: null, comparisons: [], comparison: null, cache: null,
    discoveryReady: false, discoveryRequest: {},
    requested: params.get("temporal_sequence") ? { sequence_id: params.get("temporal_sequence"),
      comparison_id: params.get("temporal_comparison") || params.get("tracking_comparison"),
      lane_index: Number(params.get("temporal_lane") || 0) } : null };
  const stringify = (value) => JSON.stringify(value);
  const dirty = () => Boolean(view.draft && !view.readonly && (stringify(view.draft) !== view.baseline || Object.keys(view.pending).length));
  const source = () => view.sequence?.manifest.frames[view.position];
  const frame = () => view.draft?.frames.find((item) => item.frame_index === source()?.frame_index);
  const object = () => frame()?.objects[view.selected];
  const capped = () => (view.sequence?.manifest.frames.length || 0) > 500;
  const editable = () => Boolean(view.draft && !view.readonly && !capped() && view.imagesReady && !view.busy && !view.loading && !view.playing);
  const newID = () => `ref_${crypto.randomUUID().replaceAll("-", "")}`;
  const mismatchedConfirmations = () => (view.draft?.frames || []).some((current) => Object.hasOwn(view.pending, current.frame_index) && current.review.reviewer !== field("reviewer").value.trim());
  function error(value) { field("error").textContent = value?.message || value || ""; field("error").hidden = !value; }
  function stop() { clearTimeout(view.timer); view.timer = null; view.playing = false; field("play").textContent = "Play"; }
  function options(select, rows, label, empty, value) {
    select.replaceChildren();
    if (empty !== null) select.append(new Option(empty, ""));
    rows.forEach((row) => select.append(new Option(label(row), row.id)));
    if (rows.some((row) => String(row.id) === String(value))) select.value = value;
  }
  function shortID(id) { return id ? (id.length > 18 ? `${id.slice(0, 14)}…` : id) : "Unassigned"; }
  function className(label) { return view.sequence?.manifest.taxonomy.classes.find((item) => item.id === label)?.name || label; }
  function invalidateImage() {
    stop(); view.imageGeneration++; view.imagesReady = false; view.svg = null; view.gesture = null;
    field("stage").replaceChildren(); update();
  }
  function consentDiscard(message = "Discard unsaved temporal identity changes?") { return !dirty() || window.confirm(message); }
  function snapshot() { return { draft: tools.clone(view.draft), pending: tools.clone(view.pending), selected: view.selected }; }
  function restore(item) { view.draft = item.draft; view.pending = item.pending; view.selected = item.selected; render(); }
  function commit(next, confirmation = null) {
    if (!editable()) return;
    if (stringify(next) === stringify(view.draft) && !confirmation) return;
    view.undo.push(snapshot()); if (view.undo.length > 80) view.undo.shift(); view.redo = [];
    for (const index of tools.changedFrames(view.draft, next)) delete view.pending[index];
    view.draft = next;
    if (confirmation) view.pending[confirmation.frame_index] = confirmation.coverage;
    error(null); render();
  }
  function perform(action) {
    if (!editable()) return;
    try { action(); } catch (failure) { error(failure); renderInspector(); }
  }
  function update() {
    const blocked = view.busy || view.loading, canEdit = editable(), current = frame();
    field("refresh").disabled = blocked;
    field("sequence").disabled = blocked || !view.sequences.length;
    field("history").disabled = blocked || !view.references.length;
    field("reload").disabled = blocked || !view.sequence;
    field("undo").disabled = !canEdit || !view.undo.length;
    field("redo").disabled = !canEdit || !view.redo.length;
    field("previous").disabled = blocked || !view.draft || view.position === 0;
    field("next").disabled = blocked || !view.draft || view.position >= (view.sequence?.manifest.frames.length || 0) - 1;
    field("play").disabled = blocked || !view.draft || (!view.imagesReady && !view.playing) || (view.sequence?.manifest.frames.length || 0) < 2;
    field("scrub").disabled = blocked || !view.draft;
    field("draw").disabled = !canEdit; field("add-absence").disabled = !canEdit;
    field("draw").setAttribute("aria-pressed", String(view.drawing));
    field("draw").textContent = view.drawing ? "Cancel drawing" : "Draw a new box";
    field("stage").dataset.drawing = String(view.drawing);
    field("object-fields").disabled = !canEdit || !object();
    field("notes").disabled = !canEdit;
    field("reviewer").disabled = blocked || view.readonly;
    const reviewer = field("reviewer").value.trim();
    field("confirm-partial").disabled = !canEdit || !reviewer;
    field("confirm-complete").disabled = !canEdit || !reviewer;
    field("confirm-complete").textContent = current?.objects.length ? "Confirm identities and all objects on this frame" : "Confirm this frame is empty (complete human review)";
    field("save").disabled = !canEdit || !reviewer || !dirty() || mismatchedConfirmations();
    field("comparison").disabled = blocked || !view.comparisons.length;
    field("lane").disabled = blocked || !view.comparison?.report;
    const selects = [...field("mapping").querySelectorAll("select")];
    field("seed").disabled = blocked || !canEdit || Boolean(view.latest) || !view.comparison?.report || !selects.length || selects.some((select) => select.value === "");
    for (const select of selects) select.disabled = blocked || Boolean(view.latest);
    $("#identities-workspace").setAttribute("aria-busy", String(blocked));
  }
  function renderSummary() {
    if (!view.sequence || !view.draft) return;
    const manifest = view.sequence.manifest, summary = tools.summary(view.draft, manifest), latest = view.latest;
    const record = view.references.find((item) => item.id === view.recordID) || latest;
    field("sequence-summary").textContent = `${view.sequence.name} · ${manifest.frames.length} available source frames · ${playback.clockLabel(manifest.clock)}. ${manifest.clock.provenance}`;
    const saved = view.readonly ? `Read-only revision ${record?.revision}` : latest ? `Latest saved revision ${latest.revision}` : "New reference · no saved revision";
    field("saved").textContent = `${saved}${dirty() ? " · Unsaved draft" : " · No unsaved changes"}${record?.payload.provenance?.author ? ` · Author: ${record.payload.provenance.author}` : ""}`;
    field("saved").dataset.dirty = String(dirty());
    field("summary").textContent = `${summary.dense_human_reference ? "Dense human reference" : "Sparse reference"} · ${summary.human_complete_frames}/${summary.available_frames} complete human-reviewed frames · ${summary.partial_frames} partial · ${summary.unreviewed_frames} unreviewed · ${summary.identity_count} identities · ${summary.object_count} objects · ${summary.uncertain_objects} uncertain`;
    field("provenance").textContent = JSON.stringify({ sequence_id: view.draft.sequence_id, sequence_sha256: view.draft.sequence_sha256, reference_id: record?.id || null, saved_revision: record?.revision || null, payload_schema: view.draft.schema, provenance: view.draft.provenance || { author: "Legacy reference: see per-frame reviewer", origin: null } }, null, 2);
    const gaps = manifest.gaps || [];
    field("gaps").hidden = !gaps.length && !capped();
    field("gaps").textContent = capped() ? "This sequence exceeds the 500 available frame editor limit. Saved reference revisions and their summaries remain readable. Freeze a shorter sequence to edit." : `Unknown source gaps: ${gaps.map((gap) => `${gap.start_frame}–${gap.end_frame} (${gap.reason})`).join("; ")}. Gaps prevent a dense human reference; no identities or boxes are interpolated.`;
    field("pending").textContent = mismatchedConfirmations() ? "Restored confirmations belong to a different reviewer. Enter that reviewer or reconfirm each affected frame before saving." : `${Object.keys(view.pending).length} frame confirmation(s) pending save. Later edits clear confirmation on affected frames. Saving without confirmations keeps new and changed frames unreviewed.`;
  }
  function renderInspector() {
    const item = object(), identities = view.draft?.identities || [];
    field("object-empty").hidden = Boolean(item);
    options(field("object-class"), view.sequence?.manifest.taxonomy.classes || [], (row) => row.name || row.id, null, item?.label);
    options(field("object-id"), identities.filter((identity) => identity.label === item?.label), (identity) => identity.id, "Unassigned / unresolved", item?.identity_id);
    options(field("merge-target"), identities.filter((identity) => identity.label === item?.label && identity.id !== item?.identity_id), (identity) => identity.id, "Choose another identity", "");
    field("visibility").value = item?.visibility || "unknown";
    field("certainty").value = item?.certainty || "uncertain";
    ["x1", "y1", "x2", "y2"].forEach((name, index) => {
      field(name).value = item?.box?.[index] ?? "";
      field(name).disabled = !item || ["out_of_view", "unknown"].includes(item.visibility);
      field(name).max = index % 2 === 0 ? source()?.width || 0 : source()?.height || 0;
    });
    field("apply-box").disabled = !item || ["out_of_view", "unknown"].includes(item.visibility);
    field("clear-box").disabled = item?.visibility !== "occluded" || !item.box;
    field("split").disabled = !item?.identity_id;
    field("merge").disabled = !item?.identity_id;
    field("split-boundary").value = source()?.frame_index ?? 0;
    update();
  }
  function renderObjects() {
    const current = frame(); field("objects").replaceChildren();
    if (view.selected !== null && !current?.objects[view.selected]) view.selected = null;
    (current?.objects || []).forEach((item, index) => {
      const button = node("button", "identity-object", `${index + 1}. ${shortID(item.identity_id)} · ${className(item.label)} · ${item.visibility.replaceAll("_", " ")} · ${item.certainty}${item.box ? ` · [${item.box.map((coordinate) => Number(coordinate.toFixed(2))).join(", ")}]` : " · no box"}`);
      button.type = "button"; button.title = item.identity_id || "Unassigned";
      button.setAttribute("aria-pressed", String(view.selected === index));
      button.addEventListener("click", () => { stop(); view.selected = index; renderObjects(); renderInspector(); draw(); });
      field("objects").append(button);
    });
    if (!current?.objects.length) field("objects").append(node("p", "field-hint", "No reference objects on this frame. This is unreviewed until an explicit frame confirmation."));
    field("frame-review").textContent = `Source ${source()?.frame_index}: ${current?.coverage || "unreviewed"} · ${current?.review.status.replaceAll("_", " ") || "unreviewed"}${current?.review.reviewer ? ` · ${current.review.reviewer}` : ""}${Object.hasOwn(view.pending, source()?.frame_index) ? " · confirmation pending save" : ""}`;
  }
  function svgNode(tag, attributes = {}, text) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function draw() {
    view.svg?.remove(); view.svg = null;
    if (!view.imagesReady || !source()) return;
    const { width, height } = source(), svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Temporal reference boxes with draggable corners" });
    const size = Math.max(4, width / (95 * Number(field("zoom").value)));
    (frame()?.objects || []).forEach((item, index) => {
      if (!item.box) return;
      const [x1, y1, x2, y2] = item.box, group = svgNode("g", { "data-object": index });
      group.append(svgNode("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: `identity-box ${item.certainty}${index === view.selected ? " selected" : ""}` }));
      group.append(svgNode("text", { x: x1 + 2, y: Math.max(size * 1.8, y1 - 4), "font-size": size * 1.8 }, `${index + 1} · ${shortID(item.identity_id)}`));
      if (index === view.selected && editable()) for (const [corner, x, y] of [["nw", x1, y1], ["ne", x2, y1], ["sw", x1, y2], ["se", x2, y2]]) group.append(svgNode("rect", { x: x - size / 2, y: y - size / 2, width: size, height: size, class: "identity-handle", "data-handle": corner }));
      svg.append(group);
    });
    field("stage").append(svg); view.svg = svg;
    svg.addEventListener("pointerdown", pointerDown); svg.addEventListener("pointermove", pointerMove);
    svg.addEventListener("pointerup", pointerUp); svg.addEventListener("pointercancel", cancelGesture);
  }
  function render() {
    if (!view.draft) { update(); return; }
    field("notes").value = view.draft.notes;
    renderSummary(); renderObjects(); renderInspector(); draw();
  }
  function point(event) {
    const bounds = view.svg.getBoundingClientRect(), current = source();
    return [Math.max(0, Math.min(current.width, (event.clientX - bounds.left) * current.width / bounds.width)), Math.max(0, Math.min(current.height, (event.clientY - bounds.top) * current.height / bounds.height))];
  }
  function pointerDown(event) {
    if (!editable() || event.button !== 0) return;
    const group = event.target.closest("[data-object]"), start = point(event);
    if (!view.drawing && !group) return;
    const index = group ? Number(group.dataset.object) : null;
    if (!view.drawing) view.selected = index;
    view.gesture = { start, index, kind: view.drawing ? "draw" : event.target.dataset.handle || "move", original: index === null ? null : [...frame().objects[index].box], box: null, pointer: event.pointerId, imageGeneration: view.imageGeneration };
    view.svg.setPointerCapture(event.pointerId); event.preventDefault();
    renderObjects(); renderInspector();
  }
  function pointerMove(event) {
    const gesture = view.gesture;
    if (!gesture || gesture.pointer !== event.pointerId || gesture.imageGeneration !== view.imageGeneration || !editable()) return;
    const current = point(event), [startX, startY] = gesture.start, dimensions = source(); let box;
    if (gesture.kind === "draw") box = [Math.min(startX, current[0]), Math.min(startY, current[1]), Math.max(startX, current[0]), Math.max(startY, current[1])];
    else if (gesture.kind === "move") {
      const [x1, y1, x2, y2] = gesture.original;
      const dx = Math.max(-x1, Math.min(dimensions.width - x2, current[0] - startX)), dy = Math.max(-y1, Math.min(dimensions.height - y2, current[1] - startY));
      box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy];
    } else {
      box = [...gesture.original]; box[gesture.kind.includes("w") ? 0 : 2] = current[0]; box[gesture.kind.includes("n") ? 1 : 3] = current[1];
    }
    gesture.box = box.map((coordinate) => Math.round(coordinate * 100) / 100);
    view.svg.querySelector(".identity-preview")?.remove();
    view.svg.append(svgNode("rect", { class: "identity-preview", x: box[0], y: box[1], width: Math.max(0, box[2] - box[0]), height: Math.max(0, box[3] - box[1]) }));
  }
  function cancelGesture() { view.gesture = null; draw(); }
  function pointerUp(event) {
    const gesture = view.gesture; if (!gesture || gesture.pointer !== event.pointerId) return;
    view.gesture = null;
    if (!gesture.box || gesture.imageGeneration !== view.imageGeneration) return draw();
    perform(() => {
      validateBox(gesture.box);
      if (gesture.kind === "draw") addObject("visible", gesture.box);
      else commit(tools.editObject(view.draft, source().frame_index, gesture.index, { box: gesture.box }));
    });
    view.drawing = false; draw(); update();
  }
  function validateBox(box) {
    const current = source();
    if (box.some((value) => !Number.isFinite(value)) || !(0 <= box[0] && box[0] < box[2] && box[2] <= current.width && 0 <= box[1] && box[1] < box[3] && box[3] <= current.height)) throw new Error("Box coordinates must have positive area within this source image.");
  }
  function addObject(visibility, box) {
    const label = view.sequence.manifest.taxonomy.classes[0]?.id;
    if (!label) throw new Error("The frozen sequence taxonomy has no classes.");
    const index = frame()?.objects.length || 0;
    const next = tools.addObject(view.draft, source().frame_index, { identity_id: null, label, visibility, box, certainty: "uncertain" });
    commit(next); view.selected = index; renderObjects(); renderInspector(); draw();
  }
  function showFrame(position) {
    if (!view.visible || !view.sequence || !view.draft || capped()) return;
    clearTimeout(view.timer); view.timer = null; view.gesture = null; view.drawing = false;
    view.position = Math.max(0, Math.min(position, view.sequence.manifest.frames.length - 1));
    const generation = view.generation, imageGeneration = ++view.imageGeneration, current = source();
    view.imagesReady = false; view.selected = null; view.svg = null; field("stage").replaceChildren();
    field("scrub").max = view.sequence.manifest.frames.length - 1; field("scrub").value = view.position;
    field("position").textContent = `${view.position + 1}/${view.sequence.manifest.frames.length} · source ${current.frame_index} · ${playback.timeLabel(current)}`;
    const gap = playback.gapBefore(view.sequence.manifest.frames, view.position);
    field("frame-gap").hidden = !gap; field("frame-gap").textContent = `${gap} unavailable source frame(s) before this image; visibility remains unknown across the gap.`;
    field("image-status").textContent = "Loading and verifying source pixels. Editing is disabled until this frame is ready…";
    render();
    const image = node("img"); image.alt = `Verified source frame ${current.frame_index}`; image.draggable = false;
    image.onload = () => {
      if (generation !== view.generation || imageGeneration !== view.imageGeneration || !view.visible) return;
      if (image.naturalWidth !== current.width || image.naturalHeight !== current.height) return failed("Source dimensions do not match the frozen sequence.");
      view.imagesReady = true; field("stage").append(image); field("stage").style.width = `${Number(field("zoom").value) * 100}%`;
      field("image-status").textContent = `Verified source image · ${current.width} × ${current.height} pixels${view.readonly ? " · saved revision is read-only" : ""}`;
      render(); schedulePlayback();
    };
    function failed(message) {
      if (generation !== view.generation || imageGeneration !== view.imageGeneration || !view.visible) return;
      view.imagesReady = false; stop(); field("image-status").textContent = `${message} Editing is disabled. Reload the latest revision to retry.`; update();
    }
    image.onerror = () => failed("Source image is missing or failed verification.");
    image.src = projectURL(`/api/temporal/sequences/${safe(view.sequence.id)}/frames/${safe(current.frame_id)}/image`);
  }
  function schedulePlayback() {
    if (!view.playing || !view.imagesReady || !view.visible) return;
    const frames = view.sequence.manifest.frames, next = frames[view.position + 1];
    if (!next) { stop(); render(); return; }
    view.timer = setTimeout(() => showFrame(view.position + 1), Math.min(playback.playbackDelay(source(), next, view.sequence.manifest.clock), 2147483647));
  }
  function setDraft(record, readonly = false) {
    view.draft = tools.clone(record?.payload || tools.blank(view.sequence)); view.baseline = stringify(view.draft);
    view.pending = {}; view.undo = []; view.redo = []; view.readonly = readonly; view.selected = null; view.recordID = record?.id || null;
    field("conflict").hidden = true; field("notes").value = view.draft.notes;
    field("editor").hidden = capped(); render();
  }
  function renderHistory(selected) {
    options(field("history"), view.references, (record) => `Revision ${record.revision}${record.id === view.latest?.id ? " · latest (editable)" : " · read-only"} · ${record.payload.provenance?.author || "legacy provenance"}`, view.references.length ? null : "No saved reference", selected || view.latest?.id);
  }
  async function loadSequence(id, request = {}) {
    const generation = ++view.generation; invalidateImage(); view.loading = true; view.sequence = null; view.draft = null;
    view.discoveryReady = false; view.discoveryRequest = { ...request };
    view.comparison = null; view.cache = null; view.comparisons = []; view.comparisonGeneration++;
    field("editor").hidden = true; field("seed-summary").textContent = ""; field("mapping").replaceChildren(); error(null); update();
    try {
      const [sequence, references, caches] = await Promise.all([api(`/api/temporal/sequences/${safe(id)}`), api(`/api/temporal/sequences/${safe(id)}/references`), api(`/api/temporal/sequences/${safe(id)}/detection-caches`)]);
      if (generation !== view.generation || !view.visible) return;
      view.sequence = sequence; view.references = references; view.latest = references[0] || sequence.latest_reference || null; view.position = 0;
      field("sequence").value = id; renderHistory(); setDraft(view.latest);
      // History discovery does not launch cache inference or tracker work.
      const histories = await Promise.allSettled(caches.map((cache) => api(`/api/temporal/detection-caches/${safe(cache.id)}/tracking-comparisons`)));
      if (generation !== view.generation || !view.visible) return;
      histories.forEach((result) => { if (result.status === "fulfilled") view.comparisons.push(...result.value); });
      view.comparisons = view.comparisons.filter((row) => row.job?.status === "succeeded" || row.report || row.status === "completed" || row.job?.status === "completed");
      options(field("comparison"), view.comparisons, (row) => row.name || row.id, view.comparisons.length ? null : "No completed saved comparisons", request.comparison_id);
      if (request.comparison_id && !view.comparisons.some((row) => row.id === request.comparison_id)) field("seed-summary").textContent = "The requested comparison is unavailable for this sequence; its saved reference is still open.";
      if (view.comparisons.length) await loadComparison(field("comparison").value, request.lane_index || 0);
      if (generation !== view.generation || !view.visible) return;
      view.discoveryReady = true;
      field("seed-panel").open = Boolean(request.comparison_id);
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.loading = false; if (view.draft) showFrame(0); update(); } }
  }
  async function refresh(request = null) {
    if (!view.visible || view.busy || view.loading) return;
    if (!consentDiscard()) return;
    const generation = ++view.generation; view.loading = true; error(null); update();
    try {
      const sequences = await api("/api/temporal/sequences");
      if (generation !== view.generation || !view.visible) return;
      view.sequences = sequences;
      const requestedID = request?.sequence_id || view.sequence?.id;
      const selected = sequences.find((row) => row.id === requestedID) || (!requestedID ? sequences[0] : null);
      options(field("sequence"), sequences, (row) => `${row.name} · ${row.manifest.frames.length} frames`, sequences.length ? null : "No saved sequences · freeze frames in Tracking", selected?.id);
      if (!selected && requestedID) throw new Error("The requested temporal sequence is unavailable in this project.");
      view.loading = false;
      if (selected) await loadSequence(selected.id, request || {});
      else { view.sequence = null; view.draft = null; view.latest = null; field("editor").hidden = true; field("saved").textContent = "Freeze a video sequence from saved frames in Tracking to begin."; }
    } catch (failure) { if (generation === view.generation) error(failure); }
    finally { if (generation === view.generation) { view.loading = false; update(); } }
  }
  async function loadComparison(id, lane = Number(field("lane").value)) {
    const generation = view.generation, request = ++view.comparisonGeneration;
    view.comparison = null; view.cache = null; field("mapping").replaceChildren(); update();
    try {
      const comparison = await api(`/api/temporal/tracking-comparisons/${safe(id)}`);
      if (generation !== view.generation || request !== view.comparisonGeneration || !view.visible) return;
      if (comparison.sequence_id !== view.sequence.id) throw new Error("Comparison belongs to a different frozen sequence.");
      const cache = await api(`/api/temporal/detection-caches/${safe(comparison.cache_id)}`);
      if (generation !== view.generation || request !== view.comparisonGeneration || !view.visible) return;
      view.comparison = comparison; view.cache = cache; field("lane").value = lane === 1 ? "1" : "0";
      [...field("lane").options].forEach((option, index) => { option.textContent = comparison.report?.lanes[index]?.name || `Lane ${index + 1}`; });
      renderMapping();
    } catch (failure) { if (generation === view.generation && request === view.comparisonGeneration) error(failure); }
    update();
  }
  function renderMapping() {
    field("mapping").replaceChildren();
    const detector = view.cache?.config.detector, lane = view.comparison?.report?.lanes[Number(field("lane").value)], classes = view.sequence?.manifest.taxonomy.classes || [];
    if (!lane || !detector) return update();
    const contract = detector.class_contract, known = {};
    if (contract?.taxonomy_id === "coco-2017-v1") classes.filter((item) => Number.isInteger(item.coco_id)).forEach((item) => { known[item.coco_id] = item.id; });
    else if (stringify(contract?.taxonomy) === stringify(view.sequence.manifest.taxonomy)) Object.entries(contract?.output_class_mapping || {}).forEach(([label, id]) => { known[id] = label; });
    for (const id of lane.report.profile.class_ids) {
      const native = detector.classes.find((item) => item.id === id), wrapper = node("div"), label = node("label", "", `${native?.name || "Native class"} · detector ${id}`), select = node("select");
      select.id = `identity-map-${id}`; select.dataset.nativeId = id; label.htmlFor = select.id;
      select.append(new Option("Choose a taxonomy class or explicitly ignore", ""), new Option("Ignore this detector class", "__ignore__"));
      classes.forEach((item) => select.append(new Option(`${item.name || item.id} · ${item.id}`, item.id)));
      if (known[id]) select.value = known[id]; select.addEventListener("change", update); wrapper.append(label, select); field("mapping").append(wrapper);
    }
    field("seed-summary").textContent = view.latest ? "A saved reference exists. Reload it and edit its identities; seeding cannot replace saved provenance." : "Review every mapping, then press Seed to copy observations into an unsaved draft. A suggested mapping is not a review.";
    update();
  }
  async function mutation(action) {
    if (view.busy || view.loading || !view.draft) return;
    view.busy = true; const generation = view.generation; stop(); error(null); update();
    try { await action(generation); } catch (failure) { if (generation === view.generation) { error(failure); if (failure.status === 409) field("conflict").hidden = false; } }
    finally { view.busy = false; render(); }
  }
  field("refresh").addEventListener("click", () => refresh());
  field("sequence").addEventListener("change", () => { const id = field("sequence").value; if (!consentDiscard()) { field("sequence").value = view.sequence?.id || ""; return; } loadSequence(id); });
  field("reload").addEventListener("click", () => { if (consentDiscard("Discard this unsaved draft and reload the latest saved reference?")) loadSequence(view.sequence.id); });
  field("history").addEventListener("change", () => {
    const id = field("history").value;
    if (!consentDiscard()) { renderHistory(view.recordID); return; }
    const record = view.references.find((row) => row.id === id); if (!record) return;
    stop(); setDraft(record, record.id !== view.latest?.id); showFrame(view.position);
  });
  field("comparison").addEventListener("change", () => loadComparison(field("comparison").value));
  field("lane").addEventListener("change", renderMapping);
  field("seed").addEventListener("click", () => {
    if (!editable() || view.latest || !consentDiscard("Replace the unsaved draft with uncertain, unreviewed observations from this lane?")) return;
    mutation(async (generation) => {
      const class_mapping = Object.fromEntries([...field("mapping").querySelectorAll("select")].map((select) => [select.dataset.nativeId, select.value === "__ignore__" ? null : select.value]));
      const result = await api(`/api/temporal/sequences/${safe(view.sequence.id)}/identity-proposals`, { method: "POST", body: stringify({ comparison_id: view.comparison.id, lane_index: Number(field("lane").value), class_mapping }) });
      if (generation !== view.generation || !view.visible) return;
      view.undo.push(snapshot()); view.redo = []; view.draft = result.payload; view.pending = {}; view.selected = null;
      const summary = result.proposal_summary;
      field("seed-summary").textContent = `Unsaved proposals: ${summary.seeded_objects} observed objects · ${summary.seeded_identities} fresh human-reference IDs · ${summary.skipped_observations} skipped observations · ${summary.excluded_predictions} predictions excluded. All frames remain unreviewed.`;
      render();
    });
  });
  field("undo").addEventListener("click", () => { if (editable() && view.undo.length) { view.redo.push(snapshot()); restore(view.undo.pop()); } });
  field("redo").addEventListener("click", () => { if (editable() && view.redo.length) { view.undo.push(snapshot()); restore(view.redo.pop()); } });
  field("previous").addEventListener("click", () => { stop(); showFrame(view.position - 1); });
  field("next").addEventListener("click", () => { stop(); showFrame(view.position + 1); });
  field("scrub").addEventListener("input", () => { stop(); showFrame(Number(field("scrub").value)); });
  field("play").addEventListener("click", () => {
    if (view.playing) { stop(); render(); return; }
    view.playing = true; view.drawing = false; field("play").textContent = "Pause"; render();
    if (view.position >= view.sequence.manifest.frames.length - 1) showFrame(0); else schedulePlayback();
  });
  field("zoom").addEventListener("change", () => { field("stage").style.width = `${Number(field("zoom").value) * 100}%`; draw(); });
  field("draw").addEventListener("click", () => { if (editable()) { view.drawing = !view.drawing; update(); } });
  field("add-absence").addEventListener("click", () => perform(() => addObject("unknown", null)));
  field("object-class").addEventListener("change", () => perform(() => commit(tools.editObject(view.draft, source().frame_index, view.selected, { label: field("object-class").value, identity_id: null, certainty: "uncertain" }))));
  field("object-id").addEventListener("change", () => perform(() => commit(tools.editObject(view.draft, source().frame_index, view.selected, { identity_id: field("object-id").value || null }))));
  field("new-id").addEventListener("click", () => perform(() => {
    const id = newID(), next = tools.newIdentity(view.draft, object().label, id);
    commit(tools.editObject(next, source().frame_index, view.selected, { identity_id: id }));
  }));
  field("visibility").addEventListener("change", () => perform(() => {
    const visibility = field("visibility").value;
    if (visibility === "visible" && !object().box) throw new Error("Set visibility to occluded, enter the observed box coordinates, then select visible. Visible records require an observed box.");
    commit(tools.editObject(view.draft, source().frame_index, view.selected, { visibility }));
  }));
  field("certainty").addEventListener("change", () => perform(() => {
    if (field("certainty").value === "certain" && (!object().identity_id || object().visibility === "unknown")) throw new Error("Assign a human identity and known visibility before declaring certainty.");
    commit(tools.editObject(view.draft, source().frame_index, view.selected, { certainty: field("certainty").value }));
  }));
  field("apply-box").addEventListener("click", () => perform(() => {
    const box = ["x1", "y1", "x2", "y2"].map((name) => field(name).value.trim() === "" ? NaN : Number(field(name).value));
    validateBox(box); commit(tools.editObject(view.draft, source().frame_index, view.selected, { box }));
  }));
  field("clear-box").addEventListener("click", () => perform(() => commit(tools.editObject(view.draft, source().frame_index, view.selected, { box: null }))));
  field("delete-object").addEventListener("click", () => perform(() => { commit(tools.removeObject(view.draft, source().frame_index, view.selected)); view.selected = null; render(); }));
  field("split").addEventListener("click", () => perform(() => {
    const boundary = Number(field("split-boundary").value);
    if (!view.sequence.manifest.frames.some((item) => item.frame_index === boundary)) throw new Error("Choose an available source frame as the inclusive split boundary.");
    commit(tools.splitIdentity(view.draft, object().identity_id, boundary, newID()));
  }));
  field("merge").addEventListener("click", () => perform(() => {
    if (!field("merge-target").value) throw new Error("Choose an identity to merge into.");
    commit(tools.mergeIdentities(view.draft, field("merge-target").value, object().identity_id));
  }));
  field("notes").addEventListener("change", () => perform(() => { const next = tools.clone(view.draft); next.notes = field("notes").value; commit(next); }));
  // Input is committed immediately so navigation cannot overlook an unfinished note.
  field("notes").addEventListener("input", () => perform(() => { const next = tools.clone(view.draft); next.notes = field("notes").value; commit(next); }));
  field("reviewer").addEventListener("input", () => {
    if (Object.keys(view.pending).length) {
      view.undo.push(snapshot()); view.redo = [];
      view.draft = tools.clone(view.draft);
      for (const current of view.draft.frames) if (Object.hasOwn(view.pending, current.frame_index)) { current.coverage = "unreviewed"; current.review = { status: "unreviewed", reviewer: "" }; }
      view.pending = {}; renderSummary(); renderObjects();
    }
    renderSummary(); update();
  });
  for (const coverage of ["partial", "complete"]) field(`confirm-${coverage}`).addEventListener("click", () => perform(() => {
    const index = source().frame_index;
    commit(tools.confirmFrame(view.draft, index, field("reviewer").value.trim(), coverage), { frame_index: index, coverage });
  }));
  field("save").addEventListener("click", () => {
    if (!editable() || !field("reviewer").value.trim() || mismatchedConfirmations()) return;
    mutation(async (generation) => {
      const result = await api(`/api/temporal/sequences/${safe(view.sequence.id)}/identity-edits`, { method: "POST", body: stringify({ expected_revision: view.latest?.revision || 0, payload: view.draft, reviewer: field("reviewer").value.trim(), reviewed_frames: Object.entries(view.pending).map(([index, coverage]) => ({ frame_index: Number(index), coverage })) }) });
      if (generation !== view.generation || !view.visible) return;
      view.latest = result; view.references.unshift(result); renderHistory(); setDraft(result);
      field("seed-summary").textContent = "Saved provenance is immutable. Continue editing this reference or reload a revision.";
    });
  });
  window.addEventListener("beforeunload", (event) => { if (dirty() || view.busy) { event.preventDefault(); event.returnValue = ""; } });
  window.addEventListener("iris:before-workspace", (event) => {
    if (!view.visible || event.detail.name === "identities") return;
    if (view.busy) { event.preventDefault(); error("Wait for the current save or proposal request to finish before leaving."); return; }
    if (!consentDiscard("Discard unsaved temporal identity changes and leave this workspace?")) event.preventDefault();
    else if (dirty()) setDraft(view.latest);
  });
  window.addEventListener("iris:workspace", (event) => {
    view.visible = event.detail.name === "identities";
    if (!view.visible) { view.generation++; view.loading = false; invalidateImage(); return; }
    if (view.requested) { const request = view.requested; view.requested = null; refresh(request); }
    else if (view.sequence && !view.discoveryReady) loadSequence(view.sequence.id, view.discoveryRequest);
    else if (view.sequence && view.draft) showFrame(view.position);
    else refresh();
  });
  function open(request) {
    if (!request?.sequence_id) return;
    if (view.visible) refresh(request);
    else { view.requested = request; if (!window.IRISNavigation.open("identities")) view.requested = null; }
  }
  window.addEventListener("iris:temporal-identities-open", (event) => open(event.detail));
  window.addEventListener("iris:project-initialized", () => { if (view.requested) window.IRISNavigation.open("identities"); });
  document.addEventListener("visibilitychange", () => { if (document.hidden) { stop(); update(); } });
  $("#identities-workspace").addEventListener("keydown", (event) => {
    if (event.key === "Escape") { view.drawing = false; cancelGesture(); update(); return; }
    if (event.target.matches("input, textarea, select")) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); field(event.shiftKey ? "redo" : "undo").click(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); field("previous").click(); }
    else if (event.key === "ArrowRight") { event.preventDefault(); field("next").click(); }
  });
  window.IRISTemporalIdentities = Object.freeze({ open });
  update();
})();
