"use strict";

(() => {
  const field = (name) => $(`#benchmark-editor-${name}`);
  const dialog = field("dialog");
  const canvas = field("canvas");
  const geometry = window.IrisAnnotationTools;
  const tools = window.IRISBenchmarkTools;
  const taxonomy = window.IRISTaxonomyTools;
  const editor = {
    id: null, record: null, boxes: [], selected: null, view: null, tool: "select", drag: null,
    token: null, timer: null, active: false, busy: false, loading: false, dirty: false,
    request: 0, pauseRequested: false, heartbeat: null, idle: null, tick: null,
    acknowledgedAt: 0, baseline: null, history: new geometry.SnapshotHistory(), pointers: new Set(),
    timingUncertain: false,
  };
  const path = (suffix = "correction") => `/api/benchmark-outputs/${encodeURIComponent(editor.id)}/${suffix}`;
  const selected = () => editor.boxes.find((box) => box.id === editor.selected);
  const editable = () => editor.active && !editor.busy && !editor.loading && Boolean(editor.record);
  const className = (label) => taxonomy.className(editor.record?.taxonomy, label);
  const snapshot = () => ({ boxes: structuredClone(editor.boxes), reviewer: field("reviewer").value, notes: field("notes").value });
  const savedKey = () => tools.canonical(snapshot());

  function error(failure) {
    field("error").textContent = failure?.message || failure || "";
    field("error").hidden = !failure;
  }

  function ownerToken(id) {
    const key = `iris.benchmark.timer.${id}`;
    try {
      const old = sessionStorage.getItem(key);
      if (old) return old;
      const token = crypto.randomUUID();
      sessionStorage.setItem(key, token);
      return token;
    } catch { return crypto.randomUUID(); }
  }

  function clock() {
    const recorded = editor.timer?.elapsed_ms;
    const live = editor.active && typeof recorded === "number"
      ? recorded + Math.min(30000, Math.max(0, performance.now() - editor.acknowledgedAt)) : recorded;
    field("clock").textContent = tools.duration(live);
    const states = { running: "Recording", paused: "Paused", unmeasured: "Not started" };
    let message = `${states[editor.timer?.state] || "Not started"} · ${tools.duration(recorded)} recorded by the server`;
    if (editor.active) message += ". The live display includes time awaiting the next receipt.";
    if (editor.timer?.interruption_reason) message += ` · ${editor.timer.interruption_reason}`;
    if (recorded != null && editor.timer?.fully_timed === false) message += " · incomplete timing; unrecorded gaps are excluded";
    if (editor.timer?.state === "running" && editor.timer.owner_token !== editor.token) message += " · another editor owns this timer";
    field("timer-status").textContent = message;
  }

  function update() {
    const allowed = editable();
    field("body").hidden = !editor.active;
    field("mask").hidden = editor.active;
    field("reviewer").disabled = editor.busy || editor.loading || editor.active;
    field("start").disabled = editor.busy || editor.loading || !editor.record || !field("reviewer").value.trim() || editor.timer?.state === "running";
    field("start").textContent = editor.timer?.elapsed_ms == null ? "Start review" : "Resume review";
    field("pause").disabled = editor.busy || !tools.ownsRunningTimer(editor.timer, editor.token);
    field("close").disabled = editor.busy || editor.loading;
    for (const input of field("body").querySelectorAll("button, input, select, textarea")) input.disabled = !allowed;
    field("undo").disabled = !allowed || !editor.history.canUndo;
    field("redo").disabled = !allowed || !editor.history.canRedo;
    field("focus").disabled = !allowed || !selected();
    field("save").disabled = editor.busy || editor.loading || !editor.record || !field("reviewer").value.trim();
    field("save").disabled ||= !tools.canSaveCorrection(editor.timer, editor.token, editor.active);
    field("complete").disabled = field("save").disabled || editor.timer?.elapsed_ms == null;
    field("save-status").textContent = editor.record
      ? `${editor.dirty ? "Unsaved corrections" : "Saved correction state"} · revision ${editor.record.revision} · ${editor.record.status || "draft"}. Completing review is an explicit human decision, including for an empty image.` : "Loading correction record…";
    for (const name of ["select", "draw", "pan"]) {
      field(name).classList.toggle("active", editor.tool === name);
      field(name).setAttribute("aria-pressed", String(editor.tool === name));
    }
    canvas.classList.toggle("drawing", editor.tool === "draw");
    canvas.classList.toggle("pan-tool", editor.tool === "pan");
    clock();
  }

  function stopLocal() {
    editor.active = false;
    clearInterval(editor.heartbeat); editor.heartbeat = null;
    clearInterval(editor.tick); editor.tick = null;
    clearTimeout(editor.idle); editor.idle = null;
    pointerEnd(null, true);
    update();
  }

  function interacted() {
    if (!editor.active) return;
    clearTimeout(editor.idle);
    editor.idle = setTimeout(() => pause("Paused after 60 seconds without interaction. Resume when ready."), 60000);
  }

  function recordTimer(timer) {
    editor.timer = timer?.timer || timer;
    editor.acknowledgedAt = performance.now();
    clock();
  }

  async function reconcileTimer(id, request) {
    try {
      const record = await api(`/api/benchmark-outputs/${encodeURIComponent(id)}/correction`);
      if (request !== editor.request || editor.id !== id) return;
      recordTimer(record.timer);
      if (record.revision !== editor.record?.revision)
        error("The saved correction changed while this editor was open. Your local edits are preserved. Close and reopen this output to load the current saved revision before editing again.");
    } catch { /* Keep the original failure and the masked editor; never retry a timer mutation. */ }
    update();
  }

  async function timerAction(action) {
    if (editor.busy || !editor.record) return;
    if (action === "start" && (document.hidden || !document.hasFocus() || !field("reviewer").value.trim())) return;
    if (action === "heartbeat" && !editor.active) return;
    const id = editor.id, request = editor.request;
    editor.busy = true;
    error(null);
    update();
    try {
      const result = await api(path("timer"), { method: "POST", body: JSON.stringify({
        action, expected_revision: editor.timer?.revision ?? 0, token: editor.token,
        reviewer: field("reviewer").value.trim(), operation_id: crypto.randomUUID(),
        ...(action === "pause" && editor.timingUncertain ? { discard_unconfirmed: true } : {}),
      }) });
      if (id !== editor.id || request !== editor.request) return;
      recordTimer(result);
      if (action === "start" || action === "pause") editor.timingUncertain = false;
      if (action === "start" && !editor.pauseRequested && !document.hidden && document.hasFocus() && tools.ownsRunningTimer(editor.timer, editor.token)) {
        editor.active = true;
        field("image").setAttribute("href", projectURL(editor.record.frame.image_url || `/api/frames/${editor.record.frame.id}/image`));
        editor.heartbeat = setInterval(() => timerAction("heartbeat"), 10000);
        editor.tick = setInterval(clock, 200);
        interacted();
        requestAnimationFrame(paint);
      } else if (action === "start" && tools.ownsRunningTimer(editor.timer, editor.token)) {
        editor.timingUncertain = true;
      }
      if (editor.timer?.state !== "running") stopLocal();
    } catch (failure) {
      if (id !== editor.id || request !== editor.request) return;
      editor.timingUncertain = true;
      stopLocal();
      error(`${failure.message} Timing has stopped in this editor. Saved receipts will be checked; no timer request is repeated automatically.`);
      await reconcileTimer(id, request);
    } finally {
      if (id === editor.id && request === editor.request) {
        editor.busy = false;
        update();
        if (editor.pauseRequested) {
          editor.pauseRequested = false;
          if (tools.ownsRunningTimer(editor.timer, editor.token)) timerAction("pause");
        }
      }
    }
  }

  function pause(message = "") {
    if (!editor.id || !dialog.open) return;
    stopLocal();
    if (message) field("save-status").textContent = message;
    if (editor.busy) { editor.pauseRequested = true; return; }
    if (tools.ownsRunningTimer(editor.timer, editor.token)) timerAction("pause");
  }

  function changed(merge = null) {
    editor.history.commit(snapshot(), merge);
    editor.dirty = savedKey() !== editor.baseline;
    interacted(); update();
  }

  function restore(value) {
    editor.boxes = structuredClone(value.boxes);
    field("reviewer").value = value.reviewer;
    field("notes").value = value.notes;
    if (!selected()) editor.selected = null;
    editor.dirty = savedKey() !== editor.baseline;
    renderBoxes(); update();
  }

  function applyRecord(record) {
    editor.record = record;
    editor.boxes = structuredClone(record.boxes || []);
    if (!record.revision && !editor.boxes.length)
      editor.boxes = (record.proposals || []).map((proposal) => ({ id: crypto.randomUUID(), label: proposal.label, box: [...proposal.box], proposal_id: proposal.id }));
    field("reviewer").value = record.reviewer || field("reviewer").value || "";
    field("notes").value = record.notes || "";
    editor.history.reset(snapshot());
    editor.baseline = savedKey(); editor.dirty = false; editor.selected = null;
    recordTimer(record.timer);
    editor.timingUncertain = record.timer?.state === "running";
    editor.view = geometry.fitViewport(record.frame.width, record.frame.height);
    field("image").removeAttribute("href");
    field("image").setAttribute("width", record.frame.width);
    field("image").setAttribute("height", record.frame.height);
    for (const name of ["class", "box-class"]) {
      field(name).replaceChildren(...record.taxonomy.classes.map((item) => new Option(item.name, item.id)));
    }
    field("definitions").replaceChildren();
    for (const category of record.taxonomy.classes)
      field("definitions").append(node("p", "field-hint", `${category.name} (${category.id}): ${category.definition}`));
    field("history-list").replaceChildren();
    for (const revision of record.history || []) {
      const details = node("details", "benchmark-correction-revision");
      const saved = node("pre", "", "Open to inspect this saved correction revision.");
      details.append(node("summary", "", `Revision ${revision.revision} · ${revision.status} · ${revision.reviewer || "reviewer not recorded"}`), saved);
      const id = editor.id, request = editor.request;
      let loaded = false;
      details.addEventListener("toggle", async () => {
        if (!details.open || loaded) return;
        loaded = true;
        try {
          const historical = await api(`/api/benchmark-outputs/${encodeURIComponent(id)}/corrections/${revision.revision}`);
          if (id === editor.id && request === editor.request) saved.textContent = JSON.stringify(historical, null, 2);
        } catch (failure) { loaded = false; if (id === editor.id && request === editor.request) saved.textContent = failure.message; }
      });
      field("history-list").append(details);
    }
    renderBoxes(); update();
  }

  async function open(id, label, linked = false) {
    if (!id || !mayClose()) return;
    stopLocal();
    editor.id = id; editor.record = null; editor.timer = null; editor.loading = true;
    editor.dirty = false; editor.token = ownerToken(id); editor.pauseRequested = false;
    const request = ++editor.request;
    field("context").textContent = label || `Output ${id}`;
    field("link-notice").hidden = !linked;
    field("reviewer").value = "";
    field("history-list").replaceChildren();
    error(null);
    if (!dialog.open) dialog.showModal();
    update();
    try {
      const record = await api(path());
      if (request !== editor.request) return;
      applyRecord(record);
    } catch (failure) { if (request === editor.request) error(failure); }
    finally { if (request === editor.request) { editor.loading = false; update(); } }
  }

  async function save(status) {
    if (editor.busy || editor.loading || !editor.record || !field("reviewer").value.trim()) return;
    if (!tools.canSaveCorrection(editor.timer, editor.token, editor.active)) return;
    if (status === "reviewed" && editor.timer?.elapsed_ms == null) return;
    pointerEnd(null, true);
    const id = editor.id, request = editor.request, revision = editor.record.revision;
    const data = { expected_revision: revision, boxes: structuredClone(editor.boxes), status,
      reviewer: field("reviewer").value.trim(), notes: field("notes").value,
      timer_revision: editor.timer?.revision ?? 0, timer_token: editor.token };
    editor.busy = true;
    stopLocal(); error(null); update();
    try {
      const record = await api(path(), { method: "PUT", body: JSON.stringify(data) });
      if (id !== editor.id || request !== editor.request) return;
      applyRecord(record);
      window.dispatchEvent(new CustomEvent("iris:benchmark-correction-saved", { detail: { output_id: id } }));
      notify(status === "reviewed" ? "Benchmark correction reviewed. The independent reference is unchanged." : "Benchmark correction draft saved; timing paused.");
    } catch (failure) {
      if (id !== editor.id || request !== editor.request) return;
      let receipt = null;
      try { receipt = await api(path()); } catch { /* No write retry. */ }
      if (id !== editor.id || request !== editor.request) return;
      if (receipt && tools.correctionMatches(receipt, data, revision)) {
        applyRecord(receipt);
        window.dispatchEvent(new CustomEvent("iris:benchmark-correction-saved", { detail: { output_id: id } }));
        notify("The correction was saved. Its recorded revision was recovered without repeating the request.");
      } else {
        if (receipt) {
          recordTimer(receipt.timer);
          if (receipt.timer?.state === "running") editor.timingUncertain = true;
        } else editor.timingUncertain = true;
        error(`${failure.message} Your local edits are preserved. No save was repeated automatically.`);
      }
    } finally {
      if (id === editor.id && request === editor.request) { editor.busy = false; update(); }
    }
  }

  function svg(tag, attributes) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    return element;
  }
  const scale = () => { const matrix = canvas.getScreenCTM(); return matrix ? Math.hypot(matrix.a, matrix.b) || 1 : 1; };
  function shape(box, active = false, drawing = false) {
    const [x1, y1, x2, y2] = box.box;
    const group = svg("g", { class: `annotation-box${active ? " selected" : ""}` });
    const rect = svg("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, "vector-effect": "non-scaling-stroke" });
    if (!drawing) rect.dataset.boxId = box.id;
    group.append(rect);
    const text = svg("text", { x: x1 + 3 / scale(), y: Math.max(14 / scale(), y1 - 5 / scale()), "font-size": 12 / scale(), "paint-order": "stroke", "stroke-width": 3 / scale() });
    text.textContent = className(box.label); group.append(text);
    if (active && editor.tool === "select" && !drawing) {
      const size = 9 / scale();
      for (const [corner, x, y] of [["nw", x1, y1], ["ne", x2, y1], ["sw", x1, y2], ["se", x2, y2]]) {
        const handle = svg("rect", { x: x - size / 2, y: y - size / 2, width: size, height: size, class: "annotation-handle", "vector-effect": "non-scaling-stroke" });
        handle.dataset.boxId = box.id; handle.dataset.corner = corner; group.append(handle);
      }
    }
    return group;
  }
  function paint() {
    if (!editor.record || !editor.view) return;
    const { x, y, width, height } = editor.view;
    canvas.setAttribute("viewBox", `${x} ${y} ${width} ${height}`);
    field("box-layer").replaceChildren(...editor.boxes.map((box) => shape(box, box.id === editor.selected)));
  }
  function properties() {
    const box = selected();
    field("properties").hidden = !box;
    if (!box) return;
    field("box-class").value = box.label;
    ["x1", "y1", "x2", "y2"].forEach((name, index) => {
      field(name).value = Number(box.box[index].toFixed(2));
      field(name).max = index % 2 ? editor.record.frame.height : editor.record.frame.width;
    });
  }
  function renderBoxes() {
    field("boxes").replaceChildren();
    if (!editor.boxes.length) field("boxes").append(node("p", "field-hint", "No boxes. Inspect for missed targets and draw any missing labels."));
    for (const [index, box] of editor.boxes.entries()) {
      const button = node("button", `annotation-box-row${box.id === editor.selected ? " selected" : ""}`, `${index + 1} · ${className(box.label)}`);
      button.type = "button"; button.disabled = !editable();
      button.addEventListener("click", () => { if (!editable()) return; editor.selected = box.id; renderBoxes(); update(); interacted(); });
      field("boxes").append(button);
    }
    properties(); paint();
  }
  function point(event, bounded = true) {
    const matrix = canvas.getScreenCTM();
    if (!matrix || !editor.record) return null;
    const value = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    if (![value.x, value.y].every(Number.isFinite)) return null;
    return bounded ? [Math.max(0, Math.min(editor.record.frame.width, value.x)), Math.max(0, Math.min(editor.record.frame.height, value.y))] : [value.x, value.y];
  }
  function pointerDown(event) {
    editor.pointers.add(event.pointerId);
    if (editor.pointers.size > 1 || event.isPrimary === false) { pointerEnd(null, true); return; }
    if (!editable() || ![0, 1].includes(event.button)) return;
    const start = point(event, false); if (!start) return;
    const frame = editor.record.frame, pan = editor.tool === "pan" || event.button === 1;
    if (!pan && (start[0] < 0 || start[1] < 0 || start[0] > frame.width || start[1] > frame.height)) return;
    event.preventDefault(); canvas.focus({ preventScroll: true }); interacted(); editor.history.breakMerge();
    const id = event.target.dataset.boxId;
    if (pan) editor.drag = { kind: "pan", pointer: event.pointerId, startScreen: [event.clientX, event.clientY], originalView: structuredClone(editor.view), scale: scale() };
    else if (editor.tool === "draw") editor.drag = { kind: "draw", pointer: event.pointerId, start, current: start, before: snapshot(), selection: editor.selected };
    else if (editor.boxes.some((box) => box.id === id)) {
      editor.selected = id;
      editor.drag = { kind: event.target.dataset.corner || "move", pointer: event.pointerId, start, original: [...selected().box], before: snapshot(), selection: id };
    } else { editor.selected = null; renderBoxes(); update(); return; }
    canvas.setPointerCapture(event.pointerId); renderBoxes(); update();
  }
  function pointerMove(event) {
    const drag = editor.drag;
    if (!drag || drag.pointer !== event.pointerId) return;
    interacted(); const frame = editor.record.frame;
    if (drag.kind === "pan") {
      editor.view = geometry.panViewport(drag.originalView, -(event.clientX - drag.startScreen[0]) / drag.scale, -(event.clientY - drag.startScreen[1]) / drag.scale, frame.width, frame.height);
      paint(); return;
    }
    const current = point(event); if (!current) return; drag.current = current;
    if (drag.kind === "draw") {
      const box = [Math.min(drag.start[0], current[0]), Math.min(drag.start[1], current[1]), Math.max(drag.start[0], current[0]), Math.max(drag.start[1], current[1])];
      field("draw-layer").replaceChildren(shape({ box, label: field("class").value }, true, true)); return;
    }
    if (selected()) selected().box = tools.boxAfterDrag(drag.original, drag.kind, drag.start, current, frame.width, frame.height);
    paint(); properties();
  }
  function pointerEnd(event, cancel = false) {
    const drag = editor.drag;
    if (!drag || event?.pointerId != null && event.pointerId !== drag.pointer) return;
    editor.drag = null;
    if (canvas.hasPointerCapture(drag.pointer)) canvas.releasePointerCapture(drag.pointer);
    field("draw-layer").replaceChildren();
    if (drag.kind === "pan") { if (cancel) editor.view = drag.originalView; }
    else if (cancel) { restore(drag.before); editor.selected = drag.selection; }
    else if (drag.kind === "draw") {
      const current = drag.current;
      const box = [Math.min(drag.start[0], current[0]), Math.min(drag.start[1], current[1]), Math.max(drag.start[0], current[0]), Math.max(drag.start[1], current[1])];
      if (box[2] - box[0] >= 1 && box[3] - box[1] >= 1) {
        const entry = { id: crypto.randomUUID(), label: field("class").value, box, proposal_id: null };
        editor.boxes.push(entry); editor.selected = entry.id; editor.tool = "select"; changed();
      }
    } else if (selected() && tools.canonical(selected().box) !== tools.canonical(drag.original)) changed();
    renderBoxes(); update();
  }
  function zoom(action, factor = 1, anchor = null) {
    if (!editable()) return;
    const frame = editor.record.frame;
    if (action === "fit") editor.view = geometry.fitViewport(frame.width, frame.height);
    else if (action === "focus" && selected()) editor.view = geometry.focusViewport(selected().box, frame.width, frame.height);
    else if (action === "zoom") editor.view = geometry.zoomViewport(editor.view, factor, anchor || { x: editor.view.x + editor.view.width / 2, y: editor.view.y + editor.view.height / 2 }, frame.width, frame.height);
    paint(); interacted();
  }
  function deleteBox() {
    if (!editable() || !selected()) return;
    editor.boxes = editor.boxes.filter((box) => box.id !== editor.selected); editor.selected = null; changed(); renderBoxes();
  }
  function history(direction) {
    if (!editable() || editor.drag) return;
    const value = editor.history[direction](); if (value) restore(value);
    interacted();
  }
  function mayClose() {
    if (!dialog.open) return true;
    if (editor.busy || editor.loading) { notify("Wait for the correction or timer request to finish before leaving.", true); return false; }
    if (editor.active || tools.ownsRunningTimer(editor.timer, editor.token)) {
      pause(); notify("Timing is being paused. Save your correction draft, then close the editor."); return false;
    }
    return !editor.dirty || window.confirm("Discard unsaved benchmark corrections? Recorded timing remains saved.");
  }
  function close() { if (mayClose()) dialog.close(); }
  field("close").addEventListener("click", close);
  dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
  dialog.addEventListener("close", () => { stopLocal(); editor.request++; editor.record = null; editor.id = null; editor.dirty = false; field("image").removeAttribute("href"); });
  field("start").addEventListener("click", () => timerAction("start"));
  field("pause").addEventListener("click", () => pause());
  field("save").addEventListener("click", () => save("draft"));
  field("complete").addEventListener("click", () => save("reviewed"));
  field("reviewer").addEventListener("input", () => { if (editor.record) changed("reviewer"); else update(); });
  field("notes").addEventListener("input", () => { if (editable()) changed("notes"); });
  for (const name of ["select", "draw", "pan"]) field(name).addEventListener("click", () => { if (editable()) { editor.tool = name; paint(); update(); interacted(); } });
  for (const direction of ["undo", "redo"]) field(direction).addEventListener("click", () => history(direction));
  field("fit").addEventListener("click", () => zoom("fit"));
  field("focus").addEventListener("click", () => zoom("focus"));
  field("zoom-in").addEventListener("click", () => zoom("zoom", 1.25));
  field("zoom-out").addEventListener("click", () => zoom("zoom", 0.8));
  field("delete").addEventListener("click", deleteBox);
  field("add").addEventListener("click", () => {
    if (!editable()) return;
    const { width, height } = editor.record.frame;
    const box = { id: crypto.randomUUID(), label: field("class").value, box: [width / 4, height / 4, width * 0.75, height * 0.75], proposal_id: null };
    editor.boxes.push(box); editor.selected = box.id; changed(); renderBoxes();
  });
  field("box-class").addEventListener("change", () => { if (editable() && selected()) { selected().label = field("box-class").value; changed(); renderBoxes(); } });
  field("coordinates").addEventListener("click", () => {
    if (!editable() || !selected()) return;
    const inputs = ["x1", "y1", "x2", "y2"].map(field);
    if (!inputs.every((input) => input.value !== "" && input.reportValidity())) return;
    const values = inputs.map((input) => Number(input.value));
    if (values[2] <= values[0] || values[3] <= values[1]) { error("Right and bottom coordinates must exceed left and top."); return; }
    selected().box = values; changed(); renderBoxes(); error(null);
  });
  canvas.addEventListener("pointerdown", pointerDown);
  canvas.addEventListener("pointermove", pointerMove);
  canvas.addEventListener("pointerup", (event) => pointerEnd(event));
  canvas.addEventListener("pointercancel", (event) => pointerEnd(event, true));
  canvas.addEventListener("lostpointercapture", (event) => pointerEnd(event, true));
  canvas.addEventListener("auxclick", (event) => { if (event.button === 1) event.preventDefault(); });
  for (const name of ["pointerup", "pointercancel"]) window.addEventListener(name, (event) => editor.pointers.delete(event.pointerId));
  canvas.addEventListener("wheel", (event) => {
    if (!editable() || document.activeElement !== canvas || event.ctrlKey || event.metaKey) return;
    const anchor = point(event, false); if (!anchor) return;
    event.preventDefault(); zoom("zoom", Math.exp(-Math.max(-500, Math.min(500, event.deltaY * (event.deltaMode === 1 ? 16 : 1))) * 0.002), { x: anchor[0], y: anchor[1] });
  }, { passive: false });
  dialog.addEventListener("keydown", (event) => {
    interacted();
    if (!editable() || event.target.closest("input,textarea,select") || event.altKey) return;
    const key = event.key.toLowerCase(), command = event.ctrlKey || event.metaKey;
    if (command && ["z", "y"].includes(key)) { event.preventDefault(); history(key === "y" || event.shiftKey ? "redo" : "undo"); }
    else if (!command && ["v", "b", "h"].includes(key)) { event.preventDefault(); editor.tool = { v: "select", b: "draw", h: "pan" }[key]; paint(); update(); }
    else if (!command && ["delete", "backspace"].includes(key)) { event.preventDefault(); deleteBox(); }
    else if (key === "escape" && editor.drag) { event.preventDefault(); event.stopPropagation(); pointerEnd(null, true); }
  });
  for (const name of ["pointerdown", "pointermove", "input", "wheel"]) dialog.addEventListener(name, interacted, { passive: true });
  window.addEventListener("blur", () => { if (editor.active) pause("Paused while this window is not focused."); });
  document.addEventListener("visibilitychange", () => { if (document.hidden && editor.active) pause("Paused while this tab is hidden."); });
  for (const eventName of ["iris:before-session", "iris:before-workspace"]) window.addEventListener(eventName, (event) => {
    if (!dialog.open || eventName === "iris:before-workspace" && event.detail?.name === "benchmark") return;
    if (!mayClose()) event.preventDefault(); else dialog.close();
  });
  window.addEventListener("beforeunload", (event) => {
    if (!dialog.open) return;
    if (editor.active || editor.dirty || editor.busy) { pause(); event.preventDefault(); event.returnValue = ""; }
  });
  window.addEventListener("iris:benchmark-correct", (event) => open(event.detail?.output_id, event.detail?.label));
  const linkedReview = tools.correctionLink(window.location.search);
  let linkOpened = false;
  function openLinkedReview() {
    if (!linkedReview || linkOpened || !state.projectInitialized) return;
    linkOpened = true;
    if (linkedReview.error) {
      field("context").textContent = linkedReview.label;
      field("link-notice").hidden = false;
      error(linkedReview.error);
      dialog.showModal(); update();
      return;
    }
    // Project and initial session are ready. This only reads the correction;
    // the human must explicitly start/resume timing and save their own review.
    open(linkedReview.output_id, linkedReview.label, true);
  }
  window.addEventListener("iris:project-initialized", openLinkedReview);
  if (state.projectInitialized) openLinkedReview();
  new ResizeObserver(() => { if (editor.active) { pointerEnd(null, true); paint(); } }).observe(canvas);
})();
