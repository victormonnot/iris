"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const projectScope = window.IRISProjectScope.create(window.location.href, {
  getItem: (key) => localStorage.getItem(key),
  setItem: (key, value) => localStorage.setItem(key, value),
});
const projectURL = (path) => projectScope.url(path);
const state = {
  projectId: projectScope.id,
  projects: [],
  sessions: [],
  sessionId: null,
  assets: [],
  frames: [],
  collectionLoading: false,
  collectionError: "",
  jobs: [],
  filter: "all",
  galleryFilters: { source: "", search: "", review: "all", signal: "all" },
  insights: new Map(),
  insightsRequest: 0,
  insightsLoading: false,
  insightsFrameKey: null,
  insightsMessage: "",
  insightsError: false,
  insightsWarnings: [],
  inspecting: null,
  inspectionIds: [],
  extracting: null,
  extractionPreview: {
    generation: 0,
    timer: null,
    plan: null,
    configKey: null,
    imagesPending: false,
    imageUrls: [],
  },
  pendingExtractions: new Set(),
  frameFetch: 0,
  pendingSelections: new Map(),
  bulkSelecting: false,
  polling: null,
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  let response;
  try {
    response = await fetch(projectURL(path), { ...options, headers });
  } catch {
    throw new Error(
      "Cannot reach IRIS. Check that the local server is running and try again.",
    );
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg || String(item)).join("; ")
      : typeof detail === "string"
        ? detail
        : `Request failed (${response.status}).`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function notify(message, isError = false) {
  const notice = $("#notice");
  notice.className = `notice${isError ? " error" : ""}`;
  notice.setAttribute("role", isError ? "alert" : "status");
  notice.setAttribute("aria-live", isError ? "assertive" : "polite");
  notice.textContent = message;
  notice.hidden = false;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function timestamp(seconds) {
  if (seconds === null || seconds === undefined) return "Still image";
  const minutes = Math.floor(seconds / 60);
  const remainder = (seconds % 60).toFixed(2).padStart(5, "0");
  return `${String(minutes).padStart(2, "0")}:${remainder}`;
}

function sourceFor(frame) {
  return state.assets.find((asset) => asset.id === frame.asset_id);
}

function visibleFrames() {
  return window.IRISIntakeTools.filterFrames(state.frames, state.assets, state.insights,
    { ...state.galleryFilters, selected: state.filter === "selected" });
}

function reviewLabel(info) {
  if (!info) return "Review status unavailable";
  if (info.negative === true) return "Validated negative";
  if (info.positive === true) return "Validated · objects present";
  return { unannotated: "Unreviewed", draft: "Draft labels", pending_suggestions: "Pending proposals", validated: "Validated" }[info.review_status] || "Review status unavailable";
}

function uploadNavigationBlocked() {
  if (!uploadQueue.state.busy) return false;
  notify("An import is still active. Let it finish, or cancel the remaining files and wait for the active file.", true);
  return true;
}

function isActive(job) {
  return job.status === "queued" || job.status === "running";
}

function rememberSession(id) {
  projectScope.rememberSession(id);
}

function recalledSession() {
  return projectScope.recalledSession();
}

function renderProjects() {
  const select = $("#project-select");
  select.replaceChildren();
  for (const project of state.projects) {
    const option = node("option", "", project.name);
    option.value = project.id;
    select.append(option);
  }
  select.value = state.projectId;
  select.disabled = !state.projects.length;
  const active = state.projects.find((project) => project.id === state.projectId);
  select.title = active?.name || "Choose a project";
  $("#project-description").textContent = active?.description ||
    "Sessions, datasets and results stay together in this project.";
  document.title = active ? `${active.name} · IRIS` : "IRIS · Vision workbench";
}

function openProject(id) {
  // Keep the current selector and scope intact if beforeunload is cancelled.
  $("#project-select").value = state.projectId;
  if (id === state.projectId || uploadNavigationBlocked()) return;
  // Navigation resets every module, pending preview and modal in one operation.
  // Existing annotation/report beforeunload handlers protect unsaved edits.
  window.location.assign(projectScope.location(id));
}

function renderSessions() {
  const list = $("#session-list");
  list.replaceChildren();
  $("#session-count").textContent = state.sessions.length;
  if (!state.sessions.length) {
    list.append(node("p", "session-empty", "No sessions in this project yet."));
  }
  for (const session of state.sessions) {
    const button = node(
      "button",
      `session-item${session.id === state.sessionId ? " active" : ""}`,
    );
    button.type = "button";
    button.title = `${session.name} · ${session.scene_group || "No scene group"}`;
    button.setAttribute(
      "aria-current",
      session.id === state.sessionId ? "true" : "false",
    );
    button.append(
      node("strong", "", session.name),
      node("span", "", session.scene_group || "No scene group"),
    );
    button.addEventListener("click", () =>
      selectSession(session.id).catch((error) => notify(error.message, true)),
    );
    list.append(button);
  }
}

async function selectSession(id) {
  if (state.sessionId !== id) {
    if (uploadNavigationBlocked() || state.bulkSelecting || state.pendingSelections.size) {
      if (!uploadQueue.state.busy) notify("Wait for the selection update before changing sessions.", true);
      return;
    }
    const event = new CustomEvent("iris:before-session", {
      cancelable: true,
      detail: { sessionId: id },
    });
    if (!window.dispatchEvent(event)) return;
  }
  state.sessionId = id;
  if ($("#extract-dialog").open) $("#extract-dialog").close();
  state.assets = [];
  state.frames = [];
  state.collectionLoading = true;
  state.collectionError = "";
  state.filter = "all";
  state.galleryFilters = { source: "", search: "", review: "all", signal: "all" };
  state.insights = new Map();
  state.insightsRequest++;
  state.insightsLoading = false;
  state.insightsFrameKey = null;
  state.insightsMessage = "";
  state.insightsError = false;
  state.insightsWarnings = [];
  if ($("#frame-dialog").open) $("#frame-dialog").close();
  state.inspecting = null;
  state.inspectionIds = [];
  for (const key of ["source", "search", "review", "signal"]) $(`#gallery-${key}`).value = state.galleryFilters[key];
  rememberSession(id);
  renderSessions();
  const session = state.sessions.find((item) => item.id === id);
  $("#welcome").hidden = true;
  $("#session-workspace").hidden = false;
  $("#active-session-name").textContent = session?.name || "Session";
  $("#active-session-group").textContent = session?.scene_group
    ? `Scene group / ${session.scene_group}`
    : "No scene group assigned";
  renderAssets();
  renderFrames();
  window.dispatchEvent(new Event("iris:session"));
  await refreshSession();
}

async function refreshSession(forceInsights = false) {
  if (!state.sessionId) return;
  const id = state.sessionId;
  const request = ++state.frameFetch;
  let assets, frames;
  try {
    [assets, frames] = await Promise.all([
      api(`/api/sessions/${encodeURIComponent(id)}/assets`),
      api(`/api/sessions/${encodeURIComponent(id)}/frames`),
    ]);
  } catch (error) {
    if (state.sessionId === id && request === state.frameFetch) {
      state.collectionLoading = false;
      state.collectionError = error.message;
      renderAssets();
      renderFrames();
    }
    throw error;
  }
  if (state.sessionId !== id || request !== state.frameFetch) return;
  const assetsChanged = JSON.stringify(state.assets) !== JSON.stringify(assets);
  const wasLoading = state.collectionLoading || Boolean(state.collectionError);
  state.collectionLoading = false;
  state.collectionError = "";
  state.assets = assets;
  state.frames = frames.map((frame) =>
    state.pendingSelections.has(frame.id)
      ? { ...frame, selected: state.pendingSelections.get(frame.id) }
      : frame,
  );
  if (assetsChanged || wasLoading || !assets.length) renderAssets();
  renderGallerySources();
  renderFrames();
  const frameKey = frames.map((frame) => `${frame.id}:${frame.duplicate_count || 0}`).join("|");
  if (forceInsights || state.insightsFrameKey !== frameKey) {
    state.insightsFrameKey = frameKey;
    await refreshInsights();
  }
}

async function refreshInsights() {
  const sessionId = state.sessionId;
  if (!sessionId) return;
  const request = ++state.insightsRequest;
  state.insightsLoading = true;
  state.insightsError = false;
  state.insightsMessage = "Refreshing saved review and similarity signals…";
  renderGalleryStatus();
  try {
    const result = await api(`/api/sessions/${encodeURIComponent(sessionId)}/selection-insights`);
    if (request !== state.insightsRequest || sessionId !== state.sessionId) return;
    state.insights = new Map(result.frames.map((frame) => [frame.frame_id, frame]));
    state.insightsWarnings = result.warnings || [];
    const total = result.summary?.total_frames ?? state.frames.length;
    state.insightsMessage = `Review signals available for ${result.frames.length} / ${total} frames. Low confidence means a saved score from 0.1 to below 0.5.`;
    if (result.frames.length < total) state.insightsMessage += " Unavailable frames are excluded from review and signal filters.";
    if (result.limits?.similarity_truncated) state.insightsMessage += ` Similarity inspection is limited to the first ${result.limits.max_similarity_frames} frames.`;
    if (result.frames.some((frame) => frame.prediction_sources_truncated)) state.insightsMessage += " Some older prediction sources were outside the inspection limit.";
  } catch (error) {
    if (request !== state.insightsRequest || sessionId !== state.sessionId) return;
    state.insights = new Map();
    state.insightsWarnings = [];
    state.insightsError = true;
    state.insightsMessage = `Review signals unavailable: ${error.message} Use Refresh review signals to try again.`;
  } finally {
    if (request === state.insightsRequest && sessionId === state.sessionId) {
      state.insightsLoading = false;
      renderGalleryStatus();
      renderFrames();
      if (state.inspecting && $("#frame-dialog").open) renderInspection();
    }
  }
}

function renderGalleryStatus() {
  $("#gallery-insights-status").textContent = state.insightsMessage;
  if (state.insightsError) $("#intake-workspace .gallery-explanation").open = true;
  $("#gallery-refresh").disabled = state.insightsLoading || !state.sessionId;
  $("#gallery-warnings").replaceChildren(...state.insightsWarnings.map((warning) => node("p", "field-hint", warning)));
}

function renderGallerySources() {
  const select = $("#gallery-source");
  select.replaceChildren(new Option("All sources", ""));
  for (const asset of state.assets) select.append(new Option(asset.filename, asset.id));
  if (!state.assets.some((asset) => asset.id === state.galleryFilters.source)) state.galleryFilters.source = "";
  select.value = state.galleryFilters.source;
  select.title = select.selectedOptions[0]?.textContent || "All sources";
}

function renderAssets() {
  $("#asset-total").textContent = state.assets.length;
  $("#intake-source-count").textContent = state.assets.length;
  const list = $("#asset-list");
  list.replaceChildren();
  if (!state.assets.length) {
    const empty = node("div", "empty-assets");
    empty.append(
      node("strong", "", state.collectionLoading ? "Loading source files…" : state.collectionError ? "Source files unavailable" : "Bring in your first source file"),
      node(
        "p",
        "",
        state.collectionLoading ? "Opening the saved sources for this session." : state.collectionError ? "Use Try again in the collection to reload this session." : "Import images or a video. Source files stay on this machine, with their original content preserved.",
      ),
    );
    list.append(empty);
    return;
  }
  for (const asset of state.assets) {
    const row = node("div", "asset-row");
    const icon = node(
      "span",
      "asset-icon",
      asset.kind === "video" ? "VID" : "IMG",
    );
    icon.setAttribute("aria-hidden", "true");
    const info = node("div", "asset-info");
    const name = node("p", "asset-name", asset.filename);
    name.title = asset.filename;
    const metadata = node("div", "asset-meta");
    const details = [formatBytes(asset.size_bytes)];
    if (asset.metadata?.width && asset.metadata?.height)
      details.push(`${asset.metadata.width} × ${asset.metadata.height}`);
    if (asset.kind === "video" && asset.metadata?.duration_seconds != null)
      details.push(timestamp(asset.metadata.duration_seconds));
    if (asset.kind === "video" && asset.metadata?.fps)
      details.push(`${Number(asset.metadata.fps).toFixed(1)} fps`);
    for (const detail of details) metadata.append(node("span", "", detail));
    info.append(name, metadata);
    row.append(icon, info);
    if (asset.kind === "video") {
      const button = node(
        "button",
        "button button-secondary asset-action",
        "Extract frames",
      );
      button.type = "button";
      button.setAttribute(
        "aria-label",
        `Extract frames from ${asset.filename}`,
      );
      button.addEventListener("click", () => openExtraction(asset));
      const actions = node("div", "asset-video-actions");
      const review = node("button", "button button-secondary asset-action", "Suggest passages");
      review.type = "button";
      review.setAttribute("aria-label", `Suggest passages from ${asset.filename}`);
      review.addEventListener("click", () => window.dispatchEvent(new CustomEvent(
        "iris:video-review", { detail: { asset } },
      )));
      actions.append(button, review);
      row.append(actions);
    } else {
      row.append(node("span", "asset-image-note", "Image imported"));
    }
    list.append(row);
  }
}

function renderFrames() {
  const focused = document.activeElement;
  const focusedCard = focused?.closest(".frame-card");
  const focusedId = focusedCard?.dataset.frameId;
  const focusedControl = focused?.tagName === "INPUT" ? "input" : "button";
  const selected = state.frames.filter((frame) => frame.selected).length;
  $("#intake-review-count").textContent = selected;
  $("#intake-review-selection").disabled = !selected || state.bulkSelecting || state.pendingSelections.size > 0 || state.collectionLoading;
  $("#intake-review-selection").setAttribute("aria-label", `Review selection · ${selected} selected frame${selected === 1 ? "" : "s"}`);
  $("#frame-total").textContent = state.frames.length;
  $("#selected-total").textContent = selected;
  $("#filter-selected-count").textContent = selected;
  $("#selection-summary").textContent = selected
    ? `${selected} frame${selected === 1 ? "" : "s"} selected for comparison and annotation`
    : "No frames selected";
  for (const filter of ["all", "selected"]) {
    const button = $(`#filter-${filter}`);
    button.classList.toggle("active", state.filter === filter);
    button.setAttribute("aria-pressed", String(state.filter === filter));
  }
  const frames = visibleFrames();
  const sourceName = state.assets.find((asset) => asset.id === state.galleryFilters.source)?.filename;
  $("#intake-collection-context").textContent = state.collectionLoading ? "Loading this session…" : `${frames.length} of ${state.frames.length} frames shown · ${sourceName || "All sources"}`;
  $("#intake-load-status").hidden = !state.collectionError;
  $("#intake-load-message").textContent = state.collectionError ? `Could not refresh this collection. ${state.collectionError}${state.frames.length ? " Previously loaded frames are still shown." : ""}` : "";
  $("#frame-grid").setAttribute("aria-busy", String(state.collectionLoading));
  $("#visible-frame-count").textContent = frames.length;
  $("#select-visible").disabled =
    state.bulkSelecting || state.pendingSelections.size > 0 || !frames.some((frame) => !frame.selected);
  $("#clear-selection").disabled = state.bulkSelecting || state.pendingSelections.size > 0 || !frames.some((frame) => frame.selected);
  const grid = $("#frame-grid");
  grid.replaceChildren();
  if (!frames.length) {
    const empty = node("div", "empty-frames");
    const onlySelected = state.filter === "selected";
    const filtered = state.frames.length > 0;
    empty.append(
      node(
        "strong",
        "",
        state.collectionLoading ? "Loading your collection…" : state.collectionError ? "Collection unavailable" : filtered ? "No frames match these filters" : onlySelected ? "Your selection is empty" : "No frames to review yet",
      ),
      node(
        "p",
        "",
        state.collectionLoading ? "Your saved sources and frame selection will appear here." : state.collectionError ? "Try loading the session again. No source files or selections have been changed." : filtered ? "Reset the filters or choose another source or review state. Frames remain saved in this session." : onlySelected
          ? "Choose frames from the collection using their checkboxes. Your selection is saved automatically."
          : "Import an image, or extract frames from a video in the source library. Each frame keeps its source and original timestamp.",
      ),
    );
    grid.append(empty);
  }
  for (const frame of frames) grid.append(frameCard(frame));
  if (focusedId) {
    const card = [...grid.children].find(
      (item) => item.dataset.frameId === focusedId,
    );
    if (card) $(focusedControl, card)?.focus({ preventScroll: true });
    else $("#filter-selected").focus({ preventScroll: true });
  }
  if (state.inspecting && $("#frame-dialog").open) renderInspectionSelection();
  window.dispatchEvent(new Event("iris:frames"));
}

function frameCard(frame) {
  const source = sourceFor(frame);
  const card = node(
    "article",
    `frame-card${frame.selected ? " selected" : ""}`,
  );
  card.dataset.frameId = frame.id;
  const open = node("button", "frame-open");
  open.type = "button";
  open.setAttribute(
    "aria-label",
    `Inspect ${source?.filename || "frame"}${source?.kind === "video" ? ` at ${timestamp(frame.timestamp_seconds)}` : ""}`,
  );
  const image = node("img");
  image.src = projectURL(`/api/frames/${encodeURIComponent(frame.id)}/image`);
  image.alt = "";
  image.loading = "lazy";
  image.decoding = "async";
  open.append(image);
  if (source?.kind === "video")
    open.append(
      node("span", "frame-time", `≈ ${timestamp(frame.timestamp_seconds)}`),
    );
  open.addEventListener("click", () => openInspection(frame.id));
  const bottom = node("div", "frame-card-bottom");
  const label = node("label");
  const checkbox = node("input");
  checkbox.type = "checkbox";
  checkbox.checked = Boolean(frame.selected);
  checkbox.disabled = state.bulkSelecting;
  checkbox.setAttribute(
    "aria-label",
    `Select ${source?.filename || "frame"}${source?.kind === "video" ? ` at ${timestamp(frame.timestamp_seconds)}` : ""}`,
  );
  checkbox.addEventListener("change", () => {
    if (state.pendingSelections.has(frame.id)) {
      checkbox.checked = state.pendingSelections.get(frame.id);
      return;
    }
    updateSelection(frame.id, checkbox.checked).catch((error) =>
      notify(error.message, true),
    );
  });
  const caption = node("span", "frame-card-label");
  const title = node("strong", "", source?.filename || "Source frame");
  title.title = source?.filename || "Source frame";
  caption.append(
    title,
    node(
      "small",
      "",
      `${frame.width} × ${frame.height}${source?.kind === "video" ? ` · Frame ${frame.frame_index}` : " · Still image"}`,
    ),
  );
  label.append(checkbox, caption);
  bottom.append(label);
  if (frame.duplicate_count > 0) {
    const duplicates = node(
      "span",
      "frame-duplicate",
      `+${frame.duplicate_count}`,
    );
    duplicates.title = `${frame.duplicate_count} duplicate occurrence(s)`;
    bottom.append(duplicates);
  }
  const info = state.insights.get(frame.id);
  const signals = node("div", "frame-signals");
  signals.append(node("span", "frame-review-state", reviewLabel(info)));
  if (info?.low_confidence_count > 0) signals.append(node("span", "", `${info.low_confidence_count} low confidence`));
  if (info?.no_target_predictions === true) signals.append(node("span", "", "No saved target detections"));
  if (info?.exact_duplicate_ids?.length || info?.similar_frame_ids?.length) {
    const related = node("button", "text-button", `${info.exact_duplicate_count ?? info.exact_duplicate_ids?.length ?? 0} identical · ${info.similar_frame_count ?? info.similar_frame_ids?.length ?? 0} similar`);
    related.type = "button";
    related.setAttribute("aria-label", `Inspect related frames for ${source?.filename || frame.id}`);
    related.addEventListener("click", () => { openInspection(frame.id); $("#inspect-related").scrollIntoView({ block: "nearest" }); });
    signals.append(related);
  }
  card.append(open, bottom, signals);
  return card;
}

async function updateSelection(id, selected, deferRender = false) {
  if (state.pendingSelections.has(id)) return;
  state.pendingSelections.set(id, selected);
  const frame = state.frames.find((item) => item.id === id);
  const previous = frame?.selected;
  if (frame) frame.selected = selected;
  if (!deferRender) renderFrames();
  try {
    const result = await api(`/api/frames/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ selected }),
    });
    const index = state.frames.findIndex((item) => item.id === id);
    if (index !== -1)
      state.frames[index] = { ...state.frames[index], ...result };
  } catch (error) {
    const current = state.frames.find((item) => item.id === id);
    if (current) current.selected = previous;
    throw error;
  } finally {
    state.pendingSelections.delete(id);
    if (!deferRender) renderFrames();
  }
}

async function bulkSelection(selected) {
  if (state.bulkSelecting || state.pendingSelections.size) return;
  const sessionId = state.sessionId;
  const payload = window.IRISIntakeTools.selectionPayload(visibleFrames(), selected);
  if (!payload.frame_ids.length) return;
  if (payload.frame_ids.length > 1000) {
    notify("Select at most 1,000 frames in one action. Narrow the source or gallery filters first.", true);
    return;
  }
  state.bulkSelecting = true;
  renderFrames();
  try {
    const result = await api(`/api/sessions/${encodeURIComponent(sessionId)}/selection`, { method: "POST", body: JSON.stringify(payload) });
    if (state.sessionId !== sessionId) return;
    const updates = new Map(result.frames.map((frame) => [frame.id, frame.selected]));
    for (const frame of state.frames) if (updates.has(frame.id)) frame.selected = updates.get(frame.id);
    notify(`${result.changed_count} frame selection${result.changed_count === 1 ? "" : "s"} updated. Only the visible filtered frames were affected.`);
  } catch (error) {
    notify(error.status === 409 ? "Selection changed in another view. No frames were updated. Review the refreshed selection and try again." : error.message, true);
    if (state.sessionId === sessionId) await refreshSession().catch((failure) => notify(failure.message, true));
  } finally {
    state.bulkSelecting = false;
    if (state.sessionId === sessionId) renderFrames();
  }
}

function openExtraction(asset) {
  clearExtractionPreview();
  state.extracting = asset.id;
  $("#extract-source").textContent = asset.filename;
  $("#extract-error").hidden = true;
  $("#extract-form").reset();
  $("#extract-dialog").showModal();
  $("#extract-dialog").scrollTop = 0;
  scheduleExtractionPreview(0);
}

function clearExtractionPreview() {
  const preview = state.extractionPreview;
  preview.generation += 1;
  clearTimeout(preview.timer);
  preview.timer = null;
  preview.plan = null;
  preview.configKey = null;
  preview.imagesPending = false;
  for (const url of preview.imageUrls) URL.revokeObjectURL(url);
  preview.imageUrls = [];
  $("#extract-thumbnails").replaceChildren();
  $("#extract-plan-details").hidden = true;
  $("#extract-preview-retry").hidden = true;
  $("#extract-images-status").textContent =
    "View up to 12 positions locally. This preview adds no frames and performs no scene analysis.";
}

function extractionConfig() {
  if (!$("#extract-form").checkValidity()) {
    throw new Error("Enter valid sampling settings to preview the extraction.");
  }
  const mode = $("#extract-mode").value;
  const start = Number($("#extract-start").value);
  const end = $("#extract-end").value === "" ? null : Number($("#extract-end").value);
  if (end !== null && end <= start) {
    throw new Error("The end time must be after the start time.");
  }
  return {
    sampling_mode: mode,
    interval_seconds: mode === "uniform" ? 2 : Number($("#extract-interval").value),
    start_seconds: start,
    end_seconds: end,
    max_frames: Number($("#extract-limit").value),
    dedup_hamming: $("#extract-dedup").checked ? Number($("#extract-hamming").value) : null,
  };
}

function extractionPreviewCurrent(generation, assetId, sessionId, configKey) {
  return $("#extract-dialog").open &&
    state.extractionPreview.generation === generation &&
    state.extracting === assetId && state.sessionId === sessionId &&
    state.extractionPreview.configKey === configKey;
}

function updateExtractionControls() {
  const pending = state.pendingExtractions.has(state.extracting);
  const preview = state.extractionPreview;
  for (const control of $("#extract-form").querySelectorAll("input, select")) {
    control.disabled = pending;
  }
  $("#extract-interval").disabled = pending || $("#extract-mode").value === "uniform";
  $("#extract-hamming").disabled = pending || !$("#extract-dedup").checked;
  $("#dedup-settings").hidden = !$("#extract-dedup").checked;
  $("#extract-mode-hint").textContent = $("#extract-mode").value === "uniform"
    ? "Distribute the maximum frame count over the chosen range, including its first and last available frames. One frame samples the midpoint."
    : "Sample from the start at each interval. The maximum frame count can stop sampling before the end of the range.";
  const ready = preview.plan && preview.plan.planned_count > 0;
  $("#extract-submit").disabled = pending || !ready || preview.imagesPending;
  $("#extract-preview-images").disabled = pending || !ready || preview.imagesPending;
  $("#extract-preview-images").textContent = preview.imagesPending ? "Loading images…" : "Preview images";
  $("#extract-preview-retry").disabled = pending;
}

function scheduleExtractionPreview(delay = 220) {
  clearExtractionPreview();
  updateExtractionControls();
  $("#extract-error").hidden = true;
  let config;
  try {
    config = extractionConfig();
  } catch (error) {
    $("#extract-plan-status").textContent = error.message;
    return;
  }
  const preview = state.extractionPreview;
  const configKey = JSON.stringify(config);
  preview.configKey = configKey;
  const generation = preview.generation;
  const assetId = state.extracting;
  const sessionId = state.sessionId;
  $("#extract-plan-status").textContent = "Planning sampling…";
  preview.timer = setTimeout(async () => {
    preview.timer = null;
    try {
      const plan = await api(`/api/assets/${encodeURIComponent(assetId)}/extract/preview`, {
        method: "POST", body: configKey,
      });
      if (!extractionPreviewCurrent(generation, assetId, sessionId, configKey)) return;
      preview.plan = plan;
      renderExtractionPlan(plan);
    } catch (error) {
      if (!extractionPreviewCurrent(generation, assetId, sessionId, configKey)) return;
      $("#extract-plan-status").textContent = error.message;
      $("#extract-preview-retry").hidden = false;
    } finally {
      if (extractionPreviewCurrent(generation, assetId, sessionId, configKey)) updateExtractionControls();
    }
  }, delay);
}

function renderExtractionPlan(plan) {
  const count = plan.planned_count;
  $("#extract-plan-status").textContent =
    `${count} planned position${count === 1 ? "" : "s"} · first ≈ ${timestamp(plan.first_timestamp_seconds)} · last ≈ ${timestamp(plan.last_timestamp_seconds)}`;
  $("#extract-plan-range").textContent =
    `Sampling range ${timestamp(plan.start_seconds)}–${timestamp(plan.end_seconds)} · video ${timestamp(plan.duration_seconds)}`;
  $("#extract-plan-details").hidden = false;
  const timeline = $("#extract-timeline");
  timeline.replaceChildren();
  timeline.setAttribute("aria-label", `${count} planned sampling positions across a ${plan.duration_seconds.toFixed(2)} second video. First at ${timestamp(plan.first_timestamp_seconds)}, last at ${timestamp(plan.last_timestamp_seconds)}.`);
  const duration = plan.duration_seconds || 1;
  const percent = (seconds) => `${Math.max(0, Math.min(100, 100 * seconds / duration))}%`;
  const range = node("span", "extract-timeline-range");
  range.style.left = percent(plan.start_seconds);
  range.style.width = percent(plan.end_seconds - plan.start_seconds);
  timeline.append(range);
  for (const position of plan.positions) {
    const marker = node("span", "extract-timeline-position");
    marker.style.left = percent(position.timestamp_seconds);
    marker.title = `Frame ${position.frame_index} · ≈ ${timestamp(position.timestamp_seconds)}`;
    marker.setAttribute("aria-hidden", "true");
    timeline.append(marker);
  }
  $("#extract-video-end").textContent = timestamp(plan.duration_seconds);
  $("#extract-plan-warning").hidden = !plan.truncated;
  $("#extract-plan-warning").textContent =
    "The frame limit stops this interval sampling before the end of the range. Increase the limit, increase the interval or choose Across the whole range.";
}

async function previewExtractionImages() {
  const preview = state.extractionPreview;
  if (!preview.plan || preview.imagesPending || state.pendingExtractions.has(state.extracting)) return;
  const { generation, configKey } = preview;
  const assetId = state.extracting;
  const sessionId = state.sessionId;
  preview.imagesPending = true;
  updateExtractionControls();
  $("#extract-images-status").textContent = "Decoding up to 12 preview images locally…";
  try {
    const result = await api(`/api/assets/${encodeURIComponent(assetId)}/extract/preview-images`, {
      method: "POST", body: configKey,
    });
    if (!extractionPreviewCurrent(generation, assetId, sessionId, configKey)) return;
    for (const url of preview.imageUrls) URL.revokeObjectURL(url);
    preview.imageUrls = [];
    const thumbnails = $("#extract-thumbnails");
    thumbnails.replaceChildren();
    for (const thumbnail of result.thumbnails) {
      const prefix = "data:image/jpeg;base64,";
      if (!thumbnail.image_data_url.startsWith(prefix)) throw new Error("Invalid preview image format.");
      const bytes = Uint8Array.from(atob(thumbnail.image_data_url.slice(prefix.length)), (character) => character.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
      preview.imageUrls.push(url);
      const figure = node("figure");
      const image = node("img");
      image.src = url;
      image.alt = `Video preview at approximately ${timestamp(thumbnail.timestamp_seconds)}`;
      figure.append(image, node("figcaption", "", `≈ ${timestamp(thumbnail.timestamp_seconds)}`));
      thumbnails.append(figure);
    }
    $("#extract-images-status").textContent =
      `${result.thumbnails.length} of ${result.planned_count} planned positions shown. Preview only: no frames added and no scene analysis. Extracted frames remain unselected for your review.`;
  } catch (error) {
    if (extractionPreviewCurrent(generation, assetId, sessionId, configKey)) {
      $("#extract-images-status").textContent = `${error.message} Use Preview images to retry.`;
    }
  } finally {
    if (extractionPreviewCurrent(generation, assetId, sessionId, configKey)) {
      preview.imagesPending = false;
      updateExtractionControls();
    }
  }
}

function openInspection(id) {
  state.inspectionIds = visibleFrames().map((frame) => frame.id);
  state.inspectionContext = state.galleryFilters.source ? "Filtered source · chronological order" : "Filtered gallery at opening";
  state.inspecting = id;
  renderInspection();
  if (!$("#frame-dialog").open) $("#frame-dialog").showModal();
}

function renderInspectionSelection() {
  const frame = state.frames.find((item) => item.id === state.inspecting);
  if (!frame) return;
  const button = $("#inspect-select");
  button.textContent = frame.selected
    ? "✓ Selected for annotation"
    : "Select for annotation";
  button.setAttribute("aria-pressed", String(Boolean(frame.selected)));
  button.disabled =
    state.pendingSelections.has(frame.id) || state.bulkSelecting;
  button.classList.toggle("button-primary", !frame.selected);
  button.classList.toggle("button-secondary", Boolean(frame.selected));
}

function renderInspection() {
  const frame = state.frames.find((item) => item.id === state.inspecting);
  if (!frame) return;
  const source = sourceFor(frame);
  $("#inspect-image").src = projectURL(`/api/frames/${encodeURIComponent(frame.id)}/image`);
  $("#inspect-image").alt =
    `${source?.filename || "Frame"}${source?.kind === "video" ? ` at approximately ${timestamp(frame.timestamp_seconds)}` : ""}`;
  renderInspectionSelection();
  const position = state.inspectionIds.indexOf(frame.id);
  $("#inspect-position").textContent =
    `${position + 1} / ${state.inspectionIds.length}`;
  $("#inspect-browse-context").textContent = state.inspectionContext;
  const insight = state.insights.get(frame.id);
  $("#inspect-review-status").textContent = reviewLabel(insight) + (insight?.taxonomy_outdated ? " · uses an older class version" : "");
  const signal = $("#inspect-prediction-signal");
  signal.textContent = insight?.prediction_signal_status === "available"
    ? `${insight.low_confidence_count ?? 0} saved low confidence predictions (0.1 to below 0.5). ${insight.no_target_predictions === true ? "No target detections were saved; this is not a negative annotation." : "Model output is an inspection signal only."}${insight.prediction_mapping_complete === false ? " This source covers only some target classes." : ""}`
    : insight?.prediction_signal_reason || "No compatible saved detector signal. This does not describe the image's content.";
  if (insight?.annotation_error) $("#inspect-review-status").textContent += ` · ${insight.annotation_error}`;
  renderRelatedFrames(frame, insight);
  $("#previous-frame").disabled = position <= 0;
  $("#next-frame").disabled = position >= state.inspectionIds.length - 1;
  const session = state.sessions.find((item) => item.id === frame.session_id);
  const metadata = [
    ["Original source", source?.filename || frame.asset_id],
    ["Session", session?.name || frame.session_id],
    ["Dimensions", `${frame.width} × ${frame.height}`],
    [
      "Source position",
      source?.kind === "video"
        ? `Frame ${frame.frame_index} · ≈ ${timestamp(frame.timestamp_seconds)}`
        : "Original still image",
    ],
    ["Duplicate occurrences", String(frame.duplicate_count || 0)],
    ["Frame pixels · SHA-256", frame.sha256],
    ["Source file · SHA-256", source?.sha256 || "Unavailable"],
  ];
  if (insight?.prediction_source_id) metadata.push(
    ["Saved detector signal", insight.prediction_model_id || "Saved model"],
    ["Prediction record", insight.prediction_source_id],
    ["Target classes covered", (insight.prediction_target_class_ids || []).join(", ")],
  );
  if (frame.extraction?.sampling_mode === "passages") {
    const config = frame.extraction;
    const plan = config.passages_plan;
    metadata.splice(4, 0,
      ["Extraction settings", `Human-chosen passages · ${plan?.frames_per_passage ?? config.frames_per_passage} frames per passage · ${plan?.context_seconds ?? config.context_seconds}s context`],
      ["Regular coverage", `${plan?.coverage_frames ?? config.coverage_frames ?? 0} extra images requested across the original range`],
      ["Source review", config.video_review_id],
      ["Chosen passages", (config.passage_ids || []).join(", ")],
      ["Sampling method", config.sampling_algorithm || plan?.algorithm || "iris-video-passages-v1"],
      ["Similarity filtering", "Exact duplicates only"],
    );
  } else if (frame.extraction?.sampling_mode || frame.extraction?.interval_seconds != null) {
    const config = frame.extraction;
    const sampling = config.sampling_plan;
    const method = config.sampling_mode === "uniform"
      ? "Across the whole range"
      : `Every ${config.interval_seconds}s`;
    const end = sampling?.end_seconds ?? config.end_seconds;
    metadata.splice(
      4,
      0,
      [
        "Extraction settings",
        `${method} · start ${timestamp(config.start_seconds || 0)}${end == null ? "" : ` · end ${timestamp(end)}`} · ${config.max_frames} frames maximum`,
      ],
      [
        "Similarity filtering",
        config.dedup_hamming == null
          ? "Exact duplicates only"
          : `Hash distance ≤ ${config.dedup_hamming}`,
      ],
    );
  }
  const list = $("#inspect-metadata");
  list.replaceChildren();
  for (const [name, value] of metadata) {
    const entry = node("div");
    entry.append(
      node("dt", "", name),
      node("dd", name.includes("SHA-256") ? "monospace" : "", value),
    );
    list.append(entry);
  }
  const video = $("#source-video");
  video.pause();
  $("#video-preview").hidden = source?.kind !== "video";
  $("#video-error").hidden = true;
  if (source?.kind === "video") {
    const url = projectURL(`/api/assets/${encodeURIComponent(source.id)}/media`);
    if (video.getAttribute("src") !== url) video.src = url;
    seekOriginal();
  } else {
    video.removeAttribute("src");
    video.load();
  }
  $("#download-source").href =
    projectURL(`/api/assets/${encodeURIComponent(frame.asset_id)}/media`);
  $("#download-source").download = source?.filename || "original";
}

function renderRelatedFrames(frame, insight) {
  const container = $("#inspect-related");
  container.replaceChildren();
  for (const [key, title] of [["exact_duplicate_ids", "Identical image pixels"], ["similar_frame_ids", "Visually similar candidates"], ["neighbor_frame_ids", "Nearby video frames"]]) {
    const related = (insight?.[key] || []).map((id) => state.frames.find((item) => item.id === id)).filter(Boolean);
    if (!related.length) continue;
    const section = node("div", "inspect-related-group");
    const total = key === "exact_duplicate_ids" ? insight.exact_duplicate_count : key === "similar_frame_ids" ? insight.similar_frame_count : related.length;
    section.append(node("h3", "", title + (total > related.length ? ` · showing ${related.length} of ${total}` : "")));
    const list = node("div", "inspect-related-list");
    for (const item of related) {
      const source = sourceFor(item);
      const button = node("button", "inspect-related-frame");
      button.type = "button";
      const image = node("img");
      image.src = projectURL(`/api/frames/${encodeURIComponent(item.id)}/image`);
      image.alt = "";
      image.loading = "lazy";
      const text = `${source?.filename || item.id}${source?.kind === "video" ? ` · ${timestamp(item.timestamp_seconds)}` : ""}`;
      button.title = text;
      button.setAttribute("aria-label", `Inspect ${title.toLowerCase()}: ${text}`);
      button.append(image, node("span", "", text));
      button.addEventListener("click", () => {
        state.inspectionIds = [frame.id, ...related.map((row) => row.id)];
        state.inspectionContext = `${title} · may include frames outside the gallery filters`;
        state.inspecting = item.id;
        renderInspection();
        $("#inspect-image").scrollIntoView({ block: "nearest" });
      });
      list.append(button);
    }
    section.append(list);
    container.append(section);
  }
  if (!container.childElementCount) container.append(node("p", "field-hint", "No related frames in the available review signals."));
  else container.append(node("p", "field-hint", "Inspect these candidates yourself. Similarity does not establish scene identity. No frames are removed or validated automatically."));
}

function navigateInspection(direction) {
  const index = state.inspectionIds.indexOf(state.inspecting) + direction;
  if (index < 0 || index >= state.inspectionIds.length) return;
  state.inspecting = state.inspectionIds[index];
  renderInspection();
}

function seekOriginal() {
  const frame = state.frames.find((item) => item.id === state.inspecting);
  const video = $("#source-video");
  if (frame?.timestamp_seconds == null || video.readyState < 1) return;
  video.currentTime = frame.timestamp_seconds;
}

const jobTools = window.IRISJobTools;
const jobView = {
  limit: 8, detailId: null, detail: null, request: 0, loading: false,
  recoveryRequest: 0, recovery: null, busy: false, cache: new Map(),
};

function renderJobs() {
  const list = $("#job-list");
  const expanded = new Set([...list.querySelectorAll("details[open]")].map((details) => details.dataset.jobId));
  const focused = document.activeElement;
  const focusedJob = focused?.closest("[data-job-id]")?.dataset.jobId;
  const focusedAction = focused?.dataset.jobAction;
  list.replaceChildren();
  const active = state.jobs.filter(isActive);
  $("#jobs-section").hidden = false;
  $("#jobs-active-count").textContent = `${active.length} active · ${state.jobs.length} saved`;
  const history = jobTools.history(state.jobs, { status: $("#jobs-status").value, kind: $("#jobs-kind").value, query: $("#jobs-search").value, limit: jobView.limit });
  $("#jobs-history-status").textContent = `${history.rows.length} of ${history.total} matching tasks · current project`;
  $("#jobs-more").hidden = !history.more;
  if (!history.rows.length) list.append(node("p", "job-history-empty", state.jobs.length ? "No saved jobs match these filters." : "No jobs saved in this project yet."));
  for (const job of history.rows) {
    const detail = jobView.cache.get(job.id);
    const asset = state.assets.find((item) => item.id === job.params?.asset_id);
    const row = node("article", "job-row");
    row.dataset.jobId = job.id;
    const header = node("div", "job-header");
    header.append(node("h3", "job-name", detail?.context?.name || asset?.filename || `${jobTools.kindNames[job.kind] || job.kind} · ${String(job.id).slice(0, 8)}`), node("span", `job-status ${job.status}`, jobTools.statusName(job.status)));
    row.append(header, node("p", "job-timestamp", `${new Date(job.created_at).toLocaleString()}${detail?.context?.session_name ? ` · ${detail.context.session_name}` : ""}`));
    if (job.message) row.append(node("p", "job-message", job.message));
    if (detail?.dispatch?.state === "outcome_unknown") row.append(node("p", "job-outcome-unknown", "Delivery outcome unknown · no automatic resend"));
    if (job.kind === "extract" && job.result) {
      const result = job.result;
      const parts = [];
      const continued = Boolean(job.params?.recovery_of || result.inherited_completed_count > 0);
      const planned = result.plan?.planned_count ?? result.planned_count;
      if (planned != null) parts.push(`${planned} planned`);
      if (result.sampled != null) parts.push(`${result.sampled} ${continued ? "total positions processed" : "sampled"}`);
      if (result.created != null) parts.push(`${result.created} ${continued ? "total images saved across linked attempts, including retained results" : "added for review"}`);
      for (const [key, label] of [["skipped_existing", "already extracted"], ["skipped_exact", "exact duplicates"], ["skipped_similar", "visually similar"]]) if (result[key] != null) parts.push(`${result[key]} ${label}`);
      if (parts.length) row.append(node("p", "job-message", parts.join(" · ")));
    }
    if (isActive(job)) {
      const progress = node("progress", "job-progress");
      progress.max = 1;
      if (typeof job.progress === "number" && Number.isFinite(job.progress)) progress.value = Math.max(0, Math.min(1, job.progress));
      progress.setAttribute("aria-label", `Processing progress for ${asset?.filename || job.id}`);
      row.append(progress);
    }
    if (job.error) row.append(node("p", "job-error", job.error));
    const bottom = node("div", "job-bottom");
    const details = node("button", "button button-secondary", "Details and next action");
    details.type = "button";
    details.dataset.jobAction = "details";
    details.setAttribute("aria-label", `Details and next action for job ${job.id}`);
    details.addEventListener("click", () => openJobDetails(job.id));
    bottom.append(details);
    if (job.logs?.length || job.started_at) {
      const logs = node("details", "job-logs");
      logs.dataset.jobId = String(job.id);
      logs.open = expanded.has(String(job.id));
      const download = node("a", "text-button", "Download full worker log ↗");
      download.href = projectURL(`/api/jobs/${encodeURIComponent(job.id)}/log`);
      download.download = `${job.id}.log`;
      logs.append(node("summary", "", "View processing log"), node("pre", "", job.logs?.length ? job.logs.join("\n") : "No progress messages recorded."), download);
      bottom.append(logs);
    }
    if (isActive(job)) {
      const cancel = node("button", "text-button", job.cancel_requested ? "Cancellation requested" : "Cancel job");
      cancel.type = "button";
      cancel.disabled = Boolean(job.cancel_requested);
      cancel.dataset.jobAction = "cancel";
      cancel.addEventListener("click", () => cancelSavedJob(job.id, cancel));
      bottom.append(cancel);
    }
    row.append(bottom);
    list.append(row);
  }
  if (focusedJob && focusedAction) {
    const row = [...list.children].find((item) => item.dataset.jobId === focusedJob);
    row?.querySelector(`[data-job-action="${focusedAction}"]`)?.focus({ preventScroll: true });
  }
}

function jobError(selector, error) {
  $(selector).textContent = error?.message || error || "";
  $(selector).hidden = !error;
}

function jobMetadata(selector, pairs) {
  const list = $(selector);
  list.replaceChildren();
  for (const [name, value] of pairs) {
    if (value == null || value === "") continue;
    const entry = node("div");
    entry.append(node("dt", "", name), node("dd", "", String(value)));
    list.append(entry);
  }
}

function updateJobActions() {
  const detail = jobView.detail;
  const blocked = jobView.loading || jobView.busy;
  $("#job-detail-dialog [data-close]").disabled = jobView.busy;
  $("#job-detail-refresh").disabled = blocked || !jobView.detailId;
  $("#job-detail-content").setAttribute("aria-busy", String(jobView.loading));
  for (const id of ["job-open-results", "job-check-continuation", "job-new-run", "job-open-batch"])
    $(`#${id}`).disabled = blocked || !detail;
  $("#job-confirm-continuation").disabled = blocked || !jobTools.canContinue(detail, jobView.recovery);
  $("#job-confirm-continuation").textContent = jobTools.continuationPresentation(detail, jobView.recovery).label;
  $("#job-detail-cancel").disabled = blocked || Boolean(detail?.job.cancel_requested);
}

function renderJobDetail(detail) {
  const { job, context, dispatch, artifacts, lineage, recovery, next_action: action } = detail;
  $("#job-detail-content").hidden = false;
  $("#job-detail-title").textContent = context?.name || jobTools.kindNames[job.kind] || "Job details";
  $("#job-detail-status").textContent = jobTools.statusName(job.status);
  $("#job-detail-status").className = `job-status ${job.status}`;
  $("#job-detail-context").textContent = `${jobTools.kindNames[job.kind] || job.kind} · ${context?.session_name || "Project task"} · ${job.id}`;
  $("#job-detail-message").textContent = job.message || "";
  jobError("#job-detail-failure", job.error);
  const time = (value) => value ? new Date(value).toLocaleString() : null;
  jobMetadata("#job-detail-times", [["Created", time(job.created_at)], ["Started", time(job.started_at)], ["Finished", time(job.finished_at)], ["Cancellation", job.cancel_requested ? "Requested; saved results are preserved" : null]]);
  const presentation = jobTools.dispatchPresentation(dispatch);
  $("#job-detail-dispatch").hidden = !presentation;
  if (presentation) {
    $("#job-detail-dispatch").classList.toggle("unknown", presentation.unknown);
    $("#job-dispatch-title").textContent = presentation.label;
    $("#job-dispatch-message").textContent = dispatch.message || "";
    $("#job-dispatch-explanation").textContent = presentation.explanation;
    jobMetadata("#job-dispatch-metadata", [["Execution", dispatch.external ? "External provider" : "Local provider"], ["Provider", dispatch.provider], ["Model", dispatch.model], ["Dispatch attempted", time(dispatch.attempted_at)], ["Response received", time(dispatch.response_received_at)], ["Provider request ID", dispatch.request_id]]);
  }
  $("#job-detail-artifacts").replaceChildren(...(artifacts || []).map((item) => node("li", "", `${item.count ?? ""} ${item.label}`.trim())));
  if (!artifacts?.length) $("#job-detail-artifacts").append(node("li", "field-hint", "No saved result artifacts were reported."));
  $("#job-open-results").hidden = !action?.workspace;
  const links = $("#job-detail-lineage");
  links.replaceChildren();
  for (const [label, ids] of [["Original task", lineage?.parent_job_id ? [lineage.parent_job_id] : []], ["Continuation", lineage?.child_job_ids || []]]) {
    for (const id of ids) {
      const button = node("button", "text-button", `${label} · ${id.slice(0, 8)}`);
      button.type = "button";
      button.addEventListener("click", () => openJobDetails(id));
      links.append(button);
    }
  }
  $("#job-check-continuation").hidden = !recovery?.can_check;
  $("#job-new-run").hidden = isActive(job) || !action?.workspace || Boolean(context?.batch_id);
  $("#job-open-batch").hidden = !context?.batch_id;
  $("#job-detail-cancel").hidden = !isActive(job);
  $("#job-detail-cancel").textContent = job.cancel_requested ? "Cancellation requested" : "Cancel job";
  const notes = [recovery?.reason, action?.reason];
  if (job.kind === "train") notes.push("Preparing a new run does not resume optimizer state.");
  if (dispatch?.external) notes.push("A new external review requires a new image and cost preview with explicit approval.");
  notes.push("Opening a workspace does not launch a task.");
  $("#job-recovery-reason").textContent = [...new Set(notes.filter(Boolean))].join(" ");
  $("#job-detail-record").textContent = JSON.stringify({ params: job.params, result: job.result }, null, 2);
  $("#job-detail-log").textContent = job.logs?.join("\n") || "No progress messages recorded.";
  $("#job-detail-log-download").href = projectURL(`/api/jobs/${encodeURIComponent(job.id)}/log`);
  $("#job-detail-log-download").download = `${job.id}.log`;
  updateJobActions();
}

async function loadJobDetails(id) {
  const request = ++jobView.request;
  jobView.loading = true;
  $("#job-detail-loading").textContent = "Loading saved task evidence…";
  updateJobActions();
  try {
    const detail = await api(`/api/jobs/${encodeURIComponent(id)}`);
    if (request !== jobView.request || id !== jobView.detailId || !$("#job-detail-dialog").open) return;
    jobView.detail = detail;
    jobView.cache.set(id, detail);
    jobError("#job-detail-error", null);
    renderJobDetail(detail);
    renderJobs();
  } catch (error) {
    if (request === jobView.request && id === jobView.detailId) jobError("#job-detail-error", error);
  } finally {
    if (request === jobView.request) {
      jobView.loading = false;
      $("#job-detail-loading").textContent = "";
      updateJobActions();
    }
  }
}

function openJobDetails(id) {
  if (jobView.busy) return;
  jobView.detailId = id;
  jobView.detail = null;
  jobView.recovery = null;
  jobView.recoveryRequest++;
  $("#job-detail-content").hidden = true;
  $("#job-recovery-preview").hidden = true;
  jobError("#job-recovery-error", null);
  jobError("#job-detail-error", null);
  if (!$("#job-detail-dialog").open) $("#job-detail-dialog").showModal();
  loadJobDetails(id);
}

async function cancelSavedJob(id, button) {
  if (button) button.disabled = true;
  try {
    await api(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
    await refreshJobs();
    if (jobView.detailId === id && $("#job-detail-dialog").open) await loadJobDetails(id);
  } catch (error) {
    notify(error.message, true);
    if (button) button.disabled = false;
  }
}

async function checkJobContinuation() {
  const detail = jobView.detail;
  if (!detail?.recovery?.can_check || jobView.busy) return;
  const id = detail.job.id;
  const request = ++jobView.recoveryRequest;
  jobView.busy = true;
  jobView.recovery = null;
  jobError("#job-recovery-error", null);
  updateJobActions();
  try {
    const preview = await api(`/api/jobs/${encodeURIComponent(id)}/recovery`);
    if (request !== jobView.recoveryRequest || id !== jobView.detailId) return;
    jobView.recovery = preview;
    const presentation = jobTools.continuationPresentation(detail, preview);
    $("#job-recovery-preview").hidden = false;
    $("#job-recovery-summary").textContent = presentation.summary;
    $("#job-recovery-notice").textContent = presentation.notice;
    $("#job-recovery-notice").hidden = !preview.available;
    if (preview.successor_job_id) $("#job-recovery-summary").append(document.createTextNode(` Existing continuation: ${preview.successor_job_id}. Open it from the task links.`));
    await loadJobDetails(id);
  } catch (error) {
    if (request === jobView.recoveryRequest) jobError("#job-recovery-error", error);
  } finally {
    if (request === jobView.recoveryRequest) { jobView.busy = false; updateJobActions(); }
  }
}

async function continueJob() {
  const detail = jobView.detail;
  const preview = jobView.recovery;
  if (jobView.busy || !jobTools.canContinue(detail, preview)) return;
  const id = detail.job.id;
  jobView.busy = true;
  jobError("#job-recovery-error", null);
  updateJobActions();
  let created;
  try {
    created = await api(`/api/jobs/${encodeURIComponent(id)}/recover`, { method: "POST", body: JSON.stringify({ fingerprint: preview.fingerprint }) });
    jobView.recovery = null;
    await refreshJobs();
    notify(jobTools.continuationPresentation(detail, preview).queuedMessage);
  } catch (error) {
    jobView.recovery = null;
    $("#job-recovery-preview").hidden = true;
    jobError("#job-recovery-error", `${error.message} Refresh this task and check its continuation links before trying again. No request was repeated automatically.`);
    await loadJobDetails(id);
  } finally {
    jobView.busy = false;
    updateJobActions();
  }
  if (created?.id) openJobDetails(created.id);
}

async function selectJobHistory(selector, id) {
  const sessionId = state.sessionId;
  const select = $(selector);
  if (!select || !id) return;
  const ready = () => [...select.options].some((option) => option.value === id);
  if (!ready()) await new Promise((resolve) => {
    const observer = new MutationObserver(() => { if (ready()) { observer.disconnect(); clearTimeout(timer); resolve(); } });
    const timer = setTimeout(() => { observer.disconnect(); resolve(); }, 5000);
    observer.observe(select, { childList: true, subtree: true });
  });
  if (ready() && sessionId === state.sessionId) { select.value = id; select.dispatchEvent(new Event("change", { bubbles: true })); }
}

async function openJobWorkspace({ results = false, batch = false } = {}) {
  const detail = jobView.detail;
  if (!detail || jobView.busy || !detail.next_action?.workspace) return;
  const context = detail.context || {};
  if (context.session_id && context.session_id !== state.sessionId) {
    await selectSession(context.session_id);
    if (state.sessionId !== context.session_id) return;
  }
  $("#job-detail-dialog").close();
  if (!window.IRISNavigation.open(detail.next_action.workspace)) return;
  if (detail.next_action.workspace === "training" && !$("#training-workspace").hidden) {
    const view = context.target_type === "model_exports" ? "exports" : results ? "runs" : "plan";
    window.IRISTrainingNavigation.open(view, { focus: true });
  }
  if (detail.next_action.workspace === "evaluation" && !$("#evaluation-workspace").hidden)
    window.IRISEvaluationNavigation.open(results ? "results" : "plan", { focus: true });
  if (detail.next_action.workspace === "benchmark" && !$("#benchmark-workspace").hidden && context.target_type === "benchmark_trials") {
    const opened = await window.IRISBenchmarkNavigation.openTrial(context.target_id);
    if (!opened) { notify("The saved benchmark trial could not be opened. Check the selected reference or use Refresh to try again.", true); return; }
  }
  $(`#${detail.next_action.workspace}-workspace`)?.scrollIntoView({ block: "start" });
  if (detail.next_action.workspace === "tracking") {
    const comparisonId = context.comparison_id;
    const sequenceId = context.sequence_id;
    if (context.tracking_study_id) window.dispatchEvent(new CustomEvent("iris:tracking-study-open", { detail: { study_id: context.tracking_study_id } }));
    else if (context.tracking_cost_id) window.dispatchEvent(new CustomEvent("iris:tracking-cost-open", { detail: { run_id: context.tracking_cost_id, comparison_id: comparisonId } }));
    else if (comparisonId) window.dispatchEvent(new CustomEvent("iris:tracking-comparison-open", { detail: { comparison_id: comparisonId } }));
    else if (sequenceId) window.dispatchEvent(new CustomEvent("iris:tracking-sequence-open", { detail: { sequence_id: sequenceId } }));
    return;
  }
  if ((batch || (results && detail.job.kind === "dinox")) && context.batch_id) {
    window.dispatchEvent(new CustomEvent(detail.job.kind === "dinox" ? "iris:dinox-batch-open" : "iris:assistance-batch-open", { detail: { batch_id: context.batch_id } }));
    return;
  }
  let resultNotice = "Opened the saved results workspace. No task was launched.";
  if (results) {
    const selector = { comparisons: "#comparison-history", training_runs: "#training-history", evaluations: "#evaluation-history", model_exports: "#model-export-history" }[context.target_type];
    if (selector) await selectJobHistory(selector, context.target_id);
    else if (context.target_type === "assistance_records" && detail.job.params?.frame_id) {
      const frameId = detail.job.params.frame_id;
      if (state.frames.some((frame) => frame.id === frameId && frame.selected)) window.dispatchEvent(new CustomEvent("iris:annotation-open-frame", { detail: { frame_id: frameId } }));
      else resultNotice = "This saved frame is not selected. Select it in Data intake to open its annotation; no frame selection was changed.";
    }
  }
  if (detail.next_action.workspace === "intake") {
    let assetId = detail.job.params?.asset_id;
    if (!assetId && context.target_type === "assets") assetId = context.target_id;
    if (!assetId && context.target_type === "video_reviews" && context.target_id) {
      const record = await api(`/api/video-reviews/${encodeURIComponent(context.target_id)}`);
      if (context.session_id && state.sessionId !== context.session_id) return;
      assetId = record.asset_id;
    }
    const asset = state.assets.find((item) => item.id === assetId);
    if (asset && detail.job.kind === "video_review") {
      window.dispatchEvent(new CustomEvent("iris:video-review", { detail: { asset } }));
      if (results) await selectJobHistory("#video-review-history", context.target_id);
    } else if (asset && !results) openExtraction(asset);
  }
  notify(results ? resultNotice : "Review the settings in this workspace to prepare a new run. Nothing was launched; saved work remains unchanged.");
}

for (const id of ["jobs-status", "jobs-kind", "jobs-search"]) $(`#${id}`).addEventListener(id === "jobs-search" ? "input" : "change", () => { jobView.limit = 8; renderJobs(); });
$("#jobs-more").addEventListener("click", () => { jobView.limit += 12; renderJobs(); });
$("#jobs-refresh").addEventListener("click", async () => {
  $("#jobs-refresh").disabled = true;
  try { await refreshJobs(); } catch (error) { notify(error.message, true); }
  finally { $("#jobs-refresh").disabled = false; }
});
$("#job-detail-refresh").addEventListener("click", () => {
  jobView.recovery = null;
  $("#job-recovery-preview").hidden = true;
  loadJobDetails(jobView.detailId);
});
$("#job-check-continuation").addEventListener("click", checkJobContinuation);
$("#job-confirm-continuation").addEventListener("click", continueJob);
$("#job-open-results").addEventListener("click", () => openJobWorkspace({ results: true }).catch((error) => notify(error.message, true)));
$("#job-new-run").addEventListener("click", () => openJobWorkspace().catch((error) => notify(error.message, true)));
$("#job-open-batch").addEventListener("click", () => openJobWorkspace({ batch: true }).catch((error) => notify(error.message, true)));
$("#job-detail-cancel").addEventListener("click", () => cancelSavedJob(jobView.detailId, $("#job-detail-cancel")));
$("#job-detail-dialog").addEventListener("cancel", (event) => { if (jobView.busy) event.preventDefault(); });
$("#job-detail-dialog").addEventListener("close", () => {
  jobView.request++;
  jobView.recoveryRequest++;
  jobView.loading = false;
  jobView.recovery = null;
});
window.addEventListener("iris:job-open", (event) => { if (event.detail?.job_id) openJobDetails(event.detail.job_id); });

async function refreshJobs() {
  if (state.polling) clearTimeout(state.polling);
  state.polling = null;
  const previouslyActive = state.jobs.some(isActive);
  const previousStatuses = new Map(
    state.jobs.map((job) => [job.id, job.status]),
  );
  state.jobs = await api("/api/jobs");
  renderJobs();
  const shownJob = state.jobs.find((job) => job.id === jobView.detailId);
  if ($("#job-detail-dialog").open && !jobView.loading && !jobView.busy && shownJob &&
      (shownJob.status !== jobView.detail?.job.status || shownJob.progress !== jobView.detail?.job.progress)) loadJobDetails(shownJob.id);
  window.dispatchEvent(new Event("iris:jobs"));
  const changed = state.jobs.some(
    (job) => previousStatuses.get(job.id) !== job.status,
  );
  const changedSignals = state.jobs.some((job) => ["infer", "assist", "dinox", "extract"].includes(job.kind) && !isActive(job) && previousStatuses.get(job.id) !== job.status);
  if (previouslyActive || state.jobs.some(isActive) || changed)
    await refreshSession(changedSignals);
  schedulePolling();
}

function schedulePolling() {
  if (state.jobs.some(isActive)) {
    state.polling = setTimeout(
      () =>
        refreshJobs().catch((error) => {
          notify(error.message, true);
          schedulePolling();
        }),
      1800,
    );
  }
}

$("#session-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (uploadNavigationBlocked()) return;
  const button = $("button[type=submit]", event.currentTarget);
  const name = $("#session-name").value.trim();
  const sceneGroup = $("#scene-group").value.trim();
  if (!name) return $("#session-name").focus();
  if (!sceneGroup) return $("#scene-group").focus();
  button.disabled = true;
  try {
    const session = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ name, scene_group: sceneGroup }),
    });
    state.sessions = await api("/api/sessions");
    $("#session-form").reset();
    await selectSession(session.id);
    notify(
      `Session “${session.name}” is ready. Import your source files to begin.`,
    );
  } catch (error) {
    notify(error.message, true);
  } finally {
    button.disabled = false;
  }
});

$("#start-session").addEventListener("click", () => $("#session-name").focus());

function uploadFile(file, sessionId, progress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", projectURL(`/api/sessions/${encodeURIComponent(sessionId)}/assets`));
    request.upload.addEventListener("progress", (event) => progress({ loaded: event.loaded, total: event.lengthComputable ? event.total : null }));
    request.upload.addEventListener("load", () => progress({ processing: true }));
    request.addEventListener("load", () => {
      let response;
      try { response = JSON.parse(request.responseText); } catch { /* Invalid responses remain visible as failures. */ }
      if (request.status >= 200 && request.status < 300 && response?.id) resolve(response);
      else {
        const detail = response?.detail;
        reject(new Error(typeof detail === "string" ? detail : `Import failed (${request.status}). Check the source list before retrying.`));
      }
    });
    request.addEventListener("error", () => reject(new Error("Connection lost. Check the source list before retrying; the server may already have received this file.")));
    const body = new FormData();
    body.append("file", file);
    request.send(body);
  });
}

function renderUploadQueue(queue) {
  $("#upload-queue").hidden = !queue.entries.length;
  $("#file-input").disabled = queue.busy;
  $(".upload-trigger").classList.toggle("disabled", queue.busy);
  $("#source-dropzone").setAttribute("aria-disabled", String(queue.busy));
  const sessionName = state.sessions.find((session) => session.id === queue.sessionId)?.name || "this session";
  const complete = queue.entries.filter((entry) => ["succeeded", "existing"].includes(entry.status)).length;
  const failed = queue.entries.filter((entry) => entry.status === "failed").length;
  const cancelled = queue.entries.filter((entry) => entry.status === "cancelled").length;
  $("#upload-progress").textContent = `${queue.busy ? "Importing into" : "Import results for"} ${sessionName} · ${complete} completed · ${failed} failed${cancelled ? ` · ${cancelled} cancelled` : ""}`;
  $("#upload-retry").disabled = queue.busy || !failed;
  $("#upload-cancel").disabled = !queue.busy || !queue.entries.some((entry) => entry.status === "pending");
  $("#project-select").disabled = queue.busy || !state.projects.length;
  for (const button of document.querySelectorAll(".session-item, #session-form button[type=submit], #project-form button[type=submit]")) button.disabled = queue.busy;
  const list = $("#upload-files");
  list.replaceChildren();
  for (const entry of queue.entries) {
    const row = node("li", `upload-file ${entry.status}`);
    const heading = node("div", "upload-file-heading");
    const name = node("strong", "", entry.name);
    name.title = entry.name;
    const labels = { pending: "Queued", uploading: "Sending file…", processing: "Processing on this computer…", succeeded: "Imported", existing: "Already imported · source reused", failed: "Failed", cancelled: "Cancelled before upload" };
    let label = labels[entry.status];
    if (entry.status === "uploading" && entry.total) label += ` ${Math.floor(100 * entry.loaded / entry.total)}%`;
    heading.append(name, node("span", "small", label));
    row.append(heading);
    if (["uploading", "processing"].includes(entry.status)) {
      const bar = node("progress");
      bar.setAttribute("aria-label", `Import progress for ${entry.name}`);
      if (entry.status === "uploading" && entry.total) { bar.max = entry.total; bar.value = entry.loaded; }
      row.append(bar);
    }
    if (entry.error) row.append(node("p", "job-error", entry.error));
    list.append(row);
  }
}

const uploadQueue = window.IRISIntakeTools.createUploadQueue({ upload: uploadFile, onChange: renderUploadQueue });
async function startUploads(files, retry = false) {
  if (uploadQueue.state.busy || !state.sessionId || (!retry && !files.length)) return;
  const sessionId = retry ? uploadQueue.state.sessionId : state.sessionId;
  await (retry ? uploadQueue.retryFailed() : uploadQueue.start(files, sessionId));
  if (state.sessionId === sessionId) await refreshSession().catch((error) => notify(error.message, true));
}
$("#file-input").addEventListener("change", (event) => {
  const files = [...event.target.files];
  event.target.value = "";
  startUploads(files);
});
$("#upload-retry").addEventListener("click", () => startUploads([], true));
$("#upload-cancel").addEventListener("click", () => uploadQueue.cancelRemaining());
const dropzone = $("#source-dropzone");
dropzone.addEventListener("click", () => { if (!uploadQueue.state.busy) $("#file-input").click(); });
dropzone.addEventListener("keydown", (event) => {
  if (["Enter", " "].includes(event.key)) { event.preventDefault(); dropzone.click(); }
});
for (const type of ["dragenter", "dragover"]) dropzone.addEventListener(type, (event) => {
  event.preventDefault();
  if (!uploadQueue.state.busy) dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", (event) => { if (!dropzone.contains(event.relatedTarget)) dropzone.classList.remove("dragging"); });
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  if (uploadNavigationBlocked()) return;
  startUploads([...event.dataTransfer.files]);
});
window.addEventListener("beforeunload", (event) => {
  if (!uploadQueue.state.busy && !state.bulkSelecting && !state.pendingSelections.size) return;
  event.preventDefault();
  event.returnValue = "";
});
for (const key of ["source", "search", "review", "signal"]) $(`#gallery-${key}`).addEventListener(key === "search" ? "input" : "change", (event) => {
  state.galleryFilters[key] = event.target.value;
  if (key === "source") event.target.title = event.target.selectedOptions[0]?.textContent || "All sources";
  renderFrames();
});
$("#gallery-refresh").addEventListener("click", () => refreshSession(true).catch((error) => notify(error.message, true)));
$("#intake-load-retry").addEventListener("click", async () => {
  const sessionId = state.sessionId;
  $("#intake-load-retry").disabled = true;
  try { await refreshSession(true); }
  catch (error) { if (state.sessionId === sessionId) notify(error.message, true); }
  finally { $("#intake-load-retry").disabled = false; }
});
$("#intake-review-selection").addEventListener("click", () => {
  if (!state.sessionId || state.collectionLoading || state.bulkSelecting || state.pendingSelections.size || !state.frames.some((frame) => frame.selected)) return;
  if (window.IRISNavigation.open("annotation")) $("#main").focus();
});
$("#gallery-reset").addEventListener("click", () => {
  state.filter = "all";
  state.galleryFilters = { source: "", search: "", review: "all", signal: "all" };
  for (const key of ["source", "search", "review", "signal"]) $(`#gallery-${key}`).value = state.galleryFilters[key];
  renderGallerySources();
  renderFrames();
});
window.addEventListener("iris:workspace", (event) => {
  if (event.detail.name === "intake" && state.sessionId) refreshSession(true).catch((error) => notify(error.message, true));
});

for (const filter of ["all", "selected"]) {
  $(`#filter-${filter}`).addEventListener("click", () => {
    state.filter = filter;
    renderFrames();
  });
}
$("#select-visible").addEventListener("click", () => bulkSelection(true));
$("#clear-selection").addEventListener("click", () => bulkSelection(false));
$("#extract-form").addEventListener("input", (event) => {
  if (event.target.matches("input, select")) scheduleExtractionPreview();
});
$("#extract-preview-retry").addEventListener("click", () => scheduleExtractionPreview(0));
$("#extract-preview-images").addEventListener("click", previewExtractionImages);
$("#extract-dialog").addEventListener("close", () => {
  if ($("#extract-dialog").open) return;
  clearExtractionPreview();
  state.extracting = null;
  updateExtractionControls();
});

$("#extract-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const preview = state.extractionPreview;
  const assetId = state.extracting;
  const sessionId = state.sessionId;
  const generation = preview.generation;
  const configKey = preview.configKey;
  if (!preview.plan || preview.imagesPending || state.pendingExtractions.has(assetId)) return;
  const errorBox = $("#extract-error");
  let config;
  try {
    config = extractionConfig();
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
    return;
  }
  if (JSON.stringify(config) !== configKey) {
    scheduleExtractionPreview(0);
    return;
  }
  state.pendingExtractions.add(assetId);
  updateExtractionControls();
  errorBox.hidden = true;
  try {
    await api(`/api/assets/${encodeURIComponent(assetId)}/extract`, {
      method: "POST", body: configKey,
    });
    if (extractionPreviewCurrent(generation, assetId, sessionId, configKey)) $("#extract-dialog").close();
    notify("Extraction queued. Review the unselected frames in the gallery when it finishes.");
    await refreshJobs();
  } catch (error) {
    if (extractionPreviewCurrent(generation, assetId, sessionId, configKey)) {
      errorBox.textContent = error.message;
      errorBox.hidden = false;
    } else notify(error.message, true);
  } finally {
    state.pendingExtractions.delete(assetId);
    updateExtractionControls();
  }
});

for (const button of document.querySelectorAll("[data-close]")) {
  button.addEventListener("click", () =>
    document.getElementById(button.dataset.close).close(),
  );
}
$("#previous-frame").addEventListener("click", () => navigateInspection(-1));
$("#next-frame").addEventListener("click", () => navigateInspection(1));
$("#seek-source").addEventListener("click", seekOriginal);
$("#source-video").addEventListener("loadedmetadata", seekOriginal);
$("#source-video").addEventListener("error", () => {
  $("#video-error").hidden = false;
});
$("#inspect-select").addEventListener("click", () => {
  const frame = state.frames.find((item) => item.id === state.inspecting);
  if (frame)
    updateSelection(frame.id, !frame.selected).catch((error) =>
      notify(error.message, true),
    );
});
$("#frame-dialog").addEventListener("close", () => {
  $("#source-video").pause();
  state.inspecting = null;
});
$("#frame-dialog").addEventListener("keydown", (event) => {
  if (["INPUT", "TEXTAREA", "VIDEO"].includes(event.target.tagName)) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    navigateInspection(event.key === "ArrowLeft" ? -1 : 1);
  } else if (
    event.key.toLowerCase() === "s" &&
    !event.ctrlKey &&
    !event.metaKey &&
    !event.altKey
  ) {
    event.preventDefault();
    $("#inspect-select").click();
  }
});

async function initialize() {
  try {
    const [system, projects] = await Promise.all([
      api("/api/system"),
      api("/api/projects"),
    ]);
    $("#version").textContent = system.version ? `v${system.version}` : "";
    $("#storage-path").textContent = system.data_dir;
    state.projects = projects;
    if (!projects.some((project) => project.id === state.projectId)) {
      window.location.replace(projectScope.location(projects[0]?.id || "default"));
      return;
    }
    projectScope.remember();
    // Keep browser history and separate tabs pinned to their own project.
    window.history.replaceState(null, "", projectScope.location(state.projectId));
    renderProjects();
    window.dispatchEvent(new Event("iris:project-ready"));
    const sessions = await api("/api/sessions");
    state.sessions = sessions;
    renderSessions();
    const recalled = recalledSession();
    const active =
      sessions.find((session) => String(session.id) === recalled) ||
      sessions[0];
    if (active) await selectSession(active.id);
    else if (window.IRISNavigation) window.IRISNavigation.syncSession();
    else $("#welcome").hidden = false;
    await refreshJobs();
    state.projectInitialized = true;
    window.dispatchEvent(new Event("iris:project-initialized"));
  } catch (error) {
    notify(error.message, true);
    $("#storage-path").textContent = "Local server unavailable";
    if (!state.sessions.length) {
      renderSessions();
      if (window.IRISNavigation) window.IRISNavigation.syncSession();
      else $("#welcome").hidden = false;
    }
  }
}

$("#project-select").addEventListener("change", (event) => openProject(event.target.value));
$("#project-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("button[type=submit]", event.currentTarget);
  if (uploadNavigationBlocked()) return;
  const name = $("#project-name").value.trim();
  if (!name) return $("#project-name").focus();
  button.disabled = true;
  $("#project-error").hidden = true;
  try {
    const project = await api("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name, description: $("#project-new-description").value.trim() }),
    });
    state.projects.push(project);
    renderProjects();
    $("#project-form").reset();
    $("#project-create").open = false;
    notify(`Project “${project.name}” is ready.`);
    openProject(project.id);
  } catch (error) {
    $("#project-error").textContent = error.message;
    $("#project-error").hidden = false;
  } finally {
    button.disabled = false;
  }
});

initialize();
