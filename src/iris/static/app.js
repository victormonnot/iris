"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const state = {
  sessions: [],
  sessionId: null,
  assets: [],
  frames: [],
  jobs: [],
  filter: "all",
  inspecting: null,
  inspectionIds: [],
  extracting: null,
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
    response = await fetch(path, { ...options, headers });
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
  return state.frames.filter(
    (frame) => state.filter !== "selected" || frame.selected,
  );
}

function isActive(job) {
  return job.status === "queued" || job.status === "running";
}

function rememberSession(id) {
  try {
    localStorage.setItem("iris.session", String(id));
  } catch {
    /* Storage is optional. */
  }
}

function recalledSession() {
  try {
    return localStorage.getItem("iris.session");
  } catch {
    return null;
  }
}

function renderSessions() {
  const list = $("#session-list");
  list.replaceChildren();
  $("#session-count").textContent = state.sessions.length;
  if (!state.sessions.length) {
    list.append(node("p", "session-empty", "No flight sessions yet."));
  }
  for (const session of state.sessions) {
    const button = node(
      "button",
      `session-item${session.id === state.sessionId ? " active" : ""}`,
    );
    button.type = "button";
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
    const event = new CustomEvent("iris:before-session", {
      cancelable: true,
      detail: { sessionId: id },
    });
    if (!window.dispatchEvent(event)) return;
  }
  state.sessionId = id;
  state.assets = [];
  state.frames = [];
  state.filter = "all";
  state.inspecting = null;
  rememberSession(id);
  renderSessions();
  const session = state.sessions.find((item) => item.id === id);
  $("#welcome").hidden = true;
  $("#session-workspace").hidden = false;
  $("#active-session-name").textContent = session?.name || "Flight session";
  $("#active-session-group").textContent = session?.scene_group
    ? `Scene group / ${session.scene_group}`
    : "No scene group assigned";
  renderAssets();
  renderFrames();
  window.dispatchEvent(new Event("iris:session"));
  $("#asset-list").replaceChildren(
    node("p", "empty-assets", "Loading source files…"),
  );
  await refreshSession();
}

async function refreshSession() {
  if (!state.sessionId) return;
  const id = state.sessionId;
  const request = ++state.frameFetch;
  const [assets, frames] = await Promise.all([
    api(`/api/sessions/${encodeURIComponent(id)}/assets`),
    api(`/api/sessions/${encodeURIComponent(id)}/frames`),
  ]);
  if (state.sessionId !== id || request !== state.frameFetch) return;
  const assetsChanged = JSON.stringify(state.assets) !== JSON.stringify(assets);
  state.assets = assets;
  state.frames = frames.map((frame) =>
    state.pendingSelections.has(frame.id)
      ? { ...frame, selected: state.pendingSelections.get(frame.id) }
      : frame,
  );
  if (assetsChanged || !assets.length) renderAssets();
  renderFrames();
}

function renderAssets() {
  $("#asset-total").textContent = state.assets.length;
  const list = $("#asset-list");
  list.replaceChildren();
  if (!state.assets.length) {
    const empty = node("div", "empty-assets");
    empty.append(
      node("strong", "", "Bring in your first source file"),
      node(
        "p",
        "",
        "Import images or a flight video. Source files stay on this machine, with their original content preserved.",
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
      row.append(button);
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
  $("#visible-frame-count").textContent = frames.length;
  $("#select-visible").disabled =
    state.bulkSelecting || !frames.some((frame) => !frame.selected);
  $("#clear-selection").disabled = state.bulkSelecting || !selected;
  const grid = $("#frame-grid");
  grid.replaceChildren();
  if (!frames.length) {
    const empty = node("div", "empty-frames");
    const onlySelected = state.filter === "selected";
    empty.append(
      node(
        "strong",
        "",
        onlySelected ? "Your selection is empty" : "No frames to review yet",
      ),
      node(
        "p",
        "",
        onlySelected
          ? "Choose frames from the collection using their checkboxes. Your selection is saved automatically."
          : "Import an image, or extract frames from a video above. Each frame will retain its source and original timestamp.",
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
  image.src = `/api/frames/${encodeURIComponent(frame.id)}/image`;
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
  card.append(open, bottom);
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
  const targets = (selected ? visibleFrames() : state.frames).filter(
    (frame) => frame.selected !== selected,
  );
  state.bulkSelecting = true;
  renderFrames();
  let failures = 0;
  try {
    // Limit parallel writes so larger collections do not flood the local server.
    for (let offset = 0; offset < targets.length; offset += 6) {
      const results = await Promise.allSettled(
        targets
          .slice(offset, offset + 6)
          .map((frame) => updateSelection(frame.id, selected, true)),
      );
      failures += results.filter(
        (result) => result.status === "rejected",
      ).length;
    }
    if (failures)
      notify(
        `${failures} frame selection(s) could not be saved. Please try again.`,
        true,
      );
  } finally {
    state.bulkSelecting = false;
    renderFrames();
  }
}

function openExtraction(asset) {
  state.extracting = asset.id;
  $("#extract-source").textContent = asset.filename;
  $("#extract-error").hidden = true;
  $("#extract-form").reset();
  $("#dedup-settings").hidden = true;
  $("#extract-hamming").disabled = true;
  $("#extract-dialog").showModal();
}

function openInspection(id) {
  state.inspectionIds = visibleFrames().map((frame) => frame.id);
  state.inspecting = id;
  renderInspection();
  $("#frame-dialog").showModal();
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
  $("#inspect-image").src = `/api/frames/${encodeURIComponent(frame.id)}/image`;
  $("#inspect-image").alt =
    `${source?.filename || "Frame"}${source?.kind === "video" ? ` at approximately ${timestamp(frame.timestamp_seconds)}` : ""}`;
  renderInspectionSelection();
  const position = state.inspectionIds.indexOf(frame.id);
  $("#inspect-position").textContent =
    `${position + 1} / ${state.inspectionIds.length}`;
  $("#previous-frame").disabled = position <= 0;
  $("#next-frame").disabled = position >= state.inspectionIds.length - 1;
  const session = state.sessions.find((item) => item.id === frame.session_id);
  const metadata = [
    ["Original source", source?.filename || frame.asset_id],
    ["Flight session", session?.name || frame.session_id],
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
  if (frame.extraction?.interval_seconds != null) {
    const config = frame.extraction;
    metadata.splice(
      4,
      0,
      [
        "Extraction settings",
        `Every ${config.interval_seconds}s · start ${timestamp(config.start_seconds || 0)} · ${config.max_frames} frames maximum`,
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
    const url = `/api/assets/${encodeURIComponent(source.id)}/media`;
    if (video.getAttribute("src") !== url) video.src = url;
    seekOriginal();
  } else {
    video.removeAttribute("src");
    video.load();
  }
  $("#download-source").href =
    `/api/assets/${encodeURIComponent(frame.asset_id)}/media`;
  $("#download-source").download = source?.filename || "original";
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

function renderJobs() {
  const list = $("#job-list");
  const expanded = new Set(
    [...list.querySelectorAll("details[open]")].map(
      (details) => details.dataset.jobId,
    ),
  );
  list.replaceChildren();
  const active = state.jobs.filter(isActive);
  $("#jobs-section").hidden = !state.jobs.length;
  $("#jobs-active-count").textContent = active.length
    ? `${active.length} active`
    : "Recent jobs";
  const jobs = [...state.jobs].sort(
    (a, b) =>
      Number(isActive(b)) - Number(isActive(a)) ||
      String(b.created_at).localeCompare(String(a.created_at)),
  );
  const displayed = [
    ...jobs.filter(isActive),
    ...jobs.filter((job) => !isActive(job)).slice(0, 8),
  ];
  for (const job of displayed) {
    const asset = state.assets.find((item) => item.id === job.params?.asset_id);
    const row = node("article", "job-row");
    const header = node("div", "job-header");
    header.append(
      node(
        "h3",
        "job-name",
        asset?.filename ||
          `${job.kind === "infer" ? "Model comparison" : job.kind === "assist" ? "Annotation assistance" : job.kind === "train" ? "Detector training" : job.kind === "evaluate" ? "Quality evaluation" : "Frame extraction"} · ${String(job.id).slice(0, 8)}`,
      ),
      node("span", `job-status ${job.status}`, job.status),
    );
    row.append(header);
    if (job.message) row.append(node("p", "job-message", job.message));
    if (isActive(job)) {
      const progress = node("progress", "job-progress");
      progress.max = 1;
      progress.value = Math.max(0, Math.min(1, job.progress || 0));
      progress.setAttribute(
        "aria-label",
        `Processing progress for ${asset?.filename || job.id}`,
      );
      row.append(progress);
    }
    if (job.error) row.append(node("p", "job-error", job.error));
    const bottom = node("div", "job-bottom");
    if (job.logs?.length || job.started_at) {
      const logs = node("details", "job-logs");
      logs.dataset.jobId = String(job.id);
      logs.open = expanded.has(String(job.id));
      const download = node("a", "text-button", "Download full worker log ↗");
      download.href = `/api/jobs/${encodeURIComponent(job.id)}/log`;
      download.download = `${job.id}.log`;
      logs.append(
        node("summary", "", "View processing log"),
        node(
          "pre",
          "",
          job.logs?.length
            ? job.logs.join("\n")
            : "No progress messages recorded.",
        ),
        download,
      );
      bottom.append(logs);
    }
    if (isActive(job)) {
      const cancel = node("button", "text-button", "Cancel job");
      cancel.type = "button";
      cancel.addEventListener("click", async () => {
        cancel.disabled = true;
        try {
          await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, {
            method: "POST",
          });
          await refreshJobs();
          await refreshSession();
        } catch (error) {
          notify(error.message, true);
          cancel.disabled = false;
        }
      });
      bottom.append(cancel);
    }
    row.append(bottom);
    list.append(row);
  }
}

async function refreshJobs() {
  if (state.polling) clearTimeout(state.polling);
  state.polling = null;
  const previouslyActive = state.jobs.some(isActive);
  const previousStatuses = new Map(
    state.jobs.map((job) => [job.id, job.status]),
  );
  state.jobs = await api("/api/jobs");
  renderJobs();
  window.dispatchEvent(new Event("iris:jobs"));
  const changed = state.jobs.some(
    (job) => previousStatuses.get(job.id) !== job.status,
  );
  if (previouslyActive || state.jobs.some(isActive) || changed)
    await refreshSession();
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

$("#file-input").addEventListener("change", async (event) => {
  const files = [...event.target.files];
  if (!files.length || !state.sessionId) return;
  const sessionId = state.sessionId;
  const input = event.target;
  const progress = $("#upload-progress");
  const trigger = $(".upload-trigger");
  input.disabled = true;
  trigger.classList.add("disabled");
  progress.hidden = false;
  let imported = 0;
  const failures = [];
  try {
    for (let index = 0; index < files.length; index++) {
      const file = files[index];
      progress.textContent = `Importing ${index + 1} of ${files.length} · ${file.name}`;
      const body = new FormData();
      body.append("file", file);
      try {
        await api(`/api/sessions/${encodeURIComponent(sessionId)}/assets`, {
          method: "POST",
          body,
        });
        imported++;
      } catch (error) {
        failures.push(`${file.name}: ${error.message}`);
      }
    }
    await refreshSession();
    const summary = `${imported} source file${imported === 1 ? "" : "s"} imported.`;
    notify(
      failures.length ? `${summary} ${failures.join(" ")}` : summary,
      Boolean(failures.length),
    );
  } catch (error) {
    notify(error.message, true);
  } finally {
    input.value = "";
    input.disabled = false;
    trigger.classList.remove("disabled");
    progress.hidden = true;
  }
});

for (const filter of ["all", "selected"]) {
  $(`#filter-${filter}`).addEventListener("click", () => {
    state.filter = filter;
    renderFrames();
  });
}
$("#select-visible").addEventListener("click", () => bulkSelection(true));
$("#clear-selection").addEventListener("click", () => bulkSelection(false));
$("#extract-dedup").addEventListener("change", (event) => {
  $("#dedup-settings").hidden = !event.target.checked;
  $("#extract-hamming").disabled = !event.target.checked;
});

$("#extract-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const start = Number($("#extract-start").value);
  const end =
    $("#extract-end").value === "" ? null : Number($("#extract-end").value);
  const errorBox = $("#extract-error");
  if (end !== null && end <= start) {
    errorBox.textContent = "The end time must be after the start time.";
    errorBox.hidden = false;
    return;
  }
  const button = $("button[type=submit]", event.currentTarget);
  button.disabled = true;
  errorBox.hidden = true;
  try {
    const config = {
      interval_seconds: Number($("#extract-interval").value),
      start_seconds: start,
      end_seconds: end,
      max_frames: Number($("#extract-limit").value),
      dedup_hamming: $("#extract-dedup").checked
        ? Number($("#extract-hamming").value)
        : null,
    };
    await api(`/api/assets/${encodeURIComponent(state.extracting)}/extract`, {
      method: "POST",
      body: JSON.stringify(config),
    });
    $("#extract-dialog").close();
    notify("Extraction queued. You can follow its progress below.");
    await refreshJobs();
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
    if (!$("#extract-dialog").open) notify(error.message, true);
  } finally {
    button.disabled = false;
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
    const [system, sessions] = await Promise.all([
      api("/api/system"),
      api("/api/sessions"),
    ]);
    $("#version").textContent = system.version ? `v${system.version}` : "";
    $("#storage-path").textContent = system.data_dir;
    state.sessions = sessions;
    renderSessions();
    const recalled = recalledSession();
    const active =
      sessions.find((session) => String(session.id) === recalled) ||
      sessions[0];
    if (active) await selectSession(active.id);
    else $("#welcome").hidden = false;
    await refreshJobs();
  } catch (error) {
    notify(error.message, true);
    $("#storage-path").textContent = "Local server unavailable";
    if (!state.sessions.length) {
      renderSessions();
      $("#welcome").hidden = false;
    }
  }
}

initialize();
