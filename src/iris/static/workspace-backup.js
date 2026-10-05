"use strict";

(() => {
  const field = (name) => $(`#workspace-backup-${name}`);
  const active = (operation) =>
    ["queued", "running"].includes(operation.status);
  const terminal = (operation) => !active(operation);
  const view = {
    context: 0,
    tab: "backup",
    preview: null,
    operations: [],
    previewRequest: 0,
    operationsRequest: 0,
    previewPending: false,
    operationsPending: false,
    fresh: false,
    mutation: false,
    timer: null,
    inspectionId: null,
    limit: 8,
    cancelling: new Set(),
    upload: null,
    uploadError: "",
    notice: "",
    previousActive: new Set(),
  };
  const labels = {
    backup: "Workspace backup",
    inspection: "Archive inspection",
    restore: "New workspace restoration",
  };
  const countNames = {
    projects: "Projects",
    sessions: "Sessions",
    assets: "Source files",
    frames: "Extracted images",
    dataset_versions: "Dataset versions",
    training_runs: "Training runs",
    trained_models: "Trained checkpoints",
    comparisons: "Comparisons",
    evaluations: "Evaluations",
    experiment_reports: "Experiment reports",
    annotation_revisions: "Annotation revisions",
    annotation_suggestions: "Annotation suggestions",
    assistance_records: "Assisted annotation records",
    assistance_batches: "Assisted annotation batches",
    assistance_previews: "Annotation previews",
    jobs: "Processing jobs",
    runs: "Model runs",
    predictions: "Saved predictions",
    evaluation_models: "Evaluated pipelines",
    evaluation_predictions: "Evaluation predictions",
    model_references: "Reference decisions",
    dataset_imports: "Dataset imports",
    video_reviews: "Video passage reviews",
  };
  const bytes = (value) =>
    Number.isFinite(value) ? formatBytes(value) : "Not available";
  const date = (value) =>
    new Date(value).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  const open = () => field("dialog").open;
  const current = (context) => open() && context === view.context;
  const activeOperations = () => view.operations.filter(active);
  const inspection = () =>
    view.operations.find(
      (item) =>
        item.id === view.inspectionId &&
        item.kind === "inspection" &&
        item.status === "succeeded",
    );
  const unavailable = () =>
    view.mutation ||
    Boolean(view.upload) ||
    !view.fresh ||
    Boolean(activeOperations().length);

  function setError(name, error) {
    field(name).textContent = error?.message || error || "";
    field(name).hidden = !error;
  }

  function status(message) {
    view.notice = message;
    field("status").textContent = message;
  }

  function setTab(tab, focus = false) {
    view.tab = tab;
    for (const name of ["backup", "restore"]) {
      const button = field(`tab-${name}`);
      button.setAttribute("aria-selected", String(name === tab));
      button.tabIndex = name === tab ? 0 : -1;
      field(`panel-${name}`).hidden = name !== tab;
    }
    if (focus) field(`tab-${tab}`).focus();
  }

  function renderCounts(container, counts = {}, compact = false) {
    container.replaceChildren();
    const entries = Object.entries(counts).filter(([, value]) =>
      Number.isInteger(value),
    );
    const shown = compact
      ? entries.filter(([key]) =>
          [
            "sessions",
            "frames",
            "dataset_versions",
            "experiment_reports",
          ].includes(key),
        )
      : entries;
    for (const [key, value] of shown) {
      const card = node("div", "workspace-backup-count");
      card.append(
        node("strong", "", value.toLocaleString()),
        node("span", "", countNames[key] || key.replaceAll("_", " ")),
      );
      container.append(card);
    }
  }

  function updateControls() {
    const disabled = unavailable();
    const selected = inspection();
    field("create").disabled =
      disabled || view.previewPending || !view.preview?.can_create;
    field("create").textContent = view.mutation
      ? "Please wait…"
      : "Create backup";
    field("refresh").disabled =
      view.previewPending ||
      view.operationsPending ||
      view.mutation ||
      Boolean(view.upload);
    field("file").disabled = disabled || !view.preview;
    field("upload").disabled =
      disabled || !field("file").files.length || !view.preview;
    field("inspection-select").disabled = Boolean(view.upload) || view.mutation;
    field("folder").disabled = !selected || view.mutation;
    field("confirm").disabled = !selected || view.mutation;
    const validName = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/.test(
      field("folder").value,
    );
    field("restore").disabled =
      disabled ||
      !selected ||
      !view.preview ||
      !validName ||
      !field("confirm").checked;
    field("restore").textContent = view.mutation
      ? "Please wait…"
      : "Restore as new workspace";
    const locked =
      activeOperations().some((item) => item.kind === "backup") ||
      view.preview?.write_locked;
    field("write-lock").hidden = !locked;
    field("open-state").textContent = activeOperations().length
      ? "Operation in progress"
      : "";
    field("busy-note").hidden = !activeOperations().length;
    field("destination").textContent =
      view.preview && validName
        ? `${view.preview.restore_parent.replace(/\/$/, "")}/${field("folder").value}`
        : "Enter a new folder name to see its destination.";
  }

  function renderPreview() {
    const preview = view.preview;
    field("preview").hidden = !preview;
    if (!preview) return updateControls();
    field("source").textContent = preview.workspace_path;
    field("volume").textContent = bytes(preview.total_bytes);
    field("file-count").textContent =
      `${Number(preview.file_count).toLocaleString()} files · installed local weights included`;
    field("space").textContent =
      `${bytes(preview.free_bytes)} available · ${bytes(preview.required_bytes)} required by the local operation`;
    field("upload-limit").textContent =
      `ZIP archives up to ${bytes(preview.max_upload_bytes)}. The file is sent only to this local IRIS server.`;
    renderCounts(field("counts"), preview.counts, true);
    renderCounts(field("all-counts"), preview.counts);
    field("categories").replaceChildren();
    for (const category of preview.categories || []) {
      const row = node("div", "workspace-backup-category");
      row.append(
        node("span", "", category.label),
        node("span", "", `${category.file_count.toLocaleString()} files`),
        node("strong", "", bytes(category.size_bytes)),
      );
      field("categories").append(row);
    }
    field("blockers").replaceChildren();
    for (const text of preview.blocking_issues || [])
      field("blockers").append(node("p", "", text));
    field("blockers").hidden = !preview.blocking_issues?.length;
    field("warnings").replaceChildren();
    for (const text of preview.warnings || [])
      field("warnings").append(node("p", "field-hint", text));
    field("excluded").replaceChildren();
    for (const item of preview.excluded || [])
      field("excluded").append(node("li", "", `${item.path} — ${item.reason}`));
    field("excluded-details").hidden = !preview.excluded?.length;
    updateControls();
  }

  function renderInspection() {
    const selected = inspection();
    field("inspection-summary").hidden = !selected;
    if (selected) {
      const summary = selected.result?.summary || {};
      field("inspection-title").textContent = "Integrity verified";
      field("inspection-size").textContent =
        `${Number(summary.file_count || 0).toLocaleString()} files · ${bytes(summary.total_bytes)} unpacked`;
      field("inspection-version").textContent =
        `${selected.result?.filename || "Verified archive"} · IRIS ${summary.app_version || "unknown version"} · created ${summary.created_at ? date(summary.created_at) : "at an unrecorded time"}`;
      renderCounts(field("inspection-counts"), summary.counts, true);
    }
    updateControls();
  }

  function chooseInspection(id) {
    view.inspectionId = id || null;
    field("inspection-select").value = id || "";
    field("confirm").checked = false;
    setError("restore-error", null);
    renderInspection();
  }

  function renderOperation(operation) {
    const card = node("article", "workspace-backup-operation");
    card.dataset.operationId = operation.id;
    card.tabIndex = -1;
    card.setAttribute(
      "aria-label",
      `${labels[operation.kind] || "Workspace operation"} · ${operation.status} · ${date(operation.created_at)}`,
    );
    const heading = node("div", "workspace-backup-operation-heading");
    const title = node("div");
    title.append(
      node("h4", "", labels[operation.kind] || "Workspace operation"),
      node("span", "workspace-backup-caption", date(operation.created_at)),
    );
    heading.append(
      title,
      node(
        "span",
        `workspace-backup-state ${operation.status}`,
        operation.status,
      ),
    );
    card.append(heading);
    if (active(operation)) {
      const progress = operation.progress || {};
      card.append(
        node(
          "p",
          "workspace-backup-operation-message",
          progress.message || "Waiting for the local operation…",
        ),
      );
      const meter = node("progress");
      meter.setAttribute("aria-label", `${labels[operation.kind]} progress`);
      if (Number.isFinite(progress.bytes_total) && progress.bytes_total > 0) {
        meter.max = progress.bytes_total;
        meter.value = Math.min(
          progress.bytes_total,
          Math.max(0, progress.bytes_done || 0),
        );
      }
      card.append(meter);
      const details = [];
      if (Number.isFinite(progress.bytes_total) && progress.bytes_total > 0)
        details.push(
          `${bytes(progress.bytes_done || 0)} / ${bytes(progress.bytes_total)}`,
        );
      if (Number.isFinite(progress.files_total) && progress.files_total > 0)
        details.push(
          `${progress.files_done || 0} / ${progress.files_total} files`,
        );
      if (details.length)
        card.append(node("p", "workspace-backup-caption", details.join(" · ")));
    }
    if (operation.error)
      card.append(node("p", "inline-error", operation.error));
    if (operation.warning)
      card.append(node("p", "field-hint", operation.warning));
    if (operation.status === "interrupted")
      card.append(
        node(
          "p",
          "field-hint",
          "This operation stopped when IRIS restarted. Start a new operation when ready.",
        ),
      );
    const actions = node("div", "workspace-backup-operation-actions");
    if (active(operation)) {
      const cancel = node(
        "button",
        "text-button",
        view.cancelling.has(operation.id)
          ? "Cancellation requested…"
          : "Cancel operation",
      );
      cancel.type = "button";
      cancel.dataset.operationAction = "cancel";
      cancel.disabled = view.mutation || view.cancelling.has(operation.id);
      cancel.addEventListener("click", () => cancelOperation(operation.id));
      actions.append(cancel);
    }
    if (operation.status === "succeeded" && operation.kind === "backup") {
      const download = node(
        "a",
        "button button-secondary",
        "Download archive ↓",
      );
      download.href = `/api/workspace/operations/${encodeURIComponent(operation.id)}/archive`;
      download.dataset.operationAction = "download";
      download.download = operation.result?.filename || "iris-workspace.zip";
      download.addEventListener("click", () =>
        status(
          "Your browser manages the archive download. Its download panel shows progress and cancellation.",
        ),
      );
      actions.append(download);
      card.append(
        node(
          "p",
          "field-hint",
          `${bytes(operation.result?.archive_size_bytes)} ZIP archive · kept locally until you remove it`,
        ),
      );
    }
    if (operation.status === "succeeded" && operation.kind === "inspection") {
      if (operation.result?.filename)
        card.append(
          node("p", "workspace-backup-path", operation.result.filename),
        );
      const use = node("button", "button button-secondary", "Use for restore");
      use.type = "button";
      use.dataset.operationAction = "restore";
      use.disabled = view.mutation || Boolean(view.upload);
      use.addEventListener("click", () => {
        setTab("restore");
        chooseInspection(operation.id);
        field("folder").focus();
      });
      actions.append(use);
    }
    if (operation.status === "succeeded" && operation.kind === "restore") {
      const result = operation.result || {};
      card.append(
        node(
          "p",
          "field-hint",
          "Restored separately. Your current workspace is still open.",
        ),
      );
      const destination = node(
        "code",
        "workspace-backup-path",
        result.destination || "Destination was not recorded.",
      );
      card.append(destination);
      if (result.launch_command) {
        const command = node(
          "pre",
          "workspace-backup-command",
          result.launch_command,
        );
        command.tabIndex = 0;
        command.dataset.operationAction = "command";
        card.append(command);
        const copy = node("button", "text-button", "Copy launch command");
        copy.type = "button";
        copy.dataset.operationAction = "copy";
        copy.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(result.launch_command);
            if (open())
              status(
                "Launch command copied. Run it yourself when you want to open the restored workspace.",
              );
          } catch {
            if (open())
              status(
                "Copy is unavailable in this browser. Select the command above and copy it manually.",
              );
          }
        });
        actions.append(copy);
      }
    }
    if (
      terminal(operation) &&
      ["backup", "inspection"].includes(operation.kind)
    ) {
      const remove = node(
        "button",
        "text-button workspace-backup-remove",
        "Remove local archive",
      );
      remove.type = "button";
      remove.dataset.operationAction = "remove";
      remove.disabled =
        view.mutation ||
        Boolean(view.upload) ||
        Boolean(activeOperations().length);
      remove.addEventListener("click", () => removeOperation(operation));
      actions.append(remove);
    }
    card.append(actions);
    return card;
  }

  function renderOperations() {
    const focused = document.activeElement?.closest("[data-operation-id]");
    const focus = focused
      ? {
          id: focused.dataset.operationId,
          action: document.activeElement.dataset.operationAction,
          index: Array.from(field("operations").children).indexOf(focused),
        }
      : null;
    field("operations").replaceChildren();
    field("history-count").textContent = `${view.operations.length}`;
    field("history-empty").hidden = Boolean(view.operations.length);
    for (const operation of view.operations.slice(0, view.limit)) {
      const card = renderOperation(operation);
      field("operations").append(card);
    }
    if (focus) {
      const cards = Array.from(field("operations").children);
      const card = cards.find((item) => item.dataset.operationId === focus.id);
      const action = card &&
        Array.from(card.querySelectorAll("[data-operation-action]")).find(
          (item) => item.dataset.operationAction === focus.action && !item.disabled,
        );
      const target = action || card ||
        cards[Math.min(focus.index, cards.length - 1)] || field("history-title");
      target.focus({ preventScroll: true });
    }
    field("more").hidden = view.operations.length <= view.limit;
    field("more").textContent =
      `Show more (${Math.min(view.limit, view.operations.length)} / ${view.operations.length})`;
    const selected = view.inspectionId;
    const options = view.operations.filter(
      (item) => item.kind === "inspection" && item.status === "succeeded",
    );
    field("inspection-select").replaceChildren(
      new Option(
        options.length
          ? "Choose a verified archive…"
          : "No verified archives yet",
        "",
      ),
    );
    for (const option of options)
      field("inspection-select").append(
        new Option(
          `${option.result?.filename || "Verified archive"} · ${date(option.created_at)} · ${bytes(option.result?.summary?.total_bytes)}`,
          option.id,
        ),
      );
    if (options.some((item) => item.id === selected))
      field("inspection-select").value = selected;
    else if (
      selected &&
      !view.operations.some(
        (item) =>
          item.id === selected && item.kind === "inspection" && active(item),
      )
    )
      chooseInspection(null);
    renderInspection();
    updateControls();
  }

  function schedulePoll() {
    clearTimeout(view.timer);
    if (open() && !view.mutation && !view.upload)
      view.timer = setTimeout(() => refreshOperations(), 1800);
  }

  function invalidateOperations() {
    ++view.operationsRequest;
    view.operationsPending = false;
    view.fresh = false;
    clearTimeout(view.timer);
  }

  async function refreshOperations() {
    if (!open() || view.operationsPending || view.mutation || view.upload)
      return;
    const context = view.context,
      request = ++view.operationsRequest;
    view.operationsPending = true;
    updateControls();
    try {
      const operations = await api("/api/workspace/operations");
      if (!current(context) || request !== view.operationsRequest) return;
      const completed = [...view.previousActive].some(
        (id) => !operations.some((item) => item.id === id && active(item)),
      );
      view.previousActive = new Set(
        operations.filter(active).map((item) => item.id),
      );
      view.operations = operations;
      view.fresh = true;
      setError("history-error", null);
      for (const id of view.cancelling)
        if (!operations.some((item) => item.id === id && active(item)))
          view.cancelling.delete(id);
      renderOperations();
      if (completed) refreshPreview();
    } catch (error) {
      if (current(context)) {
        view.fresh = false;
        setError("history-error", error);
      }
    } finally {
      if (current(context) && request === view.operationsRequest) {
        view.operationsPending = false;
        updateControls();
        schedulePoll();
      }
    }
  }

  async function refreshPreview() {
    if (!open() || view.previewPending) return;
    const context = view.context,
      request = ++view.previewRequest;
    view.previewPending = true;
    field("preview-status").textContent =
      "Checking local contents and available storage…";
    setError("preview-error", null);
    updateControls();
    try {
      const preview = await api("/api/workspace/backup-preview");
      if (!current(context) || request !== view.previewRequest) return;
      view.preview = preview;
      field("preview-status").textContent = preview.can_create
        ? "Ready to preserve your saved work."
        : "A backup cannot start yet. Review the details below.";
      renderPreview();
    } catch (error) {
      if (current(context)) {
        view.preview = null;
        field("preview-status").textContent =
          "The workspace could not be checked. Refresh to try again.";
        setError("preview-error", error);
        renderPreview();
      }
    } finally {
      if (current(context) && request === view.previewRequest) {
        view.previewPending = false;
        updateControls();
      }
    }
  }

  async function mutate(path, options, message) {
    if (view.mutation) return null;
    invalidateOperations();
    view.mutation = true;
    setError("action-error", null);
    updateControls();
    renderOperations();
    try {
      const result = await api(path, options);
      status(message);
      if (result?.id) {
        view.operations = [
          result,
          ...view.operations.filter((item) => item.id !== result.id),
        ];
        if (active(result)) view.previousActive.add(result.id);
        if (open()) {
          renderOperations();
          field("operations")
            .querySelector(`[data-operation-id="${CSS.escape(result.id)}"]`)
            ?.scrollIntoView({ block: "nearest" });
        }
      }
      return result;
    } catch (error) {
      setError("action-error", error);
      return null;
    } finally {
      view.mutation = false;
      if (open()) {
        renderOperations();
        refreshOperations();
        refreshPreview();
      }
    }
  }

  async function cancelOperation(id) {
    if (view.mutation || view.cancelling.has(id)) return;
    view.cancelling.add(id);
    const operation = await mutate(
      `/api/workspace/operations/${encodeURIComponent(id)}/cancel`,
      { method: "POST" },
      "Cancellation requested. Wait for the local operation to finish cleaning up.",
    );
    if (!operation) view.cancelling.delete(id);
    if (open()) renderOperations();
  }

  async function removeOperation(operation) {
    if (unavailable()) return;
    if (
      !window.confirm(
        "Remove this local backup or uploaded archive and its operation record? Restored workspaces and your current workspace are kept.",
      )
    )
      return;
    const id = operation.id;
    await mutate(
      `/api/workspace/operations/${encodeURIComponent(id)}`,
      { method: "DELETE" },
      "Local archive removed. Workspace data is unchanged.",
    );
  }

  function renderUpload() {
    const upload = view.upload;
    field("upload-progress").hidden = !upload;
    if (upload) {
      field("upload-message").textContent =
        `${upload.filename} · ${bytes(upload.done)} / ${bytes(upload.total)}`;
      field("upload-meter").max = Math.max(1, upload.total);
      field("upload-meter").value = upload.done;
      field("upload-phase").textContent =
        upload.done >= upload.total
          ? "Upload received. Waiting for archive inspection to be queued…"
          : "Uploading to the local IRIS server…";
    }
    setError("upload-error", view.uploadError);
    updateControls();
  }

  function uploadArchive(event) {
    event.preventDefault();
    if (unavailable() || !view.preview) return;
    const file = field("file").files[0];
    if (!file) return;
    if (!/\.zip$/i.test(file.name)) {
      view.uploadError = "Choose an IRIS workspace ZIP archive.";
      return renderUpload();
    }
    if (!file.size || file.size > view.preview.max_upload_bytes) {
      view.uploadError = `Choose a non-empty ZIP archive no larger than ${bytes(view.preview.max_upload_bytes)}.`;
      return renderUpload();
    }
    const xhr = new XMLHttpRequest();
    const upload = { xhr, filename: file.name, done: 0, total: file.size };
    invalidateOperations();
    view.upload = upload;
    view.uploadError = "";
    renderUpload();
    xhr.open("POST", "/api/workspace/restore-inspections");
    xhr.responseType = "json";
    xhr.upload.addEventListener("progress", (event) => {
      if (view.upload !== upload) return;
      upload.total = event.lengthComputable ? event.total : file.size;
      upload.done = event.loaded;
      if (open()) renderUpload();
    });
    xhr.addEventListener("load", () => {
      if (view.upload !== upload) return;
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response?.id) {
        const operation = xhr.response;
        view.operations = [
          operation,
          ...view.operations.filter((item) => item.id !== operation.id),
        ];
        view.previousActive.add(operation.id);
        view.inspectionId = operation.id;
        field("confirm").checked = false;
        status(
          "Archive uploaded. Local integrity inspection is running; restore becomes available after it passes.",
        );
        field("file").value = "";
      } else
        view.uploadError =
          typeof xhr.response?.detail === "string"
            ? xhr.response.detail
            : `Archive upload failed (${xhr.status}). Choose the file again and retry.`;
    });
    xhr.addEventListener("error", () => {
      if (view.upload === upload)
        view.uploadError =
          "Cannot reach IRIS. The upload did not complete. Retry when the local server is available.";
    });
    xhr.addEventListener("abort", () => {
      if (view.upload === upload)
        status(
          "Upload cancelled. Any operation already received by the server appears in the history below.",
        );
    });
    xhr.addEventListener("loadend", () => {
      if (view.upload !== upload) return;
      view.upload = null;
      if (open()) {
        renderUpload();
        renderOperations();
        refreshOperations();
        refreshPreview();
      }
    });
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  }

  field("open").addEventListener("click", () => {
    if (open()) return;
    view.context++;
    view.fresh = false;
    view.previewPending = false;
    view.operationsPending = false;
    field("dialog").showModal();
    setTab(view.tab);
    field("status").textContent = view.notice;
    renderPreview();
    renderOperations();
    renderUpload();
    refreshOperations();
    refreshPreview();
  });
  field("close").addEventListener("click", () => field("dialog").close());
  field("dialog").addEventListener("close", () => {
    if (open()) return;
    view.context++;
    view.previewRequest++;
    view.operationsRequest++;
    view.previewPending = false;
    view.operationsPending = false;
    clearTimeout(view.timer);
    field("open").focus({ preventScroll: true });
  });
  for (const tab of ["backup", "restore"])
    field(`tab-${tab}`).addEventListener("click", () => setTab(tab));
  field("tabs").addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    setTab(
      event.key === "Home"
        ? "backup"
        : event.key === "End"
          ? "restore"
          : view.tab === "backup"
            ? "restore"
            : "backup",
      true,
    );
  });
  field("refresh").addEventListener("click", () => {
    refreshOperations();
    refreshPreview();
  });
  field("more").addEventListener("click", () => {
    view.limit += 8;
    renderOperations();
  });
  field("file").addEventListener("change", () => {
    view.uploadError = "";
    renderUpload();
  });
  field("upload-form").addEventListener("submit", uploadArchive);
  field("upload-cancel").addEventListener("click", () =>
    view.upload?.xhr.abort(),
  );
  field("inspection-select").addEventListener("change", () =>
    chooseInspection(field("inspection-select").value),
  );
  field("folder").addEventListener("input", () => {
    field("confirm").checked = false;
    updateControls();
  });
  field("confirm").addEventListener("change", updateControls);
  field("create").addEventListener("click", () => {
    if (field("create").disabled) return;
    mutate(
      "/api/workspace/backups",
      { method: "POST" },
      "Backup started. You can close this panel and keep reading your saved work.",
    );
  });
  field("restore-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (field("restore").disabled || !field("restore-form").reportValidity())
      return;
    const result = await mutate(
      "/api/workspace/restores",
      {
        method: "POST",
        body: JSON.stringify({
          inspection_id: view.inspectionId,
          folder_name: field("folder").value,
        }),
      },
      "Restoration started in a new folder. Your current workspace stays open.",
    );
    if (result) field("confirm").checked = false;
  });
  window.addEventListener("beforeunload", (event) => {
    if (view.upload) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
})();
