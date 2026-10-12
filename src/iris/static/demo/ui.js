"use strict";

// Demo policy lives here; the shared workspaces retain their local behavior.
(() => {
  const adapter = window.IRISRuntimeAdapter;
  if (adapter?.mode !== "demo") return;

  const installURL = "https://github.com/victormonnot/iris/blob/main/docs/setup.md";
  const supportedWorkspaces = new Set(["intake", "annotation", "comparison"]);
  const localReason = "Install Iris locally to use your own files, run models and access the full workbench.";
  const $ = (selector) => document.querySelector(selector);
  const element = (tag, className, text) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text) item.textContent = text;
    return item;
  };
  const button = (id, text, className = "button button-secondary") => {
    const item = element("button", className, text);
    item.id = id;
    item.type = "button";
    return item;
  };
  const installLink = () => {
    const item = element("a", "iris-demo-install", "Install Iris locally ↗");
    item.href = installURL;
    item.target = "_blank";
    item.rel = "noopener noreferrer";
    return item;
  };

  document.documentElement.classList.add("iris-demo");
  const toolbar = element("section", "iris-demo-toolbar");
  toolbar.id = "iris-demo-toolbar";
  toolbar.setAttribute("aria-label", "Interactive demo");
  const introduction = element("div", "iris-demo-introduction");
  introduction.append(
    element("strong", "iris-demo-badge", "Demo"),
    element("span", "", "Explore the sample, review its labels and inspect recorded model outputs."),
  );
  const actions = element("div", "iris-demo-actions");
  const exportButton = button("iris-demo-export", "Export saved annotations");
  const resetButton = button("iris-demo-reset", "Reset demo", "text-button");
  const reloadButton = button("iris-demo-reload", "Reload saved demo", "text-button");
  reloadButton.hidden = true;
  actions.append(exportButton, resetButton, reloadButton, installLink());
  const storage = element("p", "iris-demo-storage", "Opening this browser’s demo workspace…");
  storage.id = "iris-demo-storage";
  storage.setAttribute("role", "status");
  const notice = element("p", "iris-demo-message");
  notice.id = "iris-demo-message";
  notice.setAttribute("role", "status");
  notice.hidden = true;
  const editHint = element("p", "iris-demo-edit-hint");
  editHint.id = "iris-demo-edit-hint";
  editHint.hidden = true;
  for (const control of [exportButton, resetButton, reloadButton])
    control.setAttribute("aria-describedby", editHint.id);
  toolbar.append(introduction, actions, storage, editHint, notice);
  const topbar = $(".topbar");
  if (topbar) topbar.after(toolbar);
  else document.body.prepend(toolbar);
  const localLabel = $(".sidebar-heading .local-label");
  if (localLabel) localLabel.textContent = "DEMO";
  const workspaceState = $(".workspace-state");
  if (workspaceState) {
    for (const child of workspaceState.childNodes)
      if (child.nodeType === Node.TEXT_NODE && child.textContent.includes("Local workspace"))
        child.textContent = "Browser demo";
  }
  const storageDetail = $(".workspace-location > .small");
  if (storageDetail) storageDetail.textContent = "Public sample images · your edits stay in this browser.";

  const dialog = element("dialog", "modal studio-dialog iris-demo-dialog");
  dialog.id = "iris-demo-local-dialog";
  dialog.setAttribute("aria-labelledby", "iris-demo-local-title");
  dialog.setAttribute("aria-describedby", "iris-demo-local-reason");
  const heading = element("div", "modal-heading");
  const title = element("h2", "", "Available in local Iris");
  title.id = "iris-demo-local-title";
  const close = button("iris-demo-local-close", "×", "icon-button");
  close.setAttribute("aria-label", "Close local feature notice");
  heading.append(title, close);
  const reason = element("p", "", localReason);
  reason.id = "iris-demo-local-reason";
  const dialogActions = element("div", "modal-actions");
  dialogActions.append(installLink());
  dialog.append(heading, reason, dialogActions);
  document.body.append(dialog);
  close.addEventListener("click", () => dialog.close());
  let returnFocus = null;
  dialog.addEventListener("close", () => {
    if (returnFocus?.isConnected && !returnFocus.closest("[hidden], [inert]"))
      returnFocus.focus({ preventScroll: true });
  });
  function explain(message = localReason) {
    reason.textContent = message;
    if (!dialog.open) {
      returnFocus = document.activeElement;
      dialog.showModal();
      close.focus();
    }
  }

  function message(text, error = false) {
    notice.textContent = text;
    notice.hidden = !text;
    notice.classList.toggle("iris-demo-error", error);
  }
  let ready = false;
  let failed = false;
  let canRecover = false;
  let actionPending = false;
  let context = { dirty: false, busy: false };
  function controls() {
    const blocked = context.dirty || context.busy;
    exportButton.disabled = !ready || actionPending || blocked;
    resetButton.hidden = failed && !canRecover;
    resetButton.disabled = (!ready && !canRecover) || actionPending || blocked;
    reloadButton.disabled = actionPending || blocked;
    editHint.hidden = !blocked;
    editHint.textContent = context.busy
      ? "Finish the current annotation action before exporting, resetting or reloading."
      : "Save or discard your current annotation edits before exporting, resetting or reloading. Only saved annotations are exported.";
  }
  function canAct() {
    window.dispatchEvent(new Event("iris:annotation-context-request"));
    if (context.dirty || context.busy || actionPending) {
      message(editHint.textContent || "Wait for the current demo action to finish.");
      return false;
    }
    return true;
  }
  function updateStorage(detail = adapter.status || {}) {
    if (!["persistent", "memory"].includes(detail.mode)) {
      if (detail.mode === "error" && detail.message) message(detail.message, true);
      return;
    }
    const memory = detail.mode === "memory";
    storage.classList.toggle("iris-demo-memory", memory);
    storage.textContent = memory
      ? "Temporary memory only. Saved edits can disappear when this page closes or reloads; export them to keep a copy."
      : "Saved edits stay in this browser. Export them to keep a separate copy; clearing site data removes them.";
  }
  window.addEventListener("iris:annotation-context", (event) => {
    context = event.detail || { dirty: false, busy: false };
    controls();
  });
  window.addEventListener("iris:demo-status", (event) => updateStorage(event.detail));
  window.addEventListener("iris:demo-change", () => {
    reloadButton.hidden = false;
    message("Saved demo data changed in another tab. Reload to use the latest version; your current unsaved edits have been kept.");
    controls();
  });
  exportButton.addEventListener("click", async () => {
    if (!canAct() || !ready) return;
    actionPending = true;
    controls();
    try {
      const annotations = await adapter.exportAnnotations();
      const blob = new Blob([JSON.stringify(annotations, null, 2) + "\n"], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const download = element("a");
      download.href = url;
      download.download = "iris-demo-annotations.json";
      document.body.append(download);
      download.click();
      download.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      message("Saved annotations exported. Recorded model outputs and unsaved edits are not included.");
    } catch (failure) {
      message(failure.message || "The saved annotations could not be exported.", true);
    } finally {
      actionPending = false;
      controls();
    }
  });
  resetButton.addEventListener("click", async () => {
    if ((!ready && !canRecover) || !canAct()) return;
    if (!window.confirm("Reset this demo to its original sample? This permanently removes the saved demo edits in this browser, including data from an older demo version. Export your saved annotations first if you want to keep them.")) return;
    actionPending = true;
    controls();
    try {
      await adapter.reset();
      window.dispatchEvent(new Event("iris:demo-reset"));
      window.location.reload();
    } catch (failure) {
      message(failure.message || "The demo could not be reset. Your existing data was not intentionally replaced.", true);
      actionPending = false;
      controls();
    }
  });
  reloadButton.addEventListener("click", () => {
    if (canAct()) window.location.reload();
  });

  // Controls remain discoverable, with one delegated policy guarding their
  // native, keyboard and programmatic event paths before workspace handlers.
  const policies = [
    ["#project-create, #project-form, #session-form, #start-session", "Create projects and sessions in local Iris. This demo provides one public example to explore."],
    ["#taxonomy-open, #taxonomy-form, #annotation-adopt-taxonomy", "Edit class definitions in local Iris. You can read the sample’s definitions in Annotation → Label definitions."],
    [".upload-trigger, #file-input, #source-dropzone, #upload-retry, #upload-cancel, #dataset-import-open, #dataset-import-dialog", "Import your own images, videos or annotated datasets in local Iris. This demo uses only the included public sample."],
    ["#workspace-backup-open, #workspace-backup-dialog", "Full workspace backups are available in local Iris. Use Export saved annotations in the demo toolbar to keep your browser edits."],
    ["#extract-dialog, #video-review-extract-form", "Extract video frames and review video passages in local Iris. This demo contains two public photos."],
    ["#comparison-form", "Run new comparisons in local Iris. The demo lets you inspect real recorded predictions and change their display threshold without running a model."],
    [".annotation-proposal-setup, .annotation-assist, #preannotations, #dinox-preannotations, #assistance-batches", "Generate detector proposals and request model assistance in local Iris. You can accept, correct or reject the sample’s imported proposals here."],
    ["#review-queue-comparison, #review-queue-confidence, #review-queue-iou, #review-queue-order", "Comparison-based review prioritization is available in local Iris. Use review status filters here, or open Model comparison to inspect the recorded outputs."],
  ];
  for (const [selector, explanation] of policies) {
    for (const target of document.querySelectorAll(selector)) {
      target.dataset.demoLocalOnly = explanation;
      if (target.matches("button, input, select, [role=button]"))
        target.setAttribute("aria-disabled", "true");
      if (target.matches("details")) {
        target.open = false;
        target.querySelector("summary")?.setAttribute("aria-disabled", "true");
      }
    }
  }
  for (const workspace of ["training", "evaluation", "experiments", "benchmark", "tracking", "identities"]) {
    const target = $(`#workspace-${workspace}`);
    target?.setAttribute("aria-disabled", "true");
    target?.setAttribute("title", "Available in local Iris");
  }
  window.addEventListener("iris:before-workspace", (event) => {
    if (supportedWorkspaces.has(event.detail?.name)) return;
    event.preventDefault();
    explain("Training, evaluation, experiments, benchmarks and tracking are available in local Iris. Explore Data intake, Annotation and Model comparison in this demo.");
  }, { capture: true });
  const guardedTarget = (event) => event.target instanceof Element
    ? event.target.closest("[data-demo-local-only]") : null;
  function guard(event) {
    const target = guardedTarget(event);
    if (!target) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (event.type === "change" && event.target.matches('input[type="file"]'))
      event.target.value = "";
    explain(target.dataset.demoLocalOnly);
  }
  for (const type of ["click", "submit", "change", "drop"])
    document.addEventListener(type, guard, { capture: true });
  document.addEventListener("dragover", (event) => {
    if (guardedTarget(event)) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, { capture: true });
  document.addEventListener("keydown", (event) => {
    if (guardedTarget(event) && ["Enter", " ", "ArrowDown", "ArrowUp"].includes(event.key))
      guard(event);
  }, { capture: true });
  document.addEventListener("toggle", (event) => {
    const target = guardedTarget(event);
    if (event.target.matches?.("details") && target && event.target.open) {
      event.stopImmediatePropagation();
      event.target.open = false;
      explain(target.dataset.demoLocalOnly);
    }
  }, { capture: true });

  function replaceWithNote(selector, text) {
    const target = $(selector);
    if (!target) return;
    target.classList.add("iris-demo-hidden");
    const note = element("p", "iris-demo-local-note", text + " ");
    note.append(installLink());
    target.after(note);
  }
  replaceWithNote("#session-form", "Create your own sessions in local Iris.");
  replaceWithNote("#source-dropzone", "Explore the included photos here. Import your own files in local Iris.");
  replaceWithNote("#comparison-form", "These are recorded model outputs. Display filters below remain interactive; new inference runs locally.");
  replaceWithNote(".annotation-proposal-setup", "Review the imported proposals below. Additional model proposals are available in local Iris.");
  const reviewExplanation = element("p", "iris-demo-local-note", "This demo supports review status filters. Comparison-based prioritization is available in local Iris.");
  $(".review-queue-controls")?.after(reviewExplanation);
  for (const selector of ["#review-queue-comparison", "#review-queue-confidence", "#review-queue-iou", "#review-queue-order"])
    if ($(selector)) $(selector).disabled = true;

  controls();
  window.dispatchEvent(new Event("iris:annotation-context-request"));
  Promise.resolve(adapter.ready).then(() => {
    ready = true;
    updateStorage();
    controls();
  }).catch((failure) => {
    failed = true;
    canRecover = typeof adapter.error?.reset === "function";
    reloadButton.textContent = "Reload demo";
    reloadButton.hidden = false;
    storage.textContent = "The demo workspace could not be opened. Existing browser data has been kept.";
    message(failure?.message || (canRecover
      ? "Reset the demo explicitly to start over, or install Iris locally."
      : "Reload the demo to try again, or install Iris locally."), true);
    controls();
  });
})();
