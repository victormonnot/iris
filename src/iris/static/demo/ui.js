"use strict";

// Demo policy lives here; the shared workspaces retain their local behavior.
(() => {
  const adapter = window.IRISRuntimeAdapter;
  if (adapter?.mode !== "demo") return;

  const installURL = "https://github.com/victormonnot/iris/blob/main/docs/setup.md";
  const unavailable = "Unavailable in demo mode";
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
  const revision = document.documentElement.dataset.irisRevision;
  if (typeof revision === "string" && revision.length === 40 && /^[0-9a-f]{40}$/.test(revision)) {
    const source = element("a", "iris-demo-revision", revision.slice(0, 7));
    source.id = "iris-demo-revision";
    source.href = `https://github.com/victormonnot/iris/commit/${revision}`;
    source.target = "_blank";
    source.rel = "noopener noreferrer";
    source.title = `Source commit ${revision}`;
    source.setAttribute("aria-label", `Source commit ${revision.slice(0, 7)}`);
    introduction.append(source);
  }
  const actions = element("div", "iris-demo-actions");
  const resetButton = button("iris-demo-reset", "Reset demo", "text-button");
  const reloadButton = button("iris-demo-reload", "Reload saved demo", "text-button");
  reloadButton.hidden = true;
  actions.append(resetButton, reloadButton, installLink());
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
  for (const control of [resetButton, reloadButton])
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
  const title = element("h2", "", unavailable);
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
    resetButton.hidden = failed && !canRecover;
    resetButton.disabled = (!ready && !canRecover) || actionPending || blocked;
    reloadButton.disabled = actionPending || blocked;
    editHint.hidden = !blocked;
    editHint.textContent = context.busy
      ? "Finish the current annotation action before resetting or reloading."
      : "Save or discard your current annotation edits before resetting or reloading.";
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
      ? "Temporary memory only. Demo edits disappear when this page closes or reloads. Install Iris locally to keep your own workspace."
      : "Demo edits stay in this browser; resetting the demo or clearing site data removes them.";
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
  resetButton.addEventListener("click", async () => {
    if ((!ready && !canRecover) || !canAct()) return;
    if (!window.confirm("Reset this demo to its original sample? This permanently removes the saved demo edits in this browser, including data from an older demo version.")) return;
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

  // Keep the native screens, disclosures and read-only navigation. Only the
  // unavailable controls are disabled; their explanation stays beside them.
  const fields = "button, input, select, textarea, [role=button]";
  const readOnlyControls = [
    "[data-close]", "[id$='-close']", "[data-training-open]", "[data-evaluation-open]",
    "[data-benchmark-open]", "[data-evaluation-section]", "[role=tab]",
    "button[id*='refresh']", "select[id$='-history']", "[id$='-job']", "[id$='-raw']",
    "#training-models-open", "#evaluation-open-datasets", "#training-resume-existing",
  ].join(", ");
  // The optional third entry restricts a mixed region to its mutable controls.
  const policies = [
    ["#project-form, #session-form", "Create projects and sessions in your local workspace."],
    ["#source-dropzone", "Import your own images and videos in local Iris."],
    [".dataset-import-launch", "Import annotated datasets in local Iris.", "#dataset-import-open"],
    [".workspace-backup-location", "Create and restore workspace backups in local Iris.", "#workspace-backup-open"],
    ["#taxonomy-form", "Edit class definitions in local Iris. The sample definitions remain available to inspect."],
    ["#dataset-import-dialog", "Import annotated datasets in local Iris.", "#dataset-import-upload input, #dataset-import-preview-button, #dataset-import-commit-form input, #dataset-import-commit-form select, #dataset-import-commit"],
    ["#workspace-backup-dialog", "Create, download and restore full workspace backups in local Iris.", "#workspace-backup-create, #workspace-backup-upload-form input, #workspace-backup-upload, #workspace-backup-upload-cancel, #workspace-backup-restore-form input, #workspace-backup-restore, [data-operation-action='download'], [data-operation-action='restore'], [data-operation-action='remove'], [data-operation-action='cancel']"],
    ["#extract-form, #video-review-form, #video-review-extract-form", "Analyze videos and extract frames in local Iris."],
    ["#comparison-form", "Run models in local Iris. The saved comparison below contains real recorded outputs; its display filters remain interactive."],
    [".comparison-model-library", "These models produced the recorded comparison. Install Iris locally to run your own comparisons.", "#model-catalog input"],
    [".annotation-proposal-setup", "Import new detector proposals in local Iris. You can review the sample’s existing proposals below."],
    [".annotation-assist, #preannotations, #dinox-preannotations, #assistance-batches", "Generate new proposals and request model assistance in local Iris. Manual annotation remains available in this demo."],
    [".review-queue-controls", "Comparison-based prioritization requires local Iris. Review status filters remain available.", "#review-queue-comparison, #review-queue-confidence, #review-queue-iou, #review-queue-order"],
    ["#dataset-form", "Prepare and freeze dataset releases in local Iris."],
    ["#training-form, #training-recovery", "Train models and continue saved training runs in local Iris."],
    ["#model-export-form, .model-export-measurement-import", "Create model packages and import performance measurements in local Iris."],
    ["#evaluation-form, #evaluation-reference-form, #evaluation-test-audit", "Run evaluations and save reference decisions in local Iris."],
    [".tracking-prepare, #selection-prepare, #bundle-prepare", "Run tracking, measure performance and prepare deployment bundles in local Iris."],
    ["#tracking-quality-panel", "Calculate tracking quality reports in local Iris.", "#quality-reference, #quality-iou, #quality-mapping select, #quality-calculate"],
    ["#identity-seed-panel, #identity-object-fields, .identity-review", "Create and save temporal identity references in local Iris."],
    ["#benchmark-prepare, #benchmark-config-form, .benchmark-trial-setup", "Prepare benchmarks and run model trials in local Iris."],
    ["#benchmark-pane-reports", "Create benchmark reports in local Iris.", "#benchmark-report-preview, #benchmark-report-save, #benchmark-report-title, #benchmark-report-notes"],
    ["#experiments-workspace", "Create, edit and export experiment reports in local Iris.", "#experiments-new, #experiments-empty-create, #experiments-edit, #experiments-export, #experiments-export-cancel"],
    ["#experiments-compose-form, #experiments-edit-form", "Create and edit experiment reports in local Iris."],
  ];
  const notes = new WeakMap();
  let noteCount = 0;
  function noteFor(region) {
    let note = notes.get(region);
    if (note?.isConnected) return note;
    note = element("p", "iris-demo-local-note");
    note.id = `iris-demo-local-note-${++noteCount}`;
    note.append(element("strong", "iris-demo-unavailable-label", unavailable),
      document.createTextNode(". "), installLink());
    const heading = region.querySelector(":scope > summary, :scope > legend, :scope > .modal-heading, :scope > .panel-heading");
    if (region.matches("label, [role=button]")) region.after(note);
    else if (region.matches(".dataset-import-launch, .workspace-backup-location")) region.append(note);
    else if (heading) heading.after(note);
    else region.prepend(note);
    notes.set(region, note);
    return note;
  }
  function mark(target, explanation, note) {
    if (target.closest(".iris-demo-local-note") || target.matches(readOnlyControls)) return;
    target.dataset.demoLocalOnly = explanation;
    target.classList.add("iris-demo-unavailable");
    if (target.getAttribute("aria-disabled") !== "true") target.setAttribute("aria-disabled", "true");
    if (target.title !== unavailable) target.title = unavailable;
    if ("disabled" in target && !target.disabled) target.disabled = true;
    // A disabled download must not remain usable through the link context menu.
    if (target.matches("a[href]")) target.removeAttribute("href");
    if (note) {
      const describedBy = new Set((target.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean));
      if (!describedBy.has(note.id)) {
        describedBy.add(note.id);
        target.setAttribute("aria-describedby", [...describedBy].join(" "));
      }
    }
  }
  const extraActions = [
    "#start-session", "#annotation-adopt-taxonomy", "#file-input", ".upload-trigger",
    "#upload-retry", "#upload-cancel", "#benchmark-lock", "#evaluation-save-experiment",
    "#identity-undo", "#identity-redo", "#identity-draw", "#identity-add-absence",
    "#tracking-cancel-cache", "#tracking-cancel", "#cost-cancel", "#study-cancel",
    "#selection-cancel", "#bundle-cancel", "#job-detail-cancel", "#job-check-continuation",
    "#job-confirm-continuation", "#benchmark-editor-start", "#benchmark-editor-save",
    "#benchmark-editor-complete", "#dataset-coco-download", "#dataset-manifest-download",
    "#model-export-download", "#cost-download", "#study-download", "#selection-download",
    "#bundle-download", "#bundle-download-manifest", "a[download]",
  ].join(", ");
  function applyPolicies() {
    for (const [selector, explanation, controlsSelector = fields] of policies) {
      for (const region of document.querySelectorAll(selector)) {
        const note = noteFor(region.closest(".comparison-setup") || region);
        if (region.matches("form")) region.dataset.demoLocalSubmit = explanation;
        for (const form of region.querySelectorAll("form")) form.dataset.demoLocalSubmit = explanation;
        for (const target of region.querySelectorAll(controlsSelector)) mark(target, explanation, note);
      }
    }
    for (const target of document.querySelectorAll(extraActions))
      mark(target, localReason);
    const dropzone = $("#source-dropzone");
    if (dropzone) {
      dropzone.dataset.demoLocalDrop = "Import your own files in local Iris.";
      dropzone.classList.add("iris-demo-unavailable-zone");
    }
  }
  applyPolicies();
  // Shared renderers may replace controls or recompute disabled state. Reapply
  // only this demo policy, leaving navigation and recorded result filters alone.
  new MutationObserver(applyPolicies).observe(document.body, {
    childList: true, subtree: true,
    attributes: true, attributeFilter: ["disabled", "href", "aria-disabled"],
  });

  function guardedTarget(event) {
    if (!(event.target instanceof Element)) return null;
    if (event.type === "submit") return event.target.closest("[data-demo-local-submit]");
    if (["drop", "dragover"].includes(event.type))
      return event.target.closest("[data-demo-local-drop], [data-demo-local-only]");
    return event.target.closest("[data-demo-local-only]");
  }
  function guard(event) {
    const target = guardedTarget(event);
    if (!target) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (event.type === "change" && event.target.matches('input[type="file"]'))
      event.target.value = "";
    if (event.type !== "dragover")
      explain(target.dataset.demoLocalOnly || target.dataset.demoLocalSubmit || target.dataset.demoLocalDrop);
  }
  for (const type of ["click", "submit", "change", "drop", "dragover"])
    document.addEventListener(type, guard, { capture: true });
  document.addEventListener("keydown", (event) => {
    if (guardedTarget(event) && ["Enter", " ", "ArrowDown", "ArrowUp"].includes(event.key))
      guard(event);
  }, { capture: true });

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
