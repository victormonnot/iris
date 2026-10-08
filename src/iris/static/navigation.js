"use strict";

// Workspace navigation keeps the original transition guards and event contract.
// A session is required only for intake, comparison and annotation.
(() => {
  const workspaces = {
    intake: [
      "01", "Data intake", "From sources to useful frames.",
      "Import source footage, sample frames and select the data worth keeping.",
    ],
    comparison: [
      "02", "Model comparison", "See what your models see.",
      "Run detectors on a shared selection, inspect their differences and keep a reproducible baseline.",
    ],
    annotation: [
      "03", "Annotation", "Review annotations.",
      "Review proposals, correct bounding boxes and validate each frame before it becomes training data.",
    ],
    training: [
      "04", "Dataset & training", "Build on what you have learned.",
      "Freeze validated data, train a local detector and bring its checkpoint back into comparison.",
    ],
    evaluation: [
      "05", "Quality evaluation", "Measure gains and regressions.",
      "Evaluate frozen labels, inspect detection errors and choose a reference model with evidence.",
    ],
    experiments: [
      "06", "Experiments", "Keep the evidence together.",
      "Turn a completed evaluation into a clear experiment record, add your conclusions and share a self-contained report.",
    ],
    benchmark: [
      "07", "Preannotation benchmark", "Measure the work behind the labels.",
      "Freeze an independent human reference, separate tuning from evaluation and measure corrections to candidate proposals.",
    ],
    tracking: [
      "08", "Tracking comparison", "Follow the observations.",
      "Replay ByteTrack and BoT-SORT on the same saved detections, and inspect identity continuity without assuming correctness.",
    ],
    identities: [
      "09", "Temporal identities", "Keep identity through time.",
      "Correct identities and observed boxes, declare visibility, and record human review on each available source frame.",
    ],
  };
  const sessionWorkspaces = new Set(["intake", "comparison", "annotation"]);
  const sidebar = $("#workspace-sidebar");
  const toggle = $("#sidebar-toggle");
  const shell = $(".shell");
  const compact = window.matchMedia("(max-width: 900px)");
  let current = "intake";
  let compactOpen = false;

  function renderSidebar() {
    const expanded = !compact.matches || compactOpen;
    if (compact.matches) toggle.hidden = false;
    // Move focus before hiding its ancestor; there is no modal focus trap.
    if (!expanded && sidebar.contains(document.activeElement)) toggle.focus();
    if (!compact.matches && document.activeElement === toggle) $("#main").focus();
    sidebar.hidden = !expanded;
    toggle.hidden = !compact.matches;
    toggle.setAttribute("aria-expanded", String(expanded));
    shell.dataset.sidebarOpen = String(expanded);
  }

  function openSidebar() {
    if (compact.matches) compactOpen = true;
    renderSidebar();
  }

  function syncSession() {
    const sessionView = sessionWorkspaces.has(current);
    for (const [selector, hidden] of [
      ["#session-workspace", !state.sessionId || !sessionView],
      ["#welcome", Boolean(state.sessionId) || !sessionView],
    ]) {
      const panel = $(selector);
      if (hidden && panel.contains(document.activeElement)) $("#main").focus();
      panel.hidden = hidden;
    }
  }

  function open(name) {
    const info = workspaces[name];
    if (!info) return false;
    if (!window.dispatchEvent(new CustomEvent("iris:before-workspace", {
      cancelable: true, detail: { name },
    }))) return false;

    // A programmatic handoff may start inside the panel about to disappear.
    // Keep keyboard navigation in the destination instead of resetting to body.
    const leavingPanel = current !== name
      && $(`#${current}-workspace`).contains(document.activeElement);
    current = name;
    for (const workspace of Object.keys(workspaces)) {
      const active = workspace === name;
      $(`#${workspace}-workspace`).hidden = !active;
      const button = $(`#workspace-${workspace}`);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    }
    $("#workspace-step").replaceChildren(
      node("span", "step-marker", info[0]),
      document.createTextNode(info[1]),
    );
    $("#workspace-title").textContent = info[2];
    $("#workspace-description").textContent = info[3];
    if (leavingPanel) $("#main").focus();
    syncSession();
    compactOpen = false;
    renderSidebar();
    window.dispatchEvent(new CustomEvent("iris:workspace", { detail: { name } }));
    return true;
  }

  for (const name of Object.keys(workspaces)) {
    $(`#workspace-${name}`).addEventListener("click", () => open(name));
  }
  $("#workspace-intake").setAttribute("aria-current", "page");
  toggle.addEventListener("click", () => {
    compactOpen = !compactOpen;
    renderSidebar();
  });
  // The intake handler focuses the field; reveal its ancestor first.
  $("#start-session").addEventListener("click", openSidebar, { capture: true });
  sidebar.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !compact.matches || !compactOpen || event.defaultPrevented) return;
    // Native selects and inputs retain their own Escape behavior.
    if (event.target.matches("input, textarea, select")) return;
    compactOpen = false;
    renderSidebar();
  });
  compact.addEventListener("change", renderSidebar);
  window.addEventListener("iris:session", () => {
    syncSession();
    compactOpen = false;
    renderSidebar();
  });
  window.IRISNavigation = Object.freeze({ open, syncSession, openSidebar });
  renderSidebar();
})();
