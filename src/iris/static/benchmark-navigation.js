"use strict";

// Keep frozen settings and trial approvals intact when changing the visible view.
(() => {
  const names = ["candidates", "trials", "reports"];
  function open(name, { focus = false } = {}) {
    if (!names.includes(name)) return false;
    const tab = $(`#benchmark-tab-${name}`);
    const moveFocus = focus || names.some((item) => item !== name &&
      $(`#benchmark-pane-${item}`).contains(document.activeElement));
    for (const item of names) {
      const active = item === name;
      $(`#benchmark-pane-${item}`).hidden = !active;
      const button = $(`#benchmark-tab-${item}`);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    }
    if (moveFocus && !$("#benchmark-detail").hidden && !$("#benchmark-workspace").hidden) {
      tab.focus();
      tab.scrollIntoView({ block: "nearest" });
    }
    return true;
  }
  names.forEach((name, index) => {
    const tab = $(`#benchmark-tab-${name}`);
    tab.addEventListener("click", () => open(name));
    tab.addEventListener("keydown", (event) => {
      const destination = event.key === "ArrowRight" ? (index + 1) % names.length
        : event.key === "ArrowLeft" ? (index + names.length - 1) % names.length
          : event.key === "Home" ? 0 : event.key === "End" ? names.length - 1 : null;
      if (destination === null || event.altKey || event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      open(names[destination], { focus: true });
    });
  });
  $("#benchmark-workspace").addEventListener("click", (event) => {
    const button = event.target.closest("[data-benchmark-open]");
    if (button) open(button.dataset.benchmarkOpen, { focus: true });
  });
  window.IRISBenchmarkNavigation = Object.freeze({ open,
    openTrial: (id) => window.IRISBenchmark?.openTrial(id) ?? Promise.resolve(false),
  });
})();
