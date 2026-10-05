"use strict";

// Switching a view only changes visibility. Forms and approved plans keep their nodes.
(() => {
  const names = ["datasets", "plan", "runs", "exports"];
  function open(name, { focus = false } = {}) {
    if (!names.includes(name)) return false;
    const tab = $(`#training-tab-${name}`);
    const moveFocus = focus || names.some((item) => item !== name &&
      $(`#training-pane-${item}`).contains(document.activeElement));
    for (const item of names) {
      const active = item === name;
      $(`#training-pane-${item}`).hidden = !active;
      const button = $(`#training-tab-${item}`);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    }
    if (moveFocus) {
      tab.focus();
      tab.scrollIntoView({ block: "nearest" });
    }
    return true;
  }
  names.forEach((name, index) => {
    const tab = $(`#training-tab-${name}`);
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
  $("#training-workspace").addEventListener("click", (event) => {
    const button = event.target.closest("[data-training-open]");
    if (button) open(button.dataset.trainingOpen, { focus: true });
  });
  window.IRISTrainingNavigation = Object.freeze({ open });
})();
