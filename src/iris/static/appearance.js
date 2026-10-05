"use strict";

((root, factory) => {
  const appearance = factory();
  if (typeof module === "object" && module.exports) module.exports = appearance;
  if (typeof window !== "undefined") root.IRISAppearance = appearance.create(window);
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const storageKey = "iris.appearance";
  const normalize = (value) => value === "light" || value === "dark" ? value : "system";

  function create(browser) {
    const document = browser.document;
    let storage;
    let preference = "system";
    let media;
    let select;
    let theme;

    try {
      storage = browser.localStorage;
      preference = normalize(storage?.getItem(storageKey));
    } catch { /* Appearance still works when browser storage is unavailable. */ }
    try {
      media = browser.matchMedia?.("(prefers-color-scheme: dark)");
    } catch { /* Older or restricted browsers use the light fallback. */ }

    function apply() {
      theme = preference === "system" ? (media?.matches ? "dark" : "light") : preference;
      document.documentElement.setAttribute("data-theme", theme);
      document.documentElement.style.colorScheme = theme;
      if (select) select.value = preference;
    }

    function setPreference(value) {
      preference = normalize(value);
      apply();
      try { storage?.setItem(storageKey, preference); } catch { /* Keep this page's choice. */ }
    }

    // This script runs in the head, before styles and the appearance control exist.
    apply();

    function bindControl() {
      select = document.getElementById("appearance-select");
      if (!select) return;
      select.value = preference;
      select.addEventListener("change", () => setPreference(select.value));
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", bindControl, { once: true });
    } else {
      bindControl();
    }

    function systemChanged() {
      if (preference === "system") apply();
    }
    if (media?.addEventListener) media.addEventListener("change", systemChanged);
    else if (media?.addListener) media.addListener(systemChanged);

    browser.addEventListener("storage", (event) => {
      if (event.storageArea && event.storageArea !== storage) return;
      if (event.key !== storageKey && event.key !== null) return;
      preference = normalize(event.key === null ? null : event.newValue);
      apply();
    });

    return Object.freeze({
      getPreference: () => preference,
      getTheme: () => theme,
      setPreference,
    });
  }

  return { create };
});
