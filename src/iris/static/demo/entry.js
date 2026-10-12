"use strict";

// Installed before data/bootstrap and shared scripts; static HTML is already inert.
(() => {
  const entry = document.getElementById("iris-demo-entry");
  if (!entry || document.documentElement.dataset.irisRuntime !== "demo") return;
  // If a later bootstrap script fails, the app must never select local transport.
  const unavailable = new Error("The demo runtime could not be loaded. Retry loading the demo.");
  const unavailableReady = Promise.reject(unavailable);
  unavailableReady.catch(() => {});
  const reject = () => { throw unavailable; };
  window.IRISRuntimeAdapter = {
    mode: "demo", capabilities: {}, error: unavailable, ready: unavailableReady,
    request: async () => reject(), resourceURL: reject,
    upload: () => ({ promise: Promise.reject(unavailable), abort() {} }),
  };
  const runtime = () => window.IRISRuntimeAdapter;
  const workbench = document.getElementById("iris-demo-workbench");
  const enter = document.getElementById("iris-demo-try");
  const status = document.getElementById("iris-demo-entry-status");
  const notice = document.getElementById("iris-demo-entry-notice");
  const retry = document.getElementById("iris-demo-entry-retry");
  const reset = document.getElementById("iris-demo-entry-reset");
  const listeners = new AbortController();
  let phase = "loading";
  let checking = false;
  let timeout;
  const scriptsReady = new Promise((resolve) => document.addEventListener("DOMContentLoaded", resolve, { once: true }));

  const localLink = () => {
    notice.textContent = "This link opens a feature available in local Iris. Explore the sample here, or install Iris locally.";
    notice.hidden = false;
  };
  const address = new URL(window.location.href);
  const localParameters = ["benchmark_review", "review_label", "tracking_comparison", "tracking_quality",
    "tracking_cost", "tracking_study", "tracking_selection", "pipeline_bundle", "temporal_sequence",
    "temporal_comparison", "temporal_lane"];
  if (localParameters.some((key) => address.searchParams.has(key))) {
    localParameters.forEach((key) => address.searchParams.delete(key));
    window.history.replaceState(null, "", address);
    localLink();
  }

  const setPhase = (value) => {
    phase = value;
    entry.dataset.state = value;
  };
  const ownsFocus = () => [entry, status, enter, retry, reset].includes(document.activeElement);
  function fail(error) {
    if (phase !== "loading") return;
    clearTimeout(timeout);
    const focused = ownsFocus();
    setPhase("error");
    enter.disabled = true;
    status.textContent = error?.message || "The example could not be loaded. Retry to try again.";
    retry.hidden = false;
    reset.hidden = typeof runtime().error?.reset !== "function";
    if (focused) retry.focus({ preventScroll: true });
  }

  // Upgrade the static, no-JavaScript fallback to a native modal focus boundary.
  entry.close();
  entry.showModal();
  status.focus({ preventScroll: true });
  retry.hidden = true;
  setPhase("loading");
  entry.addEventListener("cancel", (event) => event.preventDefault());

  // A processing deep link cannot open a second workspace/dialog over the entry.
  window.addEventListener("iris:before-workspace", (event) => {
    event.preventDefault();
    event.stopImmediatePropagation();
    if (!["intake", "annotation", "comparison"].includes(event.detail?.name)) localLink();
  }, { capture: true, signal: listeners.signal });

  async function prepared() {
    if (checking || phase !== "loading") return;
    checking = true;
    try {
      await scriptsReady;
      if (phase !== "loading") return;
      if (!document.getElementById("iris-demo-toolbar") || !document.getElementById("iris-demo-install-dialog"))
        throw new Error("The interface could not be started. Retry loading the demo.");
      const { seed } = await runtime().ready;
      const images = [...document.querySelectorAll("#frame-grid img")];
      const expected = seed.responses[`/api/sessions/${seed.session_id}/frames`].length;
      if (!expected || images.length !== expected)
        throw new Error("The sample gallery could not be prepared. Retry loading the demo.");
      status.textContent = "Loading the sample images…";
      await Promise.all(images.map(async (image) => {
        // Both public photos belong to the initial view, including on small screens.
        image.loading = "eager";
        try { await image.decode(); }
        catch { throw new Error("A sample image could not be loaded. Retry loading the demo."); }
        if (!image.isConnected || !image.naturalWidth)
          throw new Error("A sample image is unavailable. Retry loading the demo.");
      }));
      if (phase !== "loading") return;
      clearTimeout(timeout);
      const focused = ownsFocus();
      setPhase("ready");
      status.textContent = "";
      enter.disabled = false;
      if (focused) enter.focus({ preventScroll: true });
    } catch (error) { fail(error); }
  }
  window.addEventListener("iris:project-initialized", prepared, { once: true, signal: listeners.signal });
  window.addEventListener("iris:project-initialization-error", (event) => fail(event.detail), { signal: listeners.signal });
  window.addEventListener("error", (event) => {
    if (event.target instanceof HTMLScriptElement || event.target === window)
      fail(new Error("The interface could not be started. Retry loading the demo."));
  }, { capture: true, signal: listeners.signal });
  scriptsReady.then(() => runtime().ready.catch(fail));
  timeout = window.setTimeout(() => fail(new Error("The example is taking too long to load. Check your connection and retry.")), 30000);

  retry.addEventListener("click", (event) => {
    event.preventDefault();
    if (!reset.disabled) window.location.reload();
  });
  reset.addEventListener("click", async () => {
    if (phase !== "error" || typeof runtime().error?.reset !== "function") return;
    if (!window.confirm("Reset the saved demo to this version? This permanently removes demo annotations, history and selection in this browser.")) return;
    reset.disabled = true;
    retry.setAttribute("aria-disabled", "true");
    try {
      await runtime().reset();
      window.location.reload();
    } catch (error) {
      status.textContent = error.message || "The saved demo could not be reset. Try again.";
      reset.disabled = false;
      retry.removeAttribute("aria-disabled");
    }
  });
  enter.addEventListener("click", () => {
    if (phase !== "ready") return;
    setPhase("entering");
    enter.disabled = true;
    const finish = () => {
      if (phase !== "entering") return;
      setPhase("entered");
      listeners.abort();
      entry.close();
      workbench.inert = false;
      document.documentElement.classList.remove("iris-demo-entering");
      document.getElementById("main").focus({ preventScroll: true });
      window.dispatchEvent(new Event("iris:demo-entered"));
    };
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) finish();
    else {
      entry.classList.add("iris-demo-entry-leaving");
      entry.addEventListener("transitionend", (event) => {
        if (event.target === entry && event.propertyName === "opacity") finish();
      }, { once: true });
      window.setTimeout(finish, 200);
    }
  });
})();
