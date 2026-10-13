"use strict";

// Loaded only by the browser demo entry point, before the shared application.
(() => {
  const script = document.currentScript;
  const seedURL = new URL(script.dataset.seed || "./seed/", window.location.href);
  const address = new URL(window.location.href);
  if (!address.searchParams.has("project")) {
    address.searchParams.set("project", "web-demo");
    window.history.replaceState(null, "", address);
  }
  let channel;
  // Storage belongs to this demo mount, not its versioned asset directory.
  // A new sample gets its own workspace, preserving the previous sample's edits.
  // Revisions of the same sample still require explicit recovery on mismatch.
  const sample = script.dataset.sample;
  const namespace = `iris-web-demo:${new URL("./seed/", address).pathname}${sample ? `:${sample}` : ""}`;
  try {
    channel = new BroadcastChannel(namespace);
    channel.addEventListener("message", () => {
      window.dispatchEvent(new Event("iris:demo-change"));
    });
  } catch { /* Conflicts are still enforced transactionally without notifications. */ }
  const status = (detail) => window.dispatchEvent(new CustomEvent("iris:demo-status", { detail }));

  async function load() {
    const [fixtureResponse, provenanceResponse] = await Promise.all([
      fetch(new URL("fixture.json", seedURL), { cache: "no-store", mode: "same-origin", redirect: "error" }),
      fetch(new URL("provenance.json", seedURL), { cache: "no-store", mode: "same-origin", redirect: "error" }),
    ]);
    if (!fixtureResponse.ok || !provenanceResponse.ok) throw new Error("The demo examples could not be loaded. Reload to try again.");
    const [bytes, provenance] = await Promise.all([fixtureResponse.arrayBuffer(), provenanceResponse.json()]);
    if (!globalThis.crypto?.subtle) throw new Error("Open the demo over HTTPS or localhost to verify its example data.");
    const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
      .map((value) => value.toString(16).padStart(2, "0")).join("");
    if (hash !== provenance.files?.["fixture.json"]?.sha256)
      throw new Error("The example files are from different versions. Reload the demo before editing.");
    const seed = JSON.parse(new TextDecoder().decode(bytes));
    const store = await window.IRISDemoStorage.create({
      seedVersion: hash,
      initialState: window.IRISDemoAnnotations.createState(seed),
      name: namespace,
      onStatus: status,
    });
    return { seed, store, provenance, baseURL: seedURL.href };
  }
  window.IRISRuntimeAdapter = window.IRISDemoAdapter.create({
    load,
    onChange(detail) { try { channel?.postMessage(detail); } catch { /* Notifications are optional. */ } },
  });
  window.IRISRuntimeAdapter.ready.catch((error) => status({ mode: "error", message: error.message }));
  // Keep storage and notifications alive for pages restored from the back/forward
  // cache. The browser closes these connections when the document is destroyed.
})();
