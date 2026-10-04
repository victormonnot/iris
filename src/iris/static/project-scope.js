"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISProjectScope = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  function create(href, storage) {
    const page = new URL(href);
    const read = (key) => {
      try { return storage?.getItem(key); } catch { return null; }
    };
    const write = (key, value) => {
      try { storage?.setItem(key, value); } catch { /* Storage is optional. */ }
    };
    // A page keeps its project even if another tab changes the remembered one.
    const id = page.searchParams.get("project") || read("iris.project") || "default";
    const sessionKey = `iris.session.${id}`;
    return Object.freeze({
      id,
      url(path) {
        const url = new URL(path, page);
        if (url.origin !== page.origin || !url.pathname.startsWith("/api/"))
          throw new Error("Project resources must come from the local IRIS server.");
        const global = /^\/api\/(system|projects|annotation-providers|workspace)(\/|$)/.test(url.pathname);
        if (!global) url.searchParams.set("project_id", id);
        return `${url.pathname}${url.search}${url.hash}`;
      },
      location(projectId) {
        const target = new URL(page);
        target.searchParams.set("project", projectId);
        target.hash = "";
        return `${target.pathname}${target.search}`;
      },
      remember() { write("iris.project", id); },
      rememberSession(sessionId) { write(sessionKey, String(sessionId)); },
      recalledSession() {
        return read(sessionKey) || (id === "default" ? read("iris.session") : null);
      },
    });
  }
  return { create };
});
