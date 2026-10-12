"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISRuntime = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  function createLocalAdapter({ fetch: fetchRequest, XMLHttpRequest: Request } = {}) {
    return {
      mode: "local",
      request(path, options) {
        return fetchRequest ? fetchRequest(path, options) : globalThis.fetch(path, options);
      },
      resourceURL(path) { return path; },
      upload(path, options = {}) {
        let request;
        let settled = false;
        let rejectUpload;
        const signal = options.signal;
        const abortError = () => new DOMException("The upload was aborted.", "AbortError");
        const finish = (complete, value) => {
          if (settled) return;
          settled = true;
          signal?.removeEventListener("abort", abort);
          complete(value);
        };
        function abort() {
          if (settled) return;
          finish(rejectUpload, abortError());
          request?.abort();
        }
        const promise = new Promise((resolve, reject) => {
          rejectUpload = reject;
          if (signal?.aborted) {
            finish(reject, abortError());
            return;
          }
          try {
            const responseType = options.responseType ?? "text";
            if (!["text", "json", "arraybuffer", "blob"].includes(responseType))
              throw new TypeError("Unsupported upload response type.");
            request = new (Request || globalThis.XMLHttpRequest)();
            request.open(options.method || "POST", path);
            request.responseType = responseType;
            for (const [name, value] of new Headers(options.headers))
              request.setRequestHeader(name, value);
            request.upload.addEventListener("progress", (event) => {
              if (!settled) options.onProgress?.({
                loaded: event.loaded,
                total: event.total,
                lengthComputable: event.lengthComputable,
              });
            });
            request.upload.addEventListener("load", () => {
              if (!settled) options.onUploaded?.();
            });
            request.addEventListener("load", () => {
              if (settled) return;
              try {
                if (!request.status) throw new TypeError("Network request failed.");
                const headers = new Headers();
                for (const line of request.getAllResponseHeaders().split(/\r?\n/)) {
                  const colon = line.indexOf(":");
                  if (colon > 0) headers.append(line.slice(0, colon), line.slice(colon + 1).trim());
                }
                const body = [204, 205, 304].includes(request.status) ? null
                  : responseType === "text" ? request.responseText
                  : responseType === "json" ? JSON.stringify(request.response)
                  : request.response;
                finish(resolve, new Response(body, {
                  status: request.status,
                  statusText: request.statusText,
                  headers,
                }));
              } catch (error) { finish(reject, error); }
            });
            request.addEventListener("error", () => finish(reject, new TypeError("Network request failed.")));
            request.addEventListener("abort", () => finish(reject, abortError()));
            request.addEventListener("timeout", () => finish(reject, new DOMException("The upload timed out.", "TimeoutError")));
            signal?.addEventListener("abort", abort, { once: true });
            request.send(options.body);
          } catch (error) { finish(reject, error); }
        });
        return { promise, abort };
      },
    };
  }

  function create({ projectScope, adapter } = {}) {
    if (typeof projectScope?.url !== "function")
      throw new TypeError("An IRIS project scope is required.");
    const transport = adapter === undefined ? createLocalAdapter() : adapter;
    if (!transport || typeof transport.mode !== "string" || !transport.mode.trim() ||
        ["request", "resourceURL", "upload"].some((name) => typeof transport[name] !== "function"))
      throw new TypeError("An IRIS adapter must define mode, request, resourceURL and upload.");
    return Object.freeze({
      mode: transport.mode,
      capabilities: Object.freeze({ ...transport.capabilities }),
      request(path, options) { return transport.request(projectScope.url(path), options); },
      resourceURL(path) { return transport.resourceURL(projectScope.url(path)); },
      upload(path, options) { return transport.upload(projectScope.url(path), options); },
    });
  }

  return { create, createLocalAdapter };
});
