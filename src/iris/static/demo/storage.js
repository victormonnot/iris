"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISDemoStorage = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const SCHEMA_VERSION = 1;
  const STORE = "snapshots";
  const KEY = "current";
  const clone = (value) => structuredClone(value);
  const closedError = () => new DOMException("The demo storage is closed.", "InvalidStateError");
  const abortedError = () => new DOMException("The browser could not save this change.", "AbortError");

  function fresh(config) {
    const epoch = globalThis.crypto?.randomUUID?.() ||
      `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
    return { ...clone(config.initialState), schemaVersion: SCHEMA_VERSION, seedVersion: config.seedVersion, epoch };
  }

  function mismatch(config, factory) {
    const error = new Error("Saved demo data belongs to another version. Reset the demo to start again.");
    error.name = "SeedMismatchError";
    // This is deliberately an explicit recovery action; merely opening the demo never resets it.
    error.reset = () => persistent(config, factory, true);
    return error;
  }

  function compatible(state, config, factory) {
    if (!state || state.schemaVersion !== SCHEMA_VERSION || state.seedVersion !== config.seedVersion ||
        typeof state.epoch !== "string" || !state.epoch)
      throw mismatch(config, factory);
  }

  function mutate(state, mutator) {
    if (typeof mutator !== "function") throw new TypeError("A synchronous storage mutator is required.");
    const working = clone(state);
    const result = mutator(working);
    if (result && typeof result.then === "function") {
      Promise.resolve(result).catch(() => {});
      throw new TypeError("Storage mutators must be synchronous.");
    }
    for (const field of ["schemaVersion", "seedVersion", "epoch"])
      if (working[field] !== state[field]) throw new TypeError(`Storage mutators cannot change ${field}.`);
    return clone(working);
  }

  function report(config, mode) {
    const status = Object.freeze(mode === "memory"
      ? { mode, message: "Browser storage is unavailable. Changes last only while this tab is open." }
      : { mode, message: "Changes are saved in this browser." });
    try { config.onStatus?.(status); } catch { /* A notice cannot change storage behavior. */ }
    return status;
  }

  function memory(config) {
    let state = fresh(config);
    let closed = false;
    let pending = Promise.resolve();
    const enqueue = (operation) => {
      const result = pending.then(() => {
        if (closed) throw closedError();
        return operation();
      });
      pending = result.catch(() => {});
      return result;
    };
    return Object.freeze({
      status: report(config, "memory"),
      read: () => enqueue(() => clone(state)),
      update: (mutator) => enqueue(() => {
        state = mutate(state, mutator);
        return clone(state);
      }),
      reset: () => enqueue(() => {
        state = fresh(config);
        return clone(state);
      }),
      close() { closed = true; },
    });
  }

  function open(factory, name) {
    return new Promise((resolve, reject) => {
      let finished = false;
      const request = factory.open(name);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE);
      };
      request.onerror = () => { finished = true; reject(request.error); };
      request.onblocked = () => {
        finished = true;
        reject(new Error("Browser storage is blocked by another tab."));
      };
      request.onsuccess = () => {
        if (finished) request.result.close();
        else resolve(request.result);
      };
    });
  }

  function removeDatabase(factory, name) {
    return new Promise((resolve, reject) => {
      const request = factory.deleteDatabase(name);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error("Close other demo tabs before resetting this version."));
    });
  }

  function transaction(database, mode, transform) {
    return new Promise((resolve, reject) => {
      let tx;
      let failure;
      let result;
      try {
        tx = database.transaction(STORE, mode);
        tx.oncomplete = () => resolve(result);
        tx.onabort = () => reject(failure || tx.error || abortedError());
        tx.onerror = () => { failure ||= tx.error; };
        const objectStore = tx.objectStore(STORE);
        const request = objectStore.get(KEY);
        request.onerror = () => { failure ||= request.error; };
        request.onsuccess = () => {
          try {
            const update = transform(request.result);
            result = clone(update.state);
            if (update.write) {
              const write = objectStore.put(result, KEY);
              write.onerror = () => { failure ||= write.error; };
            }
          } catch (error) {
            failure = error;
            tx.abort();
          }
        };
      } catch (error) {
        if (tx) {
          failure = error;
          try { tx.abort(); } catch { reject(error); }
        } else reject(error);
      }
    });
  }

  async function persistent(config, factory, replace = false) {
    let database = await open(factory, config.name);
    if (!database.objectStoreNames.contains(STORE)) {
      database.close();
      if (!replace) throw mismatch(config, factory);
      await removeDatabase(factory, config.name);
      database = await open(factory, config.name);
    }
    let closed = false;
    database.onversionchange = () => { closed = true; database.close(); };
    try {
      await transaction(database, "readwrite", (state) => {
        if (replace || state === undefined) return { state: fresh(config), write: true };
        compatible(state, config, factory);
        return { state, write: false };
      });
    } catch (error) {
      database.close();
      throw error;
    }
    const run = (mode, transform) => closed ? Promise.reject(closedError()) : transaction(database, mode, transform);
    return Object.freeze({
      status: report(config, "persistent"),
      read: () => run("readonly", (state) => {
        compatible(state, config, factory);
        return { state, write: false };
      }),
      update: (mutator) => run("readwrite", (state) => {
        compatible(state, config, factory);
        return { state: mutate(state, mutator), write: true };
      }),
      reset: () => run("readwrite", () => ({ state: fresh(config), write: true })),
      close() { closed = true; database.close(); },
    });
  }

  async function create(options = {}) {
    const seedVersion = options.seedVersion;
    if (!(typeof seedVersion === "string" && seedVersion) &&
        !(Number.isInteger(seedVersion) && seedVersion >= 0))
      throw new TypeError("A demo seed version is required.");
    const initialState = clone(options.initialState ?? {});
    if (!initialState || typeof initialState !== "object" || Array.isArray(initialState))
      throw new TypeError("The initial demo state must be an object.");
    const config = {
      seedVersion, initialState, name: options.name ?? "iris-web-demo", onStatus: options.onStatus,
    };
    let factory;
    try {
      // Access is guarded too: privacy settings can make the indexedDB getter throw.
      factory = options.indexedDB === undefined ? globalThis.indexedDB : options.indexedDB;
      if (!factory || typeof factory.open !== "function") return memory(config);
      return await persistent(config, factory);
    } catch (error) {
      if (error?.name === "SeedMismatchError") throw error;
      return memory(config);
    }
  }

  return { create };
});
