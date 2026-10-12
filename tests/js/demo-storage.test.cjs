"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const { create } = require("../../src/iris/static/demo/storage.js");

const initial = () => ({ annotations: {}, selection: { "frame-6723": true } });
const memory = (options = {}) => create({ seedVersion: "example-1", initialState: initial(), indexedDB: null, ...options });

test("unavailable storage clearly reports temporary memory and isolates returned snapshots", async () => {
  const notices = [];
  const seed = initial();
  const store = await memory({ initialState: seed, onStatus: (status) => notices.push(status) });
  assert.equal(store.status.mode, "memory");
  assert.match(store.status.message, /only while this tab is open/);
  assert.deepEqual(notices, [store.status]);
  seed.selection["frame-6723"] = false;
  const first = await store.read();
  assert.equal(first.schemaVersion, 1);
  assert.equal(first.seedVersion, "example-1");
  assert.equal(typeof first.epoch, "string");
  assert.equal(first.selection["frame-6723"], true);
  first.selection["frame-6723"] = false;
  assert.equal((await store.read()).selection["frame-6723"], true);
});

test("updates serialize revision checks and preserve failed drafts without mutating saved data", async () => {
  const store = await memory();
  const epoch = (await store.read()).epoch;
  const conflict = new Error("The annotation changed in another tab");
  const save = (draft) => store.update((state) => {
    const revision = state.annotations.a?.revision || 0;
    if (state.epoch !== epoch || revision !== 0) throw conflict;
    state.annotations.a = { revision: revision + 1, draft };
  });
  const [first, second] = await Promise.allSettled([save("first"), save("second")]);
  assert.equal(first.status, "fulfilled");
  assert.equal(second.status, "rejected");
  assert.equal(second.reason, conflict);
  assert.deepEqual((await store.read()).annotations.a, { revision: 1, draft: "first" });
  await assert.rejects(store.update((state) => {
    state.annotations.a.draft = "partial mutation";
    throw conflict;
  }), (error) => error === conflict);
  assert.equal((await store.read()).annotations.a.draft, "first");
  assert.equal((await store.update((state) => { state.selection["frame-6723"] = false; })).selection["frame-6723"], false);
});

test("reset restores pristine seed and changes epoch so an old revision-zero save conflicts", async () => {
  const store = await memory();
  const before = await store.read();
  await store.update((state) => { state.selection["frame-6723"] = false; state.annotations.a = { revision: 3 }; });
  const reset = await store.reset();
  assert.notEqual(reset.epoch, before.epoch);
  assert.deepEqual(reset.annotations, {});
  assert.deepEqual(reset.selection, initial().selection);
  await assert.rejects(store.update((state) => {
    if (state.epoch !== before.epoch) throw new Error("Demo was reset");
    state.annotations.a = { revision: 1 };
  }), /Demo was reset/);
  reset.selection["frame-6723"] = false;
  assert.equal((await store.read()).selection["frame-6723"], true);
  assert.notEqual((await store.reset()).epoch, reset.epoch);
});

test("async mutators and envelope edits cannot commit partial changes", async () => {
  const store = await memory();
  const original = await store.read();
  await assert.rejects(store.update(async (state) => {
    state.selection["frame-6723"] = false;
    await Promise.resolve();
    state.annotations.a = { revision: 1 };
  }), /must be synchronous/);
  for (const field of ["schemaVersion", "seedVersion", "epoch"])
    await assert.rejects(store.update((state) => { delete state[field]; }), /cannot change/);
  assert.deepEqual(await store.read(), original);
  let retained;
  const updated = await store.update((state) => { retained = state; state.annotations.a = { revision: 1 }; });
  retained.annotations.a.revision = 50;
  updated.annotations.a.revision = 60;
  assert.equal((await store.read()).annotations.a.revision, 1);
});

test("read, update and reset all reject after close", async () => {
  const store = await memory();
  store.close();
  store.close();
  for (const operation of [() => store.read(), () => store.update(() => {}), () => store.reset()])
    await assert.rejects(operation(), { name: "InvalidStateError" });
});

test("a blocked IndexedDB getter and an opening failure retain a usable temporary session", async () => {
  const source = fs.readFileSync(require.resolve("../../src/iris/static/demo/storage.js"), "utf8");
  const sandbox = { module: { exports: {} }, structuredClone, DOMException };
  Object.defineProperty(sandbox, "indexedDB", { get() { throw new DOMException("Blocked", "SecurityError"); } });
  vm.runInNewContext(source, sandbox);
  const store = await sandbox.module.exports.create({ seedVersion: 1, initialState: initial() });
  assert.equal(store.status.mode, "memory");
  assert.equal((await store.update((state) => { state.selection["frame-6723"] = false; })).selection["frame-6723"], false);
  const failed = await memory({ indexedDB: { open() { throw new DOMException("Blocked", "SecurityError"); } } });
  assert.equal(failed.status.mode, "memory");
});

// This event stub exercises error boundaries only. IndexedDB isolation and durability
// are covered by browser checks, not by pretending this implements a database engine.
function storedSnapshot(snapshot) {
  let value = structuredClone(snapshot);
  let writes = 0;
  let writeError = null;
  let closed = false;
  const database = {
    objectStoreNames: { contains: () => true },
    close() { closed = true; },
    transaction() {
      const tx = {
        abort() { queueMicrotask(() => tx.onabort?.()); },
        objectStore() {
          return {
            get() {
              const request = {};
              queueMicrotask(() => {
                request.result = structuredClone(value);
                request.onsuccess();
                if (!tx.failed) queueMicrotask(() => tx.oncomplete?.());
              });
              return request;
            },
            put(next) {
              writes++;
              const request = {};
              if (writeError) {
                tx.failed = true;
                queueMicrotask(() => {
                  request.error = writeError;
                  request.onerror?.();
                  tx.error = writeError;
                  tx.onabort?.();
                });
              } else value = structuredClone(next);
              return request;
            },
          };
        },
      };
      return tx;
    },
  };
  return {
    factory: { open() { const request = {}; queueMicrotask(() => { closed = false; request.result = database; request.onsuccess(); }); return request; } },
    read: () => structuredClone(value),
    writes: () => writes,
    closed: () => closed,
    failWrites(error) { writeError = error; },
  };
}

test("seed and schema mismatches preserve stored edits until the explicit reset action", async () => {
  for (const change of [{ seedVersion: "older-seed" }, { schemaVersion: 99 }]) {
    const snapshot = { ...initial(), schemaVersion: 1, seedVersion: "example-1", epoch: "old-epoch", ...change };
    snapshot.annotations.a = { revision: 4, draft: "preserve this" };
    const backing = storedSnapshot(snapshot);
    let mismatch;
    await assert.rejects(memory({ indexedDB: backing.factory }), (error) => {
      mismatch = error;
      return error.name === "SeedMismatchError";
    });
    assert.equal(backing.writes(), 0);
    assert.equal(backing.closed(), true);
    assert.deepEqual(backing.read(), snapshot);
    const reset = await mismatch.reset();
    assert.equal(reset.status.mode, "persistent");
    const state = await reset.read();
    assert.equal(state.schemaVersion, 1);
    assert.equal(state.seedVersion, "example-1");
    assert.notEqual(state.epoch, "old-epoch");
    assert.deepEqual(state.annotations, {});
    reset.close();
  }
});

test("a persistent write failure stays an error and never reports a memory fallback", async () => {
  const original = { ...initial(), schemaVersion: 1, seedVersion: "example-1", epoch: "saved-epoch" };
  const backing = storedSnapshot(original);
  const notices = [];
  const store = await memory({ indexedDB: backing.factory, onStatus: (status) => notices.push(status.mode) });
  const failure = new DOMException("Storage is full", "QuotaExceededError");
  backing.failWrites(failure);
  await assert.rejects(store.update((state) => { state.annotations.a = { revision: 1 }; }), (error) => error === failure);
  assert.equal(store.status.mode, "persistent");
  assert.deepEqual(notices, ["persistent"]);
  assert.deepEqual(await store.read(), original);
  await assert.rejects(store.reset(), (error) => error === failure);
  assert.deepEqual(backing.read(), original);
});

test("a domain conflict in persistent mode retains its original error and stored revision", async () => {
  const original = { ...initial(), schemaVersion: 1, seedVersion: "example-1", epoch: "saved-epoch" };
  const backing = storedSnapshot(original);
  const store = await memory({ indexedDB: backing.factory });
  const conflict = Object.assign(new Error("Conflict"), { status: 409 });
  await assert.rejects(store.update((state) => { state.annotations.a = { revision: 2 }; throw conflict; }), (error) => error === conflict);
  assert.deepEqual(backing.read(), original);
  assert.equal(backing.writes(), 0);
});
