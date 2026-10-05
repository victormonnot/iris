"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { create } = require("../../src/iris/static/appearance.js");

function eventTarget() {
  const listeners = new Map();
  return {
    addEventListener(name, listener, options) {
      const entries = listeners.get(name) || [];
      entries.push({ listener, once: options?.once });
      listeners.set(name, entries);
    },
    emit(name, event = {}) {
      for (const entry of [...(listeners.get(name) || [])]) {
        entry.listener(event);
        if (entry.once) listeners.set(name, listeners.get(name).filter((item) => item !== entry));
      }
    },
  };
}

function environment({ saved = null, dark = false, loading = false, control = true } = {}) {
  const values = new Map(saved === null ? [] : [["iris.appearance", saved]]);
  const writes = [];
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem(key, value) { values.set(key, value); writes.push([key, value]); },
  };
  const select = { ...eventTarget(), value: "system" };
  const attributes = new Map();
  const document = {
    ...eventTarget(),
    readyState: loading ? "loading" : "complete",
    documentElement: { style: {}, setAttribute: (key, value) => attributes.set(key, value) },
    getElementById(id) { return control && id === "appearance-select" ? select : null; },
  };
  const media = { ...eventTarget(), matches: dark };
  const browser = {
    ...eventTarget(), document, localStorage: storage,
    matchMedia(query) {
      assert.equal(query, "(prefers-color-scheme: dark)");
      return media;
    },
  };
  return {
    browser, media, select, storage, writes,
    theme: () => attributes.get("data-theme"),
    changeSystem(value) { media.matches = value; media.emit("change", { matches: value }); },
    choose(value) { select.value = value; select.emit("change"); },
    storageEvent(key, newValue, storageArea = storage) {
      browser.emit("storage", { key, newValue, storageArea });
    },
  };
}

test("the head script restores explicit appearance before DOM ready and binds the control later", () => {
  const env = environment({ saved: "light", dark: true, loading: true });
  const context = vm.createContext({ window: env.browser });
  vm.runInContext(fs.readFileSync(require.resolve("../../src/iris/static/appearance.js"), "utf8"), context);
  const api = context.IRISAppearance;
  assert.equal(api.getPreference(), "light");
  assert.equal(env.theme(), "light");
  assert.equal(env.browser.document.documentElement.style.colorScheme, "light");
  assert.equal(env.select.value, "system");
  env.browser.document.emit("DOMContentLoaded");
  assert.equal(env.select.value, "light");
  assert.deepEqual(env.writes, []);
  env.choose("dark");
  assert.equal(api.getTheme(), "dark");
  assert.deepEqual(env.writes, [["iris.appearance", "dark"]]);
});

test("system mode follows OS changes while an explicit choice stays fixed", () => {
  const env = environment({ dark: true });
  const api = create(env.browser);
  assert.equal(api.getPreference(), "system");
  assert.equal(env.theme(), "dark");
  env.changeSystem(false);
  assert.equal(env.theme(), "light");
  env.choose("dark");
  env.changeSystem(true);
  env.changeSystem(false);
  assert.equal(env.theme(), "dark");
  assert.equal(env.select.value, "dark");
  env.choose("system");
  assert.equal(env.theme(), "light");
  env.changeSystem(true);
  assert.equal(env.theme(), "dark");
  assert.equal(env.storage.getItem("iris.appearance"), "system");
});

test("storage read and write failures keep appearance usable for the current page", () => {
  const env = environment();
  env.browser.localStorage = {
    getItem() { throw new Error("Storage denied"); },
    setItem() { throw new Error("Storage full"); },
  };
  const api = create(env.browser);
  assert.equal(api.getPreference(), "system");
  assert.doesNotThrow(() => env.choose("dark"));
  assert.equal(env.theme(), "dark");
  env.changeSystem(false);
  assert.equal(env.theme(), "dark");
  assert.equal(env.select.value, "dark");
});

test("a blocked localStorage getter and missing media queries do not break initial rendering", () => {
  const env = environment({ control: false });
  Object.defineProperty(env.browser, "localStorage", { get() { throw new Error("Storage denied"); } });
  delete env.browser.matchMedia;
  const api = create(env.browser);
  assert.equal(env.theme(), "light");
  api.setPreference("dark");
  assert.equal(env.theme(), "dark");
  assert.equal(env.browser.document.documentElement.style.colorScheme, "dark");
  api.setPreference("system");
  assert.equal(env.theme(), "light");
});

test("malformed preferences fall back to the system theme without rewriting storage at startup", () => {
  const env = environment({ saved: '{"theme":"dark"}', dark: true });
  const api = create(env.browser);
  assert.equal(api.getPreference(), "system");
  assert.equal(env.select.value, "system");
  assert.equal(env.theme(), "dark");
  assert.deepEqual(env.writes, []);
  api.setPreference("light");
  api.setPreference("invalid");
  assert.equal(env.theme(), "dark");
  assert.equal(env.select.value, "system");
});

test("appearance changes and removal in another tab update this page without a write loop", () => {
  const env = environment({ saved: "light", dark: true });
  const api = create(env.browser);
  env.storageEvent("iris.appearance", "dark");
  assert.equal(api.getPreference(), "dark");
  assert.equal(env.select.value, "dark");
  assert.equal(env.theme(), "dark");
  env.storageEvent("iris.appearance", "light", {});
  env.storageEvent("iris.project", "light");
  assert.equal(env.theme(), "dark");
  env.changeSystem(false);
  env.storageEvent("iris.appearance", null);
  assert.equal(api.getPreference(), "system");
  assert.equal(env.theme(), "light");
  env.storageEvent("iris.appearance", "dark");
  env.storageEvent(null, null);
  assert.equal(env.theme(), "light");
  assert.equal(env.select.value, "system");
  assert.deepEqual(env.writes, []);
});

test("legacy media query listeners continue to update system appearance", () => {
  const env = environment();
  let listener;
  delete env.media.addEventListener;
  env.media.addListener = (callback) => { listener = callback; };
  const api = create(env.browser);
  env.media.matches = true;
  listener();
  assert.equal(api.getTheme(), "dark");
  api.setPreference("light");
  listener();
  assert.equal(api.getTheme(), "light");
});
