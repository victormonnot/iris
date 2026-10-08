"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

test("temporal identities is independent workspace09 and honors unsaved transition cancellation", () => {
  const nodes = new Map(), document = { activeElement: null, createTextNode: (text) => text };
  class Element extends EventTarget {
    constructor() { super(); this.attributes = {}; this.dataset = {}; this.classList = { toggle() {} }; }
    contains() { return false; }
    setAttribute(key, value) { this.attributes[key] = value; }
    removeAttribute(key) { delete this.attributes[key]; }
    replaceChildren(...children) { this.children = children; }
    focus() { document.activeElement = this; }
  }
  const $ = (selector) => { if (!nodes.has(selector)) nodes.set(selector, new Element()); return nodes.get(selector); };
  const window = new EventTarget(), media = new EventTarget(); media.matches = false; window.matchMedia = () => media;
  const state = { sessionId: null };
  vm.runInNewContext(fs.readFileSync(require.resolve("../../src/iris/static/navigation.js"), "utf8"), { window, document, state, $, CustomEvent, node: (tag, className, text) => text });
  assert.equal(window.IRISNavigation.open("identities"), true);
  assert.equal($("#identities-workspace").hidden, false);
  assert.equal($("#tracking-workspace").hidden, true);
  assert.equal($("#annotation-workspace").hidden, true);
  assert.equal($("#session-workspace").hidden, true);
  assert.equal($("#welcome").hidden, true);
  assert.equal($("#workspace-identities").attributes["aria-current"], "page");
  assert.equal($("#workspace-step").children[0], "09");
  state.sessionId = "a-later-session";
  window.dispatchEvent(new Event("iris:session"));
  assert.equal($("#identities-workspace").hidden, false);
  assert.equal($("#session-workspace").hidden, true);
  window.addEventListener("iris:before-workspace", (event) => event.preventDefault());
  assert.equal(window.IRISNavigation.open("tracking"), false);
  assert.equal($("#identities-workspace").hidden, false);
  assert.equal($("#tracking-workspace").hidden, true);
});
