"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

function environment({ compact = false, sessionId = "session" } = {}) {
  const document = { activeElement: null, createTextNode: (text) => ({ text }) };
  class Element extends EventTarget {
    constructor(selector) {
      super();
      this.selector = selector;
      this.hidden = false;
      this.attributes = new Map();
      this.dataset = {};
      this.value = "";
      this.classes = new Set();
      this.classList = { toggle: (name, active) => active ? this.classes.add(name) : this.classes.delete(name) };
    }
    setAttribute(name, value) { this.attributes.set(name, value); }
    removeAttribute(name) { this.attributes.delete(name); }
    replaceChildren(...children) { this.children = children; }
    contains(element) { return element === this || element?.parent === this; }
    focus() { if (!this.hidden && !this.parent?.hidden) document.activeElement = this; }
    matches(selector) { return selector.split(", ").includes(this.tagName); }
    click() { this.dispatchEvent(new Event("click")); }
  }
  const nodes = new Map();
  function $(selector) {
    if (!nodes.has(selector)) nodes.set(selector, new Element(selector));
    return nodes.get(selector);
  }
  const window = new EventTarget();
  const media = new EventTarget();
  media.matches = compact;
  window.matchMedia = () => media;
  const state = { sessionId };
  $("#session-name").parent = $("#workspace-sidebar");
  $("#session-name").tagName = "input";
  vm.runInNewContext(fs.readFileSync(require.resolve("../../src/iris/static/navigation.js"), "utf8"), {
    window, document, state, $, CustomEvent,
    node: (tag, className, text) => ({ tag, className, text }),
  });
  return {
    $, window, document, state, api: window.IRISNavigation,
    resize(compact) { media.matches = compact; media.dispatchEvent(new Event("change")); },
  };
}

test("a rejected workspace transition preserves the view, sidebar and form state", () => {
  const env = environment({ compact: true });
  env.api.open("annotation");
  env.api.openSidebar();
  env.$("#session-name").value = "Unsubmitted session";
  const changed = [];
  env.window.addEventListener("iris:workspace", (event) => changed.push(event.detail.name));
  env.window.addEventListener("iris:before-workspace", (event) => event.preventDefault());
  assert.equal(env.api.open("training"), false);
  assert.equal(env.$("#annotation-workspace").hidden, false);
  assert.equal(env.$("#training-workspace").hidden, true);
  assert.equal(env.$("#workspace-annotation").attributes.get("aria-current"), "page");
  assert.equal(env.$("#workspace-sidebar").hidden, false);
  assert.equal(env.$("#session-name").value, "Unsubmitted session");
  assert.deepEqual(changed, []);
});

test("project destinations work without a session, including when session loading completes later", () => {
  const env = environment({ sessionId: null });
  const changed = [];
  env.window.addEventListener("iris:workspace", (event) => changed.push(event.detail.name));
  for (const name of ["training", "evaluation", "experiments", "benchmark"]) {
    assert.equal(env.api.open(name), true);
    env.api.syncSession();
    assert.equal(env.$(`#${name}-workspace`).hidden, false);
    assert.equal(env.$("#session-workspace").hidden, true);
    assert.equal(env.$("#welcome").hidden, true);
  }
  assert.deepEqual(changed, ["training", "evaluation", "experiments", "benchmark"]);
  for (const name of ["intake", "comparison", "annotation"]) {
    env.api.open(name);
    assert.equal(env.$("#session-workspace").hidden, true);
    assert.equal(env.$("#welcome").hidden, false);
  }
});

test("creating the first session preserves an open project view and makes session views usable", () => {
  const env = environment({ compact: true, sessionId: null });
  env.api.open("training");
  env.api.openSidebar();
  env.state.sessionId = "first-session";
  env.window.dispatchEvent(new Event("iris:session"));
  assert.equal(env.$("#training-workspace").hidden, false);
  assert.equal(env.$("#session-workspace").hidden, true);
  assert.equal(env.$("#welcome").hidden, true);
  assert.equal(env.$("#workspace-sidebar").hidden, true);
  env.api.open("intake");
  assert.equal(env.$("#intake-workspace").hidden, false);
  assert.equal(env.$("#session-workspace").hidden, false);
  assert.equal(env.$("#welcome").hidden, true);
});

test("compact sidebar moves focus before hiding and keeps unsubmitted input on resize", () => {
  const env = environment();
  const field = env.$("#session-name");
  const toggle = env.$("#sidebar-toggle");
  field.value = "Field survey";
  field.focus();
  env.resize(true);
  assert.equal(env.document.activeElement, toggle);
  assert.equal(env.$("#workspace-sidebar").hidden, true);
  assert.equal(toggle.attributes.get("aria-expanded"), "false");
  env.$("#start-session").click();
  assert.equal(env.$("#workspace-sidebar").hidden, false);
  assert.equal(toggle.attributes.get("aria-expanded"), "true");
  assert.equal(field.value, "Field survey");
  toggle.focus();
  env.resize(false);
  assert.equal(env.$("#workspace-sidebar").hidden, false);
  assert.equal(toggle.hidden, true);
  assert.equal(env.document.activeElement, env.$("#main"));
});
