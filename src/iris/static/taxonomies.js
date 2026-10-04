"use strict";

(() => {
  const tools = window.IRISTaxonomyTools;
  const field = (name) => $(`#taxonomy-${name}`);
  const dialog = field("dialog");
  const editor = { versions: [], currentId: null, selectedId: null, classes: [], dirty: false, busy: false, request: 0 };
  const endpoint = `/api/projects/${encodeURIComponent(state.projectId)}/taxonomies`;
  const selected = () => editor.versions.find((item) => item.id === editor.selectedId);
  const editing = () => Boolean(selected() && editor.selectedId === editor.currentId);
  const published = (id) => selected()?.id !== tools.builtinId && selected()?.classes.some((item) => item.id === id);

  function error(message) {
    field("error").textContent = message || "";
    field("error").hidden = !message;
  }

  function controls() {
    field("save").disabled = editor.busy || !editing() || !editor.dirty || !editor.classes.length;
    field("add").disabled = editor.busy || !editing() || editor.classes.length >= 100;
    field("history").disabled = editor.busy || !editor.versions.length;
    field("refresh").disabled = editor.busy;
    field("close").disabled = editor.busy;
    field("form").setAttribute("aria-busy", String(editor.busy));
    for (const input of field("rows").querySelectorAll("input, textarea, button"))
      input.disabled = editor.busy || !editing();
    field("save").textContent = editor.busy ? "Please wait…" : "Save new class version";
    field("status").textContent = !selected() ? "Loading project classes…"
      : !editing() ? "Historical version · read only. Current project classes are unchanged."
        : editor.dirty ? "Unsaved class changes. Existing images and annotation history retain their saved definitions."
          : "Current project classes. New images and import previews use this version.";
  }

  function discardAllowed() {
    if (editor.busy) return false;
    return !editor.dirty || window.confirm("Discard unsaved class definitions?");
  }

  function updateSummary() {
    const current = editor.versions.find((item) => item.id === editor.currentId);
    if (!current) return;
    $("#project-class-summary").textContent = `${current.classes.length} ${current.classes.length === 1 ? "class" : "classes"} · ${tools.versionLabel(current)}`;
    $("#project-class-summary").title = current.classes.map((item) => item.name).join(", ");
    window.dispatchEvent(new CustomEvent("iris:taxonomy", { detail: { taxonomy: current } }));
  }

  function renderRows() {
    field("rows").replaceChildren();
    for (const [index, category] of editor.classes.entries()) {
      const row = node("fieldset", "taxonomy-class");
      row.append(node("legend", "", `Class ${index + 1}`));
      for (const [key, title, multiline] of [
        ["id", "Stable identifier"], ["name", "Display name"],
        ["definition", "Definition and annotation rules", true], ["coco_id", "Official COCO category ID · optional"],
      ]) {
        const group = node("div", key === "definition" ? "taxonomy-definition" : "");
        const input = node(multiline ? "textarea" : "input");
        input.id = `taxonomy-class-${index}-${key}`;
        input.dataset.field = key;
        input.value = category[key] ?? "";
        input.required = key !== "coco_id";
        if (key === "id") {
          input.maxLength = 64;
          input.pattern = "(?!exclude$)[a-z][a-z0-9_\\-]{0,63}";
          input.readOnly = Boolean(published(category.id));
          input.title = input.readOnly ? "Published class identifiers stay stable across versions." : "Lowercase letters, digits, hyphens or underscores; start with a letter.";
        } else if (key === "coco_id") {
          input.type = "number";
          input.min = "1";
          input.max = "90";
          input.step = "1";
          input.placeholder = "Leave blank for a custom class";
        } else if (key === "name") input.maxLength = 120;
        else { input.rows = 3; input.maxLength = 2000; }
        const label = node("label", "", title);
        label.htmlFor = input.id;
        input.addEventListener("input", () => {
          category[key] = key === "coco_id" ? input.value === "" ? null : Number(input.value) : input.value;
          editor.dirty = true;
          controls();
        });
        group.append(label, input);
        row.append(group);
      }
      if (!published(category.id)) {
        const remove = node("button", "text-button taxonomy-remove", "Remove class");
        remove.type = "button";
        remove.addEventListener("click", () => {
          editor.classes.splice(index, 1);
          editor.dirty = true;
          renderRows();
        });
        row.append(remove);
      }
      field("rows").append(row);
    }
    controls();
  }

  function loadSelection(id) {
    editor.selectedId = id;
    editor.classes = tools.editableClasses(selected());
    editor.dirty = false;
    field("history").value = id;
    field("version-id").textContent = selected()?.id || "";
    error(null);
    renderRows();
  }

  async function refresh() {
    const request = ++editor.request;
    editor.busy = true;
    controls();
    error(null);
    try {
      const result = await api(endpoint);
      if (request !== editor.request) return;
      editor.versions = result.versions;
      editor.currentId = result.current_taxonomy_id;
      field("history").replaceChildren();
      for (const version of editor.versions) {
        const option = node("option", "", `${tools.versionLabel(version)}${version.id === editor.currentId ? " · current" : ""}${version.created_at ? ` · ${new Date(version.created_at).toLocaleString()}` : ""}`);
        option.value = version.id;
        field("history").append(option);
      }
      loadSelection(editor.currentId);
      updateSummary();
    } catch (failure) {
      error(failure.message);
      if (!dialog.open) $("#project-class-summary").textContent = "Open Manage classes to load class definitions.";
    } finally {
      if (request === editor.request) { editor.busy = false; controls(); }
    }
  }

  $("#taxonomy-open").addEventListener("click", () => {
    dialog.showModal();
    if (!editor.dirty) refresh();
  });
  field("close").addEventListener("click", () => { if (discardAllowed()) dialog.close(); });
  dialog.addEventListener("cancel", (event) => { if (!discardAllowed()) event.preventDefault(); });
  dialog.addEventListener("close", () => { editor.dirty = false; });
  field("refresh").addEventListener("click", () => { if (discardAllowed()) refresh(); });
  field("history").addEventListener("change", (event) => {
    if (discardAllowed()) loadSelection(event.target.value);
    else event.target.value = editor.selectedId;
  });
  field("add").addEventListener("click", () => {
    if (editor.busy || !editing() || editor.classes.length >= 100) return;
    editor.classes.push({ id: "", name: "", definition: "", coco_id: null });
    editor.dirty = true;
    renderRows();
    field("rows").lastElementChild.querySelector("input").focus();
  });
  field("form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (editor.busy || !editing() || !editor.dirty) return;
    editor.busy = true;
    controls();
    error(null);
    try {
      await api(endpoint, { method: "POST", body: JSON.stringify({
        expected_taxonomy_id: editor.currentId,
        classes: editor.classes.map((item) => ({ ...item, id: item.id.trim(), name: item.name.trim(), definition: item.definition.trim() })),
      }) });
      editor.dirty = false;
      await refresh();
      notify("Class version saved. Existing images keep their definitions until you explicitly update them in annotation.");
    } catch (failure) {
      error(`${failure.message}${failure.status === 409 ? " Reload versions before saving again." : ""}`);
    } finally {
      editor.busy = false;
      controls();
    }
  });
  window.addEventListener("beforeunload", (event) => {
    if (editor.dirty || editor.busy && dialog.open) { event.preventDefault(); event.returnValue = ""; }
  });
  window.addEventListener("iris:project-ready", refresh);
  if (state.projects.some((project) => project.id === state.projectId)) refresh();
})();
