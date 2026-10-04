"use strict";

(() => {
  const dialog = $("#dataset-import-dialog");
  const importer = {
    records: [],
    detail: null,
    mapping: {},
    imageIndex: 0,
    busy: false,
    loading: false,
    request: 0,
  };
  const base = "/api/dataset-imports";
  const svgNS = "http://www.w3.org/2000/svg";
  const fields = {
    name: "#dataset-import-name",
    scene_group: "#dataset-import-group",
    source_url: "#dataset-import-source",
    license_name: "#dataset-import-license",
    attribution: "#dataset-import-attribution",
    source_split: "#dataset-import-split",
  };

  function report(message) {
    $("#dataset-import-error").textContent = message || "";
    $("#dataset-import-error").hidden = !message;
    if (message && !dialog.open) notify(message, true);
  }

  function status(message) {
    $("#dataset-import-status").textContent = message;
  }

  function updateControls() {
    const disabled = importer.busy || importer.loading;
    const imported = importer.detail?.status === "imported";
    for (const selector of [
      "#dataset-import-file",
      "#dataset-import-preview-button",
      "#dataset-import-record",
    ]) $(selector).disabled = disabled;
    $("#dataset-import-settings").disabled = disabled || imported;
    $("#dataset-import-split").disabled = disabled || imported || !!importer.detail?.source_split;
    for (const select of $("#dataset-import-mapping").querySelectorAll("select"))
      select.disabled = disabled || imported;
    $("#dataset-import-commit").disabled = disabled || imported || !mappingComplete();
    $("#dataset-import-review").disabled = disabled;
    const count = importer.detail?.images?.length || 0;
    $("#dataset-import-previous").disabled = disabled || importer.imageIndex <= 0;
    $("#dataset-import-next").disabled = disabled || importer.imageIndex >= count - 1;
    $("#dataset-import-image-select").disabled = disabled;
    $("#dataset-import-dialog").setAttribute("aria-busy", String(disabled));
  }

  function mappingComplete() {
    return !!importer.detail && importer.detail.categories.every((category) =>
      ["person", "car", "exclude"].includes(importer.mapping[String(category.id)]),
    );
  }

  function mappingSummary() {
    let proposals = 0;
    let excluded = 0;
    let unassigned = 0;
    for (const category of importer.detail?.categories || []) {
      const value = importer.mapping[String(category.id)];
      if (value === "exclude") excluded += category.count;
      else if (value === "person" || value === "car") proposals += category.count;
      else unassigned += 1;
    }
    $("#dataset-import-mapping-summary").textContent =
      `${proposals} review proposals · ${excluded} source annotations excluded${unassigned ? ` · ${unassigned} categories still need a choice` : ""}.`;
  }

  function renderRecords() {
    const select = $("#dataset-import-record");
    select.replaceChildren(node("option", "", "Choose a saved archive…"));
    select.firstElementChild.value = "";
    for (const record of importer.records) {
      const option = node(
        "option", "",
        `${record.filename} · ${record.image_count} images · ${record.status === "imported" ? "imported" : "preview"}`,
      );
      option.value = record.id;
      select.append(option);
    }
    select.value = importer.detail?.id || "";
  }

  async function refreshRecords() {
    const records = await api(base);
    importer.records = records;
    renderRecords();
  }

  function renderMappings() {
    const container = $("#dataset-import-mapping");
    container.replaceChildren();
    importer.detail.categories.forEach((category, index) => {
      const row = node("div", "dataset-import-category");
      const label = node("label", "", `${category.name} · ${category.count}`);
      const select = node("select");
      select.id = `dataset-import-category-${index}`;
      label.htmlFor = select.id;
      select.required = true;
      select.dataset.categoryId = String(category.id);
      for (const [value, title] of [
        ["", "Choose a mapping…"],
        ["person", "Person"],
        ["car", "Car"],
        ["exclude", "Exclude this category"],
      ]) {
        const option = node("option", "", title);
        option.value = value;
        select.append(option);
      }
      select.value = importer.mapping[String(category.id)] || "";
      select.addEventListener("change", () => {
        importer.mapping[String(category.id)] = select.value;
        $("#dataset-import-confirm").checked = false;
        mappingSummary();
        renderBoxes();
        updateControls();
      });
      row.append(label, select);
      container.append(row);
    });
    if (!importer.detail.categories.length)
      container.append(node("p", "field-hint", "No source categories. Review all images for missing objects."));
    mappingSummary();
  }

  function svg(tag, attributes) {
    const element = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attributes))
      element.setAttribute(key, String(value));
    return element;
  }

  function renderBoxes() {
    const image = importer.detail?.images[importer.imageIndex];
    const layer = $("#dataset-import-boxes");
    layer.replaceChildren();
    if (!image) return;
    layer.setAttribute("viewBox", `0 0 ${image.width} ${image.height}`);
    layer.hidden = !$("#dataset-import-show-boxes").checked;
    if (layer.hidden) return;
    const fontSize = Math.max(image.width / 60, 10);
    for (const box of image.boxes) {
      const target = importer.mapping[String(box.category_id)];
      const color = target === "person" ? "#8ef1ac" : target === "car" ? "#83caff" : "#e0e3e5";
      const [x1, y1, x2, y2] = box.box;
      const group = svg("g", { class: "dataset-import-box" });
      group.append(svg("rect", {
        x: x1, y: y1, width: x2 - x1, height: y2 - y1,
        fill: "none", stroke: color, "stroke-width": 2,
        "vector-effect": "non-scaling-stroke",
        "stroke-dasharray": target === "person" || target === "car" ? "none" : "4 3",
      }));
      const title = svg("title", {});
      title.textContent = `${box.category_name} → ${target || "unmapped"}`;
      group.append(title);
      const text = svg("text", {
        x: Math.max(0, Math.min(x1, image.width - fontSize * 8)),
        y: Math.min(image.height - 2, Math.max(fontSize, y1 - 4)),
        fill: color, "font-size": fontSize, "paint-order": "stroke",
        stroke: "#17231e", "stroke-width": Math.max(2, fontSize / 5),
        "stroke-linejoin": "round",
      });
      text.textContent = `${box.category_name} → ${target || "?"}`.slice(0, 45);
      group.append(text);
      layer.append(group);
    }
  }

  function renderImage() {
    const item = importer.detail?.images[importer.imageIndex];
    if (!item) return;
    const image = $("#dataset-import-image");
    const expected = `${base}/${encodeURIComponent(importer.detail.id)}/images/${encodeURIComponent(item.id)}`;
    const source = new URL(item.image_url || expected, window.location.href);
    if (source.origin !== window.location.origin) {
      image.removeAttribute("src");
      report("The preview image has an unexpected external URL.");
      return;
    }
    image.alt = `Source image ${item.filename}`;
    image.width = item.width;
    image.height = item.height;
    image.src = projectURL(source.href);
    $("#dataset-import-image-select").value = String(importer.imageIndex);
    $("#dataset-import-image-info").textContent =
      `${importer.imageIndex + 1} / ${importer.detail.images.length} · ${item.width} × ${item.height} · ${item.annotation_count} source boxes`;
    renderBoxes();
    updateControls();
  }

  function showDetail(detail) {
    importer.detail = detail;
    importer.imageIndex = 0;
    importer.mapping = {};
    const config = detail.config || {};
    for (const category of detail.categories) {
      const suggested = ["person", "car"].includes(category.name.toLowerCase().trim())
        ? category.name.toLowerCase().trim() : "";
      importer.mapping[String(category.id)] = config.category_mapping?.[String(category.id)] || suggested;
    }
    $("#dataset-import-detail").hidden = false;
    $("#dataset-import-filename").textContent = detail.filename;
    $("#dataset-import-counts").textContent =
      `${detail.image_count} images · ${detail.annotation_count} annotations · ${detail.categories.length} source categories`;
    const warnings = $("#dataset-import-warnings");
    warnings.replaceChildren();
    for (const warning of detail.warnings || []) warnings.append(node("p", "field-hint", warning));
    renderMappings();
    const imageSelect = $("#dataset-import-image-select");
    imageSelect.replaceChildren();
    detail.images.forEach((item, index) => {
      const option = node("option", "", `${index + 1}. ${item.filename}`);
      option.value = String(index);
      imageSelect.append(option);
    });
    for (const [key, selector] of Object.entries(fields)) $(selector).value = config[key] || "";
    if (detail.source_split) $("#dataset-import-split").value = detail.source_split;
    $("#dataset-import-split-hint").hidden = !detail.source_split;
    $("#dataset-import-split-hint").textContent = detail.source_split
      ? `The archive declares the ${detail.source_split} split. This assignment is preserved.` : "";
    if (!config.name) $(fields.name).value = detail.filename.replace(/\.zip$/i, "");
    $("#dataset-import-confirm").checked = false;
    const imported = detail.status === "imported";
    $("#dataset-import-complete").hidden = !imported;
    $("#dataset-import-commit-actions").hidden = imported;
    if (imported) {
      const result = detail.result;
      $("#dataset-import-result").textContent =
        `${result.frame_ids.length} frames added, ${result.proposal_count} proposals awaiting review and ${result.excluded_annotation_count} source annotations excluded. No image has been validated by this import.`;
    }
    $("#dataset-import-provenance").textContent = JSON.stringify({
      import_id: detail.id,
      filename: detail.filename,
      archive_sha256: detail.sha256,
      created_at: detail.created_at,
      source_info: detail.info,
      source_licenses: detail.licenses,
      ...(imported ? { imported_as: config, result: detail.result } : {}),
    }, null, 2);
    renderRecords();
    renderImage();
    updateControls();
  }

  async function loadDetail(id) {
    if (!id || importer.busy) return;
    const request = ++importer.request;
    importer.loading = true;
    updateControls();
    report(null);
    status("Loading saved archive…");
    try {
      const detail = await api(`${base}/${encodeURIComponent(id)}`);
      if (request !== importer.request) return;
      showDetail(detail);
      status("");
    } catch (error) {
      if (request === importer.request) report(error.message);
    } finally {
      if (request === importer.request) {
        importer.loading = false;
        updateControls();
      }
    }
  }

  $("#dataset-import-open").addEventListener("click", async () => {
    if (!dialog.open) dialog.showModal();
    if (importer.busy || importer.loading) return;
    const request = ++importer.request;
    importer.loading = true;
    updateControls();
    report(null);
    status("Loading saved imports…");
    try {
      await refreshRecords();
      if (request !== importer.request) return;
      status(importer.records.length ? "Saved previews can be resumed after restarting IRIS." : "Choose a local archive to preview its images and labels.");
    } catch (error) {
      if (request === importer.request) report(error.message);
    } finally {
      if (request === importer.request) {
        importer.loading = false;
        updateControls();
      }
    }
  });
  $("#dataset-import-close").addEventListener("click", () => dialog.close());
  $("#dataset-import-record").addEventListener("change", (event) => loadDetail(event.target.value));
  $("#dataset-import-upload").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (importer.busy || importer.loading) return;
    const file = $("#dataset-import-file").files[0];
    if (!file) return;
    if (file.size > 64 * 1024 * 1024) {
      report("Choose a ZIP archive no larger than 64 MiB. Prepare a small subset first.");
      return;
    }
    if (!/\.zip$/i.test(file.name)) {
      report("Choose a ZIP archive containing a COCO JSON file and its images.");
      return;
    }
    importer.busy = true;
    updateControls();
    report(null);
    status("Checking archive and preparing local image previews…");
    const body = new FormData();
    body.append("file", file);
    try {
      const detail = await api(base, { method: "POST", body });
      importer.records = [detail, ...importer.records.filter((item) => item.id !== detail.id)];
      showDetail(detail);
      $("#dataset-import-file").value = "";
      status("Preview saved. Check the boxes, mapping and source before importing.");
      if (!dialog.open) notify("COCO archive preview saved. Reopen Import annotated dataset to continue.");
    } catch (error) {
      status("");
      report(error.message);
    } finally {
      importer.busy = false;
      updateControls();
    }
  });

  $("#dataset-import-commit-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (importer.busy || importer.loading || !importer.detail || importer.detail.status === "imported") return;
    if (!mappingComplete()) {
      report("Choose a target class or explicitly exclude every source category.");
      return;
    }
    if (!$("#dataset-import-confirm").checked) {
      report("Check the source, mapping and exclusions before importing.");
      return;
    }
    const payload = { category_mapping: { ...importer.mapping } };
    for (const [key, selector] of Object.entries(fields)) payload[key] = $(selector).value.trim();
    payload.source_split ||= null;
    const id = importer.detail.id;
    importer.busy = true;
    updateControls();
    report(null);
    status("Importing images and review proposals…");
    try {
      const result = await api(`${base}/${encodeURIComponent(id)}/commit`, {
        method: "POST", body: JSON.stringify(payload),
      });
      showDetail({ ...importer.detail, status: "imported", result, config: payload });
      const record = importer.records.find((item) => item.id === id);
      if (record) record.status = "imported";
      renderRecords();
      status("Import saved. Review every frame in Annotation before publishing a dataset version.");
      state.sessions = await api("/api/sessions");
      renderSessions();
      if (!dialog.open) notify("COCO import complete. Open the imported session to review its annotations.");
    } catch (error) {
      status("");
      report(error.message);
    } finally {
      importer.busy = false;
      updateControls();
    }
  });

  $("#dataset-import-review").addEventListener("click", async () => {
    const sessionId = importer.detail?.result?.session_id;
    if (!sessionId || importer.busy || importer.loading) return;
    importer.busy = true;
    updateControls();
    report(null);
    try {
      state.sessions = await api("/api/sessions");
      renderSessions();
      if (state.sessionId !== sessionId) await selectSession(sessionId);
      if (state.sessionId !== sessionId) {
        status("The imported session is saved. Finish or save your current annotation before switching.");
        return;
      }
      $("#workspace-annotation").click();
      dialog.close();
    } catch (error) {
      report(error.message);
    } finally {
      importer.busy = false;
      updateControls();
    }
  });

  $("#dataset-import-image-select").addEventListener("change", (event) => {
    importer.imageIndex = Number(event.target.value);
    renderImage();
  });
  for (const [selector, delta] of [["#dataset-import-previous", -1], ["#dataset-import-next", 1]])
    $(selector).addEventListener("click", () => {
      importer.imageIndex += delta;
      renderImage();
    });
  $("#dataset-import-show-boxes").addEventListener("change", renderBoxes);
  $("#dataset-import-image").addEventListener("error", () => {
    if (importer.detail) report("The preview image could not be loaded. Reopen this saved archive or check the local server.");
  });
  for (const selector of Object.values(fields)) $(selector).addEventListener("input", () => {
    $("#dataset-import-confirm").checked = false;
  });
})();
