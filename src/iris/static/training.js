"use strict";

(() => {
  const datasetTools = window.IRISDatasetTools;
  const taxonomyTools = window.IRISTaxonomyTools;
  const recoveryTools = window.IRISTrainingRecovery;
  const workspace = {
    visible: false,
    candidates: null,
    taxonomyId: null,
    taxonomies: [],
    choices: new Map(),
    candidateRequest: 0,
    candidateLoading: false,
    candidateNeedsRefresh: false,
    partitionPlan: null,
    partitionRequest: 0,
    partitionLoading: false,
    partitionApplying: false,
    partitionApplied: false,
    datasetBusy: false,
    datasets: [],
    datasetId: null,
    datasetRequest: 0,
    datasetHistoryRequest: 0,
    datasetDetail: null,
    exportRequest: 0,
    exportController: null,
    exportUrls: new Map(),
    models: [],
    trainings: [],
    trainingId: null,
    trainingDetail: null,
    trainingRequest: 0,
    trainingBusy: false,
    trainingOperation: null,
    trainingPreview: null,
    trainingPreviewKey: null,
    trainingPreviewRequest: 0,
    trainingModelsRequest: 0,
    trainingModelsLoading: false,
    trainingModelsError: null,
    trainingDevices: recoveryTools.deviceChoices(null),
    trainingDevicesRequest: 0,
    trainingDevicesLoading: false,
    trainingDevicesError: null,
    trainingDeviceNotes: [],
    resumePreview: null,
    resumeRequest: 0,
    resumeBusy: false,
    resumeOperation: null,
    lossPage: 0,
    historyRequest: 0,
    jobStatuses: new Map(),
  };
  const splitNames = { train: "Train", val: "Validation", test: "Test" };
  const trainingScopes = {
    prediction_head_only: {
      label: "Light · prediction head",
      description:
        "Update the final classification and box prediction layers. Keep visual features frozen for a small baseline run.",
      cost:
        "The light scope updates the fewest parameters. The detector still processes each training image on the selected device.",
    },
    partial_backbone: {
      label: "Partial · late features and detection heads",
      description:
        "Update late visual features and the detection heads. This is an option when adapting to a different camera or image appearance, including analog footage.",
      cost:
        "Updating features and detection heads requires more computation and memory than the light scope. No duration estimate is available.",
    },
    full_model: {
      label: "Full · all trainable layers",
      description:
        "Update all trainable layers for the broadest adaptation. This can overfit a small dataset; use held-out evaluation to check the result.",
      cost:
        "Updating all trainable layers requires more computation and memory than the light scope. No duration estimate is available.",
    },
  };
  const scopeLabel = (scope, modelId) => {
    const id = scope || "prediction_head_only";
    return modelScopes(workspace.models.find((model) => model.id === modelId))
      .find((item) => item.id === id)?.label || trainingScopes[id]?.label || id;
  };
  const selectedTrainingModel = () => workspace.models.find(
    (item) => item.id === $("#training-parent").value,
  );
  function modelScopes(model) {
    if (!model?.training) return [];
    return Array.isArray(model.training_scopes)
      ? model.training_scopes
      : Object.entries(trainingScopes).map(([id, scope]) => ({ id, ...scope }));
  }

  function renderTrainingScopes() {
    const select = $("#training-scope");
    const previous = select.value;
    const scopes = modelScopes(selectedTrainingModel());
    select.replaceChildren();
    for (const scope of scopes) select.append(new Option(scope.label, scope.id));
    if (!scopes.length) select.append(new Option("Choose a supported checkpoint", ""));
    select.value = scopes.some((scope) => scope.id === previous) ? previous : scopes[0]?.id || "";
    return select.value !== previous;
  }

  function showError(selector, error) {
    $(selector).textContent = error?.message || "";
    $(selector).hidden = !error;
  }

  function warnings(selector, items) {
    const container = $(selector);
    container.replaceChildren();
    for (const text of items || [])
      container.append(node("p", "field-hint", text));
  }

  function chosenGroups() {
    return (workspace.candidates?.groups || []).filter(
      (group) => splitNames[workspace.choices.get(group.scene_group)],
    );
  }

  function updateDatasetLaunch() {
    const counts = { train: 0, val: 0, test: 0 };
    const groups = chosenGroups();
    for (const group of groups)
      counts[workspace.choices.get(group.scene_group)] += group.frames.length;
    $("#dataset-selection-summary").textContent = groups.length
      ? `${groups.length} scene groups · ${counts.train} train / ${counts.val} validation / ${counts.test} test frames`
      : "No groups included";
    const readiness = $("#dataset-readiness");
    if (workspace.datasetBusy)
      readiness.textContent = "Saving the reviewed images and labels in a frozen version…";
    else if (workspace.candidateLoading || workspace.partitionApplying)
      readiness.textContent = "Checking the latest reviewed data…";
    else if (workspace.candidateNeedsRefresh)
      readiness.textContent = "Reviewed data changed. Refresh candidates before freezing.";
    else if (!workspace.candidates)
      readiness.textContent = "Refresh candidates to load the current reviewed data.";
    else if (!workspace.candidates.groups?.length)
      readiness.textContent = "Choose a class version with eligible frames, or validate selected images in Annotation.";
    else if (!counts.train || !counts.val)
      readiness.textContent = "Assign at least one group to train and another to validation.";
    else
      readiness.textContent = "Train and validation are assigned. Review the groups and name your release before freezing.";
    $("#dataset-create").disabled =
      workspace.datasetBusy ||
      workspace.partitionApplying ||
      workspace.candidateLoading ||
      workspace.candidateNeedsRefresh ||
      !counts.train ||
      !counts.val;
    $("#dataset-refresh").disabled =
      workspace.datasetBusy || workspace.candidateLoading || workspace.partitionApplying;
    $("#dataset-taxonomy").disabled =
      workspace.datasetBusy || workspace.candidateLoading || workspace.partitionApplying || !workspace.taxonomies.length;
    $("#dataset-parent").disabled = workspace.datasetBusy || workspace.candidateLoading || workspace.partitionApplying;
    for (const select of $("#dataset-groups").querySelectorAll("select")) {
      select.disabled = workspace.datasetBusy || workspace.candidateLoading || workspace.partitionApplying;
      select.closest(".dataset-group").dataset.included = String(Boolean(select.value));
    }
    $("#dataset-create").textContent = workspace.datasetBusy
      ? "Freezing release…"
      : "Freeze release →";
    updatePartitionControls();
  }

  function renderCandidates() {
    const container = $("#dataset-groups");
    container.replaceChildren();
    const candidates = workspace.candidates;
    const groups = candidates?.groups || [];
    const count = groups.reduce(
      (total, group) => total + group.frames.length,
      0,
    );
    const exclusions = Object.entries(candidates?.excluded || {})
      .filter(([, value]) => value > 0)
      .filter(([key]) => key !== "different_taxonomy")
      .map(([key, value]) => `${value} ${key.replaceAll("_", " ")}`);
    $("#dataset-candidate-status").textContent =
      `${count} eligible frames in ${groups.length} scene groups.` +
      (exclusions.length ? ` Excluded: ${exclusions.join(" · ")}.` : "");
    const different = candidates?.excluded?.different_taxonomy || 0;
    $("#dataset-taxonomy-excluded").textContent = different
      ? `${different} validated frame${different === 1 ? " uses" : "s use"} another class version and ${different === 1 ? "is" : "are"} excluded. Choose that version to build a separate release, or explicitly update and revalidate the images in Annotation.`
      : "A release includes one exact class version. Images reviewed with other versions stay separate.";
    if (!groups.length)
      container.append(
        node(
          "p",
          "dataset-empty",
          "No eligible frames for this class version. Select frames in Data intake, resolve every proposal and save a human validation in Annotation, or choose another saved class version.",
        ),
      );
    for (const [index, group] of groups.entries()) {
      const row = node("div", "dataset-group");
      const context = node("div", "dataset-group-context");
      context.append(node("strong", "", group.scene_group));
      context.append(
        node(
          "p",
          "small muted",
          (group.sessions || []).map((session) => session.name).join(" · "),
        ),
      );
      const boxes = group.frames.reduce(
        (total, frame) => total + (frame.box_count || 0),
        0,
      );
      context.append(
        node(
          "p",
          "field-hint",
          `${group.frames.length} validated frames · ${boxes} boxes`,
        ),
      );
      if (group.reserved_split)
        context.append(
          node(
            "p",
            "dataset-reservation",
            `Reserved for ${splitNames[group.reserved_split].toLowerCase()} by an existing release`,
          ),
        );
      const pixelSplits = new Set(
        group.frames.map((frame) => frame.reserved_split).filter(Boolean),
      );
      if (group.reserved_split) pixelSplits.add(group.reserved_split);
      if (pixelSplits.size > 1 || (!group.reserved_split && pixelSplits.size))
        context.append(
          node(
            "p",
            "dataset-reservation",
            pixelSplits.size === 1
              ? `Previously frozen image pixels must stay in ${splitNames[[...pixelSplits][0]].toLowerCase()}.`
              : "This group contains pixels reserved for different splits. Exclude conflicting frames in Data intake before freezing it.",
          ),
        );
      const field = node("div", "dataset-group-split");
      const label = node("label", "", "Split");
      const select = node("select");
      select.id = `dataset-group-${index}`;
      label.htmlFor = select.id;
      select.setAttribute("aria-label", `Split for ${group.scene_group}`);
      select.append(new Option("Exclude", ""));
      for (const [value, text] of Object.entries(splitNames)) {
        const option = new Option(text, value);
        option.disabled =
          pixelSplits.size > 1 ||
          (pixelSplits.size === 1 && !pixelSplits.has(value));
        select.append(option);
      }
      let choice = workspace.choices.get(group.scene_group);
      if (choice === undefined) choice = group.reserved_split || "";
      if (
        pixelSplits.size > 1 ||
        (choice && pixelSplits.size === 1 && !pixelSplits.has(choice))
      )
        choice = "";
      workspace.choices.set(group.scene_group, choice);
      select.value = choice;
      select.disabled = workspace.datasetBusy || workspace.candidateLoading || workspace.partitionApplying;
      select.addEventListener("change", () => {
        workspace.choices.set(group.scene_group, select.value);
        invalidatePartitionPlan("Split changed manually. Preview again to propose new assignments.");
        updateDatasetLaunch();
      });
      field.append(label, select);
      row.append(context, field);
      container.append(row);
    }
    warnings("#dataset-warnings", candidates?.warnings);
    updateDatasetLaunch();
  }

  async function refreshCandidates() {
    invalidatePartitionPlan();
    const request = ++workspace.candidateRequest;
    const requestedTaxonomy = workspace.taxonomyId;
    workspace.candidateLoading = true;
    updateDatasetLaunch();
    showError("#dataset-error", null);
    try {
      const candidates = await api(`/api/dataset-candidates${requestedTaxonomy ? `?taxonomy_id=${encodeURIComponent(requestedTaxonomy)}` : ""}`);
      if (request !== workspace.candidateRequest) return;
      candidates.taxonomy = taxonomyTools.snapshot(candidates.taxonomy);
      if (workspace.taxonomyId !== candidates.taxonomy.id)
        workspace.choices.clear();
      workspace.taxonomyId = candidates.taxonomy.id;
      workspace.taxonomies = candidates.taxonomies || [candidates.taxonomy];
      workspace.candidates = candidates;
      workspace.candidateNeedsRefresh = false;
      renderCandidateTaxonomy();
      datasetOptions();
    } catch (error) {
      if (request !== workspace.candidateRequest) return;
      workspace.candidates = null;
      $("#dataset-groups").replaceChildren();
      $("#dataset-candidate-status").textContent =
        "Could not load dataset candidates. Refresh to try again.";
      showError("#dataset-error", error);
    } finally {
      if (request === workspace.candidateRequest) {
        workspace.candidateLoading = false;
        if (workspace.candidates) renderCandidates();
        updateDatasetLaunch();
      }
    }
  }

  function renderDefinitions(selector, taxonomy) {
    const target = $(selector);
    target.replaceChildren();
    for (const category of taxonomy.classes) {
      const row = node("p");
      row.append(node("strong", "", `${category.name} (${category.id}). `),
        document.createTextNode(category.definition));
      target.append(row);
    }
  }

  function updatePartitionControls() {
    const blocked = workspace.datasetBusy || workspace.candidateLoading || workspace.partitionApplying;
    for (const input of $(".partition-settings").querySelectorAll("input")) input.disabled = blocked;
    $("#dataset-plan-preview").disabled = blocked || workspace.partitionLoading || workspace.candidateNeedsRefresh || !workspace.candidates?.groups?.length;
    $("#dataset-plan-preview").textContent = workspace.partitionLoading ? "Preparing preview…" : "Preview partitions";
    $("#dataset-plan-apply").disabled = blocked || workspace.partitionLoading || workspace.partitionApplied || !workspace.partitionPlan?.can_freeze;
    $("#dataset-plan-apply").textContent = workspace.partitionApplying ? "Checking reviewed frames…" : workspace.partitionApplied ? "Partitions applied" : "Apply proposed partitions";
  }

  function invalidatePartitionPlan(message = "") {
    workspace.partitionRequest++;
    workspace.partitionPlan = null;
    workspace.partitionLoading = false;
    workspace.partitionApplying = false;
    workspace.partitionApplied = false;
    $("#dataset-plan-result").hidden = true;
    $("#dataset-plan-status").textContent = message;
    showError("#dataset-plan-error", null);
    updatePartitionControls();
  }

  function partitionSettings() {
    const values = {};
    for (const split of ["train", "val", "test"]) {
      const input = $(`#dataset-plan-${split}`);
      if (!input.reportValidity() || input.value === "") throw new Error("Enter a percentage for every split.");
      values[split] = Number(input.value);
    }
    if (values.train + values.val + values.test !== 100) throw new Error("Train, validation and test percentages must add up to 100.");
    const seed = $("#dataset-plan-seed");
    if (!seed.reportValidity() || seed.value === "") throw new Error("Enter a whole-number seed.");
    return { taxonomy_id: workspace.taxonomyId, ratios: Object.fromEntries(Object.entries(values).map(([split, value]) => [split, value / 100])), seed: Number(seed.value) };
  }

  function renderPartitionPlan(plan) {
    $("#dataset-plan-result").hidden = false;
    const summary = $("#dataset-plan-summary");
    summary.replaceChildren();
    for (const [split, label] of Object.entries(splitNames)) {
      const item = node("div");
      item.append(node("strong", "", String(plan.summary.split_counts[split])),
        node("span", "", `${label} frames`));
      summary.append(item);
    }
    const body = $("#dataset-plan-coverage");
    body.replaceChildren();
    const rows = [
      ...plan.taxonomy.classes.map((item) => ({ name: item.name, counts: Object.fromEntries(Object.keys(splitNames).map((split) => [split, plan.summary.split_class_counts[split]?.[item.id] || 0])) })),
      { name: "Validated negative images", counts: plan.summary.split_negative_counts },
    ];
    for (const row of rows) {
      const tr = node("tr");
      tr.append(node("th", "", row.name));
      tr.firstChild.scope = "row";
      for (const split of Object.keys(splitNames)) tr.append(node("td", "", String(row.counts[split] || 0)));
      body.append(tr);
    }
    warnings("#dataset-plan-warnings", plan.warnings);
    $("#dataset-plan-blockers").replaceChildren(...(plan.blockers || []).map((blocker) => node("p", "", blocker.message)));
    $("#dataset-plan-blockers").hidden = !plan.blockers?.length;
    const groups = $("#dataset-plan-groups");
    groups.replaceChildren();
    for (const group of plan.groups) {
      const item = node("div", "partition-preview-group");
      item.append(node("strong", "", `${group.scene_group} → ${splitNames[group.split] || "Unassigned"}`));
      item.append(node("p", "field-hint", `${group.count} frames · ${group.negative_count} validated negatives${group.reserved_split ? ` · reserved ${splitNames[group.reserved_split].toLowerCase()}` : ""}${group.related_groups?.length ? ` · linked groups: ${group.related_groups.join(", ")}` : ""}`));
      groups.append(item);
    }
    const related = $("#dataset-plan-related");
    related.replaceChildren();
    for (const pair of plan.similar_pairs || []) {
      const row = node("div", "partition-preview-group");
      row.append(node("p", "field-hint", `${pair.scene_groups.join(" / ")} · similarity distance ${pair.distance}${pair.cross_split ? " · crosses proposed splits" : " · same proposed split"}`));
      const images = node("div", "partition-similar-images");
      for (const id of pair.frame_ids) {
        const link = node("a");
        link.href = projectURL(`/api/frames/${encodeURIComponent(id)}/image`);
        link.target = "_blank";
        link.rel = "noopener";
        link.title = `Open original-sized frame ${id}`;
        const image = node("img");
        image.src = link.href;
        image.alt = `Similarity candidate ${id}`;
        image.loading = "lazy";
        link.append(image);
        images.append(link);
      }
      row.append(images);
      related.append(row);
    }
    if (plan.similar_pairs_truncated) related.append(node("p", "field-hint", `Showing ${plan.similar_pairs.length} of ${plan.similar_pairs_total} similar pairs. Review source groups before freezing.`));
    for (const duplicate of plan.exact_duplicates || []) related.append(node("p", "field-hint", `Identical pixels in ${duplicate.scene_groups.join(" / ")}. Frame IDs: ${duplicate.frame_ids.join(", ")}. Choose which frames to include in Data intake.`));
    $("#dataset-plan-status").textContent = `${plan.groups.length} scene groups · ${plan.summary.frame_count} eligible frames · seed ${plan.seed}. Whole groups and reservations can change the requested percentages.${plan.can_freeze ? " Review the preview, then apply it explicitly." : " Resolve the listed blockers before applying."}`;
    updatePartitionControls();
  }

  async function previewPartitions() {
    if ($("#dataset-plan-preview").disabled) return;
    let settings;
    try { settings = partitionSettings(); } catch (error) { showError("#dataset-plan-error", error); return; }
    invalidatePartitionPlan();
    const request = ++workspace.partitionRequest;
    workspace.partitionLoading = true;
    updatePartitionControls();
    try {
      const plan = await api("/api/datasets/plan", { method: "POST", body: JSON.stringify(settings) });
      if (request !== workspace.partitionRequest || settings.taxonomy_id !== workspace.taxonomyId) return;
      workspace.partitionPlan = plan;
      renderPartitionPlan(plan);
    } catch (error) {
      if (request === workspace.partitionRequest) showError("#dataset-plan-error", error);
    } finally {
      if (request === workspace.partitionRequest) { workspace.partitionLoading = false; updatePartitionControls(); }
    }
  }

  async function applyPartitions() {
    if ($("#dataset-plan-apply").disabled) return;
    const plan = workspace.partitionPlan;
    const request = ++workspace.partitionRequest;
    workspace.partitionApplying = true;
    updateDatasetLaunch();
    showError("#dataset-plan-error", null);
    try {
      const current = await api(`/api/dataset-candidates?taxonomy_id=${encodeURIComponent(plan.taxonomy_id)}`);
      if (request !== workspace.partitionRequest || plan !== workspace.partitionPlan) return;
      const choices = window.IRISPartitionTools.validatePlan(plan, current);
      workspace.candidates = current;
      workspace.candidateNeedsRefresh = false;
      workspace.choices = choices;
      workspace.partitionApplied = true;
      renderCandidates();
      $("#dataset-plan-status").textContent = "Proposed partitions applied to the menus below. Review them and use Freeze release when ready; no release was created.";
      $("#dataset-groups").scrollIntoView({ block: "nearest" });
    } catch (error) {
      if (request !== workspace.partitionRequest) return;
      workspace.partitionPlan = null;
      workspace.candidateNeedsRefresh = true;
      showError("#dataset-plan-error", error);
      $("#dataset-plan-status").textContent = "No split menus changed. Refresh candidates, review the latest labels and preview again.";
    } finally {
      if (request === workspace.partitionRequest) { workspace.partitionApplying = false; updateDatasetLaunch(); }
    }
  }

  function renderCandidateTaxonomy() {
    const select = $("#dataset-taxonomy");
    select.replaceChildren();
    for (const taxonomy of workspace.taxonomies)
      select.append(new Option(`${taxonomyTools.versionLabel(taxonomy)} · ${taxonomy.classes.map((item) => item.name).join(", ")}`, taxonomy.id));
    select.value = workspace.taxonomyId;
    select.title = select.selectedOptions[0]?.textContent || "Class version";
    const taxonomy = workspace.candidates.taxonomy;
    $("#dataset-taxonomy-context").textContent = `${taxonomyTools.versionLabel(taxonomy)} · ${taxonomy.classes.length} ${taxonomy.classes.length === 1 ? "class" : "classes"}. Names, definitions and export mappings are saved with the release.`;
    renderDefinitions("#dataset-taxonomy-definitions", taxonomy);
  }

  function datasetOptions() {
    const parent = $("#dataset-parent");
    const training = $("#training-dataset");
    const oldParent = parent.value;
    const oldTraining = training.value;
    const parents = datasetTools.compatibleParents(workspace.datasets, workspace.taxonomyId, state.projectId);
    const supported = workspace.datasets.filter(datasetTools.mlSupported);
    parent.replaceChildren(new Option("First / independent release", ""));
    training.replaceChildren();
    for (const dataset of workspace.datasets) {
      const label = `${dataset.name} · ${dataset.summary.frame_count} frames`;
      if (parents.includes(dataset)) parent.append(new Option(label, dataset.id));
      if (datasetTools.mlSupported(dataset)) training.append(new Option(label, dataset.id));
    }
    if (!supported.length)
      training.append(new Option(workspace.datasets.length ? "No compatible training release" : "Freeze a dataset first", ""));
    if (parents.some((item) => item.id === oldParent))
      parent.value = oldParent;
    if (supported.some((item) => item.id === oldTraining))
      training.value = oldTraining;
    training.disabled = !supported.length || workspace.trainingBusy;
    renderTrainingModels();
    if (training.value !== oldTraining) invalidateTrainingPreview();
    else updateTrainingLaunch();
  }

  function renderDatasetHistory() {
    const select = $("#dataset-history");
    select.replaceChildren();
    $("#dataset-history-count").textContent = String(workspace.datasets.length);
    $("#dataset-history-empty").hidden = Boolean(workspace.datasets.length);
    select.disabled = !workspace.datasets.length;
    if (!workspace.datasets.length) {
      select.append(new Option("No releases yet", ""));
      $("#dataset-detail").hidden = true;
    }
    for (const dataset of workspace.datasets)
      select.append(
        new Option(
          `${dataset.name} · ${new Date(dataset.created_at).toLocaleString()}`,
          dataset.id,
        ),
      );
    select.value = workspace.datasetId || "";
    datasetOptions();
  }

  async function refreshDatasets() {
    const request = ++workspace.datasetHistoryRequest;
    ++workspace.datasetRequest;
    resetDatasetExport();
    try {
      const datasets = await api("/api/datasets");
      if (request !== workspace.datasetHistoryRequest) return;
      workspace.datasets = datasets;
      if (!workspace.datasets.some((item) => item.id === workspace.datasetId))
        workspace.datasetId = workspace.datasets[0]?.id || null;
      renderDatasetHistory();
      showError("#dataset-history-error", null);
      if (workspace.datasetId) await loadDataset(workspace.datasetId);
      else ++workspace.datasetRequest;
    } catch (error) {
      if (request === workspace.datasetHistoryRequest)
        showError("#dataset-history-error", error);
    }
  }

  async function loadDataset(id) {
    const request = ++workspace.datasetRequest;
    resetDatasetExport();
    $("#dataset-detail").hidden = true;
    try {
      const detail = await api(`/api/datasets/${encodeURIComponent(id)}`);
      if (request !== workspace.datasetRequest || id !== workspace.datasetId)
        return;
      $("#dataset-detail").hidden = false;
      $("#dataset-detail-name").textContent = detail.name;
      const parent = workspace.datasets.find(
        (item) => item.id === detail.parent_id,
      );
      $("#dataset-detail-context").textContent =
        `${new Date(detail.created_at).toLocaleString()} · ${detail.summary.frame_count} frames · ${detail.summary.box_count} boxes · ${detail.summary.negative_count} validated negatives` +
        (detail.parent_id
          ? ` · Previous version: ${parent?.name || detail.parent_id}`
          : "");
      const counts = $("#dataset-detail-counts");
      counts.replaceChildren();
      for (const [split, label] of Object.entries(splitNames)) {
        const item = node("div");
        item.append(
          node("strong", "", String(detail.summary.split_counts[split] || 0)),
          node("span", "", `${label} frames`),
        );
        counts.append(item);
      }
      const taxonomy = detail.taxonomy || detail.manifest?.taxonomy || taxonomyTools.snapshot();
      const classes = datasetTools.classRows(detail, taxonomy);
      $("#dataset-detail-classes").textContent = `${classes.length} saved ${classes.length === 1 ? "class" : "classes"} · classes without boxes are retained in the export.`;
      $("#dataset-detail-taxonomy").textContent = `${taxonomyTools.versionLabel(taxonomy)} · ${taxonomy.id}`;
      const rows = $("#dataset-detail-class-rows");
      rows.replaceChildren();
      for (const category of classes) {
        const row = node("tr");
        const heading = node("th", "", `${category.name} (${category.id})`);
        heading.scope = "row";
        row.append(heading);
        for (const value of [category.count, category.class_id, category.export_id, category.source_coco_id])
          row.append(node("td", "", value == null ? "—" : String(value)));
        rows.append(row);
      }
      renderDefinitions("#dataset-detail-definitions", taxonomy);
      $("#dataset-detail-ml").textContent = datasetTools.mlSupported(detail)
        ? "Compatible with current training and evaluation."
        : detail.ml_limitation || "This release is available for inspection and export; the current runtime cannot train or evaluate it.";
      warnings("#dataset-detail-warnings", detail.summary.warnings);
      $("#dataset-manifest-download").href =
        projectURL(`/api/datasets/${encodeURIComponent(id)}/manifest`);
      $("#dataset-manifest").textContent = JSON.stringify(
        { manifest_sha256: detail.manifest_sha256, manifest: detail.manifest },
        null,
        2,
      );
      workspace.datasetDetail = detail;
      $("#dataset-coco-download").disabled = false;
      updateTrainingLaunch();
      showError("#dataset-history-error", null);
    } catch (error) {
      if (request === workspace.datasetRequest)
        showError("#dataset-history-error", error);
    }
  }

  function resetDatasetExport() {
    ++workspace.exportRequest;
    workspace.exportController?.abort();
    workspace.exportController = null;
    workspace.datasetDetail = null;
    $("#dataset-use-training").disabled = true;
    $("#dataset-coco-download").disabled = true;
    $("#dataset-coco-download").textContent = "Download COCO ZIP";
    $("#dataset-export-status").textContent = "";
    showError("#dataset-export-error", null);
  }

  function releaseExportUrl(url) {
    clearTimeout(workspace.exportUrls.get(url));
    workspace.exportUrls.delete(url);
    URL.revokeObjectURL(url);
  }

  async function downloadDatasetCoco() {
    const detail = workspace.datasetDetail;
    if (
      !detail ||
      detail.id !== workspace.datasetId ||
      workspace.exportController
    )
      return;
    const id = detail.id;
    const request = ++workspace.exportRequest;
    const controller = new AbortController();
    workspace.exportController = controller;
    const current = () =>
      request === workspace.exportRequest && id === workspace.datasetId;
    const button = $("#dataset-coco-download");
    button.disabled = true;
    button.textContent = "Preparing COCO ZIP…";
    $("#dataset-export-status").textContent =
      "Preparing a local ZIP from the frozen release…";
    showError("#dataset-export-error", null);
    try {
      let response;
      try {
        response = await fetch(
          projectURL(`/api/datasets/${encodeURIComponent(id)}/export/coco`),
          { signal: controller.signal, mode: "same-origin", redirect: "error" },
        );
      } catch (error) {
        if (error.name === "AbortError") throw error;
        throw new Error(
          "Cannot reach IRIS. Check that the local server is running and try again.",
        );
      }
      if (!current()) return;
      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(
          typeof error?.detail === "string"
            ? error.detail
            : `COCO export failed (HTTP ${response.status}). Try again.`,
        );
      }
      if (
        response.headers.get("content-type")?.split(";")[0].trim() !==
        "application/zip"
      )
        throw new Error("The server did not return a COCO ZIP. Try again.");
      const blob = await response.blob();
      if (!current()) return;
      if (!blob.size || blob.size > 256 * 1024 * 1024)
        throw new Error("The COCO ZIP is empty or exceeds the 256 MiB limit.");
      const url = URL.createObjectURL(blob);
      workspace.exportUrls.set(
        url,
        setTimeout(() => releaseExportUrl(url), 60_000),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = `iris-dataset-${id}-coco.zip`;
      document.body.append(link);
      link.click();
      link.remove();
      $("#dataset-export-status").textContent =
        `Download started for “${detail.name}”.`;
    } catch (error) {
      if (current() && error.name !== "AbortError") {
        $("#dataset-export-status").textContent = "";
        showError("#dataset-export-error", error);
      }
    } finally {
      if (current()) {
        workspace.exportController = null;
        button.disabled = false;
        button.textContent = "Download COCO ZIP";
      }
    }
  }

  function trainingPayload() {
    return {
      name: $("#training-name").value.trim(),
      dataset_id: $("#training-dataset").value,
      parent_model_id: $("#training-parent").value,
      steps: Number($("#training-steps").value),
      learning_rate: Number($("#training-rate").value),
      seed: Number($("#training-seed").value),
      scope: $("#training-scope").value,
      checkpoint_interval: Number($("#training-checkpoint-interval").value),
      device: $("#training-device").value,
    };
  }

  function invalidateTrainingPreview() {
    ++workspace.trainingPreviewRequest;
    workspace.trainingPreview = null;
    workspace.trainingPreviewKey = null;
    $("#training-preview-result").hidden = true;
    $("#training-preview-summary").textContent = "";
    $("#training-preview-counts").replaceChildren();
    $("#training-preview-notes").replaceChildren();
    updateTrainingLaunch();
  }

  function updateTrainingLaunch() {
    const dataset = workspace.datasets.find((item) => item.id === $("#training-dataset").value);
    const model = selectedTrainingModel();
    const selectedScope = modelScopes(model).find((scope) => scope.id === $("#training-scope").value);
    const compatibility = datasetTools.modelCompatibility(dataset, model, "training");
    const device = workspace.trainingDevices.find((item) => item.id === $("#training-device").value);
    const unavailable =
      workspace.trainingBusy ||
      workspace.trainingModelsLoading ||
      workspace.trainingDevicesLoading ||
      !device?.available ||
      !datasetTools.mlSupported(dataset) ||
      !compatibility.compatible ||
      model?.status !== "ready" ||
      !model.training ||
      !selectedScope;
    for (const field of $("#training-form").querySelectorAll("input, select"))
      field.disabled = workspace.trainingBusy;
    $("#training-device").disabled = workspace.trainingBusy || workspace.trainingDevicesLoading;
    $("#training-devices-refresh").disabled = workspace.trainingBusy || workspace.trainingDevicesLoading;
    $("#training-models-refresh").disabled = workspace.trainingBusy || workspace.trainingModelsLoading;
    $("#training-models-open").disabled = workspace.trainingBusy || !state.sessionId;
    $("#training-models-open").title = state.sessionId ? "Inspect local checkpoints and setup instructions" : "Choose a session to open the comparison model library";
    $("#dataset-use-training").disabled = workspace.trainingBusy ||
      !workspace.datasetDetail || workspace.datasetDetail.id !== workspace.datasetId ||
      !datasetTools.mlSupported(workspace.datasetDetail);
    $("#training-device-status").textContent = workspace.trainingDevicesLoading
      ? "Checking the local PyTorch runtime and visible devices…"
      : [workspace.trainingDevicesError, device?.reason,
        ...workspace.trainingDeviceNotes].filter(Boolean).join(" ") ||
        `${device?.label || recoveryTools.deviceLabel($("#training-device").value)} selected. The device for deployment is chosen separately when exporting a model.`;
    $("#training-dataset").disabled =
      workspace.trainingBusy || !workspace.datasets.some(datasetTools.mlSupported);
    $("#training-parent").disabled =
      workspace.trainingBusy ||
      workspace.trainingModelsLoading ||
      !workspace.models.some((item) => item.status === "ready" && datasetTools.modelCompatibility(dataset, item, "training").compatible);
    $("#training-scope").disabled = workspace.trainingBusy || !modelScopes(model).length;
    const taxonomy = datasetTools.taxonomyOf(dataset);
    $("#training-dataset-limitation").textContent = dataset
      ? `${taxonomyTools.versionLabel(taxonomy)} · ${taxonomy.classes.map((item) => item.name).join(", ")}. Trained parents must use these exact saved definitions.`
      : "Choose a frozen release to see compatible starting checkpoints.";
    $("#training-model-status").textContent = workspace.trainingModelsLoading
      ? "Checking local checkpoints and supported training depths…"
      : workspace.trainingModelsError || (model?.status === "ready"
        ? compatibility.reason
        : "A ready Faster R-CNN or SSDLite checkpoint and the optional PyTorch runtime are required. Check Model comparison for setup instructions.");
    $("#training-model-description").textContent = model?.training_summary || "";
    $("#training-preview").disabled = unavailable;
    $("#training-readiness").textContent = workspace.trainingBusy
      ? workspace.trainingOperation === "start" ? "Queuing this reviewed plan…" : "Checking the frozen release, checkpoint and settings…"
      : workspace.trainingModelsLoading || workspace.trainingDevicesLoading
        ? "Checking local models and compute availability…"
        : !datasetTools.mlSupported(dataset)
          ? "Choose a compatible frozen release in Datasets to prepare a training plan."
          : workspace.trainingModelsError || !model || !compatibility.compatible || model.status !== "ready" || !model.training || !selectedScope
            ? "A compatible, ready starting checkpoint is required. Review model availability beside the selector."
            : !device?.available
              ? "The selected training device is unavailable. Choose an available device or refresh availability."
              : workspace.trainingPreview && workspace.trainingPreviewKey === JSON.stringify(trainingPayload())
                ? "Plan checked. Review the details below, then start training when ready."
                : !$("#training-name").value.trim()
                  ? "Name this run, review its settings, then preview the plan."
                  : "Preview this plan to check the dataset, checkpoint and settings before starting.";
    $("#training-preview").textContent =
      workspace.trainingOperation === "preview"
        ? "Checking training plan…"
        : "Preview training plan";
    $("#training-start").disabled =
      unavailable ||
      !workspace.trainingPreview ||
      workspace.trainingPreviewKey !== JSON.stringify(trainingPayload());
    $("#training-start").textContent =
      workspace.trainingOperation === "start"
        ? "Queuing training…"
        : `Start ${recoveryTools.deviceLabel($("#training-device").value)} training →`;
    $("#training-scope-description").textContent =
      selectedScope?.description || "Choose a training depth.";
    $("#training-scope-cost").textContent = selectedScope?.cost || trainingScopes[selectedScope?.id]?.cost || "";
    $("#training-scope-modules").textContent = selectedScope?.trainable_modules?.length
      ? `Layers to update: ${selectedScope.trainable_modules.join(", ")}.`
      : "";
    const steps = Number($("#training-steps").value);
    const interval = $("#training-checkpoint-interval");
    const minimum = recoveryTools.checkpointMinimum(steps);
    interval.min = String(minimum);
    $("#training-checkpoint-hint").textContent =
      `Minimum ${minimum} step(s) for this plan, up to 200 periodic saves. Keep the latest two recovery states, up to 512 MiB each. ` +
      (Number(interval.value) > steps
        ? "This short plan ends before its first periodic save; reduce the interval to save progress during the run."
        : "After cancellation or interruption, continue explicitly from the latest complete state; later unsaved steps may be repeated.");
  }

  function renderTrainingPreview(preview, payload) {
    if (!recoveryTools.previewMatches(preview, payload))
      throw new Error(
        "The training preview does not match these settings. Preview the plan again.",
      );
    workspace.trainingPreview = preview;
    workspace.trainingPreviewKey = JSON.stringify(payload);
    $("#training-preview-result").hidden = false;
    $("#training-preview-scope").textContent = preview.scope.label || scopeLabel(preview.scope.id);
    $("#training-preview-summary").textContent =
      `${preview.dataset.name} · starting from ${preview.parent.name} · learning rate ${payload.learning_rate} · seed ${payload.seed}`;
    const counts = $("#training-preview-counts");
    counts.replaceChildren();
    for (const [label, count] of [
      ["Training images", preview.dataset.train_images],
      ["Image visits / optimizer steps", preview.workload.image_visits],
      [
        "Distinct images reached by the plan",
        preview.workload.unique_images_min,
      ],
    ]) {
      const item = node("div");
      item.append(node("strong", "", String(count)), node("span", "", label));
      counts.append(item);
    }
    $("#training-preview-workload").textContent = [
      `${recoveryTools.deviceLabel(preview.workload.device)} · batch size ${preview.workload.batch_size} · ${preview.workload.steps} optimizer steps maximum`,
      `${preview.workload.full_passes} complete pass(es) through the training images + ${preview.workload.remainder_images} additional image visits.`,
      `${preview.dataset.positive_train_images} training images contain target boxes; ${preview.dataset.annotation_count} training annotations.`,
    ].join(" · ");
    $("#training-preview-modules").textContent =
      `Layers to update: ${(preview.scope.trainable_modules || []).join(", ")}. Parameter counts are recorded when the training model is loaded.`;
    $("#training-preview-recovery").textContent =
      `Save recovery state every ${payload.checkpoint_interval} steps; retain the latest two states. ` +
      "Continuation uses the same frozen dataset, settings, optimizer and random state. No duration estimate is available before observing this run.";
    warnings("#training-preview-notes", [...(preview.notes || []), ...(preview.device_notes || [])]);
    updateTrainingLaunch();
  }

  async function previewTraining(event) {
    event.preventDefault();
    if (
      $("#training-preview").disabled ||
      !$("#training-form").reportValidity()
    )
      return;
    const payload = trainingPayload();
    if (!payload.name) return $("#training-name").focus();
    invalidateTrainingPreview();
    const request = workspace.trainingPreviewRequest;
    workspace.trainingBusy = true;
    workspace.trainingOperation = "preview";
    updateTrainingLaunch();
    showError("#training-error", null);
    try {
      const preview = await api("/api/trainings/preview", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (
        request !== workspace.trainingPreviewRequest ||
        !workspace.visible ||
        JSON.stringify(payload) !== JSON.stringify(trainingPayload())
      )
        return;
      renderTrainingPreview(preview, payload);
    } catch (error) {
      if (request === workspace.trainingPreviewRequest && workspace.visible)
        showError("#training-error", error);
    } finally {
      workspace.trainingBusy = false;
      workspace.trainingOperation = null;
      updateTrainingLaunch();
    }
  }

  function renderTrainingModels() {
    const select = $("#training-parent");
    const previous = select.value;
    const dataset = workspace.datasets.find((item) => item.id === $("#training-dataset").value);
    const eligible = workspace.models.filter((model) => model.training);
    select.replaceChildren();
    for (const model of eligible) {
      const compatibility = datasetTools.modelCompatibility(dataset, model, "training");
      const option = new Option(`${model.name}${!compatibility.compatible ? " · incompatible classes" : model.status !== "ready" ? " · setup required" : ""}`, model.id);
      option.disabled = model.status !== "ready" || !compatibility.compatible;
      option.title = compatibility.reason;
      select.append(option);
    }
    const ready = eligible.filter((model) => model.status === "ready" && datasetTools.modelCompatibility(dataset, model, "training").compatible);
    if (!ready.length) select.prepend(new Option("No compatible ready checkpoint", ""));
    const defaultModel = ready.find((model) => model.id === "fasterrcnn_mobilenet_v3_large_320_fpn") || ready[0];
    select.value = ready.some((model) => model.id === previous) ? previous : defaultModel?.id || "";
    const scopeChanged = renderTrainingScopes();
    if (select.value !== previous || scopeChanged) invalidateTrainingPreview();
    else updateTrainingLaunch();
  }

  async function refreshTrainingModels() {
    const request = ++workspace.trainingModelsRequest;
    const select = $("#training-parent");
    workspace.trainingModelsLoading = true;
    workspace.trainingModelsError = null;
    invalidateTrainingPreview();
    try {
      const models = await api("/api/models");
      if (request !== workspace.trainingModelsRequest) return;
      workspace.models = models;
      renderTrainingModels();
    } catch (error) {
      if (request !== workspace.trainingModelsRequest) return;
      workspace.models = [];
      workspace.trainingModelsError = error.message;
      select.replaceChildren(new Option("Model availability unavailable", ""));
      renderTrainingScopes();
      select.disabled = true;
      $("#training-model-status").textContent = error.message;
    } finally {
      if (request === workspace.trainingModelsRequest) {
        workspace.trainingModelsLoading = false;
        updateTrainingLaunch();
      }
    }
  }

  async function refreshTrainingDevices() {
    const request = ++workspace.trainingDevicesRequest;
    const select = $("#training-device");
    const previous = select.value || "cpu";
    workspace.trainingDevicesLoading = true;
    workspace.trainingDevicesError = null;
    invalidateTrainingPreview();
    let report;
    try {
      report = await api("/api/training/devices");
    } catch (error) {
      if (request !== workspace.trainingDevicesRequest) return;
      workspace.trainingDevicesError = `Device availability could not be checked: ${error.message}`;
    }
    if (request !== workspace.trainingDevicesRequest) return;
    workspace.trainingDevices = recoveryTools.deviceChoices(report, previous);
    workspace.trainingDeviceNotes = [...new Set([
      ...(report?.notes || []),
      ...workspace.trainingDevices.filter((device) => !device.available && device.id !== previous).map((device) => device.reason).filter(Boolean),
    ])];
    select.replaceChildren();
    for (const device of workspace.trainingDevices) {
      const option = new Option(`${device.label || recoveryTools.deviceLabel(device.id)}${device.available ? "" : " · unavailable"}`, device.id);
      option.disabled = !device.available;
      option.title = device.reason || "";
      select.append(option);
    }
    select.value = previous;
    workspace.trainingDevicesLoading = false;
    updateTrainingLaunch();
  }

  function renderTrainingHistory() {
    const select = $("#training-history");
    select.replaceChildren();
    $("#training-history-count").textContent = String(
      workspace.trainings.length,
    );
    $("#training-history-empty").hidden = Boolean(workspace.trainings.length);
    select.disabled = !workspace.trainings.length;
    if (!workspace.trainings.length) {
      select.append(new Option("No training runs yet", ""));
      $("#training-detail").hidden = true;
    }
    for (const run of workspace.trainings)
      select.append(
        new Option(
          `${run.name} · ${run.job?.status || "Unknown status"} · ${new Date(run.created_at).toLocaleString()}`,
          run.id,
        ),
      );
    select.value = workspace.trainingId || "";
  }

  async function refreshTrainings(event) {
    const request = ++workspace.historyRequest;
    const announce = event?.type === "click" ||
      $("#training-history-status").textContent === "Refreshing run history…" ||
      !$("#training-history-error").hidden;
    $("#training-history-refresh").disabled = true;
    $("#training-history").setAttribute("aria-busy", "true");
    if (announce) $("#training-history-status").textContent = "Refreshing run history…";
    try {
      const trainings = await api("/api/trainings");
      if (request !== workspace.historyRequest) return;
      workspace.trainings = trainings;
      if (!trainings.some((item) => item.id === workspace.trainingId))
        workspace.trainingId = trainings[0]?.id || null;
      renderTrainingHistory();
      showError("#training-history-error", null);
      if (workspace.trainingId) await loadTraining(workspace.trainingId);
      if (request === workspace.historyRequest && announce)
        $("#training-history-status").textContent = $("#training-history-error").hidden
          ? "Run history refreshed."
          : "The run list loaded, but the selected run could not be read. Try Refresh runs again.";
    } catch (error) {
      if (request === workspace.historyRequest) {
        showError("#training-history-error", error);
        $("#training-history-status").textContent = "Run history could not be refreshed. Try Refresh runs again.";
      }
    } finally {
      if (request === workspace.historyRequest) {
        $("#training-history-refresh").disabled = false;
        $("#training-history").setAttribute("aria-busy", "false");
      }
    }
  }

  function renderLossHistory(history) {
    const container = $("#training-loss-history");
    container.replaceChildren();
    const page = recoveryTools.lossPage(history, workspace.lossPage);
    workspace.lossPage = page.page;
    $("#training-loss-count").textContent = page.total
      ? `${page.total.toLocaleString()} recorded steps`
      : "No recorded steps";
    $("#training-loss-page").textContent = page.total
      ? `Showing ${page.start + 1}–${page.end} of ${page.total} recorded steps. Full history remains saved with the run.`
      : "";
    $("#training-loss-earlier").disabled = page.page >= page.pages - 1;
    $("#training-loss-later").disabled = page.page === 0;
    if (!history.length) {
      container.append(
        node(
          "p",
          "field-hint",
          "Loss values will appear after the first optimizer step.",
        ),
      );
      return;
    }
    const table = node("table");
    table.append(node("caption", "sr-only", "Training loss by optimizer step"));
    const head = node("thead");
    const headings = node("tr");
    for (const label of ["Step", "Training loss", "Active time"]) {
      const cell = node("th", "", label);
      cell.scope = "col";
      headings.append(cell);
    }
    head.append(headings);
    const body = node("tbody");
    for (const item of page.items) {
      const row = node("tr");
      row.append(
        node("td", "", String(item.step)),
        node(
          "td",
          "",
          Number.isFinite(item.loss) ? item.loss.toFixed(5) : "Unavailable",
        ),
        node(
          "td",
          "",
          Number.isFinite(item.elapsed_seconds)
            ? `${item.elapsed_seconds.toFixed(1)} s`
            : "Unavailable",
        ),
      );
      body.append(row);
    }
    table.append(head, body);
    container.append(table);
  }

  function invalidateResumePreview() {
    ++workspace.resumeRequest;
    workspace.resumePreview = null;
    $("#training-resume-plan").hidden = true;
    $("#training-resume-start").disabled = true;
  }

  function selectTraining(id) {
    invalidateResumePreview();
    workspace.trainingId = id;
    workspace.trainingDetail = null;
    workspace.lossPage = 0;
    $("#training-history").value = id;
    $("#training-detail").hidden = true;
    showError("#training-resume-error", null);
    if (id) loadTraining(id);
  }

  function renderRecovery(detail) {
    const recovery = detail.recovery || {};
    $("#training-recovery-status").textContent = recovery.reason ||
      (recovery.supported ? "Recovery availability is being checked." : "This run has no saved recovery state. Start a new plan to enable periodic saves.");
    const lineage = $("#training-recovery-lineage");
    lineage.replaceChildren();
    if (detail.config.resume_from) {
      lineage.append(document.createTextNode(`This attempt continues from step ${detail.config.resume_from.step}. The earlier attempt and its outcome remain saved. `));
      const source = node("button", "text-button", "Open earlier attempt");
      source.type = "button";
      source.addEventListener("click", () => selectTraining(detail.config.resume_from.training_id));
      lineage.append(source);
    }
    const snapshots = $("#training-recovery-checkpoints");
    snapshots.replaceChildren();
    for (const checkpoint of [...(detail.checkpoints || [])].sort((a, b) => b.step - a.step).slice(0, 2)) {
      const size = Number.isFinite(checkpoint.size_bytes) ? ` · ${(checkpoint.size_bytes / 1024 / 1024).toFixed(1)} MiB` : "";
      const saved = node("div", "training-recovery-state");
      saved.append(
        node("strong", "", `Saved step ${checkpoint.step.toLocaleString()}`),
        node("span", "field-hint", `${new Date(checkpoint.created_at).toLocaleString()}${size}`),
      );
      snapshots.append(saved);
    }
    const existing = recovery.existing_training_id || recoveryTools.findResume(workspace.trainings, detail.id)?.id;
    $("#training-resume-existing").hidden = !existing;
    $("#training-resume-existing").disabled = workspace.resumeBusy;
    $("#training-resume-existing").dataset.trainingId = existing || "";
    $("#training-resume-preview").disabled = workspace.resumeBusy || !recovery.can_resume || Boolean(existing);
    $("#training-resume-preview").textContent = workspace.resumeOperation === "preview" ? "Checking saved progress…" : "Preview continuation";
    $("#training-resume-start").disabled = workspace.resumeBusy || !recoveryTools.resumeMatches(workspace.resumePreview, detail);
    $("#training-resume-start").textContent = workspace.resumeOperation === "start" ? "Queuing continuation…" : `Continue ${recoveryTools.deviceLabel(detail.config.device)} training →`;
  }

  async function previewResume() {
    if ($("#training-resume-preview").disabled) return;
    const detail = workspace.trainingDetail;
    if (!detail?.recovery?.can_resume) return;
    invalidateResumePreview();
    const request = workspace.resumeRequest;
    workspace.resumeBusy = true;
    workspace.resumeOperation = "preview";
    renderRecovery(detail);
    showError("#training-resume-error", null);
    try {
      const preview = await api(`/api/trainings/${encodeURIComponent(detail.id)}/resume-preview`, { method: "POST", body: "{}" });
      if (request !== workspace.resumeRequest || workspace.trainingId !== detail.id || !workspace.visible) return;
      if (!recoveryTools.resumeMatches(preview, workspace.trainingDetail))
        throw new Error("Saved progress changed. Refresh the run and preview its continuation again.");
      workspace.resumePreview = preview;
      $("#training-resume-plan").hidden = false;
      $("#training-resume-summary").textContent =
        `Continue from saved step ${preview.checkpoint_step} to the original target of ${preview.target_steps} steps (${preview.remaining_steps} remaining). ` +
        (preview.remaining_steps === 0 ? "Optimization is complete; this attempt will finish publishing the model checkpoint. " : "") +
        `${preview.recorded_steps} steps were recorded in the earlier attempt; ${preview.recomputed_steps} recorded step(s) after this state will be repeated. ` +
        `The dataset, training depth, learning rate, seed and ${recoveryTools.deviceLabel(detail.config.device)} device stay fixed. This creates a new attempt and preserves the earlier one.`;
      warnings("#training-resume-warnings", preview.warnings);
    } catch (error) {
      if (request === workspace.resumeRequest && workspace.trainingId === detail.id && workspace.visible)
        showError("#training-resume-error", error);
    } finally {
      workspace.resumeBusy = false;
      workspace.resumeOperation = null;
      if (workspace.trainingDetail) renderRecovery(workspace.trainingDetail);
    }
  }

  async function startResume() {
    const detail = workspace.trainingDetail;
    const approved = workspace.resumePreview;
    if (workspace.resumeBusy || !recoveryTools.resumeMatches(approved, detail)) return;
    const request = workspace.resumeRequest;
    workspace.resumeBusy = true;
    workspace.resumeOperation = "start";
    renderRecovery(detail);
    showError("#training-resume-error", null);
    let saved;
    try {
      try {
        saved = await api(`/api/trainings/${encodeURIComponent(detail.id)}/resume`, {
          method: "POST", body: JSON.stringify({ expected_fingerprint: approved.fingerprint }),
        });
      } catch (error) {
        try { saved = recoveryTools.findResume(await api("/api/trainings"), detail.id); }
        catch { /* Retain the original request failure if history is unavailable. */ }
        if (!saved) throw error;
      }
      if (request !== workspace.resumeRequest || workspace.trainingId !== detail.id || !workspace.visible) return;
      invalidateResumePreview();
      workspace.trainingId = saved.id;
      workspace.lossPage = 0;
      await refreshTrainings();
      await refreshJobs();
      window.IRISTrainingNavigation?.open("runs", { focus: true });
      notify(`Continuation queued from saved step ${approved.checkpoint_step}. Follow progress or cancel in Processing jobs.`);
    } catch (error) {
      if (request === workspace.resumeRequest && workspace.trainingId === detail.id && workspace.visible) {
        invalidateResumePreview();
        showError("#training-resume-error", new Error(`${error.message} Check run history before starting another continuation.`));
      }
    } finally {
      workspace.resumeBusy = false;
      workspace.resumeOperation = null;
      if (workspace.trainingDetail) renderRecovery(workspace.trainingDetail);
    }
  }

  async function loadTraining(id) {
    const request = ++workspace.trainingRequest;
    try {
      const detail = await api(`/api/trainings/${encodeURIComponent(id)}`);
      if (request !== workspace.trainingRequest || workspace.trainingId !== id)
        return;
      if (recoveryTools.recoveryKey(workspace.trainingDetail) !== recoveryTools.recoveryKey(detail))
        invalidateResumePreview();
      workspace.trainingDetail = detail;
      $("#training-detail").hidden = false;
      $("#training-detail-name").textContent = detail.name;
      const dataset = workspace.datasets.find(
        (item) => item.id === detail.dataset_id,
      );
      $("#training-detail-context").textContent =
        `${dataset?.name || detail.dataset_id} · ${scopeLabel(detail.config.scope, detail.parent_model_id)} · ${recoveryTools.deviceLabel(detail.config.device)}`;
      const trained = detail.metadata?.trainable_parameters;
      const total = detail.metadata?.total_parameters;
      const modules =
        detail.metadata?.trainable_modules ||
        detail.config.trainable_modules ||
        modelScopes(workspace.models.find((item) => item.id === detail.parent_model_id))
          .find((scope) => scope.id === (detail.config.scope || "prediction_head_only"))?.trainable_modules || [];
      const counts =
        Number.isFinite(trained) && Number.isFinite(total)
          ? `${trained.toLocaleString()} of ${total.toLocaleString()} parameters trainable. `
          : "Parameter counts become available when the training model is loaded. ";
      $("#training-detail-scope").textContent =
        counts +
        (modules.length ? `Trainable layers: ${modules.join(", ")}.` : "");
      const badge = $("#training-detail-status");
      badge.textContent = detail.job?.status || "Unknown status";
      badge.className = `job-status ${detail.job?.status || ""}`;
      $("#training-detail-job").hidden = !detail.job?.id;
      $("#training-detail-message").textContent = detail.job?.message || "";
      showError(
        "#training-detail-error",
        detail.job?.error ? new Error(detail.job.error) : null,
      );
      $("#training-checkpoint").hidden = !detail.checkpoint_id;
      $("#training-checkpoint-id").textContent = detail.checkpoint_id || "";
      const latest = detail.history.at(-1);
      const completed = latest?.step || 0;
      const target = detail.config.steps;
      $("#training-detail-steps").textContent = `${completed.toLocaleString()} / ${target.toLocaleString()}`;
      const progress = $("#training-detail-progress");
      progress.max = Math.max(1, target);
      progress.value = Math.min(target, Math.max(0, completed));
      progress.setAttribute("aria-valuetext", `${completed} of ${target} optimizer steps recorded`);
      $("#training-detail-loss").textContent = Number.isFinite(latest?.loss) ? latest.loss.toFixed(5) : "Not recorded";
      const duration = recoveryTools.observedDuration(detail.history, detail.config.steps);
      const elapsed = latest?.elapsed_seconds;
      $("#training-detail-active-time").textContent = Number.isFinite(elapsed) ? recoveryTools.durationText(elapsed) : "Not recorded";
      const active = detail.job?.status === "running";
      $("#training-detail-duration").textContent =
        (Number.isFinite(elapsed) ? "Recorded active time excludes downtime. " : "") +
        (active && duration
          ? `About ${recoveryTools.durationText(duration.remainingSeconds)} remaining, based on the latest ${duration.observedSteps} observed steps. Saving and completion can add time; this estimate may change.`
          : active ? "A duration estimate appears after enough steps have been observed." : "");
      renderLossHistory(detail.history);
      renderRecovery(detail);
      $("#training-provenance").textContent = JSON.stringify(
        {
          dataset_id: detail.dataset_id,
          parent_model_id: detail.parent_model_id,
          config: detail.config,
          metadata: detail.metadata,
          checkpoint_id: detail.checkpoint_id,
          recovery: detail.recovery,
          recovery_checkpoints: detail.checkpoints,
        },
        null,
        2,
      );
      showError("#training-history-error", null);
    } catch (error) {
      if (request === workspace.trainingRequest)
        showError("#training-history-error", error);
    }
  }

  $("#dataset-refresh").addEventListener("click", refreshCandidates);
  $("#dataset-plan-preview").addEventListener("click", previewPartitions);
  $("#dataset-plan-apply").addEventListener("click", applyPartitions);
  for (const input of $(".partition-settings").querySelectorAll("input")) input.addEventListener("input", () => invalidatePartitionPlan("Settings changed. Preview again to see their effect."));
  $("#dataset-coco-download").addEventListener("click", downloadDatasetCoco);
  $("#dataset-taxonomy").addEventListener("change", (event) => {
    if (workspace.datasetBusy || workspace.candidateLoading || workspace.partitionApplying) return;
    workspace.taxonomyId = event.target.value;
    workspace.choices.clear();
    workspace.candidates = null;
    $("#dataset-parent").value = "";
    $("#dataset-groups").replaceChildren();
    $("#dataset-candidate-status").textContent = "Loading reviewed images for this class version…";
    $("#dataset-taxonomy-definitions").replaceChildren();
    $("#dataset-taxonomy-context").textContent = "Loading class definitions…";
    $("#dataset-taxonomy-excluded").textContent = "";
    datasetOptions();
    refreshCandidates();
  });
  $("#dataset-history").addEventListener("change", (event) => {
    workspace.datasetId = event.target.value;
    loadDataset(workspace.datasetId);
  });
  $("#training-history-refresh").addEventListener("click", refreshTrainings);
  $("#training-detail-job").addEventListener("click", () => {
    const id = workspace.trainingDetail?.job?.id;
    if (id) window.dispatchEvent(new CustomEvent("iris:job-open", { detail: { job_id: id } }));
  });
  $("#training-history").addEventListener("change", (event) => {
    selectTraining(event.target.value);
  });
  $("#training-resume-preview").addEventListener("click", previewResume);
  $("#training-resume-start").addEventListener("click", startResume);
  $("#training-resume-existing").addEventListener("click", (event) => {
    const id = event.currentTarget.dataset.trainingId;
    if (id) selectTraining(id);
  });
  $("#training-loss-earlier").addEventListener("click", () => {
    workspace.lossPage += 1;
    renderLossHistory(workspace.trainingDetail?.history || []);
  });
  $("#training-loss-later").addEventListener("click", () => {
    workspace.lossPage -= 1;
    renderLossHistory(workspace.trainingDetail?.history || []);
  });
  $("#training-form").addEventListener("input", invalidateTrainingPreview);
  $("#dataset-use-training").addEventListener("click", () => {
    if ($("#dataset-use-training").disabled) return;
    $("#training-dataset").value = workspace.datasetDetail.id;
    renderTrainingModels();
    invalidateTrainingPreview();
    window.IRISTrainingNavigation.open("plan", { focus: true });
  });
  $("#training-models-refresh").addEventListener("click", refreshTrainingModels);
  $("#training-models-open").addEventListener("click", () => {
    if (window.IRISNavigation.open("comparison")) $("#main").focus();
  });
  $("#training-dataset").addEventListener("change", renderTrainingModels);
  $("#training-parent").addEventListener("change", renderTrainingScopes);
  $("#training-form").addEventListener("change", invalidateTrainingPreview);
  $("#dataset-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if ($("#dataset-create").disabled) return;
    const groups = chosenGroups();
    let expectedRevisions;
    try { expectedRevisions = datasetTools.revisionTokens(groups); }
    catch (error) { showError("#dataset-error", error); return; }
    const payload = {
      name: $("#dataset-name").value.trim(),
      frame_ids: groups.flatMap((group) =>
        group.frames.map((frame) => frame.id),
      ),
      splits: Object.fromEntries(
        groups.map((group) => [
          group.scene_group,
          workspace.choices.get(group.scene_group),
        ]),
      ),
      parent_id: $("#dataset-parent").value || null,
      taxonomy_id: workspace.taxonomyId,
      expected_revisions: expectedRevisions,
    };
    if (!payload.name) return $("#dataset-name").focus();
    workspace.datasetBusy = true;
    renderCandidates();
    showError("#dataset-error", null);
    try {
      const dataset = await api("/api/datasets", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      workspace.datasetId = dataset.id;
      $("#dataset-name").value = "";
      await Promise.all([refreshDatasets(), refreshCandidates()]);
      if (datasetTools.mlSupported(dataset)) $("#training-dataset").value = dataset.id;
      $("#dataset-parent").value = dataset.id;
      renderTrainingModels();
      invalidateTrainingPreview();
      notify(
        `Dataset release “${dataset.name}” saved with frozen labels and split assignments.`,
      );
    } catch (error) {
      if (error.status === 409) {
        workspace.candidateNeedsRefresh = true;
        error.message += " Refresh candidates and review the updated selection before freezing again.";
      }
      showError("#dataset-error", error);
    } finally {
      workspace.datasetBusy = false;
      renderCandidates();
    }
  });
  $("#training-form").addEventListener("submit", previewTraining);
  $("#training-start").addEventListener("click", async () => {
    updateTrainingLaunch();
    if (
      $("#training-start").disabled ||
      !$("#training-form").reportValidity()
    )
      return;
    const payload = trainingPayload();
    if (!payload.name) return $("#training-name").focus();
    const approved = workspace.trainingPreview;
    const request = workspace.trainingPreviewRequest;
    workspace.trainingBusy = true;
    workspace.trainingOperation = "start";
    updateTrainingLaunch();
    showError("#training-error", null);
    try {
      let detail;
      try {
        detail = await api("/api/trainings", {
          method: "POST",
          body: JSON.stringify({ ...payload, request_id: approved.request_id, expected_fingerprint: approved.fingerprint }),
        });
      } catch (error) {
        try { detail = recoveryTools.findRequest(await api("/api/trainings"), approved.request_id); }
        catch { /* Retain the original request failure if history is unavailable. */ }
        if (!detail) throw error;
      }
      if (request !== workspace.trainingPreviewRequest || !workspace.visible) return;
      workspace.trainingId = detail.id;
      workspace.lossPage = 0;
      $("#training-name").value = "";
      invalidateTrainingPreview();
      await refreshTrainings();
      await refreshJobs();
      notify(
        `Training “${detail.name}” queued for ${payload.steps} ${recoveryTools.deviceLabel(payload.device)} steps. Follow progress or cancel in Processing jobs.`,
      );
      window.IRISTrainingNavigation.open("runs", { focus: true });
    } catch (error) {
      if (request === workspace.trainingPreviewRequest && workspace.visible) {
        invalidateTrainingPreview();
        showError("#training-error", new Error(`${error.message} Check run history before starting another plan.`));
      }
    } finally {
      workspace.trainingBusy = false;
      workspace.trainingOperation = null;
      updateTrainingLaunch();
    }
  });
  $("#training-compare").addEventListener("click", () => {
    const detail = workspace.trainingDetail;
    if (!detail?.checkpoint_id) return;
    window.dispatchEvent(
      new CustomEvent("iris:models", {
        detail: {
          model_ids: [detail.parent_model_id, detail.checkpoint_id],
          openComparison: true,
        },
      }),
    );
    notify(
      "Parent and trained checkpoint selected. Choose frames in the active session and run a comparison to inspect their outputs.",
    );
  });
  window.addEventListener("iris:workspace", (event) => {
    workspace.visible = event.detail.name === "training";
    if (workspace.visible) {
      refreshCandidates();
      refreshDatasets();
      refreshTrainingModels();
      refreshTrainingDevices();
      refreshTrainings();
    } else {
      invalidatePartitionPlan();
      resetDatasetExport();
      invalidateTrainingPreview();
      invalidateResumePreview();
    }
  });
  $("#training-devices-refresh").addEventListener("click", refreshTrainingDevices);
  window.addEventListener("iris:taxonomy", (event) => {
    const taxonomy = event.detail.taxonomy;
    if (!workspace.candidates || workspace.taxonomies.some((item) => item.id === taxonomy.id)) return;
    workspace.taxonomies.push(taxonomy);
    renderCandidateTaxonomy();
    updateDatasetLaunch();
  });
  window.addEventListener("pagehide", () => {
    ++workspace.datasetRequest;
    invalidateTrainingPreview();
    invalidateResumePreview();
    resetDatasetExport();
    for (const url of workspace.exportUrls.keys()) releaseExportUrl(url);
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted && workspace.visible) refreshDatasets();
  });
  window.addEventListener("iris:jobs", () => {
    let trainingChanged = false;
    let newCheckpoint = false;
    for (const job of state.jobs) {
      if (job.kind !== "train") continue;
      const previous = workspace.jobStatuses.get(job.id);
      workspace.jobStatuses.set(job.id, job.status);
      if (previous !== job.status || isActive(job)) trainingChanged = true;
      if (previous !== job.status && job.status === "succeeded")
        newCheckpoint = true;
    }
    if (trainingChanged && workspace.visible) refreshTrainings();
    if (newCheckpoint) {
      window.dispatchEvent(new Event("iris:models"));
      if (workspace.visible) refreshTrainingModels();
    }
  });
})();
