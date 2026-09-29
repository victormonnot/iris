"use strict";

(() => {
  const workspace = {
    visible: false,
    candidates: null,
    choices: new Map(),
    candidateRequest: 0,
    candidateLoading: false,
    datasetBusy: false,
    datasets: [],
    datasetId: null,
    datasetRequest: 0,
    models: [],
    trainings: [],
    trainingId: null,
    trainingDetail: null,
    trainingRequest: 0,
    trainingBusy: false,
    historyRequest: 0,
    jobStatuses: new Map(),
  };
  const splitNames = { train: "Train", val: "Validation", test: "Test" };

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
    $("#dataset-create").disabled =
      workspace.datasetBusy ||
      workspace.candidateLoading ||
      !counts.train ||
      !counts.val;
    $("#dataset-refresh").disabled =
      workspace.datasetBusy || workspace.candidateLoading;
    $("#dataset-create").textContent = workspace.datasetBusy
      ? "Freezing release…"
      : "Freeze release →";
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
      .map(([key, value]) => `${value} ${key.replaceAll("_", " ")}`);
    $("#dataset-candidate-status").textContent =
      `${count} eligible frames in ${groups.length} scene groups.` +
      (exclusions.length ? ` Excluded: ${exclusions.join(" · ")}.` : "");
    if (!groups.length)
      container.append(
        node(
          "p",
          "dataset-empty",
          "No eligible frames yet. Select frames in Data intake, resolve every proposal and save a human validation in Annotation.",
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
      select.disabled = workspace.datasetBusy;
      select.addEventListener("change", () => {
        workspace.choices.set(group.scene_group, select.value);
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
    const request = ++workspace.candidateRequest;
    workspace.candidateLoading = true;
    updateDatasetLaunch();
    showError("#dataset-error", null);
    try {
      const candidates = await api("/api/dataset-candidates");
      if (request !== workspace.candidateRequest) return;
      workspace.candidates = candidates;
      renderCandidates();
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
        updateDatasetLaunch();
      }
    }
  }

  function datasetOptions() {
    const parent = $("#dataset-parent");
    const training = $("#training-dataset");
    const oldParent = parent.value;
    const oldTraining = training.value;
    parent.replaceChildren(new Option("First / independent release", ""));
    training.replaceChildren();
    for (const dataset of workspace.datasets) {
      const label = `${dataset.name} · ${dataset.summary.frame_count} frames`;
      parent.append(new Option(label, dataset.id));
      training.append(new Option(label, dataset.id));
    }
    if (!workspace.datasets.length)
      training.append(new Option("Freeze a dataset first", ""));
    if (workspace.datasets.some((item) => item.id === oldParent))
      parent.value = oldParent;
    if (workspace.datasets.some((item) => item.id === oldTraining))
      training.value = oldTraining;
    training.disabled = !workspace.datasets.length || workspace.trainingBusy;
    updateTrainingLaunch();
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
    try {
      workspace.datasets = await api("/api/datasets");
      if (!workspace.datasets.some((item) => item.id === workspace.datasetId))
        workspace.datasetId = workspace.datasets[0]?.id || null;
      renderDatasetHistory();
      showError("#dataset-history-error", null);
      if (workspace.datasetId) await loadDataset(workspace.datasetId);
    } catch (error) {
      showError("#dataset-history-error", error);
    }
  }

  async function loadDataset(id) {
    const request = ++workspace.datasetRequest;
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
      $("#dataset-detail-classes").textContent = Object.entries(
        detail.summary.class_counts || {},
      )
        .map(([label, count]) => `${label}: ${count} boxes`)
        .join(" · ");
      warnings("#dataset-detail-warnings", detail.summary.warnings);
      $("#dataset-manifest-download").href =
        `/api/datasets/${encodeURIComponent(id)}/manifest`;
      $("#dataset-manifest").textContent = JSON.stringify(
        { manifest_sha256: detail.manifest_sha256, manifest: detail.manifest },
        null,
        2,
      );
      showError("#dataset-history-error", null);
    } catch (error) {
      if (request === workspace.datasetRequest)
        showError("#dataset-history-error", error);
    }
  }

  function updateTrainingLaunch() {
    const model = workspace.models.find(
      (item) => item.id === $("#training-parent").value,
    );
    $("#training-start").disabled =
      workspace.trainingBusy ||
      !$("#training-dataset").value ||
      model?.status !== "ready" ||
      !model.training;
    $("#training-start").textContent = workspace.trainingBusy
      ? "Queuing training…"
      : "Start CPU training →";
  }

  async function refreshTrainingModels() {
    const select = $("#training-parent");
    const previous = select.value;
    try {
      workspace.models = await api("/api/models");
      const eligible = workspace.models.filter((model) => model.training);
      select.replaceChildren();
      for (const model of eligible) {
        const option = new Option(
          `${model.name}${model.status === "ready" ? "" : " · setup required"}`,
          model.id,
        );
        option.disabled = model.status !== "ready";
        select.append(option);
      }
      const ready = eligible.filter((model) => model.status === "ready");
      select.disabled = !ready.length || workspace.trainingBusy;
      if (!eligible.length)
        select.append(new Option("No supported training model", ""));
      select.value = ready.some((model) => model.id === previous)
        ? previous
        : ready[0]?.id || "";
      $("#training-model-status").textContent = ready.length
        ? "Local weights available. Training creates a new checkpoint and preserves its parent."
        : "A ready Faster R-CNN MobileNet V3 checkpoint and the optional CPU runtime are required. Check Model comparison for setup instructions.";
      updateTrainingLaunch();
    } catch (error) {
      workspace.models = [];
      select.replaceChildren(new Option("Model availability unavailable", ""));
      select.disabled = true;
      $("#training-model-status").textContent = error.message;
      updateTrainingLaunch();
    }
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

  async function refreshTrainings() {
    const request = ++workspace.historyRequest;
    try {
      const trainings = await api("/api/trainings");
      if (request !== workspace.historyRequest) return;
      workspace.trainings = trainings;
      if (!trainings.some((item) => item.id === workspace.trainingId))
        workspace.trainingId = trainings[0]?.id || null;
      renderTrainingHistory();
      showError("#training-history-error", null);
      if (workspace.trainingId) await loadTraining(workspace.trainingId);
    } catch (error) {
      if (request === workspace.historyRequest)
        showError("#training-history-error", error);
    }
  }

  function renderLossHistory(history) {
    const container = $("#training-loss-history");
    container.replaceChildren();
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
    for (const label of ["Step", "Training loss", "Elapsed time"]) {
      const cell = node("th", "", label);
      cell.scope = "col";
      headings.append(cell);
    }
    head.append(headings);
    const body = node("tbody");
    for (const item of history) {
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

  async function loadTraining(id) {
    const request = ++workspace.trainingRequest;
    try {
      const detail = await api(`/api/trainings/${encodeURIComponent(id)}`);
      if (request !== workspace.trainingRequest || workspace.trainingId !== id)
        return;
      workspace.trainingDetail = detail;
      $("#training-detail").hidden = false;
      $("#training-detail-name").textContent = detail.name;
      const dataset = workspace.datasets.find(
        (item) => item.id === detail.dataset_id,
      );
      $("#training-detail-context").textContent =
        `${dataset?.name || detail.dataset_id} · CPU · ${detail.history.length}/${detail.config.steps} optimizer steps · seed ${detail.config.seed}`;
      const badge = $("#training-detail-status");
      badge.textContent = detail.job?.status || "Unknown status";
      badge.className = `job-status ${detail.job?.status || ""}`;
      $("#training-detail-message").textContent = detail.job?.message || "";
      showError(
        "#training-detail-error",
        detail.job?.error ? new Error(detail.job.error) : null,
      );
      $("#training-checkpoint").hidden = !detail.checkpoint_id;
      $("#training-checkpoint-id").textContent = detail.checkpoint_id || "";
      renderLossHistory(detail.history);
      $("#training-provenance").textContent = JSON.stringify(
        {
          dataset_id: detail.dataset_id,
          parent_model_id: detail.parent_model_id,
          config: detail.config,
          metadata: detail.metadata,
          checkpoint_id: detail.checkpoint_id,
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
  $("#dataset-history").addEventListener("change", (event) => {
    workspace.datasetId = event.target.value;
    loadDataset(workspace.datasetId);
  });
  $("#training-history").addEventListener("change", (event) => {
    workspace.trainingId = event.target.value;
    workspace.trainingDetail = null;
    $("#training-detail").hidden = true;
    loadTraining(workspace.trainingId);
  });
  $("#training-parent").addEventListener("change", updateTrainingLaunch);
  $("#training-dataset").addEventListener("change", updateTrainingLaunch);
  $("#dataset-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if ($("#dataset-create").disabled) return;
    const groups = chosenGroups();
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
      $("#training-dataset").value = dataset.id;
      $("#dataset-parent").value = dataset.id;
      updateTrainingLaunch();
      notify(
        `Dataset release “${dataset.name}” saved with frozen labels and split assignments.`,
      );
    } catch (error) {
      showError("#dataset-error", error);
    } finally {
      workspace.datasetBusy = false;
      renderCandidates();
    }
  });
  $("#training-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if ($("#training-start").disabled) return;
    const payload = {
      name: $("#training-name").value.trim(),
      dataset_id: $("#training-dataset").value,
      parent_model_id: $("#training-parent").value,
      steps: Number($("#training-steps").value),
      learning_rate: Number($("#training-rate").value),
      seed: Number($("#training-seed").value),
    };
    if (!payload.name) return $("#training-name").focus();
    workspace.trainingBusy = true;
    updateTrainingLaunch();
    showError("#training-error", null);
    try {
      const detail = await api("/api/trainings", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      workspace.trainingId = detail.id;
      $("#training-name").value = "";
      await refreshTrainings();
      await refreshJobs();
      notify(
        `Training “${detail.name}” queued for ${payload.steps} CPU steps. Follow progress or cancel in Processing jobs.`,
      );
    } catch (error) {
      showError("#training-error", error);
    } finally {
      workspace.trainingBusy = false;
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
      refreshTrainings();
    }
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
