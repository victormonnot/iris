"use strict";

(() => {
  const field = (name) => $(`#benchmark-${name}`);
  const tools = window.IRISBenchmarkTools;
  const taxonomy = window.IRISTaxonomyTools;
  const view = {
    active: false, candidates: null, roles: new Map(), chosen: new Set(), models: [],
    referencePreview: null, configPreview: null, trialPreview: null,
    list: [], id: null, detail: null, trialId: null, trial: null,
    busy: null, candidateLoading: false, loading: false, trialLoading: false,
    generation: 0, candidateRequest: 0, listRequest: 0, detailRequest: 0, trialRequest: 0, catalogRequest: 0,
    jobsKey: "", loaded: false,
  };
  const base = () => `/api/benchmarks/${encodeURIComponent(view.id)}`;
  const selection = () => tools.referenceSelection(view.candidates?.groups || [], view.roles, view.chosen);
  const locked = () => view.detail?.status === "locked";
  const configOptions = () => ({ approach: "local_detector", model_id: field("model").value,
    threshold: Number(field("threshold").value), device: field("device").value,
    inference_mode: field("mode").value, tile_size: field("mode").value === "tiled" ? Number(field("tile-size").value) : 640,
    overlap: field("mode").value === "tiled" ? Number(field("overlap").value) : 0.2 });
  const trialOptions = () => ({ config_id: field("trial-config").value, role: field("trial-role").value });
  const referenceOptions = () => {
    const { frame_ids, roles } = selection();
    return { frame_ids, roles, reviewer: field("reviewer").value.trim(), independence_notes: field("notes").value.trim(),
      independent_reference: field("independent").checked, taxonomy_id: field("taxonomy").value || null };
  };
  const percentage = (value) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "N/A";
  const roleName = (role) => role === "evaluation" ? "Evaluation" : "Tuning";

  function error(name, failure) {
    field(name).textContent = failure?.message || failure || "";
    field(name).hidden = !failure;
  }
  function invalidate(kind = "reference") {
    view.generation++;
    if (kind === "reference") { view.referencePreview = null; field("preview-result").hidden = true; }
    if (kind === "config") { view.configPreview = null; field("config-preview-summary").textContent = ""; }
    if (kind === "trial") { view.trialPreview = null; field("trial-preview-summary").textContent = ""; }
    update();
  }
  function update() {
    const busy = Boolean(view.busy), selected = selection();
    for (const input of field("prepare").querySelectorAll("input,select,textarea")) input.disabled = busy || view.candidateLoading;
    field("refresh-candidates").disabled = busy || view.candidateLoading;
    field("preview").disabled = busy || view.candidateLoading || !selected.valid || !field("reviewer").value.trim() || !field("notes").value.trim() || !field("independent").checked;
    field("create").disabled = busy || !view.referencePreview || view.referencePreview.key !== tools.canonical(referenceOptions()) || !field("name").value.trim();
    field("preview").textContent = view.busy === "reference-preview" ? "Checking independent reference…" : "Preview reference";
    field("candidate-status").textContent = view.candidateLoading ? "Checking eligible human references…" : `${view.candidates?.groups.length || 0} scene groups · ${selected.counts.tuning} tuning images · ${selected.counts.evaluation} evaluation images`;
    field("refresh").disabled = busy || view.loading;
    field("history").disabled = busy || view.loading || !view.list.length;
    field("detail").hidden = !view.detail;
    const full = (view.detail?.configs.length || 0) >= 8;
    for (const input of field("config-form").querySelectorAll("input,select,button")) input.disabled = busy || view.loading || locked() || full || !view.detail;
    field("config-preview").disabled ||= !view.models.some((model) => model.id === field("model").value && model.status === "ready");
    field("config-create").disabled ||= !view.configPreview || view.configPreview.key !== tools.canonical(configOptions()) || !field("config-name").value.trim();
    field("tiling").hidden = field("mode").value !== "tiled";
    field("lock").disabled = busy || view.loading || !view.detail?.configs.length || locked() || Boolean(view.detail.trials.some((trial) => isActive(trial.job || {})));
    field("lock-status").textContent = locked()
      ? "Configurations are locked. Only evaluation trials can be created; all tuning records remain available."
      : `${view.detail?.configs.length || 0} / 8 frozen configurations. Inspect tuning results before locking. Locking permanently ends configuration changes and tuning runs for this reference.`;
    if (view.detail?.warnings?.length) field("lock-status").textContent += ` ${view.detail.warnings.join(" ")}`;
    for (const name of ["trial-config", "trial-role", "trial-preview", "trial-history"]) field(name).disabled = busy || view.loading || !view.detail;
    field("trial-role").querySelector('[value="tuning"]').disabled = locked();
    field("trial-role").querySelector('[value="evaluation"]').disabled = !locked();
    field("trial-preview").disabled ||= !field("trial-config").value;
    field("trial-create").disabled = busy || !view.trialPreview || view.trialPreview.key !== tools.canonical(trialOptions());
    field("protocol").disabled = busy || view.loading || !view.detail;
    field("trial-raw").disabled = busy || view.loading || !tools.currentTrial(view.trial, view.id, view.trialId) || view.trialLoading;
    field("trial-job").disabled = field("trial-raw").disabled || !view.trial?.job_id;
    for (const button of field("outputs").querySelectorAll("button"))
      button.disabled = busy || view.loading || view.trialLoading || !view.trial ||
        !tools.currentTrial(view.trial, view.id, view.trialId) || button.dataset.ready !== "true";
  }
  function renderGroups() {
    const list = field("groups"); list.replaceChildren();
    for (const [index, group] of (view.candidates?.groups || []).entries()) {
      const section = node("fieldset", "benchmark-group");
      section.append(node("legend", "", group.scene_group));
      const select = node("select", "");
      select.id = `benchmark-role-${index}`;
      select.setAttribute("aria-label", `Benchmark role for ${group.scene_group}`);
      select.append(new Option("Exclude this scene", ""));
      for (const role of ["tuning", "evaluation"]) {
        const option = new Option(roleName(role), role);
        option.disabled = group.allowed_roles && !group.allowed_roles.includes(role);
        select.append(option);
      }
      select.value = view.roles.get(group.scene_group) || "";
      select.addEventListener("change", () => { view.roles.set(group.scene_group, select.value); invalidate(); });
      section.append(select, node("p", "field-hint", `${group.count} eligible images${group.allowed_roles?.length < 2 ? ` · reserved role: ${group.allowed_roles.map(roleName).join(", ") || "no compatible role"}` : ""}`));
      const frames = node("details", "benchmark-frame-picker");
      frames.append(node("summary", "", "Choose images in this scene"));
      for (const frame of group.frames) {
        const label = node("label", "benchmark-check"); const input = document.createElement("input");
        input.type = "checkbox"; input.checked = view.chosen.has(frame.id); input.value = frame.id;
        input.addEventListener("change", () => { if (input.checked) view.chosen.add(frame.id); else view.chosen.delete(frame.id); invalidate(); });
        label.append(input, node("span", "", `${frame.source_filename || frame.id} · ${frame.box_count} labels${frame.negative ? " · validated negative" : ""}`));
        frames.append(label);
      }
      section.append(frames); list.append(section);
    }
    if (!view.candidates?.groups.length) list.append(node("p", "field-hint", "No eligible reference images. Select and independently review manual or imported labels in Annotation first."));
    const excluded = Object.entries(view.candidates?.excluded || {}).filter(([, count]) => count > 0).map(([reason, count]) => `${count} ${reason.replaceAll("_", " ")}`);
    field("candidate-warnings").textContent = `${excluded.length ? `Excluded: ${excluded.join("; ")}. ` : ""}Existing training partitions reserve train scenes for tuning, and validation/test scenes for evaluation. Earlier benchmarks also reserve roles.`;
    update();
  }
  async function loadCandidates() {
    if (view.candidateLoading || view.busy) return;
    const request = ++view.candidateRequest, desired = field("taxonomy").value;
    view.candidateLoading = true; invalidate(); error("error", null);
    try {
      const data = await api(`/api/benchmark-candidates${desired ? `?taxonomy_id=${encodeURIComponent(desired)}` : ""}`);
      if (request !== view.candidateRequest) return;
      const same = view.candidates?.taxonomy.id === data.taxonomy.id;
      const previousIds = new Set((view.candidates?.groups || []).flatMap((group) => group.frames.map((frame) => frame.id)));
      if (!same) { view.roles.clear(); view.chosen.clear(); }
      for (const group of data.groups) for (const frame of group.frames) if (!same || !previousIds.has(frame.id)) view.chosen.add(frame.id);
      view.candidates = data;
      field("taxonomy").replaceChildren(...(data.taxonomies || [data.taxonomy]).map((item) => new Option(taxonomy.versionLabel(item), item.id)));
      field("taxonomy").value = data.taxonomy.id;
      renderGroups();
    } catch (failure) { if (request === view.candidateRequest) error("error", failure); }
    finally { if (request === view.candidateRequest) { view.candidateLoading = false; update(); } }
  }
  async function loadModels() {
    const request = ++view.catalogRequest;
    try {
      const data = await api("/api/preannotation-providers");
      if (request !== view.catalogRequest) return;
      view.models = data.providers.find((provider) => provider.id === "local_detector")?.models || [];
      const previous = field("model").value;
      field("model").replaceChildren();
      for (const model of view.models) {
        const option = new Option(`${model.name || model.id}${model.status === "ready" ? "" : " · unavailable"}`, model.id);
        option.disabled = model.status !== "ready"; option.title = model.reason || ""; field("model").append(option);
      }
      field("model").value = view.models.find((model) => model.id === previous && model.status === "ready")?.id || view.models.find((model) => model.status === "ready")?.id || "";
      if (!field("model").value) { field("model").prepend(new Option("No ready local detector", "")); field("model").value = ""; }
      update();
    } catch (failure) { error("action-error", failure); }
  }
  async function operation(name, execute, apply, errorField = "action-error") {
    if (view.busy) return;
    view.busy = name; const id = view.id, generation = view.generation; error(errorField, null); update();
    try {
      const result = await execute();
      if (id !== view.id || generation !== view.generation) return;
      await apply(result);
    } catch (failure) { if (id === view.id) error(errorField, failure); }
    finally { view.busy = null; update(); }
  }
  function showRecord(title, record) {
    const dialog = node("dialog", "annotation-record-dialog"); const heading = node("h2", "", title);
    heading.id = "benchmark-record-title"; dialog.setAttribute("aria-labelledby", heading.id);
    const close = node("button", "button button-secondary", "Close"); close.type = "button"; close.addEventListener("click", () => dialog.close());
    dialog.append(heading, node("pre", "", JSON.stringify(record, null, 2)), close);
    dialog.addEventListener("close", () => dialog.remove()); document.body.append(dialog); dialog.showModal();
  }
  async function previewReference() {
    if (field("preview").disabled) return;
    const options = referenceOptions();
    invalidate();
    await operation("reference-preview", () => api("/api/benchmarks/preview", { method: "POST", body: JSON.stringify(options) }), (result) => {
      view.referencePreview = { ...result, key: tools.canonical(options) };
      field("preview-result").hidden = false;
      field("preview-summary").textContent = `${result.summary.frame_count} images · ${result.summary.role_counts.tuning} tuning · ${result.summary.role_counts.evaluation} evaluation · ${result.summary.negative_count} validated negatives. This freezes the reviewed reference, source pixels, classes and scene roles.`;
      field("preview-warnings").replaceChildren(...(result.warnings || []).map((warning) => node("li", "", warning)));
    }, "error");
  }
  async function createReference() {
    if (field("create").disabled) return;
    const data = { ...referenceOptions(), name: field("name").value.trim(), expected_fingerprint: view.referencePreview.fingerprint };
    await operation("reference-create", () => api("/api/benchmarks", { method: "POST", body: JSON.stringify(data) }), async (detail) => {
      view.id = detail.id; view.detail = detail; view.detailRequest++; view.loading = false;
      view.trialId = null; view.trial = null; view.trialRequest++; view.trialLoading = false;
      invalidate(); invalidate("config"); invalidate("trial"); renderTrial(); field("prepare").open = false;
      renderDetail(); await loadList(false); notify("Independent benchmark reference frozen. Candidate configurations can now be prepared.");
    }, "error");
  }
  async function loadList(refreshDetail = true) {
    const request = ++view.listRequest;
    try {
      const records = await api("/api/benchmarks"); if (request !== view.listRequest) return;
      view.list = records;
      if (!view.id && records.length) view.id = records[0].id;
      field("history").replaceChildren(new Option("Choose a saved benchmark", ""));
      for (const record of records) field("history").append(new Option(`${record.name} · ${record.status} · ${new Date(record.created_at).toLocaleDateString()}`, record.id));
      field("history").value = view.id || "";
      if (view.id && refreshDetail) await loadDetail();
      update();
    } catch (failure) { if (request === view.listRequest) error("history-error", failure); }
  }
  async function loadDetail() {
    if (!view.id) return;
    const id = view.id, request = ++view.detailRequest; view.loading = true; update();
    try {
      const detail = await api(base()); if (request !== view.detailRequest || id !== view.id) return;
      const old = view.detail;
      view.detail = detail;
      if (old?.status !== detail.status || tools.canonical(old?.configs) !== tools.canonical(detail.configs)) invalidate("config");
      renderDetail();
      if (view.trialId) loadTrial();
    } catch (failure) { if (request === view.detailRequest) error("history-error", failure); }
    finally { if (request === view.detailRequest) { view.loading = false; update(); } }
  }
  function renderDetail() {
    if (!view.detail) { update(); return; }
    const detail = view.detail, summary = detail.summary || {}, manifest = detail.manifest;
    if (view.trialId && !detail.trials.some((trial) => trial.id === view.trialId)) {
      view.trialId = null; view.trial = null; view.trialRequest++; view.trialLoading = false; renderTrial();
    }
    const counts = summary.role_counts || { tuning: manifest.frames.filter((frame) => frame.role === "tuning").length, evaluation: manifest.frames.filter((frame) => frame.role === "evaluation").length };
    field("reference-summary").textContent = `${detail.name} · ${taxonomy.versionLabel(manifest.taxonomy)} · ${counts.tuning} tuning / ${counts.evaluation} evaluation images · independent reference declared by ${manifest.reference.reviewer}.`;
    field("configs").replaceChildren();
    const previous = field("trial-config").value;
    field("trial-config").replaceChildren(new Option("Choose a frozen configuration", ""));
    for (const config of detail.configs) {
      const section = node("article", "benchmark-config");
      section.append(node("strong", "", config.name), node("p", "field-hint", `${config.config.model_name || config.config.model_id} · ${config.config.inference.mode} · proposal score ≥ ${config.config.threshold}`));
      const button = node("button", "text-button", "Inspect frozen configuration"); button.type = "button"; button.addEventListener("click", () => showRecord("Frozen candidate configuration", config)); section.append(button); field("configs").append(section);
      field("trial-config").append(new Option(config.name, config.id));
    }
    field("trial-config").value = detail.configs.some((config) => config.id === previous) ? previous : detail.configs[0]?.id || "";
    field("trial-role").value = locked() ? "evaluation" : "tuning";
    field("trial-history").replaceChildren(new Option("Choose a saved trial", ""));
    for (const trial of detail.trials) field("trial-history").append(new Option(`${trial.config_name} · ${roleName(trial.split)} · ${trial.job?.status || "saved"} · ${new Date(trial.created_at).toLocaleString()}`, trial.id));
    if (!view.trialId && detail.trials.length) view.trialId = detail.trials[0].id;
    field("trial-history").value = view.trialId || "";
    renderResults(); update();
  }
  function renderResults() {
    const container = field("results-table"); container.replaceChildren();
    if (!view.detail.trials.length) { container.append(node("p", "field-hint", "No measured trial yet. Preview and run a tuning trial with a frozen configuration.")); return; }
    const table = node("table", "benchmark-results-table"), head = node("thead", ""), row = node("tr", "");
    for (const label of ["Configuration / role", "Job / coverage", "Extra / missed boxes", "Class conflicts", "Precision / recall", "Matched IoU", "Local processing", "Human corrections"]) row.append(node("th", "", label));
    head.append(row); table.append(head); const body = node("tbody", "");
    for (const trial of view.detail.trials) {
      const metrics = trial.quality?.metrics?.summary, corrections = trial.corrections;
      const tr = node("tr", "");
      const values = [`${trial.config_name} · ${roleName(trial.split)}`, `${trial.job?.status || "saved"} · ${trial.counts?.ready || 0}/${trial.counts?.total || 0} outputs`,
        metrics ? `${metrics.fp} extra / ${metrics.fn} missed` : "Incomplete · not scored", metrics ? String(metrics.class_conflicts) : "N/A", metrics ? `${percentage(metrics.precision)} / ${percentage(metrics.recall)}` : "N/A",
        metrics ? percentage(metrics.matched_iou_mean) : "N/A", trial.latency ? `${tools.duration(trial.latency.total_ms)} · ${trial.latency.measured_count}/${trial.latency.planned_count} images timed` : "See saved trial details", corrections ? `${corrections.reviewed_count}/${corrections.output_count} reviewed · ${tools.duration(corrections.recorded_review_ms)} recorded${corrections.fully_timed_count < corrections.timed_count ? " · interruptions recorded" : ""}` : "Unmeasured"];
      for (const value of values) tr.append(node("td", "", value)); body.append(tr);
    }
    table.append(body); container.append(table);
    container.append(node("p", "field-hint", "Operating-point box matching against the frozen reference; see the saved scoring protocol for matching rules. Native model scores are not comparable probabilities. Incomplete outputs are never counted as successful empty predictions."));
  }
  function configValid() {
    return ["threshold", ...(field("mode").value === "tiled" ? ["tile-size", "overlap"] : [])].every((name) => field(name).value.trim() && field(name).reportValidity());
  }
  async function previewConfig() {
    if (field("config-preview").disabled || !configValid()) return;
    const options = configOptions(); invalidate("config");
    await operation("config-preview", () => api(`${base()}/configs/preview`, { method: "POST", body: JSON.stringify(options) }), (result) => {
      view.configPreview = { ...result, key: tools.canonical(options) };
      const coverage = result.config.proposal_contract;
      field("config-preview-summary").textContent = `${result.config.model_name} · covered class IDs: ${coverage.supported_class_ids.join(", ")}${coverage.unsupported_class_ids.length ? ` · uncovered: ${coverage.unsupported_class_ids.join(", ")}` : ""}. ${result.work.tuning.total_forward_passes} tuning / ${result.work.evaluation.total_forward_passes} evaluation detector passes including warm-up. ${(result.warnings || []).join(" ")}`;
    });
  }
  async function createConfig() {
    if (field("config-create").disabled) return;
    const data = { ...configOptions(), name: field("config-name").value.trim(), expected_fingerprint: view.configPreview.fingerprint };
    await operation("config-create", () => api(`${base()}/configs`, { method: "POST", body: JSON.stringify(data) }), async () => { invalidate("config"); await loadDetail(); notify("Candidate configuration frozen. Run tuning trials before locking for evaluation."); });
  }
  async function lock() {
    if (field("lock").disabled || !window.confirm("Lock all current candidate configurations for evaluation? You will no longer be able to add configurations or run tuning trials for this benchmark.")) return;
    await operation("lock", () => api(`${base()}/lock`, { method: "POST", body: JSON.stringify({ expected_fingerprint: view.detail.lock_fingerprint }) }), (detail) => {
      view.detail = detail; invalidate("config"); invalidate("trial"); renderDetail(); notify("Configurations locked. Evaluation trials now use this frozen candidate set.");
    });
  }
  async function previewTrial() {
    if (field("trial-preview").disabled) return;
    const options = trialOptions(); invalidate("trial");
    await operation("trial-preview", () => api(`${base()}/trials/preview`, { method: "POST", body: JSON.stringify(options) }), (result) => {
      view.trialPreview = { ...result, key: tools.canonical(options) };
      field("trial-preview-summary").textContent = `${result.frame_ids.length} ${roleName(options.role).toLowerCase()} images · ${result.work.total_forward_passes} detector passes including warm-up. Reference labels are withheld from the candidate. ${(result.warnings || []).join(" ")}`;
    });
  }
  async function createTrial() {
    if (field("trial-create").disabled) return;
    const data = { ...trialOptions(), expected_fingerprint: view.trialPreview.fingerprint };
    await operation("trial-create", () => api(`${base()}/trials`, { method: "POST", body: JSON.stringify(data) }), async (trial) => {
      view.trialRequest++; view.trialId = trial.id; view.trial = trial; view.trialLoading = false;
      invalidate("trial"); renderTrial(); await loadDetail();
      refreshJobs().catch((failure) => error("action-error", failure)); notify("Benchmark trial queued. The reference labels are not part of the candidate request.");
    });
  }
  async function loadTrial() {
    if (!view.trialId) return;
    const id = view.trialId, benchmarkId = view.id, request = ++view.trialRequest; view.trialLoading = true; update();
    try {
      const trial = await api(`/api/benchmark-trials/${encodeURIComponent(id)}`);
      if (request !== view.trialRequest || id !== view.trialId || benchmarkId !== view.id || trial.benchmark_id !== benchmarkId) return;
      view.trial = trial; renderTrial();
    } catch (failure) { if (request === view.trialRequest) error("action-error", failure); }
    finally { if (request === view.trialRequest) { view.trialLoading = false; update(); } }
  }
  function renderTrial() {
    field("trial-detail").hidden = !view.trial;
    if (!view.trial) return;
    const trial = view.trial;
    field("trial-summary").textContent = `${trial.config_name} · ${roleName(trial.split)} · ${trial.job?.status || "saved"} · ${trial.counts.ready}/${trial.counts.total} usable outputs${trial.quality?.reason ? `. ${trial.quality.reason}` : ""}`;
    const measured = (trial.outputs || []).map((output) => output.metadata?.timing?.elapsed_ms).filter((value) => typeof value === "number" && Number.isFinite(value) && value >= 0);
    field("trial-summary").textContent += ` · local image processing: ${measured.length ? tools.duration(measured.reduce((sum, value) => sum + value, 0)) : "unmeasured"} across ${measured.length}/${trial.counts.total} images. Includes image decode and local inference; separate from human correction time. Monetary cost is not measured.`;
    field("outputs").replaceChildren();
    if (trial.quality?.metrics?.per_class) {
      const classes = node("details", "benchmark-class-results"); classes.append(node("summary", "", "Results for each saved class"));
      for (const category of view.detail.manifest.taxonomy.classes) {
        const counts = trial.quality.metrics.per_class[category.id];
        if (counts) classes.append(node("p", "field-hint", `${category.name} (${category.id}): ${counts.tp} matched · ${counts.fp} extra · ${counts.fn} missed`));
      }
      field("outputs").append(classes);
    }
    for (const frame of trial.frames) {
      const row = node("article", "benchmark-output");
      const correction = trial.corrections?.frames.find((entry) => entry.output_id === frame.output_id);
      row.append(node("strong", "", frame.source_filename || frame.frame_id), node("p", "field-hint", `${frame.state} · ${frame.proposal_count} candidate boxes${frame.error ? ` · ${frame.error}` : ""}${correction ? ` · correction ${correction.status}, revision ${correction.revision} · ${tools.duration(correction.timing?.elapsed_ms)} recorded` : " · correction not measured"}`));
      const button = node("button", "button button-secondary", correction ? "Open correction record" : "Measure human correction");
      button.type = "button";
      button.dataset.ready = String(Boolean(frame.output_id) && !frame.error && frame.state === "ready");
      button.addEventListener("click", () => {
        if (button.disabled || !tools.currentTrial(trial, view.id, view.trialId) || view.trial?.id !== trial.id) return;
        window.dispatchEvent(new CustomEvent("iris:benchmark-correct", { detail: { output_id: frame.output_id, label: `${trial.config_name} · ${roleName(trial.split)} · ${frame.source_filename || frame.frame_id}` } }));
      });
      row.append(button); field("outputs").append(row);
    }
    update();
  }
  field("preview").addEventListener("click", previewReference); field("create").addEventListener("click", createReference);
  field("config-preview").addEventListener("click", previewConfig); field("config-create").addEventListener("click", createConfig);
  field("trial-preview").addEventListener("click", previewTrial); field("trial-create").addEventListener("click", createTrial);
  field("lock").addEventListener("click", lock); field("refresh-candidates").addEventListener("click", loadCandidates);
  field("refresh").addEventListener("click", () => { loadList(); loadModels(); });
  field("taxonomy").addEventListener("change", loadCandidates);
  for (const name of ["reviewer", "notes", "independent"]) field(name).addEventListener(name === "independent" ? "change" : "input", () => invalidate());
  field("name").addEventListener("input", update); field("config-name").addEventListener("input", update);
  for (const name of ["model", "threshold", "device", "mode", "tile-size", "overlap"]) field(name).addEventListener(["threshold", "tile-size", "overlap"].includes(name) ? "input" : "change", () => invalidate("config"));
  for (const name of ["trial-config", "trial-role"]) field(name).addEventListener("change", () => invalidate("trial"));
  field("history").addEventListener("change", () => {
    view.id = field("history").value || null; view.detail = null; view.trialId = null; view.trial = null;
    view.detailRequest++; view.trialRequest++; view.loading = false; view.trialLoading = false; invalidate("config"); invalidate("trial"); renderTrial();
    if (view.id) loadDetail();
  });
  field("trial-history").addEventListener("change", () => { view.trialId = field("trial-history").value || null; view.trial = null; view.trialRequest++; renderTrial(); if (view.trialId) loadTrial(); });
  field("protocol").addEventListener("click", () => showRecord("Frozen independent reference and benchmark protocol", view.detail));
  field("trial-raw").addEventListener("click", () => showRecord("Saved trial protocol, model inputs and raw outputs", view.trial));
  field("trial-job").addEventListener("click", () => window.dispatchEvent(new CustomEvent("iris:job-open", { detail: { job_id: view.trial?.job_id } })));
  window.addEventListener("iris:workspace", (event) => {
    view.active = event.detail.name === "benchmark";
    if (!view.active) return;
    if (!view.loaded) { view.loaded = true; loadCandidates(); loadModels(); }
    loadList();
  });
  for (const name of ["iris:before-workspace", "iris:before-session"]) window.addEventListener(name, (event) => {
    if (!view.busy) return;
    event.preventDefault(); notify("Wait for the benchmark request to finish before changing context.", true);
  });
  window.addEventListener("beforeunload", (event) => { if (view.busy) { event.preventDefault(); event.returnValue = ""; } });
  window.addEventListener("iris:benchmark-correction-saved", () => { if (view.id) loadDetail(); });
  window.addEventListener("iris:jobs", () => {
    if (!view.active || !view.id || view.busy) return;
    const key = tools.canonical(state.jobs.filter((job) => job.kind === "benchmark").map((job) => [job.id, job.status, job.progress]));
    if (key !== view.jobsKey) { view.jobsKey = key; loadDetail(); }
  });
  update();
})();
