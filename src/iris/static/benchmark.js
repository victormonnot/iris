"use strict";

(() => {
  const field = (name) => $(`#benchmark-${name}`);
  const tools = window.IRISBenchmarkTools;
  const externalTools = window.IRISBenchmarkExternalTools;
  const samTools = window.IRISBenchmarkSAMTools;
  const taxonomy = window.IRISTaxonomyTools;
  const view = {
    active: false, candidates: null, roles: new Map(), chosen: new Set(), models: [],
    referencePreview: null, configPreview: null, trialPreview: null,
    list: [], id: null, detail: null, trialId: null, trial: null,
    busy: null, candidateLoading: false, loading: false, trialLoading: false,
    generation: 0, candidateRequest: 0, listRequest: 0, detailRequest: 0, trialRequest: 0, catalogRequest: 0,
    jobsKey: "", loaded: false,
    providers: null, providerRequest: 0, externalImages: new Set(), externalExpiry: null,
    samPrompts: null,
  };
  const base = () => `/api/benchmarks/${encodeURIComponent(view.id)}`;
  const selection = () => tools.referenceSelection(view.candidates?.groups || [], view.roles, view.chosen);
  const locked = () => view.detail?.status === "locked";
  const multimodal = () => field("approach").value === "multimodal";
  const segmentation = () => field("approach").value === "segmentation";
  const configOptions = () => multimodal() ? { approach: "multimodal", model_id: view.providers?.multimodal?.model || "gpt-6-astra",
    multimodal: { image_long_edge: Number(field("image-edge").value), reasoning_effort: field("reasoning").value, max_output_tokens: Number(field("output-tokens").value) } }
    : segmentation() ? { approach: "segmentation", model_id: view.providers?.segmentation?.model_id || "sam3",
      segmentation: { class_prompts: samTools.promptPayload(view.samPrompts, view.detail?.manifest.taxonomy), threshold: Number(field("sam-threshold").value), device: field("sam-device").value } }
    : ({ approach: "local_detector", model_id: field("model").value,
    threshold: Number(field("threshold").value), device: field("device").value,
    inference_mode: field("mode").value, tile_size: field("mode").value === "tiled" ? Number(field("tile-size").value) : 640,
    overlap: field("mode").value === "tiled" ? Number(field("overlap").value) : 0.2 });
  const trialOptions = () => ({ config_id: field("trial-config").value, role: field("trial-role").value });
  const selectedConfig = () => view.detail?.configs.find((config) => config.id === field("trial-config").value);
  const trialApproach = () => selectedConfig()?.approach || selectedConfig()?.config?.approach || "local_detector";
  const externalTrial = () => selectedConfig()?.approach === "multimodal" || selectedConfig()?.config?.approach === "multimodal";
  const externalBudget = () => field("external-budget").value.trim() ? Number(field("external-budget").value) : NaN;
  const approval = () => externalTools.approval(view.trialPreview, { budget: externalBudget(), consent: field("external-consent").checked, loaded: view.externalImages });
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
    if (kind === "trial" || kind === "config") {
      view.trialPreview = null; field("trial-preview-summary").textContent = "";
      field("local-plan").hidden = true; field("local-plan-record").textContent = "";
      clearExternalPreview();
    }
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
    for (const input of field("config-form").querySelectorAll("input,select,textarea,button")) input.disabled = busy || view.loading || locked() || full || !view.detail;
    field("config-preview").disabled ||= multimodal() ? !view.providers?.multimodal_settings : segmentation() ? !view.providers?.segmentation_settings : !view.models.some((model) => model.id === field("model").value && model.status === "ready");
    field("config-create").disabled ||= !view.configPreview || view.configPreview.key !== tools.canonical(configOptions()) || !field("config-name").value.trim();
    field("local-settings").hidden = multimodal() || segmentation();
    field("multimodal-settings").hidden = !multimodal();
    field("sam-settings").hidden = !segmentation();
    field("tiling").hidden = multimodal() || segmentation() || field("mode").value !== "tiled";
    field("lock").disabled = busy || view.loading || !view.detail?.configs.length || locked() || Boolean(view.detail.trials.some((trial) => isActive(trial.job || {})));
    field("lock-status").textContent = locked()
      ? "Configurations are locked. Only evaluation trials can be created; all tuning records remain available."
      : `${view.detail?.configs.length || 0} / 8 frozen configurations. Inspect tuning results before locking. Locking permanently ends configuration changes and tuning runs for this reference.`;
    if (view.detail?.warnings?.length) field("lock-status").textContent += ` ${view.detail.warnings.join(" ")}`;
    for (const name of ["trial-config", "trial-role", "trial-preview", "trial-history"]) field(name).disabled = busy || view.loading || !view.detail;
    field("trial-role").querySelector('[value="tuning"]').disabled = locked();
    field("trial-role").querySelector('[value="evaluation"]').disabled = !locked();
    field("trial-preview").disabled ||= !field("trial-config").value;
    field("trial-create").disabled = busy || view.loading || !view.trialPreview || view.trialPreview.key !== tools.canonical(trialOptions());
    field("trial-create").disabled ||= externalTrial() && !approval().allowed;
    field("trial-create").disabled ||= !samTools.launchAllowed(view.trialPreview, trialApproach());
    field("trial-create").textContent = externalTrial() ? "Send approved external trial" : trialApproach() === "segmentation" ? "Run checked local SAM trial" : "Run checked trial";
    field("external-budget").disabled = busy || !view.trialPreview?.external_plan;
    field("external-consent").disabled = busy || !view.trialPreview?.external_plan;
    if (view.trialPreview?.external_plan) {
      field("external-status").textContent = approval().reason;
      field("external-consent-label").textContent = `I approve sending these ${view.trialPreview.external_plan.requests.length} images and their displayed prompts to ${view.trialPreview.external_plan.provider} / ${view.trialPreview.external_plan.model}, with a planning budget of ${externalTools.money(externalBudget())} for this trial.`;
      field("external-image-status").textContent = `${view.externalImages.size}/${view.trialPreview.external_plan.requests.length} outgoing images displayed. Inspect every image and prompt before approving.`;
    }
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
  async function loadProviders() {
    const request = ++view.providerRequest;
    try {
      const providers = await api("/api/benchmark-providers");
      if (request !== view.providerRequest) return;
      const first = !view.providers;
      view.providers = providers;
      const settings = providers.multimodal_settings;
      for (const [name, values, preferred] of [["image-edge", settings.image_long_edges, settings.defaults.image_long_edge], ["reasoning", settings.reasoning_efforts, settings.defaults.reasoning_effort]]) {
        const previous = field(name).value;
        field(name).replaceChildren(...values.map((value) => new Option(String(value), String(value))));
        field(name).value = values.map(String).includes(previous) ? previous : String(preferred);
      }
      field("output-tokens").min = settings.max_output_tokens.min;
      field("output-tokens").max = settings.max_output_tokens.max;
      if (first) field("output-tokens").value = settings.defaults.max_output_tokens;
      field("multimodal-model").textContent = `Exact model: ${providers.multimodal.model} · image detail: ${settings.detail}`;
      field("provider-status").textContent = externalTools.providerStatus(providers.multimodal);
      field("provider-setup").textContent = "The server reads IRIS_OPENAI_API_KEY or OPENAI_API_KEY. Keys are never entered or displayed here. This offline check does not verify model access; settings can be prepared without a key.";
      const sam = providers.segmentation_settings;
      field("sam-provider-status").textContent = samTools.availability(providers.segmentation);
      field("sam-availability").textContent = samTools.availability(providers.segmentation);
      if (sam) {
        field("sam-threshold").min = sam.threshold.min;
        field("sam-threshold").max = sam.threshold.max;
        if (first) field("sam-threshold").value = sam.defaults.threshold;
        const previous = field("sam-device").value;
        field("sam-device").replaceChildren(...sam.devices.map((device) => {
          const status = providers.segmentation?.devices?.find((item) => item.id === device);
          return new Option(`${device.toUpperCase()}${status?.available === false ? " · setup required" : ""}`, device);
        }));
        field("sam-device").value = sam.devices.includes(previous) ? previous : sam.defaults.device;
        renderSAMPrompts();
      }
    } catch (failure) {
      if (request !== view.providerRequest) return;
      view.providers = null;
      field("provider-status").textContent = failure.message;
      field("sam-provider-status").textContent = failure.message;
      field("sam-availability").textContent = failure.message;
    }
    update();
  }
  function renderSAMPrompts() {
    const next = samTools.promptState(view.samPrompts, view.id, view.detail?.manifest.taxonomy);
    if (next === view.samPrompts) return;
    view.samPrompts = next;
    const container = field("sam-prompts"); container.replaceChildren();
    for (const [index, category] of (view.detail?.manifest.taxonomy.classes || []).entries()) {
      const row = node("div", "benchmark-sam-prompt");
      const label = node("label", "", `${category.name} · ${category.id}`);
      const input = node("textarea", ""); input.id = `benchmark-sam-prompt-${index}`;
      label.htmlFor = input.id;
      const definition = node("p", "field-hint", category.definition);
      definition.id = `${input.id}-definition`; input.setAttribute("aria-describedby", definition.id);
      input.rows = 2; input.required = true;
      input.maxLength = view.providers?.segmentation_settings?.prompt_limits.max_length || 120;
      input.value = next.values.get(category.id);
      input.addEventListener("input", () => { next.values.set(category.id, input.value); invalidate("config"); });
      row.append(label, definition, input); container.append(row);
    }
  }
  function clearExternalPreview() {
    clearTimeout(view.externalExpiry); view.externalExpiry = null;
    view.externalImages.clear();
    field("external-consent").checked = false;
    field("external-budget").value = "";
    field("external-preview").hidden = true;
    field("external-images").replaceChildren();
  }
  function renderExternalPreview() {
    clearExternalPreview();
    const preview = view.trialPreview, plan = preview?.external_plan;
    if (!plan) return;
    field("external-preview").hidden = false;
    field("external-provider").textContent = `${plan.provider} / ${plan.model} · ${plan.requests.length} image requests · preview expires ${new Date(preview.expires_at).toLocaleString()}. ${externalTools.providerStatus(plan.provider_status)}`;
    field("external-cost").textContent = `Conservative planning estimate: ${externalTools.money(plan.estimate?.upper_bound_usd)} for this trial.`;
    field("external-basis").textContent = typeof plan.estimate?.basis === "string" ? plan.estimate.basis : JSON.stringify(plan.estimate?.basis || "");
    const amount = plan.estimate?.upper_bound_usd;
    if (typeof amount === "number" && Number.isFinite(amount) && amount >= 0) {
      field("external-budget").value = String(amount);
      field("external-budget").min = String(amount);
    }
    for (const [index, request] of plan.requests.entries()) {
      const figure = node("figure", "benchmark-external-image");
      const image = node("img", ""); image.alt = `Outgoing image ${index + 1} for frame ${request.frame_id}`;
      const info = request.input.image;
      const caption = node("figcaption", "", `Image ${index + 1} · ${info.width ?? info.original_width} × ${info.height ?? info.original_height} original → ${info.sent_width} × ${info.sent_height} sent · ${externalTools.money(request.estimate?.upper_bound_usd)} planning estimate`);
      image.addEventListener("load", () => {
        if (preview !== view.trialPreview) return;
        if (image.naturalWidth > 0) view.externalImages.add(request.frame_id);
        update();
      });
      image.addEventListener("error", () => {
        if (preview !== view.trialPreview) return;
        view.externalImages.delete(request.frame_id); field("external-consent").checked = false;
        caption.textContent += " · Image could not be loaded. Prepare a new preview before sending."; update();
      });
      const details = node("details", "benchmark-request-details");
      details.append(node("summary", "", "Exact prompt, class definitions and image transform"), node("pre", "", request.input.prompt), node("pre", "", JSON.stringify({ image: info, request_sha256: request.input.request_sha256, estimate: request.estimate }, null, 2)));
      figure.append(image, caption, details); field("external-images").append(figure);
      try { image.src = projectURL(request.image_url); }
      catch { caption.textContent += " · Invalid local preview image URL; sending is blocked."; }
    }
    const remaining = Date.parse(preview.expires_at) - Date.now();
    if (remaining > 0) view.externalExpiry = setTimeout(() => {
      if (preview !== view.trialPreview) return;
      field("external-consent").checked = false; update();
    }, Math.min(remaining + 20, 2147483647));
    update();
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
    renderSAMPrompts();
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
      const external = config.config.approach === "multimodal";
      const sam = config.config.approach === "segmentation";
      section.append(node("strong", "", config.name), node("p", "field-hint", external
        ? `A · ${config.config.model_name || config.config.model_id} · external API · image edge ${config.config.provider_config.image_encoding.long_edge}px · reasoning ${config.config.provider_config.settings.reasoning.effort} · output limit ${config.config.provider_config.settings.max_output_tokens} tokens · no detector confidence scores`
        : sam ? `B · ${config.config.model_name || config.config.model_id} · local ${config.config.provider_config.settings.device.toUpperCase()} · ${config.config.provider_config.prompts.length} class prompts · native SAM score > ${config.config.provider_config.settings.threshold} · boxes only; no masks`
        : `${config.config.model_name || config.config.model_id} · ${config.config.inference.mode} · proposal score ≥ ${config.config.threshold}`));
      if (sam) {
        const prompts = node("details", "benchmark-sam-saved-prompts");
        prompts.append(node("summary", "", "Frozen class prompts"));
        for (const item of config.config.provider_config.prompts) prompts.append(node("p", "field-hint", `${item.class_id}: ${item.text}`));
        section.append(prompts);
      }
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
    for (const label of ["Configuration / role", "Job / coverage", "Extra / missed boxes", "Class conflicts", "Precision / recall", "Matched IoU", "Processing / API time", "Usage cost estimate", "Human corrections"]) row.append(node("th", "", label));
    head.append(row); table.append(head); const body = node("tbody", "");
    for (const trial of view.detail.trials) {
      const metrics = trial.quality?.metrics?.summary, corrections = trial.corrections;
      const modelLoading = trial.config?.candidate_config?.approach === "segmentation" ? ` · model loading ${tools.duration(trial.latency?.model_load_ms)} separately` : "";
      const tr = node("tr", "");
      const values = [`${trial.config_name} · ${roleName(trial.split)}`, `${trial.job?.status || "saved"} · ${trial.counts?.ready || 0}/${trial.counts?.total || 0} outputs`,
        metrics ? `${metrics.fp} extra / ${metrics.fn} missed` : "Incomplete · not scored", metrics ? String(metrics.class_conflicts) : "N/A", metrics ? `${percentage(metrics.precision)} / ${percentage(metrics.recall)}` : "N/A",
        metrics ? percentage(metrics.matched_iou_mean) : "N/A", trial.latency ? `${tools.duration(trial.latency.total_ms)} · ${trial.latency.measured_count}/${trial.latency.planned_count} images timed${modelLoading}` : "See saved trial details", externalTools.costPresentation(trial.external_dispatch), corrections ? `${corrections.reviewed_count}/${corrections.output_count} reviewed · ${tools.duration(corrections.recorded_review_ms)} recorded${corrections.fully_timed_count < corrections.timed_count ? " · interruptions recorded" : ""}` : "Unmeasured"];
      for (const value of values) tr.append(node("td", "", value)); body.append(tr);
    }
    table.append(body); container.append(table);
    container.append(node("p", "field-hint", "Operating-point box matching against the frozen reference; see the saved scoring protocol for matching rules. Native model scores are not comparable probabilities. Incomplete outputs are never counted as successful empty predictions."));
  }
  function configValid() {
    if (multimodal()) return Boolean(field("image-edge").value && field("reasoning").value && field("output-tokens").value.trim() && field("output-tokens").reportValidity());
    if (segmentation()) {
      const promptError = samTools.promptError(view.samPrompts, view.detail?.manifest.taxonomy, view.providers?.segmentation_settings?.prompt_limits);
      if (promptError) { error("action-error", promptError); return false; }
      return Boolean(field("sam-device").value && field("sam-threshold").value.trim() && field("sam-threshold").reportValidity());
    }
    return ["threshold", ...(field("mode").value === "tiled" ? ["tile-size", "overlap"] : [])].every((name) => field(name).value.trim() && field(name).reportValidity());
  }
  async function previewConfig() {
    if (field("config-preview").disabled || !configValid()) return;
    const options = configOptions(); invalidate("config");
    await operation("config-preview", () => api(`${base()}/configs/preview`, { method: "POST", body: JSON.stringify(options) }), (result) => {
      view.configPreview = { ...result, key: tools.canonical(options) };
      if (options.approach === "multimodal") {
        field("config-preview-summary").textContent = `A · ${result.config.model_name || result.config.model_id} · ${result.work.tuning.request_count} tuning / ${result.work.evaluation.request_count} evaluation image requests. Saving this configuration sends nothing externally. Each trial requires its own exact-image preview and explicit budget approval. ${(result.warnings || []).join(" ")}`;
        return;
      }
      if (options.approach === "segmentation") {
        field("config-preview-summary").textContent = `B · ${result.config.model_name || result.config.model_id}. Tuning: ${samTools.workSummary(result.work.tuning)}. Evaluation: ${samTools.workSummary(result.work.evaluation)}. ${samTools.availability(result.provider_status)} Saving freezes the published model identity and these prompts; it installs nothing and runs no model. ${(result.warnings || []).join(" ")}`;
        return;
      }
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
      const work = result.external_plan ? `${result.external_plan.requests.length} external image requests`
        : result.local_plan ? `local SAM · ${samTools.workSummary(result.work)}` : `${result.work.total_forward_passes} detector passes including warm-up`;
      field("trial-preview-summary").textContent = `${result.frame_ids.length} ${roleName(options.role).toLowerCase()} images · ${work}. Reference labels are withheld from the candidate. ${result.launch_allowed === false ? `Launch unavailable: ${result.launch_reason || "Local setup must be completed."} ` : ""}${(result.warnings || []).join(" ")}`;
      field("local-plan").hidden = !result.local_plan;
      field("local-plan").open = false;
      field("local-plan-record").textContent = result.local_plan ? JSON.stringify({ local_plan: result.local_plan, work: result.work, configuration: selectedConfig()?.config, provider_status: result.provider_status }, null, 2) : "";
      renderExternalPreview();
    });
  }
  async function createTrial() {
    if (field("trial-create").disabled) return;
    if (!samTools.launchAllowed(view.trialPreview, trialApproach())) { update(); return; }
    const preview = view.trialPreview, isExternal = externalTrial(), benchmarkId = view.id, path = base();
    if (isExternal && !approval().allowed) { update(); return; }
    const data = { ...trialOptions(), expected_fingerprint: preview.fingerprint,
      ...(isExternal ? { approve_external: true, max_cost_usd: externalBudget(), preview_token: preview.preview_token } : {}) };
    let recovered = false;
    try {
      await operation("trial-create", async () => {
        try { return await api(`${path}/trials`, { method: "POST", body: JSON.stringify(data) }); }
        catch (failure) {
          if (!isExternal) throw failure;
          let receipt = null;
          try { receipt = externalTools.findTrialReceipt((await api(path)).trials, preview.fingerprint, benchmarkId); } catch { /* No repeated external POST. */ }
          if (receipt) { recovered = true; return receipt; }
          throw new Error(`${failure.message} IRIS could not confirm whether this trial was queued. Check saved trials and Project jobs before preparing another request; another request may incur another charge. Nothing was resent automatically.`);
        }
      }, async (trial) => {
        view.trialRequest++; view.trialId = trial.id; view.trial = trial; view.trialLoading = false;
        invalidate("trial"); renderTrial(); await loadDetail();
        refreshJobs().catch((failure) => error("action-error", failure));
        notify(recovered ? "The approved trial was already recorded. Its saved receipt is open; no second request was sent." : "Benchmark trial queued. The reference labels are not part of the candidate request.");
      });
    } finally { if (isExternal) invalidate("trial"); }
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
    const remote = trial.config?.candidate_config?.approach === "multimodal" || Boolean(trial.external_dispatch);
    const sam = trial.config?.candidate_config?.approach === "segmentation";
    field("trial-summary").textContent += ` · ${remote ? "API/image processing" : "Local image processing"}: ${measured.length ? tools.duration(measured.reduce((sum, value) => sum + value, 0)) : "unmeasured"} across ${measured.length}/${trial.counts.total} images. ${remote ? "Includes observed request processing; separate from human correction time." : "Includes image decode and local inference; separate from human correction time. Monetary cost is not measured."}`;
    if (remote) field("trial-summary").textContent += ` ${externalTools.costPresentation(trial.external_dispatch)}. Usage-based estimates are not the provider's invoice.`;
    if (sam) field("trial-summary").textContent += ` SAM 3 model loading: ${tools.duration(trial.latency?.model_load_ms)}, recorded separately from image processing. Image times include the first pass; there is no warm-up pass. Native SAM scores are not calibrated probabilities. Metrics cover native boxes; masks are neither calculated nor saved.`;
    field("outputs").replaceChildren();
    if (trial.external_dispatch) {
      const dispatch = trial.external_dispatch, presentation = window.IRISJobTools.dispatchPresentation(dispatch);
      const receipt = node("section", `benchmark-dispatch${presentation.unknown ? " unknown" : ""}`);
      receipt.append(node("strong", "", presentation.label), node("p", "field-hint", presentation.explanation),
        node("p", "field-hint", `Approved planning budget: ${externalTools.money(dispatch.budget_microusd / 1000000)} · reserved for recorded attempts: ${externalTools.money(dispatch.reserved_microusd / 1000000)}. These reservations are not provider charges.`));
      field("outputs").append(receipt);
    }
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
      const dispatch = trial.external_dispatch?.outputs?.find((item) => item.frame_id === frame.frame_id);
      if (dispatch) {
        const presentation = window.IRISJobTools.dispatchPresentation({ ...dispatch, external: true });
        const detail = node("div", `benchmark-dispatch${presentation.unknown ? " unknown" : ""}`);
        detail.append(node("strong", "", presentation.label), node("p", "field-hint", presentation.explanation),
          node("p", "field-hint", typeof dispatch.usage_cost_usd === "number" ? `${externalTools.money(dispatch.usage_cost_usd)} estimated from recorded usage; not an invoice.` : "Usage cost unknown; no zero charge is inferred."));
        const record = node("details", "");
        record.append(node("summary", "", "Dispatch and usage receipt"), node("pre", "", JSON.stringify(dispatch, null, 2)));
        detail.append(record); row.append(detail);
      }
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
  field("refresh").addEventListener("click", () => { invalidate("trial"); loadList(); loadModels(); loadProviders(); });
  field("taxonomy").addEventListener("change", loadCandidates);
  for (const name of ["reviewer", "notes", "independent"]) field(name).addEventListener(name === "independent" ? "change" : "input", () => invalidate());
  field("name").addEventListener("input", update); field("config-name").addEventListener("input", update);
  for (const name of ["model", "threshold", "device", "mode", "tile-size", "overlap"]) field(name).addEventListener(["threshold", "tile-size", "overlap"].includes(name) ? "input" : "change", () => invalidate("config"));
  field("approach").addEventListener("change", () => {
    if (["Local detector control", "A · GPT-6 Astra", "B · SAM 3"].includes(field("config-name").value)) field("config-name").value = multimodal() ? "A · GPT-6 Astra" : segmentation() ? "B · SAM 3" : "Local detector control";
    invalidate("config");
  });
  for (const name of ["sam-threshold", "sam-device"]) field(name).addEventListener(name === "sam-threshold" ? "input" : "change", () => invalidate("config"));
  field("sam-setup").addEventListener("click", () => showRecord("SAM 3 local setup requirements", {
    guide: "docs/sam-preannotation-adapter.md",
    setup: "Configure the isolated Python runtime with IRIS_SAM_PYTHON, the official weights under models/sam3/sam3.pt, and a compatible CUDA GPU. Preparing configurations requires none of these to be installed. This interface does not install or download anything. Availability does not prove a successful model run.",
    local_status: view.providers?.segmentation || null,
  }));
  for (const name of ["image-edge", "reasoning", "output-tokens"]) field(name).addEventListener(name === "output-tokens" ? "input" : "change", () => invalidate("config"));
  field("external-budget").addEventListener("input", () => { field("external-consent").checked = false; update(); });
  field("external-consent").addEventListener("change", update);
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
    if (!view.active) { invalidate("trial"); return; }
    if (!view.loaded) { view.loaded = true; loadCandidates(); loadModels(); loadProviders(); }
    loadList();
  });
  window.addEventListener("iris:session", () => invalidate("trial"));
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
