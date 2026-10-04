"use strict";

(() => {
  const field = (name) => $(`#video-review-${name}`);
  const dialog = field("dialog");
  const review = {
    asset: null, sessionId: null, context: 0, generation: 0,
    catalog: null, catalogLoading: false, catalogRequest: 0,
    history: [], historyLoading: false, historyRequest: 0,
    detail: null, detailLoading: false, detailRequest: 0,
    operation: null, operationToken: 0, selected: new Set(),
    plan: null, planKey: null, configKey: null, imagesKey: null,
    imagesPending: 0, imagesFailed: 0, expiryTimer: null, jobKey: "",
  };
  const active = (job) => job && ["queued", "running"].includes(job.status);
  const current = (context) => dialog.open && context === review.context &&
    state.sessionId === review.sessionId;
  const assetURL = (suffix = "") =>
    `/api/assets/${encodeURIComponent(review.asset.id)}/video-reviews${suffix}`;
  const detailURL = (suffix = "") =>
    `/api/video-reviews/${encodeURIComponent(review.detail.id)}${suffix}`;
  const provider = () => review.catalog?.providers?.find((item) => item.id === field("provider").value);
  const model = () => provider()?.models?.find((item) => item.id === field("model").value);
  const modelReady = () => ["ready", "configured"].includes(model()?.status);
  const external = () => review.detail?.config?.provider?.provider === "alibaba";
  const cost = () => review.detail?.config?.estimated_cost?.upper_bound_usd;
  const validCost = () => typeof cost() === "number" && Number.isFinite(cost()) && cost() >= 0;
  const money = (value) => `$${Number(value).toLocaleString("en-US", { maximumFractionDigits: 8 })} USD`;
  const expired = () => review.detail?.status === "expired" ||
    !Number.isFinite(Date.parse(review.detail?.expires_at)) ||
    Date.parse(review.detail.expires_at) <= Date.now();
  const imagesReady = () => Boolean(review.detail?.images?.length) &&
    review.imagesPending === 0 && review.imagesFailed === 0;

  function error(failure) {
    field("error").textContent = failure?.message || failure || "";
    field("error").hidden = !failure;
  }

  function configuration() {
    return {
      provider: field("provider").value,
      model: field("model").value,
      start_seconds: Number(field("start").value),
      end_seconds: field("end").value === "" ? null : Number(field("end").value),
      sample_count: Number(field("samples").value),
      instructions: field("instructions").value.trim(),
    };
  }

  function extraction() {
    return {
      passage_ids: (review.detail?.result?.passages || [])
        .filter((passage) => review.selected.has(passage.id)).map((passage) => passage.id),
      frames_per_passage: Number(field("frames").value),
      context_seconds: Number(field("context").value),
      coverage_frames: Number(field("coverage").value),
    };
  }

  function invalidatePlan() {
    review.plan = null;
    review.planKey = null;
    field("extraction-plan").hidden = true;
    field("plan-summary").textContent = "";
    field("plan-ranges").replaceChildren();
    updateControls();
  }

  function invalidateSettings() {
    review.generation++;
    review.detailRequest++;
    review.detailLoading = false;
    review.detail = null;
    review.configKey = null;
    review.imagesKey = null;
    review.selected.clear();
    clearTimeout(review.expiryTimer);
    review.expiryTimer = null;
    field("consent").checked = false;
    field("detail").hidden = true;
    field("images").replaceChildren();
    field("history").value = "";
    field("status").textContent = "Prepare a new preview after changing the review settings.";
    invalidatePlan();
    error(null);
  }

  function updateControls() {
    const busy = Boolean(review.operation || review.detailLoading);
    for (const input of field("form").querySelectorAll("input, select, textarea"))
      input.disabled = busy || review.catalogLoading;
    field("provider").disabled = busy || review.catalogLoading || !review.catalog?.providers?.length;
    field("model").disabled = busy || review.catalogLoading || !provider()?.models?.length;
    field("refresh-models").disabled = busy || review.catalogLoading;
    field("preview").disabled = busy || review.catalogLoading || !modelReady();
    field("preview").textContent = review.operation === "preview" ? "Preparing images…" : "Prepare image preview";
    field("history").disabled = busy || review.catalogLoading || review.historyLoading || !review.history.length;
    field("refresh-history").disabled = busy || review.catalogLoading || review.historyLoading;
    const runnable = review.detail?.status === "preview" && !review.detail.job && !expired() &&
      review.detail.config.provider.provider === field("provider").value &&
      review.detail.config.provider.model === field("model").value &&
      review.configKey === JSON.stringify(configuration()) && imagesReady() && modelReady();
    field("consent").disabled = busy || !runnable || !validCost();
    field("run").disabled = busy || !runnable || (external() && (!validCost() || !field("consent").checked));
    field("run").textContent = review.operation === "run" ? "Starting review…" :
      external() ? "Send approved images and analyze" : "Analyze locally";
    field("cancel-job").disabled = busy || !active(review.detail?.job) || Boolean(review.detail?.job?.cancel_requested);
    for (const input of field("extract-form").querySelectorAll("input")) input.disabled = busy;
    for (const input of field("passages").querySelectorAll("input")) input.disabled = busy;
    field("plan").disabled = busy || review.detail?.status !== "succeeded" || !review.selected.size;
    field("plan").textContent = review.operation === "plan" ? "Planning extraction…" : "Preview extraction plan";
    field("extract").disabled = busy || !review.plan?.planned_count || !review.selected.size ||
      review.planKey !== JSON.stringify(extraction());
    field("extract").textContent = review.operation === "extract" ? "Queuing extraction…" : "Extract chosen passages";
    if (["preview", "expired"].includes(review.detail?.status)) {
      field("expiry").textContent = expired()
        ? "This preview has expired. Prepare a new preview before analyzing."
        : `Preview valid until ${new Date(review.detail.expires_at).toLocaleString()}. Analysis requires this explicit action.`;
    }
  }

  function renderProviderStatus() {
    const choice = model();
    field("provider-status").textContent = review.catalogLoading ? "Checking available models…" :
      !choice ? "No model is available. Configure a local model or an API key, then refresh." :
        modelReady() ? `${provider().local ? "Local model ready" : "API key configured · connection not tested"} · ${choice.label || choice.id}${choice.endpoint ? ` · ${choice.endpoint}` : ""}` :
          `Unavailable · ${choice.reason || choice.status || "Model is not configured"}. Regular frame extraction remains available.`;
    updateControls();
  }

  function renderModels(preferred) {
    field("model").replaceChildren();
    for (const entry of provider()?.models || [])
      field("model").append(new Option(entry.label || entry.id, entry.id));
    if (provider()?.models?.some((entry) => entry.id === preferred)) field("model").value = preferred;
    renderProviderStatus();
  }

  function renderProviders(preferredProvider, preferredModel) {
    field("provider").replaceChildren();
    for (const entry of review.catalog?.providers || [])
      field("provider").append(new Option(entry.name || entry.id, entry.id));
    if (review.catalog?.providers?.some((entry) => entry.id === preferredProvider))
      field("provider").value = preferredProvider;
    renderModels(preferredModel);
  }

  async function loadCatalog() {
    const context = review.context;
    const request = ++review.catalogRequest;
    const previous = configuration();
    review.catalogLoading = true;
    renderProviderStatus();
    try {
      const result = await api("/api/annotation-providers");
      if (!current(context) || request !== review.catalogRequest) return;
      review.catalog = result;
      renderProviders(previous.provider || result.default_provider, previous.model || result.default_model);
    } catch (failure) {
      if (!current(context) || request !== review.catalogRequest) return;
      review.catalog = null;
      renderProviders();
      error(failure);
    } finally {
      if (current(context) && request === review.catalogRequest) {
        review.catalogLoading = false;
        renderProviderStatus();
      }
    }
  }

  async function loadHistory() {
    const context = review.context;
    const request = ++review.historyRequest;
    review.historyLoading = true;
    updateControls();
    try {
      const rows = await api(assetURL());
      if (!current(context) || request !== review.historyRequest) return;
      review.history = rows;
      field("history").replaceChildren(new Option("Choose a saved review…", ""));
      for (const item of [...rows].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))) {
        const providerConfig = item.config?.provider;
        field("history").append(new Option(
          `${new Date(item.created_at).toLocaleString()} · ${item.status} · ${providerConfig?.model || item.id.slice(0, 8)}`,
          item.id,
        ));
      }
      if (rows.some((item) => item.id === review.detail?.id)) field("history").value = review.detail.id;
    } catch (failure) {
      if (current(context) && request === review.historyRequest) error(failure);
    } finally {
      if (current(context) && request === review.historyRequest) {
        review.historyLoading = false;
        updateControls();
      }
    }
  }

  function imageURL(item) {
    const url = new URL(item.url, window.location.href);
    if (url.origin !== window.location.origin || !url.pathname.startsWith("/api/video-reviews/"))
      throw new Error("Preview images must be stored by IRIS. Prepare a new preview.");
    return projectURL(url.href);
  }

  function imageStatus() {
    field("image-status").textContent = review.imagesFailed
      ? "Some preview images could not be loaded. Reload this review from history before analyzing."
      : review.imagesPending ? `Loading ${review.imagesPending} preview image(s)…`
        : "All prepared images are shown above. No images have been sent by opening this preview.";
    updateControls();
  }

  function renderImages(record) {
    const context = review.context;
    const key = `${record.id}:${++review.generation}`;
    review.imagesKey = key;
    review.imagesPending = record.images?.length || 0;
    review.imagesFailed = 0;
    field("images").replaceChildren();
    for (const item of record.images || []) {
      const figure = node("figure", "video-review-image");
      figure.dataset.sampleId = item.id;
      const img = node("img");
      img.alt = `Sample ${item.id} at approximately ${timestamp(item.timestamp_seconds)}`;
      const complete = (failed) => {
        if (!current(context) || key !== review.imagesKey) return;
        review.imagesPending--;
        if (failed) review.imagesFailed++;
        imageStatus();
      };
      img.addEventListener("load", () => complete(false), { once: true });
      img.addEventListener("error", () => complete(true), { once: true });
      try { img.src = imageURL(item); } catch { complete(true); }
      figure.append(img, node("figcaption", "field-hint",
        `${item.id} · ${timestamp(item.timestamp_seconds)} · ${item.width} × ${item.height} · ${formatBytes(item.size_bytes)}`));
      field("images").append(figure);
    }
    imageStatus();
  }

  function renderPassages(record) {
    const passages = record.result?.passages || [];
    const ids = new Set(passages.map((item) => item.id));
    for (const id of review.selected) if (!ids.has(id)) review.selected.delete(id);
    field("passages").replaceChildren();
    field("empty").hidden = passages.length > 0;
    field("extract-form").hidden = !passages.length;
    for (const passage of passages) {
      const card = node("article", "video-review-passage");
      const label = node("label", "checkbox-label");
      const input = node("input");
      input.type = "checkbox";
      input.value = passage.id;
      input.checked = review.selected.has(passage.id);
      const title = `${timestamp(passage.start_seconds)} – ${timestamp(passage.end_seconds)}`;
      label.append(input, node("strong", "", title));
      input.addEventListener("change", () => {
        if (input.checked) review.selected.add(passage.id); else review.selected.delete(passage.id);
        invalidatePlan();
      });
      card.append(label, node("p", "small", passage.reason), node("p", "field-hint",
        `Model-reported uncertainty: ${passage.uncertainty} · anchors ${passage.start_sample_id}–${passage.end_sample_id} · end time exclusive`));
      const anchors = node("div", "video-review-anchors");
      for (const id of new Set([passage.start_sample_id, passage.end_sample_id])) {
        const source = record.images?.find((item) => item.id === id);
        if (!source) continue;
        const figure = node("figure");
        const img = node("img");
        img.alt = `Passage anchor ${id} at ${timestamp(source.timestamp_seconds)}`;
        try { img.src = imageURL(source); } catch { /* The main gallery reports image errors. */ }
        figure.append(img, node("figcaption", "field-hint", `${id} · ${timestamp(source.timestamp_seconds)}`));
        anchors.append(figure);
      }
      card.append(anchors);
      field("passages").append(card);
    }
  }

  function showRecord(record, reset = false) {
    if (record.asset_id !== review.asset.id) throw new Error("This review belongs to another video.");
    const changed = reset || review.detail?.id !== record.id;
    review.detail = record;
    if (changed) {
      review.selected.clear();
      field("consent").checked = false;
      field("extract-form").reset();
      invalidatePlan();
      const config = record.config;
      renderProviders(config.provider.provider, config.provider.model);
      if (field("provider").value !== config.provider.provider) {
        field("provider").append(new Option(`${config.provider.provider} (unavailable)`, config.provider.provider));
        field("provider").value = config.provider.provider;
        renderModels(config.provider.model);
      }
      if (field("model").value !== config.provider.model) {
        field("model").append(new Option(`${config.provider.model} (unavailable)`, config.provider.model));
        field("model").value = config.provider.model;
      }
      renderProviderStatus();
      field("start").value = config.plan.start_seconds;
      field("end").value = config.plan.end_seconds;
      field("samples").value = config.sample_count || config.plan.max_frames || config.plan.planned_count;
      field("instructions").value = config.instructions || "";
      review.configKey = JSON.stringify(configuration());
      renderImages(record);
    }
    field("detail").hidden = false;
    field("history").value = record.id;
    field("record-status").textContent = record.status;
    field("preview-summary").textContent = `${record.images.length} exact JPEG images · ${timestamp(record.config.plan.first_timestamp_seconds)} to ${timestamp(record.config.plan.last_timestamp_seconds)} · ${record.config.provider.model}`;
    field("run-actions").hidden = !["preview", "expired"].includes(record.status);
    field("external").hidden = !external();
    if (external()) {
      field("destination").textContent = `${record.config.provider.model} · ${record.config.provider.endpoint}`;
      field("cost").textContent = validCost()
        ? `Maximum approved charge for this request: ${money(cost())}. The provider's bill is authoritative.`
        : "No valid cost ceiling is available. Prepare a new preview before sending.";
      field("consent-label").textContent = `I approve sending these ${record.images.length} images and my instructions to Alibaba Cloud and a charge of up to ${validCost() ? money(cost()) : "the displayed limit"} for this review.`;
    }
    clearTimeout(review.expiryTimer);
    if (record.status === "preview" && !expired()) review.expiryTimer = setTimeout(updateControls,
      Math.min(2147483647, Math.max(0, Date.parse(record.expires_at) - Date.now() + 20)));
    field("job").hidden = !record.job;
    if (record.job) {
      field("job-status").textContent = `${record.job.status} · ${record.job.message || "Video passage review"}${record.job.cancel_requested ? " · cancellation requested" : ""}`;
      field("progress").value = Math.max(0, Math.min(1, record.job.progress || 0));
      field("progress").hidden = !active(record.job);
      field("cancel-job").hidden = !active(record.job);
    }
    field("result").hidden = record.status !== "succeeded" || !record.result;
    if (record.status === "succeeded" && record.result) {
      field("summary").textContent = record.result.summary || "";
      renderPassages(record);
    }
    field("provenance").textContent = JSON.stringify({
      id: record.id, created_at: record.created_at, config: record.config,
      images: record.images.map(({ url, ...image }) => image),
      prompt: record.prompt, metadata: record.metadata, raw_response: record.raw_response,
    }, null, 2);
    if (record.error || record.job?.error) error(record.error || record.job.error);
    field("status").textContent = record.status === "succeeded" ? "Review complete. Select the passages you want to inspect more closely." :
      ["failed", "cancelled", "interrupted"].includes(record.status)
        ? `Review ${record.status}. No passages were extracted. A new analysis requires a new preview and explicit action.` : "";
    updateControls();
  }

  async function loadDetail(id, reset = false) {
    const context = review.context;
    const request = ++review.detailRequest;
    review.detailLoading = true;
    updateControls();
    try {
      const result = await api(`/api/video-reviews/${encodeURIComponent(id)}`);
      if (!current(context) || request !== review.detailRequest) return;
      showRecord(result, reset);
    } catch (failure) {
      if (current(context) && request === review.detailRequest) error(failure);
    } finally {
      if (current(context) && request === review.detailRequest) {
        review.detailLoading = false;
        updateControls();
      }
    }
  }

  async function operation(name, callback) {
    if (review.operation || review.detailLoading) return;
    const context = review.context;
    const token = ++review.operationToken;
    review.operation = name;
    error(null);
    updateControls();
    try { await callback(context); } catch (failure) {
      if (current(context) && token === review.operationToken) error(failure);
    } finally {
      if (current(context) && token === review.operationToken) {
        review.operation = null;
        updateControls();
      }
    }
  }

  async function preparePreview(event) {
    event.preventDefault();
    if (!field("form").reportValidity() || !modelReady()) return;
    const payload = configuration();
    invalidateSettings();
    await operation("preview", async (context) => {
      const record = await api(assetURL("/preview"), { method: "POST", body: JSON.stringify(payload) });
      if (!current(context)) return;
      showRecord(record, true);
      await loadHistory();
    });
  }

  async function runReview() {
    updateControls();
    if (field("run").disabled) return;
    const id = review.detail.id;
    const endpoint = detailURL("/run");
    const payload = { allow_external: external(), max_cost_usd: external() ? cost() : null };
    await operation("run", async (context) => {
      try {
        await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
      } catch (failure) {
        if (!payload.allow_external) throw failure;
        let saved = null;
        try { saved = await api(`/api/video-reviews/${encodeURIComponent(id)}`); }
        catch { /* An unreadable record is not evidence that no request was sent. */ }
        if (!current(context)) return;
        field("consent").checked = false;
        if (saved?.job?.id || saved?.job_id) {
          showRecord(saved, true);
          await refreshJobs().catch(() => {});
          notify("This approved video review is already recorded. No second request was sent; inspect its saved task for delivery and results.");
          window.dispatchEvent(new CustomEvent("iris:job-open", { detail: { job_id: saved.job?.id || saved.job_id } }));
          return;
        }
        invalidateSettings();
        throw new Error(`${failure.message} IRIS could not confirm whether the approval was queued. Check Project jobs and review history before preparing another request; another request may incur another charge. No request was repeated automatically.`);
      }
      await refreshJobs();
      if (!current(context)) return;
      field("consent").checked = false;
      await loadDetail(id);
      await loadHistory();
    });
  }

  async function cancelReview() {
    if (field("cancel-job").disabled || !review.detail?.job) return;
    const id = review.detail.id;
    const jobId = review.detail.job.id;
    await operation("cancel", async (context) => {
      await api(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
      await refreshJobs();
      if (current(context)) await loadDetail(id);
    });
  }

  async function planExtraction(event) {
    event.preventDefault();
    if (!field("extract-form").reportValidity() || !review.selected.size) return;
    const payload = extraction();
    const endpoint = detailURL("/extract/preview");
    const id = review.detail.id;
    invalidatePlan();
    await operation("plan", async (context) => {
      const plan = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
      if (!current(context) || review.detail?.id !== id || JSON.stringify(extraction()) !== JSON.stringify(payload)) return;
      review.plan = plan;
      review.planKey = JSON.stringify(payload);
      field("extraction-plan").hidden = false;
      field("plan-summary").textContent = `${plan.planned_count} unique planned positions after merging overlaps · ${timestamp(plan.first_timestamp_seconds)} to ${timestamp(plan.last_timestamp_seconds)} · ${payload.coverage_frames} extra coverage images requested. This preview adds no frames.`;
      field("plan-ranges").replaceChildren();
      for (const range of plan.ranges || []) field("plan-ranges").append(node("li", "field-hint",
        `${range.passage_id || "Passage"}: ${timestamp(range.start_seconds)} – ${timestamp(range.end_seconds)} including context`));
    });
  }

  async function extractPassages() {
    updateControls();
    if (field("extract").disabled) return;
    const endpoint = detailURL("/extract");
    const payload = extraction();
    await operation("extract", async (context) => {
      await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
      await refreshJobs();
      if (!current(context)) return;
      invalidatePlan();
      field("status").textContent = "Extraction queued. Review the new images in Data intake when the job completes; they remain unselected and unannotated.";
      notify("Passage extraction queued. Review the new frames in Data intake when it completes.");
    });
  }

  async function open(asset) {
    if (asset.kind !== "video" || !state.sessionId) return;
    review.context++;
    review.asset = asset;
    review.sessionId = state.sessionId;
    review.operation = null;
    review.detailLoading = false;
    review.catalogLoading = false;
    review.historyLoading = false;
    review.history = [];
    review.jobKey = "";
    field("form").reset();
    field("extract-form").reset();
    field("source").textContent = `${asset.filename} · ${timestamp(asset.metadata?.duration_seconds)}`;
    field("history").replaceChildren(new Option("Choose a saved review…", ""));
    invalidateSettings();
    field("status").textContent = "Choose a model and prepare the images before requesting a review.";
    dialog.showModal();
    dialog.scrollTop = 0;
    await Promise.allSettled([loadCatalog(), loadHistory()]);
  }

  field("form").addEventListener("submit", preparePreview);
  field("form").addEventListener("input", () => { invalidateSettings(); renderProviderStatus(); });
  field("provider").addEventListener("change", () => { invalidateSettings(); renderModels(); });
  field("model").addEventListener("change", () => { invalidateSettings(); renderProviderStatus(); });
  field("refresh-models").addEventListener("click", () => { invalidateSettings(); loadCatalog(); });
  field("refresh-history").addEventListener("click", async () => {
    await loadHistory();
    if (review.detail) await loadDetail(review.detail.id, true);
  });
  field("history").addEventListener("change", () => {
    const id = field("history").value;
    invalidateSettings();
    if (id) loadDetail(id, true);
  });
  field("consent").addEventListener("change", updateControls);
  field("run").addEventListener("click", runReview);
  field("cancel-job").addEventListener("click", cancelReview);
  field("extract-form").addEventListener("input", invalidatePlan);
  field("extract-form").addEventListener("submit", planExtraction);
  field("extract").addEventListener("click", extractPassages);
  field("close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    if (dialog.open) return;
    review.context++;
    review.catalogRequest++;
    review.historyRequest++;
    review.operationToken++;
    review.operation = null;
    invalidateSettings();
  });
  window.addEventListener("iris:session", () => { if (dialog.open) dialog.close(); });
  window.addEventListener("iris:video-review", (event) => open(event.detail.asset));
  window.addEventListener("iris:jobs", () => {
    if (!dialog.open || !review.detail?.job || review.operation || review.detailLoading) return;
    const job = state.jobs.find((item) => item.id === review.detail.job.id);
    const key = JSON.stringify(job);
    if (job && key !== review.jobKey) {
      review.jobKey = key;
      loadDetail(review.detail.id);
    }
  });
})();
