"use strict";

(() => {
  let instanceCount = 0;
  const finiteTime = (value) => Number.isFinite(value) && value >= 0;
  const clock = (value) => {
    if (!finiteTime(value)) return "Unknown time";
    const centiseconds = Math.round(value * 100);
    if (!Number.isSafeInteger(centiseconds)) return "Unknown time";
    const minutes = Math.floor(centiseconds / 6000);
    const seconds = Math.floor((centiseconds % 6000) / 100);
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(centiseconds % 100).padStart(2, "0")}`;
  };
  const element = (tag, className, text) => {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
  };
  const hasTime = (sample) => finiteTime(sample?.timestamp_seconds);
  const count = (sample) =>
    Number.isInteger(sample.prediction_count)
      ? sample.prediction_count
      : sample.predicted_run_ids?.length || 0;
  const coverage = (sample) =>
    sample.complete ? "complete" : count(sample) ? "partial" : "empty";
  function sampleAtOrBefore(samples, time, selectedId = null) {
    if (!finiteTime(time)) return null;
    let candidate = null;
    for (const sample of samples) {
      if (!hasTime(sample) || sample.timestamp_seconds > time) continue;
      if (
        !candidate ||
        sample.timestamp_seconds > candidate.timestamp_seconds ||
        (sample.timestamp_seconds === candidate.timestamp_seconds &&
          sample.frame_id === selectedId)
      )
        candidate = sample;
    }
    return candidate;
  }
  function samplePosition(time, timestamp) {
    if (!finiteTime(time) || !finiteTime(timestamp))
      return { kind: "unknown", delta: null };
    const delta = time - timestamp;
    return {
      kind: delta === 0 ? "recorded-timestamp" : delta > 0 ? "after" : "before",
      delta,
    };
  }

  function create(root, { onSelectFrame = () => {}, urlFor = (url) => url } = {}) {
    if (!(root instanceof HTMLElement))
      throw new TypeError("A replay mount element is required.");
    const prefix = `comparison-replay-${++instanceCount}`;
    const refs = {};
    const state = {
      detail: null,
      sources: [],
      sourceId: null,
      selectedId: null,
      still: false,
      active: true,
      loadedURL: null,
      mediaKey: null,
      pendingSeek: null,
      mediaError: "",
      seekWarning: "",
      signature: null,
      timedSamples: [],
      holdSample: false,
    };
    const ref = (tag, role, className, text) => {
      const result = element(tag, className, text);
      if (role) {
        result.dataset.replayRole = role;
        refs[role] = result;
      }
      return result;
    };
    const source = () =>
      state.sources.find((item) => item.asset_id === state.sourceId);
    const locate = (id) => {
      for (const item of state.sources) {
        const sample = item.samples.find((entry) => entry.frame_id === id);
        if (sample) return { source: item, sample };
      }
      return null;
    };
    const selected = () => locate(state.selectedId)?.sample;
    const runCount = () =>
      state.detail?.lanes?.length ||
      state.detail?.model_ids?.length ||
      state.detail?.runs?.length ||
      0;
    const coverageLabel = (sample) =>
      `${count(sample)}${runCount() ? `/${runCount()}` : ""} run${runCount() === 1 ? "" : "s"} saved${sample.complete ? "" : count(sample) ? " · partial" : " · no results"}`;
    const sampleLabel = (sample, index) =>
      `${clock(sample.timestamp_seconds)} · sample ${index + 1} · ${coverageLabel(sample)}${sample.image_available === false ? " · image unavailable" : ""}`;
    const currentTime = () =>
      finiteTime(refs.video.currentTime) ? refs.video.currentTime : 0;
    const duration = () => {
      const item = source();
      const recorded = finiteTime(item?.duration_seconds)
        ? item.duration_seconds
        : 0;
      const playable =
        Number.isFinite(refs.video.duration) && refs.video.duration > 0
          ? refs.video.duration
          : 0;
      const last = state.timedSamples.at(-1)?.timestamp_seconds || 0;
      return Math.max(recorded, playable, last, 0.001);
    };

    root.classList.add("comparison-replay");
    root.replaceChildren();
    const header = element("div", "comparison-replay-heading");
    const headingText = element("div");
    headingText.append(
      element("span", "eyebrow", "Video replay"),
      element("h3", "", "Follow the video. Inspect saved frames."),
    );
    header.append(
      headingText,
      ref("span", "sample-count", "comparison-replay-badge"),
    );
    root.append(
      header,
      element(
        "p",
        "comparison-replay-intro",
        "Replay the source video. Model predictions stay on the saved frames below.",
      ),
    );
    const sourceField = element("div", "comparison-replay-source");
    const sourceLabel = element("label", "", "Source video");
    sourceLabel.htmlFor = `${prefix}-source`;
    const sourceSelect = ref("select", "source");
    sourceSelect.id = sourceLabel.htmlFor;
    sourceField.append(sourceLabel, sourceSelect);
    root.append(sourceField, ref("p", "still", "comparison-replay-notice"));
    const body = ref("div", "body", "comparison-replay-body");
    const media = element("div", "comparison-replay-media");
    const video = ref("video", "video");
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.setAttribute(
      "aria-label",
      "Source video playback without detection overlays",
    );
    media.append(video);
    const mediaFallback = ref(
      "div",
      "media-fallback",
      "comparison-replay-media-fallback",
    );
    mediaFallback.append(
      element("span", "comparison-replay-unavailable-mark", "↗"),
      ref("p", "media-error"),
    );
    const retry = ref(
      "button",
      "retry",
      "button button-secondary",
      "Retry video",
    );
    retry.type = "button";
    mediaFallback.append(retry);
    media.append(mediaFallback);
    body.append(media);
    const controls = element("div", "comparison-replay-controls");
    const controlsTitle = element("div", "comparison-replay-controls-heading");
    controlsTitle.append(
      element("span", "eyebrow", "Saved samples"),
      ref("span", "coverage", "comparison-replay-caption"),
    );
    controls.append(controlsTitle);
    const sampleLabelElement = element(
      "label",
      "",
      "Frame shown in the comparison below",
    );
    sampleLabelElement.htmlFor = `${prefix}-sample`;
    const sampleSelect = ref("select", "sample");
    sampleSelect.id = sampleLabelElement.htmlFor;
    controls.append(sampleLabelElement, sampleSelect);
    const navigation = element("div", "comparison-replay-navigation");
    const previous = ref(
      "button",
      "previous",
      "button button-secondary",
      "← Previous sample",
    );
    const next = ref(
      "button",
      "next",
      "button button-secondary",
      "Next sample →",
    );
    previous.type = next.type = "button";
    navigation.append(previous, next);
    controls.append(navigation);
    const followLabel = element("label", "comparison-replay-follow");
    const follow = ref("input", "follow");
    follow.type = "checkbox";
    follow.checked = true;
    followLabel.append(
      follow,
      document.createTextNode("Follow saved samples during playback"),
    );
    controls.append(
      followLabel,
      element(
        "p",
        "comparison-replay-caption",
        "Following advances to the last sampled frame at or before playback. Gaps contain no recorded predictions.",
      ),
    );
    controls.append(ref("p", "untimed", "comparison-replay-caption"));
    body.append(controls);
    root.append(body);
    const timeline = ref("div", "timeline", "comparison-replay-timeline");
    timeline.setAttribute(
      "aria-label",
      "Saved sample positions along the source video",
    );
    const timelineHeading = element(
      "div",
      "comparison-replay-timeline-heading",
    );
    timelineHeading.append(
      element("span", "", "Sparse comparison timeline"),
      ref("span", "range", "comparison-replay-caption"),
    );
    timeline.append(timelineHeading);
    const track = ref("div", "track", "comparison-replay-track");
    track.append(
      ref("span", "playhead", "comparison-replay-playhead"),
      ref("div", "markers", "comparison-replay-markers"),
    );
    refs.playhead.setAttribute("aria-hidden", "true");
    timeline.append(track);
    const legend = element("div", "comparison-replay-legend");
    for (const [kind, text] of [
      ["complete", "All runs saved"],
      ["partial", "Some runs saved"],
      ["empty", "No saved results"],
    ]) {
      const entry = element("span");
      entry.append(element("i", kind), document.createTextNode(text));
      legend.append(entry);
    }
    timeline.append(legend);
    root.append(
      timeline,
      element(
        "p",
        "comparison-replay-timestamp-note",
        "Times are approximate: saved timestamps use the video's nominal frame rate. Variable frame rates can shift alignment. Predictions exist only for samples with saved results; gaps were not analysed.",
      ),
    );
    const context = ref("div", "context", "comparison-replay-context");
    context.setAttribute("role", "group");
    context.setAttribute(
      "aria-label",
      "Playback and frozen comparison context",
    );
    const playback = element("div");
    playback.append(
      element("span", "", "Video playback"),
      ref("strong", "playback-time"),
    );
    const frozen = element("div");
    frozen.append(
      element("span", "", "Saved frame shown below"),
      ref("strong", "saved-time"),
    );
    context.append(playback, frozen, ref("p", "gap"));
    root.append(context);
    root.hidden = true;

    function renderSourceOptions() {
      const options = [new Option("Choose a source video…", "")];
      for (const item of state.sources)
        options.push(
          new Option(
            `${item.filename} · ${item.samples.length} saved samples`,
            item.asset_id,
          ),
        );
      refs.source.replaceChildren(...options);
      refs.source.value = state.still ? "" : state.sourceId || "";
    }

    function renderMediaState() {
      const item = source();
      refs.body.hidden = state.still || !item;
      refs.timeline.hidden = state.still || !item;
      refs["media-fallback"].hidden = !state.mediaError;
      refs.video.hidden = Boolean(state.mediaError);
      refs["media-error"].textContent = state.mediaError;
      refs.retry.hidden = !item?.media_available || !state.mediaError;
      refs.still.hidden = !state.still;
      refs.still.textContent = state.detail?.replay?.still_frame_ids?.includes(
        state.selectedId,
      )
        ? "This saved frame is a still image. Choose a source video to replay its samples; the still image remains available in the comparison below."
        : "No source video is selected for playback. Use the saved comparison below, or choose a source video.";
    }

    function renderSamples() {
      const item = source();
      const samples = item?.samples || [];
      const selectedSample = selected();
      const index = samples.findIndex(
        (sample) => sample.frame_id === state.selectedId,
      );
      const options = samples.map(
        (sample, position) =>
          new Option(sampleLabel(sample, position), sample.frame_id),
      );
      refs.sample.replaceChildren(...options);
      refs.sample.value = index >= 0 ? state.selectedId : "";
      refs.sample.disabled = !samples.length;
      refs.previous.disabled = index <= 0;
      refs.next.disabled = index < 0 || index >= samples.length - 1;
      refs["sample-count"].textContent =
        `${state.sources.reduce((sum, value) => sum + value.samples.length, 0)} saved video samples`;
      const complete = samples.filter((sample) => sample.complete).length;
      const partial = samples.filter(
        (sample) => !sample.complete && count(sample),
      ).length;
      refs.coverage.textContent = `${complete} complete · ${partial} partial · ${samples.length - complete - partial} without results`;
      refs.untimed.textContent =
        samples.length !== state.timedSamples.length
          ? `${samples.length - state.timedSamples.length} sample(s) have no recorded timestamp. They remain available in the list; their video position is unknown.`
          : "";
      refs.untimed.hidden = !refs.untimed.textContent;
      refs.follow.disabled =
        !state.timedSamples.length || Boolean(state.mediaError);
      for (const marker of refs.markers.children) {
        const isSelected =
          marker.dataset.replayFrameId === selectedSample?.frame_id &&
          !state.still;
        marker.classList.toggle("selected", isSelected);
        marker.setAttribute("aria-pressed", String(isSelected));
      }
    }

    function renderTimeline() {
      const item = source();
      const focusedId = document.activeElement?.dataset.replayFrameId;
      refs.markers.replaceChildren();
      refs.range.textContent = `00:00.00 — ${clock(duration())}`;
      for (const [index, sample] of (item?.samples || []).entries()) {
        if (!hasTime(sample)) continue;
        const marker = element(
          "button",
          `comparison-replay-marker ${coverage(sample)}`,
        );
        marker.type = "button";
        marker.dataset.replayFrameId = sample.frame_id;
        marker.style.left = `${(100 * sample.timestamp_seconds) / duration()}%`;
        marker.style.setProperty("--replay-marker-row", index % 2);
        marker.setAttribute("aria-label", sampleLabel(sample, index));
        marker.title = sampleLabel(sample, index);
        marker.addEventListener("click", () =>
          chooseFrame(sample.frame_id, { seek: true, emit: true }),
        );
        refs.markers.append(marker);
        if (sample.frame_id === focusedId)
          marker.focus({ preventScroll: true });
      }
      renderSamples();
      renderContext();
    }

    function renderContext() {
      const sample = selected();
      const time = currentTime();
      const item = source();
      refs["playback-time"].textContent =
        state.still || !item
          ? "Paused · no video selected"
          : state.mediaError
            ? "Video unavailable"
            : `${clock(time)}${refs.video.paused ? " · paused" : ""}`;
      refs["saved-time"].textContent = sample
        ? `${clock(sample.timestamp_seconds)} · frozen sample`
        : state.detail?.replay?.still_frame_ids?.includes(state.selectedId)
          ? "Still image · frozen comparison"
          : "Saved frame · timestamp unavailable";
      let message;
      if (state.still || !sample)
        message =
          "The cards below show the selected saved frame. No video predictions are being generated.";
      else if (!hasTime(sample))
        message =
          "The selected saved frame has no recorded video timestamp. Its comparison remains frozen; playback cannot locate it reliably.";
      else if (state.mediaError)
        message = `The video is unavailable. The saved comparison at ${clock(sample.timestamp_seconds)} can still be inspected below.`;
      else {
        const position = samplePosition(time, sample.timestamp_seconds);
        const delta = position.delta;
        const before =
          state.timedSamples.length &&
          time < state.timedSamples[0].timestamp_seconds;
        message = before ? "Playback is before the first saved sample. " : "";
        message +=
          position.kind === "recorded-timestamp"
            ? "Playback is at the recorded timestamp (approximate). Overlays belong only to its saved image below."
            : `Playback is ${Math.abs(delta) < 0.01 ? "less than 0.01" : Math.abs(delta).toFixed(2)} s ${position.kind} the displayed saved frame. The video between samples has no recorded predictions.`;
      }
      if (sample) message += ` ${coverageLabel(sample)}.`;
      if (sample?.image_available === false)
        message +=
          " This saved image is unavailable; its recorded predictions remain unchanged.";
      if (state.seekWarning) message += ` ${state.seekWarning}`;
      refs.gap.textContent = message;
      refs.playhead.style.left = `${Math.min(100, Math.max(0, (100 * time) / duration()))}%`;
    }

    function mediaURL(item) {
      if (!item?.media_available || !item.media_url) return null;
      try {
        const url = new URL(item.media_url, location.href);
        if (
          url.origin !== location.origin ||
          !url.pathname.startsWith("/api/assets/")
        )
          return null;
        return new URL(urlFor(url.href), location.href).href;
      } catch {
        return null;
      }
    }

    function loadMedia(force = false) {
      const item = source();
      const url = mediaURL(item);
      const key = JSON.stringify([item?.asset_id, item?.media_status, url]);
      if (!force && key === state.mediaKey) return;
      state.mediaKey = key;
      refs.video.pause();
      state.loadedURL = url;
      state.mediaError = "";
      state.seekWarning = "";
      if (url) {
        refs.video.src = url;
        refs.video.load();
      } else {
        refs.video.removeAttribute("src");
        refs.video.load();
        const reasons = {
          missing: "The source video is missing from this workspace.",
          size_mismatch:
            "The source video no longer matches its saved file size.",
          unsafe: "The source video path is unavailable.",
        };
        state.mediaError = `${reasons[item?.media_status] || "The source video is unavailable."} You can still navigate the saved comparison frames below.`;
      }
      renderMediaState();
    }

    function seekSample(sample) {
      state.seekWarning = "";
      if (!hasTime(sample)) {
        state.pendingSeek = null;
        return;
      }
      state.pendingSeek = sample.timestamp_seconds;
      if (!state.loadedURL || refs.video.readyState < 1) return;
      const target = state.pendingSeek;
      state.pendingSeek = null;
      if (
        Number.isFinite(refs.video.duration) &&
        target >= refs.video.duration
      ) {
        state.seekWarning =
          "This saved timestamp is outside the playable video duration; inspect its frozen image below.";
        refs.video.pause();
        return;
      }
      try {
        refs.video.currentTime = target;
      } catch {
        state.seekWarning =
          "This browser could not seek to the saved timestamp. The saved frame remains available below.";
      }
    }

    function useSource(item) {
      const changed = item?.asset_id !== state.sourceId;
      if (changed) {
        refs.video.pause();
        state.holdSample = true;
        state.pendingSeek = null;
        state.sourceId = item?.asset_id || null;
      }
      state.still = false;
      state.timedSamples = (item?.samples || [])
        .filter(hasTime)
        .sort(
          (left, right) => left.timestamp_seconds - right.timestamp_seconds,
        );
      loadMedia();
      renderMediaState();
      refs.source.value = state.sourceId || "";
      return changed;
    }

    function chooseFrame(id, { seek = false, emit = false } = {}) {
      const match = locate(id);
      state.selectedId = id || null;
      state.seekWarning = "";
      if (!match) {
        refs.video.pause();
        state.pendingSeek = null;
        state.still = true;
        refs.source.value = "";
        renderMediaState();
        renderSamples();
        renderContext();
      } else {
        const changed = useSource(match.source);
        if (changed) renderTimeline();
        if (seek) {
          // A manual sample selection pauses playback so its saved context remains visible.
          state.holdSample = true;
          refs.video.pause();
          seekSample(match.sample);
        }
        renderSamples();
        renderContext();
      }
      if (emit) onSelectFrame(id);
    }

    function followPlayback(allowPaused = false) {
      if (
        !state.active ||
        state.still ||
        state.holdSample ||
        !refs.follow.checked ||
        state.mediaError ||
        (refs.video.paused && !allowPaused)
      )
        return;
      const time = currentTime();
      const candidate = sampleAtOrBefore(
        state.timedSamples,
        time,
        state.selectedId,
      );
      const currentSample = selected();
      if (
        candidate &&
        currentSample?.timestamp_seconds === candidate.timestamp_seconds
      )
        return;
      if (candidate && candidate.frame_id !== state.selectedId) {
        state.selectedId = candidate.frame_id;
        renderSamples();
        onSelectFrame(candidate.frame_id);
      }
    }

    function mediaEvent() {
      return state.loadedURL && refs.video.currentSrc === state.loadedURL;
    }
    refs.video.addEventListener("loadedmetadata", () => {
      if (!mediaEvent()) return;
      if (state.pendingSeek !== null)
        seekSample({ timestamp_seconds: state.pendingSeek });
      renderTimeline();
    });
    refs.video.addEventListener("durationchange", () => {
      if (mediaEvent()) renderTimeline();
    });
    for (const name of ["timeupdate", "seeked"])
      refs.video.addEventListener(name, () => {
        if (!mediaEvent()) return;
        followPlayback(name === "seeked");
        renderContext();
      });
    for (const name of ["play", "pause", "ended"])
      refs.video.addEventListener(name, () => {
        if (name === "play") {
          if (!state.active || state.still) refs.video.pause();
          else {
            state.holdSample = false;
            followPlayback();
          }
        }
        renderContext();
      });
    refs.video.addEventListener("error", () => {
      if (!mediaEvent() || !refs.video.error) return;
      refs.video.pause();
      state.mediaError =
        "This browser could not play the source video. The file may be unavailable or use an unsupported codec. Saved samples are still available below.";
      renderMediaState();
      renderSamples();
      renderContext();
    });
    refs.retry.addEventListener("click", () => {
      loadMedia(true);
      seekSample(selected());
      renderContext();
    });
    for (const name of ["pointerdown", "keydown"])
      refs.video.addEventListener(name, () => {
        state.holdSample = false;
      });
    refs.source.addEventListener("change", () => {
      const item = state.sources.find(
        (entry) => entry.asset_id === refs.source.value,
      );
      if (!item) {
        refs.video.pause();
        state.still = true;
        renderMediaState();
        renderContext();
        return;
      }
      useSource(item);
      const sample = item.samples.find(hasTime) || item.samples[0];
      if (sample) chooseFrame(sample.frame_id, { seek: true, emit: true });
      renderTimeline();
    });
    refs.sample.addEventListener("change", () =>
      chooseFrame(refs.sample.value, { seek: true, emit: true }),
    );
    for (const [role, offset] of [
      ["previous", -1],
      ["next", 1],
    ])
      refs[role].addEventListener("click", () => {
        const samples = source()?.samples || [];
        const index = samples.findIndex(
          (sample) => sample.frame_id === state.selectedId,
        );
        if (samples[index + offset])
          chooseFrame(samples[index + offset].frame_id, {
            seek: true,
            emit: true,
          });
      });
    refs.follow.addEventListener("change", () => {
      state.holdSample = false;
      followPlayback(true);
      renderContext();
    });

    function reset() {
      refs.video.pause();
      state.loadedURL = null;
      state.mediaKey = null;
      refs.video.removeAttribute("src");
      refs.video.load();
      state.detail = null;
      state.sources = [];
      state.sourceId = null;
      state.selectedId = null;
      state.still = false;
      state.pendingSeek = null;
      state.mediaError = "";
      state.seekWarning = "";
      state.signature = null;
      state.timedSamples = [];
      state.holdSample = false;
      refs.follow.checked = true;
      root.hidden = true;
    }

    function update(detail, selectedFrameId) {
      const sources = detail?.replay?.sources || [];
      const newComparison = detail?.id !== state.detail?.id;
      if (newComparison) reset();
      state.active = true;
      state.detail = detail;
      state.sources = sources.map((item) => ({
        ...item,
        samples: item.samples || [],
      }));
      root.hidden = !state.sources.length;
      if (root.hidden) {
        refs.video.pause();
        return;
      }
      const signature = JSON.stringify([state.sources, runCount()]);
      const changed = signature !== state.signature;
      state.signature = signature;
      if (changed) renderSourceOptions();
      const desired =
        selectedFrameId ||
        state.selectedId ||
        state.sources[0]?.samples[0]?.frame_id;
      if (desired !== state.selectedId || newComparison)
        chooseFrame(desired, { seek: newComparison });
      else {
        if (!state.still) useSource(source());
        renderMediaState();
        renderContext();
      }
      if (changed) renderTimeline();
    }

    return Object.freeze({
      update,
      selectFrame: (id, options = {}) => chooseFrame(id, options),
      deactivate: () => {
        state.active = false;
        refs.video.pause();
        renderContext();
      },
      reset,
    });
  }

  if (typeof module !== "undefined" && module.exports)
    module.exports = {
      clock,
      sampleAtOrBefore,
      samplePosition,
      coverage,
      count,
    };
  if (typeof window !== "undefined")
    window.IRISComparisonReplay = Object.freeze({ create });
})();
