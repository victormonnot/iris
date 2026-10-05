const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/training-recovery.js");

test("long plans bound periodic writes without silently changing the chosen interval", () => {
  assert.equal(tools.checkpointMinimum(1), 1);
  assert.equal(tools.checkpointMinimum(200), 1);
  assert.equal(tools.checkpointMinimum(201), 2);
  assert.equal(tools.checkpointMinimum(10000), 50);
  assert.equal(tools.checkpointMinimum(NaN), 1);
});

test("a training confirmation binds all optimization settings and the durable plan", () => {
  const payload = { dataset_id: "release", parent_model_id: "parent", scope: "full_model", steps: 10000, learning_rate: 0.001, seed: 42, checkpoint_interval: 50 };
  const preview = {
    fingerprint: "approved", request_id: "request", config: { ...payload }, scope: { id: "full_model" },
    dataset: { id: "release" }, parent: { id: "parent" }, workload: { steps: 10000, device: "cpu" },
  };
  assert.equal(tools.previewMatches(preview, payload), true);
  for (const change of [
    { dataset_id: "other" }, { parent_model_id: "other" }, { scope: "prediction_head_only" },
    { steps: 9999 }, { learning_rate: 0.01 }, { seed: 0 }, { checkpoint_interval: 100 }, { device: "cuda:0" },
  ]) assert.equal(tools.previewMatches(preview, { ...payload, ...change }), false);
  assert.equal(tools.previewMatches({ ...preview, request_id: null }, payload), false);
  assert.equal(tools.previewMatches({ ...preview, fingerprint: null }, payload), false);
  assert.equal(tools.previewMatches({ ...preview, workload: { steps: 10000, device: "cuda" } }, payload), false);
});

test("device discovery keeps CPU usable and never silently replaces a missing selected GPU", () => {
  const reported = { devices: [{ id: "cpu", available: false }, { id: "cuda:0", label: "NVIDIA fixture", available: true }] };
  const choices = tools.deviceChoices(reported, "cuda:1");
  assert.equal(choices.find((item) => item.id === "cpu").available, true);
  assert.equal(choices.find((item) => item.id === "cuda:0").available, true);
  assert.equal(choices.find((item) => item.id === "cuda:1").available, false);
  assert.match(choices.find((item) => item.id === "cuda:1").reason, /Choose a device explicitly/);
  assert.equal(reported.devices.length, 2);
  assert.equal(tools.deviceChoices(null).find((item) => item.id === "cuda").available, false);
  assert.equal(tools.deviceChoices(null)[0].id, "cpu");
  assert.equal(tools.deviceLabel(), "CPU");
  assert.equal(tools.deviceLabel("cuda:1"), "NVIDIA GPU (cuda:1)");
});

test("CUDA previews and continuations bind the exact selected device", () => {
  const payload = { dataset_id: "release", parent_model_id: "parent", scope: "full_model", steps: 50, learning_rate: 0.001, seed: 42, checkpoint_interval: 10, device: "cuda:1" };
  const preview = { fingerprint: "approved", request_id: "request", config: { ...payload },
    scope: { id: payload.scope }, dataset: { id: payload.dataset_id }, parent: { id: payload.parent_model_id }, workload: { steps: 50, device: "cuda:1" } };
  assert.equal(tools.previewMatches(preview, payload), true);
  assert.equal(tools.previewMatches(preview, { ...payload, device: "cuda:0" }), false);
  assert.equal(tools.previewMatches({ ...preview, config: { ...payload, device: "cpu" } }, payload), false);
  const detail = { id: "source", config: payload, recovery: { can_resume: true, checkpoint_id: "state", checkpoint_step: 20, recorded_steps: 25, recomputed_steps: 5 } };
  const continuation = { source_training_id: "source", checkpoint_id: "state", checkpoint_step: 20,
    target_steps: 50, remaining_steps: 30, recorded_steps: 25, recomputed_steps: 5, fingerprint: "approved", config: payload };
  assert.equal(tools.resumeMatches(continuation, detail), true);
  assert.equal(tools.resumeMatches({ ...continuation, config: { ...payload, device: "cpu" } }, detail), false);
  assert.notEqual(tools.recoveryKey(detail), tools.recoveryKey({ ...detail, config: { ...payload, device: "cuda:0" } }));
});

test("lost acknowledgements reconcile exact requests or source attempts without resubmission", () => {
  const rows = [
    { id: "similar", config: { request_id: "another", resume_from: { training_id: "other" } } },
    { id: "new", config: { request_id: "approved" } },
    { id: "continued", config: { resume_from: { training_id: "interrupted", checkpoint_id: "snapshot" } } },
  ];
  assert.equal(tools.findRequest(rows, "approved").id, "new");
  assert.equal(tools.findRequest(rows, "missing"), null);
  assert.equal(tools.findRequest(rows, undefined), null);
  assert.equal(tools.findRequest(null, "approved"), null);
  assert.equal(tools.findResume(rows, "interrupted").id, "continued");
  assert.equal(tools.findResume(rows, "snapshot"), null);
  assert.equal(tools.findResume(rows, undefined), null);
  assert.equal(tools.findResume(null, "interrupted"), null);
});

test("continuation binds the source state and makes repeated recorded work explicit", () => {
  const detail = { id: "source", config: { steps: 1000 }, job: { status: "interrupted" }, recovery: {
    can_resume: true, checkpoint_id: "snapshot", checkpoint_step: 100, recorded_steps: 117, recomputed_steps: 17,
  } };
  const preview = { source_training_id: "source", checkpoint_id: "snapshot", checkpoint_step: 100,
    target_steps: 1000, remaining_steps: 900, recorded_steps: 117, recomputed_steps: 17, fingerprint: "approved" };
  assert.equal(tools.resumeMatches(preview, detail), true);
  for (const change of [
    { source_training_id: "other" }, { checkpoint_id: "newer" }, { checkpoint_step: 110 },
    { target_steps: 2000 }, { remaining_steps: 899 }, { recorded_steps: 118 }, { recomputed_steps: 0 }, { fingerprint: null },
  ]) assert.equal(tools.resumeMatches({ ...preview, ...change }, detail), false);
  assert.equal(tools.resumeMatches(preview, { ...detail, recovery: { ...detail.recovery, can_resume: false } }), false);
  assert.equal(tools.resumeMatches({ ...preview, checkpoint_step: 1000, remaining_steps: 0, recorded_steps: 1000, recomputed_steps: 0 },
    { ...detail, recovery: { ...detail.recovery, checkpoint_step: 1000, recorded_steps: 1000, recomputed_steps: 0 } }), true);
  assert.notEqual(tools.recoveryKey(detail), tools.recoveryKey({ ...detail, job: { status: "running" } }));
  assert.notEqual(tools.recoveryKey(detail), tools.recoveryKey({ ...detail, recovery: { ...detail.recovery, existing_training_id: "child" } }));
});

test("loss pages keep the latest progress bounded and every earlier step reachable", () => {
  const history = Array.from({ length: 10000 }, (_, i) => ({ step: i + 1 }));
  const latest = tools.lossPage(history);
  assert.equal(latest.items.length, 100);
  assert.equal(latest.items[0].step, 9901);
  assert.equal(latest.items.at(-1).step, 10000);
  assert.equal(latest.pages, 100);
  const visited = Array.from({ length: latest.pages }, (_, page) => tools.lossPage(history, page).items).flat();
  assert.equal(new Set(visited.map((item) => item.step)).size, 10000);
  assert.equal(tools.lossPage(history, 1000).items[0].step, 1);
  assert.equal(tools.lossPage(history, -1).page, 0);
  assert.equal(tools.lossPage(history.slice(0, 101), 1).items.length, 1);
  assert.deepEqual(tools.lossPage(null).items, []);
  assert.equal(history[0].step, 1);
});

test("duration is estimated only from finite, ordered, observed active time", () => {
  const history = Array.from({ length: 100 }, (_, i) => ({ step: i + 1, elapsed_seconds: 100 + i * 3 }));
  assert.deepEqual(tools.observedDuration(history, 150), { secondsPerStep: 3, remainingSeconds: 150, observedSteps: 20 });
  assert.equal(tools.observedDuration(history.slice(0, 2), 150), null);
  assert.equal(tools.observedDuration(history, 99), null);
  assert.equal(tools.observedDuration(history, 100).remainingSeconds, 0);
  for (const change of [{ elapsed_seconds: NaN }, { elapsed_seconds: Infinity }, { elapsed_seconds: 1 }, { step: 102 }]) {
    assert.equal(tools.observedDuration([...history.slice(0, -1), { ...history.at(-1), ...change }], 150), null);
  }
  assert.equal(tools.durationText(0), "0 s");
  assert.equal(tools.durationText(61), "2 min");
  assert.equal(tools.durationText(5400), "1.5 h");
  assert.equal(tools.durationText(-1), "Unavailable");
});
