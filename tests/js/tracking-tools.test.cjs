"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const tools = require("../../src/iris/static/tracking-tools.js");
const observation = (id, box = [0, 0, 10, 10], confirmed = true) => ({ track_id: id, box, confirmed });
const frame = (index, observations = [], timestamp = index / 25) => ({ frame_id: `frame-${index}`, frame_index: index, timestamp_seconds: timestamp, observations });

test("sequence creation uses one saved video, sorts source indices, and makes omissions explicit", () => {
  const assets = [{ id: "video", kind: "video", filename: "Source.mp4" }, { id: "image", kind: "image" }];
  const selected = [8, 2, 3].map((index) => ({ id: `f-${index}`, frame_index: index, selected: true, asset_id: "video" }));
  const result = tools.selectedSource(selected, assets);
  assert.deepEqual(result.frames.map((item) => item.frame_index), [2, 3, 8]);
  assert.equal(result.gap_count, 4);
  assert.deepEqual(selected.map((item) => item.frame_index), [8, 2, 3], "sorting does not mutate intake state");
  assert.match(tools.selectedSource(selected.slice(0, 1), assets).error, /at least two/);
  assert.match(tools.selectedSource([...selected, { ...selected[0], asset_id: "image" }], assets).error, /exactly one video/);
  assert.match(tools.selectedSource(selected.map((item) => ({ ...item, asset_id: "image" })), assets).error, /exactly one video/);
  assert.match(tools.selectedSource([...selected, { ...selected[0], id: "duplicate-image" }], assets).error, /one saved image/);
  assert.match(tools.selectedSource(selected.map((item) => ({ ...item, frame_index: null })), assets).error, /source frame index/);
});

test("unknown time never becomes zero; nominal and provided clocks retain their meaning", () => {
  assert.equal(tools.timeLabel(frame(0, [], null)), "Time unknown");
  assert.equal(tools.timeLabel(frame(0, [], 0)), "0.000 s");
  assert.equal(tools.timeLabel(frame(0, [], NaN)), "Time unknown");
  assert.match(tools.clockLabel({ basis: "nominal_fps", fps: 25 }), /Nominal.*estimated, not capture/);
  assert.match(tools.clockLabel({ basis: "provided" }), /caller-declared/);
  assert.match(tools.clockLabel({ basis: "unknown" }), /display cadence/);
  assert.equal(tools.playbackDelay(frame(0), frame(250), { basis: "nominal_fps" }, 2), 5000, "long source gaps are not compressed or filled");
  assert.equal(tools.playbackDelay(frame(0, [], null), frame(250, [], null), { basis: "unknown" }, 2), 250);
  assert.equal(tools.playbackDelay(frame(0), null, { basis: "provided" }), null);
});

test("observation trails break on loss, a missing source frame and a new ID", () => {
  const frames = [frame(0, [observation(1)]), frame(1, [observation(1, [2, 2, 12, 12])]), frame(2),
    frame(3, [observation(1)]), frame(4, [observation(1)]), frame(8, [observation(1)]), frame(9, [observation(1)]),
    frame(10, [observation(2)]), frame(11, [observation(2)])];
  frames[2].predictions = [{ track_id: 1, box: [100, 100, 110, 110] }];
  assert.deepEqual(tools.trails(frames, 8), [
    { track_id: 1, points: [[5, 5], [7, 7]] }, { track_id: 1, points: [[5, 5], [5, 5]] },
    { track_id: 1, points: [[5, 5], [5, 5]] }, { track_id: 2, points: [[5, 5], [5, 5]] },
  ]);
  assert.deepEqual(tools.trails(frames, 2), [{ track_id: 1, points: [[5, 5], [7, 7]] }], "predictions never become observed trail points");
  assert.equal(tools.trails(frames, 1, 1).length, 0, "a single observation is not a continuous trail");
});

test("continuity events describe saved observations without inferring real identity or visibility", () => {
  const frames = [frame(0, [observation(1)]), frame(1), frame(2, [observation(1)]), frame(5, [observation(2, [0, 0, 10, 10], false)])];
  const events = tools.events(frames);
  assert.equal(events.filter((event) => event.kind === "disappearance").length, 2);
  const returned = events.find((event) => event.kind === "return");
  assert.equal(returned.track_id, 1); assert.equal(returned.position, 2); assert.match(returned.text, /real identity is unverified/);
  assert.match(events.find((event) => event.kind === "new_id" && event.track_id === 2).text, /unconfirmed.*does not establish.*identity switch/);
  assert.match(events.find((event) => event.kind === "source_gap").text, /2 missing source frames.*no tracker updates/);
  assert.equal(events.some((event) => event.kind === "occluded" || event.kind === "identity_switch"), false);
  const otherLane = tools.events([frame(0, [observation(1)]), frame(1, [observation(1)])]);
  assert.equal(otherLane.filter((event) => event.kind === "return").length, 0, "equal numeric IDs in another lane share no history");
});

test("overlap is inspectable ambiguity and never an asserted crossing or identity error", () => {
  const touching = frame(0, [observation(1), observation(2, [10, 0, 20, 10])]);
  const overlapping = frame(1, [observation(1), observation(2, [9, 0, 19, 10])]);
  const events = tools.events([touching, overlapping]);
  assert.equal(events.filter((event) => event.kind === "overlap").length, 1);
  assert.equal(events.find((event) => event.kind === "overlap").position, 1);
  assert.match(events.find((event) => event.kind === "overlap").text, /possible crossing or occlusion.*does not prove an identity error/);
});
