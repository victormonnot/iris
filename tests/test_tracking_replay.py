"""Cache reuse, source evidence and complete-only standalone tracker reports."""

import json
import sqlite3
import sys
from copy import deepcopy

import pytest
from test_temporal_api import publish_sequence
from test_temporal_detection_api import client as client
from test_temporal_detection_api import create, sequence
from test_temporal_detections import execute

from iris import cli
from iris import temporal_detections as caches
from iris import tracking_replay as replay
from iris.tracking_contracts import FRAME_SCHEMA, make_profile, validate_tracking_frame


@pytest.fixture
def completed(client, tmp_path):
    cache = create(client, sequence(client, tmp_path))
    store = client.app.state.store
    execute(store, cache)
    return store, cache


class SyntheticTracker:
    """Transparent output adapter; association quality is tested separately."""

    def __init__(self, profile, *, pass_index, calls, mismatch=False, failure=None):
        self.profile = profile
        self.pass_index = pass_index
        self.calls = calls
        self.mismatch = mismatch
        self.failure = failure
        self.metadata = {"synthetic": True, "algorithm": profile["algorithm"]}

    def reset(self, sequence_id):
        self.sequence_id = sequence_id
        self.position = 0
        self.calls.append(("reset", self.pass_index, sequence_id))

    def verify_runtime(self):
        pass

    def update(self, frame, *, image=None):
        self.position += 1
        self.calls.append(("update", self.pass_index, deepcopy(frame), image))
        if self.failure:
            self.failure(self.pass_index, self.position)
        observed, unassigned = [], []
        for detection in frame["detections"]:
            if detection["score"] > self.profile["high_threshold"]:
                observed.append(
                    {
                        **deepcopy(detection),
                        "track_id": detection["detection_index"]
                        + 1
                        + (self.pass_index if self.mismatch else 0),
                        "confirmed": True,
                        "estimated_box": list(detection["box"]),
                    }
                )
            else:
                unassigned.append(
                    {
                        **deepcopy(detection),
                        "reason": "below_low_threshold"
                        if detection["score"] <= self.profile["low_threshold"]
                        else "unmatched_low_confidence",
                    }
                )
        gmc = self.profile["gmc_method"]
        result = {
            "schema": FRAME_SCHEMA,
            "sequence_id": self.sequence_id,
            **{
                key: frame[key]
                for key in ("frame_id", "frame_index", "timestamp_seconds", "input_size")
            },
            "update_index": self.position,
            "observations": observed,
            "predictions": [],
            "unassigned": unassigned,
            "gmc": {
                "method": gmc,
                "status": "disabled" if gmc == "none" else "initialized",
                "matrix": None if gmc == "none" else [[1, 0, 0], [0, 1, 0]],
                "downscale": 2,
            },
            "timing": {
                "gmc_ms": 0,
                "association_ms": 1 + self.pass_index,
                "total_ms": 2 + self.pass_index,
            },
        }
        return validate_tracking_frame(result, frame, self.profile)


def synthetic(monkeypatch, *, mismatch=False, failure=None):
    calls, trackers = [], []

    def factory(profile):
        tracker = SyntheticTracker(
            profile,
            pass_index=len(trackers),
            calls=calls,
            mismatch=mismatch,
            failure=failure,
        )
        trackers.append(tracker)
        return tracker

    monkeypatch.setattr(replay, "_factory", factory)
    return calls, trackers


def forbidden(*_args, **_kwargs):
    pytest.fail("Replay must not load a detector or request unnecessary source pixels")


@pytest.mark.parametrize("algorithm", ["bytetrack", "botsort"])
def test_complete_cache_replays_without_media_or_detector_and_preserves_native_indices(
    completed, monkeypatch, algorithm
):
    store, cache = completed
    calls, trackers = synthetic(monkeypatch)
    before = store.list("temporal_detection_frames", cache_id=cache["id"])
    jobs = store.list("jobs")
    for frame in store.list("frames"):
        store.artifact_path(frame["path"]).unlink()
    monkeypatch.setattr(replay, "_source_image", forbidden)
    monkeypatch.setattr(caches, "prepare_detector", forbidden)
    profile = make_profile(algorithm, class_ids=[1], gmc_method="none")
    report = replay.replay_detection_cache(store, cache["id"], profile=profile)
    assert report["complete"] is True
    assert report["repeatability"]["status"] == "observed_match"
    assert report["profile"] == profile
    assert report["cache"]["config"] == cache["config"]
    assert report["input_filter"] == {"min_score": 0.001, "class_ids": [1]}
    assert 3 in report["excluded_class_ids"]
    assert len(trackers) == 2
    assert len([call for call in calls if call[0] == "reset"]) == 2
    frames = report["passes"][0]["frames"]
    assert frames[0]["observations"][0]["detection_index"] == 2
    assert frames[0]["unassigned"][0]["detection_index"] == 0
    assert frames[0]["unassigned"][0]["score"] == 0.02
    assert frames[1]["observations"] == frames[1]["unassigned"] == []
    assert [frame["update_index"] for frame in frames] == [1, 2, 3]
    assert [frame["timestamp_seconds"] for frame in frames] == [0.0, 0.1, 0.2]
    assert report["passes"][0]["timing"]["source_image_read_ms"] == 0
    assert report["passes"][0]["frames"][0]["timing"] != report["passes"][1]["frames"][0]["timing"]
    assert report["passes"][0]["semantic_sha256"] == report["passes"][1]["semantic_sha256"]
    assert store.list("temporal_detection_frames", cache_id=cache["id"]) == before
    assert store.list("jobs") == jobs


def test_default_profile_uses_all_known_classes_and_sorting_does_not_renumber(
    completed, monkeypatch
):
    store, cache = completed
    synthetic(monkeypatch)
    report = replay.replay_detection_cache(store, cache["id"], repeats=1)
    assert report["profile"]["class_ids"] == sorted(
        item["id"] for item in cache["config"]["detector"]["classes"]
    )
    assert report["repeatability"]["status"] == "not_checked"
    restricted = replay.replay_detection_cache(store, cache["id"], class_ids=[3, 1], repeats=1)
    assert restricted["profile"]["class_ids"] == [1, 3]
    assert [d["detection_index"] for d in restricted["passes"][0]["frames"][0]["unassigned"]] == [
        0,
        1,
    ]


def test_timing_changes_do_not_mask_semantic_mismatch(completed, monkeypatch):
    store, cache = completed
    synthetic(monkeypatch, mismatch=True)
    report = replay.replay_detection_cache(store, cache["id"], class_ids=[1])
    assert report["complete"] is True
    assert report["repeatability"]["status"] == "observed_mismatch"
    assert len(set(report["repeatability"]["semantic_sha256"])) == 2


@pytest.mark.parametrize("during_pass", [False, True])
def test_runtime_drift_cannot_publish_a_repeatability_claim(
    completed, monkeypatch, tmp_path, during_pass
):
    store, cache = completed
    _, trackers = synthetic(monkeypatch)
    original_factory = replay._factory

    def changing_factory(profile):
        tracker = original_factory(profile)
        if during_pass:
            tracker.failure = lambda *_args: tracker.metadata.update({"runtime": "changed"})
        elif len(trackers) == 2:
            tracker.metadata["runtime"] = "changed"
        return tracker

    monkeypatch.setattr(replay, "_factory", changing_factory)
    destination = tmp_path / "runtime-drift.json"
    with pytest.raises(ValueError, match="runtime or source provenance changed"):
        replay.replay_to_file(store, cache["id"], destination)
    assert not destination.exists()


@pytest.mark.parametrize("repeats", [0, 6, True, 2.0, None])
def test_bad_repeats_fail_before_adapter(completed, monkeypatch, repeats):
    store, cache = completed
    monkeypatch.setattr(replay, "_factory", forbidden)
    with pytest.raises(ValueError, match="repeats"):
        replay.replay_detection_cache(store, cache["id"], repeats=repeats)


@pytest.mark.parametrize("classes", [[], [1, 1], [True], [0], [1000], "1"])
def test_unknown_duplicate_or_invalid_classes_fail_before_adapter(completed, monkeypatch, classes):
    store, cache = completed
    monkeypatch.setattr(replay, "_factory", forbidden)
    with pytest.raises(ValueError, match="Class IDs"):
        replay.replay_detection_cache(store, cache["id"], class_ids=classes)


def test_profile_class_selection_is_complete_not_an_override(completed, monkeypatch):
    store, cache = completed
    monkeypatch.setattr(replay, "_factory", forbidden)
    with pytest.raises(ValueError, match="not both"):
        replay.replay_detection_cache(
            store, cache["id"], profile=make_profile("bytetrack", class_ids=[1]), class_ids=[3]
        )
    with pytest.raises(ValueError, match="native labels"):
        replay.replay_detection_cache(
            store, cache["id"], profile=make_profile("bytetrack", class_ids=[999])
        )


def test_saved_floor_and_partial_cache_refuse_before_adapter(client, tmp_path, monkeypatch):
    store = client.app.state.store
    source = sequence(client, tmp_path)
    high_floor = create(client, source, min_score=0.2)
    execute(store, high_floor)
    monkeypatch.setattr(replay, "_factory", forbidden)
    with pytest.raises(ValueError, match="lower-floor detector cache"):
        replay.replay_detection_cache(store, high_floor["id"], class_ids=[1])
    partial = create(client, source)
    execute(
        store,
        partial,
        stop=lambda: bool(store.list("temporal_detection_frames", cache_id=partial["id"])),
    )
    with pytest.raises(caches.DetectionCacheConflict, match="incomplete"):
        replay.replay_detection_cache(store, partial["id"], class_ids=[1])


def test_profile_low_threshold_below_supported_cache_floor_has_an_actionable_error(
    completed, monkeypatch
):
    store, cache = completed
    monkeypatch.setattr(replay, "_factory", forbidden)
    with pytest.raises(ValueError, match="raise the tracker low threshold"):
        replay.replay_detection_cache(
            store,
            cache["id"],
            profile=make_profile("botsort", class_ids=[1], low_threshold=0),
        )


@pytest.mark.parametrize("clock_basis", ["unknown", "provided"])
def test_source_gaps_and_clock_are_retained_without_inventing_empty_updates(
    client, tmp_path, monkeypatch, clock_basis
):
    original = sequence(client, tmp_path)
    manifest = original["manifest"]
    identifiers = [manifest["frames"][position]["frame_id"] for position in [0, 2]]
    source = publish_sequence(
        client,
        {
            "name": "Sparse replay",
            "asset_id": manifest["asset"]["id"],
            "frame_ids": identifiers,
            "clock": {"basis": clock_basis, "fps": None, "provenance": "Synthetic test clock"},
            **(
                {"timestamps": dict(zip(identifiers, [5.0, 200.0], strict=True))}
                if clock_basis == "provided"
                else {}
            ),
        },
    )
    cache = create(client, source)
    store = client.app.state.store
    execute(store, cache)
    calls, _ = synthetic(monkeypatch)
    report = replay.replay_detection_cache(store, cache["id"], repeats=1)
    assert report["sequence"] == source["manifest"]
    assert report["sequence"]["gaps"] == [{"start_frame": 1, "end_frame": 1, "reason": "unknown"}]
    assert [call[2]["frame_index"] for call in calls if call[0] == "update"] == [0, 2]
    frames = report["passes"][0]["frames"]
    assert [row["update_index"] for row in frames] == [1, 2]
    assert [row["timestamp_seconds"] for row in frames] == (
        [None, None] if clock_basis == "unknown" else [5.0, 200.0]
    )


def test_gmc_reads_exact_frozen_pixels_in_bgr_for_every_pass(completed, monkeypatch):
    store, cache = completed
    calls, _ = synthetic(monkeypatch)
    report = replay.replay_detection_cache(store, cache["id"], algorithm="botsort", class_ids=[1])
    assert report["source_images_required"] is True
    assert report["passes"][0]["timing"]["source_image_read_ms"] > 0
    image_calls = [call for call in calls if call[0] == "update"]
    assert [call[3][0, 0].tolist() for call in image_calls] == [
        [90, 60, 0],
        [90, 60, 13],
        [90, 60, 26],
        [90, 60, 0],
        [90, 60, 13],
        [90, 60, 26],
    ]
    assert all(call[3].dtype.name == "uint8" and call[3].flags.c_contiguous for call in image_calls)


@pytest.mark.parametrize("damage", ["missing", "png", "pixels"])
def test_gmc_source_damage_does_not_publish_or_fall_back(completed, monkeypatch, tmp_path, damage):
    store, cache = completed
    synthetic(monkeypatch)
    frame = store.get("frames", cache["config"]["frame_ids"][0])
    path = store.artifact_path(frame["path"])
    if damage == "missing":
        path.unlink()
    elif damage == "png":
        path.write_bytes(b"Changed source PNG")
    else:
        # Matching bytes alone cannot hide inconsistent decoded/persisted pixels.
        store.update("frames", frame["id"], {"sha256": "f" * 64})
    destination = tmp_path / "failed.json"
    with pytest.raises((ValueError, OSError)):
        replay.replay_to_file(store, cache["id"], destination, algorithm="botsort", class_ids=[1])
    assert not destination.exists()
    assert not list(tmp_path.glob(".iris-replay-*"))


def test_native_failure_on_later_pass_leaves_no_report(completed, monkeypatch, tmp_path):
    store, cache = completed

    def fail(pass_index, position):
        if pass_index == 1 and position == 2:
            raise RuntimeError("Synthetic native GMC failure")

    synthetic(monkeypatch, failure=fail)
    destination = tmp_path / "native-failure.json"
    with pytest.raises(RuntimeError, match="native GMC"):
        replay.replay_to_file(store, cache["id"], destination)
    assert not destination.exists()


@pytest.mark.parametrize("stop_at", [0, 1, 3, 6])
def test_cancel_before_first_or_after_last_update_cannot_publish(
    completed, monkeypatch, tmp_path, stop_at
):
    store, cache = completed
    calls, _ = synthetic(monkeypatch)
    destination = tmp_path / "cancelled.json"
    with pytest.raises(replay.TrackingReplayCancelled):
        replay.replay_to_file(
            store,
            cache["id"],
            destination,
            cancelled=lambda: len([call for call in calls if call[0] == "update"]) >= stop_at,
        )
    assert not destination.exists()


def test_atomic_output_never_overwrites_even_if_destination_appears_mid_replay(
    completed, monkeypatch, tmp_path
):
    store, cache = completed
    destination = tmp_path / "receipt.json"

    def create_destination(pass_index, position):
        if pass_index == 0 and position == 1:
            destination.write_text("Keep this concurrent output")

    synthetic(monkeypatch, failure=create_destination)
    with pytest.raises(FileExistsError):
        replay.replay_to_file(store, cache["id"], destination)
    assert destination.read_text() == "Keep this concurrent output"
    assert not list(tmp_path.glob(".iris-replay-*"))
    monkeypatch.setattr(replay, "_factory", forbidden)
    with pytest.raises(FileExistsError):
        replay.replay_to_file(store, cache["id"], destination)


def test_read_only_workspace_and_cli_report_do_not_write_rows(
    completed, monkeypatch, tmp_path, capsys
):
    store, cache = completed
    synthetic(monkeypatch)
    with store.connect() as conn:
        before = list(conn.iterdump())
    readonly = replay.ReadOnlyReplayStore(store.root)
    assert readonly.get("temporal_detection_caches", cache["id"])["id"] == cache["id"]
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        with readonly.connect() as conn:
            conn.execute("DELETE FROM temporal_detection_frames")
    destination = tmp_path / "successful.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "iris",
            "tracking",
            "replay",
            "--data-dir",
            str(store.root),
            "--cache-id",
            cache["id"],
            "--tracker",
            "bytetrack",
            "--class-id",
            "1",
            "--output",
            str(destination),
        ],
    )
    cli.main()
    result = json.loads(destination.read_text())
    summary = json.loads(capsys.readouterr().out)
    assert result["complete"] and summary["frames"] == 3
    assert summary["repeatability"]["status"] == "observed_match"
    assert (
        result["cache"]["result_sha256"]
        == caches.get_detection_cache(store, cache["id"])["coverage"]["result_sha256"]
    )
    with store.connect() as conn:
        assert list(conn.iterdump()) == before


def test_read_only_handle_cannot_create_or_migrate_a_workspace(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="initialized"):
        replay.ReadOnlyReplayStore(missing)
    assert not missing.exists()
    old = tmp_path / "old"
    old.mkdir()
    with sqlite3.connect(old / "iris.sqlite3") as conn:
        conn.execute("PRAGMA user_version=20")
    before = (old / "iris.sqlite3").read_bytes()
    with pytest.raises(ValueError, match="upgrade"):
        replay.ReadOnlyReplayStore(old)
    assert (old / "iris.sqlite3").read_bytes() == before


def test_cli_complete_profile_and_existing_output_failure(completed, monkeypatch, tmp_path, capsys):
    store, cache = completed
    synthetic(monkeypatch)
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(make_profile("botsort", class_ids=[1], gmc_method="none")))
    output = tmp_path / "profile-report.json"
    argv = [
        "iris",
        "--data-dir",
        str(store.root),
        "tracking",
        "replay",
        "--cache-id",
        cache["id"],
        "--profile",
        str(profile_file),
        "--output",
        str(output),
        "--repeats",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert json.loads(output.read_text())["profile"]["gmc_method"] == "none"
    assert json.loads(capsys.readouterr().out)["repeatability"]["status"] == "not_checked"
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 1
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize("algorithm", ["bytetrack", "botsort"])
def test_real_native_adapter_reuses_complete_cache_with_repeatable_outputs(completed, algorithm):
    pytest.importorskip("lap")
    pytest.importorskip("cython_bbox")
    store, cache = completed
    profile = make_profile(algorithm, class_ids=[1, 3], gmc_method="none")
    report = replay.replay_detection_cache(store, cache["id"], profile=profile)
    assert report["complete"] is True
    assert report["repeatability"]["status"] == "observed_match"
    assert report["passes"][0]["metadata"]["algorithm"] == algorithm
    frames = report["passes"][0]["frames"]
    assert frames[0]["observations"][0]["detection_index"] == 2
    assert frames[1]["observations"] == []
    assert frames[1]["predictions"]
    assert frames[2]["observations"][0]["track_id"] == frames[0]["observations"][0]["track_id"]
