"""Durable visual jobs reuse complete caches and publish both lanes atomically."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_temporal_api import publish_sequence, sequence_source
from test_temporal_detection_api import client as client
from test_temporal_detection_api import create, sequence
from test_temporal_detections import execute
from test_tracking_replay import forbidden, synthetic

from iris import tracking, worker
from iris import tracking_comparisons as comparisons
from iris.jobs import update_running
from iris.projects import record_project
from iris.store import Store, now
from iris.temporal import _digest
from iris.tracking_contracts import semantic_frame
from iris.workspace_archive import ArchiveError, _inventory, create_archive, validate_database
from iris.workspace_restore import inspect_archive, restore_archive


@pytest.fixture
def completed(client, tmp_path, monkeypatch):
    cache = create(client, sequence(client, tmp_path))
    execute(client.app.state.store, cache)
    monkeypatch.setattr(
        tracking,
        "tracking_status",
        lambda: {
            "available": True,
            "packages": {},
            "installation": "Synthetic tracker runtime",
        },
    )
    return cache


def launch(client, cache, *, project_id="default", **settings):
    response = client.post(
        f"/api/temporal/detection-caches/{cache['id']}/tracking-comparisons",
        params={"project_id": project_id},
        json={"name": "Visual replay", "class_ids": [3, 1], "gmc_method": "none", **settings},
    )
    assert response.status_code == 201, response.text
    return response.json()


def run_worker(client, record, monkeypatch):
    store = client.app.state.store
    store.update("jobs", record["id"], {"status": "running", "started_at": now()})
    with monkeypatch.context() as context:
        context.setattr(worker.signal, "signal", lambda *_args: None)
        context.setattr(
            worker.ctypes,
            "CDLL",
            lambda *_args, **_kwargs: SimpleNamespace(prctl=lambda *_args: 0),
        )
        context.setattr(worker.os, "getppid", lambda: 123)
        worker.run(store.root, record["id"], 123)
    return store.get("jobs", record["id"])


def test_two_lanes_reuse_exact_inputs_and_publish_only_complete_report(
    client, completed, monkeypatch
):
    store = client.app.state.store
    calls, trackers = synthetic(monkeypatch)
    original = store.list("temporal_detection_frames")
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    record = launch(client, completed)
    assert record["report"] is None
    assert record["job"]["result"] is None
    assert record["job"]["kind"] == "tracking_compare"
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    report = job["result"]
    assert report["schema"] == "iris-tracking-comparison-v1"
    assert [lane["name"] for lane in report["lanes"]] == ["ByteTrack", "BoT-SORT"]
    assert len(trackers) == 2
    assert trackers[0] is not trackers[1]
    assert trackers[0].sequence_id == trackers[1].sequence_id == record["sequence_id"]
    updates = [[row[2] for row in calls if row[:2] == ("update", index)] for index in (0, 1)]
    assert updates[0] == updates[1]
    assert [frame["frame_index"] for frame in updates[0]] == [0, 1, 2]
    assert all(
        lane["report"]["repeatability"]["status"] == "not_checked" for lane in report["lanes"]
    )
    assert all(lane["report"]["profile"]["class_ids"] == [1, 3] for lane in report["lanes"])
    assert store.list("temporal_detection_frames") == original
    assert store.list("temporal_references") == []
    detail = client.get(f"/api/temporal/tracking-comparisons/{record['id']}").json()
    assert detail["report"] == report
    assert detail["job"]["result"] == {
        "schema": "iris-tracking-comparison-v1",
        "cache_id": completed["id"],
        "sequence_id": record["sequence_id"],
        "lane_count": 2,
        "frame_count": 3,
    }
    history = client.get(
        f"/api/temporal/detection-caches/{completed['id']}/tracking-comparisons"
    ).json()
    assert len(history) == 1 and history[0]["report"] is None
    generic = next(row for row in client.get("/api/jobs").json() if row["id"] == record["id"])
    assert generic["result"] == detail["job"]["result"]
    activity = client.get(f"/api/jobs/{record['id']}").json()
    assert activity["job"]["result"] == generic["result"]
    assert activity["next_action"]["workspace"] == "tracking"
    assert activity["context"]["comparison_id"] == record["id"]
    assert activity["context"]["name"] == "Visual replay"
    assert activity["context"]["sequence_id"] == record["sequence_id"]
    assert activity["artifacts"][0]["count"] == 1
    assert activity["recovery"]["can_check"] is False


@pytest.mark.parametrize("gmc", ["none", "sparseOptFlow"])
def test_gmc_is_explicit_and_only_botsort_reads_original_images(
    client, completed, monkeypatch, gmc
):
    calls, _ = synthetic(monkeypatch)
    record = launch(client, completed, gmc_method=gmc)
    assert run_worker(client, record, monkeypatch)["status"] == "succeeded"
    images = [[row[3] for row in calls if row[:2] == ("update", index)] for index in (0, 1)]
    assert all(image is None for image in images[0])
    assert all((image is None) == (gmc == "none") for image in images[1])


@pytest.mark.parametrize(
    "settings,status",
    [
        ({"class_ids": []}, 422),
        ({"class_ids": [True]}, 422),
        ({"class_ids": ["1"]}, 422),
        ({"class_ids": [1, 1]}, 409),
        ({"class_ids": [999]}, 409),
        ({"class_ids": [0]}, 422),
        ({"gmc_method": "orb"}, 422),
        ({"repeats": 2}, 422),
        ({"profile": {}}, 422),
        ({"name": "   "}, 409),
    ],
)
def test_strict_launch_rejects_unsupported_inputs_without_a_job(
    client, completed, settings, status
):
    before = client.app.state.store.list("jobs")
    response = client.post(
        f"/api/temporal/detection-caches/{completed['id']}/tracking-comparisons",
        json={"name": "Replay", "class_ids": [1], "gmc_method": "none", **settings},
    )
    assert response.status_code == status, response.text
    assert client.app.state.store.list("jobs") == before


def test_launch_requires_complete_lower_floor_bounded_cache_and_ready_runtime(
    client, tmp_path, completed, monkeypatch
):
    source = sequence(client, tmp_path)
    incomplete = create(client, source)
    high_floor = create(client, source, min_score=0.2)
    execute(client.app.state.store, high_floor)
    for cache in (incomplete, high_floor):
        response = client.post(
            f"/api/temporal/detection-caches/{cache['id']}/tracking-comparisons",
            json={"name": "Replay", "class_ids": [1], "gmc_method": "none"},
        )
        assert response.status_code == 409
    monkeypatch.setattr(comparisons, "MAX_COMPARISON_FRAMES", 2)
    response = client.post(
        f"/api/temporal/detection-caches/{completed['id']}/tracking-comparisons",
        json={"name": "Replay", "class_ids": [1], "gmc_method": "none"},
    )
    assert response.status_code == 409 and "at most 2" in response.text
    monkeypatch.setattr(comparisons, "MAX_COMPARISON_FRAMES", 500)
    monkeypatch.setattr(
        tracking,
        "tracking_status",
        lambda: {
            "available": False,
            "packages": {},
            "installation": "Install the tracking extra",
        },
    )
    assert client.get("/api/temporal/tracking-status").json()["available"] is False
    response = client.post(
        f"/api/temporal/detection-caches/{completed['id']}/tracking-comparisons",
        json={"name": "Replay", "class_ids": [1], "gmc_method": "none"},
    )
    assert response.status_code == 409 and "tracking extra" in response.text


def test_tracking_jobs_and_verified_frames_are_project_scoped(client, tmp_path, monkeypatch):
    project = client.post("/api/projects", json={"name": "Foreign video"}).json()["id"]
    source = sequence(client, tmp_path, project_id=project)
    cache = create(client, source, project_id=project)
    execute(client.app.state.store, cache)
    monkeypatch.setattr(tracking, "tracking_status", lambda: {"available": True})
    record = launch(client, cache, project_id=project)
    assert record_project(client.app.state.store, "jobs", record["job"]) == project
    image_url = (
        f"/api/temporal/sequences/{source['id']}/frames/"
        f"{source['manifest']['frames'][0]['frame_id']}/image"
    )
    for path in (
        f"/api/temporal/detection-caches/{cache['id']}/tracking-comparisons",
        f"/api/temporal/tracking-comparisons/{record['id']}",
        f"/api/jobs/{record['id']}",
        image_url,
    ):
        assert client.get(path).status_code == 404
        assert client.get(path, params={"project_id": project}).status_code == 200
    assert client.get("/api/jobs").json() == []
    for path in (
        f"/api/jobs/{record['id']}/cancel",
        f"/api/temporal/detection-caches/{cache['id']}/tracking-comparisons",
    ):
        assert (
            client.post(
                path, json={"name": "Replay", "class_ids": [1], "gmc_method": "none"}
            ).status_code
            == 404
        )


def test_frame_endpoint_returns_verified_snapshot_and_refuses_tampered_or_foreign_frames(
    client, completed, tmp_path
):
    store = client.app.state.store
    source = store.get("temporal_sequences", completed["sequence_id"])
    frame = store.get("frames", source["manifest"]["frames"][0]["frame_id"])
    path = f"/api/temporal/sequences/{source['id']}/frames/{frame['id']}/image"
    response = client.get(path)
    assert response.status_code == 200
    assert response.content == store.artifact_path(frame["path"]).read_bytes()
    assert response.headers["cache-control"] == "no-store"
    other = sequence(client, tmp_path)
    alien = other["manifest"]["frames"][0]["frame_id"]
    assert (
        client.get(f"/api/temporal/sequences/{source['id']}/frames/{alien}/image").status_code
        == 404
    )
    store.artifact_path(frame["path"]).write_bytes(b"changed PNG bytes")
    assert client.get(path).status_code == 409


@pytest.mark.parametrize("stop", ["cancel", "failure", "supervisor"])
def test_stopped_second_lane_never_publishes_half_a_report_and_retry_is_fresh(
    client, completed, monkeypatch, stop
):
    store = client.app.state.store
    record = launch(client, completed)

    def failure(pass_index, position):
        if pass_index == 1 and position == 1:
            if stop == "failure":
                raise RuntimeError("Native tracker failed")
            if stop == "cancel":
                client.app.state.jobs.cancel(record["id"])
            else:
                store.update("jobs", record["id"], {"status": "interrupted", "finished_at": now()})

    _, trackers = synthetic(monkeypatch, failure=failure)
    job = run_worker(client, record, monkeypatch)
    assert (
        job["status"]
        == {"cancel": "cancelled", "failure": "failed", "supervisor": "interrupted"}[stop]
    )
    assert job["result"] is None
    assert client.get(f"/api/temporal/tracking-comparisons/{record['id']}").json()["report"] is None
    assert len(trackers) == 2
    preview = client.get(f"/api/jobs/{record['id']}/recovery").json()
    assert preview["available"] is False
    retry = launch(client, completed)
    assert retry["id"] != record["id"]
    assert "recovery_of" not in retry["job"]["params"]
    synthetic(monkeypatch)
    assert run_worker(client, retry, monkeypatch)["status"] == "succeeded"
    assert store.get("jobs", record["id"]) == job


def test_queued_cancel_and_final_publication_cancellation_race(client, completed, monkeypatch):
    record = launch(client, completed)
    cancelled = client.post(f"/api/jobs/{record['id']}/cancel").json()
    assert cancelled["status"] == "cancelled" and cancelled["result"] is None
    record = launch(client, completed)
    synthetic(monkeypatch)

    def race(store, job_id, changes, **options):
        if changes.get("status") == "succeeded":
            client.app.state.jobs.cancel(job_id)
        return update_running(store, job_id, changes, **options)

    monkeypatch.setattr(worker, "update_running", race)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "cancelled"
    assert job["result"] is None
    assert job["cancel_requested"] == 1


def test_report_size_bound_fails_without_publication(client, completed, monkeypatch):
    synthetic(monkeypatch)
    record = launch(client, completed)
    monkeypatch.setattr(comparisons, "MAX_REPORT_BYTES", 100)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "failed"
    assert job["result"] is None
    assert "too large" in job["error"]


@pytest.mark.parametrize("status", ["queued", "running"])
def test_server_restart_interrupts_comparison_without_resume_or_partial_report(
    client, completed, monkeypatch, status
):
    store = client.app.state.store
    record = launch(client, completed)
    store.update("jobs", record["id"], {"status": status, "progress": 0.4})
    client.app.state.jobs._interrupt_unfinished()
    detail = client.get(f"/api/temporal/tracking-comparisons/{record['id']}").json()
    assert detail["job"]["status"] == "interrupted"
    assert detail["report"] is None and detail["job"]["result"] is None
    assert client.get(f"/api/jobs/{record['id']}/recovery").json()["available"] is False
    retry = launch(client, completed)
    synthetic(monkeypatch)
    assert run_worker(client, retry, monkeypatch)["status"] == "succeeded"
    assert store.get("jobs", record["id"])["status"] == "interrupted"


def test_gapped_unknown_clock_keeps_two_available_updates_in_both_lanes(
    client, tmp_path, monkeypatch
):
    source = sequence_source(client, tmp_path)
    source["frame_ids"] = [source["frame_ids"][0], source["frame_ids"][2]]
    source["clock"] = {"basis": "unknown", "fps": None, "provenance": "Synthetic unknown clock"}
    frozen = publish_sequence(client, source)
    cache = create(client, frozen)
    execute(client.app.state.store, cache)
    monkeypatch.setattr(tracking, "tracking_status", lambda: {"available": True})
    synthetic(monkeypatch)
    record = launch(client, cache)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    assert job["result"]["sequence"]["gaps"] == [
        {"start_frame": 1, "end_frame": 1, "reason": "unknown"}
    ]
    for lane in job["result"]["lanes"]:
        frames = lane["report"]["passes"][0]["frames"]
        assert [frame["frame_index"] for frame in frames] == [0, 2]
        assert [frame["update_index"] for frame in frames] == [1, 2]
        assert [frame["timestamp_seconds"] for frame in frames] == [None, None]


def test_complete_comparison_round_trips_backup_without_current_runtime(
    client, completed, tmp_path, monkeypatch
):
    synthetic(monkeypatch)
    record = launch(client, completed)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded"
    monkeypatch.setattr(tracking, "tracking_status", forbidden)
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    archive = create_archive(client.app.state.store.root, tmp_path / "tracking.zip")
    inspected = inspect_archive(archive["path"])
    restored = tmp_path / "restored"
    result = restore_archive(
        archive["path"], restored, expected_archive_sha256=inspected["archive_sha256"]
    )
    assert result["verified"]
    store = Store(restored)
    assert store.get("jobs", record["id"]) == job
    assert comparisons.get_tracking_comparison(store, record["id"])["report"] == job["result"]


@pytest.mark.parametrize(
    "corruption",
    [
        "lane",
        "cache",
        "profile",
        "frame",
        "prediction",
        "repeatability",
        "partial",
        "timing",
        "image_timing",
        "envelope",
    ],
)
def test_archive_refuses_inconsistent_comparison_evidence(
    client, completed, monkeypatch, corruption
):
    store = client.app.state.store
    synthetic(monkeypatch)
    record = launch(client, completed)
    job = run_worker(client, record, monkeypatch)
    result = deepcopy(job["result"])
    replay = result["lanes"][0]["report"]
    if corruption == "lane":
        result["lanes"].pop()
    elif corruption == "cache":
        replay["cache"]["fingerprint"] = "0" * 64
    elif corruption == "profile":
        replay["profile"]["buffer_updates"] = 5
    elif corruption == "frame":
        replay["passes"][0]["frames"].pop()
    elif corruption == "repeatability":
        replay["repeatability"]["status"] = "observed_match"
    elif corruption == "timing":
        replay["passes"][0]["timing"]["replay_ms"] = -1
    elif corruption == "image_timing":
        replay["passes"][0]["image_reads"][0]["frame_id"] = "foreign-frame"
    elif corruption == "envelope":
        del replay["timing_scope"]
    elif corruption == "prediction":
        frames = replay["passes"][0]["frames"]
        first, last = frames[0], frames[-1]
        last["predictions"].append(
            {
                "track_id": 999,
                "label_id": 1,
                "label": "person",
                "box": [1, 2, 10, 20],
                "confirmed": True,
                "last_observed_frame_id": first["frame_id"],
                "last_observed_frame_index": first["frame_index"],
                "last_observed_timestamp_seconds": first["timestamp_seconds"],
                "last_observed_update_index": first["update_index"],
                "age_updates": last["update_index"] - first["update_index"],
                "age_seconds": last["timestamp_seconds"] - first["timestamp_seconds"],
            }
        )
        digest = _digest(
            {
                "profile_sha256": replay["profile_sha256"],
                "cache_fingerprint": replay["cache"]["fingerprint"],
                "sequence_sha256": replay["cache"]["config"]["sequence_sha256"],
                "frames": [semantic_frame(frame) for frame in frames],
            }
        )
        replay["passes"][0]["semantic_sha256"] = digest
        replay["repeatability"]["semantic_sha256"] = [digest]
    store.update(
        "jobs",
        record["id"],
        {
            "result": result,
            "status": "cancelled" if corruption == "partial" else "succeeded",
        },
    )
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="tracking comparison records"):
        validate_database(store.root, inventory)
