"""Fresh pipeline jobs are complete-only, scoped, portable and explicitly imported."""

import json
import sqlite3
import sys
from copy import deepcopy

import pytest
from test_temporal_detection_api import client as client
from test_temporal_detector import runtime_metadata
from test_temporal_identities import comparison as comparison
from test_temporal_identities import completed as completed
from test_tracking_comparisons import run_worker
from test_tracking_replay import forbidden

from iris import cli, tracking_costs, worker
from iris.jobs import update_running
from iris.projects import record_project
from iris.store import Store, now
from iris.temporal_detector import saved_execution_signature
from iris.tracking_cost_contracts import (
    LIMITATIONS,
    PROTOCOL,
    REPORT_SCHEMA,
    TIMING_FIELDS,
    digest,
    frame_schedule,
    next_frame,
    source_binding,
    summarize,
    validate_cost_report,
    validate_cost_request,
)
from iris.tracking_replay import ReadOnlyReplayStore
from iris.workspace_archive import ArchiveError, _inventory, create_archive, validate_database
from iris.workspace_restore import inspect_archive, restore_archive


def synthetic_report(comparison, **options):
    """Declared measurements solely for persistence/protocol integration tests."""
    config = validate_cost_request(
        {
            "lane_index": 0,
            "device": "cpu",
            "repeats": 1,
            "policy": "offline_all",
            "cadence_fps": None,
            **options,
        }
    )
    sequence = deepcopy(comparison["report"]["sequence"])
    lane = comparison["report"]["lanes"][config["lane_index"]]["report"]
    detector = deepcopy(lane["cache"]["config"]["detector"])
    detector["device"] = config["device"]
    metadata = {
        "schema": "iris-tracker-runtime-v1",
        "algorithm": lane["profile"]["algorithm"],
        "execution_policy": {
            "device": "cpu",
            "learned_reid": False,
            "opencv_threads": lane["profile"]["opencv_threads"],
            "seed": lane["profile"]["seed"],
            "native_buffer_unit": "available_frame_updates",
            "native_time_step": 1,
            "skipped_source_frames": "no_synthetic_updates",
        },
    }
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "request": config,
        "source": source_binding(comparison, config["lane_index"]),
        "sequence": sequence,
        "profile": deepcopy(lane["profile"]),
        "detector_config": detector,
        "execution": {
            "started_at": now(),
            "host": {
                "cpu": "Synthetic CPU",
                "platform": "Synthetic OS",
                "machine": "test",
                "python": "3.12",
                "logical_cpus": 2,
                "affinity_cpus": [0, 1],
            },
            "detector_signature": saved_execution_signature(detector, runtime_metadata(detector)),
            "tracker_metadata": metadata,
            "tracker_metadata_sha256": digest(metadata),
            "pipeline_sources": {
                "tracking_cost_runtime.py": "a" * 64,
                "tracking_cost_contracts.py": "b" * 64,
            },
            "setup_ms": {"detector_load_ms": 2, "tracker_setup_ms": 1, "warmup_ms": 10},
            "warmup": {
                "frame_id": sequence["frames"][0]["frame_id"],
                "passes": 1,
                "tracker_reset": True,
            },
        },
        "passes": [],
        "summary": {},
        "protocol": deepcopy(PROTOCOL),
        "limitations": deepcopy(LIMITATIONS),
    }
    indices = [frame["frame_index"] for frame in sequence["frames"]]
    for iteration in range(config["repeats"]):
        frames, dropped, position, clock = [], [], 0, 0.0
        while position < len(indices):
            selected, omitted, start = next_frame(indices, position, clock, config)
            dropped.extend(omitted)
            source = sequence["frames"][selected]
            timing = {
                **dict.fromkeys(TIMING_FIELDS, 0.0),
                "pipeline_ms": 10.0,
                "verify_decode_ms": 1.0,
                "detector_call_ms": 5.0,
                "tracker_call_ms": 2.0,
                "filter_ms": 1.0,
            }
            schedule = frame_schedule(source["frame_index"], indices[0], start, 10.0, config)
            frames.append(
                {
                    **{
                        key: source[key] for key in ("frame_id", "frame_index", "timestamp_seconds")
                    },
                    "input_size": [source["width"], source["height"]],
                    "timing": timing,
                    "schedule": schedule,
                    "work": {
                        "detection_count": 0,
                        "observation_count": 0,
                        "prediction_count": 0,
                        "unassigned_count": 0,
                        "forward_passes": 1,
                        "tile_count": 0,
                    },
                    "outputs_sha256": "c" * 64,
                }
            )
            clock = schedule["finish_ms"] if schedule else 0.0
            position = selected + 1
        memory = {
            "rss_start_bytes": 100,
            "rss_end_bytes": 110,
            "rss_sampled_peak_bytes": 120,
            "process_lifetime_peak_bytes": 200,
            "cuda": None,
        }
        if config["device"] == "cuda":
            memory["cuda"] = {
                "device": "cuda:0",
                "allocated_start_bytes": 100,
                "reserved_start_bytes": 200,
                "allocated_peak_bytes": 120,
                "reserved_peak_bytes": 220,
            }
        report["passes"].append(
            {
                "pass_index": iteration,
                "frames": frames,
                "dropped_frame_indices": dropped,
                "wall_ms": len(frames) * 10.0 + 1,
                "memory": memory,
            }
        )
    report["summary"] = summarize(report)
    return validate_cost_report(report, comparison)


@pytest.fixture
def measurement(client, comparison, monkeypatch):
    calls = []

    def measure(store, source, *, progress=None, cancelled=None, **settings):
        calls.append((source["id"], settings))
        if progress:
            progress(0.5, "Synthetic measured repetition")
        return synthetic_report(source, **settings)

    monkeypatch.setattr("iris.tracking_cost_runtime.measure_tracking_cost", measure)
    return calls


def create_cost(client, comparison, *, project="default", **settings):
    return client.post(
        f"/api/temporal/tracking-comparisons/{comparison['id']}/cost-runs",
        params={"project_id": project},
        json={
            "name": "Synthetic pipeline cost",
            "lane_index": 0,
            "device": "cpu",
            "repeats": 1,
            "policy": "offline_all",
            "cadence_fps": None,
            **settings,
        },
    )


def detail_url(record):
    return f"/api/temporal/tracking-cost-runs/{record['id']}"


def import_report(client, comparison, report, **kwargs):
    return client.post(
        f"/api/temporal/tracking-comparisons/{comparison['id']}/cost-runs/import",
        json={"report": report},
        **kwargs,
    )


def test_cost_job_measures_fresh_and_generic_history_never_contains_raw_frames(
    client, comparison, measurement, monkeypatch
):
    store = client.app.state.store
    original = {
        table: store.list(table) for table in ("temporal_references", "temporal_detection_frames")
    }
    queued = create_cost(client, comparison, lane_index=1, repeats=2).json()
    assert queued["job"]["kind"] == "tracking_cost" and queued["origin"] == "local_worker"
    assert queued["report"] is None
    assert record_project(store, "jobs", queued["job"]) == "default"
    assert client.get(detail_url(queued) + "/report").status_code == 409
    job = run_worker(client, queued, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    assert len(measurement) == 1
    detail = client.get(detail_url(queued)).json()
    assert detail["report"] == job["result"]
    assert detail["job"]["result"] == {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "request": job["result"]["request"],
        "summary": job["result"]["summary"],
    }
    history = client.get(f"/api/temporal/tracking-comparisons/{comparison['id']}/cost-runs").json()
    assert len(history) == 1 and history[0]["report"] is None
    assert history[0]["job"] == detail["job"]
    generic = next(row for row in client.get("/api/jobs").json() if row["id"] == job["id"])
    assert generic == detail["job"]
    activity = client.get(f"/api/jobs/{job['id']}").json()
    assert activity["job"] == generic
    assert activity["context"]["tracking_cost_id"] == job["id"]
    assert activity["context"]["comparison_id"] == comparison["id"]
    assert activity["context"]["session_id"]
    assert activity["artifacts"][0]["count"] == 1
    assert activity["recovery"]["can_check"] is False
    response = client.get(detail_url(queued) + "/report")
    assert response.json() == job["result"]
    assert "attachment" in response.headers["content-disposition"]
    assert {table: store.list(table) for table in original} == original


@pytest.mark.parametrize(
    "settings,status",
    [
        ({"lane_index": True}, 422),
        ({"lane_index": 2}, 422),
        ({"repeats": 0}, 422),
        ({"device": "auto"}, 422),
        ({"name": " "}, 409),
        ({"policy": "simulated_latest"}, 409),
        ({"cadence_fps": 30}, 409),
        ({"policy": "simulated_latest", "cadence_fps": 241}, 422),
        ({"origin": "imported_declaration"}, 422),
        ({"result": {}}, 422),
    ],
)
def test_invalid_requests_do_not_create_jobs(client, comparison, settings, status):
    before = client.app.state.store.list("jobs")
    response = create_cost(client, comparison, **settings)
    assert response.status_code == status, response.text
    assert client.app.state.store.list("jobs") == before


@pytest.mark.parametrize("stop", ["cancel", "failure", "supervisor", "publication"])
def test_stop_never_publishes_partial_or_late_result(client, comparison, monkeypatch, stop):
    store = client.app.state.store
    queued = create_cost(client, comparison).json()

    def measure(_store, source, **_settings):
        if stop == "cancel":
            client.app.state.jobs.cancel(queued["id"])
        elif stop == "supervisor":
            store.update("jobs", queued["id"], {"status": "interrupted", "finished_at": now()})
        elif stop == "failure":
            raise RuntimeError("Synthetic pipeline failure")
        return synthetic_report(source)

    monkeypatch.setattr("iris.tracking_cost_runtime.measure_tracking_cost", measure)
    if stop == "publication":

        def race(store, job_id, changes, **options):
            if changes.get("status") == "succeeded":
                client.app.state.jobs.cancel(job_id)
            return update_running(store, job_id, changes, **options)

        monkeypatch.setattr(worker, "update_running", race)
    job = run_worker(client, queued, monkeypatch)
    expected = {
        "cancel": "cancelled",
        "failure": "failed",
        "supervisor": "interrupted",
        "publication": "cancelled",
    }
    assert job["status"] == expected[stop]
    assert job["result"] is None
    assert client.get(detail_url(queued)).json()["report"] is None
    assert client.get(f"/api/jobs/{job['id']}/recovery").json()["available"] is False


def test_queued_cancel_restart_and_new_attempt_remain_distinct(
    client, comparison, measurement, monkeypatch
):
    store = client.app.state.store
    first = create_cost(client, comparison).json()
    cancelled = client.post(f"/api/jobs/{first['id']}/cancel").json()
    assert cancelled["status"] == "cancelled" and cancelled["result"] is None
    second = create_cost(client, comparison).json()
    store.update("jobs", second["id"], {"status": "running"})
    client.app.state.jobs._interrupt_unfinished()
    assert client.get(detail_url(second)).json()["job"]["status"] == "interrupted"
    third = create_cost(client, comparison).json()
    assert len({first["id"], second["id"], third["id"]}) == 3
    assert run_worker(client, third, monkeypatch)["status"] == "succeeded"
    assert store.get("jobs", first["id"])["result"] is None
    assert store.get("jobs", second["id"])["result"] is None


def test_import_remains_declared_and_cpu_to_cuda_preserves_recipe(client, comparison, monkeypatch):
    store = client.app.state.store
    monkeypatch.setattr("iris.tracking.tracking_status", forbidden)
    monkeypatch.setattr("iris.tracking_cost_runtime.measure_tracking_cost", forbidden)
    report = synthetic_report(comparison, device="cuda", policy="simulated_latest", cadence_fps=240)
    response = import_report(client, comparison, report)
    assert response.status_code == 201, response.text
    record = response.json()
    assert record["origin"] == record["job"]["params"]["origin"] == "imported_declaration"
    assert record["job"]["status"] == "succeeded"
    assert record["report"]["detector_config"]["device"] == "cuda"
    assert record["report"]["summary"]["dropped_frames"] == 1
    assert client.get(detail_url(record)).json() == record
    assert client.get(detail_url(record) + "/report").json() == report
    assert len([job for job in store.list("jobs") if job["kind"] == "tracking_cost"]) == 1


@pytest.mark.parametrize(
    "change", ["summary", "source", "profile", "recipe", "partial", "origin", "schedule", "memory"]
)
def test_inconsistent_import_is_rejected_atomically(client, comparison, change):
    store = client.app.state.store
    report = synthetic_report(comparison)
    if change == "summary":
        report["summary"]["service_fps"] += 1
    elif change == "source":
        report["source"]["comparison_id"] = "foreign"
    elif change == "profile":
        report["profile"]["buffer_updates"] += 1
    elif change == "recipe":
        report["detector_config"]["min_score"] = 0.02
    elif change == "partial":
        report["complete"] = False
    elif change == "origin":
        report["origin"] = "local_worker"
    elif change == "schedule":
        report["passes"][0]["frames"][0]["schedule"] = {"arrival_ms": 1}
    else:
        report["passes"][0]["memory"]["rss_sampled_peak_bytes"] = 1
    before = store.list("jobs")
    response = import_report(client, comparison, report)
    assert response.status_code == 409, response.text
    assert store.list("jobs") == before


@pytest.mark.parametrize(
    "content",
    [
        '{"report":NaN}',
        '{"report":{"a":Infinity}}',
        '{"report":{},"report":{}}',
        '{"report":null}',
        '{"report":[]}',
        '{"report":123}',
        "[]",
        '{"report":',
        '{"report":' + "[" * 1000 + "0" + "]" * 1000 + "}",
    ],
)
def test_bad_import_json_is_controlled_without_writes(client, comparison, content):
    store = client.app.state.store
    before = store.list("jobs")
    response = client.post(
        f"/api/temporal/tracking-comparisons/{comparison['id']}/cost-runs/import",
        content=content,
        headers={"content-type": "application/json"},
    )
    assert response.status_code in (409, 422), response.text
    assert store.list("jobs") == before


def test_oversize_import_and_nonfinite_create_fail_before_publication(
    client, comparison, monkeypatch
):
    store = client.app.state.store
    before = store.list("jobs")
    monkeypatch.setattr(tracking_costs, "MAX_REPORT_BYTES", 64)
    response = import_report(client, comparison, {"oversized": "a" * 100})
    assert response.status_code == 409
    response = client.post(
        f"/api/temporal/tracking-comparisons/{comparison['id']}/cost-runs",
        content='{"name":"bad","lane_index":0,"device":"cpu","repeats":1,"policy":"simulated_latest","cadence_fps":NaN}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 409
    assert store.list("jobs") == before


def test_cost_ownership_applies_to_all_routes(client, comparison):
    foreign = client.post("/api/projects", json={"name": "Other project"}).json()["id"]
    report = synthetic_report(comparison)
    record = import_report(client, comparison, report).json()
    base = f"/api/temporal/tracking-comparisons/{comparison['id']}/cost-runs"
    assert create_cost(client, comparison, project=foreign).status_code == 404
    assert (
        import_report(client, comparison, report, params={"project_id": foreign}).status_code == 404
    )
    for url in (
        base,
        detail_url(record),
        detail_url(record) + "/report",
        f"/api/jobs/{record['id']}",
    ):
        assert client.get(url, params={"project_id": foreign}).status_code == 404
    assert (
        client.post(f"/api/jobs/{record['id']}/cancel", params={"project_id": foreign}).status_code
        == 404
    )
    with pytest.raises(KeyError):
        tracking_costs.get_cost_run(client.app.state.store, record["id"], project_id=foreign)


def test_import_archive_round_trips_offline_without_execution(
    client, comparison, tmp_path, monkeypatch
):
    store = client.app.state.store
    record = import_report(client, comparison, synthetic_report(comparison)).json()
    monkeypatch.setattr("iris.tracking.tracking_status", forbidden)
    monkeypatch.setattr("iris.tracking_cost_runtime.measure_tracking_cost", forbidden)
    archive = create_archive(store.root, tmp_path / "cost.zip")
    preview = inspect_archive(archive["path"])
    target = tmp_path / "restored"
    assert restore_archive(
        archive["path"], target, expected_archive_sha256=preview["archive_sha256"]
    )["verified"]
    restored = Store(target)
    assert tracking_costs.get_cost_run(restored, record["id"]) == record
    assert restored.get("jobs", record["id"]) == store.get("jobs", record["id"])


@pytest.mark.parametrize("corruption", ["summary", "origin", "partial", "comparison", "config"])
def test_archive_and_detail_reject_inconsistent_saved_cost(client, comparison, corruption):
    store = client.app.state.store
    record = import_report(client, comparison, synthetic_report(comparison)).json()
    job = store.get("jobs", record["id"])
    if corruption == "summary":
        job["result"]["summary"]["service_fps"] += 1
    elif corruption == "origin":
        job["params"]["origin"] = "remote_authenticated"
    elif corruption == "partial":
        job["status"] = "failed"
    elif corruption == "comparison":
        job["params"]["comparison_sha256"] = "0" * 64
    else:
        job["params"]["config"]["repeats"] = 2
    store.update("jobs", record["id"], {key: job[key] for key in ("result", "params", "status")})
    assert client.get(detail_url(record)).status_code == 409
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="tracking cost"):
        validate_database(store.root, inventory)


def test_cli_measurement_is_read_only_and_output_is_atomic_no_overwrite(
    client, comparison, measurement, tmp_path, monkeypatch, capsys
):
    store = client.app.state.store
    with store.connect() as conn:
        before = list(conn.iterdump())
    readonly = ReadOnlyReplayStore(store.root)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        with readonly.connect() as conn:
            conn.execute("DELETE FROM jobs")
    target = tmp_path / "cost.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "iris",
            "--data-dir",
            str(store.root),
            "tracking",
            "measure",
            "--comparison-id",
            comparison["id"],
            "--lane-index",
            "1",
            "--repeats",
            "2",
            "--output",
            str(target),
        ],
    )
    cli.main()
    result = json.loads(target.read_text())
    assert result["complete"] is True and result["request"]["lane_index"] == 1
    assert result["request"]["repeats"] == 2
    assert json.loads(capsys.readouterr().out)["summary"] == result["summary"]
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1 and "already exists" in capsys.readouterr().err
    assert len(measurement) == 1
    with store.connect() as conn:
        assert list(conn.iterdump()) == before
    assert not list(tmp_path.glob(".iris-cost-*"))


def test_atomic_output_race_and_failure_preserve_existing_destination(
    client, comparison, tmp_path, monkeypatch
):
    target = tmp_path / "cost.json"

    def racing(_store, source, **_settings):
        target.write_text("Other writer")
        return synthetic_report(source)

    monkeypatch.setattr("iris.tracking_cost_runtime.measure_tracking_cost", racing)
    with pytest.raises(FileExistsError):
        tracking_costs.measure_to_file(
            ReadOnlyReplayStore(client.app.state.store.root), comparison["id"], target
        )
    assert target.read_text() == "Other writer"
    assert not list(tmp_path.glob(".iris-cost-*"))
