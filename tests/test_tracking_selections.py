"""Frozen selected-object sources, explicit publication and portable evidence."""

from copy import deepcopy

import pytest
from test_temporal_detection_api import client as client
from test_temporal_identities import comparison as comparison
from test_temporal_identities import completed as completed
from test_tracking_comparisons import run_worker
from test_tracking_quality import reviewed as reviewed
from test_tracking_replay import forbidden
from test_tracking_studies import request_payload as request_payload

from iris import temporal, tracking_selections, worker
from iris.jobs import update_running
from iris.projects import record_project
from iris.store import Store
from iris.tracking_selection_contracts import selection_status
from iris.workspace_archive import ArchiveError, _inventory, create_archive, validate_database
from iris.workspace_restore import inspect_archive, restore_archive


@pytest.fixture
def selection_payload(comparison, reviewed):
    replay = comparison["report"]["lanes"][0]["report"]
    first = replay["passes"][0]["frames"][0]
    return {
        "name": "Known object scenario",
        "source": {
            "kind": "comparison",
            "job_id": comparison["id"],
            "sequence_id": comparison["sequence_id"],
            "profile_sha256": replay["profile_sha256"],
        },
        "selection": {
            "frame_id": first["frame_id"],
            "detection_index": first["observations"][0]["detection_index"],
        },
        "release_frame_id": None,
        "policy": selection_status()["default_policy"],
        "evaluation": {
            "reference_id": reviewed["id"],
            "identity_id": reviewed["payload"]["identities"][0]["id"],
            "class_mapping": {"1": "person", "3": None},
            "iou_threshold": 0.5,
        },
        "max_seconds": 60,
    }


def launch(client, payload):
    response = client.post("/api/temporal/tracking-selections/preview", json=payload)
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/temporal/tracking-selections",
        json={**payload, "expected_fingerprint": response.json()["fingerprint"]},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_source_preview_run_reads_and_tasks_are_explicit_without_ml(
    client, selection_payload, reviewed, monkeypatch
):
    store = client.app.state.store
    original = {
        table: store.list(table)
        for table in ("jobs", "temporal_references", "temporal_detection_frames")
    }
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr("iris.tracking.tracking_status", forbidden)
    catalogue = client.get("/api/temporal/tracking-selection-sources")
    assert catalogue.status_code == 200, catalogue.text
    assert len(catalogue.json()["sources"]) == 2
    assert catalogue.json()["sources"][0]["frame_count"] == 3
    context = client.post(
        "/api/temporal/tracking-selection-source", json=selection_payload["source"]
    )
    assert context.status_code == 200, context.text
    assert context.json()["references"][0]["id"] == reviewed["id"]
    assert context.json()["replay"]["passes"][0]["frames"]
    assert client.get("/api/temporal/tracking-selection-status").status_code == 200
    preview = client.post("/api/temporal/tracking-selections/preview", json=selection_payload)
    assert preview.status_code == 200, preview.text
    assert preview.json()["work"]["tracker_runs"] == 0
    assert preview.json()["source_binding"]["reference_sha256"] == reviewed["payload_sha256"]
    assert all(store.list(table) == rows for table, rows in original.items())
    record = launch(client, selection_payload)
    assert record["report"] is None
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    report = job["result"]
    assert report["complete"] is True
    assert len(report["lanes"]) == 2
    url = f"/api/temporal/tracking-selections/{record['id']}"
    response = client.get(url)
    assert response.status_code == 200, response.text
    assert response.json()["report"] == report
    assert client.get(url + "/report").json() == report
    assert client.get("/api/temporal/tracking-selections").json()[0]["report"] is None
    generic = next(row for row in client.get("/api/jobs").json() if row["id"] == job["id"])
    assert generic["result"] == {"schema": report["schema"], "complete": True}
    activity = client.get(f"/api/jobs/{job['id']}").json()
    assert activity["context"]["tracking_selection_id"] == job["id"]
    assert activity["next_action"]["workspace"] == "tracking"
    assert activity["artifacts"][0]["count"] == 1
    assert record_project(store, "jobs", job) == "default"
    for table in ("temporal_references", "temporal_detection_frames"):
        assert store.list(table) == original[table]


def test_behavior_without_reference_remains_available(client, selection_payload, monkeypatch):
    payload = {**selection_payload, "evaluation": None}
    record = launch(client, payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    for lane in job["result"]["lanes"]:
        assert lane["quality"]["status"] == "unavailable"
        assert lane["summary"]["selected_frames"] > 0


def test_api_and_direct_sources_scenarios_are_project_scoped(client, selection_payload):
    store = client.app.state.store
    record = launch(client, selection_payload)
    foreign = client.post("/api/projects", json={"name": "Other"}).json()["id"]
    assert client.get(
        "/api/temporal/tracking-selection-sources", params={"project_id": foreign}
    ).json() == {"sources": []}
    assert (
        client.get("/api/temporal/tracking-selections", params={"project_id": foreign}).json() == []
    )
    for url, payload in (
        ("/api/temporal/tracking-selection-source", selection_payload["source"]),
        ("/api/temporal/tracking-selections/preview", selection_payload),
    ):
        assert client.post(url, params={"project_id": foreign}, json=payload).status_code == 404
    assert (
        client.get(
            f"/api/temporal/tracking-selections/{record['id']}", params={"project_id": foreign}
        ).status_code
        == 404
    )
    with pytest.raises(KeyError):
        tracking_selections.get_selection(store, record["id"], project_id=foreign)
    with pytest.raises(KeyError):
        tracking_selections.source_detail(store, selection_payload["source"], project_id=foreign)


@pytest.mark.parametrize(
    "change,status",
    [
        ({"max_seconds": True}, 422),
        ({"max_seconds": 121}, 422),
        ({"unknown": 1}, 422),
        ({"selection": {"frame_id": "missing", "detection_index": 0}}, 409),
        ({"selection": {"frame_id": "missing", "detection_index": True}}, 422),
        ({"release_frame_id": "missing"}, 409),
    ],
)
def test_invalid_requests_never_admit_a_job(client, selection_payload, change, status):
    store = client.app.state.store
    before = store.list("jobs")
    response = client.post(
        "/api/temporal/tracking-selections/preview", json={**selection_payload, **change}
    )
    assert response.status_code == status, response.text
    assert store.list("jobs") == before


def test_wrong_frozen_source_kind_profile_and_stale_preview_rejected(client, selection_payload):
    store = client.app.state.store
    before = store.list("jobs")
    for change, code in (({"kind": "study"}, 404), ({"profile_sha256": "0" * 64}, 409)):
        payload = {**selection_payload, "source": {**selection_payload["source"], **change}}
        response = client.post("/api/temporal/tracking-selections/preview", json=payload)
        assert response.status_code == code, response.text
    response = client.post(
        "/api/temporal/tracking-selections",
        json={**selection_payload, "expected_fingerprint": "0" * 64},
    )
    assert response.status_code == 409 and "preview" in response.text
    assert store.list("jobs") == before


def test_reference_anchor_mismatch_and_other_sequence_reference_rejected(
    client, selection_payload, reviewed, comparison
):
    payload = deepcopy(selection_payload)
    payload["evaluation"]["identity_id"] = "missing-identity"
    assert client.post("/api/temporal/tracking-selections/preview", json=payload).status_code == 409
    changed = deepcopy(reviewed["payload"])
    changed["frames"][0]["objects"][0]["box"] = [40, 5, 60, 35]
    revised = temporal.save_reference(
        client.app.state.store, comparison["sequence_id"], payload=changed, expected_revision=1
    )
    payload["evaluation"]["identity_id"] = selection_payload["evaluation"]["identity_id"]
    payload["evaluation"]["reference_id"] = revised["id"]
    assert client.post("/api/temporal/tracking-selections/preview", json=payload).status_code == 409


def test_unfinished_source_is_never_selectable(client, selection_payload):
    store = client.app.state.store
    store.update(
        "jobs", selection_payload["source"]["job_id"], {"status": "failed", "result": None}
    )
    response = client.post("/api/temporal/tracking-selections/preview", json=selection_payload)
    assert response.status_code == 409 and "complete" in response.text


def test_report_pins_reference_and_survives_later_dataset_membership(
    client, selection_payload, reviewed, comparison, monkeypatch
):
    store = client.app.state.store
    record = launch(client, selection_payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    temporal.create_temporal_dataset(
        store,
        name="Later development freeze",
        entries=[
            {
                "sequence_id": comparison["sequence_id"],
                "reference_id": reviewed["id"],
                "split": "train",
            }
        ],
    )
    report = client.get(f"/api/temporal/tracking-selections/{record['id']}")
    assert report.status_code == 200, report.text
    assert report.json()["report"] == job["result"]
    source = client.post(
        "/api/temporal/tracking-selection-source", json=selection_payload["source"]
    ).json()
    assert source["context"]["current_dataset_memberships"][0]["split"] == "train"


def test_source_study_profile_is_exact_and_no_runtime_is_executed(
    client, selection_payload, request_payload, monkeypatch
):
    from test_tracking_studies import launch as launch_study

    study = launch_study(client, request_payload)
    result = run_worker(client, study, monkeypatch)
    assert result["status"] == "succeeded", result["error"]
    replay = result["result"]["runs"][0]["replays"][1]
    payload = {
        **selection_payload,
        "source": {
            **selection_payload["source"],
            "kind": "study",
            "job_id": study["id"],
            "profile_sha256": replay["profile_sha256"],
        },
    }
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr("iris.tracking.tracking_status", forbidden)
    source = client.post("/api/temporal/tracking-selection-source", json=payload["source"])
    assert source.status_code == 200, source.text
    assert source.json()["source_binding"]["inherited_dataset"]["split"] == "train"
    assert len(client.get("/api/temporal/tracking-selection-sources").json()["sources"]) == 4
    record = launch(client, payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    assert job["result"]["profile_sha256"] == replay["profile_sha256"]


@pytest.mark.parametrize("kind", ["failure", "deadline", "cancel"])
def test_failed_cancelled_or_over_budget_attempts_publish_no_partial_report(
    client, selection_payload, monkeypatch, kind
):
    store = client.app.state.store
    record = launch(client, selection_payload)

    def stopped(*_args, **_kwargs):
        if kind == "cancel":
            store.update("jobs", record["id"], {"cancel_requested": True})
        raise ValueError("Wall-time budget exceeded" if kind == "deadline" else "Stopped scenario")

    monkeypatch.setattr(tracking_selections, "run_selection", stopped)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == ("cancelled" if kind == "cancel" else "failed")
    assert job["result"] is None
    assert client.get(f"/api/temporal/tracking-selections/{record['id']}/report").status_code == 409


def test_cancel_publication_race_never_publishes_complete_report(
    client, selection_payload, monkeypatch
):
    record = launch(client, selection_payload)

    def update(store, job_id, changes, **kwargs):
        if changes.get("status") == "succeeded":
            store.update("jobs", job_id, {"cancel_requested": True})
        return update_running(store, job_id, changes, **kwargs)

    monkeypatch.setattr(worker, "update_running", update)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "cancelled" and job["result"] is None


def test_reads_and_archives_recompute_report_without_ml(
    client, selection_payload, monkeypatch, tmp_path
):
    store = client.app.state.store
    record = launch(client, selection_payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr("iris.tracking.tracking_status", forbidden)
    target = tmp_path / "selection.iris-workspace"
    create_archive(store.root, target)
    info = inspect_archive(target)
    restored_root = tmp_path / "restored"
    restore_archive(target, restored_root, expected_archive_sha256=info["archive_sha256"])
    restored = Store(restored_root)
    assert tracking_selections.get_selection(restored, record["id"])["report"] == job["result"]
    corrupt = deepcopy(job["result"])
    corrupt["lanes"][0]["frames"][0]["state"] = "lost"
    store.update("jobs", record["id"], {"result": corrupt})
    response = client.get(f"/api/temporal/tracking-selections/{record['id']}")
    assert response.status_code == 409, response.text
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="selected-object"):
        validate_database(store.root, inventory)


def test_duplicate_and_deep_json_do_not_admit_jobs(client, selection_payload):
    import json

    encoded = json.dumps(selection_payload)
    duplicate = encoded[:-1] + ',"max_seconds":60}'
    response = client.post(
        "/api/temporal/tracking-selections/preview",
        content=duplicate,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 409
    response = client.post(
        "/api/temporal/tracking-selections/preview",
        content="[" * 33 + "0" + "]" * 33,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 409


def test_cooperative_deadline_covers_post_replay_validation(client, selection_payload, monkeypatch):
    from types import SimpleNamespace

    record = launch(client, selection_payload)
    elapsed = [0.0]
    monkeypatch.setattr(tracking_selections, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    original = tracking_selections.run_selection

    def crosses_deadline(*args, **kwargs):
        result = original(*args, **kwargs)
        elapsed[0] = 61.0
        return result

    monkeypatch.setattr(tracking_selections, "run_selection", crosses_deadline)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "failed"
    assert "wall-time budget" in job["error"]
    assert job["result"] is None


def test_forged_source_job_parent_cannot_form_an_ownership_cycle(client, selection_payload):
    store = client.app.state.store
    record = launch(client, selection_payload)
    job = store.get("jobs", record["id"])
    store.update("jobs", record["id"], {"params": {**job["params"], "source_job_id": record["id"]}})
    assert record_project(store, "jobs", store.get("jobs", record["id"])) is None
    assert client.get(f"/api/temporal/tracking-selections/{record['id']}").status_code == 404
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="selected-object"):
        validate_database(store.root, inventory)
