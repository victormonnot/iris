"""Explicit source freezing, project ownership and complete-only profile study jobs."""

from copy import deepcopy

import pytest
from test_temporal_detection_api import client as client
from test_temporal_identities import comparison as comparison
from test_temporal_identities import completed as completed
from test_tracking_comparisons import run_worker
from test_tracking_quality import reviewed as reviewed
from test_tracking_replay import SyntheticTracker, forbidden

from iris import temporal, tracking_studies, worker
from iris.jobs import update_running
from iris.projects import record_project
from iris.store import Store
from iris.workspace_archive import ArchiveError, _inventory, create_archive, validate_database
from iris.workspace_restore import inspect_archive, restore_archive

SYNTHETIC_UPDATE = SyntheticTracker.update


@pytest.fixture
def request_payload(client, comparison, reviewed, monkeypatch):
    original = SyntheticTracker.__init__

    def initialize(self, profile, **kwargs):
        original(self, profile, **kwargs)
        from test_tracking_study_runtime import synthetic_metadata

        self.metadata = synthetic_metadata(profile)

    monkeypatch.setattr(SyntheticTracker, "__init__", initialize)
    monkeypatch.setattr(SyntheticTracker, "update", SYNTHETIC_UPDATE)
    dataset = temporal.create_temporal_dataset(
        client.app.state.store,
        name="Development take",
        entries=[
            {
                "sequence_id": comparison["sequence_id"],
                "split": "train",
                "reference_id": reviewed["id"],
            }
        ],
    )
    baseline = comparison["report"]["lanes"][0]["report"]["profile"]
    return {
        "name": "Bounded study",
        "dataset_id": dataset["id"],
        "sources": [{"sequence_id": comparison["sequence_id"], "comparison_id": comparison["id"]}],
        "baseline": {"name": "Current ByteTrack", "profile": baseline},
        "candidates": [{"name": "Longer buffer", "profile": {**baseline, "buffer_updates": 60}}],
        "class_mapping": {"1": "person", "3": None},
        "iou_threshold": 0.5,
        "repeats": 2,
        "max_updates": 100,
        "max_seconds": 120,
    }


def launch(client, payload):
    preview = client.post("/api/temporal/tracking-studies/preview", json=payload)
    assert preview.status_code == 200, preview.text
    response = client.post(
        "/api/temporal/tracking-studies",
        json={**payload, "expected_fingerprint": preview.json()["fingerprint"]},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_preview_is_read_only_pins_sources_and_complete_report_is_project_scoped(
    client,
    comparison,
    reviewed,
    request_payload,
    monkeypatch,
):
    store = client.app.state.store
    before = store.list("jobs")
    preview = client.post("/api/temporal/tracking-studies/preview", json=request_payload)
    assert preview.status_code == 200, preview.text
    assert store.list("jobs") == before
    assert preview.json()["budget"]["required_updates"] == 12
    assert preview.json()["coverage"]["development_only"] is True
    assert preview.json()["coverage"]["sources"][0]["reference_id"] == reviewed["id"]
    catalogue = client.get("/api/temporal/tracking-study-sources").json()
    assert catalogue["datasets"][0]["id"] == request_payload["dataset_id"]
    assert catalogue["sequences"][0]["comparisons"][0]["id"] == comparison["id"]
    assert catalogue["sequences"][0]["taxonomy"]["id"]
    assert client.get("/api/temporal/tracking-study-status").status_code == 200
    suggestions = client.post(
        "/api/temporal/tracking-studies/suggestions",
        json={"baseline_profile": request_payload["baseline"]["profile"]},
    )
    assert suggestions.status_code == 200 and suggestions.json()["candidates"]
    source_rows = store.list("temporal_detection_frames")
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    record = launch(client, request_payload)
    assert record["report"] is None
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    assert job["result"]["complete"] is True
    assert store.list("temporal_detection_frames") == source_rows
    assert store.list("temporal_references") == [store.get("temporal_references", reviewed["id"])]
    url = f"/api/temporal/tracking-studies/{record['id']}"
    report = client.get(url)
    assert report.status_code == 200, report.text
    assert report.json()["report"] == job["result"]
    assert client.get(url + "/report").json() == job["result"]
    assert client.get("/api/temporal/tracking-studies").json()[0]["report"] is None
    generic = next(row for row in client.get("/api/jobs").json() if row["id"] == record["id"])
    assert "runs" not in generic["result"]
    activity = client.get(f"/api/jobs/{record['id']}").json()
    assert activity["context"]["tracking_study_id"] == record["id"]
    assert activity["context"]["dataset_id"] == request_payload["dataset_id"]
    assert activity["next_action"]["workspace"] == "tracking"
    assert activity["artifacts"][0]["count"] == 1
    assert record_project(store, "jobs", job) == "default"
    foreign = client.post("/api/projects", json={"name": "Other"}).json()["id"]
    assert client.get(url, params={"project_id": foreign}).status_code == 404
    assert client.get("/api/temporal/tracking-studies", params={"project_id": foreign}).json() == []
    assert client.get(
        "/api/temporal/tracking-study-sources", params={"project_id": foreign}
    ).json() == {"datasets": [], "sequences": []}
    assert (
        client.post(
            "/api/temporal/tracking-studies/preview",
            params={"project_id": foreign},
            json=request_payload,
        ).status_code
        == 404
    )
    with pytest.raises(KeyError):
        tracking_studies.get_study(store, record["id"], project_id=foreign)


@pytest.mark.parametrize(
    "change,status",
    [
        ({"max_updates": 1}, 409),
        ({"repeats": True}, 422),
        ({"max_seconds": 0}, 422),
        ({"max_updates": 20001}, 422),
        ({"sources": []}, 422),
        ({"class_mapping": {"1": "person"}}, 409),
        ({"unknown": 1}, 422),
        ({"candidates": []}, 422),
        ({"dataset_id": "missing"}, 404),
    ],
)
def test_invalid_preview_never_admits_a_job(client, request_payload, change, status):
    before = client.app.state.store.list("jobs")
    response = client.post(
        "/api/temporal/tracking-studies/preview", json={**request_payload, **change}
    )
    assert response.status_code == status, response.text
    assert client.app.state.store.list("jobs") == before


def test_stale_preview_and_changed_baseline_are_rejected(client, request_payload):
    response = client.post(
        "/api/temporal/tracking-studies", json={**request_payload, "expected_fingerprint": "0" * 64}
    )
    assert response.status_code == 409 and "preview" in response.text
    changed = deepcopy(request_payload)
    changed["baseline"]["profile"]["buffer_updates"] = 31
    response = client.post("/api/temporal/tracking-studies/preview", json=changed)
    assert response.status_code == 409 and "baseline" in response.text


def test_source_requires_a_pinned_reference_and_exact_comparison(
    client, comparison, request_payload
):
    store = client.app.state.store
    dataset = temporal.create_temporal_dataset(
        store,
        name="No reference",
        entries=[{"sequence_id": comparison["sequence_id"], "split": "train"}],
    )
    response = client.post(
        "/api/temporal/tracking-studies/preview",
        json={**request_payload, "dataset_id": dataset["id"]},
    )
    assert response.status_code == 409 and "reference" in response.text
    store.update("jobs", comparison["id"], {"status": "failed", "result": None})
    response = client.post("/api/temporal/tracking-studies/preview", json=request_payload)
    assert response.status_code == 409 and "comparison" in response.text


def test_cancel_during_publication_cannot_publish_report(client, request_payload, monkeypatch):
    record = launch(client, request_payload)

    def cancelled_publication(current_store, job_id, changes, **kwargs):
        if changes.get("status") == "succeeded":
            current_store.update("jobs", job_id, {"cancel_requested": True})
        return update_running(current_store, job_id, changes, **kwargs)

    monkeypatch.setattr(worker, "update_running", cancelled_publication)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "cancelled" and job["result"] is None
    assert client.get(f"/api/temporal/tracking-studies/{record['id']}/report").status_code == 409


@pytest.mark.parametrize("kind", ["failure", "deadline", "cancel"])
def test_stopped_runs_have_no_partial_report(client, request_payload, monkeypatch, kind):
    from iris.tracking_study_runtime import TrackingStudyBudgetExceeded

    store = client.app.state.store
    record = launch(client, request_payload)

    def stop(*_args, **_kwargs):
        if kind == "cancel":
            store.update("jobs", record["id"], {"cancel_requested": True})
        if kind == "deadline":
            raise TrackingStudyBudgetExceeded("Time budget exhausted")
        raise RuntimeError("Stopped synthetic study")

    monkeypatch.setattr("iris.tracking_study_runtime.run_study", stop)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == ("cancelled" if kind == "cancel" else "failed")
    assert job["result"] is None


def test_report_reads_and_archives_recompute_evidence_without_tracker_runtime(
    client, request_payload, monkeypatch, tmp_path
):
    store = client.app.state.store
    record = launch(client, request_payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr("iris.tracking.tracking_status", forbidden)
    assert client.get(f"/api/temporal/tracking-studies/{record['id']}").status_code == 200
    target = tmp_path / "study.iris-workspace"
    create_archive(store.root, target)
    inspected = inspect_archive(target)
    assert inspected
    restored_root = tmp_path / "restored"
    restore_archive(target, restored_root, expected_archive_sha256=inspected["archive_sha256"])
    restored = Store(restored_root)
    assert tracking_studies.get_study(restored, record["id"])["report"] == job["result"]
    corrupt = deepcopy(job["result"])
    corrupt["summary"] = {}
    store.update("jobs", record["id"], {"result": corrupt})
    response = client.get(f"/api/temporal/tracking-studies/{record['id']}")
    assert response.status_code == 409
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="tracking study"):
        validate_database(store.root, inventory)


def test_all_active_sources_required_and_test_never_replayed_or_scored(
    client, comparison, reviewed, request_payload, monkeypatch, tmp_path
):
    from test_temporal_api import publish_sequence, reference_payload, sequence_source
    from test_temporal_detection_api import create
    from test_temporal_detections import execute
    from test_tracking_comparisons import launch as launch_comparison

    store = client.app.state.store
    validation = publish_sequence(
        client, sequence_source(client, tmp_path, group="validation", color=90)
    )
    reserved = publish_sequence(
        client, sequence_source(client, tmp_path, group="reserved-test", color=160)
    )
    val_ref = temporal.save_reference(
        store, validation["id"], payload=reference_payload(validation)
    )
    cache = create(client, validation)
    execute(store, cache)
    val_comparison = launch_comparison(client, cache)
    assert run_worker(client, val_comparison, monkeypatch)["status"] == "succeeded"
    dataset = temporal.create_temporal_dataset(
        store,
        name="Train validation and reserved test",
        entries=[
            {
                "sequence_id": comparison["sequence_id"],
                "split": "train",
                "reference_id": reviewed["id"],
            },
            {"sequence_id": validation["id"], "split": "val", "reference_id": val_ref["id"]},
            {"sequence_id": reserved["id"], "split": "test"},
        ],
    )
    payload = {**request_payload, "dataset_id": dataset["id"]}
    incomplete = client.post("/api/temporal/tracking-studies/preview", json=payload)
    assert incomplete.status_code == 409 and "every non-test" in incomplete.text
    payload["sources"] = [
        *payload["sources"],
        {"sequence_id": validation["id"], "comparison_id": val_comparison["id"]},
    ]
    expected = client.post("/api/temporal/tracking-studies/preview", json=payload)
    assert expected.status_code == 200, expected.text
    assert expected.json()["coverage"]["reserved_test_count"] == 1
    assert expected.json()["coverage"]["development_only"] is False
    assert expected.json()["budget"]["required_updates"] == 24
    forbidden_payload = deepcopy(payload)
    forbidden_payload["sources"].append(
        {"sequence_id": reserved["id"], "comparison_id": comparison["id"]}
    )
    assert (
        client.post("/api/temporal/tracking-studies/preview", json=forbidden_payload).status_code
        == 409
    )
    from iris import tracking_study_runtime

    original_replay = tracking_study_runtime.replay_detection_cache
    replayed = []

    def replay(store, cache_id, **kwargs):
        replayed.append(cache_id)
        return original_replay(store, cache_id, **kwargs)

    monkeypatch.setattr(tracking_study_runtime, "replay_detection_cache", replay)
    record = launch(client, payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    assert set(replayed) == {comparison["cache_id"], cache["id"]}
    assert len(replayed) == 4
    assert {run["sequence_id"] for run in job["result"]["runs"]} == {
        comparison["sequence_id"],
        validation["id"],
    }
    assert job["result"]["dataset"]["reserved_test_entries"][0]["sequence_id"] == reserved["id"]
    assert (
        job["result"]["summary"]["splits"]["train"]["profiles"][0]["identity"]["available"] is True
    )
    assert (
        job["result"]["summary"]["splits"]["val"]["profiles"][0]["identity"]["available"] is False
    )


def test_source_recipe_mismatch_is_rejected(
    client, comparison, reviewed, request_payload, monkeypatch, tmp_path
):
    from test_temporal_api import publish_sequence, reference_payload, sequence_source
    from test_temporal_detection_api import create
    from test_temporal_detections import execute
    from test_tracking_comparisons import launch as launch_comparison

    store = client.app.state.store
    source = publish_sequence(client, sequence_source(client, tmp_path, group="other", color=90))
    reference = temporal.save_reference(store, source["id"], payload=reference_payload(source))
    cache = create(client, source, min_score=0.05)
    execute(store, cache)
    second = launch_comparison(client, cache)
    assert run_worker(client, second, monkeypatch)["status"] == "succeeded"
    dataset = temporal.create_temporal_dataset(
        store,
        name="Recipe mismatch",
        entries=[
            {
                "sequence_id": comparison["sequence_id"],
                "split": "train",
                "reference_id": reviewed["id"],
            },
            {"sequence_id": source["id"], "split": "val", "reference_id": reference["id"]},
        ],
    )
    payload = {
        **request_payload,
        "dataset_id": dataset["id"],
        "sources": [
            *request_payload["sources"],
            {"sequence_id": source["id"], "comparison_id": second["id"]},
        ],
    }
    response = client.post("/api/temporal/tracking-studies/preview", json=payload)
    assert response.status_code == 409 and "recipe" in response.text


def test_new_conflicting_image_import_reservations_block_preview(
    client, comparison, request_payload
):
    store = client.app.state.store
    sequence = store.get("temporal_sequences", comparison["sequence_id"])
    asset = store.get("assets", sequence["asset_id"])
    store.update(
        "assets",
        asset["id"],
        {"metadata": {**asset["metadata"], "dataset_import": {"source_split": "test"}}},
    )
    response = client.post("/api/temporal/tracking-studies/preview", json=request_payload)
    assert response.status_code == 409 and "conflicting" in response.text


@pytest.mark.parametrize(
    "body",
    [b'{"name":"x","name":"y"}', b'{"repeats":NaN}', b"[" * 33 + b"0" + b"]" * 33, b" " * 131073],
)
def test_bounded_json_rejects_ambiguous_or_oversized_requests(client, body):
    response = client.post(
        "/api/temporal/tracking-studies/preview",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 409


def test_deadline_includes_report_validation_and_publication(client, request_payload, monkeypatch):
    record = launch(client, request_payload)
    clock = [0.0]
    monkeypatch.setattr(tracking_studies.time, "monotonic", lambda: clock[0])
    original = tracking_studies.validate_report

    def delayed_validation(*args, **kwargs):
        value = original(*args, **kwargs)
        clock[0] = 121.0
        return value

    monkeypatch.setattr(tracking_studies, "validate_report", delayed_validation)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "failed" and job["result"] is None
    assert "time budget" in job["error"]
