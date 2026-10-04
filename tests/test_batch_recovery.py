"""Unfinished local batch retry uses synthetic media and provider availability fixtures."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from test_assistance_batches import queue
from test_assistance_batches import workspace as batch_workspace
from test_assistance_batches_api import create, payload
from test_assistance_batches_api import workspace as api_workspace

from iris.annotations import AnnotationConflict
from iris.batch_recovery import preview_retry_batch, retry_batch
from iris.jobs import JobManager
from iris.store import new_id, now

workspace = batch_workspace
api = api_workspace


def stopped(workspace):
    store, _, _, _ = workspace
    batch = queue(workspace)
    for job_id, status in zip(
        batch["job_ids"], ("succeeded", "failed", "interrupted"), strict=True
    ):
        store.update("jobs", job_id, {"status": status, "finished_at": now()})
    return batch


def test_retry_retains_successes_and_is_a_new_immutable_batch(workspace):
    store, jobs, _, _ = workspace
    batch = stopped(workspace)
    before = {
        table: store.list(table) for table in ("jobs", "assistance_records", "annotation_revisions")
    }
    preview = preview_retry_batch(store, batch["id"])
    assert preview["retained_count"] == 1 and preview["eligible_count"] == 2
    assert preview["frame_ids"] == batch["frame_ids"][1:]
    assert {table: store.list(table) for table in before} == before
    created = retry_batch(
        store, jobs, batch["id"], name="New requests", expected_fingerprint=preview["fingerprint"]
    )
    assert created["config"]["retry_of"] == batch["id"]
    assert created["frame_ids"] == batch["frame_ids"][1:]
    assert not set(created["job_ids"]) & set(batch["job_ids"])
    assert [store.get("jobs", row["id"]) for row in before["jobs"]] == before["jobs"]
    assert store.list("annotation_revisions") == before["annotation_revisions"]
    assert (
        retry_batch(
            store,
            jobs,
            batch["id"],
            name="Lost response",
            expected_fingerprint=preview["fingerprint"],
        )["id"]
        == created["id"]
    )
    assert len(store.list("jobs")) == 5
    with pytest.raises(AnnotationConflict, match="already created"):
        preview_retry_batch(store, batch["id"])


def test_saved_proposals_from_interrupted_child_are_never_replayed(workspace):
    store, jobs, _, _ = workspace
    batch = stopped(workspace)
    store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": batch["frame_ids"][2],
            "job_id": batch["job_ids"][2],
            "kind": "multimodal",
            "label": "person",
            "box": [2, 3, 30, 35],
            "metadata": {"synthetic": True},
            "created_at": now(),
        },
    )
    preview = preview_retry_batch(store, batch["id"])
    assert preview["retained_count"] == 2 and preview["eligible_count"] == 1
    created = retry_batch(
        store, jobs, batch["id"], name="Remaining", expected_fingerprint=preview["fingerprint"]
    )
    assert created["frame_ids"] == [batch["frame_ids"][1]]
    assert len(store.list("annotation_suggestions")) == 1


def test_changed_parent_or_annotations_requires_new_preview(workspace):
    store, jobs, _, _ = workspace
    batch = stopped(workspace)
    preview = preview_retry_batch(store, batch["id"])
    store.update("jobs", batch["job_ids"][1], {"status": "cancelled"})
    with pytest.raises(AnnotationConflict, match="changed"):
        retry_batch(
            store, jobs, batch["id"], name="Stale", expected_fingerprint=preview["fingerprint"]
        )
    assert len(store.list("jobs")) == 3


def test_active_and_all_successful_batches_have_no_retry(workspace):
    store, _, _, _ = workspace
    batch = queue(workspace)
    with pytest.raises(RuntimeError, match="Wait"):
        preview_retry_batch(store, batch["id"])
    for identifier in batch["job_ids"]:
        store.update("jobs", identifier, {"status": "succeeded"})
    with pytest.raises(ValueError, match="No unfinished"):
        preview_retry_batch(store, batch["id"])


def test_external_batch_config_is_rejected_without_dispatch(workspace):
    store, _, _, _ = workspace
    batch = stopped(workspace)
    config = dict(batch["config"])
    config["provider"] = {"provider": "alibaba"}
    store.update("assistance_batches", batch["id"], {"config": config})
    with pytest.raises(ValueError, match="Only local"):
        preview_retry_batch(store, batch["id"])
    assert len(store.list("jobs")) == 3


def test_concurrent_confirmations_create_one_successor(workspace):
    store, _, _, _ = workspace
    batch = stopped(workspace)
    preview = preview_retry_batch(store, batch["id"])
    barrier = Barrier(2)

    def confirm(_):
        barrier.wait()
        try:
            return retry_batch(
                store,
                JobManager(store),
                batch["id"],
                name="Concurrent",
                expected_fingerprint=preview["fingerprint"],
            )["id"]
        except (AnnotationConflict, RuntimeError):
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(confirm, range(2)))
    assert len({value for value in results if value}) == 1
    assert len(store.list("assistance_batches")) == 2
    assert len(store.list("jobs")) == 5


def test_retry_http_preview_confirmation_receipt_and_project_scope(api):
    client, store, session, frames = api
    batch = create(client, session, payload(frames))
    for identifier, status in zip(batch["job_ids"], ("succeeded", "failed"), strict=True):
        store.update("jobs", identifier, {"status": status, "finished_at": now()})
    endpoint = f"/api/assistance-batches/{batch['id']}"
    other = client.post("/api/projects", json={"name": "Other project"}).json()["id"]
    assert client.post(endpoint + "/retry-preview", params={"project_id": other}).status_code == 404
    response = client.post(endpoint + "/retry-preview", json={})
    assert response.status_code == 200, response.text
    preview = response.json()
    body = {"name": "New local requests", "expected_fingerprint": preview["fingerprint"]}
    assert (
        client.post(endpoint + "/retry", json={**body, "allow_external": True}).status_code == 422
    )
    assert (
        client.post(endpoint + "/retry", params={"project_id": other}, json=body).status_code == 404
    )
    response = client.post(endpoint + "/retry", json=body)
    assert response.status_code == 202, response.text
    assert response.json()["frame_ids"] == [batch["frame_ids"][1]]
    assert client.post(endpoint + "/retry", json=body).json()["id"] == response.json()["id"]
    assert len(store.list("jobs")) == 3
