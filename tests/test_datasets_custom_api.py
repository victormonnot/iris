"""Frozen custom dataset workflows through HTTP; synthetic media and no model calls."""

import hashlib
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from test_custom_classes_api import HELMET, intake, publish, save

from iris import evaluation, training
from iris.app import create_app
from iris.store import new_id, now
from iris.taxonomies import TAXONOMY

CLASSES = [
    HELMET,
    {"id": "bottle", "name": "Bottle", "definition": "A visible bottle.", "coco_id": 44},
    {"id": "damaged_panel", "name": "Damaged panel", "definition": "A panel with visible cracks."},
]
PARENT = "fasterrcnn_mobilenet_v3_large_320_fpn"


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(tmp_path / "workspace", run_jobs=False), base_url="http://127.0.0.1"
    ) as api:
        yield api


@pytest.fixture
def reviewed(client):
    taxonomy = publish(client, CLASSES)
    frames = [intake(client, color=color) for color in (35, 55, 75)]
    for frame, label in zip(frames, ("helmet", "bottle", None), strict=True):
        boxes = [{"id": "box", "label": label, "box": [2, 3, 20, 22]}] if label else []
        response = save(client, frame, taxonomy, boxes=boxes)
        assert response.status_code == 200, response.text
    candidates = client.get("/api/dataset-candidates").json()
    payload = {
        "name": "Synthetic custom release",
        "taxonomy_id": taxonomy["id"],
        "frame_ids": [frame["id"] for frame in frames],
        "splits": {"35": "train", "55": "val", "75": "test"},
        "expected_revisions": {
            frame["id"]: frame["annotation_revision_id"]
            for group in candidates["groups"]
            for frame in group["frames"]
        },
    }
    return taxonomy, frames, payload


def freeze(client, payload):
    response = client.post("/api/datasets", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_custom_release_download_and_lists_use_frozen_classes_after_edits(client, reviewed):
    taxonomy, frames, payload = reviewed
    dataset = freeze(client, payload)
    assert dataset["taxonomy_id"] == taxonomy["id"]
    assert not dataset["ml_supported"] and dataset["ml_limitation"]
    assert dataset["manifest"]["schema_version"] == 2
    assert dataset["manifest"]["taxonomy"] == taxonomy
    assert (
        dataset["class_mapping"]
        == dataset["coco_mapping"]
        == {
            "helmet": 1,
            "bottle": 2,
            "damaged_panel": 3,
        }
    )
    assert dataset["summary"]["class_counts"] == {"helmet": 1, "bottle": 1, "damaged_panel": 0}
    assert dataset["summary"]["negative_count"] == 1
    endpoint = f"/api/datasets/{dataset['id']}"
    before = client.get(endpoint + "/export/coco")
    assert before.status_code == 200, before.text
    raw_manifest = client.get(endpoint + "/manifest").content
    assert "path" not in client.get("/api/datasets").json()[0]
    publish(
        client,
        [{**HELMET, "definition": "Only helmets worn on a head."}, *CLASSES[1:]],
        taxonomy["id"],
    )
    assert (
        save(client, frames[0], taxonomy, revision=1, boxes=[], status="draft").status_code == 200
    )
    assert client.get(endpoint).json() == dataset
    assert client.get(endpoint + "/manifest").content == raw_manifest
    assert client.get(endpoint + "/export/coco").content == before.content
    with zipfile.ZipFile(io.BytesIO(before.content)) as archive:
        assert archive.read("iris-manifest.json") == raw_manifest
        metadata = json.loads(archive.read("export.json"))
        assert metadata["protocol"] == "iris-coco-export-v2"
        assert metadata["coco_mapping"]["bottle"] == 2
        assert len(json.loads(archive.read("test/annotations.json"))["images"]) == 1
    with TestClient(
        create_app(client.app.state.store.root, run_jobs=False), base_url="http://127.0.0.1"
    ) as reopened:
        assert reopened.get(endpoint).json() == dataset
        assert reopened.get(endpoint + "/export/coco").content == before.content


def test_candidate_version_selection_and_parent_compatibility_are_explicit(client, reviewed):
    taxonomy, frames, payload = reviewed
    parent = freeze(client, payload)
    updated = publish(client, [{**HELMET, "name": "Worn helmet"}, *CLASSES[1:]], taxonomy["id"])
    current = client.get("/api/dataset-candidates").json()
    assert current["taxonomy"] == updated and current["groups"] == []
    assert current["excluded"]["different_taxonomy"] == 3
    historical = client.get(
        "/api/dataset-candidates", params={"taxonomy_id": taxonomy["id"]}
    ).json()
    assert historical["taxonomy"] == taxonomy
    assert sum(group["count"] for group in historical["groups"]) == 3
    child = freeze(client, {**payload, "parent_id": parent["id"]})
    assert child["parent_id"] == parent["id"]
    for frame in frames:
        endpoint = f"/api/frames/{frame['id']}/annotation"
        adopted = client.post(
            endpoint + "/taxonomy",
            json={
                "expected_revision": 1,
                "expected_taxonomy_id": taxonomy["id"],
                "target_taxonomy_id": updated["id"],
            },
        )
        assert adopted.status_code == 200, adopted.text
        assert save(client, frame, updated, revision=2, boxes=[]).status_code == 200
    changed = {
        **payload,
        "taxonomy_id": updated["id"],
        "expected_revisions": None,
        "parent_id": parent["id"],
    }
    assert client.post("/api/datasets", json=changed).status_code == 422
    assert len(client.get("/api/datasets").json()) == 2
    changed["parent_id"] = None
    assert freeze(client, changed)["taxonomy"] == updated


@pytest.mark.parametrize("status", ["draft", "validated"])
def test_changed_review_since_preview_returns_conflict_without_artifacts(client, reviewed, status):
    taxonomy, frames, payload = reviewed
    assert save(client, frames[0], taxonomy, revision=1, boxes=[], status=status).status_code == 200
    response = client.post("/api/datasets", json=payload)
    assert response.status_code == 409, response.text
    store = client.app.state.store
    assert store.list("dataset_versions") == []
    assert not list((store.root / "datasets").glob("*"))


def test_mixed_annotation_versions_cannot_be_frozen_together(client, reviewed):
    taxonomy, frames, payload = reviewed
    updated = publish(client, [{**HELMET, "name": "Worn helmet"}, *CLASSES[1:]], taxonomy["id"])
    adopted = client.post(
        f"/api/frames/{frames[0]['id']}/annotation/taxonomy",
        json={
            "expected_revision": 1,
            "expected_taxonomy_id": taxonomy["id"],
            "target_taxonomy_id": updated["id"],
        },
    )
    assert adopted.status_code == 200, adopted.text
    assert save(client, frames[0], updated, revision=2).status_code == 200
    for chosen in (None, taxonomy["id"], updated["id"]):
        response = client.post(
            "/api/datasets", json={**payload, "taxonomy_id": chosen, "expected_revisions": None}
        )
        assert response.status_code == 422, response.text
    assert client.app.state.store.list("dataset_versions") == []


def test_other_project_cannot_read_freeze_or_export_custom_release(client, reviewed):
    taxonomy, _, payload = reviewed
    dataset = freeze(client, payload)
    other = client.post("/api/projects", json={"name": "Separate project"}).json()
    params = {"project_id": other["id"]}
    assert client.get("/api/datasets", params=params).json() == []
    for suffix in ("", "/manifest", "/export/coco"):
        assert (
            client.get(f"/api/datasets/{dataset['id']}" + suffix, params=params).status_code == 404
        )
    assert client.post("/api/datasets", params=params, json=payload).status_code == 422
    assert (
        client.get(
            "/api/dataset-candidates", params={**params, "taxonomy_id": taxonomy["id"]}
        ).status_code
        == 409
    )
    assert len(client.app.state.store.list("dataset_versions")) == 1


def test_custom_releases_cannot_launch_training_or_evaluation_before_model_access(
    client, reviewed, monkeypatch
):
    dataset = freeze(client, reviewed[2])

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported custom releases must fail before accessing a model")

    monkeypatch.setattr(training, "catalog", unexpected)
    monkeypatch.setattr(evaluation, "catalog", unexpected)
    for kind, fields in (
        ("trainings", {"parent_model_id": PARENT}),
        ("evaluations", {"model_ids": [PARENT]}),
    ):
        for suffix in ("", "/preview"):
            response = client.post(
                "/api/" + kind + suffix,
                json={"name": "Unsupported custom run", "dataset_id": dataset["id"], **fields},
            )
            assert response.status_code == 422 and "custom" in response.text.lower(), response.text
    store = client.app.state.store
    assert store.list("jobs") == store.list("training_runs") == store.list("evaluations") == []


def test_custom_evaluation_worker_rejects_persisted_request_before_detector_load(client, reviewed):
    dataset = freeze(client, reviewed[2])
    store = client.app.state.store
    job = store.insert(
        "jobs",
        {"id": new_id(), "kind": "evaluate", "status": "queued", "params": {}, "created_at": now()},
    )
    row = store.insert(
        "evaluations",
        {
            "id": new_id(),
            "name": "Unsupported persisted fixture",
            "dataset_id": dataset["id"],
            "split": "val",
            "model_ids": [PARENT],
            "config": {"frame_ids": [reviewed[1][1]["id"]]},
            "job_id": job["id"],
            "created_at": now(),
        },
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported persisted request must not load a detector")

    with pytest.raises(ValueError, match="custom class"):
        evaluation.run_evaluation(
            store, row["id"], lambda *_: None, lambda: False, detector_factory=unexpected
        )
    assert store.list("evaluation_models") == store.list("evaluation_predictions") == []


def test_legacy_taxonomy_alias_cannot_override_custom_person_car_definitions(client, monkeypatch):
    custom = publish(
        client,
        [
            {**TAXONOMY["classes"][0], "definition": "Only people wearing a safety helmet."},
            TAXONOMY["classes"][1],
        ],
    )
    frames = [intake(client, color=color) for color in (35, 55)]
    for frame in frames:
        assert (
            save(
                client,
                frame,
                custom,
                boxes=[
                    {
                        "id": "person",
                        "label": "person",
                        "box": [2, 3, 20, 22],
                    }
                ],
            ).status_code
            == 200
        )
    dataset = freeze(
        client,
        {
            "name": "Custom Person / Car semantics",
            "taxonomy_id": custom["id"],
            "frame_ids": [frame["id"] for frame in frames],
            "splits": {"35": "train", "55": "val"},
        },
    )
    store = client.app.state.store
    row = store.get("dataset_versions", dataset["id"])
    manifest = dataset["manifest"]
    manifest["taxonomy_id"] = TAXONOMY["id"]
    raw = json.dumps(manifest).encode()
    store.artifact_path(row["path"]).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    store.update("dataset_versions", dataset["id"], {"manifest_sha256": digest})

    def unexpected(*args, **kwargs):
        pytest.fail("A legacy alias must never override the frozen class definitions")

    monkeypatch.setattr(training, "catalog", unexpected)
    response = client.post(
        "/api/trainings/preview",
        json={
            "name": "Conflicting alias",
            "dataset_id": dataset["id"],
            "parent_model_id": PARENT,
        },
    )
    assert response.status_code == 422 and "Custom class training" in response.text, response.text
    job = store.insert(
        "jobs",
        {
            "id": new_id(),
            "kind": "train",
            "status": "queued",
            "params": {},
            "created_at": now(),
        },
    )
    run = store.insert(
        "training_runs",
        {
            "id": new_id(),
            "name": "Persisted conflicting alias",
            "dataset_id": dataset["id"],
            "parent_model_id": PARENT,
            "config": {"dataset_manifest_sha256": digest},
            "job_id": job["id"],
            "created_at": now(),
        },
    )
    with pytest.raises(ValueError, match="Custom class training"):
        training.run_training(
            store, run["id"], lambda *_: None, lambda: False, trainer_factory=unexpected
        )
    assert store.list("trained_models") == []
