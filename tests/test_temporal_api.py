"""Temporal contracts through HTTP: immutable versions, strict inputs and project isolation."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from temporal_fixtures import video_sequence

from iris import temporal
from iris.app import create_app

BASE = "http://127.0.0.1"
SEQUENCES = "/api/temporal/sequences"
DATASETS = "/api/temporal/datasets"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE) as api:
        yield api


def sequence_source(client, tmp_path, *, project_id="default", group="take-a", color=0):
    asset, frames = video_sequence(
        client.app.state.store, tmp_path, project_id=project_id, group=group, color=color
    )
    return {
        "name": f"Sequence {group}",
        "asset_id": asset["id"],
        "frame_ids": [frame["id"] for frame in frames],
    }


def publish_sequence(client, body, project_id="default"):
    response = client.post(SEQUENCES, params={"project_id": project_id}, json=body)
    assert response.status_code == 201, response.text
    return response.json()


def reference_payload(sequence):
    manifest = sequence["manifest"]
    label = manifest["taxonomy"]["classes"][0]["id"]
    return {
        "schema": "iris-temporal-reference-v1",
        "sequence_id": sequence["id"],
        "sequence_sha256": sequence["manifest_sha256"],
        "taxonomy_id": manifest["taxonomy"]["id"],
        "identities": [{"id": "observed-a", "label": label}],
        "frames": [
            {
                "frame_index": manifest["frames"][0]["frame_index"],
                "coverage": "partial",
                "review": {"status": "human_reviewed", "reviewer": "Synthetic test"},
                "objects": [
                    {
                        "identity_id": "observed-a",
                        "label": label,
                        "box": [10, 8, 35, 48],
                        "visibility": "visible",
                        "certainty": "certain",
                    }
                ],
            }
        ],
        "notes": "Synthetic sparse reference",
    }


def publish_reference(client, sequence, *, project_id="default", expected_revision=0):
    response = client.post(
        f"{SEQUENCES}/{sequence['id']}/references",
        params={"project_id": project_id},
        json={"expected_revision": expected_revision, "payload": reference_payload(sequence)},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_sequences_preserve_source_clock_and_publish_new_versions(client, tmp_path):
    body = sequence_source(client, tmp_path)
    store = client.app.state.store
    original_frames = store.list("frames")
    first = publish_sequence(client, body)
    manifest = first["manifest"]
    assert manifest["clock"]["basis"] == "nominal_fps"
    assert manifest["clock"]["fps"] == 10
    assert [frame["timestamp_seconds"] for frame in manifest["frames"]] == [0, 0.1, 0.2]
    assert manifest["gaps"] == []
    assert len(manifest["frames"][0]["file_sha256"]) == 64
    detail = client.get(f"{SEQUENCES}/{first['id']}").json()
    assert detail == {**first, "latest_reference": None}
    assert client.get(SEQUENCES).json() == [detail]

    second = publish_sequence(
        client,
        {
            **body,
            "name": "Sparse revised selection",
            "parent_id": first["id"],
            "frame_ids": [body["frame_ids"][0], body["frame_ids"][2]],
            "clock": {"basis": "provided", "fps": None, "provenance": "Synthetic capture clock"},
            "timestamps": {body["frame_ids"][0]: 4.2, body["frame_ids"][2]: 4.47},
        },
    )
    assert second["id"] != first["id"]
    assert second["parent_id"] == first["id"]
    assert second["manifest"]["gaps"] == [{"start_frame": 1, "end_frame": 1, "reason": "unknown"}]
    assert client.get(f"{SEQUENCES}/{first['id']}").json() == detail
    assert store.list("frames") == original_frames
    assert store.list("jobs") == []


def test_reference_revisions_reject_stale_writes_and_do_not_invent_dense_evidence(client, tmp_path):
    sequence = publish_sequence(client, sequence_source(client, tmp_path))
    endpoint = f"{SEQUENCES}/{sequence['id']}/references"
    first = publish_reference(client, sequence)
    assert first["revision"] == 1
    assert first["summary"]["human_reviewed_frames"] == 1
    assert first["summary"]["omitted_frames"] == 2
    assert first["summary"]["dense_human_reference"] is False
    second = publish_reference(client, sequence, expected_revision=1)
    assert second["revision"] == 2 and second["id"] != first["id"]
    conflict = client.post(
        endpoint, json={"expected_revision": 0, "payload": reference_payload(sequence)}
    )
    assert conflict.status_code == 409
    assert "expected revision 0, found 2" in conflict.json()["detail"]
    assert client.get(endpoint).json() == [second, first]
    assert client.get(f"/api/temporal/references/{first['id']}").json() == first
    assert client.get(f"{SEQUENCES}/{sequence['id']}").json()["latest_reference"] == second
    with TestClient(create_app(client.app.state.store.root, run_jobs=False), base_url=BASE) as api:
        assert api.get(endpoint).json() == [second, first]


def test_dataset_freezes_exact_references_and_reserves_splits(client, tmp_path):
    train = publish_sequence(client, sequence_source(client, tmp_path))
    validation = publish_sequence(
        client, sequence_source(client, tmp_path, group="take-b", color=90)
    )
    reference = publish_reference(client, train)
    entries = [
        {"sequence_id": train["id"], "reference_id": reference["id"], "split": "train"},
        {"sequence_id": validation["id"], "reference_id": None, "split": "val"},
    ]
    response = client.post(DATASETS, json={"name": "Temporal pilot", "entries": entries})
    assert response.status_code == 201, response.text
    dataset = response.json()
    manifest = dataset["manifest"]
    assert manifest["entries"][0]["reference_sha256"] == reference["payload_sha256"]
    assert manifest["entries"][1]["reference_sha256"] is None
    assert manifest["evaluation_policy"]["unreviewed"] == "exclude"
    assert client.get(f"{DATASETS}/{dataset['id']}").json()["manifest"] == manifest
    assert [row["id"] for row in client.get(DATASETS).json()] == [dataset["id"]]
    next_reference = publish_reference(client, train, expected_revision=1)
    assert client.get(f"{DATASETS}/{dataset['id']}").json()["manifest"] == manifest
    revised_entries = deepcopy(entries)
    revised_entries[0]["reference_id"] = next_reference["id"]
    revised = client.post(
        DATASETS,
        json={"name": "Revised pilot", "parent_id": dataset["id"], "entries": revised_entries},
    )
    assert revised.status_code == 201, revised.text
    assert revised.json()["parent_id"] == dataset["id"]
    moved = deepcopy(entries)
    moved[0]["split"] = "test"
    conflict = client.post(DATASETS, json={"name": "Leakage", "entries": moved})
    assert conflict.status_code == 409
    assert "split conflict" in conflict.json()["detail"]
    assert len(client.get(DATASETS).json()) == 2


def test_temporal_project_scope_covers_reads_writes_parents_and_reference_links(client, tmp_path):
    project_ids = []
    for name in ("Temporal A", "Temporal B"):
        response = client.post("/api/projects", json={"name": name})
        assert response.status_code == 201, response.text
        project_ids.append(response.json()["id"])
    a, b = project_ids
    body_a = sequence_source(client, tmp_path, project_id=a)
    body_b = sequence_source(client, tmp_path, project_id=b, group="take-b", color=90)
    sequence_a = publish_sequence(client, body_a, a)
    sequence_b = publish_sequence(client, body_b, b)
    reference = publish_reference(client, sequence_a, project_id=a)
    dataset_body = {
        "name": "Scoped pilot",
        "entries": [
            {"sequence_id": sequence_a["id"], "reference_id": reference["id"], "split": "train"}
        ],
    }
    response = client.post(DATASETS, params={"project_id": a}, json=dataset_body)
    assert response.status_code == 201, response.text
    dataset = response.json()
    paths = [
        f"{SEQUENCES}/{sequence_a['id']}",
        f"{SEQUENCES}/{sequence_a['id']}/references",
        f"/api/temporal/references/{reference['id']}",
        f"{DATASETS}/{dataset['id']}",
    ]
    for path in paths:
        assert client.get(path, params={"project_id": a}).status_code == 200
        assert client.get(path, params={"project_id": b}).status_code == 404
        assert client.get(path).status_code == 404
    assert client.get(SEQUENCES).json() == []
    assert client.get(DATASETS, params={"project_id": b}).json() == []
    assert [row["id"] for row in client.get(SEQUENCES, params={"project_id": b}).json()] == [
        sequence_b["id"]
    ]
    for body in (
        body_a,
        {**body_b, "frame_ids": body_a["frame_ids"]},
        {**body_b, "parent_id": sequence_a["id"]},
    ):
        assert client.post(SEQUENCES, params={"project_id": b}, json=body).status_code == 404
    assert (
        client.post(
            f"{SEQUENCES}/{sequence_a['id']}/references",
            params={"project_id": b},
            json={"expected_revision": 1, "payload": reference_payload(sequence_a)},
        ).status_code
        == 404
    )
    own_entry = {"sequence_id": sequence_b["id"], "reference_id": None, "split": "train"}
    for body in (
        dataset_body,
        {"name": "Wrong reference", "entries": [{**own_entry, "reference_id": reference["id"]}]},
        {"name": "Wrong parent", "entries": [own_entry], "parent_id": dataset["id"]},
    ):
        assert client.post(DATASETS, params={"project_id": b}, json=body).status_code == 404
    assert len(client.get(SEQUENCES, params={"project_id": b}).json()) == 1
    assert client.get(DATASETS, params={"project_id": b}).json() == []


@pytest.mark.parametrize(
    "change",
    [
        {"project_id": "default"},
        {"name": 4},
        {"name": "x" * 161},
        {"asset_id": "x" * 129},
        {"frame_ids": []},
        {"frame_ids": ["frame"] * 10001},
        {"frame_ids": [False]},
        {"take_group": "x" * 161},
        {"timestamps": {"frame": "1.0"}},
        {"timestamps": {"frame": True}},
        {"clip": []},
        {"gaps": [{}] * 10002},
    ],
)
def test_sequence_http_input_is_strict_and_bounded(client, change):
    body = {"name": "Sequence", "asset_id": "asset", "frame_ids": ["frame"], **change}
    assert client.post(SEQUENCES, json=body).status_code == 422
    assert client.app.state.store.list("temporal_sequences") == []


@pytest.mark.parametrize(
    "body",
    [
        {"expected_revision": True, "payload": {}},
        {"expected_revision": -1, "payload": {}},
        {"expected_revision": "0", "payload": {}},
        {"expected_revision": 0, "payload": {}, "project_id": "default"},
        {"payload": {}},
    ],
)
def test_reference_revision_envelope_is_strict(client, body):
    assert client.post(f"{SEQUENCES}/missing/references", json=body).status_code == 422


@pytest.mark.parametrize(
    "change",
    [
        {"entries": []},
        {"entries": [{"sequence_id": "id", "split": "train"}] * 1001},
        {"entries": [{"sequence_id": "id", "split": "validation"}]},
        {"entries": [{"sequence_id": "id", "split": "train", "tracker_id": "1"}]},
        {"notes": "x" * 4001},
        {"parent_id": False},
        {"project_id": "default"},
    ],
)
def test_dataset_http_input_is_strict_and_bounded(client, change):
    body = {"name": "Dataset", "entries": [{"sequence_id": "id", "split": "train"}], **change}
    assert client.post(DATASETS, json=body).status_code == 422


def test_temporal_semantic_failures_and_missing_resources_are_http_errors(client, tmp_path):
    body = sequence_source(client, tmp_path)
    foreign_source = sequence_source(client, tmp_path, group="take-b", color=90)
    mixed = {**body, "frame_ids": [body["frame_ids"][0], foreign_source["frame_ids"][1]]}
    response = client.post(SEQUENCES, json=mixed)
    assert response.status_code == 409
    assert "same source video" in response.json()["detail"]
    assert client.app.state.store.list("temporal_sequences") == []
    assert client.post(SEQUENCES, json={**body, "asset_id": "absent"}).status_code == 404
    assert client.get(f"{SEQUENCES}/absent").status_code == 404
    assert client.get("/api/temporal/references/absent").status_code == 404
    assert client.get(f"{DATASETS}/absent").status_code == 404
    sequence = publish_sequence(client, body)
    bad_reference = reference_payload(sequence)
    bad_reference["sequence_sha256"] = "0" * 64
    response = client.post(
        f"{SEQUENCES}/{sequence['id']}/references",
        json={"expected_revision": 0, "payload": bad_reference},
    )
    assert response.status_code == 409
    assert "frozen manifest" in response.json()["detail"]
    assert client.app.state.store.list("temporal_references") == []


@pytest.mark.parametrize("table", ["assets", "frames"])
def test_publication_rejects_missing_source_media_without_exposing_private_paths(
    client, tmp_path, table
):
    body = sequence_source(client, tmp_path)
    store = client.app.state.store
    identifier = body["asset_id"] if table == "assets" else body["frame_ids"][0]
    record = store.get(table, identifier)
    source_path = store.artifact_path(record["path"])
    source_path.unlink()
    response = client.post(SEQUENCES, json=body)
    assert response.status_code == 409
    assert response.json()["detail"] == "Temporal source media is missing or unreadable"
    assert str(source_path) not in response.text
    assert store.list("temporal_sequences") == []


@pytest.mark.parametrize(
    "exception,status", [(KeyError, 404), (ValueError, 409), (RuntimeError, 409)]
)
def test_service_errors_remain_controlled_after_ownership_check(
    client, tmp_path, monkeypatch, exception, status
):
    sequence = publish_sequence(client, sequence_source(client, tmp_path))

    def fail(*args, **kwargs):
        raise exception("Synthetic concurrent source change")

    monkeypatch.setattr(temporal, "sequence_detail", fail)
    response = client.get(f"{SEQUENCES}/{sequence['id']}")
    assert response.status_code == status
