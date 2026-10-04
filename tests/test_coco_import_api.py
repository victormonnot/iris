"""COCO intake through review and freezing, using explicitly synthetic fixtures."""

import io
import json
import os
import sqlite3
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_training_live import wait_for_job

from iris.app import create_app
from iris.models import get_spec
from iris.store import SCHEMA_VERSION, Store, new_id, now

BASE_URL = "http://127.0.0.1"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL) as api:
        yield api


def package(color=30, *, negative=False):
    image = io.BytesIO()
    Image.new("RGB", (40, 30), (color, 70, 100)).save(image, format="PNG")
    coco = {
        "info": {"description": "Synthetic test only; no human ground truth"},
        "licenses": [{"id": 1, "name": "Synthetic fixture"}],
        "images": [{"id": 11, "file_name": "images/fixture.png", "width": 40, "height": 30}],
        "categories": [{"id": 42, "name": "Person"}, {"id": 8, "name": "Truck"}],
        "annotations": []
        if negative
        else [
            {"id": 99, "image_id": 11, "category_id": 42, "bbox": [2, 3, 10, 15]},
            {"id": 100, "image_id": 11, "category_id": 8, "bbox": [20, 3, 10, 15]},
        ],
    }
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotations.json", json.dumps(coco))
        archive.writestr("images/fixture.png", image.getvalue())
    return result.getvalue()


def preview(client, color=30, **kwargs):
    response = client.post(
        "/api/dataset-imports",
        files={"file": ("synthetic-coco.zip", package(color, **kwargs), "application/zip")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def config(group="fixture-train", split="train"):
    return {
        "name": group,
        "scene_group": group,
        "source_url": "https://example.invalid/synthetic-fixture",
        "license_name": "Synthetic test data",
        "attribution": "Automated test fixture",
        "source_split": split,
        "category_mapping": {"42": "person", "8": "exclude"},
    }


def commit(client, detail, payload=None):
    response = client.post(f"/api/dataset-imports/{detail['id']}/commit", json=payload or config())
    assert response.status_code == 201, response.text
    return response.json()


def fixture_review(client, frame_id):
    document = client.get(f"/api/frames/{frame_id}/annotation").json()
    response = client.put(
        f"/api/frames/{frame_id}/annotation",
        json={
            "expected_revision": 0,
            "status": "validated",
            "reviewer": "Automated synthetic test; not human ground truth",
            "boxes": [
                {
                    "id": item["id"],
                    "suggestion_id": item["id"],
                    "label": item["label"],
                    "box": item["box"],
                }
                for item in document["suggestions"]
            ],
            "decisions": {item["id"]: "accepted" for item in document["suggestions"]},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_preview_does_not_create_sessions_or_reviews_and_survives_restart(client):
    detail = preview(client)
    assert detail["image_count"] == 1 and detail["annotation_count"] == 2
    assert detail["status"] == "preview"
    assert detail["images"][0]["boxes"][0]["box"] == [2, 3, 12, 18]
    assert client.get(detail["images"][0]["image_url"]).headers["content-type"] == "image/png"
    assert client.get("/api/sessions").json() == []
    with TestClient(
        create_app(client.app.state.store.root, run_jobs=False), base_url=BASE_URL
    ) as restarted:
        restored = restarted.get(f"/api/dataset-imports/{detail['id']}").json()
        assert restored == detail
        (listed,) = restarted.get("/api/dataset-imports").json()
        assert listed["id"] == detail["id"]
        assert "path" not in listed and "images" not in listed


def test_import_requires_review_and_preserves_provenance_through_freeze(client):
    detail = preview(client)
    result = commit(client, detail)
    assert result["proposal_count"] == 1 and result["excluded_annotation_count"] == 1
    assert commit(client, detail) == result
    frame_id = result["frame_ids"][0]
    document = client.get(f"/api/frames/{frame_id}/annotation").json()
    assert document["revision"] == 0
    assert document["boxes"] == []
    assert document["suggestions"][0]["kind"] == "imported"
    candidates = client.get("/api/dataset-candidates").json()
    assert candidates["groups"] == []
    assert candidates["excluded"]["unannotated"] == 1
    review = fixture_review(client, frame_id)
    assert review["boxes"][0]["source"]["kind"] == "imported"
    negative = commit(client, preview(client, 100, negative=True), config("fixture-val", "val"))
    negative_id = negative["frame_ids"][0]
    assert negative["proposal_count"] == 0
    assert client.get(f"/api/frames/{negative_id}/annotation").json()["revision"] == 0
    fixture_review(client, negative_id)
    frozen = client.post(
        "/api/datasets",
        json={
            "name": "Synthetic COCO release",
            "frame_ids": [frame_id, negative_id],
            "splits": {"fixture-train": "train", "fixture-val": "val"},
        },
    )
    assert frozen.status_code == 201, frozen.text
    manifest = client.get(f"/api/datasets/{frozen.json()['id']}/manifest").json()
    for frame in manifest["frames"]:
        source = frame["source"]["metadata"]["dataset_import"]
        assert source["license_name"] == "Synthetic test data"
        assert source["source_url"] == config()["source_url"]
        assert source["source_split"] == frame["split"]
        assert source["category_mapping"] == config()["category_mapping"]
    assert frozen.json()["summary"]["negative_count"] == 1


def test_declared_split_reserves_unreviewed_groups_and_pixels(client):
    commit(client, preview(client))
    for color, group in ((70, "fixture-train"), (30, "different-group")):
        detail = preview(client, color)
        response = client.post(
            f"/api/dataset-imports/{detail['id']}/commit", json=config(group, "val")
        )
        assert response.status_code == 422, response.text
        assert client.get(f"/api/dataset-imports/{detail['id']}").json()["status"] == "preview"
    assert len(client.get("/api/sessions").json()) == 1


def test_review_cannot_reassign_imported_split(client):
    train = commit(client, preview(client))
    val = commit(client, preview(client, 100), config("fixture-val", "val"))
    ids = train["frame_ids"] + val["frame_ids"]
    for frame_id in ids:
        fixture_review(client, frame_id)
    candidates = client.get("/api/dataset-candidates").json()["groups"]
    assert {item["scene_group"]: item["reserved_split"] for item in candidates} == {
        "fixture-train": "train",
        "fixture-val": "val",
    }
    response = client.post(
        "/api/datasets",
        json={
            "name": "Incorrect split",
            "frame_ids": ids,
            "splits": {"fixture-train": "val", "fixture-val": "train"},
        },
    )
    assert response.status_code == 422 and "reserved" in response.text


def test_unknown_split_cannot_import_pixels_into_a_conflicting_group(client):
    commit(client, preview(client))
    commit(client, preview(client, 100), config("fixture-val", "val"))
    repeated_train_pixels = preview(client)
    response = client.post(
        f"/api/dataset-imports/{repeated_train_pixels['id']}/commit",
        json=config("fixture-val", None),
    )
    assert response.status_code == 422, response.text
    assert len(client.get("/api/sessions").json()) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"category_mapping": {"42": "person"}},
        {"category_mapping": {"42": "person", "8": "truck"}},
        {"source_split": "validation"},
        {"name": " "},
        {"license_name": ""},
        {"source_url": "file:///etc/passwd"},
        {"unknown": True},
    ],
)
def test_invalid_confirmation_is_rejected_without_sessions(client, change):
    detail = preview(client)
    response = client.post(
        f"/api/dataset-imports/{detail['id']}/commit", json={**config(), **change}
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/sessions").json() == []


def test_unknown_ids_and_oversized_upload_are_explicit(client, monkeypatch):
    assert client.get("/api/dataset-imports/unknown").status_code == 404
    assert client.post("/api/dataset-imports/unknown/commit", json=config()).status_code == 404
    detail = preview(client)
    assert client.get(f"/api/dataset-imports/{detail['id']}/images/unknown").status_code == 404
    monkeypatch.setattr("iris.app.MAX_DATASET_UPLOAD_BYTES", 10)
    response = client.post("/api/dataset-imports", files={"file": ("huge.zip", package())})
    assert response.status_code == 413
    assert list((client.app.state.store.root / "uploads").iterdir()) == []


def test_oversized_dataset_content_length_rejected_before_multipart_parser(client):
    response = client.post(
        "/api/dataset-imports",
        content=b"not even multipart",
        headers={"content-length": str(66 * 1024**2)},
    )
    assert response.status_code == 413
    assert "64 MiB" in response.text
    assert not (client.app.state.store.root / "uploads").exists()


def test_schema_six_migration_preserves_suggestions_and_revision_links(tmp_path):
    store = Store(tmp_path)
    created = now()
    session_id, asset_id, frame_id = new_id(), new_id(), new_id()
    store.insert(
        "sessions",
        {"id": session_id, "name": "Existing", "scene_group": "old", "created_at": created},
    )
    store.insert(
        "assets",
        {
            "id": asset_id,
            "session_id": session_id,
            "filename": "fixture.png",
            "kind": "image",
            "sha256": "source",
            "size_bytes": 100,
            "path": "old.png",
            "metadata": {},
            "created_at": created,
        },
    )
    store.insert(
        "frames",
        {
            "id": frame_id,
            "session_id": session_id,
            "asset_id": asset_id,
            "width": 40,
            "height": 30,
            "sha256": "pixels",
            "perceptual_hash": "0000",
            "path": "old.png",
            "created_at": created,
        },
    )
    with sqlite3.connect(store.db_path) as conn:
        conn.executescript("""
        DROP TABLE annotation_suggestions;
        CREATE TABLE annotation_suggestions (
            id TEXT PRIMARY KEY, frame_id TEXT NOT NULL REFERENCES frames(id),
            job_id TEXT REFERENCES jobs(id),
            kind TEXT NOT NULL CHECK(kind IN ('detector','multimodal')),
            label TEXT NOT NULL, box TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE INDEX annotation_suggestions_frame ON annotation_suggestions(frame_id);
        DROP TABLE dataset_imports;
        PRAGMA user_version=6;
        """)
        for kind in ("detector", "multimodal"):
            conn.execute(
                "INSERT INTO annotation_suggestions VALUES (?,?,NULL,?,?,?,?,?)",
                (kind, frame_id, kind, "person", "[1,2,10,20]", '{"original":true}', created),
            )
        before = conn.execute("SELECT * FROM annotation_suggestions ORDER BY id").fetchall()
    store.insert(
        "annotation_revisions",
        {
            "id": new_id(),
            "frame_id": frame_id,
            "revision": 1,
            "status": "draft",
            "taxonomy_id": "iris-objects-v1",
            "frame_sha256": "pixels",
            "boxes": [],
            "decisions": {"detector": "rejected"},
            "created_at": created,
        },
    )
    migrated = Store(tmp_path)
    with migrated.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert [
            tuple(row) for row in conn.execute("SELECT * FROM annotation_suggestions ORDER BY id")
        ] == before
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert migrated.list("annotation_revisions")[0]["decisions"] == {"detector": "rejected"}
    migrated.insert(
        "annotation_suggestions",
        {
            "id": "imported",
            "frame_id": frame_id,
            "kind": "imported",
            "label": "person",
            "box": [1, 2, 10, 20],
            "metadata": {},
            "created_at": created,
        },
    )
    assert len(Store(tmp_path).list("annotation_suggestions")) == 3


def test_live_import_review_freeze_train_and_evaluate(tmp_path):
    if os.environ.get("IRIS_TEST_TRAINING") != "1":
        pytest.skip("Set IRIS_TEST_TRAINING=1 for one real CPU step on imported synthetic fixtures")
    model_root = os.environ.get("IRIS_TEST_MODEL_DIR")
    assert model_root, "IRIS_TEST_MODEL_DIR must identify already installed weights"
    parent_id = "fasterrcnn_mobilenet_v3_large_320_fpn"
    root = tmp_path / "workspace"
    (root / "models").mkdir(parents=True)
    weight_name = get_spec(parent_id)["weight_filename"]
    (root / "models" / weight_name).symlink_to(Path(model_root).resolve() / "models" / weight_name)
    with TestClient(create_app(root), base_url=BASE_URL) as client:
        frame_ids = []
        for color, split in ((30, "train"), (100, "val")):
            imported = commit(client, preview(client, color), config(f"fixture-{split}", split))
            frame_ids.extend(imported["frame_ids"])
            fixture_review(client, imported["frame_ids"][0])
        response = client.post(
            "/api/datasets",
            json={
                "name": "Imported synthetic dataset; no human ground truth",
                "frame_ids": frame_ids,
                "splits": {"fixture-train": "train", "fixture-val": "val"},
            },
        )
        assert response.status_code == 201, response.text
        dataset = response.json()
        response = client.post(
            "/api/trainings",
            json={
                "name": "Synthetic COCO CPU pipeline verification",
                "dataset_id": dataset["id"],
                "parent_model_id": parent_id,
                "steps": 1,
            },
        )
        assert response.status_code == 202, response.text
        training_id = response.json()["id"]
        wait_for_job(client, response.json()["job"]["id"])
        training = client.get(f"/api/trainings/{training_id}").json()
        assert training["metadata"]["head_weights_changed"]
        response = client.post(
            "/api/evaluations",
            json={
                "name": "Synthetic COCO validation; no flight accuracy claim",
                "dataset_id": dataset["id"],
                "model_ids": [parent_id, training["checkpoint_id"]],
            },
        )
        assert response.status_code == 202, response.text
        evaluation_id = response.json()["id"]
        wait_for_job(client, response.json()["job"]["id"])
        evaluated = client.get(f"/api/evaluations/{evaluation_id}").json()
        assert len(evaluated["predictions"]) == 2
        assert all(
            item["metrics"]["summary"]["ground_truth_count"] == 1 for item in evaluated["models"]
        )
    with TestClient(create_app(root, run_jobs=False), base_url=BASE_URL) as restarted:
        assert restarted.get(f"/api/evaluations/{evaluation_id}").json() == evaluated
