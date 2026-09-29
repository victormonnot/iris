"""Local download contracts with synthetic reviewed releases, without model execution."""

import asyncio
import hashlib
import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from test_training_api import prepare_dataset

from iris.app import _DatasetExportResponse, create_app
from iris.dataset_export import ExportLimitError


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(tmp_path / "workspace", run_jobs=False), base_url="http://127.0.0.1"
    ) as api:
        yield api


def test_download_is_a_local_zip_and_releases_temporary_space(client):
    dataset, _ = prepare_dataset(client)
    store = client.app.state.store
    before = store.list("dataset_versions")
    response = client.get(f"/api/datasets/{dataset['id']}/export/coco")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["cache-control"] == "no-store"
    assert f"iris-dataset-{dataset['id']}-coco.zip" in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert json.loads(archive.read("iris-manifest.json")) == dataset["manifest"]
        for split in ("train", "val", "test"):
            document = json.loads(archive.read(f"{split}/annotations.json"))
            assert len(document["images"]) == 1
            assert {category["id"] for category in document["categories"]} == {1, 3}
    assert store.list("dataset_versions") == before
    assert not list((store.root / "exports").rglob("*.zip"))
    assert client.app.state.store.list("jobs") == []
    assert client.get("/api/system").json()["capabilities"]["dataset_export"] is True


def test_unknown_release_does_not_create_export(client):
    assert client.get("/api/datasets/missing/export/coco").status_code == 404
    assert not (client.app.state.store.root / "exports").exists()


@pytest.mark.parametrize("bad_path", [None, [], "../../outside.png"])
def test_malformed_snapshot_path_returns_conflict_and_can_be_repaired(client, bad_path):
    dataset, _ = prepare_dataset(client)
    store = client.app.state.store
    row = store.get("dataset_versions", dataset["id"])
    path = store.artifact_path(row["path"])
    original = path.read_bytes()
    manifest = json.loads(original)
    manifest["frames"][0]["image_path"] = bad_path
    changed = json.dumps(manifest).encode()
    path.write_bytes(changed)
    store.update(
        "dataset_versions", dataset["id"], {"manifest_sha256": hashlib.sha256(changed).hexdigest()}
    )
    url = f"/api/datasets/{dataset['id']}/export/coco"
    assert client.get(url).status_code == 409
    path.write_bytes(original)
    store.update("dataset_versions", dataset["id"], {"manifest_sha256": row["manifest_sha256"]})
    assert client.get(url).status_code == 200


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ExportLimitError("Synthetic size limit"), 413),
        (ValueError("Synthetic corruption"), 409),
        (OSError("private-path-not-for-browser"), 409),
    ],
)
def test_failed_build_reports_error_and_releases_export_slot(client, monkeypatch, error, status):
    import iris.app as app_module

    dataset, _ = prepare_dataset(client)
    original = app_module.build_coco_export

    def fail(*args):
        raise error

    monkeypatch.setattr(app_module, "build_coco_export", fail)
    response = client.get(f"/api/datasets/{dataset['id']}/export/coco")
    assert response.status_code == status
    assert "private-path-not-for-browser" not in response.text
    monkeypatch.setattr(app_module, "build_coco_export", original)
    assert client.get(f"/api/datasets/{dataset['id']}/export/coco").status_code == 200


def test_only_one_export_is_prepared_at_a_time(client, monkeypatch):
    import iris.app as app_module

    dataset, _ = prepare_dataset(client)
    original = app_module.build_coco_export
    started, finish = threading.Event(), threading.Event()

    def delayed(*args):
        started.set()
        assert finish.wait(10)
        return original(*args)

    monkeypatch.setattr(app_module, "build_coco_export", delayed)
    url = f"/api/datasets/{dataset['id']}/export/coco"
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.get, url)
        try:
            assert started.wait(5)
            second = client.get(url)
            assert second.status_code == 409
            assert "in progress" in second.json()["detail"]
        finally:
            finish.set()
        assert first.result(timeout=10).status_code == 200
    assert client.get(url).status_code == 200


def test_interrupted_download_deletes_archive_and_releases_slot(tmp_path):
    path = tmp_path / "temporary.zip"
    path.write_bytes(b"synthetic archive")
    slot = threading.Lock()
    slot.acquire()
    response = _DatasetExportResponse(path, "fixture", slot)

    async def disconnected_send(message):
        raise OSError("Synthetic client disconnect")

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(OSError, match="disconnect"):
        asyncio.run(
            response({"type": "http", "method": "GET", "headers": []}, receive, disconnected_send)
        )
    assert not path.exists()
    assert not slot.locked()


def test_export_after_process_reopen_is_identical(client):
    dataset, _ = prepare_dataset(client)
    url = f"/api/datasets/{dataset['id']}/export/coco"
    first = client.get(url)
    with TestClient(
        create_app(client.app.state.store.root, run_jobs=False), base_url="http://127.0.0.1"
    ) as reopened:
        second = reopened.get(url)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
