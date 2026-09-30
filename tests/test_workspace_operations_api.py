"""Workspace transfer flows use local generated images and real ZIP/SQLite files."""

import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from test_training_api import BASE_URL, prepare_dataset

from iris import workspace_operations
from iris.app import create_app
from iris.store import Store
from iris.workspace_operations import WorkspaceBusy, WorkspaceOperations


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "source", run_jobs=False), base_url=BASE_URL) as api:
        yield api


def wait_for(client, identifier):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get(f"/api/workspace/operations/{identifier}")
        assert response.status_code == 200, response.text
        record = response.json()
        if record["status"] not in {"queued", "running"}:
            return record
        time.sleep(0.01)
    pytest.fail("Workspace transfer did not finish")


def start_backup(client):
    response = client.post("/api/workspace/backups")
    assert response.status_code == 202, response.text
    record = wait_for(client, response.json()["id"])
    assert record["status"] == "succeeded", record
    return record


def start_inspection(client, content):
    response = client.post(
        "/api/workspace/restore-inspections",
        files={"file": ("test-workspace.zip", content, "application/zip")},
    )
    assert response.status_code == 202, response.text
    return wait_for(client, response.json()["id"])


def test_complete_backup_inspection_restore_and_preserve_source(client):
    dataset, _ = prepare_dataset(client)
    store = client.app.state.store
    before = {table: store.list(table) for table in store.columns}
    preview = client.get("/api/workspace/backup-preview").json()
    assert preview["can_create"] and preview["file_count"] > 1
    assert preview["counts"]["dataset_versions"] == 1
    assert preview["workspace_path"] == str(store.root)
    assert preview["restore_parent"] == str(store.root.parent)
    assert client.get("/api/system").json()["capabilities"]["workspace_backup"]

    backup = start_backup(client)
    result = backup["result"]
    assert result["summary"]["counts"]["annotation_revisions"] == 3
    response = client.get(result["download_url"])
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.headers["cache-control"] == "no-store"
    assert len(response.content) == result["archive_size_bytes"]
    inspection = start_inspection(client, response.content)
    assert inspection["status"] == "succeeded", inspection
    assert inspection["result"]["archive_sha256"] == result["archive_sha256"]
    payload = {"inspection_id": inspection["id"], "folder_name": "restored-demo"}
    response = client.post("/api/workspace/restores", json=payload)
    assert response.status_code == 202, response.text
    restore = wait_for(client, response.json()["id"])
    assert restore["status"] == "succeeded", restore
    restored = Store(store.root.parent / "restored-demo")
    assert restore["result"]["destination"] == str(restored.root)
    assert str(restored.root) in restore["result"]["launch_command"]
    for table, records in before.items():
        assert store.list(table) == records
        assert restored.list(table) == records
    assert restored.get("dataset_versions", dataset["id"]) is not None
    assert client.post("/api/workspace/restores", json=payload).status_code == 409
    assert client.delete(f"/api/workspace/operations/{restore['id']}").status_code == 422
    assert client.delete(f"/api/workspace/operations/{inspection['id']}").status_code == 204
    assert client.delete(f"/api/workspace/operations/{backup['id']}").status_code == 204
    assert restored.db_path.exists()
    assert client.get(result["download_url"]).status_code == 404


def test_operations_reopen_without_restarting_or_changing_data(client):
    backup = start_backup(client)
    manager = WorkspaceOperations(client.app.state.store)
    manager.recover()
    assert manager.get(backup["id"]) == backup
    assert manager.list()[0] == backup
    assert not manager.gate.frozen and manager.active_id is None


def test_running_backup_rejects_writes_but_allows_reads_and_cancellation(client, monkeypatch):
    entered, finish = threading.Event(), threading.Event()

    def delayed(*args, cancelled, **kwargs):
        entered.set()
        assert finish.wait(5)
        assert cancelled()
        raise workspace_operations.ArchiveCancelled()

    monkeypatch.setattr(workspace_operations, "create_archive", delayed)
    response = client.post("/api/workspace/backups")
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    assert entered.wait(2)
    try:
        assert client.get("/api/sessions").status_code == 200
        assert client.get("/api/workspace/backup-preview").json()["write_locked"]
        response = client.post("/api/sessions", json={"name": "Blocked", "scene_group": "blocked"})
        assert response.status_code == 409
        assert client.post("/api/workspace/backups").status_code == 409
        assert client.delete(f"/api/workspace/operations/{identifier}").status_code == 409
        assert client.post(f"/api/workspace/operations/{identifier}/cancel").status_code == 200
    finally:
        finish.set()
    assert wait_for(client, identifier)["status"] == "cancelled"
    response = client.post("/api/sessions", json={"name": "Allowed", "scene_group": "allowed"})
    assert response.status_code == 201
    assert not client.app.state.workspace_operations.gate.frozen


def test_backup_waits_for_existing_http_writes(client):
    entered, finish = threading.Event(), threading.Event()

    @client.app.post("/api/test-hold-write")
    def hold_write():
        entered.set()
        assert finish.wait(5)
        return {"finished": True}

    with ThreadPoolExecutor() as pool:
        request = pool.submit(client.post, "/api/test-hold-write")
        assert entered.wait(2)
        try:
            assert client.post("/api/workspace/backups").status_code == 409
            assert not client.app.state.workspace_operations.gate.frozen
        finally:
            finish.set()
        assert request.result().status_code == 200
    assert start_backup(client)["status"] == "succeeded"


def test_backup_failure_releases_gate_and_hides_private_paths(client, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("/private/user/secret.txt is missing")

    monkeypatch.setattr(workspace_operations, "create_archive", fail)
    response = client.post("/api/workspace/backups")
    record = wait_for(client, response.json()["id"])
    assert record["status"] == "failed"
    assert "/private" not in record["error"]
    assert not client.app.state.workspace_operations.gate.frozen
    assert (
        client.post("/api/sessions", json={"name": "Next", "scene_group": "next"}).status_code
        == 201
    )


def test_unfinished_receipts_become_interrupted_without_restoring(client):
    manager = client.app.state.workspace_operations
    record = manager._reserve("restore")
    reopened = WorkspaceOperations(client.app.state.store)
    reopened.recover()
    result = reopened.get(record["id"])
    assert result["status"] == "interrupted" and result["result"] is None
    assert not reopened.gate.frozen and reopened.active_id is None
    manager.active_id = None


def test_download_lease_prevents_removal(client):
    backup = start_backup(client)
    manager = client.app.state.workspace_operations
    path, _ = manager.archive(backup["id"])
    assert path.exists()
    try:
        with pytest.raises(WorkspaceBusy, match="download"):
            manager.delete(backup["id"])
    finally:
        manager.release_download(backup["id"])
    assert client.get(backup["result"]["download_url"]).status_code == 200
    assert manager.downloads == {}
    manager.delete(backup["id"])
    assert not path.exists()


@pytest.mark.parametrize(
    "name", ["..", "../other", "/tmp/other", "a/b", "a\\b", ".hidden", "", "x" * 81]
)
def test_restore_rejects_paths_and_invalid_names(client, name):
    response = client.post(
        "/api/workspace/restores", json={"inspection_id": "a" * 32, "folder_name": name}
    )
    assert response.status_code == 422


def test_restore_needs_verified_inspection(client):
    backup = start_backup(client)
    response = client.post(
        "/api/workspace/restores",
        json={"inspection_id": backup["id"], "folder_name": "not-an-inspection"},
    )
    assert response.status_code == 422
    inspection = start_inspection(client, b"not a ZIP")
    assert inspection["status"] == "failed"
    response = client.post(
        "/api/workspace/restores",
        json={"inspection_id": inspection["id"], "folder_name": "invalid-restore"},
    )
    assert response.status_code == 422
    assert not (client.app.state.store.root.parent / "invalid-restore").exists()


def test_uploaded_bytes_are_bounded_and_failed_spool_is_removed(client, monkeypatch):
    monkeypatch.setattr(workspace_operations, "MAX_ARCHIVE_BYTES", 32)
    response = client.post(
        "/api/workspace/restore-inspections", files={"file": ("oversize.zip", b"x" * 33)}
    )
    assert response.status_code == 413
    manager = client.app.state.workspace_operations
    assert manager.active_id is None
    assert not list(manager.root.glob("*/workspace.zip"))
    assert manager.list()[0]["status"] == "failed"


def test_empty_upload_is_rejected(client):
    response = client.post(
        "/api/workspace/restore-inspections", files={"file": ("empty.zip", io.BytesIO(b""))}
    )
    assert response.status_code == 422


def test_chunked_upload_is_rejected_before_multipart_spooling(client):
    response = client.post(
        "/api/workspace/restore-inspections",
        content=iter([b"--large multipart input"]),
        headers={"content-type": "multipart/form-data; boundary=large"},
    )
    assert response.status_code == 411
    assert client.get("/api/workspace/operations").json() == []


def test_worker_start_failure_is_terminal_and_releases_gate(client, monkeypatch):
    original_start = threading.Thread.start

    def fail_start(self):
        if self.name.startswith("iris-workspace-"):
            raise RuntimeError("thread unavailable")
        original_start(self)

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail_start)
        response = client.post("/api/workspace/backups")
    assert response.status_code == 409
    operations = client.get("/api/workspace/operations").json()
    assert operations[0]["status"] == "failed"
    assert not client.app.state.workspace_operations.gate.frozen
    assert client.delete(f"/api/workspace/operations/{operations[0]['id']}").status_code == 204
    assert start_backup(client)["status"] == "succeeded"


def test_receipt_write_failure_keeps_completed_result_available(client, monkeypatch):
    manager = client.app.state.workspace_operations
    original_write = manager._write

    def fail_receipt(record):
        if record["status"] == "succeeded":
            raise OSError("receipt disk unavailable")
        original_write(record)

    monkeypatch.setattr(manager, "_write", fail_receipt)
    backup = start_backup(client)
    assert backup["status"] == "succeeded"
    assert backup["warning"]
    assert backup["result"]["archive_sha256"]
    assert client.get(backup["result"]["download_url"]).status_code == 200
    assert manager.list()[0] == backup
    assert not manager.gate.frozen


def test_recovery_failure_closes_job_manager(tmp_path, monkeypatch):
    app = create_app(tmp_path / "recovery-failure")
    calls = []
    monkeypatch.setattr(app.state.jobs, "start", lambda: calls.append("start"))
    monkeypatch.setattr(app.state.jobs, "close", lambda: calls.append("close"))

    def fail_recovery():
        raise workspace_operations.ArchiveError("Invalid receipt")

    monkeypatch.setattr(app.state.workspace_operations, "recover", fail_recovery)
    with pytest.raises(workspace_operations.ArchiveError, match="Invalid receipt"):
        with TestClient(app, base_url=BASE_URL):
            pass
    assert calls == ["start", "close"]


@pytest.mark.parametrize("route", ["/api/workspace/backups", "/api/workspace/restore-inspections"])
def test_remote_origin_cannot_start_transfer(client, route):
    assert client.post(route, headers={"Origin": "https://example.org"}).status_code == 403
    assert client.get("/api/workspace/operations").json() == []


def test_unknown_operation_and_invalid_receipt_are_explicit(client):
    assert client.get("/api/workspace/operations/not-an-id").status_code == 404
    assert client.post("/api/workspace/operations/" + "a" * 32 + "/cancel").status_code == 404
    manager = client.app.state.workspace_operations
    directory = manager.root / ("b" * 32)
    directory.mkdir(parents=True)
    (directory / "operation.json").write_text(json.dumps({"id": "c" * 32}))
    assert client.get("/api/workspace/operations").status_code == 422
