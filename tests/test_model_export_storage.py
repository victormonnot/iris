"""Additive portable-model storage, ownership and archive retention contracts."""

import hashlib
import json
import shutil
import sqlite3
import zipfile

import pytest
import test_workspace_archive as fixtures

from iris.projects import create_project, project_records, record_project
from iris.store import (
    DINOX_TABLES,
    MODEL_EXPORT_TABLES,
    SCHEMA_V16,
    SCHEMA_V17,
    SCHEMA_VERSION,
    TABLES,
    TEMPORAL_DETECTION_TABLES,
    TEMPORAL_TABLES,
    TRACKING_QUALITY_TABLES,
    TRAINING_CHECKPOINT_TABLES,
    Store,
    new_id,
    now,
)
from iris.workspace_archive import (
    ArchiveError,
    ArchiveLimitError,
    _inventory,
    allowed_artifact_path,
    create_archive,
    validate_database,
)
from iris.workspace_restore import inspect_archive, restore_archive

OLD_TABLES = (
    TABLES
    - MODEL_EXPORT_TABLES
    - TRAINING_CHECKPOINT_TABLES
    - DINOX_TABLES
    - TEMPORAL_TABLES
    - TEMPORAL_DETECTION_TABLES
    - TRACKING_QUALITY_TABLES
)


def rows(root, tables=OLD_TABLES):
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        return {
            table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY id")]
            for table in sorted(tables)
        }


def artifacts(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("iris.sqlite3")
    }


@pytest.fixture
def workspace(tmp_path):
    store = fixtures.workspace.__wrapped__(tmp_path)
    fixtures._all_references(store)
    return store


@pytest.fixture
def schema16(workspace, tmp_path):
    root = tmp_path / "schema16"
    root.mkdir()
    for path in workspace.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMA_V16)
        for table, records in rows(workspace.root).items():
            for record in records:
                connection.execute(
                    f"INSERT INTO {table} ({','.join(record)}) "
                    f"VALUES ({','.join('?' for _ in record)})",
                    list(record.values()),
                )
        # Preserve raw serialization rather than only decoded values.
        connection.execute(
            "UPDATE training_runs SET config=?", (json.dumps({"historical": True}, indent=3),)
        )
        connection.execute("PRAGMA user_version=16")
    return root


def test_schema17_adds_only_export_tables_preserving_raw_rows_and_files(schema16):
    original, files = rows(schema16), artifacts(schema16)
    for _ in range(2):
        store = Store(schema16)
        assert rows(schema16) == original
        assert artifacts(schema16) == files
        assert all(store.list(table) == [] for table in MODEL_EXPORT_TABLES)
        with store.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema17_failed_migration_rolls_back_new_tables_and_version(schema16):
    with sqlite3.connect(schema16 / "iris.sqlite3") as connection:
        connection.execute("UPDATE trained_models SET training_id='missing'")
    original, files = rows(schema16), artifacts(schema16)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(schema16)
    assert rows(schema16) == original
    assert artifacts(schema16) == files
    with sqlite3.connect(schema16 / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 16
        assert {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        } == OLD_TABLES


@pytest.mark.parametrize("version", [16, 17])
def test_schema16_and17_archives_preserve_exact_saved_database(
    schema16, tmp_path, monkeypatch, version
):
    if version == 17:
        with sqlite3.connect(schema16 / "iris.sqlite3") as connection:
            connection.executescript(SCHEMA_V17)
            connection.execute("PRAGMA user_version=17")
    tables = (
        OLD_TABLES
        if version == 16
        else TABLES
        - TRAINING_CHECKPOINT_TABLES
        - DINOX_TABLES
        - TEMPORAL_TABLES
        - TEMPORAL_DETECTION_TABLES
        - TRACKING_QUALITY_TABLES
    )
    original = rows(schema16, tables)
    saved = create_archive(schema16, tmp_path / "historical.zip")
    assert saved["manifest"]["schema_version"] == version
    assert set(saved["manifest"]["counts"]) == tables

    def forbidden(*_args, **_kwargs):
        pytest.fail("Archive inspection must never initialize or migrate Store")

    restored = tmp_path / "restored"
    with monkeypatch.context() as context:
        context.setattr(Store, "__init__", forbidden)
        inspected = inspect_archive(saved["path"])
        restore_archive(
            saved["path"], restored, expected_archive_sha256=inspected["archive_sha256"]
        )
    with zipfile.ZipFile(saved["path"]) as archive:
        for item in saved["manifest"]["files"]:
            assert (restored / item["path"]).read_bytes() == archive.read(item["path"])
    Store(restored)
    assert rows(restored, tables) == original


def export_row(store, *, with_job=True):
    identifier = new_id()
    job = store.insert(
        "jobs",
        {
            "id": new_id(),
            "kind": "model_export",
            "status": "failed",
            "params": {"export_id": identifier},
            "created_at": now(),
        },
    )
    return store.insert(
        "model_exports",
        {
            "id": identifier,
            "trained_model_id": store.list("trained_models")[0]["id"],
            "evaluation_id": store.list("evaluations")[0]["id"],
            "name": "Synthetic storage contract",
            "config": {"frozen": [1, "two", None]},
            "request_id": new_id(),
            "job_id": job["id"] if with_job else None,
            "created_at": now(),
        },
    )


def test_export_measurement_json_uniqueness_and_project_ownership(workspace):
    project = create_project(workspace, name="Owning project")
    model = workspace.list("trained_models")[0]
    training = workspace.get("training_runs", model["training_id"])
    workspace.update("dataset_versions", training["dataset_id"], {"project_id": project["id"]})
    export = export_row(workspace)
    measurement = workspace.insert(
        "model_export_measurements",
        {
            "id": new_id(),
            "export_id": export["id"],
            "payload": {"samples": [1, {"nested": True}]},
            "summary": {"parity": "passed"},
            "fingerprint": "a" * 64,
            "created_at": now(),
        },
    )
    assert workspace.get("model_export_measurements", measurement["id"]) == measurement
    assert workspace.get("model_exports", export["id"])["config"] == export["config"]
    for table, row in (("model_exports", export), ("model_export_measurements", measurement)):
        assert record_project(workspace, table, row) == project["id"]
        assert project_records(workspace, table, "default") == []
        assert project_records(workspace, table, project["id"]) == [row]
    job = workspace.get("jobs", export["job_id"])
    assert record_project(workspace, "jobs", job) == project["id"]
    job = workspace.update("jobs", job["id"], {"params": {}})
    assert record_project(workspace, "jobs", job) == project["id"]
    with pytest.raises(sqlite3.IntegrityError):
        workspace.insert("model_exports", {**export, "id": new_id(), "job_id": None})
    with pytest.raises(sqlite3.IntegrityError):
        workspace.insert("model_export_measurements", {**measurement, "id": new_id()})


@pytest.mark.parametrize(
    "path,allowed",
    [
        ("model_exports/abc123/model.zip", True),
        ("model_exports/abc123/run.py", False),
        ("model_exports/abc123/extra/model.zip", False),
        ("model_exports/../model.zip", False),
        ("model_exports/.staging/model.zip", False),
        ("model_exports/abc123/model.zip.part", False),
    ],
)
def test_only_published_model_package_paths_are_archived(path, allowed):
    assert allowed_artifact_path(path) is allowed


@pytest.mark.parametrize(
    "damage", ["publication", "path", "job_kind", "job_owner", "missing_job", "project"]
)
def test_archive_rejects_invalid_export_storage_before_protocol_validation(
    workspace, monkeypatch, damage
):
    export = export_row(workspace)
    if damage == "publication":
        workspace.update("model_exports", export["id"], {"manifest": {}})
    elif damage in {"path", "missing_job"}:
        workspace.update(
            "model_exports",
            export["id"],
            {
                "path": "model_exports/another/model.zip",
                "manifest": {},
                "manifest_sha256": "a" * 64,
                "archive_sha256": "b" * 64,
                **({"job_id": None} if damage == "missing_job" else {}),
            },
        )
    elif damage in {"job_kind", "job_owner"}:
        workspace.update(
            "jobs",
            export["job_id"],
            {"kind": "train"} if damage == "job_kind" else {"params": {"export_id": "other"}},
        )
    else:
        project = create_project(workspace, name="Unrelated project")
        dataset = workspace.list("dataset_versions")[0]
        other = workspace.insert(
            "dataset_versions", {**dataset, "id": new_id(), "project_id": project["id"]}
        )
        workspace.update("evaluations", export["evaluation_id"], {"dataset_id": other["id"]})
        # Keep the valid original artifact identity; the source mismatch is checked below.
        workspace.update("dataset_versions", other["id"], {"path": dataset["path"]})

    def unexpected(*_args, **_kwargs):
        pytest.fail("Storage ownership and publication must be validated before package protocol")

    monkeypatch.setattr("iris.model_exports.validate_export_archive", unexpected)
    if damage == "project":
        # Directly exercise ownership independently of the intentionally duplicated dataset path.
        from iris.workspace_archive import _validate_model_exports

        with workspace.connect() as connection, pytest.raises(ArchiveError, match="projects"):
            _validate_model_exports(connection, workspace.root, lambda *_args: None)
    else:
        with pytest.raises(ArchiveError, match="Model export"):
            validate_database(workspace.root, _inventory(workspace.root)[0])


@pytest.mark.parametrize("over_limit", [False, True])
def test_archive_inspection_retains_bounded_bundle_for_offline_protocol_check(
    workspace, tmp_path, monkeypatch, over_limit
):
    export = export_row(workspace)
    relative = f"model_exports/{export['id']}/model.zip"
    content = b"synthetic package handled by the protocol validator"
    path = workspace.root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    workspace.update(
        "model_exports",
        export["id"],
        {
            "path": relative,
            "manifest": {},
            "manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
            "archive_sha256": hashlib.sha256(content).hexdigest(),
        },
    )
    checks = []

    def check(row, *, connection, root):
        assert (root / row["path"]).read_bytes() == content
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM model_exports").fetchone()[0] == 1
        checks.append(root)
        if root != workspace.root:
            assert not list(root.glob("models/trained/*.pth"))
            assert list(root.glob("datasets/*/manifest.json"))
            assert not list(root.glob("datasets/*/images/*.png"))

    monkeypatch.setattr("iris.model_exports.validate_export_archive", check)
    saved = create_archive(workspace.root, tmp_path / "exports.zip")
    if over_limit:
        monkeypatch.setattr("iris.workspace_restore.MAX_MODEL_EXPORT_BYTES", len(content) - 1)
        with pytest.raises(ArchiveLimitError, match="standalone model package"):
            inspect_archive(saved["path"])
        assert all(root == workspace.root for root in checks)
        return
    checked = inspect_archive(saved["path"])
    assert checked["verified"]
    assert any(root != workspace.root for root in checks)
