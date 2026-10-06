"""Project migration preserves historical rows, provenance and frozen artifacts."""

import hashlib
import json
import shutil
import sqlite3
import zipfile

import pytest
import test_workspace_archive as archive_fixtures

from iris import workspace_archive, workspace_restore
from iris.store import (
    BENCHMARK_TABLES,
    DEFAULT_PROJECT_ID,
    DINOX_TABLES,
    MODEL_EXPORT_TABLES,
    PROJECT_TABLES,
    SCHEMA_V12,
    SCHEMA_VERSION,
    TABLES,
    TRAINING_CHECKPOINT_TABLES,
    Store,
    new_id,
    now,
)

LEGACY_TABLES = (
    TABLES
    - BENCHMARK_TABLES
    - MODEL_EXPORT_TABLES
    - TRAINING_CHECKPOINT_TABLES
    - DINOX_TABLES
    - {"projects", "taxonomy_versions"}
)


def _rows(root, tables=LEGACY_TABLES):
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        return {
            table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY id")]
            for table in sorted(tables)
        }


def _artifacts(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("iris.sqlite3")
    }


@pytest.fixture
def legacy_workspace(tmp_path):
    source = archive_fixtures.workspace.__wrapped__(tmp_path)
    archive_fixtures._all_references(source)
    frame = source.list("frames")[0]
    job = archive_fixtures._job(source)
    suggestion = source.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frame["id"],
            "job_id": job["id"],
            "kind": "multimodal",
            "label": "person",
            "box": [1, 2, 3, 4],
            "metadata": {"provider": "historical-fixture", "original_label": "person"},
            "created_at": now(),
        },
    )
    source.insert(
        "assistance_records",
        {
            "id": new_id(),
            "frame_id": frame["id"],
            "job_id": job["id"],
            "config": {"provider": "historical-fixture"},
            "candidates": [{"id": suggestion["id"], "label": "person"}],
            "prompt": "Original prompt, accents: vérification",
            "raw_response": {"suggestion_id": suggestion["id"], "accepted": True},
            "created_at": now(),
        },
    )
    with source.connect() as connection:
        # Preserve exact serialized bytes as well as decoded identities and values.
        connection.execute(
            "UPDATE annotation_revisions SET decisions=?, notes=? WHERE frame_id=?",
            (
                json.dumps({suggestion["id"]: "accepted"}, indent=2),
                "Historical human review",
                frame["id"],
            ),
        )
    root = tmp_path / "legacy"
    root.mkdir()
    for path in source.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMA_V12)
        for table, rows in _rows(source.root).items():
            for row in rows:
                row.pop("project_id", None)
                if table == "frames":
                    row.pop("taxonomy_id")
                connection.execute(
                    f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
                    list(row.values()),
                )
        connection.execute("PRAGMA user_version=12")
    return root


def _assert_legacy_rows(root, expected):
    current = _rows(root)
    for table, rows in current.items():
        if table == "frames":
            for row in rows:
                assert row.pop("taxonomy_id") == "iris-objects-v1"
        if table in PROJECT_TABLES:
            for row in rows:
                assert row.pop("project_id") == DEFAULT_PROJECT_ID
    assert current == expected


def test_projects_migration_preserves_history_json_bytes_and_artifacts(legacy_workspace):
    expected = _rows(legacy_workspace)
    artifacts = _artifacts(legacy_workspace)
    assert expected["annotation_revisions"] and expected["assistance_records"]
    assert expected["dataset_versions"] and expected["trained_models"]
    assert expected["experiment_reports"]
    store = Store(legacy_workspace)
    project = store.get("projects", DEFAULT_PROJECT_ID)
    assert project["name"] == "Default project"
    assert project["taxonomy_id"] == "iris-objects-v1"
    _assert_legacy_rows(legacy_workspace, expected)
    assert _artifacts(legacy_workspace) == artifacts
    with store.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        for table in PROJECT_TABLES:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(f"UPDATE {table} SET project_id='missing'")
    reopened = Store(legacy_workspace)
    assert reopened.list("projects") == [project]
    _assert_legacy_rows(legacy_workspace, expected)
    assert _artifacts(legacy_workspace) == artifacts


def test_projects_migration_rolls_back_all_schema_and_row_changes(legacy_workspace):
    with sqlite3.connect(legacy_workspace / "iris.sqlite3") as connection:
        connection.execute("UPDATE assets SET session_id='missing'")
    expected = _rows(legacy_workspace)
    artifacts = _artifacts(legacy_workspace)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(legacy_workspace)
    assert _rows(legacy_workspace) == expected
    assert _artifacts(legacy_workspace) == artifacts
    with sqlite3.connect(legacy_workspace / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 12
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name='projects'").fetchone()
            is None
        )
        for table in PROJECT_TABLES:
            assert "project_id" not in {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }


@pytest.mark.parametrize("version", [12, SCHEMA_VERSION])
def test_archives_restore_each_supported_schema_unchanged_before_open(
    legacy_workspace, tmp_path, monkeypatch, version
):
    expected = _rows(legacy_workspace)
    if version == SCHEMA_VERSION:
        store = Store(legacy_workspace)
        project = store.insert(
            "projects",
            {
                "id": "second",
                "name": "Other subject",
                "description": "Archive preserves every project",
                "taxonomy_id": "iris-objects-v1",
                "created_at": now(),
            },
        )
        store.insert(
            "sessions",
            {
                "id": "second-session",
                "project_id": project["id"],
                "name": "Second session",
                "scene_group": "other",
                "created_at": now(),
            },
        )
    before = _rows(legacy_workspace, LEGACY_TABLES if version == 12 else TABLES)
    artifacts = _artifacts(legacy_workspace)
    preview = workspace_archive.preview_workspace(legacy_workspace)
    assert preview["schema_version"] == version
    assert preview["can_create"]
    result = workspace_archive.create_archive(legacy_workspace, tmp_path / f"schema{version}.zip")
    archive = result["path"]
    assert result["manifest"]["schema_version"] == version
    assert set(result["manifest"]["counts"]) == (LEGACY_TABLES if version == 12 else TABLES)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Inspection and restoration must not initialize or migrate Store")

    target = tmp_path / "restored"
    with monkeypatch.context() as context:
        context.setattr(Store, "__init__", forbidden)
        inspection = workspace_restore.inspect_archive(archive)
        workspace_restore.restore_archive(
            archive, target, expected_archive_sha256=inspection["archive_sha256"]
        )
    with zipfile.ZipFile(archive) as saved:
        for item in result["manifest"]["files"]:
            assert (target / item["path"]).read_bytes() == saved.read(item["path"])
    assert _rows(target, LEGACY_TABLES if version == 12 else TABLES) == before
    store = Store(target)
    if version == 12:
        _assert_legacy_rows(target, expected)
        assert len(store.list("projects")) == 1
    else:
        assert _rows(target, TABLES) == before
        assert store.get("sessions", "second-session")["project_id"] == "second"
    assert _artifacts(target) == artifacts
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["archive_sha256"]


@pytest.mark.parametrize("version", [12, SCHEMA_VERSION])
@pytest.mark.parametrize("damage", ["wrong_version", "missing_index", "extra_column"])
def test_legacy_support_keeps_version_specific_structural_validation(
    legacy_workspace, version, damage
):
    if version == SCHEMA_VERSION:
        Store(legacy_workspace)
    with sqlite3.connect(legacy_workspace / "iris.sqlite3") as connection:
        if damage == "wrong_version":
            connection.execute(f"PRAGMA user_version={13 if version == 12 else 12}")
        elif damage == "missing_index":
            connection.execute("DROP INDEX frames_session")
        else:
            connection.execute("ALTER TABLE frames ADD COLUMN unexpected TEXT")
    preview = workspace_archive.preview_workspace(legacy_workspace)
    assert not preview["can_create"]
    assert any("schema" in issue or "tables" in issue for issue in preview["blocking_issues"])
