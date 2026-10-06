"""Legacy workspaces and archives retain exact rows and artifacts across schema 14."""

import json
import shutil
import sqlite3
import zipfile

import pytest
import test_workspace_archive as archive_fixtures
import test_workspace_restore as restore_fixtures

from iris.projects import create_project
from iris.store import (
    BENCHMARK_TABLES,
    DEFAULT_PROJECT_ID,
    DINOX_TABLES,
    MODEL_EXPORT_TABLES,
    SCHEMA_V13,
    SCHEMA_VERSION,
    TABLES,
    TRAINING_CHECKPOINT_TABLES,
    Store,
)
from iris.taxonomies import TAXONOMY, current_taxonomy, get_taxonomy, publish_taxonomy
from iris.workspace_archive import ArchiveError, create_archive, preview_workspace
from iris.workspace_restore import inspect_archive, restore_archive

SCHEMA13_TABLES = (
    TABLES
    - BENCHMARK_TABLES
    - MODEL_EXPORT_TABLES
    - TRAINING_CHECKPOINT_TABLES
    - DINOX_TABLES
    - {"taxonomy_versions"}
)


def rows(root, tables=SCHEMA13_TABLES):
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
def schema13(tmp_path):
    source = archive_fixtures.workspace.__wrapped__(tmp_path)
    archive_fixtures._all_references(source)
    root = tmp_path / "schema13"
    root.mkdir()
    for path in source.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMA_V13)
        for table, records in rows(source.root).items():
            for record in records:
                if table == "frames":
                    record.pop("taxonomy_id")
                connection.execute(
                    f"INSERT INTO {table} ({','.join(record)}) "
                    f"VALUES ({','.join('?' for _ in record)})",
                    list(record.values()),
                )
        connection.execute("PRAGMA user_version=13")
    return root


def assert_migrated_rows(root, expected):
    current = rows(root)
    for frame in current["frames"]:
        assert frame.pop("taxonomy_id") == TAXONOMY["id"]
    assert current == expected


def test_schema13_adds_only_frame_taxonomy_and_empty_versions(schema13):
    expected, original_artifacts = rows(schema13), artifacts(schema13)
    assert expected["annotation_revisions"] and expected["experiment_reports"]
    store = Store(schema13)
    assert_migrated_rows(schema13, expected)
    assert store.list("taxonomy_versions") == []
    assert artifacts(schema13) == original_artifacts
    assert Store(schema13).list("projects") == store.list("projects")
    assert_migrated_rows(schema13, expected)
    with store.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema13_failed_migration_rolls_back_frame_column_and_new_table(schema13):
    with sqlite3.connect(schema13 / "iris.sqlite3") as connection:
        connection.execute("UPDATE assets SET session_id='missing'")
    expected = rows(schema13)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(schema13)
    assert rows(schema13) == expected
    with sqlite3.connect(schema13 / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 13
        assert "taxonomy_id" not in {
            row[1] for row in connection.execute("PRAGMA table_info(frames)")
        }
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='taxonomy_versions'"
            ).fetchone()
            is None
        )


@pytest.mark.parametrize("version", [13, SCHEMA_VERSION])
def test_taxonomy_archive_restores_original_schema_and_all_saved_snapshots(
    schema13, tmp_path, monkeypatch, version
):
    original_artifacts = artifacts(schema13)
    if version == SCHEMA_VERSION:
        store = Store(schema13)
        first = publish_taxonomy(
            store,
            DEFAULT_PROJECT_ID,
            expected_taxonomy_id=TAXONOMY["id"],
            classes=[{"id": "helmet", "name": "Helmet", "definition": "A helmet."}],
        )
        second = publish_taxonomy(
            store,
            DEFAULT_PROJECT_ID,
            expected_taxonomy_id=first["id"],
            classes=[{"id": "helmet", "name": "Helmet", "definition": "A worn helmet."}],
        )
        frame = store.list("frames")[0]
        store.update("frames", frame["id"], {"taxonomy_id": first["id"]})
    tables = SCHEMA13_TABLES if version == 13 else TABLES
    expected = rows(schema13, tables)
    archive = create_archive(schema13, tmp_path / "backup.zip")
    assert archive["manifest"]["schema_version"] == version
    assert preview_workspace(schema13)["schema_version"] == version
    target = tmp_path / "restored"

    def forbidden(*_args, **_kwargs):
        pytest.fail("Archive inspection/restoration must not migrate saved databases")

    with monkeypatch.context() as context:
        context.setattr(Store, "__init__", forbidden)
        inspected = inspect_archive(archive["path"])
        restore_archive(
            archive["path"], target, expected_archive_sha256=inspected["archive_sha256"]
        )
    with zipfile.ZipFile(archive["path"]) as saved:
        for item in archive["manifest"]["files"]:
            assert (target / item["path"]).read_bytes() == saved.read(item["path"])
    assert rows(target, tables) == expected
    reopened = Store(target)
    if version == 13:
        assert_migrated_rows(target, expected)
        assert reopened.list("taxonomy_versions") == []
    else:
        assert rows(target, tables) == expected
        assert get_taxonomy(reopened, first["id"]) == first
        assert current_taxonomy(reopened, DEFAULT_PROJECT_ID) == second
        assert reopened.get("frames", frame["id"])["taxonomy_id"] == first["id"]
    assert artifacts(target) == original_artifacts


@pytest.mark.parametrize("damage", ["version", "columns", "tables", "counts"])
def test_archive_schema13_validation_stays_strict(schema13, tmp_path, damage):
    if damage == "counts":
        archive = create_archive(schema13, tmp_path / "original.zip")
        damaged = tmp_path / "damaged.zip"
        with zipfile.ZipFile(archive["path"]) as source, zipfile.ZipFile(damaged, "w") as target:
            for item in source.infolist():
                content = source.read(item.filename)
                if item.filename == "manifest.json":
                    manifest = json.loads(content)
                    manifest["counts"]["taxonomy_versions"] = 0
                    content = json.dumps(manifest).encode()
                target.writestr(item, content)
        with pytest.raises(ArchiveError, match="counts"):
            inspect_archive(damaged)
    else:
        with sqlite3.connect(schema13 / "iris.sqlite3") as connection:
            if damage == "version":
                connection.execute("PRAGMA user_version=14")
            elif damage == "columns":
                connection.execute("ALTER TABLE frames ADD COLUMN taxonomy_id TEXT")
            else:
                connection.execute("CREATE TABLE taxonomy_versions (id TEXT)")
        assert not preview_workspace(schema13)["can_create"]


@pytest.mark.parametrize(
    "damage",
    [
        "snapshot_id",
        "snapshot_classes",
        "snapshot_parent",
        "snapshot_version",
        "removed_id",
        "project_missing",
        "project_foreign",
        "frame_missing",
        "frame_foreign",
        "annotation_missing",
        "annotation_foreign",
    ],
)
def test_archives_reject_broken_taxonomy_snapshots_and_cross_project_references(
    schema13, tmp_path, damage
):
    store = Store(schema13)
    classes = [{"id": "helmet", "name": "Helmet", "definition": "A helmet."}]
    first = publish_taxonomy(
        store, DEFAULT_PROJECT_ID, expected_taxonomy_id=TAXONOMY["id"], classes=classes
    )
    second = publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=first["id"],
        classes=classes + [{"id": "cone", "name": "Cone", "definition": "A traffic cone."}],
    )
    other = create_project(store, name="Other project")
    foreign = publish_taxonomy(
        store, other["id"], expected_taxonomy_id=TAXONOMY["id"], classes=classes
    )
    with store.connect() as connection:
        if damage.startswith("snapshot_") or damage == "removed_id":
            if damage == "snapshot_id":
                second["id"] = "wrong-id"
            elif damage == "snapshot_classes":
                second["classes"][0]["definition"] = ""
            elif damage == "snapshot_parent":
                second["parent_id"] = foreign["id"]
            elif damage == "snapshot_version":
                second["version"] = True
            else:
                second["classes"] = second["classes"][1:]
            connection.execute(
                "UPDATE taxonomy_versions SET snapshot=? WHERE project_id=? AND version=3",
                (json.dumps(second), DEFAULT_PROJECT_ID),
            )
        else:
            identifier = foreign["id"] if damage.endswith("foreign") else "missing-taxonomy"
            if damage.startswith("project_"):
                connection.execute(
                    "UPDATE projects SET taxonomy_id=? WHERE id=?", (identifier, DEFAULT_PROJECT_ID)
                )
            elif damage.startswith("frame_"):
                connection.execute("UPDATE frames SET taxonomy_id=?", (identifier,))
            else:
                connection.execute("UPDATE annotation_revisions SET taxonomy_id=?", (identifier,))
    preview = preview_workspace(schema13)
    assert not preview["can_create"]
    assert any("taxonomy" in issue for issue in preview["blocking_issues"])
    archive = restore_fixtures._write_archive(
        tmp_path / "corrupted.zip", restore_fixtures._payload(store)
    )
    with pytest.raises(ArchiveError, match="taxonomy"):
        inspect_archive(archive)
    with pytest.raises(ArchiveError, match="taxonomy"):
        restore_fixtures._restore(archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
