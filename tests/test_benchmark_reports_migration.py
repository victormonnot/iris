"""Schema 16 only adds immutable reports and preserves schema-15 archive bytes."""

import json
import shutil
import sqlite3
import zipfile

import pytest
import test_benchmarks_archive as archive_fixtures

from iris.store import MODEL_EXPORT_TABLES, SCHEMA_V15, SCHEMA_V16, SCHEMA_VERSION, TABLES, Store
from iris.workspace_archive import create_archive, preview_workspace
from iris.workspace_restore import inspect_archive, restore_archive

benchmark_workspace = archive_fixtures.benchmark_workspace
OLD_TABLES = TABLES - MODEL_EXPORT_TABLES - {"benchmark_reports"}


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
def schema15(benchmark_workspace, tmp_path):
    source = benchmark_workspace[0]
    # Raw JSON formatting is part of persisted history and must survive migration.
    output = source.list("benchmark_outputs")[0]
    with source.connect() as connection:
        connection.execute(
            "UPDATE benchmark_outputs SET raw_response=? WHERE id=?",
            (json.dumps(output["raw_response"], indent=3), output["id"]),
        )
    root = tmp_path / "schema15"
    root.mkdir()
    for path in source.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMA_V15)
        for table, records in rows(source.root).items():
            for record in records:
                connection.execute(
                    f"INSERT INTO {table} ({','.join(record)}) "
                    f"VALUES ({','.join('?' for _ in record)})",
                    list(record.values()),
                )
        connection.execute("PRAGMA user_version=15")
    return root


def test_schema16_is_additive_and_repeatable_with_raw_history_and_artifacts(schema15):
    original, files = rows(schema15), artifacts(schema15)
    for _ in range(2):
        store = Store(schema15)
        assert rows(schema15) == original
        assert artifacts(schema15) == files
        assert store.list("benchmark_reports") == []
        with store.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema16_failure_rolls_back_table_and_version(schema15):
    with sqlite3.connect(schema15 / "iris.sqlite3") as connection:
        connection.execute("UPDATE benchmark_trials SET benchmark_id='missing'")
    original = rows(schema15)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(schema15)
    assert rows(schema15) == original
    with sqlite3.connect(schema15 / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 15
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='benchmark_reports'"
            ).fetchone()
            is None
        )


@pytest.mark.parametrize("version", [15, 16, SCHEMA_VERSION])
def test_archives_restore_original_database_before_schema16_migration(
    schema15, tmp_path, monkeypatch, version
):
    if version == 16:
        with sqlite3.connect(schema15 / "iris.sqlite3") as connection:
            connection.executescript(SCHEMA_V16)
            connection.execute("PRAGMA user_version=16")
    elif version == SCHEMA_VERSION:
        Store(schema15)
    tables = (
        OLD_TABLES if version == 15 else TABLES - MODEL_EXPORT_TABLES if version == 16 else TABLES
    )
    original, files = rows(schema15, tables), artifacts(schema15)
    saved = create_archive(schema15, tmp_path / "historical.zip")
    assert saved["manifest"]["schema_version"] == version
    assert set(saved["manifest"]["counts"]) == tables

    def forbidden(*_args, **_kwargs):
        pytest.fail("Archive inspection and restoration must never open or migrate Store")

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
    assert rows(restored, tables) == original
    Store(restored)
    assert rows(restored, tables) == original
    assert artifacts(restored) == files


@pytest.mark.parametrize("damage", ["version", "column", "index", "new_table"])
def test_schema15_archive_validation_remains_exact(schema15, damage):
    with sqlite3.connect(schema15 / "iris.sqlite3") as connection:
        if damage == "version":
            connection.execute("PRAGMA user_version=16")
        elif damage == "column":
            connection.execute("ALTER TABLE benchmarks ADD COLUMN extra TEXT")
        elif damage == "index":
            connection.execute("DROP INDEX benchmark_trials_benchmark")
        else:
            connection.execute("CREATE TABLE benchmark_reports(id TEXT)")
    assert not preview_workspace(schema15)["can_create"]
