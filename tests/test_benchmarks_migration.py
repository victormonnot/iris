"""Schema 15 adds isolated benchmark records without rewriting historical data."""

import json
import shutil
import sqlite3
import zipfile

import pytest
import test_workspace_archive as archive_fixtures

from iris.projects import create_project, project_records, record_project
from iris.store import (
    BENCHMARK_TABLES,
    DEFAULT_PROJECT_ID,
    DINOX_TABLES,
    MODEL_EXPORT_TABLES,
    SCHEMA_V14,
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
from iris.taxonomies import TAXONOMY, publish_taxonomy
from iris.workspace_archive import create_archive, preview_workspace
from iris.workspace_restore import inspect_archive, restore_archive

SCHEMA14_TABLES = (
    TABLES
    - BENCHMARK_TABLES
    - MODEL_EXPORT_TABLES
    - TRAINING_CHECKPOINT_TABLES
    - DINOX_TABLES
    - TEMPORAL_TABLES
    - TEMPORAL_DETECTION_TABLES
    - TRACKING_QUALITY_TABLES
)


def rows(root, tables=SCHEMA14_TABLES):
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
def schema14(tmp_path):
    source = archive_fixtures.workspace.__wrapped__(tmp_path)
    archive_fixtures._all_references(source)
    taxonomy = publish_taxonomy(
        source,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[{"id": "helmet", "name": "Helmet", "definition": "A worn safety helmet."}],
    )
    # Preserve noncanonical JSON bytes, not only the decoded taxonomy object.
    with source.connect() as connection:
        connection.execute(
            "UPDATE taxonomy_versions SET snapshot=? WHERE id=?",
            (json.dumps(taxonomy, indent=3, ensure_ascii=False), taxonomy["id"]),
        )
    root = tmp_path / "schema14"
    root.mkdir()
    for path in source.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMA_V14)
        for table, records in rows(source.root).items():
            for record in records:
                connection.execute(
                    f"INSERT INTO {table} ({','.join(record)}) "
                    f"VALUES ({','.join('?' for _ in record)})",
                    list(record.values()),
                )
        connection.execute("PRAGMA user_version=14")
    return root


def test_schema15_is_additive_repeatable_and_preserves_every_old_row_and_artifact(schema14):
    original, files = rows(schema14), artifacts(schema14)
    store = Store(schema14)
    assert rows(schema14) == original
    assert artifacts(schema14) == files
    assert all(store.list(table) == [] for table in BENCHMARK_TABLES)
    with store.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    Store(schema14)
    assert rows(schema14) == original
    assert artifacts(schema14) == files


def test_schema15_failure_rolls_back_all_new_tables_and_version(schema14):
    with sqlite3.connect(schema14 / "iris.sqlite3") as connection:
        connection.execute("UPDATE assets SET session_id='missing'")
    original = rows(schema14)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(schema14)
    assert rows(schema14) == original
    with sqlite3.connect(schema14 / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 14
        assert {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        } == SCHEMA14_TABLES


@pytest.mark.parametrize("version", [14, SCHEMA_VERSION])
def test_schema14_and15_archive_restore_database_bytes_before_any_migration(
    schema14, tmp_path, monkeypatch, version
):
    if version == SCHEMA_VERSION:
        Store(schema14)
    tables = SCHEMA14_TABLES if version == 14 else TABLES
    original, files = rows(schema14, tables), artifacts(schema14)
    archived = create_archive(schema14, tmp_path / "backup.zip")
    assert archived["manifest"]["schema_version"] == version
    assert set(archived["manifest"]["counts"]) == tables

    def forbidden(*_args, **_kwargs):
        pytest.fail("Inspection and restoration must preserve the archived schema unchanged")

    restored = tmp_path / "restored"
    with monkeypatch.context() as context:
        context.setattr(Store, "__init__", forbidden)
        checked = inspect_archive(archived["path"])
        restore_archive(
            archived["path"], restored, expected_archive_sha256=checked["archive_sha256"]
        )
    with zipfile.ZipFile(archived["path"]) as archive:
        for item in archived["manifest"]["files"]:
            assert (restored / item["path"]).read_bytes() == archive.read(item["path"])
    assert rows(restored, tables) == original
    Store(restored)
    assert rows(restored, tables) == original
    assert artifacts(restored) == files


@pytest.mark.parametrize("damage", ["version", "column", "index", "new_table"])
def test_schema14_archive_validation_remains_exact(schema14, damage):
    with sqlite3.connect(schema14 / "iris.sqlite3") as connection:
        if damage == "version":
            connection.execute("PRAGMA user_version=15")
        elif damage == "column":
            connection.execute("ALTER TABLE frames ADD COLUMN extra TEXT")
        elif damage == "index":
            connection.execute("DROP INDEX taxonomy_versions_project")
        else:
            connection.execute("CREATE TABLE benchmarks(id TEXT)")
    assert not preview_workspace(schema14)["can_create"]


def test_benchmark_records_follow_project_and_job_ownership_and_enforce_uniqueness(tmp_path):
    store = archive_fixtures.workspace.__wrapped__(tmp_path)
    other = create_project(store, name="Benchmark owner")
    benchmark = store.insert(
        "benchmarks",
        {
            "id": new_id(),
            "project_id": other["id"],
            "name": "Owned benchmark",
            "path": "benchmarks/fixture/manifest.json",
            "manifest_sha256": "a" * 64,
            "summary": {"reference": "frozen"},
            "created_at": now(),
        },
    )
    config = store.insert(
        "benchmark_configs",
        {
            "id": new_id(),
            "benchmark_id": benchmark["id"],
            "name": "Manual",
            "approach": "manual",
            "config": {},
            "fingerprint": "b" * 64,
            "created_at": now(),
        },
    )
    trial_id = new_id()
    job = store.insert(
        "jobs",
        {
            "id": new_id(),
            "kind": "benchmark",
            "status": "succeeded",
            "params": {"trial_id": trial_id},
            "created_at": now(),
        },
    )
    trial = store.insert(
        "benchmark_trials",
        {
            "id": trial_id,
            "benchmark_id": benchmark["id"],
            "config_id": config["id"],
            "split": "tuning",
            "config": {"immutable": True},
            "job_id": job["id"],
            "created_at": now(),
        },
    )
    output = store.insert(
        "benchmark_outputs",
        {
            "id": new_id(),
            "trial_id": trial_id,
            "frame_id": store.list("frames")[0]["id"],
            "raw_response": {"fixture": []},
            "result": None,
            "created_at": now(),
        },
    )
    correction = store.insert(
        "benchmark_corrections",
        {
            "id": new_id(),
            "output_id": output["id"],
            "revision": 1,
            "status": "draft",
            "boxes": [],
            "decisions": {},
            "reviewer": "Human",
            "notes": "",
            "timing": {"elapsed_ms": 45.5},
            "created_at": now(),
        },
    )
    timer = store.insert(
        "benchmark_timers",
        {
            "id": new_id(),
            "output_id": output["id"],
            "reviewer": "Human",
            "state": "paused",
            "segments": [{"elapsed_ms": 45.5}],
            "elapsed_ms": 45.5,
            "created_at": now(),
            "updated_at": now(),
        },
    )
    for table, record in (
        ("benchmarks", benchmark),
        ("benchmark_configs", config),
        ("benchmark_trials", trial),
        ("benchmark_outputs", output),
        ("benchmark_corrections", correction),
        ("benchmark_timers", timer),
        ("jobs", job),
    ):
        assert record_project(store, table, record) == other["id"]
        assert record in project_records(store, table, other["id"])
        assert record not in project_records(store, table, DEFAULT_PROJECT_ID)
    assert timer["segments"] == [{"elapsed_ms": 45.5}]
    assert timer["metadata"] == {} and timer["revision"] == 0
    for table, record in (
        ("benchmark_trials", trial),
        ("benchmark_outputs", output),
        ("benchmark_corrections", correction),
        ("benchmark_timers", timer),
    ):
        with pytest.raises(sqlite3.IntegrityError):
            store.insert(table, {**record, "id": new_id()})
    with pytest.raises(sqlite3.IntegrityError):
        store.update("benchmark_timers", timer["id"], {"elapsed_ms": -1})
    with pytest.raises(sqlite3.IntegrityError):
        store.update("benchmark_trials", trial_id, {"split": "test"})
    with pytest.raises(sqlite3.IntegrityError):
        store.update("benchmark_outputs", output["id"], {"trial_id": "missing"})
