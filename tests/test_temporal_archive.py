"""Temporal storage remains additive and round-trips without model execution."""

import hashlib
import json
import shutil
import sqlite3
import zipfile

import pytest
import test_workspace_archive as archive_fixtures
from temporal_fixtures import video_sequence
from test_datasets import add_frame

from iris.annotations import save_annotation
from iris.datasets import create_dataset
from iris.store import (
    SCHEMA_V19,
    SCHEMA_VERSION,
    TABLES,
    TEMPORAL_DETECTION_TABLES,
    TEMPORAL_TABLES,
    TRACKING_QUALITY_TABLES,
    Store,
)
from iris.workspace_archive import (
    SCHEMA_TABLES,
    SCHEMAS,
    ArchiveError,
    _inventory,
    create_archive,
    validate_database,
)
from iris.workspace_restore import inspect_archive, restore_archive

OLD_TABLES = TABLES - TEMPORAL_TABLES - TEMPORAL_DETECTION_TABLES - TRACKING_QUALITY_TABLES


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
def schema19(tmp_path):
    source = archive_fixtures.workspace.__wrapped__(tmp_path)
    archive_fixtures._all_references(source)
    root = tmp_path / "schema19"
    root.mkdir()
    for path in source.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMA_V19)
        for table, records in rows(source.root).items():
            for record in records:
                connection.execute(
                    f"INSERT INTO {table} ({','.join(record)}) "
                    f"VALUES ({','.join('?' for _ in record)})",
                    list(record.values()),
                )
        # Migration must preserve original serialization, including reviewed boxes.
        connection.execute("UPDATE annotation_revisions SET boxes='[ ]'")
        connection.execute("PRAGMA user_version=19")
    return root


def test_schema20_migration_preserves_annotations_history_and_artifacts(schema19):
    original, files = rows(schema19), artifacts(schema19)
    for _ in range(2):
        store = Store(schema19)
        assert rows(schema19) == original
        assert artifacts(schema19) == files
        assert all(store.list(table) == [] for table in TEMPORAL_TABLES)
        with store.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema20_migration_rolls_back_on_broken_historical_reference(schema19):
    with sqlite3.connect(schema19 / "iris.sqlite3") as connection:
        connection.execute("UPDATE frames SET asset_id='missing'")
    original, files = rows(schema19), artifacts(schema19)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(schema19)
    assert rows(schema19) == original
    assert artifacts(schema19) == files
    with sqlite3.connect(schema19 / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 19
        assert not connection.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'temporal_%'"
        ).fetchall()


def test_schema19_archive_restores_frozen_database_without_migration(
    schema19, tmp_path, monkeypatch
):
    original = rows(schema19)
    archive = create_archive(schema19, tmp_path / "historical.zip")
    assert archive["manifest"]["schema_version"] == 19
    assert set(archive["manifest"]["counts"]) == OLD_TABLES

    def forbidden(*_args, **_kwargs):
        pytest.fail("Inspection and restore must not initialize or migrate Store")

    monkeypatch.setattr(Store, "__init__", forbidden)
    preview = inspect_archive(archive["path"])
    target = tmp_path / "restored"
    result = restore_archive(
        archive["path"], target, expected_archive_sha256=preview["archive_sha256"]
    )
    assert result["verified"]
    assert rows(target) == original
    with zipfile.ZipFile(archive["path"]) as saved:
        assert (target / "iris.sqlite3").read_bytes() == saved.read("iris.sqlite3")
    with sqlite3.connect(target / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 19


@pytest.mark.parametrize("version", range(12, 20))
def test_all_historical_archive_layouts_remain_supported(version, tmp_path):
    root = tmp_path / "historical"
    root.mkdir()
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMAS[version])
        connection.execute(f"PRAGMA user_version={version}")
    archive = create_archive(root, tmp_path / "historical.zip")
    assert set(archive["manifest"]["counts"]) == SCHEMA_TABLES[version]
    assert not set(archive["manifest"]["counts"]) & TEMPORAL_TABLES
    preview = inspect_archive(archive["path"])
    result = restore_archive(
        archive["path"], tmp_path / "restored", expected_archive_sha256=preview["archive_sha256"]
    )
    assert result["verified"]


@pytest.mark.parametrize("table", sorted(TEMPORAL_TABLES))
def test_temporal_store_updates_require_a_new_published_version(table, tmp_path):
    store = Store(tmp_path / "workspace")
    with pytest.raises(ValueError, match="immutable"):
        store.update(table, "existing-record", {"created_at": "replacement"})


@pytest.fixture
def temporal_workspace(tmp_path):
    from iris.temporal import create_sequence, create_temporal_dataset, save_reference

    store = Store(tmp_path / "temporal-workspace")
    asset, frames = video_sequence(store, tmp_path)
    sequence = create_sequence(
        store,
        name="Synthetic temporal sequence",
        asset_id=asset["id"],
        frame_ids=[frame["id"] for frame in frames],
    )
    payload = {
        "schema": "iris-temporal-reference-v1",
        "sequence_id": sequence["id"],
        "sequence_sha256": sequence["manifest_sha256"],
        "taxonomy_id": sequence["manifest"]["taxonomy"]["id"],
        "identities": [{"id": "subject-a", "label": "person"}],
        "frames": [
            {
                "frame_index": frame["frame_index"],
                "coverage": "complete",
                "review": {"status": "human_reviewed", "reviewer": "Synthetic fixture"},
                "objects": [
                    {
                        "identity_id": "subject-a",
                        "label": "person",
                        "box": [2, 3, 20, 40],
                        "visibility": "visible",
                        "certainty": "certain",
                    }
                ],
            }
            for frame in frames
        ],
        "notes": "Synthetic test labels, not a real human review",
    }
    reference = save_reference(store, sequence["id"], payload=payload)
    create_temporal_dataset(
        store,
        name="Synthetic frozen temporal dataset",
        entries=[
            {"sequence_id": sequence["id"], "reference_id": reference["id"], "split": "train"}
        ],
    )
    return store


def test_temporal_archive_round_trip_preserves_exact_rows_and_files(
    temporal_workspace, tmp_path, monkeypatch
):
    store = temporal_workspace
    original = rows(store.root, TABLES)
    archive = create_archive(store.root, tmp_path / "temporal.zip")
    assert archive["manifest"]["schema_version"] == SCHEMA_VERSION
    assert all(archive["manifest"]["counts"][table] == 1 for table in TEMPORAL_TABLES)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Archive inspection and restore must not initialize or migrate Store")

    monkeypatch.setattr(Store, "__init__", forbidden)
    preview = inspect_archive(archive["path"])
    target = tmp_path / "restored-temporal"
    restored = restore_archive(
        archive["path"], target, expected_archive_sha256=preview["archive_sha256"]
    )
    assert restored["verified"]
    assert rows(target, TABLES) == original
    assert artifacts(target) == artifacts(store.root)


@pytest.mark.parametrize(
    ("table", "field"),
    [
        ("temporal_sequences", "manifest_sha256"),
        ("temporal_references", "payload_sha256"),
        ("temporal_datasets", "manifest_sha256"),
    ],
)
def test_archive_rejects_changed_temporal_identity(temporal_workspace, table, field):
    store = temporal_workspace
    with store.connect() as connection:
        connection.execute(f"UPDATE {table} SET {field}=?", ("0" * 64,))
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="temporal"):
        validate_database(store.root, inventory)


def test_archive_rejects_modified_temporal_frame_file(temporal_workspace, tmp_path):
    store = temporal_workspace
    frame = store.list("frames")[0]
    path = store.artifact_path(frame["path"])
    original = path.read_bytes()
    # Same length, so only the frozen file checksum catches this change.
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ArchiveError, match="checksum"):
        create_archive(store.root, tmp_path / "corrupt.zip")
    assert not (tmp_path / "corrupt.zip").exists()


def test_archive_inspection_verifies_temporal_files_through_inventory(temporal_workspace, tmp_path):
    store = temporal_workspace
    archive = create_archive(store.root, tmp_path / "temporal.zip")
    inspection = tmp_path / "inspection"
    inspection.mkdir()
    with zipfile.ZipFile(archive["path"]) as saved:
        saved.extract("iris.sqlite3", inspection)
    inventory = {item["path"]: item for item in archive["manifest"]["files"]}
    checked = validate_database(inspection, inventory)
    for sequence in store.list("temporal_sequences"):
        for frame in sequence["manifest"]["frames"]:
            path = store.get("frames", frame["frame_id"])["path"]
            assert checked["expected_hashes"][path] == frame["file_sha256"]


def test_archive_rejects_rehashed_cross_task_split_conflict(temporal_workspace, tmp_path):
    store = temporal_workspace
    train = store.list("frames")[0]
    store.update("frames", train["id"], {"selected": True})
    save_annotation(
        store,
        train["id"],
        expected_revision=0,
        boxes=[],
        decisions={},
        status="validated",
        reviewer="Synthetic fixture",
    )
    validation = add_frame(store, tmp_path, group="independent-val", color=(70, 80, 90))
    create_dataset(
        store,
        name="Image dataset using the same training take",
        frame_ids=[train["id"], validation["id"]],
        splits={"take-a": "train", "independent-val": "val"},
    )
    inventory, _ = _inventory(store.root)
    assert validate_database(store.root, inventory)["counts"]["dataset_versions"] == 1
    dataset = store.list("temporal_datasets")[0]
    manifest = dataset["manifest"]
    manifest["entries"][0]["split"] = "val"
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with store.connect() as connection:
        connection.execute(
            "UPDATE temporal_datasets SET manifest=?,manifest_sha256=? WHERE id=?",
            (raw, hashlib.sha256(raw.encode()).hexdigest(), dataset["id"]),
        )
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="temporal dataset splits"):
        validate_database(store.root, inventory)
