"""Detection caches extend temporal storage without changing existing evidence."""

import builtins
import json
import shutil
import sqlite3
import zipfile
from copy import deepcopy

import pytest
from test_temporal_archive import artifacts, temporal_workspace
from test_temporal_detector import frozen_config, runtime_metadata

from iris.annotations import save_annotation
from iris.jobs import JobManager
from iris.store import (
    SCHEMA_V20,
    SCHEMA_VERSION,
    TABLES,
    TEMPORAL_DETECTION_TABLES,
    Store,
    new_id,
    now,
)
from iris.temporal import _digest
from iris.temporal_detection_contracts import TIMING_FIELDS, payload_hash, validate_frame_payload
from iris.workspace_archive import ArchiveError, _inventory, create_archive, validate_database
from iris.workspace_restore import inspect_archive, restore_archive

OLD_TABLES = TABLES - TEMPORAL_DETECTION_TABLES


def rows(root, tables=OLD_TABLES):
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        return {
            table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY id")]
            for table in sorted(tables)
        }


@pytest.fixture
def schema20(tmp_path):
    source = temporal_workspace.__wrapped__(tmp_path)
    save_annotation(
        source,
        source.list("frames")[0]["id"],
        expected_revision=0,
        boxes=[],
        decisions={},
        status="validated",
        reviewer="Synthetic fixture",
    )
    root = tmp_path / "schema20"
    root.mkdir()
    for path in source.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMA_V20)
        for table, records in rows(source.root).items():
            for record in records:
                connection.execute(
                    f"INSERT INTO {table} ({','.join(record)}) "
                    f"VALUES ({','.join('?' for _ in record)})",
                    list(record.values()),
                )
        connection.execute("UPDATE annotation_revisions SET boxes='[ ]'")
        connection.execute("PRAGMA user_version=20")
    return root


def test_detection_migration_preserves_image_annotations_and_temporal_evidence(schema20):
    original, files = rows(schema20), artifacts(schema20)
    assert len(original["temporal_sequences"]) == 1
    assert len(original["temporal_references"]) == 1
    assert len(original["temporal_datasets"]) == 1
    assert len(original["annotation_revisions"]) == 1
    for _ in range(2):
        store = Store(schema20)
        assert rows(schema20) == original
        assert artifacts(schema20) == files
        assert all(store.list(table) == [] for table in TEMPORAL_DETECTION_TABLES)
        with store.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 21
            assert not connection.execute("PRAGMA foreign_key_check").fetchall()


def test_detection_migration_rolls_back_on_broken_temporal_foreign_key(schema20):
    with sqlite3.connect(schema20 / "iris.sqlite3") as connection:
        connection.execute("UPDATE temporal_references SET sequence_id='missing'")
    original, files = rows(schema20), artifacts(schema20)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(schema20)
    assert rows(schema20) == original
    assert artifacts(schema20) == files
    with sqlite3.connect(schema20 / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 20
        assert not connection.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'temporal_detection_%'"
        ).fetchall()


def test_schema20_archive_keeps_exact_temporal_records_without_migration(
    schema20, tmp_path, monkeypatch
):
    original = rows(schema20)
    archive = create_archive(schema20, tmp_path / "schema20.zip")
    assert archive["manifest"]["schema_version"] == 20
    assert set(archive["manifest"]["counts"]) == OLD_TABLES

    def forbidden(*_args, **_kwargs):
        pytest.fail("Historical inspection and restore must not initialize Store")

    monkeypatch.setattr(Store, "__init__", forbidden)
    preview = inspect_archive(archive["path"])
    target = tmp_path / "restored20"
    restored = restore_archive(
        archive["path"], target, expected_archive_sha256=preview["archive_sha256"]
    )
    assert restored["verified"]
    assert rows(target) == original
    assert artifacts(target) == artifacts(schema20)
    with zipfile.ZipFile(archive["path"]) as saved:
        assert (target / "iris.sqlite3").read_bytes() == saved.read("iris.sqlite3")
    with sqlite3.connect(target / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 20


@pytest.mark.parametrize("table", sorted(TEMPORAL_DETECTION_TABLES))
def test_detection_records_are_immutable(table, tmp_path):
    store = Store(tmp_path / "workspace")
    with pytest.raises(ValueError, match="immutable"):
        store.update(table, "existing-record", {"created_at": "replacement"})


def populate_cache(store, monkeypatch, *, completed):
    """Publish schema-valid synthetic output; worker execution has separate tests."""
    from iris import temporal_detections as caches
    from iris.temporal_detector import saved_execution_signature

    detector = frozen_config()
    monkeypatch.setattr(caches, "prepare_detector", lambda *_args, **_kwargs: deepcopy(detector))
    sequence = store.list("temporal_sequences")[0]
    cache = caches.create_detection_cache(
        store,
        JobManager(store),
        sequence["id"],
        name="Synthetic archive cache",
        model_id=detector["model_id"],
    )
    metadata = runtime_metadata(detector)
    signature = saved_execution_signature(detector, metadata)
    execution = {
        "signature": signature,
        "signature_sha256": _digest(signature),
        "metadata": metadata,
        "metadata_sha256": _digest(metadata),
        "load_ms": 1.0,
        "warmup_ms": 0.5,
        "warmup_frame_id": cache["config"]["frame_ids"][0],
    }
    for index, frame in enumerate(sequence["manifest"]["frames"][:completed]):
        detections = (
            []
            if index == 1
            else [
                {
                    "detection_index": 0,
                    "label_id": 1,
                    "label": "person",
                    "score": 0.8,
                    "box": [2.0, 3.0, 20.0, 40.0],
                }
            ]
        )
        payload = validate_frame_payload(
            {
                "schema": "iris-temporal-detection-frame-v1",
                "cache_fingerprint": cache["fingerprint"],
                "frame_id": frame["frame_id"],
                "frame_index": frame["frame_index"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_sha256": frame["sha256"],
                "file_sha256": frame["file_sha256"],
                "input_size": [frame["width"], frame["height"]],
                "detections": detections,
                "native_detection_count": len(detections),
                "execution_signature_sha256": execution["signature_sha256"],
                "timing": {name: 0.1 for name in TIMING_FIELDS},
                "work": {"forward_passes": 1, "tile_count": 0},
            },
            cache["config"],
            sequence["manifest"],
        )
        store.insert(
            "temporal_detection_frames",
            {
                "id": new_id(),
                "cache_id": cache["id"],
                "frame_id": frame["frame_id"],
                "job_id": cache["job_id"],
                "payload": payload,
                "payload_sha256": payload_hash(payload),
                "created_at": now(),
            },
        )
    complete = completed == len(sequence["manifest"]["frames"])
    result = store.get("jobs", cache["job_id"])["result"]
    result.update(
        worker_token=new_id(),
        execution=execution,
        produced_count=completed,
        completed_count=completed,
        cancelled=not complete,
    )
    store.update(
        "jobs",
        cache["job_id"],
        {
            "status": "succeeded" if complete else "cancelled",
            "result": result,
            "finished_at": now(),
        },
    )
    return caches.get_detection_cache(store, cache["id"])


@pytest.mark.parametrize("completed", [1, 3], ids=["partial", "complete"])
def test_cache_archives_round_trip_without_weights_or_runtime(tmp_path, monkeypatch, completed):
    from iris import temporal_detections as caches
    from iris import temporal_detector as adapter

    store = temporal_workspace.__wrapped__(tmp_path)
    cache = populate_cache(store, monkeypatch, completed=completed)
    expected_state = "complete" if completed == 3 else "partial"
    assert cache["coverage"]["state"] == expected_state
    original, files = rows(store.root, TABLES), artifacts(store.root)
    assert not (store.root / "models").exists()

    def forbidden(*_args, **_kwargs):
        pytest.fail("Reading archived detector outputs must not load weights or current runtimes")

    real_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "torchvision"}
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(caches, "prepare_detector", forbidden)
    monkeypatch.setattr(adapter, "prepare_detector", forbidden)
    monkeypatch.setattr(adapter, "_runtime", forbidden)
    monkeypatch.setattr(adapter, "detector_factory", forbidden)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    archive = create_archive(store.root, tmp_path / "cache.zip")
    assert archive["manifest"]["counts"]["temporal_detection_frames"] == completed
    preview = inspect_archive(archive["path"])
    target = tmp_path / "restored-cache"
    restored = restore_archive(
        archive["path"], target, expected_archive_sha256=preview["archive_sha256"]
    )
    assert restored["verified"]
    assert rows(target, TABLES) == original
    assert artifacts(target) == files
    restored_store = Store(target)
    reread = caches.get_detection_cache(restored_store, cache["id"])
    assert reread["coverage"]["state"] == expected_state
    assert reread["config"] == cache["config"]


@pytest.mark.parametrize(
    "corruption",
    ["cache", "frame", "metadata", "producer", "job_params", "success_with_missing_frames"],
)
def test_archive_rejects_inconsistent_detection_evidence(tmp_path, monkeypatch, corruption):
    store = temporal_workspace.__wrapped__(tmp_path)
    cache = populate_cache(store, monkeypatch, completed=1)
    frame = store.list("temporal_detection_frames")[0]
    with store.connect() as connection:
        if corruption == "cache":
            connection.execute("UPDATE temporal_detection_caches SET fingerprint=?", ("0" * 64,))
        elif corruption == "frame":
            payload = frame["payload"]
            payload["detections"][0]["score"] = 1.1
            connection.execute(
                "UPDATE temporal_detection_frames SET payload=?,payload_sha256=?",
                (json.dumps(payload), payload_hash(payload)),
            )
        elif corruption == "metadata":
            result = store.get("jobs", cache["job_id"])["result"]
            result["execution"]["metadata"]["device"] = "cuda:0"
            result["execution"]["metadata_sha256"] = _digest(result["execution"]["metadata"])
            connection.execute(
                "UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), cache["job_id"])
            )
        elif corruption == "producer":
            result = store.get("jobs", cache["job_id"])["result"]
            result["produced_count"] = result["completed_count"] = 0
            connection.execute(
                "UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), cache["job_id"])
            )
        elif corruption == "job_params":
            connection.execute("UPDATE jobs SET params='[]' WHERE id=?", (cache["job_id"],))
        else:
            connection.execute("UPDATE jobs SET status='succeeded' WHERE id=?", (cache["job_id"],))
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="temporal detection records"):
        validate_database(store.root, inventory)
