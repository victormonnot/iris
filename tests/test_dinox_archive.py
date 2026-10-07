"""DINO-X archives preserve paid-request receipts without authentication or dispatch."""

import hashlib
import json
import shutil
import sqlite3
import zipfile
from copy import deepcopy

import pytest
import test_workspace_archive as fixtures

from iris import dinox_provider
from iris.annotations import save_annotation
from iris.jobs import JobManager
from iris.projects import create_project, project_records, record_project
from iris.store import (
    DINOX_TABLES,
    SCHEMA_V18,
    SCHEMA_VERSION,
    TABLES,
    TEMPORAL_TABLES,
    Store,
    new_id,
    now,
)
from iris.taxonomies import TAXONOMY
from iris.workspace_archive import (
    ArchiveError,
    _inventory,
    create_archive,
    preview_workspace,
    validate_database,
)
from iris.workspace_restore import inspect_archive, restore_archive

OLD_TABLES = TABLES - DINOX_TABLES - TEMPORAL_TABLES
FRAME_FIELDS = ("id", "session_id", "asset_id", "sha256", "path", "width", "height", "taxonomy_id")


def rows(root, tables=TABLES):
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        return {
            table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY id")]
            for table in sorted(tables)
        }


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@pytest.fixture
def workspace(tmp_path):
    return fixtures.workspace.__wrapped__(tmp_path)


def batch(store, frame, config):
    identifier, job_id = new_id(), new_id()
    store.insert(
        "jobs",
        {
            "id": job_id,
            "kind": "dinox",
            "status": "succeeded",
            "params": {"batch_id": identifier},
            "created_at": now(),
        },
    )
    return store.insert(
        "dinox_batches",
        {
            "id": identifier,
            "session_id": frame["session_id"],
            "name": "Synthetic DINO-X batch",
            "frame_ids": [frame["id"]],
            "config": {"protocol": "iris-dinox-batch-v1", "provider_config": config},
            "job_id": job_id,
            "created_at": now(),
        },
    )


def request(store, *, state="succeeded", with_suggestion=True):
    frame = store.list("frames")[0]
    config = dinox_provider.freeze_config(TAXONOMY)
    owner = batch(store, frame, config)
    snapshot = {
        "frame": {key: frame[key] for key in FRAME_FIELDS},
        "taxonomy_id": TAXONOMY["id"],
    }
    output = {"objects": [{"category": "person", "bbox": [-1, 2, 20, 22], "score": 0.9}]}
    received = state in {"response_received", "succeeded"}
    normalized = dinox_provider.normalize(output, config, frame["width"], frame["height"])
    saved = store.insert(
        "dinox_requests",
        {
            "id": new_id(),
            "frame_id": frame["id"],
            "job_id": owner["job_id"],
            "cache_key": digest({"snapshot": snapshot, "config": config}),
            "config": config,
            "snapshot": snapshot,
            "state": state,
            "task_id": "synthetic-task" if state not in {"not_started", "dispatching"} else None,
            "raw_response": output if received else None,
            "result": normalized if state == "succeeded" else None,
            "metadata": {"fixture": True},
            "created_at": now(),
            "updated_at": now(),
        },
    )
    store.update(
        "dinox_batches",
        owner["id"],
        {
            "metadata": {
                "frames": [{"frame_id": frame["id"], "request_id": saved["id"], "state": state}]
            }
        },
    )
    if with_suggestion and state == "succeeded":
        proposal = normalized["proposals"][0]
        store.insert(
            "annotation_suggestions",
            {
                "id": hashlib.sha256(f"dinox:{saved['id']}:{proposal['id']}".encode()).hexdigest(),
                "frame_id": frame["id"],
                "job_id": owner["job_id"],
                "kind": "detector",
                "label": proposal["label"],
                "box": proposal["box"],
                "metadata": {
                    "provider": "dinox",
                    "dinox_request_id": saved["id"],
                    "batch_id": owner["id"],
                    "target_taxonomy": TAXONOMY["id"],
                    "frame_sha256": frame["sha256"],
                    "source": proposal["source"],
                    "geometry": proposal["geometry"],
                    "score": proposal["score"],
                    "threshold": config["settings"]["bbox_threshold"],
                },
                "created_at": now(),
            },
        )
    return saved, store.get("dinox_batches", owner["id"])


def forbid_provider(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Archive operations must not read credentials or contact DINO-X")

    for name in ("_credential", "provider_status", "submit", "poll", "_request"):
        monkeypatch.setattr(dinox_provider, name, forbidden)


def test_archive_round_trip_preserves_requests_suggestions_and_excludes_credentials(
    workspace, tmp_path, monkeypatch
):
    request(workspace)
    request(workspace, state="outcome_unknown")
    original = rows(workspace.root)
    excluded = workspace.root / ".config" / "iris" / "cloud-credentials.json"
    excluded.parent.mkdir(parents=True)
    excluded.write_text('{"synthetic_fixture":"not an actual credential"}')
    forbid_provider(monkeypatch)
    preview = preview_workspace(workspace.root)
    assert preview["can_create"]
    assert preview["dinox_request_states"] == {"outcome_unknown": 1, "succeeded": 1}
    saved = create_archive(workspace.root, tmp_path / "dinox.zip")
    inspected = inspect_archive(saved["path"])
    restored = tmp_path / "restored"
    restore_archive(saved["path"], restored, expected_archive_sha256=inspected["archive_sha256"])
    assert rows(restored) == original
    with zipfile.ZipFile(saved["path"]) as archive:
        assert excluded.relative_to(workspace.root).as_posix() not in archive.namelist()
        for item in saved["manifest"]["files"]:
            assert (restored / item["path"]).read_bytes() == archive.read(item["path"])
    assert saved["manifest"]["counts"]["dinox_requests"] == 2


@pytest.mark.parametrize(
    "state",
    [
        "not_started",
        "dispatching",
        "submitted",
        "response_received",
        "succeeded",
        "failed",
        "outcome_unknown",
    ],
)
def test_archive_preserves_each_known_request_state_without_retry(workspace, monkeypatch, state):
    request(workspace, state=state)
    forbid_provider(monkeypatch)
    inventory, _ = _inventory(workspace.root)
    checked = validate_database(workspace.root, inventory)
    assert checked["dinox_request_states"] == {state: 1}


@pytest.mark.parametrize(
    "corruption",
    [
        "profile",
        "snapshot",
        "cache_key",
        "batch_session",
        "job_kind",
        "batch_request",
        "suggestion_request",
        "normalized_result",
        "suggestion_geometry",
        "previous_request",
        "nonfinite",
    ],
)
def test_archive_rejects_broken_dinox_provenance(workspace, tmp_path, corruption):
    saved, owner = request(workspace)
    if corruption == "profile":
        config = deepcopy(saved["config"])
        config["endpoint"] = "https://invalid.example.test"
        workspace.update("dinox_requests", saved["id"], {"config": config})
    elif corruption == "snapshot":
        snapshot = deepcopy(saved["snapshot"])
        snapshot["frame"]["sha256"] = "0" * 64
        workspace.update("dinox_requests", saved["id"], {"snapshot": snapshot})
    elif corruption == "cache_key":
        workspace.update("dinox_requests", saved["id"], {"cache_key": "0" * 64})
    elif corruption == "batch_session":
        other = next(row for row in workspace.list("sessions") if row["id"] != owner["session_id"])
        workspace.update("dinox_batches", owner["id"], {"session_id": other["id"]})
    elif corruption == "job_kind":
        workspace.update("jobs", owner["job_id"], {"kind": "fixture"})
    elif corruption == "batch_request":
        workspace.update(
            "dinox_batches",
            owner["id"],
            {"metadata": {"frames": [{"frame_id": saved["frame_id"], "request_id": "missing"}]}},
        )
    elif corruption == "suggestion_request":
        suggestion = workspace.list("annotation_suggestions")[0]
        workspace.update(
            "annotation_suggestions",
            suggestion["id"],
            {"metadata": {**suggestion["metadata"], "dinox_request_id": "missing"}},
        )
    elif corruption == "normalized_result":
        result = deepcopy(saved["result"])
        result["proposals"][0]["box"] = [1, 1, 10, 10]
        workspace.update("dinox_requests", saved["id"], {"result": result})
    elif corruption == "suggestion_geometry":
        suggestion = workspace.list("annotation_suggestions")[0]
        workspace.update(
            "annotation_suggestions",
            suggestion["id"],
            {"metadata": {**suggestion["metadata"], "geometry": {}}},
        )
    elif corruption == "previous_request":
        workspace.update(
            "dinox_requests", saved["id"], {"metadata": {"previous_request_id": saved["id"]}}
        )
    else:
        with workspace.connect() as connection:
            connection.execute("UPDATE dinox_requests SET raw_response=?", ('{"value":NaN}',))
    with pytest.raises(ArchiveError):
        create_archive(workspace.root, tmp_path / "invalid.zip")


def test_cache_receipt_can_be_shared_by_batches_with_distinct_jobs(workspace):
    saved, owner = request(workspace)
    frame = workspace.get("frames", saved["frame_id"])
    reused = batch(workspace, frame, saved["config"])
    workspace.update("dinox_batches", reused["id"], {"metadata": owner["metadata"]})
    inventory, _ = _inventory(workspace.root)
    assert validate_database(workspace.root, inventory)["counts"]["dinox_batches"] == 2
    duplicate = {**saved, "id": new_id(), "job_id": reused["job_id"]}
    workspace.insert("dinox_requests", duplicate)
    assert len(workspace.list("dinox_requests", cache_key=saved["cache_key"])) == 2


def test_actual_batch_workflow_and_reused_proposals_are_archivable(workspace, monkeypatch):
    from iris.dinox_batches import create_batch, preview_batch, run_batch

    frame = workspace.list("frames")[0]
    jobs = JobManager(workspace)
    monkeypatch.setattr(dinox_provider, "provider_status", lambda: {"status": "ready"})
    calls = []

    def submit(config, png, *, idempotency_key):
        calls.append(idempotency_key)
        return {"task_id": idempotency_key, "raw_response": {"code": 0}}

    monkeypatch.setattr(dinox_provider, "submit", submit)
    monkeypatch.setattr(
        dinox_provider,
        "poll",
        lambda _: {
            "status": "succeeded",
            "raw_response": {"code": 0},
            "result": {"objects": [{"category": "person", "bbox": [1, 2, 20, 22], "score": 0.8}]},
        },
    )
    for index in range(2):
        preview = preview_batch(workspace, frame["session_id"], frame_ids=[frame["id"]])
        created = create_batch(
            workspace,
            jobs,
            frame["session_id"],
            name="Synthetic workflow",
            expected_fingerprint=preview["fingerprint"],
            frame_ids=[frame["id"]],
            approve_external=True,
            max_cost_cny=0.15,
        )
        workspace.update("jobs", created["job_id"], {"status": "running"})
        run_batch(workspace, created["id"], lambda *_: None, lambda: False)
        workspace.update("jobs", created["job_id"], {"status": "succeeded"})
        if index == 0:
            save_annotation(workspace, frame["id"], expected_revision=1, boxes=[], decisions={})
    assert len(calls) == len(workspace.list("dinox_requests")) == 1
    assert len(workspace.list("annotation_suggestions")) == 1
    forbid_provider(monkeypatch)
    inventory, _ = _inventory(workspace.root)
    assert validate_database(workspace.root, inventory)["dinox_request_states"] == {"succeeded": 1}


def test_project_ownership_follows_source_frame_and_batch_job(workspace):
    saved, owner = request(workspace)
    project = create_project(workspace, name="DINO-X owner")
    workspace.update("sessions", owner["session_id"], {"project_id": project["id"]})
    assert record_project(workspace, "dinox_requests", saved) == project["id"]
    assert record_project(workspace, "dinox_batches", owner) == project["id"]
    job = workspace.get("jobs", owner["job_id"])
    assert record_project(workspace, "jobs", job) == project["id"]
    job = workspace.update("jobs", job["id"], {"params": {}})
    assert record_project(workspace, "jobs", job) == project["id"]
    assert project_records(workspace, "dinox_requests", "default") == []


@pytest.fixture
def schema18(workspace, tmp_path):
    root = tmp_path / "schema18"
    root.mkdir()
    for path in workspace.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMA_V18)
        for table, records in rows(workspace.root, OLD_TABLES).items():
            for record in records:
                connection.execute(
                    f"INSERT INTO {table} ({','.join(record)}) "
                    f"VALUES ({','.join('?' for _ in record)})",
                    list(record.values()),
                )
        connection.execute("PRAGMA user_version=18")
    return root


def test_schema18_migration_preserves_rows_and_is_idempotent(schema18):
    original = rows(schema18, OLD_TABLES)
    for _ in range(2):
        store = Store(schema18)
        assert rows(schema18, OLD_TABLES) == original
        assert store.list("dinox_requests") == store.list("dinox_batches") == []
        with store.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema19_migration_rolls_back_with_broken_historical_reference(schema18):
    with sqlite3.connect(schema18 / "iris.sqlite3") as connection:
        connection.execute("UPDATE frames SET session_id='missing'")
    original = rows(schema18, OLD_TABLES)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(schema18)
    assert rows(schema18, OLD_TABLES) == original
    with sqlite3.connect(schema18 / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 18
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name IN ('dinox_batches','dinox_requests')"
            ).fetchall()
            == []
        )


def test_schema18_archive_restores_exact_database_before_migration(schema18, tmp_path, monkeypatch):
    original = rows(schema18, OLD_TABLES)
    saved = create_archive(schema18, tmp_path / "old.zip")
    assert saved["manifest"]["schema_version"] == 18
    assert set(saved["manifest"]["counts"]) == OLD_TABLES

    def forbidden(*_args, **_kwargs):
        pytest.fail("Historical archive operations must not initialize or migrate Store")

    restored = tmp_path / "restored"
    with monkeypatch.context() as context:
        context.setattr(Store, "__init__", forbidden)
        forbid_provider(context)
        inspected = inspect_archive(saved["path"])
        restore_archive(
            saved["path"], restored, expected_archive_sha256=inspected["archive_sha256"]
        )
    with zipfile.ZipFile(saved["path"]) as archive:
        assert (restored / "iris.sqlite3").read_bytes() == archive.read("iris.sqlite3")
    Store(restored)
    assert rows(restored, OLD_TABLES) == original
