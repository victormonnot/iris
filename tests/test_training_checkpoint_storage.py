"""Opaque training-state persistence and archives, without opening a model."""

import builtins
import hashlib
import shutil
import sqlite3
import zipfile

import pytest
import test_workspace_archive as fixtures

from iris.projects import create_project, project_records, record_project
from iris.store import (
    DINOX_TABLES,
    SCHEMA_V17,
    SCHEMA_VERSION,
    TABLES,
    TRAINING_CHECKPOINT_TABLES,
    Store,
    new_id,
    now,
)
from iris.workspace_archive import (
    MAX_TRAINING_CHECKPOINT_BYTES,
    ArchiveError,
    ArchiveLimitError,
    _inventory,
    allowed_artifact_path,
    create_archive,
    validate_database,
)
from iris.workspace_restore import inspect_archive, restore_archive

OLD_TABLES = TABLES - TRAINING_CHECKPOINT_TABLES - DINOX_TABLES


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
def schema17(workspace, tmp_path):
    root = tmp_path / "schema17"
    root.mkdir()
    for path in workspace.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with sqlite3.connect(root / "iris.sqlite3") as connection:
        connection.executescript(SCHEMA_V17)
        for table, records in rows(workspace.root).items():
            for record in records:
                connection.execute(
                    f"INSERT INTO {table} ({','.join(record)}) "
                    f"VALUES ({','.join('?' for _ in record)})",
                    list(record.values()),
                )
        connection.execute("UPDATE training_runs SET config=?", ('{ "historical" : true }',))
        connection.execute("PRAGMA user_version=17")
    return root


def checkpoint_row(store):
    training = store.list("training_runs")[0]
    store.update(
        "jobs", training["job_id"], {"kind": "train", "params": {"training_id": training["id"]}}
    )
    identifier = new_id()
    artifact = fixtures._put(
        store,
        f"training_checkpoints/{training['id']}/{identifier}.pth",
        b"Opaque synthetic state fixture, not executable model parameters",
    )
    return store.insert(
        "training_checkpoints",
        {
            "id": identifier,
            "training_id": training["id"],
            "step": 10,
            "path": artifact["path"],
            "state_sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"],
            "metadata": {"fixture": True, "sampler": [4, 2, 0]},
            "created_at": now(),
        },
    )


@pytest.fixture
def validation_hooks(monkeypatch):
    # This module tests storage boundaries; the engine's frozen-contract validation
    # is exercised separately against full training fixtures.
    from iris import training_recovery

    calls = []

    def checkpoint(row, *, connection, root):
        calls.append(("checkpoint", row, root))
        assert connection.execute(
            "SELECT id FROM training_runs WHERE id=?", (row["training_id"],)
        ).fetchone()

    def recoveries(connection, root):
        calls.append(("recoveries", None, root))

    monkeypatch.setattr(training_recovery, "validate_training_checkpoint", checkpoint)
    monkeypatch.setattr(training_recovery, "validate_training_recoveries", recoveries)
    return calls


def test_schema18_adds_only_training_states_without_changing_history_or_files(schema17):
    original, files = rows(schema17), artifacts(schema17)
    for _ in range(2):
        store = Store(schema17)
        assert rows(schema17) == original
        assert artifacts(schema17) == files
        assert store.list("training_checkpoints") == []
        with store.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 19
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema18_migration_rolls_back_on_existing_broken_reference(schema17):
    with sqlite3.connect(schema17 / "iris.sqlite3") as connection:
        connection.execute("UPDATE trained_models SET training_id='missing'")
    original, files = rows(schema17), artifacts(schema17)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(schema17)
    assert rows(schema17) == original
    assert artifacts(schema17) == files
    with sqlite3.connect(schema17 / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 17
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='training_checkpoints'"
            ).fetchone()
            is None
        )


def test_schema17_archive_restores_exact_bytes_without_migration(schema17, tmp_path, monkeypatch):
    original = rows(schema17)
    saved = create_archive(schema17, tmp_path / "old.zip")
    assert saved["manifest"]["schema_version"] == 17
    assert set(saved["manifest"]["counts"]) == OLD_TABLES
    restored = tmp_path / "restored"

    def forbidden(*_args, **_kwargs):
        pytest.fail("Archive inspection and restoration must not initialize Store")

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
    assert rows(restored) == original


def test_state_json_uniqueness_and_project_ownership(workspace):
    checkpoint = checkpoint_row(workspace)
    assert workspace.get("training_checkpoints", checkpoint["id"]) == checkpoint
    training = workspace.get("training_runs", checkpoint["training_id"])
    project = create_project(workspace, name="Checkpoint owner")
    workspace.update("dataset_versions", training["dataset_id"], {"project_id": project["id"]})
    assert record_project(workspace, "training_checkpoints", checkpoint) == project["id"]
    assert project_records(workspace, "training_checkpoints", "default") == []
    assert project_records(workspace, "training_checkpoints", project["id"]) == [checkpoint]
    with pytest.raises(sqlite3.IntegrityError):
        workspace.insert("training_checkpoints", {**checkpoint, "id": new_id(), "path": "new.pth"})
    with pytest.raises(sqlite3.IntegrityError):
        workspace.insert("training_checkpoints", {**checkpoint, "id": new_id(), "step": 20})


@pytest.mark.parametrize(
    "path",
    [
        "training_checkpoints/run/state.pth",
        "training_checkpoints/other/checkpoint.pth",
    ],
)
def test_state_paths_are_retained(path):
    assert allowed_artifact_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "training_checkpoints/state.pth",
        "training_checkpoints/run/extra/state.pth",
        "training_checkpoints/run/state.pkl",
        "training_checkpoints/run/state.pth.part",
        "training_checkpoints/run/../state.pth",
        "training_checkpoints/run/.state.pth",
        "training_checkpoints/run/state.pth.json",
    ],
)
def test_only_bounded_state_file_paths_are_retained(path):
    assert not allowed_artifact_path(path)


def test_archive_checks_metadata_without_retaining_or_loading_state_on_inspection(
    workspace, tmp_path, monkeypatch, validation_hooks
):
    checkpoint = checkpoint_row(workspace)
    state = workspace.root / checkpoint["path"]
    original = state.read_bytes()
    real_import = builtins.__import__

    def import_without_models(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision"}:
            pytest.fail("Opaque training archives must not import a model runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_models)
    saved = create_archive(workspace.root, tmp_path / "states.zip")
    assert saved["manifest"]["counts"]["training_checkpoints"] == 1
    validation_hooks.clear()
    inspected = inspect_archive(saved["path"])
    inspected_states = [row for kind, row, _ in validation_hooks if kind == "checkpoint"]
    assert inspected_states == [checkpoint]
    # Inspection validates opaque bytes through the archive inventory; only the
    # database and reference documents need to be retained for its SQL checks.
    assert all(not (root / checkpoint["path"]).exists() for _, _, root in validation_hooks)
    restored = tmp_path / "restored"
    restore_archive(saved["path"], restored, expected_archive_sha256=inspected["archive_sha256"])
    assert (restored / checkpoint["path"]).read_bytes() == original
    assert Store(restored).get("training_checkpoints", checkpoint["id"]) == checkpoint


@pytest.mark.parametrize("damage", ["missing", "size", "hash", "path", "job_kind", "job_source"])
def test_state_inconsistency_blocks_backup(workspace, tmp_path, validation_hooks, damage):
    checkpoint = checkpoint_row(workspace)
    if damage == "missing":
        (workspace.root / checkpoint["path"]).unlink()
    elif damage == "size":
        workspace.update(
            "training_checkpoints", checkpoint["id"], {"size_bytes": checkpoint["size_bytes"] + 1}
        )
    elif damage == "hash":
        workspace.update("training_checkpoints", checkpoint["id"], {"state_sha256": "0" * 64})
    elif damage == "path":
        workspace.update(
            "training_checkpoints",
            checkpoint["id"],
            {"path": "training_checkpoints/foreign/state.pth"},
        )
    else:
        training = workspace.get("training_runs", checkpoint["training_id"])
        workspace.update(
            "jobs",
            training["job_id"],
            {"kind": "fixture"} if damage == "job_kind" else {"params": {"training_id": "other"}},
        )
    with pytest.raises(ArchiveError):
        create_archive(workspace.root, tmp_path / "invalid.zip")


@pytest.mark.parametrize("size", [0, -1, MAX_TRAINING_CHECKPOINT_BYTES + 1])
def test_state_size_is_bounded_before_archive_reads(workspace, validation_hooks, size):
    checkpoint = checkpoint_row(workspace)
    workspace.update("training_checkpoints", checkpoint["id"], {"size_bytes": size})
    inventory, _ = _inventory(workspace.root)
    with pytest.raises(ArchiveLimitError, match="size"):
        validate_database(workspace.root, inventory)


def test_archive_calls_global_lineage_validation_even_without_state_rows(
    workspace, validation_hooks
):
    inventory, _ = _inventory(workspace.root)
    validate_database(workspace.root, inventory)
    assert [kind for kind, _, _ in validation_hooks] == ["recoveries"]


def test_archive_checks_source_bindings_after_file_identity(
    workspace, monkeypatch, validation_hooks
):
    from iris import training_recovery

    checkpoint_row(workspace)

    def invalid(*_args, **_kwargs):
        raise ValueError("source contract changed")

    monkeypatch.setattr(training_recovery, "validate_training_checkpoint", invalid)
    inventory, _ = _inventory(workspace.root)
    with pytest.raises(ArchiveError, match="source identity"):
        validate_database(workspace.root, inventory)


def test_restore_rejects_state_bytes_that_conflict_with_recorded_database_hash(
    workspace, tmp_path, validation_hooks
):
    checkpoint = checkpoint_row(workspace)
    saved = create_archive(workspace.root, tmp_path / "states.zip")
    # Rewriting both the member and outer manifest still cannot change the SHA
    # frozen in SQLite. Exercise that boundary without tensor deserialization.
    import json

    altered = tmp_path / "changed.zip"
    with zipfile.ZipFile(saved["path"]) as original, zipfile.ZipFile(altered, "w") as target:
        manifest = json.loads(original.read("manifest.json"))
        for member in original.infolist():
            if member.filename == "manifest.json":
                continue
            data = original.read(member.filename)
            if member.filename == checkpoint["path"]:
                data = b"X" * len(data)
                next(item for item in manifest["files"] if item["path"] == member.filename)[
                    "sha256"
                ] = hashlib.sha256(data).hexdigest()
            target.writestr(member.filename, data)
        target.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ArchiveError, match="artifact hash"):
        inspect_archive(altered)


@pytest.fixture
def bound_checkpoint(workspace):
    """A complete frozen checkpoint contract with opaque synthetic state bytes."""
    from iris.annotations import save_annotation
    from iris.datasets import create_dataset, load_manifest
    from iris.model_taxonomy import dataset_contract
    from iris.training import SCOPE_VERSION, training_scope
    from iris.training_recovery import PROTOCOL, binding, sampling

    frame = next(
        frame
        for frame in workspace.list("frames")
        if workspace.get("sessions", frame["session_id"])["scene_group"] == "train"
    )
    save_annotation(
        workspace,
        frame["id"],
        expected_revision=1,
        boxes=[{"id": "person", "label": "person", "box": [2, 2, 15, 20]}],
        decisions={},
        status="validated",
        reviewer="Synthetic storage fixture",
    )
    dataset = create_dataset(
        workspace,
        name="Frozen continuation fixture",
        frame_ids=[frame["id"] for frame in workspace.list("frames")],
        splits={"train": "train", "val": "val"},
    )
    manifest = load_manifest(workspace, dataset["id"])
    config = {
        "checkpoint_protocol": PROTOCOL,
        "steps": 20,
        "checkpoint_interval": 5,
        "history_interval": 10,
        "seed": 42,
        "device": "cpu",
        "batch_size": 1,
        "scope": "prediction_head_only",
        "scope_version": SCOPE_VERSION,
        "trainable_modules": training_scope("prediction_head_only")["trainable_modules"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "parent_weight_sha256": "b" * 64,
        **dataset_contract(manifest),
    }
    frames = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    _, _, visits, sampler = sampling(config["seed"], frames, 10)
    history = [
        {
            "step": step,
            "frame_id": frame_id,
            "loss": 1.0 / step,
            "losses": {"fixture": 1.0 / step},
            "elapsed_seconds": step * 0.25,
        }
        for step, frame_id in enumerate(visits, 1)
    ]
    training = workspace.list("training_runs")[0]
    workspace.update(
        "training_runs",
        training["id"],
        {
            "dataset_id": dataset["id"],
            "config": config,
            "history": history,
        },
    )
    checkpoint = checkpoint_row(workspace)
    training = workspace.get("training_runs", training["id"])
    workspace.update(
        "training_checkpoints",
        checkpoint["id"],
        {
            "metadata": {
                "binding": binding(training, history, {"fixture": "synthetic"}),
                "sampler": sampler,
            },
        },
    )
    return workspace.get("training_checkpoints", checkpoint["id"])


def test_actual_checkpoint_validator_uses_retained_manifest_without_loading_state(
    workspace, bound_checkpoint, tmp_path, monkeypatch
):
    real_import = builtins.__import__

    def without_model_imports(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision"}:
            pytest.fail("Frozen checkpoint identities do not require a model runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_model_imports)
    saved = create_archive(workspace.root, tmp_path / "bound.zip")
    inspected = inspect_archive(saved["path"])
    restored = tmp_path / "restored-bound"
    restore_archive(saved["path"], restored, expected_archive_sha256=inspected["archive_sha256"])
    assert Store(restored).get("training_checkpoints", bound_checkpoint["id"]) == bound_checkpoint


@pytest.mark.parametrize("damage", ["sampler", "history", "dataset", "parent", "step"])
def test_actual_checkpoint_validator_rejects_changed_frozen_sources(
    workspace, bound_checkpoint, damage
):
    training = workspace.get("training_runs", bound_checkpoint["training_id"])
    if damage == "sampler":
        bound_checkpoint["metadata"]["sampler"]["remaining_order"] = [0]
        workspace.update(
            "training_checkpoints",
            bound_checkpoint["id"],
            {
                "metadata": bound_checkpoint["metadata"],
            },
        )
    elif damage == "history":
        training["history"][0]["loss"] = 99.0
        workspace.update("training_runs", training["id"], {"history": training["history"]})
    elif damage == "step":
        workspace.update("training_checkpoints", bound_checkpoint["id"], {"step": 21})
    else:
        key = "dataset_manifest_sha256" if damage == "dataset" else "parent_weight_sha256"
        training["config"][key] = "c" * 64
        workspace.update("training_runs", training["id"], {"config": training["config"]})
    inventory, _ = _inventory(workspace.root)
    with pytest.raises(ArchiveError, match="source identity"):
        validate_database(workspace.root, inventory)
