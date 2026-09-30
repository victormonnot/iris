"""Streaming backups of tiny synthetic workspaces, without model execution."""

import hashlib
import json
import os
import shutil
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from iris import workspace_archive as archives
from iris.annotations import save_annotation
from iris.datasets import create_dataset
from iris.media import import_asset
from iris.store import Store, new_id, now
from iris.workspace_archive import ArchiveCancelled, ArchiveError, ArchiveLimitError


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _put(store, relative, data):
    path = store.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "size_bytes": len(data), "sha256": _sha(data)}


def _job(store, status="succeeded"):
    return store.insert(
        "jobs",
        {"id": new_id(), "kind": "fixture", "status": status, "params": {}, "created_at": now()},
    )


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    frames = []
    for group, color in (("train", "red"), ("val", "green")):
        session = store.insert(
            "sessions",
            {"id": new_id(), "name": group, "scene_group": group, "created_at": now()},
        )
        source = tmp_path / f"{group}.png"
        Image.new("RGB", (32, 24), color).save(source)
        asset = import_asset(store, session["id"], source, source.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        store.update("frames", frame["id"], {"selected": True})
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=[],
            decisions={},
            status="validated",
            reviewer="Fixture reviewer",
        )
        frames.append(frame)
    create_dataset(
        store,
        name="Fixture release",
        frame_ids=[frame["id"] for frame in frames],
        splits={"train": "train", "val": "val"},
    )
    return store


def _all_references(store):
    """Add every remaining managed reference using bytes, never runtime models."""
    frame = store.list("frames")[0]
    dataset = store.list("dataset_versions")[0]
    job = _job(store)
    training = store.insert(
        "training_runs",
        {
            "id": new_id(),
            "name": "Fixture training",
            "dataset_id": dataset["id"],
            "parent_model_id": "fixture-parent",
            "config": {},
            "job_id": job["id"],
            "created_at": now(),
        },
    )
    checkpoint = _put(store, "models/trained/fixture.pth", b"synthetic checkpoint")
    store.insert(
        "trained_models",
        {
            "id": new_id(),
            "name": "Fixture checkpoint",
            "training_id": training["id"],
            "parent_model_id": "fixture-parent",
            "architecture": "fixture",
            "path": checkpoint["path"],
            "weight_sha256": checkpoint["sha256"],
            "metadata": {},
            "created_at": now(),
        },
    )
    evaluation = store.insert(
        "evaluations",
        {
            "id": new_id(),
            "name": "Fixture evaluation",
            "dataset_id": dataset["id"],
            "split": "val",
            "model_ids": [],
            "config": {},
            "job_id": _job(store)["id"],
            "created_at": now(),
        },
    )
    for table, prefix, extra in (
        (
            "assistance_previews",
            "assistance/previews",
            {"frame_id": frame["id"], "config": {}, "candidates": [], "expires_at": now()},
        ),
        (
            "video_reviews",
            "video_reviews",
            {"asset_id": frame["asset_id"], "config": {}, "expires_at": now()},
        ),
        (
            "experiment_reports",
            "reports",
            {
                "evaluation_id": evaluation["id"],
                "title": "Fixture report",
                "objective": "",
                "conclusion": "",
                "revision": 1,
                "snapshot": {},
                "snapshot_sha256": _sha(b"{}"),
                "updated_at": now(),
            },
        ),
    ):
        identifier = new_id()
        image = _put(store, f"{prefix}/{identifier}/0.jpg", b"fixture image bytes")
        store.insert(table, {"id": identifier, "images": [image], "created_at": now(), **extra})
    identifier = new_id()
    source = _put(store, f"imports/{identifier}/0.source", b"import source")
    normalized = _put(store, f"imports/{identifier}/0.png", b"normalized import")
    original = _put(store, f"imports/{identifier}/source.zip", b"import zip")
    store.insert(
        "dataset_imports",
        {
            "id": identifier,
            "path": original["path"],
            "sha256": original["sha256"],
            "summary": {
                "images": [
                    {
                        "source_path": source["path"],
                        "source_sha256": source["sha256"],
                        "size_bytes": source["size_bytes"],
                        "path": normalized["path"],
                        "png_sha256": normalized["sha256"],
                        "sha256": "f" * 64,
                    }
                ]
            },
            "created_at": now(),
        },
    )
    official = _put(store, "models/fixture.pth", b"official fixture weights")
    _put(
        store,
        "models/fixture.pth.json",
        json.dumps(
            {"weight_filename": "fixture.pth", "weight_sha256": official["sha256"]}
        ).encode(),
    )
    blobs = []
    for data in (b"ollama config", b"ollama model layer"):
        blob = _put(store, f"ollama/blobs/sha256-{_sha(data)}", data)
        blobs.append({"digest": f"sha256:{blob['sha256']}", "size": blob["size_bytes"]})
    _put(
        store,
        "ollama/manifests/registry.ollama.ai/library/fixture/latest",
        json.dumps({"config": blobs[0], "layers": blobs[1:]}).encode(),
    )
    _put(store, f"ollama/metadata/sha256-{'a' * 64}.json", b"{}")
    _put(store, f"logs/{job['id']}.log", b"fixture log")


def test_archive_covers_all_artifacts_and_streams_exact_checksums(workspace, tmp_path):
    _all_references(workspace)
    root = workspace.root
    events = []
    preview = archives.preview_workspace(root)
    assert preview["can_create"] and not preview["blocking_issues"]
    assert {row["id"] for row in preview["categories"]} == {
        "database",
        "assets",
        "frames",
        "datasets",
        "models",
        "assistance",
        "video_reviews",
        "reports",
        "imports",
        "ollama",
        "logs",
    }
    result = archives.create_archive(root, tmp_path / "backup.zip", progress=events.append)
    assert result["archive_sha256"] == _sha(result["path"].read_bytes())
    assert result["archive_size_bytes"] == result["path"].stat().st_size
    manifest = result["manifest"]
    assert manifest["file_count"] == preview["file_count"]
    assert manifest["counts"]["trained_models"] == 1
    assert manifest["counts"]["experiment_reports"] == 1
    assert events[-1]["phase"] == "complete"
    assert events[-1]["bytes_done"] == manifest["total_bytes"]
    assert events[-1]["files_done"] == manifest["file_count"]
    assert all(a["bytes_done"] <= b["bytes_done"] for a, b in zip(events, events[1:], strict=False))
    restored = tmp_path / "verified"
    with zipfile.ZipFile(result["path"]) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert all(info.flag_bits & 8 for info in archive.infolist())
        assert json.loads(archive.read("manifest.json")) == manifest
        assert len(archive.namelist()) == manifest["file_count"] + 1
        for item in manifest["files"]:
            data = archive.read(item["path"])
            assert len(data) == item["size_bytes"] and _sha(data) == item["sha256"]
        archive.extractall(restored)
    checked = archives.validate_database(
        restored, {item["path"]: item for item in manifest["files"]}, verify_hashes=True
    )
    assert checked["counts"] == manifest["counts"]
    assert checked["active_jobs"] == 0
    with closing(sqlite3.connect(restored / "iris.sqlite3")) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not list(tmp_path.glob(".iris-archive-*"))


def test_empty_workspace_needs_no_installed_official_models(tmp_path):
    store = Store(tmp_path / "empty")
    assert archives.preview_workspace(store.root)["can_create"]
    result = archives.create_archive(store.root, tmp_path / "empty.zip")
    assert result["manifest"]["file_count"] == 1
    assert not any(result["manifest"]["counts"].values())


def test_reference_inspection_reads_no_ordinary_artifact_files(workspace, tmp_path):
    _all_references(workspace)
    result = archives.create_archive(workspace.root, tmp_path / "backup.zip")
    inspection = tmp_path / "inspection"
    with zipfile.ZipFile(result["path"]) as archive:
        for item in result["manifest"]["files"]:
            if item["path"] == "iris.sqlite3" or archives.is_reference_document(item["path"]):
                archive.extract(item["path"], inspection)
    inventory = {item["path"]: item for item in result["manifest"]["files"]}
    checked = archives.validate_database(inspection, inventory)
    assert checked["counts"]["assets"] == 2
    assert any(name.startswith("ollama/blobs/") for name in checked["expected_hashes"])
    with pytest.raises(ArchiveError):
        archives.validate_database(inspection, inventory, verify_hashes=True)


def test_excludes_runtime_secrets_temporary_and_generated_files(workspace, tmp_path):
    names = [
        ".env",
        ".env.production",
        ".server.lock",
        ".venv/bin/python",
        "src/app.py",
        "uploads/pending.png",
        "exports/result.zip",
        "backups/old.zip",
        "restore_uploads/upload.zip",
        "workspace_operations/operation.json",
        "models/.hidden/weights.pth",
        "models/download.pth.part",
        "models/.env",
        "assets/credentials.json",
    ]
    for name in names:
        _put(workspace, name, b"must not be copied")
    before = archives.preview_workspace(workspace.root)
    assert before["can_create"]
    result = archives.create_archive(workspace.root, tmp_path / "backup.zip")
    included = {row["path"] for row in result["manifest"]["files"]}
    assert not included.intersection(names)
    assert {row["path"] for row in before["excluded"]} >= {".env", "uploads", "exports", ".venv"}


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_jobs_block_backup(workspace, tmp_path, status):
    _job(workspace, status)
    assert not archives.preview_workspace(workspace.root)["can_create"]
    with pytest.raises(ArchiveError, match="queued and running"):
        archives.create_archive(workspace.root, tmp_path / "backup.zip")
    assert not list(tmp_path.glob(".iris-archive-*"))
    assert not (tmp_path / "backup.zip").exists()


@pytest.mark.parametrize("when", ["before", "copying"])
def test_cancel_never_publishes_or_leaves_partial(workspace, tmp_path, when):
    state = {"cancel": when == "before"}

    def progress(_event):
        state["cancel"] = True

    with pytest.raises(ArchiveCancelled):
        archives.create_archive(
            workspace.root,
            tmp_path / "backup.zip",
            progress=progress,
            cancelled=lambda: state["cancel"],
        )
    assert not (tmp_path / "backup.zip").exists()
    assert not list(tmp_path.glob(".iris-archive-*"))


@pytest.mark.parametrize("race", [False, True])
def test_existing_destination_is_never_overwritten(workspace, tmp_path, race):
    destination = tmp_path / "backup.zip"
    if not race:
        destination.write_bytes(b"keep me")

    def progress(_event):
        if not destination.exists():
            destination.write_bytes(b"keep me")

    with pytest.raises(ArchiveError):
        archives.create_archive(workspace.root, destination, progress=progress)
    assert destination.read_bytes() == b"keep me"
    assert not list(tmp_path.glob(".iris-archive-*"))


def test_disk_space_checked_before_copy_and_after_database_snapshot(
    workspace, tmp_path, monkeypatch
):
    monkeypatch.setattr(archives.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    assert not archives.preview_workspace(workspace.root)["can_create"]
    with pytest.raises(ArchiveLimitError, match="disk space"):
        archives.create_archive(workspace.root, tmp_path / "backup.zip")
    assert not list(tmp_path.glob(".iris-archive-*"))
    calls = iter([2**40, 0])
    monkeypatch.setattr(
        archives.shutil, "disk_usage", lambda _path: SimpleNamespace(free=next(calls))
    )
    with pytest.raises(ArchiveLimitError, match="disk space"):
        archives.create_archive(workspace.root, tmp_path / "backup.zip")
    assert not list(tmp_path.glob(".iris-archive-*"))


@pytest.mark.parametrize("kind", ["file", "directory", "fifo"])
def test_managed_symlinks_and_special_files_rejected(workspace, tmp_path, kind):
    if kind == "file":
        (workspace.root / "assets" / "linked.bin").symlink_to(tmp_path / "train.png")
    elif kind == "directory":
        (workspace.root / "models").symlink_to(tmp_path, target_is_directory=True)
    else:
        os.mkfifo(workspace.root / "assets" / "namedpipe")
    assert not archives.preview_workspace(workspace.root)["can_create"]
    with pytest.raises(ArchiveError):
        archives.create_archive(workspace.root, tmp_path / "backup.zip")


@pytest.mark.parametrize(
    "mutation", ["content", "new_file", "removed_file", "database", "database_replaced"]
)
def test_concurrent_workspace_changes_abort_publication(workspace, tmp_path, mutation):
    changed = False
    asset = workspace.list("assets")[0]
    asset_path = workspace.root / asset["path"]

    def progress(event):
        nonlocal changed
        if changed or event["phase"] != "copying":
            return
        changed = True
        if mutation == "content":
            data = asset_path.read_bytes()
            asset_path.write_bytes(b"x" * len(data))
        elif mutation == "new_file":
            _put(workspace, "logs/concurrent.log", b"new")
        elif mutation == "removed_file":
            asset_path.unlink()
        elif mutation == "database":
            workspace.insert(
                "sessions",
                {"id": new_id(), "name": "Concurrent", "scene_group": "new", "created_at": now()},
            )
        else:
            replacement = tmp_path / "replacement.sqlite3"
            shutil.copyfile(workspace.db_path, replacement)
            os.replace(replacement, workspace.db_path)

    with pytest.raises(ArchiveError):
        archives.create_archive(workspace.root, tmp_path / "backup.zip", progress=progress)
    assert not (tmp_path / "backup.zip").exists()
    assert not list(tmp_path.glob(".iris-archive-*"))


def test_sqlite_backup_includes_committed_wal_rows(workspace, tmp_path):
    with closing(sqlite3.connect(workspace.db_path)) as writer:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO sessions VALUES (?,?,?,?)", ("wal-session", "Only in WAL", "wal", now())
        )
        writer.commit()
        assert Path(str(workspace.db_path) + "-wal").stat().st_size > 0
        result = archives.create_archive(workspace.root, tmp_path / "backup.zip")
        with (
            zipfile.ZipFile(result["path"]) as archive,
            closing(sqlite3.connect(":memory:")) as restored,
        ):
            restored.deserialize(archive.read("iris.sqlite3"))
            assert (
                restored.execute("SELECT name FROM sessions WHERE id='wal-session'").fetchone()[0]
                == "Only in WAL"
            )
    assert result["manifest"]["counts"]["sessions"] == 3


@pytest.mark.parametrize(
    "artifact", ["asset", "dataset_image", "trained", "ollama", "preview", "import", "receipt"]
)
def test_recorded_artifact_hashes_checked_during_creation(workspace, tmp_path, artifact):
    _all_references(workspace)
    if artifact == "asset":
        relative = workspace.list("assets")[0]["path"]
    elif artifact == "dataset_image":
        manifest = json.loads(
            (workspace.root / workspace.list("dataset_versions")[0]["path"]).read_bytes()
        )
        relative = manifest["frames"][0]["image_path"]
    elif artifact == "trained":
        relative = workspace.list("trained_models")[0]["path"]
    elif artifact == "ollama":
        relative = (
            next((workspace.root / "ollama/blobs").iterdir()).relative_to(workspace.root).as_posix()
        )
    elif artifact == "preview":
        relative = workspace.list("video_reviews")[0]["images"][0]["path"]
    elif artifact == "import":
        relative = workspace.list("dataset_imports")[0]["summary"]["images"][0]["path"]
    else:
        relative = "models/fixture.pth"
    path = workspace.root / relative
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ArchiveError, match="checksum"):
        archives.create_archive(workspace.root, tmp_path / "backup.zip")
    assert not (tmp_path / "backup.zip").exists()
    assert not list(tmp_path.glob(".iris-archive-*"))


def test_missing_frozen_images_or_checkpoint_block_fast_preview(workspace):
    _all_references(workspace)
    path = workspace.root / workspace.list("trained_models")[0]["path"]
    path.unlink()
    preview = archives.preview_workspace(workspace.root)
    assert not preview["can_create"]
    assert "missing" in preview["blocking_issues"][0]


@pytest.mark.parametrize(
    "value",
    [
        "not-json",
        '{"duplicate":1,"duplicate":2}',
        '{"number":NaN}',
        '{"number":Infinity}',
        "[" * 1200,
    ],
)
def test_malformed_json_columns_block_restore_compatible_backup(workspace, value):
    with workspace.connect() as connection:
        connection.execute("UPDATE assets SET metadata=?", (value,))
    assert not archives.preview_workspace(workspace.root)["can_create"]


@pytest.mark.parametrize("change", ["version", "table", "view", "trigger", "index", "column"])
def test_rejects_incompatible_or_executable_database_schema(workspace, change):
    commands = {
        "version": "PRAGMA user_version=99",
        "table": "CREATE TABLE surprise (value TEXT)",
        "view": "CREATE VIEW surprise AS SELECT * FROM assets",
        "trigger": "CREATE TRIGGER surprise AFTER INSERT ON sessions "
        "BEGIN UPDATE sessions SET name='oops'; END",
        "index": "CREATE INDEX unexpected ON assets(filename)",
        "column": "ALTER TABLE assets ADD COLUMN unexpected TEXT",
    }
    with workspace.connect() as connection:
        connection.execute(commands[change])
    assert not archives.preview_workspace(workspace.root)["can_create"]


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "a//b",
        "a/./b",
        "a/../b",
        "C:/file",
        "a\\b",
        "a\n.txt",
        "assets/NUL.png",
        "assets/con",
        "assets/COM1.bin",
        "assets/LPT9.txt",
        "assets/name.",
        "assets/name ",
    ],
)
def test_paths_reject_traversal_control_and_device_names(path):
    with pytest.raises(ArchiveError):
        archives.safe_member_path(path)


def test_archive_and_file_limits_are_enforced_and_cleanup(workspace, tmp_path, monkeypatch):
    monkeypatch.setattr(archives, "MAX_FILES", 1)
    with pytest.raises(ArchiveLimitError):
        archives.create_archive(workspace.root, tmp_path / "backup.zip")
    monkeypatch.setattr(archives, "MAX_FILES", 100000)
    monkeypatch.setattr(archives, "MAX_ARCHIVE_BYTES", 100)
    with pytest.raises(ArchiveLimitError):
        archives.create_archive(workspace.root, tmp_path / "backup.zip")
    assert not (tmp_path / "backup.zip").exists()
    assert not list(tmp_path.glob(".iris-archive-*"))


def test_nonfinite_exponent_and_report_snapshot_changes_rejected(workspace):
    _all_references(workspace)
    report = workspace.list("experiment_reports")[0]
    workspace.update("experiment_reports", report["id"], {"snapshot": {"changed": True}})
    assert not archives.preview_workspace(workspace.root)["can_create"]
    workspace.update("experiment_reports", report["id"], {"snapshot": {}})
    with workspace.connect() as connection:
        connection.execute("UPDATE assets SET metadata=?", ('{"number":1e999}',))
    assert not archives.preview_workspace(workspace.root)["can_create"]


def test_case_collisions_do_not_produce_unrestorable_archives(workspace, tmp_path):
    _put(workspace, "assets/duplicate.PNG", b"one")
    _put(workspace, "assets/DUPLICATE.png", b"two")
    assert not archives.preview_workspace(workspace.root)["can_create"]
    with pytest.raises(ArchiveError, match="collide"):
        archives.create_archive(workspace.root, tmp_path / "backup.zip")
