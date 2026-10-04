"""Small local archives exercise restore integrity; no model runtime is loaded."""

import hashlib
import json
import os
import sqlite3
import stat
import struct
import zipfile
from contextlib import closing
from types import SimpleNamespace

import pytest

from iris import workspace_restore as restore
from iris.store import SCHEMA_VERSION, TABLES, Store, new_id, now
from iris.workspace_archive import ArchiveCancelled, ArchiveError, ArchiveLimitError


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _database_bytes(root):
    target = root.parent / (root.name + "-snapshot.sqlite3")
    with (
        closing(sqlite3.connect(root / "iris.sqlite3")) as source,
        closing(sqlite3.connect(target)) as destination,
    ):
        source.backup(destination)
        destination.execute("PRAGMA journal_mode=DELETE")
    return target.read_bytes()


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "original")
    session_id, asset_id, frame_id = new_id(), new_id(), new_id()
    store.insert(
        "sessions",
        {"id": session_id, "name": "Fixture flight", "scene_group": "fixture", "created_at": now()},
    )
    # Byte fixtures suffice here: restore verifies saved file identities, never runs inference.
    media = b"local fixture bytes, not a flight video"
    asset_path, frame_path = f"assets/{asset_id}.bin", f"frames/{frame_id}.png"
    for relative in (asset_path, frame_path):
        path = store.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(media)
    store.insert(
        "assets",
        {
            "id": asset_id,
            "session_id": session_id,
            "filename": "fixture.bin",
            "kind": "image",
            "sha256": _sha(media),
            "size_bytes": len(media),
            "path": asset_path,
            "metadata": {},
            "created_at": now(),
        },
    )
    store.insert(
        "frames",
        {
            "id": frame_id,
            "session_id": session_id,
            "asset_id": asset_id,
            "width": 1,
            "height": 1,
            "sha256": "a" * 64,
            "perceptual_hash": "0",
            "path": frame_path,
            "created_at": now(),
        },
    )
    store.insert(
        "jobs",
        {
            "id": new_id(),
            "kind": "fixture",
            "status": "interrupted",
            "params": {},
            "message": "Historical job",
            "created_at": now(),
        },
    )
    return store


def _payload(store):
    result = {}
    for path in store.root.rglob("*"):
        if path.is_file() and path.name not in (
            "iris.sqlite3",
            "iris.sqlite3-wal",
            "iris.sqlite3-shm",
        ):
            result[path.relative_to(store.root).as_posix()] = path.read_bytes()
    result["iris.sqlite3"] = _database_bytes(store.root)
    return result


def _manifest(files):
    with sqlite3.connect(":memory:") as connection:
        connection.deserialize(files["iris.sqlite3"])
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in sorted(TABLES)
        }
    return {
        "protocol": "iris-workspace-archive-v1",
        "format_version": 1,
        "app_version": "0.19.0",
        "schema_version": version,
        "created_at": now(),
        "file_count": len(files),
        "total_bytes": sum(len(data) for data in files.values()),
        "counts": counts,
        "files": [
            {"path": name, "size_bytes": len(data), "sha256": _sha(data)}
            for name, data in sorted(files.items())
        ],
        "exclusions": [],
    }


def _write_archive(path, files, *, manifest=None, extras=(), compression=zipfile.ZIP_STORED):
    value = _manifest(files) if manifest is None else manifest
    document = value if isinstance(value, bytes) else json.dumps(value).encode()
    with zipfile.ZipFile(path, "w", compression=compression, allowZip64=True) as archive:
        archive.writestr("manifest.json", document)
        for name, data in files.items():
            archive.writestr(name, data)
        for info, data in extras:
            archive.writestr(info, data)
    return path


@pytest.fixture
def archive(workspace, tmp_path):
    return _write_archive(tmp_path / "workspace.zip", _payload(workspace))


def _restore(archive, destination, **kwargs):
    return restore.restore_archive(
        archive, destination, expected_archive_sha256=_sha(archive.read_bytes()), **kwargs
    )


def test_inspection_verifies_every_file_without_mutating_source(archive, workspace):
    before = {path: path.read_bytes() for path in workspace.root.rglob("*") if path.is_file()}
    progress = []
    result = restore.inspect_archive(archive, progress=progress.append)
    assert result["verified"] is True
    assert result["archive_sha256"] == _sha(archive.read_bytes())
    assert result["archive_size_bytes"] == archive.stat().st_size
    assert result["manifest"]["counts"]["frames"] == 1
    assert before == {
        path: path.read_bytes() for path in workspace.root.rglob("*") if path.is_file()
    }
    assert {step["phase"] for step in progress} >= {"inspecting", "checking_database"}


def test_restore_round_trip_bytes_ids_history_and_permissions(archive, workspace, tmp_path):
    preview = restore.inspect_archive(archive)
    target = tmp_path / "restored"
    result = restore.restore_archive(
        archive, target, expected_archive_sha256=preview["archive_sha256"]
    )
    assert result["path"] == target
    assert result["verified"] is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    with zipfile.ZipFile(archive) as saved:
        for item in preview["manifest"]["files"]:
            path = target / item["path"]
            assert path.read_bytes() == saved.read(item["path"])
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with sqlite3.connect(target / "iris.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert (
            connection.execute("SELECT id FROM frames").fetchone()[0]
            == workspace.list("frames")[0]["id"]
        )
        assert connection.execute("SELECT status FROM jobs").fetchone()[0] == "interrupted"
    assert not list(target.glob("*.lock"))
    assert not list(tmp_path.glob(".iris-restore-*"))


def test_restore_does_not_initialize_store_or_require_model_runtime(archive, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Restoration must not initialize or migrate Store")

    monkeypatch.setattr(Store, "__init__", forbidden)
    result = _restore(archive, tmp_path / "restored")
    assert result["verified"] is True


def test_zip64_archive_is_supported(workspace, tmp_path, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(zipfile, "ZIP64_LIMIT", 100)
        archive = _write_archive(tmp_path / "zip64.zip", _payload(workspace))
    assert b"PK\x06\x06" in archive.read_bytes()
    assert restore.inspect_archive(archive)["verified"]
    assert _restore(archive, tmp_path / "restored")["verified"]


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink"])
def test_restore_never_overwrites_existing_target(archive, tmp_path, kind):
    target = tmp_path / "existing"
    if kind == "file":
        target.write_bytes(b"preserve me")
    elif kind == "directory":
        target.mkdir()
    else:
        referent = tmp_path / "referent"
        if kind == "symlink":
            referent.mkdir()
        target.symlink_to(referent)
    with pytest.raises(ArchiveError, match="already exists"):
        _restore(archive, target)
    assert os.path.lexists(target)
    if kind == "file":
        assert target.read_bytes() == b"preserve me"
    assert not list(tmp_path.glob(".iris-restore-*"))


def test_atomic_publication_refuses_target_created_during_restore(archive, tmp_path):
    target = tmp_path / "racing-target"

    def progress(value):
        if value["phase"] == "publishing":
            target.mkdir()

    with pytest.raises(ArchiveError, match="already exists"):
        _restore(archive, target, progress=progress)
    assert list(target.iterdir()) == []
    assert not list(tmp_path.glob(".iris-restore-*"))


def test_publication_failure_cleans_only_owned_staging(archive, tmp_path, monkeypatch):
    unrelated = tmp_path / ".iris-restore-unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("unchanged")

    def fail(*args):
        raise ArchiveError("Fixture publication failure")

    monkeypatch.setattr(restore, "_publish", fail)
    with pytest.raises(ArchiveError, match="publication failure"):
        _restore(archive, tmp_path / "target")
    assert (unrelated / "keep").read_text() == "unchanged"
    assert list(tmp_path.glob(".iris-restore-*")) == [unrelated]
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize(
    "phase", ["checking_archive", "restoring", "checking_database", "publishing"]
)
def test_cancellation_removes_staging_without_publishing(archive, tmp_path, phase):
    state = {"stop": False}

    def progress(value):
        if value["phase"] == phase:
            state["stop"] = True

    with pytest.raises(ArchiveCancelled):
        _restore(
            archive, tmp_path / "cancelled", progress=progress, cancelled=lambda: state["stop"]
        )
    assert not (tmp_path / "cancelled").exists()
    assert not list(tmp_path.glob(".iris-restore-*"))


def test_wrong_preview_hash_refuses_restore(archive, tmp_path):
    with pytest.raises(ArchiveError, match="changed since"):
        restore.restore_archive(archive, tmp_path / "target", expected_archive_sha256="0" * 64)
    assert not (tmp_path / "target").exists()
    assert not list(tmp_path.glob(".iris-restore-*"))


@pytest.mark.parametrize("identity", [None, "", True, "F" * 64, "z" * 64, "a" * 63])
def test_restore_requires_exact_preview_identity(archive, tmp_path, identity):
    with pytest.raises(ArchiveError, match="SHA-256"):
        restore.restore_archive(archive, tmp_path / "target", expected_archive_sha256=identity)


def test_in_place_archive_change_and_reversal_is_rejected(archive, tmp_path):
    changed = False

    def progress(value):
        nonlocal changed
        if value["phase"] == "restoring" and not changed:
            original = archive.read_bytes()
            archive.write_bytes(original + b"changed")
            archive.write_bytes(original)
            changed = True

    with pytest.raises(ArchiveError, match="changed during"):
        _restore(archive, tmp_path / "target", progress=progress)
    assert changed
    assert not (tmp_path / "target").exists()
    assert not list(tmp_path.glob(".iris-restore-*"))


def test_refuses_restore_inside_current_workspace(archive, workspace):
    with pytest.raises(ArchiveError, match="separate workspace"):
        _restore(archive, workspace.root / "nested", forbidden_root=workspace.root)


def test_archive_symlink_is_not_followed(archive, tmp_path):
    alias = tmp_path / "alias.zip"
    alias.symlink_to(archive)
    with pytest.raises(ArchiveError, match="regular local file"):
        restore.inspect_archive(alias)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "uploads/../escape",
        "uploads//file",
        "uploads/./file",
        "C:/escape",
        "uploads\\escape",
        "NUL",
    ],
)
def test_unsafe_member_paths_are_rejected(workspace, tmp_path, name):
    archive = _write_archive(tmp_path / "unsafe.zip", _payload(workspace), extras=[(name, b"x")])
    with pytest.raises(ArchiveError):
        restore.inspect_archive(archive)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "mode", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFSOCK, stat.S_IFDIR]
)
def test_non_regular_members_are_rejected(workspace, tmp_path, mode):
    info = zipfile.ZipInfo("assets/special")
    info.create_system = 3
    info.external_attr = (mode | 0o777) << 16
    archive = _write_archive(
        tmp_path / "special.zip", _payload(workspace), extras=[(info, b"target")]
    )
    with pytest.raises(ArchiveError, match="regular files"):
        restore.inspect_archive(archive)


def test_duplicate_members_are_rejected(workspace, tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        archive = _write_archive(
            tmp_path / "duplicate.zip", _payload(workspace), extras=[("iris.sqlite3", b"duplicate")]
        )
    with pytest.raises(ArchiveError, match="duplicated"):
        restore.inspect_archive(archive)


def test_case_collisions_are_rejected(workspace, tmp_path):
    archive = _write_archive(
        tmp_path / "case.zip",
        _payload(workspace),
        extras=[("assets/fixture", b"one"), ("assets/FIXTURE", b"two")],
    )
    with pytest.raises(ArchiveError, match="ambiguous"):
        restore.inspect_archive(archive)


def test_file_directory_collisions_are_rejected(workspace, tmp_path):
    archive = _write_archive(
        tmp_path / "collision.zip",
        _payload(workspace),
        extras=[("ollama/manifests/a/parent", b"one"), ("ollama/manifests/a/parent/child", b"two")],
    )
    with pytest.raises(ArchiveError, match="parent directory"):
        restore.inspect_archive(archive)


@pytest.mark.parametrize("compression", [zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_compressed_archives_are_rejected_before_extraction(workspace, tmp_path, compression):
    archive = _write_archive(
        tmp_path / "compressed.zip", _payload(workspace), compression=compression
    )
    with pytest.raises(ArchiveError, match="ZIP_STORED"):
        restore.inspect_archive(archive)


def test_encrypted_flag_is_rejected(archive):
    data = bytearray(archive.read_bytes())
    position = data.index(b"PK\x01\x02")
    flags = struct.unpack_from("<H", data, position + 8)[0]
    struct.pack_into("<H", data, position + 8, flags | 1)
    archive.write_bytes(data)
    with pytest.raises(ArchiveError, match="unencrypted"):
        restore.inspect_archive(archive)


def test_inventory_must_match_archive_exactly(workspace, tmp_path):
    files = _payload(workspace)
    archive = _write_archive(tmp_path / "extra.zip", files, extras=[("logs/orphan.log", b"extra")])
    with pytest.raises(ArchiveError, match="inventory"):
        restore.inspect_archive(archive)


def test_missing_payload_is_rejected(workspace, tmp_path):
    files = _payload(workspace)
    manifest = _manifest(files)
    del files[next(name for name in files if name.startswith("frames/"))]
    archive = _write_archive(tmp_path / "missing.zip", files, manifest=manifest)
    with pytest.raises(ArchiveError, match="inventory"):
        restore.inspect_archive(archive)


def test_payload_digest_mismatch_is_rejected(workspace, tmp_path):
    files = _payload(workspace)
    manifest = _manifest(files)
    name = next(name for name in files if name.startswith("frames/"))
    files[name] = b"x" * len(files[name])
    archive = _write_archive(tmp_path / "changed.zip", files, manifest=manifest)
    with pytest.raises(ArchiveError, match="SHA-256"):
        restore.inspect_archive(archive)


def test_database_reference_missing_from_inventory_is_rejected(workspace, tmp_path):
    files = _payload(workspace)
    del files[next(name for name in files if name.startswith("frames/"))]
    archive = _write_archive(tmp_path / "broken-ref.zip", files)
    with pytest.raises(ArchiveError):
        restore.inspect_archive(archive)


def test_database_recorded_hash_must_match_manifest_hash(workspace, tmp_path):
    files = _payload(workspace)
    name = next(name for name in files if name.startswith("assets/"))
    files[name] = b"x" * len(files[name])
    archive = _write_archive(tmp_path / "lying-manifest.zip", files)
    with pytest.raises(ArchiveError, match="hash"):
        restore.inspect_archive(archive)


def test_manifest_row_counts_must_match_database(workspace, tmp_path):
    files = _payload(workspace)
    manifest = _manifest(files)
    manifest["counts"]["frames"] += 1
    archive = _write_archive(tmp_path / "count.zip", files, manifest=manifest)
    with pytest.raises(ArchiveError, match="counts"):
        restore.inspect_archive(archive)


def test_active_jobs_cannot_be_restored(workspace, tmp_path):
    with workspace.connect() as connection:
        connection.execute("UPDATE jobs SET status='queued'")
    archive = _write_archive(tmp_path / "queued.zip", _payload(workspace))
    with pytest.raises(ArchiveError, match="queued|active|running"):
        restore.inspect_archive(archive)


@pytest.mark.parametrize("change", ["old_schema", "extra_table", "trigger", "view", "foreign_key"])
def test_unexpected_or_broken_database_schema_is_rejected(workspace, tmp_path, change):
    with workspace.connect() as connection:
        if change == "old_schema":
            connection.execute("PRAGMA user_version=11")
        elif change == "extra_table":
            connection.execute("CREATE TABLE surprise (value TEXT)")
        elif change == "trigger":
            connection.execute(
                "CREATE TRIGGER surprise AFTER INSERT ON sessions BEGIN SELECT 1; END"
            )
        elif change == "view":
            connection.execute("CREATE VIEW surprise AS SELECT * FROM sessions")
        else:
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("UPDATE assets SET session_id='missing'")
    archive = _write_archive(tmp_path / "schema.zip", _payload(workspace))
    with pytest.raises(ArchiveError):
        restore.inspect_archive(archive)


def test_duplicate_json_keys_are_rejected(workspace, tmp_path):
    files = _payload(workspace)
    text = json.dumps(_manifest(files))
    document = ('{"format_version":1,' + text[1:]).encode()
    archive = _write_archive(tmp_path / "json.zip", files, manifest=document)
    with pytest.raises(ArchiveError, match="duplicate JSON keys"):
        restore.inspect_archive(archive)


def test_file_count_limit_is_checked_before_zipfile_allocation(archive, monkeypatch):
    data = bytearray(archive.read_bytes())
    struct.pack_into("<HH", data, len(data) - 22 + 8, 200, 200)
    archive.write_bytes(data)
    monkeypatch.setattr(restore, "MAX_FILES", 100)

    def forbidden(*args, **kwargs):
        pytest.fail("ZipFile must not allocate an oversized directory")

    monkeypatch.setattr(restore.zipfile, "ZipFile", forbidden)
    with pytest.raises(ArchiveLimitError, match="file count"):
        restore.inspect_archive(archive)


def test_archive_size_limit_is_checked_before_reading(archive, monkeypatch):
    monkeypatch.setattr(restore, "MAX_ARCHIVE_BYTES", 100)
    with pytest.raises(ArchiveLimitError, match="size limit"):
        restore.inspect_archive(archive)


def test_manifest_size_limit_is_checked_before_json_decode(archive, monkeypatch):
    monkeypatch.setattr(restore, "MAX_MANIFEST_BYTES", 100)
    with pytest.raises(ArchiveLimitError, match="manifest"):
        restore.inspect_archive(archive)


def test_truncated_and_appended_archives_are_rejected(archive, tmp_path):
    original = archive.read_bytes()
    for name, data in (("truncated", original[:-5]), ("appended", original + b"trailer")):
        path = tmp_path / (name + ".zip")
        path.write_bytes(data)
        with pytest.raises(ArchiveError):
            restore.inspect_archive(path)


def test_inspection_does_not_retain_extracted_media(archive, tmp_path, monkeypatch):
    captured = []
    original = restore._database

    def inspect(root, inventory, manifest):
        captured.extend(
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
        )
        return original(root, inventory, manifest)

    monkeypatch.setattr(restore, "_database", inspect)
    restore.inspect_archive(archive)
    assert captured == ["iris.sqlite3"]


def test_failed_restore_leaves_original_archive_unchanged(archive, tmp_path):
    before = archive.read_bytes()
    with pytest.raises(ArchiveError):
        _restore(archive, tmp_path / "missing-parent" / "target")
    assert archive.read_bytes() == before
    assert not (tmp_path / "missing-parent").exists()


def _reference_files(workspace):
    files = _payload(workspace)
    weight, blob = b"fixture detector bytes", b"fixture Ollama bytes"
    files["models/fixture.pth"] = weight
    files["models/fixture.pth.json"] = json.dumps(
        {"weight_filename": "fixture.pth", "weight_sha256": _sha(weight)}
    ).encode()
    files["ollama/blobs/sha256-" + _sha(blob)] = blob
    files["ollama/manifests/registry/library/fixture/latest"] = json.dumps(
        {"config": {"digest": "sha256:" + _sha(blob), "size": len(blob)}, "layers": []}
    ).encode()
    dataset_id, frame_id = new_id(), workspace.list("frames")[0]["id"]
    image_path = f"datasets/{dataset_id}/images/{frame_id}.png"
    image = b"frozen fixture image bytes"
    manifest_path = f"datasets/{dataset_id}/manifest.json"
    document = json.dumps(
        {
            "id": dataset_id,
            "frames": [
                {"frame_id": frame_id, "image_path": image_path, "image_file_sha256": _sha(image)}
            ],
        }
    ).encode()
    workspace.insert(
        "dataset_versions",
        {
            "id": dataset_id,
            "name": "Reference fixture",
            "path": manifest_path,
            "manifest_sha256": _sha(document),
            "summary": {},
            "created_at": now(),
        },
    )
    files.update(
        {
            manifest_path: document,
            image_path: image,
            "iris.sqlite3": _database_bytes(workspace.root),
        }
    )
    return files


def test_inspection_keeps_only_database_and_reference_documents(workspace, tmp_path, monkeypatch):
    files = _reference_files(workspace)
    archive = _write_archive(tmp_path / "references.zip", files)
    captured, original = [], restore._database

    def check(root, inventory, manifest):
        captured.extend(
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
        )
        return original(root, inventory, manifest)

    monkeypatch.setattr(restore, "_database", check)
    preview = restore.inspect_archive(archive)
    assert set(captured) == {
        name for name in files if name == "iris.sqlite3" or restore.is_reference_document(name)
    }
    monkeypatch.setattr(restore, "_database", original)
    result = restore.restore_archive(
        archive, tmp_path / "restored", expected_archive_sha256=preview["archive_sha256"]
    )
    assert all((result["path"] / name).read_bytes() == data for name, data in files.items())


@pytest.mark.parametrize("kind", ["dataset", "model_receipt", "ollama"])
def test_reference_document_missing_blob_or_image_is_rejected(workspace, tmp_path, kind):
    files = _reference_files(workspace)
    if kind == "dataset":
        del files[
            next(name for name in files if name.startswith("datasets/") and name.endswith(".png"))
        ]
    elif kind == "model_receipt":
        del files["models/fixture.pth"]
    else:
        del files[next(name for name in files if name.startswith("ollama/blobs/"))]
    archive = _write_archive(tmp_path / "missing-reference.zip", files)
    with pytest.raises(ArchiveError, match="artifact.*missing"):
        restore.inspect_archive(archive)


def test_reference_document_size_is_bounded(workspace, tmp_path, monkeypatch):
    archive = _write_archive(tmp_path / "references.zip", _reference_files(workspace))
    monkeypatch.setattr(restore, "MAX_REFERENCE_BYTES", 10)
    with pytest.raises(ArchiveLimitError, match="reference document"):
        restore.inspect_archive(archive)


def test_bad_crc_is_rejected_even_when_zip_is_structurally_valid(archive):
    with zipfile.ZipFile(archive) as saved:
        entry = next(item for item in saved.infolist() if item.filename.startswith("frames/"))
        offset = entry.header_offset
    data = bytearray(archive.read_bytes())
    name_size, extra_size = struct.unpack_from("<HH", data, offset + 26)
    data[offset + 30 + name_size + extra_size] ^= 1
    archive.write_bytes(data)
    with pytest.raises(ArchiveError, match="corrupt"):
        restore.inspect_archive(archive)


@pytest.mark.parametrize("document", ["not-json", '{"value":NaN}', '{"value":1,"value":2}'])
def test_invalid_saved_json_is_rejected(workspace, tmp_path, document):
    with workspace.connect() as connection:
        connection.execute("UPDATE assets SET metadata=?", (document,))
    archive = _write_archive(tmp_path / "invalid-metadata.zip", _payload(workspace))
    with pytest.raises(ArchiveError):
        restore.inspect_archive(archive)


def test_dos_directory_attribute_is_rejected(workspace, tmp_path):
    info = zipfile.ZipInfo("assets/directory")
    info.create_system = 0
    info.external_attr = 0x10
    archive = _write_archive(tmp_path / "directory.zip", _payload(workspace), extras=[(info, b"")])
    with pytest.raises(ArchiveError, match="regular files"):
        restore.inspect_archive(archive)


def test_insufficient_disk_space_is_rejected_before_extraction(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(restore.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(ArchiveLimitError, match="free disk space"):
        _restore(archive, tmp_path / "target")
    assert not (tmp_path / "target").exists()
    assert not list(tmp_path.glob(".iris-restore-*"))
