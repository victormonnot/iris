"""Portable-package integrity and publication using synthetic checkpoint bytes only."""

import hashlib
import json
import sqlite3
import stat
import threading
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy

import pytest
from test_model_exports import fixture_workspace, measurement, options, published, queued

from iris import model_exports as exports
from iris.workspace_archive import ArchiveError, create_archive
from iris.workspace_restore import inspect_archive


@pytest.fixture
def workspace(tmp_path):
    return fixture_workspace(tmp_path)


def files_in(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def write_bundle(path, files, *, special=None, extra=None):
    with warnings.catch_warnings(), zipfile.ZipFile(path, "w") as archive:
        warnings.simplefilter("ignore", UserWarning)
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            if name == "run.py" and special in {"symlink", "fifo", "directory_mode"}:
                info.create_system = 3
                kind = {
                    "symlink": stat.S_IFLNK,
                    "fifo": stat.S_IFIFO,
                    "directory_mode": stat.S_IFDIR,
                }[special]
                info.external_attr = (kind | 0o777) << 16
            if name == "run.py" and special == "deflated":
                info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        if extra:
            archive.writestr(*extra)


def assert_unpublished(store, row):
    saved = store.get("model_exports", row["id"])
    assert all(
        saved[key] is None for key in ("path", "manifest", "manifest_sha256", "archive_sha256")
    )
    assert not list((store.root / "model_exports").iterdir())


@pytest.mark.parametrize("changed", ["checkpoint", "image", "runner", "requirements", "readme"])
def test_preview_rejects_changed_source_or_packaged_resources(workspace, monkeypatch, changed):
    store = workspace[0]
    values = options(workspace)
    preview = exports.preview_export(store, **values)
    original_jobs = store.list("jobs")
    if changed in {"checkpoint", "image"}:
        relative = (
            workspace[1]["path"]
            if changed == "checkpoint"
            else preview["plan"]["frames"][0]["image_path"]
        )
        store.artifact_path(relative).write_bytes(b"source changed after preview")
    else:
        resources = exports._resources()
        name = {"runner": "run.py", "requirements": "requirements.txt", "readme": "README.md"}[
            changed
        ]
        resources[name] += b"\n# Changed after preview\n"
        monkeypatch.setattr(exports, "_resources", lambda: resources)
    with pytest.raises(ValueError, match="changed"):
        exports.create_export(
            store,
            **values,
            request_id=preview["request_id"],
            expected_fingerprint=preview["fingerprint"],
        )
    assert store.list("model_exports") == []
    assert store.list("jobs") == original_jobs


@pytest.mark.parametrize("moment", ["before_copy", "during_copy"])
def test_copy_detects_packaged_runtime_changes_and_cleans_staging(workspace, monkeypatch, moment):
    store, row = workspace[0], queued(workspace)
    original = exports._resources
    calls = 0

    def resources():
        nonlocal calls
        calls += 1
        result = original()
        if moment == "before_copy" or calls > 1:
            result["run.py"] += b"\n# Changed while preparing package\n"
        return result

    monkeypatch.setattr(exports, "_resources", resources)
    with pytest.raises(ValueError, match="changed|inconsistent"):
        exports.run_export(store, row["id"], lambda *_: None, lambda: False)
    assert_unpublished(store, row)


def test_image_changed_after_source_validation_cannot_publish(workspace):
    store, row = workspace[0], queued(workspace)
    image = store.artifact_path(row["config"]["frames"][0]["image_path"])
    changed = False

    def progress(*_):
        nonlocal changed
        if not changed:
            original = image.read_bytes()
            image.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            changed = True

    with pytest.raises(ValueError, match="changed"):
        exports.run_export(store, row["id"], progress, lambda: False)
    assert changed
    assert_unpublished(store, row)


@pytest.mark.parametrize("failure", ["rename", "database"])
def test_publication_failure_rolls_back_database_and_removes_only_staging(
    workspace, monkeypatch, failure
):
    store, row = workspace[0], queued(workspace)
    if failure == "rename":

        def fail(*_args, **_kwargs):
            raise OSError("Injected publication failure")

        monkeypatch.setattr(exports.os, "rename", fail)
        error = OSError
    else:
        original = store.connect

        class FailingConnection:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, statement, *args):
                if statement.startswith("UPDATE jobs SET status='succeeded'"):
                    raise sqlite3.OperationalError("Injected publication failure")
                return self.connection.execute(statement, *args)

            def __getattr__(self, name):
                return getattr(self.connection, name)

        @contextmanager
        def connect():
            with original() as connection:
                yield FailingConnection(connection)

        monkeypatch.setattr(store, "connect", connect)
        error = sqlite3.OperationalError
    with pytest.raises(error, match="Injected publication failure"):
        exports.run_export(store, row["id"], lambda *_: None, lambda: False)
    assert_unpublished(store, row)
    assert store.get("jobs", row["job_id"])["status"] == "queued"


def test_simultaneous_copy_attempts_publish_exactly_one_immutable_package(workspace):
    store, row = workspace[0], queued(workspace)
    barrier = threading.Barrier(2)

    def run(_):
        def progress(_amount, message):
            if message == "Copying model.pth":
                barrier.wait(timeout=10)

        try:
            return exports.run_export(store, row["id"], progress, lambda: False)
        except ValueError as exc:
            return {"rejected": str(exc)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    assert sum(result.get("published", False) for result in results) == 1
    assert all(
        result.get("published") or result.get("cancelled") or result.get("rejected")
        for result in results
    )
    saved = exports.export_detail(store, row["id"])
    assert saved["job"]["status"] == "succeeded"
    exports.read_bundle(exports.download_path(store, row["id"]), saved["manifest"])
    assert [path.name for path in (store.root / "model_exports").iterdir()] == [row["id"]]


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "missing",
        "duplicate",
        "traversal",
        "symlink",
        "fifo",
        "directory_mode",
        "deflated",
        "duplicate_manifest_key",
        "checkpoint_bytes",
        "reference_identity",
    ],
)
def test_nested_zip_rejects_unsafe_or_falsified_members(workspace, tmp_path, damage):
    store, row = workspace[0], published(workspace)
    files = files_in(exports.download_path(store, row["id"]))
    extra = None
    if damage == "extra":
        files["unplanned.py"] = b"unexpected"
    elif damage == "missing":
        del files["run.py"]
    elif damage == "duplicate":
        extra = ("run.py", files["run.py"])
    elif damage == "traversal":
        files["../outside.py"] = files.pop("run.py")
    elif damage == "duplicate_manifest_key":
        files["manifest.json"] = files["manifest.json"].replace(
            b'"format":', b'"format":"duplicate", "format":', 1
        )
    elif damage == "checkpoint_bytes":
        raw = files["model.pth"]
        files["model.pth"] = bytes([raw[0] ^ 1]) + raw[1:]
    elif damage == "reference_identity":
        reference = json.loads(files["parity/reference.json"])
        reference["frames"][0]["input_size"][0] += 1
        files["parity/reference.json"] = exports._canonical(reference)
        manifest = json.loads(files["manifest.json"])
        manifest["files"]["parity/reference.json"] = {
            "size": len(files["parity/reference.json"]),
            "sha256": hashlib.sha256(files["parity/reference.json"]).hexdigest(),
        }
        # The changed dimensions remain structurally valid, so compare against the saved manifest.
        files["manifest.json"] = exports._canonical(manifest)
    path = tmp_path / "damaged-model.zip"
    write_bundle(path, files, special=damage, extra=extra)
    with pytest.raises(ValueError):
        exports.read_bundle(path, row["manifest"])


@pytest.mark.parametrize("damage", ["parity", "timing", "execution", "fingerprint", "hardware"])
def test_archive_recomputes_measurements_instead_of_trusting_saved_summary(
    workspace, tmp_path, damage
):
    store, row = workspace[0], published(workspace)
    payload = measurement(row)
    preview = exports.preview_measurement(store, row["id"], payload)
    saved = exports.save_measurement(store, row["id"], payload, preview["fingerprint"])
    if damage == "parity":
        payload["samples"][0]["detections"][0]["score"] -= 0.01
        saved["fingerprint"] = exports._digest(payload)
    elif damage == "timing":
        payload["samples"][0]["timing"]["inference_ms"] += 10
        payload["samples"][0]["timing"]["total_ms"] += 10
        saved["fingerprint"] = exports._digest(payload)
    elif damage == "execution":
        saved["summary"]["execution_verified"] = True
    elif damage == "fingerprint":
        saved["fingerprint"] = "a" * 64
    else:
        payload["environment"]["processor"] = "A different producer declaration"
    store.update(
        "model_export_measurements",
        saved["id"],
        {"payload": payload, "summary": saved["summary"], "fingerprint": saved["fingerprint"]},
    )
    path = tmp_path / "false-measurements.zip"
    with pytest.raises(ArchiveError, match="Model export"):
        create_archive(store.root, path)
    assert not path.exists()


def test_archive_rejects_rehashed_package_reference_disagreeing_with_saved_evaluation(
    workspace, tmp_path
):
    store, row = workspace[0], published(workspace)
    path = exports.download_path(store, row["id"])
    files = files_in(path)
    config = deepcopy(row["config"])
    config["reference"]["frames"][0]["detections"][0]["score"] -= 0.01
    files["parity/reference.json"] = exports._canonical(config["reference"])
    manifest = deepcopy(row["manifest"])
    manifest["files"]["parity/reference.json"] = {
        "size": len(files["parity/reference.json"]),
        "sha256": hashlib.sha256(files["parity/reference.json"]).hexdigest(),
    }
    files["manifest.json"] = exports._canonical(manifest)
    write_bundle(path, files)
    # Every package-level descriptor is internally coherent; source evidence must catch this.
    exports.read_bundle(path)
    store.update(
        "model_exports",
        row["id"],
        {
            "config": config,
            "manifest": manifest,
            "manifest_sha256": exports._digest(manifest),
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    target = tmp_path / "rehashed-package.zip"
    with pytest.raises(ArchiveError, match="Model export"):
        create_archive(store.root, target)
    assert not target.exists()


def test_measurement_changed_after_preview_requires_fresh_preview(workspace):
    store, row = workspace[0], published(workspace)
    payload = measurement(row)
    preview = exports.preview_measurement(store, row["id"], payload)
    payload["load_ms"] += 1
    with pytest.raises(ValueError, match="changed"):
        exports.save_measurement(store, row["id"], payload, preview["fingerprint"])
    assert store.list("model_export_measurements") == []


@pytest.mark.parametrize("damage", ["missing_frozen_frame", "reference", "summary"])
def test_inspection_rejects_corrupt_database_even_when_outer_checksums_are_recomputed(
    workspace, tmp_path, damage
):
    store, row = workspace[0], published(workspace)
    payload = measurement(row)
    preview = exports.preview_measurement(store, row["id"], payload)
    exports.save_measurement(store, row["id"], payload, preview["fingerprint"])
    original = create_archive(store.root, tmp_path / "original.zip")
    files = files_in(original["path"])
    manifest = json.loads(files.pop("manifest.json"))
    database = tmp_path / "changed.sqlite3"
    database.write_bytes(files["iris.sqlite3"])
    with sqlite3.connect(database) as connection:
        if damage == "summary":
            summary = deepcopy(preview["summary"])
            summary["execution_verified"] = True
            connection.execute(
                "UPDATE model_export_measurements SET summary=?", (json.dumps(summary),)
            )
        else:
            config = deepcopy(row["config"])
            if damage == "missing_frozen_frame":
                config["frames"] = []
            else:
                config["reference"]["frames"][0]["detections"][0]["score"] -= 0.01
            connection.execute("UPDATE model_exports SET config=?", (json.dumps(config),))
    files["iris.sqlite3"] = database.read_bytes()
    for item in manifest["files"]:
        item["size_bytes"] = len(files[item["path"]])
        item["sha256"] = hashlib.sha256(files[item["path"]]).hexdigest()
    manifest["total_bytes"] = sum(item["size_bytes"] for item in manifest["files"])
    files["manifest.json"] = exports._canonical(manifest)
    forged = tmp_path / "coherent-outer-archive.zip"
    write_bundle(forged, files)
    with pytest.raises(ArchiveError, match="Model export"):
        inspect_archive(forged)
