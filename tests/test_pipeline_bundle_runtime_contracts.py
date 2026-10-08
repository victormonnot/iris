"""Versioned native inventory, safe extraction and v1 historical preservation."""

import hashlib
import json
import subprocess
import sys
import zipfile
from copy import deepcopy

import pytest
from test_pipeline_bundles import client as client
from test_pipeline_bundles import comparison as comparison
from test_pipeline_bundles import completed as completed
from test_pipeline_bundles import launch
from test_pipeline_bundles import local_fixture_weights as local_fixture_weights
from test_pipeline_bundles import payload as payload
from test_tracking_comparisons import run_worker

from iris import pipeline_bundle_runtime_contracts as runtime_contracts
from iris import pipeline_bundles
from iris.pipeline_bundle_contracts import (
    canonical,
    extract_bundle,
    inspect_bundle,
    validate_directory,
    validate_manifest,
)
from iris.pipeline_bundle_runtime_contracts import FORMAT_V2, portable_tracking_bytes


@pytest.fixture
def published(client, payload, monkeypatch):
    record = launch(client, payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    return client.app.state.store.artifact_path(job["result"]["path"]), job["result"]["manifest"]


def test_v2_inventory_port_and_safe_extract_are_standalone(published, tmp_path):
    path, manifest = published
    assert manifest["format"] == FORMAT_V2
    assert manifest["deployment_runtime"]["package_versions"]["torch"] == "2.10.0"
    target = tmp_path / "runtime"
    result = extract_bundle(path, target)
    assert result["manifest"] == manifest
    assert result["files_verified"] == len(manifest["files"]) + 1
    assert validate_directory(target, expected_manifest=manifest)["manifest"] == manifest
    original = (target / "provenance/tracking-original.py.txt").read_bytes()
    portable = (target / "iris_bundle/tracking.py").read_bytes()
    assert portable_tracking_bytes(original) == portable
    assert b"from iris." not in portable
    assert (target / "iris_bundle/_vendor/botsort/bot_sort.py").is_file()
    assert (target / "iris_bundle/_vendor/bytetrack/byte_tracker.py").is_file()
    assert not (target / "iris_bundle/_vendor/yolox").exists()
    assert b"sys.dont_write_bytecode = True" in (target / "iris_bundle/__init__.py").read_bytes()
    assert (
        manifest["tracker"]["runtime"]["provenance"]["adapter_sha256"]["tracking.py"]
        == manifest["deployment_runtime"]["tracker_port"]["source_sha256"]
    )
    assert (
        manifest["deployment_runtime"]["tracker_port"]["source_sha256"]
        != manifest["deployment_runtime"]["tracker_port"]["portable_sha256"]
    )


@pytest.mark.parametrize(
    "change",
    ["extra", "missing", "modified", "symlink", "directory", "cache_extra", "cache_symlink"],
)
def test_directory_requires_exact_regular_files(published, tmp_path, change):
    path, _ = published
    root = tmp_path / "runtime"
    extract_bundle(path, root)
    if change == "extra":
        (root / "untrusted.py").write_text("raise AssertionError('never execute')")
    elif change == "missing":
        (root / "iris_bundle/pipeline_runtime.py").unlink()
    elif change == "modified":
        (root / "iris_bundle/pipeline_runtime.py").write_text("print('not trusted')")
    elif change == "symlink":
        member = root / "requirements.txt"
        member.unlink()
        member.symlink_to(tmp_path / "outside")
    elif change == "directory":
        (root / "foreign").mkdir()
    else:
        cache = root / "iris_bundle/__pycache__"
        cache.mkdir()
        if change == "cache_extra":
            (cache / "foreign.cpython-312.pyc").write_bytes(b"not a declared module")
        else:
            (cache / "pipeline_runtime.cpython-312.pyc").symlink_to(root / "requirements.txt")
    with pytest.raises(ValueError):
        validate_directory(root)


def test_bytecode_for_declared_sources_is_also_rejected(published, tmp_path):
    path, _ = published
    root = tmp_path / "runtime"
    extract_bundle(path, root)
    cache = root / "iris_bundle/__pycache__"
    cache.mkdir()
    (cache / "pipeline_runtime.cpython-312.pyc").write_bytes(
        b"ignored bytecode; never executed by validator"
    )
    with pytest.raises(ValueError, match="bytecode"):
        validate_directory(root)


def test_standalone_inspector_never_writes_runtime_bytecode(published, tmp_path):
    path, manifest = published
    root = tmp_path / "runtime"
    extract_bundle(path, root)
    inspected = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import runpy,sys;sys.path.insert(0,sys.argv[1]);"
            "sys.argv=[sys.argv[1]+'/inspect_bundle.py',sys.argv[2]];"
            "runpy.run_path(sys.argv[0],run_name='__main__')",
            str(root),
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(inspected.stdout)["manifest"] == manifest
    assert not list(root.rglob("__pycache__"))
    assert validate_directory(root)["manifest"] == manifest


def test_packaged_launcher_propagates_parity_mismatch_exit_status(
    client, payload, monkeypatch, tmp_path
):
    resources = pipeline_bundles._resources

    def with_parity_failure(*args, **kwargs):
        result = resources(*args, **kwargs)
        result["iris_bundle/pipeline_runner.py"] = (
            b"import sys\ndef main():\n    assert sys.argv[1] == 'compare'\n    return 2\n"
        )
        return result

    monkeypatch.setattr(pipeline_bundles, "_resources", with_parity_failure)
    record = launch(client, payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    root = tmp_path / "runtime"
    extract_bundle(client.app.state.store.artifact_path(job["result"]["path"]), root)
    result = subprocess.run(
        [sys.executable, "-S", str(root / "run.py"), "compare"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    assert not list(root.rglob("__pycache__"))
    assert validate_directory(root)["manifest"] == job["result"]["manifest"]


def test_extract_existing_or_racing_destination_never_overwrites(published, tmp_path, monkeypatch):
    path, _ = published
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("Keep")
    with pytest.raises(ValueError, match="already exists"):
        extract_bundle(path, target)
    assert (target / "keep.txt").read_text() == "Keep"
    destination = tmp_path / "racing"
    rename = runtime_contracts._rename_new

    def raced(source, output):
        output.mkdir()
        (output / "keep.txt").write_text("Concurrent file")
        rename(source, output)

    monkeypatch.setattr(runtime_contracts, "_rename_new", raced)
    with pytest.raises(ValueError, match="already exists"):
        extract_bundle(path, destination)
    assert (destination / "keep.txt").read_text() == "Concurrent file"
    assert not list(tmp_path.glob(".iris-pipeline-extract-*"))


def test_extract_cancellation_leaves_no_published_directory(published, tmp_path):
    path, _ = published
    target = tmp_path / "cancelled"
    checks = [0]

    def cancel():
        checks[0] += 1
        if checks[0] > 5:
            raise RuntimeError("Cancelled")

    with pytest.raises(RuntimeError, match="Cancelled"):
        extract_bundle(path, target, checkpoint=cancel)
    assert not target.exists()
    assert not list(tmp_path.glob(".iris-pipeline-extract-*"))


def _rewrite(path, destination, member, transform):
    with zipfile.ZipFile(path) as source:
        data = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(data["manifest.json"])
    data[member] = transform(data[member])
    manifest["files"][member] = {
        "sha256": hashlib.sha256(data[member]).hexdigest(),
        "size": len(data[member]),
    }
    if member == "iris_bundle/tracking.py":
        manifest["deployment_runtime"]["tracker_port"]["portable_sha256"] = manifest["files"][
            member
        ]["sha256"]
    data["manifest.json"] = canonical(manifest)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_STORED, allowZip64=False
    ) as output:
        for name, content in data.items():
            output.writestr(name, content)
    return manifest


@pytest.mark.parametrize(
    "member",
    ["iris_bundle/tracking.py", "iris_bundle/_vendor/bytetrack/matching.py", "requirements.txt"],
)
def test_rehash_cannot_bypass_port_vendor_or_dependency_contract(published, tmp_path, member):
    path, _ = published
    tampered = tmp_path / "changed.zip"
    _rewrite(path, tampered, member, lambda raw: raw + b"\n# Changed after packaging\n")
    with pytest.raises(ValueError):
        inspect_bundle(tampered)


def test_deployment_requirements_cannot_be_silently_widened(published):
    _, manifest = published
    changed = deepcopy(manifest)
    changed["deployment_runtime"]["package_versions"]["torch"] = "3.0.0"
    with pytest.raises(ValueError, match="dependency"):
        validate_manifest(changed)


@pytest.mark.parametrize("adapter", ["tracking.py", "tracking_contracts.py"])
def test_runtime_adapters_must_equal_frozen_source_bytes(published, adapter):
    _, manifest = published
    changed = deepcopy(manifest)
    member = (
        "provenance/tracking-original.py.txt"
        if adapter == "tracking.py"
        else "iris_bundle/tracking_contracts.py"
    )
    changed["files"][member]["sha256"] = "a" * 64
    changed["deployment_runtime"] = runtime_contracts.runtime_descriptor(changed["files"])
    with pytest.raises(ValueError, match="adapters"):
        validate_manifest(changed)


def test_manifest_change_during_directory_validation_is_rejected(published, tmp_path, monkeypatch):
    path, _ = published
    target = tmp_path / "runtime"
    extract_bundle(path, target)
    check = runtime_contracts.validate_runtime_payloads

    def mutate(manifest, read):
        check(manifest, read)
        changed = deepcopy(manifest)
        changed["name"] = "Changed while checking payloads"
        (target / "manifest.json").write_bytes(canonical(changed))

    monkeypatch.setattr(runtime_contracts, "validate_runtime_payloads", mutate)
    with pytest.raises(ValueError, match="manifest changed"):
        validate_directory(target)


def test_historical_v1_job_remains_readable_after_new_v2_default(client, payload, monkeypatch):
    resources = pipeline_bundles._resources
    build = pipeline_bundles._manifest

    def old_resources(detector, profile, **_kwargs):
        return resources(detector, profile, bundle_format="iris-pipeline-bundle-v1")

    def old_manifest(*args, **kwargs):
        return build(*args, **{**kwargs, "bundle_format": "iris-pipeline-bundle-v1"})

    with monkeypatch.context() as context:
        context.setattr(pipeline_bundles, "_resources", old_resources)
        context.setattr(pipeline_bundles, "_manifest", old_manifest)
        record = launch(client, payload)
        job = run_worker(client, record, context)
        assert job["status"] == "succeeded", job["error"]
    saved = pipeline_bundles.get_bundle(client.app.state.store, record["id"])
    assert saved["bundle"] == job["result"]
    assert saved["bundle"]["manifest"]["format"] == "iris-pipeline-bundle-v1"
    assert "deployment_runtime" not in saved["bundle"]["manifest"]
    newer = launch(client, payload)
    assert (
        client.app.state.store.get("jobs", newer["id"])["params"]["manifest"]["format"] == FORMAT_V2
    )
