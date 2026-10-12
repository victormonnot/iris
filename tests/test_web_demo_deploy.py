"""Publication accepts exact artifacts and preserves an independently usable old release."""

import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECEIVER = ROOT / "deploy" / "web-demo" / "receive.py"
spec = importlib.util.spec_from_file_location("iris_demo_receiver", RECEIVER)
receiver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receiver)
A, B, C = "a" * 40, "b" * 40, "c" * 40


def artifact(revision=A, content=b"first version"):
    version = hashlib.sha256(content).hexdigest()
    prefix = f"assets/{version}/"
    files = {
        "index.html": f'<script src="./{prefix}static/app.js"></script>'.encode(),
        prefix + "static/app.js": content,
        prefix + "seed/CREDITS.txt": b"Public sample attribution",
    }
    manifest = {
        "format": "iris-web-demo-site-v1",
        "source_revision": revision,
        "asset_version": version,
        "files": {name: receiver.fingerprint(data) for name, data in files.items()},
    }
    files["manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    return manifest, files


def zipped(files, extras=()):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        for name, content in extras:
            archive.writestr(name, content)
    data.seek(0)
    return data


def publish(root, revision=A, content=b"first version"):
    value, files = artifact(revision, content)
    result = receiver.deploy(root, revision, zipped(files))
    return value, files, result


def state(root):
    return {
        p.relative_to(root).as_posix(): ("link", str(p.readlink()))
        if p.is_symlink()
        else ("directory",)
        if p.is_dir()
        else ("file", p.read_bytes())
        for p in root.rglob("*")
    }


def test_publish_retains_old_assets_and_cas_rollback(tmp_path):
    site = tmp_path / "site"
    first, _, result = publish(site)
    assert result == {
        "action": "deploy",
        "revision": A,
        "previous": None,
        "asset_version": first["asset_version"],
    }
    assert (site / "current").readlink() == Path("releases") / A
    old = site / "assets" / first["asset_version"] / "static/app.js"
    second, _, result = publish(site, B, b"second version")
    assert result["previous"] == A
    assert old.read_bytes() == b"first version"
    assert (site / "current/manifest.json").read_bytes() == (
        site / "releases" / B / "manifest.json"
    ).read_bytes()
    prior = state(site)
    with pytest.raises(ValueError, match="Current revision changed"):
        receiver.rollback(site, A)
    assert state(site) == prior
    result = receiver.rollback(site, B)
    assert result["revision"] == A and result["previous"] == B
    assert (site / "assets" / second["asset_version"] / "static/app.js").is_file()


def test_idempotent_retry_does_not_destroy_previous(tmp_path):
    publish(tmp_path)
    _, files, _ = publish(tmp_path, B, b"second")
    before = state(tmp_path)
    result = receiver.deploy(tmp_path, B, zipped(files))
    assert result["previous"] == A
    assert state(tmp_path) == before


@pytest.mark.parametrize(
    "kind",
    [
        "wrong_revision",
        "missing_revision",
        "corrupt_bytes",
        "extra_file",
        "missing_file",
        "traversal",
        "absolute",
        "backslash",
        "symlink",
        "duplicate",
        "extra_directory",
        "duplicate_json",
        "wrong_asset_prefix",
        "boolean_size",
        "manifest_too_large",
    ],
)
def test_rejected_archive_leaves_entire_published_tree_unchanged(tmp_path, monkeypatch, kind):
    publish(tmp_path)
    manifest, files = artifact(B, b"second")
    extras = []
    if kind == "wrong_revision":
        manifest["source_revision"] = C
    elif kind == "missing_revision":
        del manifest["source_revision"]
    elif kind == "corrupt_bytes":
        files["index.html"] += b"unrecorded change"
    elif kind == "extra_file":
        files["unexpected.txt"] = b"extra"
    elif kind == "missing_file":
        del files[next(name for name in files if name.endswith("app.js"))]
    elif kind in {"traversal", "absolute", "backslash"}:
        extras.append(
            (
                {"traversal": "../outside", "absolute": "/outside", "backslash": "assets\\outside"}[
                    kind
                ],
                b"x",
            )
        )
    elif kind == "symlink":
        entry = zipfile.ZipInfo("linked")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        extras.append((entry, b"/etc/passwd"))
    elif kind == "duplicate":
        extras.append(("index.html", b"duplicate"))
    elif kind == "extra_directory":
        extras.append(("unexpected/", b""))
    elif kind == "wrong_asset_prefix":
        manifest["asset_version"] = "f" * 64
    elif kind == "boolean_size":
        manifest["files"]["index.html"]["size"] = True
    elif kind == "manifest_too_large":
        monkeypatch.setattr(receiver, "MAX_MANIFEST", 10)
    files["manifest.json"] = json.dumps(manifest).encode()
    if kind == "duplicate_json":
        files["manifest.json"] = files["manifest.json"].replace(
            b'{"format":', b'{"format":"duplicate","format":', 1
        )
    before = state(tmp_path)
    with pytest.warns(UserWarning) if kind == "duplicate" else nullcontext():
        archive = zipped(files, extras)
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        receiver.deploy(tmp_path, B, archive)
    assert state(tmp_path) == before
    assert not (tmp_path.parent / "outside").exists()


@pytest.mark.parametrize("limit", ["MAX_COMPRESSED", "MAX_UNCOMPRESSED", "MAX_FILES"])
def test_archive_limits_apply_before_site_mutation(tmp_path, monkeypatch, limit):
    publish(tmp_path)
    _, files = artifact(B, b"large" * 100)
    before = state(tmp_path)
    monkeypatch.setattr(receiver, limit, 2)
    with pytest.raises(ValueError):
        receiver.deploy(tmp_path, B, zipped(files))
    assert state(tmp_path) == before


def test_truncated_transfer_and_activation_failure_preserve_current(tmp_path, monkeypatch):
    publish(tmp_path)
    publish(tmp_path, B, b"second")
    _, files = artifact(C, b"third")
    before = state(tmp_path)
    with pytest.raises(zipfile.BadZipFile):
        receiver.deploy(tmp_path, C, io.BytesIO(zipped(files).read()[:-40]))
    assert state(tmp_path) == before
    replace = os.replace

    def fail_current(source, destination):
        if destination == tmp_path / "current":
            raise OSError("Simulated atomic activation failure")
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(receiver.os, "replace", fail_current)
        with pytest.raises(OSError, match="activation failure"):
            receiver.deploy(tmp_path, C, zipped(files))
    assert receiver.pointer(tmp_path, "current") == B
    assert receiver.pointer(tmp_path, "previous") == A
    assert receiver.stored_release(tmp_path, B)["source_revision"] == B
    # A verified, unreferenced staged release can be retried without overwriting.
    assert receiver.deploy(tmp_path, C, zipped(files))["previous"] == B


def test_immutable_revision_and_asset_collision_refused(tmp_path):
    publish(tmp_path)
    _, files = artifact(A, b"changed")
    before = state(tmp_path)
    with pytest.raises(ValueError, match="immutable"):
        receiver.deploy(tmp_path, A, zipped(files))
    assert state(tmp_path) == before
    manifest, files = artifact(B)
    key = next(name for name in files if name.endswith("app.js"))
    files[key] = b"reuse same asset name with other bytes"
    manifest["files"][key] = receiver.fingerprint(files[key])
    files["manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(ValueError, match="immutable"):
        receiver.deploy(tmp_path, B, zipped(files))
    assert state(tmp_path) == before


def test_corrupt_previous_release_cannot_be_activated(tmp_path):
    first, _, _ = publish(tmp_path)
    publish(tmp_path, B, b"second")
    (tmp_path / "assets" / first["asset_version"] / "static/app.js").write_bytes(b"corrupt")
    before = state(tmp_path)
    with pytest.raises(ValueError, match="integrity"):
        receiver.rollback(tmp_path, B)
    assert state(tmp_path) == before


def test_symlinked_storage_or_pointer_cannot_escape_root(tmp_path):
    site, outside = tmp_path / "site", tmp_path / "outside"
    site.mkdir()
    outside.mkdir()
    (site / "assets").symlink_to(outside, target_is_directory=True)
    _, files = artifact()
    with pytest.raises(ValueError, match="symlink"):
        receiver.deploy(site, A, zipped(files))
    assert list(outside.iterdir()) == []
    (site / "assets").unlink()
    publish(site)
    (site / "current").unlink()
    (site / "current").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="pointer"):
        receiver.rollback(site, A)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "deploy",
        "deploy " + A + "\n",
        "deploy  " + A,
        "deploy " + A + "; id",
        "rollback HEAD",
        "sh",
        "deploy " + "A" * 40,
    ],
)
def test_ssh_original_command_is_exact(value):
    with pytest.raises(ValueError):
        receiver.command(value)


def test_forced_command_cli_and_lock_serialization(tmp_path):
    publish(tmp_path)
    _, files = artifact(B, b"second")
    with receiver.locked(tmp_path):
        process = subprocess.Popen(
            [sys.executable, str(RECEIVER), "--root", str(tmp_path), "--ssh"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "SSH_ORIGINAL_COMMAND": f"deploy {B}"},
        )
        process.stdin.write(zipped(files).read())
        process.stdin.close()
        process.stdin = None
        time.sleep(0.15)
        assert process.poll() is None
        assert receiver.pointer(tmp_path, "current") == A
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    assert json.loads(stdout)["revision"] == B
    refused = subprocess.run(
        [sys.executable, str(RECEIVER), "--root", str(tmp_path), "--ssh"],
        input=b"",
        capture_output=True,
        env={**os.environ, "SSH_ORIGINAL_COMMAND": "deploy " + B + " && id"},
        check=False,
    )
    assert refused.returncode == 1 and refused.stdout == b""
    assert receiver.pointer(tmp_path, "current") == B


def test_structural_zip_directories_and_published_permissions(tmp_path):
    value, files = artifact()
    directories = [("assets/", b""), (f"assets/{value['asset_version']}/", b"")]
    receiver.deploy(tmp_path, A, zipped(files, directories))
    for path in (tmp_path / "assets").rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o755 if path.is_dir() else 0o644)
