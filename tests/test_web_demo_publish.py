"""The deployment client authenticates build bytes and verifies before accepting promotion."""

import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "deploy/web-demo/publish.py"
REVISION = "1" * 40
PREVIOUS = "2" * 40
VERSION = "a" * 64


@pytest.fixture
def client(monkeypatch):
    spec = importlib.util.spec_from_file_location("iris_test_publish", SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifact(client, tmp_path):
    directory = tmp_path / "artifact"
    files = {
        "index.html": b"<!doctype html><title>Iris</title>",
        f"assets/{VERSION}/static/app.js": b"'use strict';",
    }
    manifest = {
        "format": client.FORMAT,
        "source_revision": REVISION,
        "asset_version": VERSION,
        "files": {
            name: {"sha256": client.digest(data), "size": len(data)} for name, data in files.items()
        },
    }
    files["manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return directory, client.digest(files["manifest.json"]), files


def arguments(tmp_path, artifact):
    identity, known_hosts = tmp_path / "identity", tmp_path / "known_hosts"
    identity.write_text("test identity placeholder")
    known_hosts.write_text("test pinned host placeholder")
    return {
        "artifact": artifact[0],
        "revision": REVISION,
        "manifest_sha256": artifact[1],
        "url": "https://iris.example/demo/",
        "host": "192.0.2.1",
        "identity": identity,
        "known_hosts": known_hosts,
    }


def test_zip_contains_only_authenticated_files_without_a_parent_directory(client, artifact):
    directory, checksum, expected = artifact
    prepared = client.prepare_artifact(directory, REVISION, checksum)
    with zipfile.ZipFile(io.BytesIO(prepared.archive)) as archive:
        assert set(archive.namelist()) == set(expected)
        assert all(not item.is_dir() for item in archive.infolist())
        assert {name: archive.read(name) for name in archive.namelist()} == expected
    assert client.prepare_artifact(directory, REVISION, checksum).archive == prepared.archive


def test_actual_export_zip_is_accepted_by_the_restricted_receiver(client, tmp_path):
    root = SOURCE.parents[2]
    output = tmp_path / "built"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples/web-demo/export.py"),
            "--output",
            str(output),
            "--revision",
            REVISION,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    manifest_bytes = (output / "manifest.json").read_bytes()
    prepared = client.prepare_artifact(output, REVISION, client.digest(manifest_bytes))
    receiver_spec = importlib.util.spec_from_file_location(
        "iris_publish_receiver", SOURCE.with_name("receive.py")
    )
    receiver = importlib.util.module_from_spec(receiver_spec)
    receiver_spec.loader.exec_module(receiver)
    deployed = tmp_path / "deployed"
    confirmation = receiver.deploy(deployed, REVISION, io.BytesIO(prepared.archive))
    assert confirmation == {
        "action": "deploy",
        "revision": REVISION,
        "previous": None,
        "asset_version": prepared.manifest["asset_version"],
    }
    assert (deployed / "current/manifest.json").read_bytes() == manifest_bytes
    assert (deployed / "current/index.html").read_bytes() == prepared.files["index.html"]


@pytest.mark.parametrize(
    "mutation",
    ["extra", "changed", "missing", "symlink", "empty_directory", "manifest", "wrong_revision"],
)
def test_artifact_changes_fail_before_any_ssh(client, artifact, tmp_path, monkeypatch, mutation):
    directory, _checksum, _files = artifact
    options = arguments(tmp_path, artifact)
    if mutation == "extra":
        (directory / "private.txt").write_text("unlisted")
    elif mutation == "changed":
        (directory / "index.html").write_text("changed")
    elif mutation == "missing":
        (directory / "index.html").unlink()
    elif mutation == "symlink":
        (directory / "alias").symlink_to(directory / "index.html")
    elif mutation == "empty_directory":
        (directory / "unlisted").mkdir()
    elif mutation == "manifest":
        (directory / "manifest.json").write_text("{}")
    else:
        options["revision"] = PREVIOUS
    monkeypatch.setattr(
        client, "remote", lambda *_args: pytest.fail("Invalid artifact reached SSH")
    )
    with pytest.raises((client.PublishError, OSError)):
        client.publish(**options)


@pytest.mark.parametrize(
    "url",
    [
        "http://iris.example/",
        "https://user:password@iris.example/",
        "https://iris.example/?token=x",
        "https://iris.example/#demo",
        "https://iris.example/../demo/",
        "https://iris.example/%2e%2e/",
        "https://iris.example/\n",
    ],
)
def test_public_destination_requires_a_fixed_https_namespace(client, url):
    with pytest.raises(client.PublishError):
        client.public_base(url)
    assert client.public_base("https://iris.example/demo") == "https://iris.example/demo/"


def test_success_verifies_every_public_byte_and_pins_ssh_identity(
    client, artifact, tmp_path, monkeypatch
):
    options = arguments(tmp_path, artifact)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "action": "deploy",
                    "revision": REVISION,
                    "previous": PREVIOUS,
                    "asset_version": VERSION,
                }
            ).encode(),
        )

    monkeypatch.setattr(client.subprocess, "run", run)
    fetched = []
    monkeypatch.setattr(
        client,
        "fetch_public",
        lambda url, content, **kwargs: fetched.append((url, content, kwargs)),
    )
    result = client.publish(**options)
    assert result == {
        "action": "deploy",
        "revision": REVISION,
        "previous": PREVIOUS,
        "asset_version": VERSION,
    }
    command, kwargs = calls[0]
    assert command[-2:] == ["iris-demo@192.0.2.1", f"deploy {REVISION}"]
    for setting in [
        "BatchMode=yes",
        "IdentitiesOnly=yes",
        "StrictHostKeyChecking=yes",
        "HostKeyAlias=iris-demo-vps",
        "GlobalKnownHostsFile=/dev/null",
        f"UserKnownHostsFile={options['known_hosts']}",
    ]:
        assert setting in command
    assert "shell" not in kwargs and kwargs["input"].startswith(b"PK")
    assert {url for url, _, _ in fetched} == {
        options["url"],
        *(options["url"] + name for name in artifact[2]),
    }
    for url, content, flags in fetched:
        name = url.removeprefix(options["url"]) or "index.html"
        assert content == artifact[2][name]
        assert flags["immutable"] == name.startswith("assets/")


def test_lost_ssh_acknowledgement_is_verified_before_deciding_to_rollback(
    client, artifact, tmp_path, monkeypatch
):
    calls = []

    def disconnected(_command, action, revision, _content=b""):
        calls.append((action, revision))
        raise client.PublishError("Connection lost after promotion")

    monkeypatch.setattr(client, "remote", disconnected)
    monkeypatch.setattr(client, "verify_public", lambda *_: None)
    result = client.publish(**arguments(tmp_path, artifact))
    assert result["revision"] == REVISION and result["previous"] is None
    assert calls == [("deploy", REVISION)]


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_failed_public_verification_rolls_back_only_the_expected_current_revision(
    client, artifact, tmp_path, monkeypatch, rollback_fails
):
    calls = []

    def receiver(_command, action, revision, _content=b""):
        calls.append((action, revision))
        if action == "rollback":
            if rollback_fails:
                raise client.PublishError("No previous release or the current revision changed")
            return {"revision": PREVIOUS}
        return {"previous": PREVIOUS, "asset_version": VERSION}

    def invalid_public(*_):
        raise client.PublishError("Public asset checksum differs")

    monkeypatch.setattr(client, "remote", receiver)
    monkeypatch.setattr(client, "verify_public", invalid_public)
    with pytest.raises(client.PublishError, match="Public asset checksum differs") as failure:
        client.publish(**arguments(tmp_path, artifact))
    assert calls == [("deploy", REVISION), ("rollback", REVISION)]
    assert (
        "Rollback was not confirmed"
        if rollback_fails
        else f"Rollback confirmed revision {PREVIOUS}"
    ) in str(failure.value)


def test_receiver_failures_do_not_echo_remote_output_or_credential_material(client, monkeypatch):
    monkeypatch.setattr(
        client.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"private receiver diagnostic"
        ),
    )
    with pytest.raises(client.PublishError, match="exit 1") as failure:
        client.remote(["ssh"], "deploy", REVISION)
    assert "private receiver diagnostic" not in str(failure.value)
    monkeypatch.setattr(
        client.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("ssh", 180)),
    )
    with pytest.raises(client.PublishError, match="did not finish"):
        client.remote(["ssh"], "deploy", REVISION)


@pytest.mark.parametrize("problem", [None, "bytes", "cache", "location", "status"])
def test_public_verification_checks_bytes_headers_and_exact_location(client, monkeypatch, problem):
    url = "https://iris.example/demo/assets/version/app.js"

    class Response(io.BytesIO):
        status = 503 if problem == "status" else 200
        headers = {
            "Cache-Control": "no-cache"
            if problem == "cache"
            else "public,max-age=31536000,immutable"
        }

        def geturl(self):
            return "https://other.example/file" if problem == "location" else url

    response = Response(b"wrong" if problem == "bytes" else b"expected")
    monkeypatch.setattr(
        client, "build_opener", lambda *_args: SimpleNamespace(open=lambda *_a, **_k: response)
    )
    if problem:
        with pytest.raises(client.PublishError):
            client.fetch_public(url, b"expected", immutable=True)
    else:
        client.fetch_public(url, b"expected", immutable=True)
    with pytest.raises(client.PublishError, match="redirected"):
        client.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example/")
