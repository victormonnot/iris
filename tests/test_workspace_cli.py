"""The recovery commands work without starting a server or loading model runtimes."""

import fcntl
import json
import sys

import pytest

from iris import cli
from iris.store import Store


def invoke(monkeypatch, capsys, *arguments):
    monkeypatch.setattr(sys, "argv", ["iris", *map(str, arguments)])
    cli.main()
    return json.loads(capsys.readouterr().out)


def test_backup_verify_restore_and_existing_target(monkeypatch, capsys, tmp_path):
    store = Store(tmp_path / "workspace")
    archive = tmp_path / "backup.zip"
    restored = tmp_path / "restored"
    backup = invoke(monkeypatch, capsys, "workspace", "backup", archive, "--data-dir", store.root)
    assert archive.is_file()
    assert backup["summary"]["schema_version"] == 12
    inspection = invoke(monkeypatch, capsys, "workspace", "inspect", archive)
    assert inspection["archive_sha256"] == backup["archive_sha256"]
    restoration = invoke(monkeypatch, capsys, "workspace", "restore", archive, "--to", restored)
    assert restoration["path"] == str(restored)
    assert (restored / "iris.sqlite3").is_file()
    monkeypatch.setattr(
        sys, "argv", ["iris", "workspace", "restore", str(archive), "--to", str(restored)]
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert (restored / "iris.sqlite3").is_file()


def test_cli_backup_refuses_workspace_open_in_server(monkeypatch, capsys, tmp_path):
    store = Store(tmp_path / "workspace")
    with (store.root / ".server.lock").open("a") as server_lock:
        fcntl.flock(server_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "iris",
                "--data-dir",
                str(store.root),
                "workspace",
                "backup",
                str(tmp_path / "blocked.zip"),
            ],
        )
        with pytest.raises(SystemExit) as raised:
            cli.main()
        assert raised.value.code == 1
        assert "Use Workspace backup in the app" in capsys.readouterr().err
        assert not (tmp_path / "blocked.zip").exists()


def test_cli_missing_source_does_not_initialize_workspace(monkeypatch, capsys, tmp_path):
    source = tmp_path / "missing"
    monkeypatch.setattr(
        sys,
        "argv",
        ["iris", "--data-dir", str(source), "workspace", "backup", str(tmp_path / "missing.zip")],
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 1
    assert "not an initialized" in capsys.readouterr().err
    assert not source.exists()
