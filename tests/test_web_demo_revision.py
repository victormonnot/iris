"""Published demo revisions bind visible source links to deterministic artifacts."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "web-demo"
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "abcdef0123456789abcdef0123456789abcdef01"


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    monkeypatch.delitem(sys.modules, "serve", raising=False)
    spec = importlib.util.spec_from_file_location("iris_revision_export", EXAMPLE / "export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delitem(sys.modules, "serve")
    return module


def contents(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_source_revision_is_deterministic_and_changes_asset_identity(exporter, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = exporter.export_demo(first, revision=REVISION)
    assert manifest["source_revision"] == REVISION
    assert json.loads((first / "manifest.json").read_bytes()) == manifest
    assert f'data-iris-revision="{REVISION}"' in (first / "index.html").read_text()
    assert exporter.export_demo(second, revision=REVISION) == manifest
    assert contents(first) == contents(second)
    updated = exporter.export_demo(first, revision=OTHER_REVISION)
    assert updated["asset_version"] != manifest["asset_version"]
    assert updated["source_revision"] == OTHER_REVISION
    assert f'data-iris-revision="{OTHER_REVISION}"' in (first / "index.html").read_text()


def test_unspecified_revision_remains_optional_for_preview_and_export(exporter, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = exporter.export_demo(first)
    assert "source_revision" not in manifest
    assert b"data-iris-revision=" not in (first / "index.html").read_bytes()
    assert b"data-iris-revision=" not in exporter.demo_index()
    assert exporter.export_demo(second, revision=None) == manifest
    assert contents(first) == contents(second)


@pytest.mark.parametrize(
    "revision",
    [
        "",
        "abc1234",
        "A" * 40,
        "g" * 40,
        "a" * 39,
        "a" * 41,
        REVISION + "\n",
        '" onload="alert(1)',
        123,
        True,
    ],
)
def test_invalid_revision_is_rejected_without_writing(exporter, tmp_path, revision):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source revision"):
        exporter.export_demo(output, revision=revision)
    assert not output.exists()
    with pytest.raises(ValueError, match="source revision"):
        exporter.demo_index(revision=revision)


def test_revision_cli_records_the_supplied_commit(tmp_path):
    output = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "export.py"),
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
    assert json.loads((output / "manifest.json").read_bytes())["source_revision"] == REVISION
    before = contents(output)
    result = subprocess.run(
        [sys.executable, str(EXAMPLE / "export.py"), "--output", str(output), "--revision", "main"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "source revision" in result.stderr
    assert contents(output) == before
