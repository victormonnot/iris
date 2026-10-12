#!/usr/bin/env python3
"""Receive a bounded static artifact and atomically switch its published release."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import stat
import sys
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

FORMAT = "iris-web-demo-site-v1"
MAX_COMPRESSED = 32 * 1024 * 1024
MAX_UNCOMPRESSED = 128 * 1024 * 1024
MAX_FILES = 4096
MAX_MANIFEST = 4 * 1024 * 1024
REVISION = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")


def revision(value: str) -> str:
    if not isinstance(value, str) or not REVISION.fullmatch(value):
        raise ValueError("A full lowercase 40-character Git revision is required")
    return value


def command(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"(deploy|rollback) ([0-9a-f]{40})", value)
    if not match:
        raise ValueError("Only 'deploy <revision>' or 'rollback <revision>' is permitted")
    return match[1], match[2]


def unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest JSON key")
        result[key] = value
    return result


def safe_path(name: str) -> str:
    if not isinstance(name, str) or not name or len(name) > 512:
        raise ValueError("Invalid artifact path")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or path.as_posix() != name
        or any(not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", p) for p in path.parts)
    ):
        raise ValueError("Invalid artifact path")
    return name


def fingerprint(data: bytes) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def manifest(data: bytes, expected: str) -> dict:
    if len(data) > MAX_MANIFEST:
        raise ValueError("Artifact manifest is too large")
    value = json.loads(data, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != {
        "format",
        "source_revision",
        "asset_version",
        "files",
    }:
        raise ValueError("Invalid artifact manifest fields")
    if value["format"] != FORMAT or value["source_revision"] != revision(expected):
        raise ValueError("Artifact format or source revision does not match the deployment")
    version = value["asset_version"]
    if not isinstance(version, str) or not DIGEST.fullmatch(version):
        raise ValueError("Invalid asset version")
    records = value["files"]
    if not isinstance(records, dict) or not 2 <= len(records) < MAX_FILES:
        raise ValueError("Invalid artifact file count")
    if "index.html" not in records:
        raise ValueError("Artifact index is missing")
    prefix = f"assets/{version}/"
    size = 0
    for name, record in records.items():
        safe_path(name)
        if name != "index.html" and not name.startswith(prefix):
            raise ValueError("Artifact files must belong to its one asset version")
        if not isinstance(record, dict) or set(record) != {"sha256", "size"}:
            raise ValueError("Invalid artifact file fingerprint")
        if not isinstance(record["sha256"], str) or not DIGEST.fullmatch(record["sha256"]):
            raise ValueError("Invalid artifact file hash")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError("Invalid artifact file size")
        size += record["size"]
    if size + len(data) > MAX_UNCOMPRESSED:
        raise ValueError("Artifact exceeds the uncompressed size limit")
    return value


def read_archive(stream, expected: str) -> tuple[dict, dict[str, bytes]]:
    """Validate all archive bytes before making any deployment mutation."""
    revision(expected)
    data = stream.read(MAX_COMPRESSED + 1)
    if len(data) > MAX_COMPRESSED:
        raise ValueError("Artifact exceeds the compressed size limit")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_FILES:
            raise ValueError("Archive contains too many entries")
        paths, directories = {}, set()
        size = 0
        for entry in entries:
            directory = entry.is_dir()
            name = safe_path(entry.filename[:-1] if directory else entry.filename)
            if entry.orig_filename != entry.filename or name in paths or name in directories:
                raise ValueError("Duplicate or malformed ZIP path")
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if mode not in (0, stat.S_IFDIR if directory else stat.S_IFREG):
                raise ValueError("ZIP links and special files are forbidden")
            if entry.flag_bits & 1 or entry.compress_type not in {
                zipfile.ZIP_STORED,
                zipfile.ZIP_DEFLATED,
            }:
                raise ValueError("Unsupported encrypted or compressed ZIP entry")
            size += entry.file_size
            if size > MAX_UNCOMPRESSED:
                raise ValueError("Archive exceeds the uncompressed size limit")
            if directory:
                if entry.file_size:
                    raise ValueError("ZIP directories cannot carry data")
                directories.add(name)
            else:
                paths[name] = entry
        if "manifest.json" not in paths:
            raise ValueError("Archive manifest is missing")
        if paths["manifest.json"].file_size > MAX_MANIFEST:
            raise ValueError("Artifact manifest is too large")
        raw = archive.read(paths["manifest.json"])
        value = manifest(raw, expected)
        if set(paths) != set(value["files"]) | {"manifest.json"}:
            raise ValueError("Archive has missing or extra files")
        allowed_directories = {
            str(parent)
            for name in paths
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        }
        if not directories <= allowed_directories:
            raise ValueError("Archive has extra directories")
        contents = {"manifest.json": raw}
        for name, record in value["files"].items():
            if paths[name].file_size != record["size"]:
                raise ValueError("ZIP size does not match the manifest")
            contents[name] = archive.read(paths[name])
            if fingerprint(contents[name]) != record:
                raise ValueError("Artifact bytes do not match the manifest")
        return value, contents


def plain_path(path: Path) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Deployment storage must not contain symlink indirection")


@contextmanager
def locked(root: Path):
    root = root.absolute()
    plain_path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o755)
    fd = os.open(root / ".deploy.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("Deployment lock must be a regular file")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        root.chmod(0o755)
        for name in ("assets", "releases"):
            directory = root / name
            plain_path(directory)
            directory.mkdir(exist_ok=True, mode=0o755)
            if not directory.is_dir():
                raise ValueError("Invalid deployment storage directory")
            directory.chmod(0o755)
        yield root


def pointer(root: Path, name: str) -> str | None:
    path = root / name
    if not path.is_symlink():
        if path.exists():
            raise ValueError("Deployment pointer must be a relative symlink")
        return None
    target = path.readlink().as_posix()
    match = re.fullmatch(r"releases/([0-9a-f]{40})", target)
    if not match:
        raise ValueError("Invalid deployment pointer target")
    return match[1]


def replace_pointer(root: Path, name: str, target: str | None) -> None:
    if target is None:
        (root / name).unlink(missing_ok=True)
        return
    temporary = root / f".{name}-{uuid.uuid4().hex}"
    try:
        temporary.symlink_to(f"releases/{revision(target)}")
        os.replace(temporary, root / name)
    finally:
        temporary.unlink(missing_ok=True)


def read_regular(path: Path) -> bytes:
    plain_path(path)
    if not path.is_file() or path.stat().st_size > MAX_UNCOMPRESSED:
        raise ValueError("Invalid stored deployment file")
    return path.read_bytes()


def stored_release(root: Path, expected: str) -> dict:
    directory = root / "releases" / revision(expected)
    value = manifest(read_regular(directory / "manifest.json"), expected)
    for name, record in value["files"].items():
        path = directory / name if name == "index.html" else root / name
        if fingerprint(read_regular(path)) != record:
            raise ValueError("Stored release failed its integrity check")
    return value


def check_tree(root: Path, expected: dict[str, bytes]) -> None:
    plain_path(root)
    if not root.is_dir():
        raise ValueError("Stored immutable directory is invalid")
    entries = list(root.rglob("*"))
    if any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in entries):
        raise ValueError("Stored immutable directory contains a link or special file")
    files = {p.relative_to(root).as_posix(): p for p in entries if p.is_file()}
    directories = {p.relative_to(root).as_posix() for p in entries if p.is_dir()}
    expected_directories = {
        str(parent)
        for name in expected
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    if set(files) != set(expected) or directories != expected_directories:
        raise ValueError("Stored immutable directory has different contents")
    if any(read_regular(files[name]) != content for name, content in expected.items()):
        raise ValueError("An immutable asset or revision cannot be overwritten")


def write_tree(root: Path, contents: dict[str, bytes]) -> None:
    root.mkdir()
    for name, data in contents.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        target.chmod(0o644)
    for directory in (root, *(p for p in root.rglob("*") if p.is_dir())):
        directory.chmod(0o755)
        sync_directory(directory)


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def activate(root: Path, target: str, current: str | None) -> None:
    previous = pointer(root, "previous")
    try:
        replace_pointer(root, "previous", current)
        replace_pointer(root, "current", target)
    except OSError:
        replace_pointer(root, "previous", previous)
        raise
    sync_directory(root)


def deploy(root: Path, expected: str, stream) -> dict:
    value, contents = read_archive(stream, expected)
    version = value["asset_version"]
    prefix = f"assets/{version}/"
    assets = {
        name.removeprefix(prefix): data
        for name, data in contents.items()
        if name.startswith(prefix)
    }
    release = {name: contents[name] for name in ("index.html", "manifest.json")}
    with locked(root) as root:
        current = pointer(root, "current")
        pointer(root, "previous")  # Reject a malformed pointer before writing anything public.
        if current:
            stored_release(root, current)
        destination = root / "releases" / expected
        asset_destination = root / "assets" / version
        for path, files in ((destination, release), (asset_destination, assets)):
            plain_path(path)
            if path.exists():
                check_tree(path, files)
        with tempfile.TemporaryDirectory(prefix=".incoming-", dir=root) as temporary:
            staging = Path(temporary)
            if not asset_destination.exists():
                write_tree(staging / "assets", assets)
                (staging / "assets").rename(asset_destination)
                sync_directory(root / "assets")
            if not destination.exists():
                write_tree(staging / "release", release)
                (staging / "release").rename(destination)
                sync_directory(root / "releases")
        if current != expected:
            activate(root, expected, current)
        return {
            "action": "deploy",
            "revision": expected,
            "previous": pointer(root, "previous"),
            "asset_version": version,
        }


def rollback(root: Path, expected: str) -> dict:
    revision(expected)
    with locked(root) as root:
        if pointer(root, "current") != expected:
            raise ValueError("Current revision changed; rollback refused")
        previous = pointer(root, "previous")
        if not previous or previous == expected:
            raise ValueError("There is no different previous release to restore")
        value = stored_release(root, previous)
        activate(root, previous, expected)
        return {
            "action": "rollback",
            "revision": previous,
            "previous": expected,
            "asset_version": value["asset_version"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ssh", action="store_true")
    parser.add_argument("action", nargs="?", choices=("deploy", "rollback"))
    parser.add_argument("revision", nargs="?")
    args = parser.parse_args()
    try:
        if args.ssh:
            if args.action or args.revision:
                raise ValueError("SSH mode does not accept a local action")
            action, expected = command(os.environ.get("SSH_ORIGINAL_COMMAND", ""))
        else:
            action, expected = args.action, revision(args.revision)
            if not action:
                raise ValueError("A deployment action is required")
        result = (
            deploy(args.root, expected, sys.stdin.buffer)
            if action == "deploy"
            else rollback(args.root, expected)
        )
        print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
        parser.exit(1, f"Deployment refused: {error}\n")


if __name__ == "__main__":
    main()
