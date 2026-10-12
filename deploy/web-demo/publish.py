#!/usr/bin/env python3
"""Publish a verified static Iris artifact through the restricted SSH receiver."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import stat
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

FORMAT = "iris-web-demo-site-v1"
MAX_ARCHIVE = 32 * 1024 * 1024
MAX_CONTENT = 128 * 1024 * 1024
MAX_MANIFEST = 4 * 1024 * 1024
MAX_FILES = 4096


class PublishError(ValueError):
    """A bounded publication or verification failure, safe to print in CI."""


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def require_hash(value: str, length: int, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise PublishError(
            f"{label} must contain exactly {length} lowercase hexadecimal characters"
        )


def unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PublishError("The manifest contains duplicate keys")
        result[key] = value
    return result


@dataclass(frozen=True)
class Artifact:
    manifest: dict
    files: dict[str, bytes]
    archive: bytes


def prepare_artifact(directory: Path, revision: str, manifest_sha256: str) -> Artifact:
    """Snapshot only the exact files authenticated by the build job's manifest hash."""
    require_hash(revision, 40, "Source revision")
    require_hash(manifest_sha256, 64, "Manifest SHA-256")
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise PublishError("Artifact paths must not use symlinks")
    entries = list(directory.rglob("*"))
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries):
        raise PublishError("Artifacts may contain only ordinary files and directories")
    if len(entries) > MAX_FILES * 2:
        raise PublishError("The artifact has too many entries")
    manifest_path = directory / "manifest.json"
    if manifest_path.stat().st_size > MAX_MANIFEST:
        raise PublishError("The manifest exceeds the publication size limit")
    manifest_bytes = manifest_path.read_bytes()
    if digest(manifest_bytes) != manifest_sha256:
        raise PublishError("The manifest differs from the verified build job")
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique_object)
    if not isinstance(manifest, dict) or set(manifest) != {
        "format",
        "source_revision",
        "asset_version",
        "files",
    }:
        raise PublishError("The publication manifest has an unsupported shape")
    if manifest["format"] != FORMAT or manifest["source_revision"] != revision:
        raise PublishError("The artifact does not identify the requested source revision")
    require_hash(manifest["asset_version"], 64, "Asset version")
    records = manifest["files"]
    if (
        not isinstance(records, dict)
        or "index.html" not in records
        or not 2 <= len(records) < MAX_FILES
    ):
        raise PublishError("The artifact must contain an index and a bounded file list")
    prefix = f"assets/{manifest['asset_version']}/"
    files = {"manifest.json": manifest_bytes}
    total = len(manifest_bytes)
    for name, expected in records.items():
        path = PurePosixPath(name)
        if (
            len(name) > 512
            or path.is_absolute()
            or path.as_posix() != name
            or any(not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", part) for part in path.parts)
            or (name != "index.html" and not name.startswith(prefix))
            or not isinstance(expected, dict)
            or set(expected) != {"sha256", "size"}
        ):
            raise PublishError("The manifest contains an unsupported file entry")
        require_hash(expected["sha256"], 64, "File SHA-256")
        size = expected["size"]
        if type(size) is not int or size < 0 or total + size > MAX_CONTENT:
            raise PublishError("Artifact files exceed the publication size limit")
        target = directory / name
        if target.stat().st_size != size:
            raise PublishError(f"Artifact file size differs: {name}")
        content = target.read_bytes()
        if len(content) != size or digest(content) != expected["sha256"]:
            raise PublishError(f"Artifact file checksum differs: {name}")
        files[name] = content
        total += size
    actual = {path.relative_to(directory).as_posix() for path in entries if path.is_file()}
    expected_dirs = {
        str(parent)
        for name in files
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    actual_dirs = {path.relative_to(directory).as_posix() for path in entries if path.is_dir()}
    if actual != set(files) or actual_dirs != expected_dirs:
        raise PublishError("The artifact contains missing or unlisted files or directories")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            item = zipfile.ZipInfo(name)
            item.create_system = 3
            item.external_attr = (stat.S_IFREG | 0o644) << 16
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, content)
    compressed = buffer.getvalue()
    if len(compressed) > MAX_ARCHIVE:
        raise PublishError("The compressed artifact exceeds the publication size limit")
    return Artifact(manifest, files, compressed)


def public_base(url: str) -> str:
    address = urlsplit(url)
    if (
        address.scheme != "https"
        or not address.hostname
        or address.username is not None
        or address.password is not None
        or address.query
        or address.fragment
        or any(part in {".", ".."} for part in address.path.split("/"))
        or any(char in url for char in ("\\", "%", "\r", "\n"))
        or any(ord(char) <= 32 for char in url)
        or "//" in address.path
    ):
        raise PublishError(
            "The public URL must be an HTTPS demo location without credentials or query"
        )
    return f"https://{address.netloc}{address.path.rstrip('/')}/"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, _request, _response, _code, _message, _headers, _url):
        raise PublishError("The public site unexpectedly redirected a verification request")


def fetch_public(url: str, expected: bytes, *, immutable: bool) -> None:
    request = Request(url, headers={"Cache-Control": "no-cache", "Accept-Encoding": "identity"})
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=20) as response:
            if response.status != 200 or response.geturl() != url:
                raise PublishError("The public site returned an unexpected response or location")
            directives = {
                part.strip().lower()
                for part in response.headers.get("Cache-Control", "").split(",")
            }
            if immutable:
                if "immutable" not in directives or not any(
                    re.fullmatch(r"max-age=[1-9][0-9]*", part) for part in directives
                ):
                    raise PublishError("Versioned public assets lack immutable cache headers")
            elif not directives.intersection({"no-cache", "no-store"}):
                raise PublishError("The public index or manifest lacks revalidation cache headers")
            if response.read(len(expected) + 1) != expected:
                raise PublishError(f"Public file bytes differ: {urlsplit(url).path}")
    except (OSError, URLError, HTTPException) as error:
        raise PublishError(f"Cannot verify public file: {urlsplit(url).path}") from error


def verify_public(base: str, artifact: Artifact) -> None:
    """Check the landing page, every published byte, and the revision again at the end."""
    fetch_public(base + "manifest.json", artifact.files["manifest.json"], immutable=False)
    fetch_public(base, artifact.files["index.html"], immutable=False)

    def verify(name: str) -> None:
        fetch_public(base + name, artifact.files[name], immutable=name.startswith("assets/"))

    with ThreadPoolExecutor(max_workers=6) as workers:
        list(workers.map(verify, artifact.manifest["files"]))
    fetch_public(base + "manifest.json", artifact.files["manifest.json"], immutable=False)


def ssh_command(host: str, user: str, identity: Path, known_hosts: Path, port: int) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", host) or not re.fullmatch(
        r"[a-z_][a-z0-9_-]*", user
    ):
        raise PublishError("Invalid SSH host or deployment user")
    if not 1 <= port <= 65535 or not identity.is_file() or not known_hosts.is_file():
        raise PublishError("SSH requires a valid port, identity file and pinned known-hosts file")
    return [
        "ssh",
        "-T",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "HostKeyAlias=iris-demo-vps",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        f"UserKnownHostsFile={known_hosts.resolve()}",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        "-i",
        str(identity.resolve()),
        "-p",
        str(port),
        f"{user}@{host}",
    ]


def remote(command: list[str], action: str, revision: str, content: bytes = b"") -> dict:
    try:
        result = subprocess.run(
            [*command, f"{action} {revision}"], input=content, capture_output=True, timeout=180
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PublishError(
            "The deployment receiver could not be reached or did not finish"
        ) from error
    if result.returncode:
        raise PublishError(
            f"The deployment receiver did not confirm {action} (exit {result.returncode})"
        )
    try:
        if len(result.stdout) > 65536:
            raise ValueError("Response too large")
        response = json.loads(result.stdout)
        if not isinstance(response, dict) or response.get("action") != action:
            raise ValueError("Unexpected receiver action")
        require_hash(response.get("revision"), 40, "Receiver revision")
        require_hash(response.get("asset_version"), 64, "Receiver asset version")
        if response.get("previous") is not None:
            require_hash(response["previous"], 40, "Previous revision")
        if action == "deploy" and response["revision"] != revision:
            raise ValueError("Unexpected deployed revision")
    except (ValueError, TypeError, KeyError) as error:
        raise PublishError("The deployment receiver returned an invalid confirmation") from error
    return response


def publish(
    artifact: Path,
    *,
    revision: str,
    manifest_sha256: str,
    url: str,
    host: str,
    identity: Path,
    known_hosts: Path,
    user: str = "iris-demo",
    port: int = 22,
) -> dict:
    prepared = prepare_artifact(artifact, revision, manifest_sha256)
    base = public_base(url)
    command = ssh_command(host, user, identity, known_hosts, port)
    confirmation = None
    connection_error = None
    try:
        confirmation = remote(command, "deploy", revision, prepared.archive)
        if confirmation["asset_version"] != prepared.manifest["asset_version"]:
            raise PublishError("The receiver confirmed a different asset version")
    except PublishError as error:
        # A lost acknowledgement does not imply the atomic promotion failed.
        confirmation = None
        connection_error = str(error)
    try:
        verify_public(base, prepared)
    except PublishError as error:
        try:
            restored = remote(command, "rollback", revision)
            recovery = f"Rollback confirmed revision {restored['revision']}."
        except PublishError as rollback_error:
            recovery = f"Rollback was not confirmed: {rollback_error}."
        initial = f" {connection_error}." if connection_error else ""
        raise PublishError(f"Public verification failed: {error}.{initial} {recovery}") from error
    return {
        "action": "deploy",
        "revision": revision,
        "previous": confirmation.get("previous") if confirmation else None,
        "asset_version": prepared.manifest["asset_version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="iris-demo")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--port", type=int, default=22)
    args = parser.parse_args()
    try:
        result = publish(**vars(args))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Publication failed: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
