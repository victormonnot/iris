#!/usr/bin/env python3
"""Export Iris's shared interface and public examples as a standalone static site.

Uses only Python's standard library. Does not start Iris, execute models, access
the network or read a user workspace. The local application remains buildless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

from serve import DEMO_URL, EXAMPLE, SEED, SOCIAL_IMAGE, STATIC, demo_index, validate_revision

ROOT = EXAMPLE.parents[1]
FORMAT = "iris-web-demo-site-v1"
FONT_NOTICES = (
    "IBM-Plex-Sans-OFL.txt",
    "IBM-Plex-Mono-OFL.txt",
    "Marcellus-OFL.txt",
    "README.md",
)
CSS_URL = re.compile(r"url\(\s*(['\"]?)([^)'\"\s]+)\1\s*\)")


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def fingerprint(content: bytes) -> dict:
    return {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}


def public_file(root: Path, relative: str) -> bytes:
    """Read a named public input, refusing traversal and symlink indirection."""
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or path.as_posix() != relative
        or ".." in path.parts
        or "\\" in relative
        or "\x00" in relative
    ):
        raise ValueError(f"Invalid public asset path: {relative!r}")
    target = root / relative
    if any(item.is_symlink() for item in (target, *target.parents)):
        raise ValueError(f"Symlinks are not public build inputs: {relative}")
    return target.read_bytes()


class BrowserAssets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list) -> None:
        values = dict(attrs)
        url = (
            values.get("src") if tag == "script" else values.get("href") if tag == "link" else None
        )
        if url:
            if not url.startswith("/static/"):
                raise ValueError(f"Demo browser dependencies must be local: {url}")
            self.paths.add(url.removeprefix("/static/"))


def collect_assets() -> dict[str, bytes]:
    # The composer also reads these templates; apply the same input boundary as
    # for the dependencies that its HTML names.
    public_file(STATIC, "index.html")
    public_file(STATIC, "demo/entry.html")
    parser = BrowserAssets()
    parser.feed(demo_index().decode())
    assets: dict[str, bytes] = {}

    def collect(relative: str) -> None:
        key = f"static/{relative}"
        if key in assets:
            return
        if PurePosixPath(relative).suffix not in {".css", ".js", ".svg", ".woff2"}:
            raise ValueError(f"Unsupported browser dependency: {relative}")
        content = public_file(STATIC, relative)
        assets[key] = content
        if relative.endswith(".css"):
            source = content.decode()
            if "@import" in source:
                raise ValueError(f"Use explicit stylesheet links in the shared HTML: {relative}")

            def resolve_url(match: re.Match) -> str:
                url = match[2]
                if url.startswith("/static/"):
                    dependency = url.removeprefix("/static/")
                else:
                    if url.startswith("/") or any(char in url for char in (":", "?", "#")):
                        raise ValueError(f"Unsupported CSS asset URL in {relative}: {url}")
                    dependency = posixpath.normpath(
                        posixpath.join(posixpath.dirname(relative), url)
                    )
                collect(dependency)
                local = posixpath.relpath(dependency, posixpath.dirname(relative) or ".")
                return f'url("{local}")'

            assets[key] = CSS_URL.sub(resolve_url, source).encode()

    for relative in sorted(parser.paths):
        collect(relative)
    for name in FONT_NOTICES:
        assets[f"static/fonts/{name}"] = public_file(STATIC, f"fonts/{name}")

    provenance_bytes = public_file(SEED, "provenance.json")
    provenance = json.loads(provenance_bytes)
    if provenance.get("format") != "iris-web-demo-provenance-v1":
        raise ValueError("Unsupported public example provenance format")
    files = provenance["files"]
    if not {"fixture.json", "CREDITS.txt"} <= files.keys():
        raise ValueError("The public examples require their fixture and credits")
    for relative, expected in files.items():
        if relative not in {"fixture.json", "CREDITS.txt"} and not re.fullmatch(
            r"media/[A-Za-z0-9_-]+\.(jpg|png)", relative
        ):
            raise ValueError(f"Not a public example asset: {relative}")
        content = public_file(SEED, relative)
        if fingerprint(content) != {"sha256": expected["sha256"], "size": expected["size_bytes"]}:
            raise ValueError(f"Public example checksum mismatch: {relative}")
        assets[f"seed/{relative}"] = content
    fixture = json.loads(assets["seed/fixture.json"])
    if fixture.get("format") != "iris-web-demo-v1":
        raise ValueError("Unsupported public example fixture format")
    for relative in fixture["resources"].values():
        if f"seed/{relative}" not in assets or not relative.startswith("media/"):
            raise ValueError(f"Public example media missing: {relative}")
    assets["seed/provenance.json"] = provenance_bytes
    assets["LICENSE"] = public_file(ROOT, "LICENSE")
    assets["THIRD_PARTY_NOTICES.md"] = public_file(EXAMPLE, "THIRD_PARTY_NOTICES.md")
    assets[SOCIAL_IMAGE] = public_file(EXAMPLE, SOCIAL_IMAGE)
    return assets


def validate_output(output: Path) -> Path:
    if any(item.is_symlink() for item in (output, *output.parents)):
        raise ValueError("The output path must not use symlinks")
    output = output.resolve()
    if ROOT.is_relative_to(output) or (
        output.is_relative_to(ROOT)
        and (not output.is_relative_to(ROOT / "dist") or output == ROOT / "dist")
    ):
        raise ValueError("Build inside a subdirectory of dist/ or outside the source repository")
    if not output.exists():
        return output
    if not output.is_dir():
        raise ValueError("The output must be a directory")
    entries = list(output.rglob("*"))
    if not entries:
        return output
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("Refusing to replace an output containing symlinks")
    try:
        manifest = json.loads((output / "manifest.json").read_bytes())
        files = manifest["files"]
        if manifest["format"] != FORMAT or "index.html" not in files:
            raise ValueError("Not a generated Iris demo")
        actual_files = {
            entry.relative_to(output).as_posix() for entry in entries if entry.is_file()
        }
        expected_dirs = {
            str(parent)
            for name in files
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        }
        actual_dirs = {entry.relative_to(output).as_posix() for entry in entries if entry.is_dir()}
        if actual_files != set(files) | {"manifest.json"} or actual_dirs != expected_dirs:
            raise ValueError("The output contains unrecognized files or directories")
        for name, expected in files.items():
            if fingerprint(public_file(output, name)) != expected:
                raise ValueError(f"Generated file was modified: {name}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Refusing to replace a directory that is not an intact Iris demo: {output}"
        ) from error
    return output


def export_demo(output: Path, *, revision: str | None = None) -> dict:
    """Create a deterministic artifact; replace only an intact previous export."""
    validate_revision(revision)
    output = validate_output(output)
    assets = collect_assets()
    # Hash the entry template too: an HTML-only change produces a new release.
    inputs = {name: fingerprint(content) for name, content in sorted(assets.items())}
    inputs["index.html"] = fingerprint(demo_index(revision=revision))
    version = hashlib.sha256(json_bytes(inputs)).hexdigest()
    prefix = f"assets/{version}/"
    files = {prefix + name: content for name, content in assets.items()}
    files["index.html"] = demo_index(
        static_prefix=f"./{prefix}static/",
        seed_prefix=f"./{prefix}seed/",
        revision=revision,
        social_image_url=f"{DEMO_URL}{prefix}{SOCIAL_IMAGE}",
    )
    manifest = {
        "format": FORMAT,
        "asset_version": version,
        "files": {name: fingerprint(content) for name, content in sorted(files.items())},
    }
    if revision is not None:
        manifest["source_revision"] = revision
    files["manifest.json"] = json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-build-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "site"
        staging.mkdir()
        for name, content in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        # The old artifact stays intact if collection or staging fails. This is
        # a local build operation; publication/retaining deployed assets is separate.
        previous = Path(temporary) / "previous"
        if output.exists():
            output.rename(previous)
        try:
            staging.rename(output)
        except OSError:
            if previous.exists():
                previous.rename(output)
            raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "demo")
    parser.add_argument(
        "--revision", help="Full 40-character lowercase Git commit SHA of this build"
    )
    args = parser.parse_args()
    try:
        manifest = export_demo(args.output, revision=args.revision)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot export the demo: {error}\n")
    size = sum(record["size"] for record in manifest["files"].values())
    print(f"Static Iris demo: {args.output.resolve()}")
    print(f"{len(manifest['files'])} files, {size:,} bytes; assets {manifest['asset_version']}")


if __name__ == "__main__":
    main()
