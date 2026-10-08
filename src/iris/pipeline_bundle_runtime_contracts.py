"""Version-two runtime inventory and directory integrity, using only stdlib.

Validation never imports packaged implementation code or deserializes weights.
Historical runtime declarations remain separate from the shipped implementation.
"""

import ctypes
import errno
import hashlib
import os
import shutil
import stat
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path

FORMAT_V2 = "iris-pipeline-bundle-v2"
PACKAGE_VERSIONS = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "pillow": "12.3.0",
    "numpy": "2.5.3",
    "opencv-python-headless": "4.14.0.94",
    "scipy": "1.17.1",
    "lap": "0.5.12",
    "cython_bbox": "0.1.5",
}
VERSION_POLICY = {
    name: "base_version" if name in {"torch", "torchvision"} else "exact"
    for name in PACKAGE_VERSIONS
}
PYTHON_VERSIONS = ["3.12", "3.13"]
OPERATING_SYSTEMS = ["Linux"]
TRACKER_FILES = {
    "bytetrack": {
        "__init__.py",
        "basetrack.py",
        "byte_tracker.py",
        "kalman_filter.py",
        "matching.py",
        "LICENSE",
        "adaptations.patch",
    },
    "botsort": {
        "__init__.py",
        "basetrack.py",
        "bot_sort.py",
        "gmc.py",
        "kalman_filter.py",
        "matching.py",
        "LICENSE",
        "adaptations.patch",
    },
}
YOLOX_FILES = {
    "__init__.py",
    "darknet.py",
    "losses.py",
    "network_blocks.py",
    "utils.py",
    "yolo_head.py",
    "yolo_pafpn.py",
    "yolox.py",
    "LICENSE",
    "NOTICE",
}
RUNTIME_MODULES = {
    "pipeline_bundle_runtime_contracts.py",
    "pipeline_runtime.py",
    "pipeline_selection.py",
    "pipeline_detector_runtime.py",
    "pipeline_runner.py",
    "tracking.py",
}
RUNTIME_PATHS = {
    "run.py",
    "example.py",
    "requirements.txt",
    "provenance/tracking-original.py.txt",
    "iris_bundle/_vendor/__init__.py",
    "iris_bundle/_vendor/tracking-provenance.json",
    *{"iris_bundle/" + name for name in RUNTIME_MODULES},
    *{
        f"iris_bundle/_vendor/{algorithm}/{name}"
        for algorithm, names in TRACKER_FILES.items()
        for name in names
    },
}


def runtime_required_paths(architecture):
    from .pipeline_bundle_contracts import required_paths

    return (
        (required_paths(architecture) - {"inspect.py"})
        | {"inspect_bundle.py"}
        | RUNTIME_PATHS
        | (
            {"iris_bundle/_vendor/yolox/" + name for name in YOLOX_FILES}
            if architecture == "yolox_nano"
            else set()
        )
    )


def portable_tracking_bytes(original):
    if not isinstance(original, bytes):
        raise ValueError("Tracker source must be UTF-8 bytes")
    try:
        return original.decode("utf-8").replace("from iris.", "from .").encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Tracker source must be UTF-8") from exc


def requirements_bytes():
    return (
        "\n".join(f"{name}=={version}" for name, version in PACKAGE_VERSIONS.items()) + "\n"
    ).encode("utf-8")


def runtime_descriptor(files):
    return {
        "schema": "iris-pipeline-runtime-v1",
        "python_versions": list(PYTHON_VERSIONS),
        "operating_systems": list(OPERATING_SYSTEMS),
        "package_versions": dict(PACKAGE_VERSIONS),
        "version_policy": dict(VERSION_POLICY),
        "tracker_port": {
            "source_path": "provenance/tracking-original.py.txt",
            "source_sha256": files["provenance/tracking-original.py.txt"]["sha256"],
            "portable_path": "iris_bundle/tracking.py",
            "portable_sha256": files["iris_bundle/tracking.py"]["sha256"],
            "transformation": "iris_imports_to_relative_v1",
        },
        "vendor_provenance": {
            "path": "iris_bundle/_vendor/tracking-provenance.json",
            "sha256": files["iris_bundle/_vendor/tracking-provenance.json"]["sha256"],
        },
    }


def validate_v2_manifest(manifest):
    from .pipeline_bundle_contracts import (
        FORMAT,
        MAX_CHECKPOINT_BYTES,
        MAX_JSON_BYTES,
        _hash,
        _integer,
        _object,
        _relative_path,
        _validate_v1_manifest,
        canonical,
        required_paths,
    )

    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_V2:
        raise ValueError("Runtime requires a version-two pipeline bundle")
    base = {key: value for key, value in manifest.items() if key != "deployment_runtime"}
    files = _object(
        manifest.get("files"),
        runtime_required_paths(manifest["detector"]["config"]["architecture"]),
        "Runtime file inventory",
    )
    base["files"] = {
        key: files["inspect_bundle.py" if key == "inspect.py" else key]
        for key in required_paths(manifest["detector"]["config"]["architecture"])
    }
    base["format"] = FORMAT
    _validate_v1_manifest(base)
    for name, identity in files.items():
        _relative_path(name)
        _object(identity, {"sha256", "size"}, "Runtime file identity")
        _hash(identity["sha256"], "Runtime member hash")
        _integer(
            identity["size"],
            "Runtime member size",
            0 if name.endswith("/__init__.py") else 1,
            MAX_CHECKPOINT_BYTES if name == "detector/model.pth" else MAX_JSON_BYTES,
        )
    if canonical(manifest.get("deployment_runtime")) != canonical(runtime_descriptor(files)):
        raise ValueError("Runtime implementation or dependency contract differs from v2")
    if (
        files["iris_bundle/_vendor/tracking-provenance.json"]["sha256"]
        != manifest["tracker"]["runtime"]["provenance"]["manifest_sha256"]
    ):
        raise ValueError(
            "Packaged tracker provenance differs from the frozen native implementation"
        )
    adapters = manifest["tracker"]["runtime"]["provenance"]["adapter_sha256"]
    if (
        files["provenance/tracking-original.py.txt"]["sha256"] != adapters["tracking.py"]
        or files["iris_bundle/tracking_contracts.py"]["sha256"] != adapters["tracking_contracts.py"]
    ):
        raise ValueError("Packaged tracker adapters differ from the frozen source implementation")
    return deepcopy(manifest)


def validate_runtime_payloads(manifest, read):
    """Validate source-to-port and native vendor provenance after member hashes."""
    from .pipeline_bundle_contracts import _provenance, canonical, read_json

    if manifest["format"] != FORMAT_V2:
        return
    if portable_tracking_bytes(read("provenance/tracking-original.py.txt")) != read(
        "iris_bundle/tracking.py"
    ):
        raise ValueError("Portable tracker differs from its declared import-only transformation")
    if read("requirements.txt") != requirements_bytes():
        raise ValueError("Requirements differ from the declared deployment dependency versions")
    provenance = read_json(read("iris_bundle/_vendor/tracking-provenance.json"))
    if not isinstance(provenance, dict) or set(provenance) != set(TRACKER_FILES):
        raise ValueError("Native tracker provenance must describe both bundled implementations")
    for algorithm, names in TRACKER_FILES.items():
        record = provenance[algorithm]
        checked = {
            **record,
            "manifest_sha256": manifest["files"]["iris_bundle/_vendor/tracking-provenance.json"][
                "sha256"
            ],
            "adapter_sha256": {
                "tracking.py": manifest["files"]["iris_bundle/tracking.py"]["sha256"],
                "tracking_contracts.py": manifest["files"]["iris_bundle/tracking_contracts.py"][
                    "sha256"
                ],
            },
        }
        _provenance(checked, algorithm)
        for name in names:
            expected = (
                record["license_sha256"]
                if name == "LICENSE"
                else record["adaptations_sha256"]
                if name == "adaptations.patch"
                else record["files"][name]["vendored_sha256"]
            )
            if manifest["files"][f"iris_bundle/_vendor/{algorithm}/{name}"]["sha256"] != expected:
                raise ValueError("Native tracker file differs from its pinned vendor provenance")
        if algorithm == manifest["tracker"]["profile"]["algorithm"]:
            source = manifest["tracker"]["runtime"]["provenance"]
            if any(canonical(source[key]) != canonical(value) for key, value in record.items()):
                raise ValueError(
                    "Frozen tracker provenance and packaged native implementation differ"
                )


def _safe_root(directory):
    path = Path(directory).absolute()
    if (
        path.is_symlink()
        or not path.is_dir()
        or any(parent.is_symlink() for parent in path.parents)
    ):
        raise ValueError("Bundle directory must be a regular directory without symbolic links")
    return path


def _signature(details):
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns


def _regular_bytes(path, maximum, *, checkpoint=None):
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_size > maximum:
            raise ValueError("Runtime member is unsafe or oversized")
        result, count, blocks = hashlib.sha256(), 0, []
        while block := stream.read(1024**2):
            if checkpoint:
                checkpoint()
            count += len(block)
            if count > maximum:
                raise ValueError("Runtime member grew beyond its declared limit")
            result.update(block)
            if maximum <= 2 * 1024**2:
                blocks.append(block)
        if _signature(os.fstat(stream.fileno())) != _signature(details):
            raise ValueError("Runtime member changed during validation")
        return {"size": count, "sha256": result.hexdigest()}, b"".join(blocks)


def validate_directory(directory, expected_manifest=None, checkpoint=None):
    from .pipeline_bundle_contracts import (
        MAX_JSON_BYTES,
        canonical,
        digest,
        read_json,
        validate_manifest,
    )

    root = _safe_root(directory)
    initial_identity, raw = _regular_bytes(
        root / "manifest.json", MAX_JSON_BYTES, checkpoint=checkpoint
    )
    manifest = validate_manifest(read_json(raw))
    if expected_manifest is not None and canonical(manifest) != canonical(expected_manifest):
        raise ValueError("Runtime directory manifest differs from the expected package")
    expected = {"manifest.json", *manifest["files"]}
    found = set()
    count = 0
    for folder, directories, filenames in os.walk(root, followlinks=False):
        for name in [*directories, *filenames]:
            count += 1
            if count > 1024:
                raise ValueError("Runtime directory inventory exceeds its bounded size")
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            details = path.lstat()
            if stat.S_ISLNK(details.st_mode):
                raise ValueError("Runtime directory cannot contain symbolic links")
            if name in directories:
                if not stat.S_ISDIR(details.st_mode):
                    raise ValueError("Runtime path is not a directory")
                if name == "__pycache__":
                    raise ValueError(
                        "Runtime bytecode caches are forbidden; use python -B before import"
                    )
                if not any(member.startswith(relative + "/") for member in expected):
                    raise ValueError("Unexpected directory inside pipeline runtime")
                continue
            if not stat.S_ISREG(details.st_mode):
                raise ValueError("Runtime directory must contain only regular files")
            if relative not in expected:
                raise ValueError("Runtime directory contains files outside its frozen inventory")
            found.add(relative)
    if found != expected:
        raise ValueError("Runtime directory is missing required bundle members")
    contents = {}
    for name, identity in manifest["files"].items():
        if checkpoint:
            checkpoint()
        measured, raw = _regular_bytes(root / name, identity["size"], checkpoint=checkpoint)
        if measured != identity:
            raise ValueError("Runtime directory member differs from its frozen hash or size")
        if name != "detector/model.pth":
            contents[name] = raw
    validate_runtime_payloads(manifest, contents.__getitem__)
    final_identity, _ = _regular_bytes(
        root / "manifest.json", MAX_JSON_BYTES, checkpoint=checkpoint
    )
    if final_identity != initial_identity:
        raise ValueError("Runtime manifest changed during directory validation")
    return {
        "manifest": manifest,
        "manifest_sha256": digest(manifest),
        "files_verified": len(expected),
    }


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_new(source, destination):
    # The supported runtime matrix is Linux; renameat2 avoids an overwrite race
    # between checking a destination and atomically publishing a whole directory.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise ValueError("Atomic new-directory extraction requires Linux renameat2")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise ValueError("Extraction destination already exists")
        raise OSError(code, os.strerror(code))


def extract_bundle(archive_path, destination, checkpoint=None):
    from .pipeline_bundle_contracts import inspect_bundle

    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Extract into a new directory; the destination already exists")
    parent = _safe_root(destination.parent)
    inspected = inspect_bundle(archive_path, checkpoint=checkpoint)
    staging = Path(tempfile.mkdtemp(prefix=".iris-pipeline-extract-", dir=parent))
    published = False
    try:
        with zipfile.ZipFile(archive_path, "r", allowZip64=False) as archive:
            expected = {"manifest.json", *inspected["manifest"]["files"]}
            for name in sorted(expected):
                if checkpoint:
                    checkpoint()
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                maximum = (
                    2 * 1024**2
                    if name == "manifest.json"
                    else inspected["manifest"]["files"][name]["size"]
                )
                count = 0
                with archive.open(name) as source, path.open("xb") as output:
                    os.chmod(path, 0o600)
                    while block := source.read(1024**2):
                        if checkpoint:
                            checkpoint()
                        count += len(block)
                        if count > maximum:
                            raise ValueError("Extraction member exceeds its frozen size")
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
        result = validate_directory(
            staging, expected_manifest=inspected["manifest"], checkpoint=checkpoint
        )
        if (
            inspect_bundle(archive_path, checkpoint=checkpoint)["archive_sha256"]
            != inspected["archive_sha256"]
        ):
            raise ValueError("Pipeline archive changed during extraction")
        for folder, _, _ in os.walk(staging, topdown=False):
            _sync_directory(folder)
        if checkpoint:
            checkpoint()
        _rename_new(staging, destination)
        published = True
        _sync_directory(parent)
        return {**result, "directory": str(destination)}
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
