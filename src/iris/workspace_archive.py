"""Streaming, checksummed archives of an idle local IRIS workspace."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath

from iris import __version__
from iris.store import (
    BENCHMARK_TABLES,
    DINOX_TABLES,
    JSON_FIELDS,
    MODEL_EXPORT_TABLES,
    SCHEMA,
    SCHEMA_V12,
    SCHEMA_V13,
    SCHEMA_V14,
    SCHEMA_V15,
    SCHEMA_V16,
    SCHEMA_V17,
    SCHEMA_V18,
    SCHEMA_VERSION,
    TABLES,
    TRAINING_CHECKPOINT_TABLES,
    now,
)
from iris.taxonomies import validate_taxonomy_records

PROTOCOL = "iris-workspace-archive-v1"
FORMAT_VERSION = 1
SCHEMAS = {
    12: SCHEMA_V12,
    13: SCHEMA_V13,
    14: SCHEMA_V14,
    15: SCHEMA_V15,
    16: SCHEMA_V16,
    17: SCHEMA_V17,
    18: SCHEMA_V18,
    SCHEMA_VERSION: SCHEMA,
}
SCHEMA_TABLES = {
    12: TABLES
    - BENCHMARK_TABLES
    - MODEL_EXPORT_TABLES
    - TRAINING_CHECKPOINT_TABLES
    - DINOX_TABLES
    - {"projects", "taxonomy_versions"},
    13: TABLES
    - BENCHMARK_TABLES
    - MODEL_EXPORT_TABLES
    - TRAINING_CHECKPOINT_TABLES
    - DINOX_TABLES
    - {"taxonomy_versions"},
    14: TABLES - BENCHMARK_TABLES - MODEL_EXPORT_TABLES - TRAINING_CHECKPOINT_TABLES - DINOX_TABLES,
    15: TABLES
    - MODEL_EXPORT_TABLES
    - TRAINING_CHECKPOINT_TABLES
    - DINOX_TABLES
    - {"benchmark_reports"},
    16: TABLES - MODEL_EXPORT_TABLES - TRAINING_CHECKPOINT_TABLES - DINOX_TABLES,
    17: TABLES - TRAINING_CHECKPOINT_TABLES - DINOX_TABLES,
    18: TABLES - DINOX_TABLES,
    SCHEMA_VERSION: TABLES,
}
CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024**3
MAX_TOTAL_BYTES = 64 * 1024**3
MAX_FILES = 100000
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_REFERENCE_BYTES = 64 * 1024**2
MAX_MODEL_EXPORT_BYTES = 1280 * 1024**2
MAX_TRAINING_CHECKPOINT_BYTES = 512 * 1024**2
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CATEGORIES = {
    "database": "Workspace database",
    "assets": "Original media",
    "frames": "Extracted images",
    "imports": "Imported datasets",
    "datasets": "Frozen dataset versions",
    "benchmarks": "Frozen benchmark references and images",
    "models": "Detector checkpoints",
    "model_exports": "Standalone trained-model packages",
    "training_checkpoints": "Training states for explicit continuation",
    "assistance": "Annotation review previews",
    "video_reviews": "Video review storyboards",
    "reports": "Experiment images",
    "logs": "Job logs",
    "ollama": "Local Ollama models",
}
_EXCLUDED_ROOTS = {"workspace_operations", "backups", "restore_uploads", "exports", "uploads"}
EXCLUSIONS = [
    "Environment files, credentials, scripts, source repositories and runtime environments",
    "Uploads, generated exports, backup archives and workspace-operation staging files",
    "SQLite journals and process locks; the database is saved through SQLite's backup API",
    "Incomplete downloads, temporary files and unpublished staging directories",
]


class ArchiveError(ValueError):
    """The workspace or archive is incomplete, incompatible or unsafe."""


class ArchiveLimitError(ArchiveError):
    """The bounded archive operation would exceed a supported size or count."""


class ArchiveCancelled(Exception):
    """The user cancelled an operation before publication."""


def safe_member_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ArchiveError("Archive paths must be bounded, nonempty relative file names")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or ":" in value
        or path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(
            part.endswith((" ", "."))
            or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", part.split(".")[0], re.I)
            for part in value.split("/")
        )
    ):
        raise ArchiveError(
            "Archive paths must not contain traversal, absolute paths or backslashes"
        )
    return value


def _excluded_part(part):
    return (
        part.startswith(".")
        or part.endswith((".part", ".tmp", ".lock", "-wal", "-shm", "-journal"))
        or part
        in {
            "id_ed25519",
            "id_ed25519.pub",
            "id_rsa",
            "id_rsa.pub",
            "credentials.json",
            "secrets.json",
            "config.yaml",
            "config.yml",
        }
        or "-partial" in part
    )


def allowed_artifact_path(value: str) -> bool:
    try:
        safe_member_path(value)
    except ArchiveError:
        return False
    if value == "iris.sqlite3":
        return True
    parts = value.split("/")
    if any(_excluded_part(part) for part in parts):
        return False
    root = parts[0]
    if root in {"assets", "frames", "logs"} and len(parts) == 2:
        return root == "assets" or parts[-1].endswith(".png" if root == "frames" else ".log")
    if root == "imports" and len(parts) == 3:
        return parts[-1] == "source.zip" or parts[-1].endswith((".source", ".png"))
    if root in {"datasets", "benchmarks"}:
        return (len(parts) == 3 and parts[-1] == "manifest.json") or (
            len(parts) == 4 and parts[2] == "images" and parts[-1].endswith(".png")
        )
    if root == "models":
        return (
            value == "models/sam3/sam3.pt"
            or (len(parts) == 2 and parts[-1].endswith((".pth", ".pth.json")))
            or (len(parts) == 3 and parts[1] == "trained" and parts[-1].endswith(".pth"))
        )
    if root == "model_exports":
        return len(parts) == 3 and parts[-1] == "model.zip"
    if root == "training_checkpoints":
        return len(parts) == 3 and parts[-1].endswith(".pth")
    if root == "assistance":
        return len(parts) == 4 and parts[1] == "previews" and parts[-1].endswith(".jpg")
    if root in {"video_reviews", "reports"}:
        return len(parts) == 3 and parts[-1].endswith(".jpg")
    if root == "ollama":
        return (
            len(parts) >= 4
            and parts[1] == "manifests"
            or len(parts) == 3
            and parts[1] == "blobs"
            and re.fullmatch(r"sha256-[0-9a-f]{64}", parts[2]) is not None
            or len(parts) == 3
            and parts[1] == "metadata"
            and re.fullmatch(r"sha256-[0-9a-f]{64}\.json", parts[2]) is not None
        )
    return False


def is_reference_document(value: str) -> bool:
    parts = value.split("/")
    return (
        len(parts) == 3
        and parts[0] in {"datasets", "benchmarks"}
        and parts[-1] == "manifest.json"
        or len(parts) >= 4
        and parts[:2] == ["ollama", "manifests"]
        or len(parts) == 2
        and parts[0] == "models"
        and parts[-1].endswith(".pth.json")
    )


def is_model_export_bundle(value: str) -> bool:
    return value.startswith("model_exports/") and allowed_artifact_path(value)


def _canonical(value):
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise ArchiveError("Archive metadata must contain finite JSON values") from exc


def _portable_names(names):
    normalized = [unicodedata.normalize("NFC", name).casefold() for name in names]
    if len(normalized) != len(set(normalized)):
        raise ArchiveError("Workspace paths collide on a case-insensitive filesystem")


def validate_manifest(value: dict) -> dict:
    keys = {
        "protocol",
        "format_version",
        "app_version",
        "schema_version",
        "created_at",
        "file_count",
        "total_bytes",
        "counts",
        "files",
        "exclusions",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ArchiveError("Unsupported workspace archive manifest fields")
    if (
        value["protocol"] != PROTOCOL
        or type(value["format_version"]) is not int
        or value["format_version"] != FORMAT_VERSION
        or type(value["schema_version"]) is not int
        or value["schema_version"] not in SCHEMAS
    ):
        raise ArchiveError("Unsupported workspace archive or database version")
    if not isinstance(value["app_version"], str) or not 1 <= len(value["app_version"]) <= 64:
        raise ArchiveError("Archive application version is missing")
    try:
        if (
            not isinstance(value["created_at"], str)
            or datetime.fromisoformat(value["created_at"]).tzinfo is None
        ):
            raise ValueError
    except ValueError as exc:
        raise ArchiveError("Archive creation time must include a timezone") from exc
    if (
        not isinstance(value["counts"], dict)
        or set(value["counts"]) != SCHEMA_TABLES[value["schema_version"]]
        or any(type(number) is not int or number < 0 for number in value["counts"].values())
    ):
        raise ArchiveError("Archive table counts are incomplete or invalid")
    files = value["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise ArchiveLimitError("Archive file count exceeds the supported limit")
    total, names = 0, []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
            raise ArchiveError("Invalid archive file descriptor")
        name = safe_member_path(item["path"])
        if (
            not allowed_artifact_path(name)
            or not isinstance(item["sha256"], str)
            or _SHA.fullmatch(item["sha256"]) is None
        ):
            raise ArchiveError("Archive contains an unsupported path or checksum")
        if type(item["size_bytes"]) is not int or not 0 <= item["size_bytes"] <= MAX_TOTAL_BYTES:
            raise ArchiveLimitError("Archive member size exceeds the supported limit")
        names.append(name)
        total += item["size_bytes"]
    if names != sorted(set(names)) or "iris.sqlite3" not in names:
        raise ArchiveError("Archive files must be unique, sorted and include iris.sqlite3")
    _portable_names(names)
    names_set = set(names)
    for name in names:
        if any(
            parent.as_posix() in names_set
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        ):
            raise ArchiveError("Archive paths have a file/directory conflict")
    if (
        type(value["file_count"]) is not int
        or value["file_count"] != len(files)
        or type(value["total_bytes"]) is not int
        or value["total_bytes"] != total
    ):
        raise ArchiveError("Archive totals do not match the file inventory")
    if total > MAX_TOTAL_BYTES:
        raise ArchiveLimitError("Expanded workspace exceeds the 64 GiB limit")
    if not isinstance(value["exclusions"], list) or any(
        not isinstance(text, str) or len(text) > 1000 for text in value["exclusions"]
    ):
        raise ArchiveError("Archive exclusions must be text")
    if len(_canonical(value)) > MAX_MANIFEST_BYTES:
        raise ArchiveLimitError("Archive manifest exceeds the 16 MiB limit")
    return value


def _stat_signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _inventory(root):
    files, excluded = {}, []

    def walk(directory, depth=0):
        if depth > 16:
            raise ArchiveError("Workspace artifact directory nesting is too deep")
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if _excluded_part(entry.name) or (depth == 0 and entry.name in _EXCLUDED_ROOTS):
                excluded.append(
                    {"path": relative, "reason": "Temporary, private or generated file"}
                )
                continue
            if depth == 0 and entry.name not in {*_CATEGORIES, "iris.sqlite3"}:
                excluded.append(
                    {"path": relative, "reason": "Outside the managed workspace artifacts"}
                )
                continue
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ArchiveError(f"Workspace contains a symbolic link: {relative}")
            if stat.S_ISDIR(info.st_mode):
                walk(path, depth + 1)
            elif stat.S_ISREG(info.st_mode):
                if not allowed_artifact_path(relative):
                    excluded.append({"path": relative, "reason": "Not a managed artifact"})
                    continue
                files[relative] = {
                    "path": relative,
                    "size_bytes": info.st_size,
                    "signature": _stat_signature(info),
                }
                if len(files) > MAX_FILES:
                    raise ArchiveLimitError("Workspace contains more than 100000 files")
            else:
                raise ArchiveError(f"Workspace contains a non-regular artifact: {relative}")

    walk(root)
    if "iris.sqlite3" not in files:
        raise ArchiveError("The workspace database iris.sqlite3 is missing")
    if sum(item["size_bytes"] for item in files.values()) > MAX_TOTAL_BYTES:
        raise ArchiveLimitError("Workspace exceeds the 64 GiB archive limit")
    _portable_names(files)
    return files, excluded


def _parse_json(raw):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON number")
        return number

    try:
        return json.loads(
            raw, parse_constant=reject, parse_float=finite_float, object_pairs_hook=unique_pairs
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise ArchiveError("Workspace metadata contains invalid or nonfinite JSON") from exc


def _read_json(path, limit=MAX_REFERENCE_BYTES):
    try:
        if path.is_symlink() or path.stat().st_size > limit:
            raise ArchiveError("Reference document is linked or exceeds its size limit")
        raw = path.read_bytes()

        return _parse_json(raw), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, RecursionError) as exc:
        raise ArchiveError(
            "Workspace reference document is missing, invalid or unreadable"
        ) from exc


def _readonly_database(path):
    if path.is_symlink() or not path.is_file():
        raise ArchiveError("Workspace database must be an existing regular file")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_REFERENCE_BYTES)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    return connection


def _schema_signature(connection, tables):
    result = {}
    for table in sorted(tables):
        columns = {
            row[1]: tuple(row[2:]) for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        foreign = sorted(
            tuple(row)[2:] for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
        )
        indexes = []
        for index in connection.execute(f'PRAGMA index_list("{table}")'):
            index_name = index[1].replace('"', '""')
            # Column order in migrated tables differs, but declared index keys do not.
            indexes.append(
                (
                    index[2],
                    index[3],
                    index[4],
                    tuple(
                        row[2] for row in connection.execute(f'PRAGMA index_info("{index_name}")')
                    ),
                )
            )
        result[table] = (columns, foreign, sorted(indexes))
    return result


def _validate_benchmarks(connection, root, require):
    """Validate frozen references and ownership without opening or migrating Store."""
    benchmarks = connection.execute("SELECT * FROM benchmarks").fetchall()
    if not benchmarks:
        return
    # The pure validator reads only the manifest. Candidate models and the active
    # project's current taxonomy are deliberately not consulted during recovery.
    from iris.benchmark import validate_benchmark_config, validate_benchmark_manifest
    from iris.store import _decode
    from iris.taxonomies import _get

    manifests, benchmark_rows, configurations = {}, {}, {}
    for row in benchmarks:
        expected_path = f"benchmarks/{row['id']}/manifest.json"
        if row["path"] != expected_path:
            raise ArchiveError("Benchmark manifest path differs from its frozen directory")
        require(row["path"], (f"benchmarks/{row['id']}/",), row["manifest_sha256"])
        manifest, digest = _read_json(root / row["path"])
        try:
            validate_benchmark_manifest(manifest)
            if (
                digest != row["manifest_sha256"]
                or manifest["id"] != row["id"]
                or manifest["project_id"] != row["project_id"]
                or _get(connection, manifest["taxonomy"]["id"], row["project_id"])
                != manifest["taxonomy"]
            ):
                raise ValueError("Frozen identity or class definitions differ")
        except ValueError as exc:
            raise ArchiveError("Benchmark manifest or frozen taxonomy is invalid") from exc
        frames = {}
        for frame in manifest["frames"]:
            if frame["image_path"] != f"benchmarks/{row['id']}/images/{frame['frame_id']}.png":
                raise ArchiveError("Frozen image path differs from its benchmark directory")
            require(
                frame["image_path"],
                (f"benchmarks/{row['id']}/images/",),
                frame["image_file_sha256"],
            )
            source = connection.execute(
                "SELECT f.session_id,s.project_id FROM frames f "
                "JOIN sessions s ON s.id=f.session_id WHERE f.id=?",
                (frame["frame_id"],),
            ).fetchone()
            annotation = _decode(
                connection.execute(
                    "SELECT * FROM annotation_revisions WHERE id=?",
                    (frame["annotation_revision_id"],),
                ).fetchone()
            )
            if (
                source is None
                or source["session_id"] != frame["session_id"]
                or source["project_id"] != row["project_id"]
                or annotation != frame["annotation"]
                or annotation["frame_id"] != frame["frame_id"]
            ):
                raise ArchiveError("Benchmark reference has an invalid frame or revision owner")
            frames[frame["frame_id"]] = frame
        manifests[row["id"]] = frames
        benchmark_rows[row["id"]] = (dict(row), manifest)
    for raw_row in connection.execute("SELECT * FROM benchmark_configs"):
        row = _decode(raw_row)
        benchmark, manifest = benchmark_rows[row["benchmark_id"]]
        try:
            validate_benchmark_config(row, benchmark, manifest)
        except ValueError as exc:
            raise ArchiveError("Benchmark frozen configuration is invalid") from exc
        configurations[row["id"]] = row
    external_trials, sam_trials, combined_trials, recorded_trials = {}, {}, {}, {}
    for row in connection.execute(
        "SELECT t.*,c.benchmark_id AS config_benchmark_id,j.kind,j.params,"
        "j.status AS job_status,j.result AS job_result "
        "FROM benchmark_trials t JOIN benchmark_configs c ON c.id=t.config_id "
        "JOIN jobs j ON j.id=t.job_id"
    ):
        frozen = _parse_json(row["config"])
        config = configurations[row["config_id"]]
        benchmark, manifest = benchmark_rows[row["benchmark_id"]]
        if (
            row["config_benchmark_id"] != row["benchmark_id"]
            or row["kind"] != "benchmark"
            or _parse_json(row["params"]).get("trial_id") != row["id"]
            or not isinstance(frozen, dict)
            or frozen.get("protocol") != manifest["protocol"]
            or frozen.get("source_config_fingerprint") != config["fingerprint"]
            or frozen.get("benchmark_manifest_sha256") != benchmark["manifest_sha256"]
            or frozen.get("candidate_config") != config["config"]
            or frozen.get("role") != row["split"]
            or frozen.get("frame_ids")
            != [frame["frame_id"] for frame in manifest["frames"] if frame["role"] == row["split"]]
            or not isinstance(frozen.get("fingerprint"), str)
            or _SHA.fullmatch(frozen["fingerprint"]) is None
        ):
            raise ArchiveError("Benchmark trial has an invalid configuration or job owner")
        if config["approach"] == "multimodal":
            from iris.benchmark_dispatch import validate_external_trial

            try:
                plan = validate_external_trial(
                    frozen,
                    config["config"],
                    [frame for frame in manifest["frames"] if frame["role"] == row["split"]],
                )
            except ValueError as exc:
                raise ArchiveError(
                    "External benchmark consent or planning budget is invalid"
                ) from exc
            job_result = _parse_json(row["job_result"]) if row["job_result"] is not None else {}
            external_trials[row["id"]] = {
                "trial": {**dict(row), "config": frozen},
                "plan": plan,
                "frames": set(),
                "reserved": 0,
                "attempt": job_result.get("benchmark_attempt_id")
                if isinstance(job_result, dict)
                else None,
            }
        elif config["approach"] == "combined":
            from iris.benchmark_combined_dispatch import validate_external_trial

            try:
                plan = validate_external_trial(
                    frozen,
                    config["config"],
                    [frame for frame in manifest["frames"] if frame["role"] == row["split"]],
                )
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ArchiveError("Combined benchmark consent or frozen plan is invalid") from exc
            job_result = _parse_json(row["job_result"]) if row["job_result"] else {}
            combined_trials[row["id"]] = {
                "trial": {**dict(row), "config": frozen},
                "plan": plan,
                "frames": set(),
                "reserved": 0,
                "attempt": job_result.get("benchmark_attempt_id")
                if isinstance(job_result, dict)
                else None,
            }
        elif config["approach"] == "segmentation":
            from iris.benchmark_segmentation import validate_trial

            try:
                plan = validate_trial(
                    frozen,
                    config["config"],
                    [frame for frame in manifest["frames"] if frame["role"] == row["split"]],
                )
            except ValueError as exc:
                raise ArchiveError("SAM benchmark runtime or work plan is invalid") from exc
            job_result = _parse_json(row["job_result"]) if row["job_result"] else {}
            sam_trials[row["id"]] = {
                "config": config,
                "plan": plan,
                "attempt": job_result.get("benchmark_attempt_id")
                if isinstance(job_result, dict)
                else None,
            }
        elif config["approach"] == "recorded_proposals":
            from iris.benchmark_recorded import validate_trial

            try:
                bundle = validate_trial(
                    frozen,
                    config["config"],
                    [frame for frame in manifest["frames"] if frame["role"] == row["split"]],
                )
                if _parse_json(row["params"]) != {
                    "trial_id": row["id"],
                    "operation": "import_recorded_proposals",
                }:
                    raise ValueError("Recorded proposal import has an invalid job operation")
                job_result = _parse_json(row["job_result"]) if row["job_result"] else {}
                if (
                    not isinstance(job_result, dict)
                    or job_result
                    and (
                        job_result.get("trial_id") != row["id"]
                        or job_result.get("operation") != "import_recorded_proposals"
                    )
                ):
                    raise ValueError("Recorded proposal result has an invalid job owner")
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ArchiveError("Recorded benchmark source bundle or job is invalid") from exc
            recorded_trials[row["id"]] = {
                "trial": {**dict(row), "config": frozen},
                "bundle": bundle,
                "frames": set(),
                "attempt": job_result.get("benchmark_attempt_id")
                if isinstance(job_result, dict)
                else None,
            }
    for row in connection.execute(
        "SELECT o.*,t.benchmark_id,t.split FROM benchmark_outputs o "
        "JOIN benchmark_trials t ON t.id=o.trial_id"
    ):
        frame = manifests[row["benchmark_id"]].get(row["frame_id"])
        if frame is None or frame["role"] != row["split"]:
            raise ArchiveError("Benchmark output is outside its trial's frozen image partition")
        if row["trial_id"] in external_trials:
            from iris.benchmark_dispatch import validate_output_row

            saved = _decode(row)
            external = external_trials[row["trial_id"]]
            try:
                dispatch = validate_output_row(
                    saved, external["trial"], validated_plan=external["plan"]
                )
                if (
                    dispatch.get("attempt_id") is not None
                    and dispatch["attempt_id"] != external["attempt"]
                ):
                    raise ValueError("External output attempt does not own its job")
            except ValueError as exc:
                raise ArchiveError("External benchmark dispatch receipt is invalid") from exc
            external["frames"].add(row["frame_id"])
            external["reserved"] += saved["metadata"]["budget"]["reserved_microusd"]
        elif row["trial_id"] in combined_trials:
            from iris.benchmark_combined_dispatch import validate_output_row

            combined = combined_trials[row["trial_id"]]
            try:
                pipeline = validate_output_row(
                    _decode(row), combined["trial"], validated_plan=combined["plan"]
                )
                for stage in pipeline["stages"].values():
                    if (
                        stage.get("attempt_id") is not None
                        and stage["attempt_id"] != combined["attempt"]
                    ):
                        raise ValueError("Combined stage attempt does not own its job")
                    combined["reserved"] += stage["budget"]["reserved_microusd"]
                combined["frames"].add(row["frame_id"])
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ArchiveError("Combined benchmark stage evidence is invalid") from exc
        elif row["trial_id"] in sam_trials:
            from iris.benchmark_segmentation import validate_saved_output

            sam = sam_trials[row["trial_id"]]
            try:
                validate_saved_output(
                    _decode(row),
                    config=sam["config"],
                    frame=frame,
                    plan=sam["plan"],
                    attempt=sam["attempt"],
                )
            except (ValueError, TypeError, AttributeError) as exc:
                raise ArchiveError("SAM benchmark native output evidence is invalid") from exc
        elif row["trial_id"] in recorded_trials:
            from iris.benchmark_recorded import validate_output_row

            recorded = recorded_trials[row["trial_id"]]
            try:
                if not isinstance(recorded["attempt"], str) or not recorded["attempt"]:
                    raise ValueError("Recorded output has no owning import attempt")
                validate_output_row(
                    _decode(row),
                    recorded["trial"],
                    validated_bundle=recorded["bundle"],
                    attempt=recorded["attempt"],
                )
                recorded["frames"].add(row["frame_id"])
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ArchiveError("Recorded benchmark native output evidence is invalid") from exc
    for external in (*external_trials.values(), *combined_trials.values()):
        if (
            external["frames"] != {request["frame_id"] for request in external["plan"]["requests"]}
            or external["reserved"] > external["plan"]["approval"]["budget_microusd"]
        ):
            raise ArchiveError("External benchmark request coverage or budget is inconsistent")
    for recorded in recorded_trials.values():
        if recorded["trial"]["job_status"] == "succeeded" and recorded["frames"] != set(
            recorded["trial"]["config"]["frame_ids"]
        ):
            raise ArchiveError("Completed recorded benchmark output coverage is inconsistent")
    for row in connection.execute("SELECT elapsed_ms,segments FROM benchmark_timers"):
        if (
            type(row["elapsed_ms"]) not in (int, float)
            or not math.isfinite(row["elapsed_ms"])
            or row["elapsed_ms"] < 0
            or not isinstance(_parse_json(row["segments"]), list)
        ):
            raise ArchiveError("Benchmark timer has invalid measured intervals")
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='benchmark_reports'"
    ).fetchone():
        from iris.benchmark_reports import validate_report_row

        for row in connection.execute("SELECT * FROM benchmark_reports"):
            try:
                validate_report_row(
                    _decode(row),
                    connection=connection,
                    manifest=benchmark_rows[row["benchmark_id"]][1],
                )
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ArchiveError(
                    "Benchmark report snapshot or original evidence is invalid"
                ) from exc


def _validate_model_exports(connection, root, require):
    """Resolve export ownership before validating frozen, portable evidence."""
    rows = connection.execute("SELECT * FROM model_exports").fetchall()
    if not rows:
        return
    from iris.model_exports import validate_export_archive
    from iris.store import _decode

    for raw_row in rows:
        row = _decode(raw_row)
        owners = connection.execute(
            "SELECT d.project_id AS model_project,e.project_id AS evaluation_project "
            "FROM trained_models m JOIN training_runs t ON t.id=m.training_id "
            "JOIN dataset_versions d ON d.id=t.dataset_id "
            "JOIN evaluations v ON v.id=? JOIN dataset_versions e ON e.id=v.dataset_id "
            "WHERE m.id=?",
            (row["evaluation_id"], row["trained_model_id"]),
        ).fetchone()
        if owners is None or owners["model_project"] != owners["evaluation_project"]:
            raise ArchiveError("Model export source records belong to different projects")
        if row["job_id"] is not None:
            job = connection.execute(
                "SELECT kind,params FROM jobs WHERE id=?", (row["job_id"],)
            ).fetchone()
            params = _parse_json(job["params"]) if job is not None else None
            if (
                job is None
                or job["kind"] != "model_export"
                or not isinstance(params, dict)
                or params.get("export_id") != row["id"]
            ):
                raise ArchiveError("Model export has an invalid job owner")
        published = [
            row[key] is not None
            for key in ("path", "manifest", "manifest_sha256", "archive_sha256")
        ]
        if any(published) and not all(published):
            raise ArchiveError("Model export publication is incomplete")
        if all(published):
            if row["job_id"] is None:
                raise ArchiveError("Model export publication has no owning job")
            if row["path"] != f"model_exports/{row['id']}/model.zip":
                raise ArchiveError("Model export package path differs from its frozen directory")
            require(row["path"], (f"model_exports/{row['id']}/",), row["archive_sha256"])
        try:
            validate_export_archive(row, connection=connection, root=root)
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ArchiveError(
                "Model export package or frozen measurement evidence is invalid"
            ) from exc


def _validate_training_checkpoints(connection, root, require):
    """Validate durable state identities without reading or deserializing tensors."""
    from iris.store import _decode
    from iris.training_recovery import (
        validate_training_checkpoint,
        validate_training_recoveries,
    )

    for raw_row in connection.execute("SELECT * FROM training_checkpoints"):
        row = _decode(raw_row)
        expected_path = f"training_checkpoints/{row['training_id']}/{row['id']}.pth"
        if row["path"] != expected_path:
            raise ArchiveError("Training checkpoint path differs from its owning run")
        if (
            type(row["step"]) is not int
            or row["step"] <= 0
            or type(row["size_bytes"]) is not int
            or not 0 < row["size_bytes"] <= MAX_TRAINING_CHECKPOINT_BYTES
        ):
            raise ArchiveLimitError("Training checkpoint step or file size is unsupported")
        owner = connection.execute(
            "SELECT j.kind,j.params FROM training_runs t JOIN jobs j ON j.id=t.job_id WHERE t.id=?",
            (row["training_id"],),
        ).fetchone()
        params = _parse_json(owner["params"]) if owner is not None else None
        if (
            owner is None
            or owner["kind"] != "train"
            or not isinstance(params, dict)
            or params.get("training_id") != row["training_id"]
        ):
            raise ArchiveError("Training checkpoint has an invalid job owner")
        require(
            row["path"],
            (f"training_checkpoints/{row['training_id']}/",),
            row["state_sha256"],
            row["size_bytes"],
        )
        try:
            validate_training_checkpoint(row, connection=connection, root=root)
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ArchiveError(
                "Training checkpoint metadata or source identity is invalid"
            ) from exc
    try:
        validate_training_recoveries(connection, root)
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        raise ArchiveError("Training continuation lineage or source checkpoint is invalid") from exc


def _validate_dinox(connection):
    """Check saved receipts and local provenance without authenticating or making requests."""
    from iris.store import _decode

    batches = {row["id"]: _decode(row) for row in connection.execute("SELECT * FROM dinox_batches")}
    requests = {
        row["id"]: _decode(row) for row in connection.execute("SELECT * FROM dinox_requests")
    }
    suggestions = [
        _decode(row)
        for row in connection.execute(
            "SELECT * FROM annotation_suggestions WHERE json_extract(metadata,'$.provider')='dinox'"
        )
    ]
    if not batches and not requests and not suggestions:
        return {}

    from iris.dinox_provider import normalize, validate_frozen_config
    from iris.taxonomies import _get as saved_taxonomy

    frames = {row["id"]: dict(row) for row in connection.execute("SELECT * FROM frames")}
    jobs = {row["id"]: _decode(row) for row in connection.execute("SELECT * FROM jobs")}
    sessions = {row["id"]: dict(row) for row in connection.execute("SELECT * FROM sessions")}
    by_job = {batch["job_id"]: batch for batch in batches.values()}
    frame_fields = {
        "id",
        "session_id",
        "asset_id",
        "sha256",
        "path",
        "width",
        "height",
        "taxonomy_id",
    }

    def invalid(message):
        raise ArchiveError(f"DINO-X {message}")

    def profile(value, session_id):
        try:
            validate_frozen_config(value)
            taxonomy = saved_taxonomy(
                connection, value["taxonomy_id"], sessions[session_id]["project_id"]
            )
            if value["taxonomy"] != taxonomy:
                raise ValueError("Frozen taxonomy differs from its saved definitions")
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ArchiveError("DINO-X frozen provider configuration is invalid") from exc

    def image_snapshot(value, frame_id):
        frame = frames.get(frame_id)
        if (
            frame is None
            or not isinstance(value, dict)
            or not isinstance(value.get("frame"), dict)
            or set(value["frame"]) != frame_fields
            or any(value["frame"][key] != frame[key] for key in frame_fields)
            or not isinstance(value.get("taxonomy_id"), str)
            or not value["taxonomy_id"]
        ):
            invalid("image snapshot differs from its saved source")

    for batch in batches.values():
        frame_ids, config, metadata = batch["frame_ids"], batch["config"], batch["metadata"]
        job = jobs.get(batch["job_id"])
        if (
            not isinstance(frame_ids, list)
            or not frame_ids
            or any(
                not isinstance(identifier, str) or identifier not in frames
                for identifier in frame_ids
            )
            or len(set(frame_ids)) != len(frame_ids)
            or any(
                frames[identifier]["session_id"] != batch["session_id"] for identifier in frame_ids
            )
            or job is None
            or job["kind"] != "dinox"
            or not isinstance(job["params"], dict)
            or job["params"].get("batch_id") != batch["id"]
        ):
            invalid("batch has an invalid frame, session or job owner")
        if (
            not isinstance(config, dict)
            or config.get("protocol") != "iris-dinox-batch-v1"
            or not isinstance(metadata, dict)
        ):
            invalid("batch configuration or metadata is invalid")
        profile(config.get("provider_config"), batch["session_id"])
        snapshots = config.get("frames", [])
        if not isinstance(snapshots, list):
            invalid("batch image snapshots must be a list")
        for item in snapshots:
            if not isinstance(item, dict) or not isinstance(item.get("snapshot"), dict):
                invalid("batch image snapshot is invalid")
            frame_id = item["snapshot"].get("frame", {}).get("id")
            if frame_id not in frame_ids:
                invalid("batch snapshot refers to an image outside its batch")
            image_snapshot(item["snapshot"], frame_id)
            if item["snapshot"]["taxonomy_id"] != config["provider_config"]["taxonomy_id"]:
                invalid("batch image taxonomy differs from its frozen provider configuration")

    states = Counter()
    for request in requests.values():
        owner = by_job.get(request["job_id"])
        if owner is None or request["frame_id"] not in owner["frame_ids"]:
            invalid("request has an invalid image or executor job owner")
        profile(request["config"], owner["session_id"])
        if request["config"] != owner["config"]["provider_config"]:
            invalid("request configuration differs from its executor batch")
        image_snapshot(request["snapshot"], request["frame_id"])
        if request["snapshot"]["taxonomy_id"] != request["config"]["taxonomy_id"]:
            invalid("request image taxonomy differs from its frozen provider configuration")
        digest = hashlib.sha256(
            json.dumps(
                {"snapshot": request["snapshot"], "config": request["config"]},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if request["cache_key"] != digest or not isinstance(request["metadata"], dict):
            invalid("request cache identity or metadata is invalid")
        previous_id = request["metadata"].get("previous_request_id")
        if previous_id is not None:
            previous = requests.get(previous_id) if isinstance(previous_id, str) else None
            if (
                previous is None
                or previous_id == request["id"]
                or previous["frame_id"] != request["frame_id"]
                or previous["cache_key"] != request["cache_key"]
            ):
                invalid("request retry provenance refers to an unrelated earlier request")
        task_id, state = request["task_id"], request["state"]
        if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
            invalid("request task identifier is invalid")
        if state == "submitted" and task_id is None:
            invalid("submitted request has no saved provider task identifier")
        if request["raw_response"] is not None and not isinstance(request["raw_response"], dict):
            invalid("provider response must be a finite JSON object")
        if request["result"] is not None and not isinstance(request["result"], dict):
            invalid("normalized result must be a finite JSON object")
        if state in {"response_received", "succeeded"} and request["raw_response"] is None:
            invalid("received request has no saved provider response")
        if state == "succeeded" and request["result"] is None:
            invalid("successful request has no normalized result")
        if state == "succeeded":
            frame = request["snapshot"]["frame"]
            try:
                normalized = normalize(
                    request["raw_response"], request["config"], frame["width"], frame["height"]
                )
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ArchiveError("DINO-X saved provider output is invalid") from exc
            if request["result"] != normalized:
                invalid("normalized result differs from its saved provider response")
        states[state] += 1

    for batch in batches.values():
        entries = batch["metadata"].get("frames", [])
        if not isinstance(entries, list):
            invalid("batch frame progress must be a list")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("frame_id"), str):
                invalid("batch frame progress is invalid")
            frame_id = entry["frame_id"]
            if frame_id in seen or frame_id not in batch["frame_ids"]:
                invalid("batch frame progress refers to a duplicate or unrelated image")
            seen.add(frame_id)
            request_id = entry.get("request_id")
            if request_id is not None:
                request = requests.get(request_id) if isinstance(request_id, str) else None
                if (
                    request is None
                    or request["frame_id"] != frame_id
                    or request["config"] != batch["config"]["provider_config"]
                ):
                    invalid("batch progress refers to an unrelated provider request")

    for suggestion in suggestions:
        metadata = suggestion["metadata"]
        request = requests.get(metadata.get("dinox_request_id"))
        batch = batches.get(metadata.get("batch_id"))
        if (
            suggestion["kind"] != "detector"
            or request is None
            or batch is None
            or request["state"] != "succeeded"
            or suggestion["frame_id"] != request["frame_id"]
            or suggestion["frame_id"] not in batch["frame_ids"]
            or suggestion["job_id"] != batch["job_id"]
            or request["config"] != batch["config"]["provider_config"]
            or metadata.get("frame_sha256") != request["snapshot"]["frame"]["sha256"]
            or metadata.get("target_taxonomy") != request["snapshot"]["taxonomy_id"]
        ):
            invalid("suggestion provenance does not match its batch and provider request")
        proposals = {
            hashlib.sha256(f"dinox:{request['id']}:{proposal['id']}".encode()).hexdigest(): proposal
            for proposal in request["result"]["proposals"]
        }
        proposal = proposals.get(suggestion["id"])
        if (
            proposal is None
            or any(suggestion[field] != proposal[field] for field in ("label", "box"))
            or any(
                metadata.get(field) != proposal[field] for field in ("score", "source", "geometry")
            )
            or metadata.get("threshold") != request["config"]["settings"]["bbox_threshold"]
        ):
            invalid("suggestion differs from its recorded normalized proposal")
    return dict(sorted(states.items()))


def validate_database(
    root: Path, inventory: dict, *, verify_hashes=False, database_path=None
) -> dict:
    """Check schema and all persisted file references without opening models.

    Inspection may supply only SQLite and reference JSON documents on disk; all
    other existence and size checks use the verified archive inventory.
    """
    root = Path(root).resolve()
    expected = {}
    if not isinstance(inventory, dict) or "iris.sqlite3" not in inventory:
        raise ArchiveError("Workspace inventory has no database")

    def require(path, prefixes, digest=None, size=None):
        safe_member_path(path)
        if (
            not allowed_artifact_path(path)
            or not any(path.startswith(prefix) for prefix in prefixes)
            or path not in inventory
        ):
            raise ArchiveError(f"A required workspace artifact is missing or unsafe: {path}")
        if digest is not None:
            if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
                raise ArchiveError("A workspace artifact has an invalid recorded checksum")
            if path in expected and expected[path] != digest:
                raise ArchiveError("Workspace records disagree about an artifact checksum")
            expected[path] = digest
        if size is not None and (
            type(size) is not int or size < 0 or inventory[path]["size_bytes"] != size
        ):
            raise ArchiveError(f"A required workspace artifact has changed size: {path}")
        return path

    connection = None
    try:
        connection = _readonly_database(
            Path(database_path) if database_path is not None else root / "iris.sqlite3"
        )
        with connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in SCHEMAS:
                raise ArchiveError(
                    "Workspace database version is unsupported; open it in IRIS first"
                )
            tables = SCHEMA_TABLES[version]
            objects = connection.execute(
                "SELECT name,type,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if {row["name"] for row in objects if row["type"] == "table"} != tables or any(
                row["type"] in {"view", "trigger"}
                or "CREATE VIRTUAL TABLE" in (row["sql"] or "").upper()
                for row in objects
            ):
                raise ArchiveError("Workspace database has unsupported tables, views or triggers")
            reference = sqlite3.connect(":memory:")
            try:
                reference.executescript(SCHEMAS[version])
                if _schema_signature(connection, tables) != _schema_signature(reference, tables):
                    raise ArchiveError(
                        "Workspace database schema does not match the supported layout"
                    )
            finally:
                reference.close()
            if (
                connection.execute("PRAGMA quick_check").fetchall()[0][0] != "ok"
                or connection.execute("PRAGMA foreign_key_check").fetchone() is not None
            ):
                raise ArchiveError("Workspace database integrity checks failed")
            counts = {
                table: connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                for table in sorted(tables)
            }
            for table in sorted(tables):
                columns = [
                    row[1]
                    for row in connection.execute(f'PRAGMA table_info("{table}")')
                    if row[1] in JSON_FIELDS
                ]
                if columns:
                    selection = ",".join(f'"{column}"' for column in columns)
                    for row in connection.execute(f'SELECT {selection} FROM "{table}"'):
                        for raw in row:
                            if raw is not None:
                                if (
                                    not isinstance(raw, str)
                                    or len(raw.encode()) > MAX_REFERENCE_BYTES
                                ):
                                    raise ArchiveLimitError(
                                        "Workspace JSON metadata exceeds its size limit"
                                    )
                                _parse_json(raw)
            if version >= 14:
                try:
                    validate_taxonomy_records(connection)
                except ValueError as exc:
                    raise ArchiveError(
                        "Workspace taxonomy definitions or ownership are invalid"
                    ) from exc
            active = connection.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            for row in connection.execute("SELECT path,sha256,size_bytes FROM assets"):
                require(row["path"], ("assets/", "imports/"), row["sha256"], row["size_bytes"])
            for row in connection.execute("SELECT path FROM frames"):
                require(row["path"], ("frames/", "imports/"))
            for row in connection.execute("SELECT path,weight_sha256 FROM trained_models"):
                require(row["path"], ("models/trained/",), row["weight_sha256"])
            for row in connection.execute("SELECT id,path,sha256,summary FROM dataset_imports"):
                require(row["path"], (f"imports/{row['id']}/",), row["sha256"])
                summary = json.loads(row["summary"])
                for image in summary["images"]:
                    require(
                        image["source_path"],
                        (f"imports/{row['id']}/",),
                        image["source_sha256"],
                        image["size_bytes"],
                    )
                    require(image["path"], (f"imports/{row['id']}/",), image.get("png_sha256"))
            for row in connection.execute("SELECT id,path,manifest_sha256 FROM dataset_versions"):
                if row["path"] != f"datasets/{row['id']}/manifest.json":
                    raise ArchiveError("Dataset manifest path differs from its version directory")
                require(row["path"], (f"datasets/{row['id']}/",), row["manifest_sha256"])
                manifest, digest = _read_json(root / row["path"])
                if (
                    digest != row["manifest_sha256"]
                    or manifest.get("id") != row["id"]
                    or not isinstance(manifest.get("frames"), list)
                ):
                    raise ArchiveError("Dataset manifest no longer matches its frozen identity")
                for frame in manifest["frames"]:
                    if (
                        frame["image_path"]
                        != f"datasets/{row['id']}/images/{frame['frame_id']}.png"
                    ):
                        raise ArchiveError("Frozen image path differs from its dataset directory")
                    require(
                        frame["image_path"],
                        (f"datasets/{row['id']}/images/",),
                        frame["image_file_sha256"],
                    )
            if version >= 15:
                _validate_benchmarks(connection, root, require)
            if version >= 17:
                _validate_model_exports(connection, root, require)
            if version >= 18:
                _validate_training_checkpoints(connection, root, require)
            dinox_states = _validate_dinox(connection) if version >= 19 else {}
            for table, prefix in (
                ("assistance_previews", "assistance/previews"),
                ("video_reviews", "video_reviews"),
                ("experiment_reports", "reports"),
            ):
                for row in connection.execute(f"SELECT id,images FROM {table}"):
                    images = json.loads(row["images"])
                    if not isinstance(images, list):
                        raise ArchiveError("Workspace image previews must contain an image list")
                    for image in images:
                        require(
                            image["path"],
                            (f"{prefix}/{row['id']}/",),
                            image["sha256"],
                            image["size_bytes"],
                        )
            for row in connection.execute(
                "SELECT snapshot,snapshot_sha256 FROM experiment_reports"
            ):
                if (
                    hashlib.sha256(_canonical(_parse_json(row["snapshot"]))).hexdigest()
                    != row["snapshot_sha256"]
                ):
                    raise ArchiveError("Experiment report no longer matches its frozen checksum")
        for path in sorted(inventory):
            if (
                path.startswith("training_checkpoints/")
                and inventory[path]["size_bytes"] > MAX_TRAINING_CHECKPOINT_BYTES
            ):
                raise ArchiveLimitError("A training checkpoint exceeds its file size limit")
            if path == "models/sam3/sam3.pt":
                from iris.sam_provider import CHECKPOINT_SHA256, CHECKPOINT_SIZE

                require(path, ("models/sam3/",), CHECKPOINT_SHA256, CHECKPOINT_SIZE)
            elif path.startswith("models/") and path.endswith(".pth.json") and path.count("/") == 1:
                receipt, _ = _read_json(root / path)
                if not isinstance(receipt, dict) or not isinstance(
                    receipt.get("weight_filename"), str
                ):
                    raise ArchiveError("Model receipt is invalid")
                require(
                    "models/" + receipt["weight_filename"], ("models/",), receipt["weight_sha256"]
                )
            elif path.startswith("ollama/manifests/"):
                manifest, _ = _read_json(root / path)
                if (
                    not isinstance(manifest, dict)
                    or not isinstance(manifest.get("config"), dict)
                    or not isinstance(manifest.get("layers"), list)
                ):
                    raise ArchiveError("Ollama manifest is invalid")
                for descriptor in [manifest["config"], *manifest["layers"]]:
                    digest = descriptor["digest"]
                    if not isinstance(digest, str) or not digest.startswith("sha256:"):
                        raise ArchiveError("Ollama manifest uses an unsupported digest")
                    require(
                        "ollama/blobs/sha256-" + digest[7:],
                        ("ollama/blobs/",),
                        digest[7:],
                        descriptor.get("size"),
                    )
        if verify_hashes:
            for path, expected_digest in expected.items():
                source = root / path
                if source.is_symlink() or not source.resolve().is_relative_to(root):
                    raise ArchiveError("Referenced artifact is outside the workspace")
                with source.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest != expected_digest:
                    raise ArchiveError(
                        f"Workspace artifact no longer matches its recorded checksum: {path}"
                    )
        return {
            "schema_version": version,
            "counts": counts,
            "expected_hashes": expected,
            "active_jobs": active,
            "dinox_request_states": dinox_states,
        }
    except (
        sqlite3.Error,
        KeyError,
        TypeError,
        AttributeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ArchiveError(
            "Workspace database or artifact references are incomplete or unreadable"
        ) from exc
    finally:
        if connection is not None:
            connection.close()


def _required_bytes(total, database_bytes, count):
    return total + database_bytes + MAX_MANIFEST_BYTES + count * 512


def preview_workspace(root: Path) -> dict:
    root = Path(root).resolve()
    result = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "app_version": __version__,
        "counts": {},
        "file_count": 0,
        "total_bytes": 0,
        "categories": [],
        "excluded": [],
        "blocking_issues": [],
        "warnings": [
            "The archive contains private media, annotations, prompts and job logs. "
            "Keep it locally or share it deliberately.",
            "API keys and software environments are excluded and must be "
            "configured separately after restoration.",
        ],
        "can_create": False,
        "free_bytes": 0,
        "required_bytes": 0,
    }
    try:
        inventory, excluded = _inventory(root)
        checked = validate_database(root, inventory)
        connection = _readonly_database(root / "iris.sqlite3")
        try:
            inventory["iris.sqlite3"]["size_bytes"] = (
                connection.execute("PRAGMA page_count").fetchone()[0]
                * connection.execute("PRAGMA page_size").fetchone()[0]
            )
        finally:
            connection.close()
        groups = Counter()
        counts = Counter()
        for path, item in inventory.items():
            category = "database" if path == "iris.sqlite3" else path.split("/", 1)[0]
            groups[category] += item["size_bytes"]
            counts[category] += 1
        total = sum(groups.values())
        required = _required_bytes(total, inventory["iris.sqlite3"]["size_bytes"], len(inventory))
        free = shutil.disk_usage(root).free
        result.update(
            schema_version=checked["schema_version"],
            counts=checked["counts"],
            dinox_request_states=checked["dinox_request_states"],
            file_count=len(inventory),
            total_bytes=total,
            excluded=excluded,
            free_bytes=free,
            required_bytes=required,
            categories=[
                {
                    "id": key,
                    "label": _CATEGORIES[key],
                    "file_count": counts[key],
                    "size_bytes": groups[key],
                }
                for key in _CATEGORIES
                if counts[key]
            ],
        )
        if checked["active_jobs"]:
            result["blocking_issues"].append(
                "Wait for queued and running jobs to finish before backing up"
            )
        if free < required:
            result["blocking_issues"].append(
                "Not enough free disk space to prepare a complete archive"
            )
        result["can_create"] = not result["blocking_issues"]
    except (ArchiveError, OSError) as exc:
        result["blocking_issues"].append(
            str(exc) if isinstance(exc, ArchiveError) else "Workspace files could not be inspected"
        )
    return result


def _cancel(cancelled):
    if cancelled is not None and cancelled():
        raise ArchiveCancelled("Workspace archive cancelled")


class _DigestWriter(io.RawIOBase):
    def __init__(self, handle, cancelled):
        self.handle = handle
        self.cancelled = cancelled
        self.digest = hashlib.sha256()
        self.count = 0

    def writable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self.count

    def write(self, data):
        _cancel(self.cancelled)
        if self.count + len(data) > MAX_ARCHIVE_BYTES:
            raise ArchiveLimitError("Workspace archive exceeds the 64 GiB limit")
        written = self.handle.write(data)
        self.digest.update(memoryview(data)[:written])
        self.count += written
        return written

    def flush(self):
        if not self.handle.closed:
            self.handle.flush()


def create_archive(root: Path, destination: Path, *, progress=None, cancelled=None) -> dict:
    """Archive an idle workspace; callers must freeze mutations for the whole call."""
    root, destination = Path(root).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ArchiveError("The archive destination already exists; choose a new file")
    if destination.resolve().is_relative_to(root) and allowed_artifact_path(
        destination.resolve().relative_to(root).as_posix()
    ):
        raise ArchiveError("The archive must not replace a managed workspace artifact")
    _cancel(cancelled)
    inventory, _ = _inventory(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = _required_bytes(
        sum(item["size_bytes"] for item in inventory.values()),
        inventory["iris.sqlite3"]["size_bytes"],
        len(inventory),
    )
    if shutil.disk_usage(destination.parent).free < required:
        raise ArchiveLimitError("Not enough free disk space to prepare the workspace archive")
    source_db = _readonly_database(root / "iris.sqlite3")
    source_version = source_db.execute("PRAGMA data_version").fetchone()[0]
    temporary_path = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=".iris-archive-", dir=destination.parent
        ) as temporary_dir:
            snapshot_path = Path(temporary_dir) / "iris.sqlite3"
            copied = sqlite3.connect(snapshot_path)
            try:
                source_db.backup(copied, pages=256, progress=lambda *_: _cancel(cancelled))
                copied.execute("PRAGMA journal_mode=DELETE")
            finally:
                copied.close()
            checked = validate_database(root, inventory, database_path=snapshot_path)
            if checked["active_jobs"]:
                raise ArchiveError("Wait for queued and running jobs to finish before backing up")
            inventory["iris.sqlite3"]["size_bytes"] = snapshot_path.stat().st_size
            total = sum(item["size_bytes"] for item in inventory.values())
            if total > MAX_TOTAL_BYTES:
                raise ArchiveLimitError("Workspace exceeds the 64 GiB archive limit")
            if shutil.disk_usage(destination.parent).free < _required_bytes(
                total, 0, len(inventory)
            ):
                raise ArchiveLimitError(
                    "Not enough free disk space for the database snapshot and archive"
                )
            completed, file_rows = 0, []
            with tempfile.NamedTemporaryFile(
                prefix=".iris-archive-", suffix=".part", dir=destination.parent, delete=False
            ) as output:
                temporary_path = Path(output.name)
                writer = _DigestWriter(output, cancelled)
                with zipfile.ZipFile(
                    writer, "w", compression=zipfile.ZIP_STORED, allowZip64=True
                ) as archive:
                    for path, item in sorted(inventory.items()):
                        _cancel(cancelled)
                        source = snapshot_path if path == "iris.sqlite3" else root / path
                        before = source.lstat()
                        if not stat.S_ISREG(before.st_mode) or (
                            path != "iris.sqlite3" and _stat_signature(before) != item["signature"]
                        ):
                            raise ArchiveError(f"Workspace artifact changed before copying: {path}")
                        digest, count = hashlib.sha256(), 0
                        info = zipfile.ZipInfo(path)
                        info.compress_type = zipfile.ZIP_STORED
                        info.external_attr = (stat.S_IFREG | 0o600) << 16
                        info.file_size = item["size_bytes"]
                        with (
                            source.open("rb") as input_file,
                            archive.open(info, "w", force_zip64=True) as member,
                        ):
                            if _stat_signature(os.fstat(input_file.fileno())) != _stat_signature(
                                before
                            ):
                                raise ArchiveError("Workspace artifact changed while opening it")
                            while chunk := input_file.read(CHUNK_BYTES):
                                _cancel(cancelled)
                                count += len(chunk)
                                if count > item["size_bytes"]:
                                    raise ArchiveError("Workspace artifact grew while copying")
                                member.write(chunk)
                                digest.update(chunk)
                                completed += len(chunk)
                                if progress:
                                    progress(
                                        {
                                            "phase": "copying",
                                            "message": f"Saving {path}",
                                            "bytes_done": completed,
                                            "bytes_total": total,
                                            "files_done": len(file_rows),
                                            "files_total": len(inventory),
                                        }
                                    )
                        if count != item["size_bytes"] or _stat_signature(
                            source.lstat()
                        ) != _stat_signature(before):
                            raise ArchiveError(f"Workspace artifact changed while copying: {path}")
                        checksum = digest.hexdigest()
                        if (
                            path in checked["expected_hashes"]
                            and checksum != checked["expected_hashes"][path]
                        ):
                            raise ArchiveError(
                                "Workspace artifact no longer matches its recorded checksum: "
                                + path
                            )
                        file_rows.append({"path": path, "size_bytes": count, "sha256": checksum})
                    manifest = {
                        "protocol": PROTOCOL,
                        "format_version": FORMAT_VERSION,
                        "app_version": __version__,
                        "schema_version": checked["schema_version"],
                        "created_at": now(),
                        "file_count": len(file_rows),
                        "total_bytes": total,
                        "counts": checked["counts"],
                        "files": file_rows,
                        "exclusions": list(EXCLUSIONS),
                    }
                    validate_manifest(manifest)
                    archive.writestr("manifest.json", _canonical(manifest))
                latest, _ = _inventory(root)
                if (
                    set(latest) != set(inventory)
                    or latest["iris.sqlite3"]["signature"][:2]
                    != inventory["iris.sqlite3"]["signature"][:2]
                    or any(
                        latest[path]["signature"] != item["signature"]
                        for path, item in inventory.items()
                        if path != "iris.sqlite3"
                    )
                    or source_db.execute("PRAGMA data_version").fetchone()[0] != source_version
                ):
                    raise ArchiveError("Workspace changed during backup; start a new archive")
                _cancel(cancelled)
                output.flush()
                os.fsync(output.fileno())
                archive_digest, archive_size = writer.digest.hexdigest(), writer.count
            # Hard-link publication is atomic and refuses to replace an existing
            # file, including a destination created after the initial check.
            os.link(temporary_path, destination)
            temporary_path.unlink()
            if progress:
                progress(
                    {
                        "phase": "complete",
                        "message": "Workspace archive ready",
                        "bytes_done": total,
                        "bytes_total": total,
                        "files_done": len(file_rows),
                        "files_total": len(file_rows),
                    }
                )
            return {
                "path": destination,
                "manifest": manifest,
                "archive_size_bytes": archive_size,
                "archive_sha256": archive_digest,
            }
    except (sqlite3.Error, OSError) as exc:
        raise ArchiveError(
            "Could not create the workspace archive; check files and free disk space"
        ) from exc
    finally:
        source_db.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
