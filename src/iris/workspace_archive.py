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
    JSON_FIELDS,
    SCHEMA,
    SCHEMA_V12,
    SCHEMA_V13,
    SCHEMA_V14,
    SCHEMA_V15,
    SCHEMA_VERSION,
    TABLES,
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
    SCHEMA_VERSION: SCHEMA,
}
SCHEMA_TABLES = {
    12: TABLES - BENCHMARK_TABLES - {"projects", "taxonomy_versions"},
    13: TABLES - BENCHMARK_TABLES - {"taxonomy_versions"},
    14: TABLES - BENCHMARK_TABLES,
    15: TABLES - {"benchmark_reports"},
    SCHEMA_VERSION: TABLES,
}
CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024**3
MAX_TOTAL_BYTES = 64 * 1024**3
MAX_FILES = 100000
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_REFERENCE_BYTES = 64 * 1024**2
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CATEGORIES = {
    "database": "Workspace database",
    "assets": "Original media",
    "frames": "Extracted images",
    "imports": "Imported datasets",
    "datasets": "Frozen dataset versions",
    "benchmarks": "Frozen benchmark references and images",
    "models": "Detector checkpoints",
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
    external_trials, sam_trials, combined_trials = {}, {}, {}
    for row in connection.execute(
        "SELECT t.*,c.benchmark_id AS config_benchmark_id,j.kind,j.params,j.result AS job_result "
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
    for external in (*external_trials.values(), *combined_trials.values()):
        if (
            external["frames"] != {request["frame_id"] for request in external["plan"]["requests"]}
            or external["reserved"] > external["plan"]["approval"]["budget_microusd"]
        ):
            raise ArchiveError("External benchmark request coverage or budget is inconsistent")
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
