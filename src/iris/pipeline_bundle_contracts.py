"""Portable pipeline format and bounded ZIP inspection, without runtime execution.

This module is copied into the bundle for standard-library-only inspection. IRIS
always invokes its own installed validator and never imports code from an archive.
Checksums establish content integrity, not publisher authenticity or model quality.
"""

import hashlib
import json
import os
import re
import stat
import struct
import zipfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath

from .pipeline_detector_contracts import (
    ARCHITECTURES,
    YOLOX,
    _json_value,
    detector_contract,
    license_contract,
)
from .tracking_contracts import profile_hash, validate_profile
from .tracking_selection_contracts import canonicalize_source, validate_policy

FORMAT = "iris-pipeline-bundle-v1"
MAX_JSON_BYTES = 2 * 1024**2
MAX_CHECKPOINT_BYTES = 1024**3
MAX_ARCHIVE_BYTES = MAX_CHECKPOINT_BYTES + 32 * 1024**2
MAX_ENTRIES = 32
MAX_DIRECTORY_BYTES = 128 * 1024
INTERFACE = {
    "input": "original_oriented_image_with_source_frame_index_and_optional_timestamp_seconds",
    "image_color": "RGB_or_BGR_explicitly_converted_by_the_detector_recipe",
    "exif_orientation": "apply_before_coordinates_and_tracking",
    "coordinates": "xyxy_pixels_original_oriented_image_exclusive_right_bottom",
    "detector_outputs": "post_native_filtering_then_saved_score_floor",
    "class_namespace": "detector_output_mapping_output_id",
    "tracker_time_step": "one_update_per_available_frame",
    "source_frame_indices": "strictly_increasing_within_one_sequence",
    "timestamps": "source_seconds_or_null_never_inferred_capture_clock",
    "source_gaps": "no_synthetic_tracker_updates_clear_selection_confirmation",
    "sequence_boundary": "explicit_reset_all_tracker_and_selection_state",
    "image_dimensions": "constant_within_a_sequence",
    "gmc_image": "original_uint8_BGR_required_when_tracker_gmc_enabled",
    "observation": "current_measured_box_and_score",
    "prediction": "separate_estimate_never_a_measured_observation",
    "track_id": "local_association_number_not_physical_identity",
    "selection": "optional_guarded_geometry_requires_explicit_initial_measured_observation",
    "selection_events": "caller_owned_select_and_release_no_saved_live_target_in_bundle",
}
VALIDATION = {
    "status": "experimental",
    "package_integrity": "sha256_checked",
    "pipeline_execution": "not_run",
    "pipeline_parity": "not_run",
    "independent_quality": "not_qualified",
    "target_device_measurement": "not_run",
    "checkpoint_payload": "not_deserialized",
    "publisher_authenticity": "not_established",
}
_BASE_PATHS = {
    "detector/model.pth",
    "README.md",
    "inspect.py",
    "iris_bundle/__init__.py",
    "iris_bundle/pipeline_bundle_contracts.py",
    "iris_bundle/pipeline_detector_contracts.py",
    "iris_bundle/tracking_contracts.py",
    "iris_bundle/tracking_selection_contracts.py",
    "licenses/tracker-LICENSE",
    "licenses/detector-LICENSE",
    "licenses/NOTICE.txt",
}


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} must contain exactly its documented fields")
    return value


def _text(value, name, maximum=128):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} must be bounded nonempty text")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    return value


def _integer(value, name, minimum=0, maximum=2**53 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in its documented range")
    return value


def _hash(value, name, length=64):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")


def _timestamp(value):
    _text(value, "Creation timestamp", 80)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid creation timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Creation timestamp must include its timezone")


def canonical(value):
    _json_value(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_JSON_BYTES:
        raise ValueError("Pipeline JSON exceeds its bounded size")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Pipeline JSON has duplicate object keys")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Pipeline JSON has a nonfinite number: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        _json_value(value)
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError) as exc:
        raise ValueError("Pipeline JSON must be bounded finite UTF-8 data") from exc


def canonicalize_request(payload):
    _json_value(payload)
    _object(payload, {"name", "source", "selection_id", "target_device"}, "Pipeline bundle request")
    result = {
        "name": _text(payload["name"], "Bundle name", 160).strip(),
        "source": canonicalize_source(payload["source"]),
        "selection_id": None,
        "target_device": payload["target_device"],
    }
    if payload["selection_id"] is not None:
        result["selection_id"] = _text(payload["selection_id"], "Selection scenario ID")
    if result["target_device"] not in ("cpu", "cuda"):
        raise ValueError("Pipeline target device must be cpu or cuda")
    return result


def required_paths(architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError("Unsupported pipeline detector architecture")
    return _BASE_PATHS | ({"licenses/detector-NOTICE"} if architecture == YOLOX else set())


def _relative_path(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("Unsafe pipeline archive path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or str(path) != value
    ):
        raise ValueError("Unsafe pipeline archive path")
    return value


def _source(value, profile):
    _object(
        value,
        {
            "descriptor",
            "source_report_sha256",
            "sequence_sha256",
            "cache_fingerprint",
            "result_sha256",
            "replay_sha256",
            "first_pass_semantic_sha256",
            "inherited_dataset",
            "repeatability",
        },
        "Pipeline source provenance",
    )
    descriptor = canonicalize_source(value["descriptor"])
    if descriptor["profile_sha256"] != profile_hash(profile):
        raise ValueError("Pipeline profile must match its source descriptor")
    for key in (
        "source_report_sha256",
        "sequence_sha256",
        "cache_fingerprint",
        "result_sha256",
        "replay_sha256",
        "first_pass_semantic_sha256",
    ):
        _hash(value[key], key)
    inherited = value["inherited_dataset"]
    if (descriptor["kind"] == "study") != (inherited is not None):
        raise ValueError("Study sources require their inherited dataset; comparisons have none")
    if inherited is not None:
        _object(inherited, {"dataset_id", "manifest_sha256", "split"}, "Inherited source dataset")
        _text(inherited["dataset_id"], "Source dataset ID")
        _hash(inherited["manifest_sha256"], "Source dataset manifest hash")
        if inherited["split"] not in ("train", "val"):
            raise ValueError("Pipeline tuning sources cannot be reserved test data")
    if value["repeatability"] not in ("not_checked", "observed_match", "observed_mismatch"):
        raise ValueError("Unsupported source repeatability declaration")


def validate_manifest(manifest):
    """Validate the full portable contract without weights, ML imports or network."""
    _json_value(manifest)
    if len(canonical(manifest)) > MAX_JSON_BYTES:
        raise ValueError("Pipeline manifest exceeds its bounded size")
    _object(
        manifest,
        {
            "format",
            "id",
            "name",
            "created_at",
            "producer",
            "detector",
            "tracker",
            "selection",
            "interface",
            "source",
            "validation",
            "licenses",
            "files",
        },
        "Pipeline manifest",
    )
    if manifest["format"] != FORMAT:
        raise ValueError("Unsupported pipeline bundle format")
    _text(manifest["id"], "Bundle ID")
    _text(manifest["name"], "Bundle name", 160)
    _timestamp(manifest["created_at"])
    producer = _object(manifest["producer"], {"name", "version"}, "Bundle producer")
    if producer["name"] != "IRIS":
        raise ValueError("Unsupported bundle producer declaration")
    _text(producer["version"], "Producer version", 80)
    detector = _object(
        manifest["detector"],
        {"config", "checkpoint", "target_device", "precision", "output_mapping"},
        "Pipeline detector",
    )
    checkpoint = _object(
        detector["checkpoint"], {"path", "sha256", "size", "encoding"}, "Detector checkpoint"
    )
    expected_detector = detector_contract(
        detector["config"], detector["target_device"], checkpoint_size=checkpoint["size"]
    )
    if canonical(detector) != canonical(expected_detector):
        raise ValueError(
            "Pipeline detector contract differs from its frozen recipe or class mapping"
        )
    tracker = _object(
        manifest["tracker"],
        {"profile", "profile_sha256", "runtime", "runtime_sha256"},
        "Pipeline tracker",
    )
    profile = validate_profile(tracker["profile"])
    if tracker["profile_sha256"] != profile_hash(profile) or canonical(profile) != canonical(
        tracker["profile"]
    ):
        raise ValueError("Pipeline tracker profile hash or canonical fields differ")
    _metadata(tracker["runtime"], profile)
    if tracker["runtime_sha256"] != digest(tracker["runtime"]):
        raise ValueError("Pipeline tracker runtime fingerprint differs")
    config = detector["config"]
    if not set(profile["class_ids"]) <= {item["id"] for item in config["classes"]}:
        raise ValueError("Tracker classes must belong to the detector output namespace")
    if config["min_score"] > profile["low_threshold"]:
        raise ValueError("Detector score floor exceeds the tracker low threshold")
    selected = manifest["selection"]
    if selected is not None:
        _object(
            selected,
            {"algorithm", "policy", "policy_sha256", "source_job_id", "source_report_sha256"},
            "Optional selected-object policy",
        )
        if selected["algorithm"] != "guarded_geometry":
            raise ValueError("Only the guarded geometry selection policy is supported")
        policy = validate_policy(selected["policy"])
        if (
            canonical(policy) != canonical(selected["policy"])
            or digest(policy) != selected["policy_sha256"]
        ):
            raise ValueError("Selection policy hash or canonical fields differ")
        _text(selected["source_job_id"], "Selection source job ID")
        _hash(selected["source_report_sha256"], "Selection source report hash")
    if canonical(manifest["interface"]) != canonical(INTERFACE):
        raise ValueError("Unsupported pipeline coordinate, class or temporal interface")
    _source(manifest["source"], profile)
    if canonical(manifest["validation"]) != canonical(VALIDATION):
        raise ValueError("Pipeline bundle v1 must remain experimental with execution unmeasured")
    if canonical(manifest["licenses"]) != canonical(license_contract(config)):
        raise ValueError("Unsupported or inflated license and weight-rights declaration")
    if tracker["runtime"]["provenance"]["license"] != "MIT":
        raise ValueError("Supported tracker provenance must declare its MIT code license")
    files = _object(
        manifest["files"], required_paths(config["architecture"]), "Pipeline file inventory"
    )
    for path, identity in files.items():
        _relative_path(path)
        _object(identity, {"sha256", "size"}, "Pipeline file identity")
        _hash(identity["sha256"], "Pipeline file SHA-256")
        _integer(
            identity["size"],
            "Pipeline file bytes",
            0 if path == "iris_bundle/__init__.py" else 1,
            MAX_CHECKPOINT_BYTES if path == "detector/model.pth" else MAX_JSON_BYTES,
        )
    if files["detector/model.pth"] != {"sha256": checkpoint["sha256"], "size": checkpoint["size"]}:
        raise ValueError("Checkpoint inventory differs from the frozen detector weights")
    if (
        files["licenses/tracker-LICENSE"]["sha256"]
        != tracker["runtime"]["provenance"]["license_sha256"]
    ):
        raise ValueError("Tracker license bytes differ from the recorded tracker provenance")
    return deepcopy(manifest)


def _directory_preflight(stream, size):
    if not 22 <= size <= MAX_ARCHIVE_BYTES:
        raise ValueError("Pipeline archive exceeds its bounded size")
    stream.seek(size - 22)
    footer = stream.read(22)
    (
        signature,
        disk,
        directory_disk,
        disk_entries,
        entries,
        directory_size,
        directory_offset,
        comment_size,
    ) = struct.unpack("<4s4H2IH", footer)
    if (
        signature != b"PK\x05\x06"
        or comment_size
        or disk
        or directory_disk
        or disk_entries != entries
        or not 1 <= entries <= MAX_ENTRIES
        or directory_size > MAX_DIRECTORY_BYTES
        or directory_offset + directory_size != size - 22
    ):
        raise ValueError(
            "Unsupported or excessive pipeline ZIP central directory; "
            "ZIP64 and comments are not supported"
        )
    stream.seek(0)
    return entries, directory_offset


def inspect_bundle(path, expected_manifest=None, checkpoint=None):
    """Stream-verify a stored ZIP without extracting or executing any member."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Pipeline archive must not be a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("Pipeline archive is missing or unsafe") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode):
                raise ValueError("Pipeline archive must be a regular file")
            entries, directory_offset = _directory_preflight(stream, details.st_size)
            if checkpoint:
                checkpoint()
            with zipfile.ZipFile(stream, allowZip64=False) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if (
                    len(infos) != entries
                    or len(set(names)) != len(names)
                    or len({name.casefold() for name in names}) != len(names)
                ):
                    raise ValueError("Pipeline ZIP contains duplicate or case-colliding paths")
                if not infos or min(info.header_offset for info in infos) != 0:
                    raise ValueError(
                        "Pipeline ZIP must not contain a leading executable or other prefix"
                    )
                next_header = 0
                for info in sorted(infos, key=lambda value: value.header_offset):
                    if info.header_offset != next_header:
                        raise ValueError("Pipeline ZIP has hidden data between members")
                    stream.seek(info.header_offset)
                    header = stream.read(30)
                    (
                        signature,
                        version,
                        flags,
                        compression,
                        _,
                        _,
                        crc,
                        compressed,
                        size,
                        name_size,
                        extra_size,
                    ) = struct.unpack("<4s5H3I2H", header)
                    if (
                        signature != b"PK\x03\x04"
                        or version > 20
                        or flags & ~0x800
                        or flags != info.flag_bits
                        or compression != zipfile.ZIP_STORED
                        or extra_size
                        or compressed != info.compress_size
                        or size != info.file_size
                        or crc != info.CRC
                    ):
                        raise ValueError("Unsupported pipeline ZIP local member header")
                    next_header = info.header_offset + 30 + name_size + compressed
                if next_header != directory_offset:
                    raise ValueError("Pipeline ZIP has data outside its exact member inventory")
                total = 0
                for info in infos:
                    _relative_path(info.filename)
                    if (
                        info.is_dir()
                        or info.flag_bits & 1
                        or info.compress_type != zipfile.ZIP_STORED
                        or info.compress_size != info.file_size
                        or info.extra
                        or info.comment
                        or info.extract_version > 20
                        or (info.external_attr >> 16) & 0o170000 not in {0, stat.S_IFREG}
                    ):
                        raise ValueError(
                            "Pipeline ZIP must contain only plain stored regular files"
                        )
                    maximum = (
                        MAX_CHECKPOINT_BYTES
                        if info.filename == "detector/model.pth"
                        else MAX_JSON_BYTES
                    )
                    if not 0 <= info.file_size <= maximum:
                        raise ValueError("Pipeline ZIP member exceeds its bounded size")
                    total += info.file_size
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError("Pipeline ZIP total exceeds its bounded size")
                manifest_info = archive.getinfo("manifest.json")
                if not 1 <= manifest_info.file_size <= MAX_JSON_BYTES:
                    raise ValueError("Pipeline manifest is empty or exceeds its bounded size")
                manifest = validate_manifest(read_json(archive.read(manifest_info)))
                if set(names) != {"manifest.json", *manifest["files"]}:
                    raise ValueError("Pipeline ZIP inventory differs from its manifest")
                if expected_manifest is not None and canonical(manifest) != canonical(
                    expected_manifest
                ):
                    raise ValueError("Pipeline manifest differs from the saved export")
                for info in infos:
                    if checkpoint:
                        checkpoint()
                    if info.filename == "manifest.json":
                        continue
                    identity = manifest["files"][info.filename]
                    if info.file_size != identity["size"]:
                        raise ValueError("Pipeline member size differs from its manifest")
                    hashed, read_bytes = hashlib.sha256(), 0
                    with archive.open(info) as member:
                        while block := member.read(1024**2):
                            if checkpoint:
                                checkpoint()
                            read_bytes += len(block)
                            if read_bytes > identity["size"]:
                                raise ValueError("Pipeline member exceeded its declared size")
                            hashed.update(block)
                    if read_bytes != identity["size"] or hashed.hexdigest() != identity["sha256"]:
                        raise ValueError(
                            "Pipeline member bytes differ from their frozen fingerprint"
                        )
            stream.seek(0)
            archive_hash, archive_bytes = hashlib.sha256(), 0
            while block := stream.read(1024**2):
                if checkpoint:
                    checkpoint()
                archive_bytes += len(block)
                if archive_bytes > MAX_ARCHIVE_BYTES:
                    raise ValueError("Pipeline archive changed or exceeded its size limit")
                archive_hash.update(block)
            if (
                archive_bytes != details.st_size
                or os.fstat(stream.fileno()).st_mtime_ns != details.st_mtime_ns
                or os.fstat(stream.fileno()).st_ctime_ns != details.st_ctime_ns
            ):
                raise ValueError("Pipeline archive changed during inspection")
            return {
                "manifest": manifest,
                "manifest_sha256": digest(manifest),
                "archive_sha256": archive_hash.hexdigest(),
                "archive_bytes": archive_bytes,
            }
    except (
        zipfile.BadZipFile,
        KeyError,
        OverflowError,
        EOFError,
        struct.error,
    ) as exc:
        raise ValueError("Pipeline ZIP is invalid or incomplete") from exc


EXECUTION_POLICY = {
    "device": "cpu",
    "learned_reid": False,
    "class_association": "independent_native_instance_per_class",
    "native_frame_rate_argument": 30,
    "native_buffer_unit": "available_frame_updates",
    "native_time_step": 1,
    "skipped_source_frames": "no_synthetic_updates",
    "native_high_and_low_comparison": "strict_greater_than",
    "native_second_pass_match_threshold": 0.5,
    "native_unconfirmed_match_threshold": 0.7,
    "native_duplicate_iou_distance": 0.15,
    "native_expiry_order": "association_before_lost_cleanup_not_a_hard_ttl",
    "native_iou_geometry": "cython_bbox_inclusive_pixel_overlap",
    "gmc_scope": "one_sequence_estimate_shared_across_classes",
    "gmc_failure": "raise_and_require_reset_no_silent_fallback",
    "gmc_insufficient_matches": "native_identity_transform_reported",
    "opencv_rng": "seed_plus_update_index_before_each_update_no_rng_restore_api",
    "repeatability": "measure_semantic_hashes_on_repeated_fresh_replays",
}


def _provenance(value, algorithm):
    _object(
        value,
        {
            "repository",
            "commit",
            "license",
            "files",
            "license_sha256",
            "adaptations_sha256",
            "manifest_sha256",
            "adapter_sha256",
        },
        "Tracker source provenance",
    )
    for key in ("repository", "license"):
        _text(value[key], f"Tracker provenance {key}", 1000)
    _hash(value["commit"], "Tracker source commit", 40)
    for key in ("license_sha256", "adaptations_sha256", "manifest_sha256"):
        _hash(value[key], key)
    _object(
        value["adapter_sha256"],
        {"tracking.py", "tracking_contracts.py"},
        "Tracker adapter source hashes",
    )
    for digest in value["adapter_sha256"].values():
        _hash(digest, "Adapter source hash")
    names = {"__init__.py", "basetrack.py", "kalman_filter.py", "matching.py"}
    names |= {"byte_tracker.py"} if algorithm == "bytetrack" else {"bot_sort.py", "gmc.py"}
    _object(value["files"], names, "Vendored tracker source files")
    for name, item in value["files"].items():
        _object(item, {"path", "upstream_sha256", "vendored_sha256"}, "Vendored source provenance")
        _hash(item["vendored_sha256"], "Vendored source hash")
        if name == "__init__.py":
            if item["path"] is not None or item["upstream_sha256"] is not None:
                raise ValueError("Local tracker package initializer has no upstream source")
        else:
            _text(item["path"], "Upstream source path", 1000)
            _hash(item["upstream_sha256"], "Upstream source hash")


def _metadata(metadata, profile):
    _object(
        metadata,
        {"schema", "algorithm", "provenance", "python", "platform", "packages", "execution_policy"},
        "Pipeline tracker runtime metadata",
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != "iris-tracker-runtime-v1"
        or metadata.get("algorithm") != profile["algorithm"]
    ):
        raise ValueError("Pipeline tracker metadata must declare its runtime schema and algorithm")
    for key in ("python", "platform"):
        _text(metadata[key], f"Tracker runtime {key}", 2000)
    _object(
        metadata["packages"],
        {"scipy", "lap", "cython_bbox", "numpy", "opencv-python-headless"},
        "Tracker runtime packages",
    )
    for version in metadata["packages"].values():
        _text(version, "Historical package version", 100)
    _provenance(metadata["provenance"], profile["algorithm"])
    policy = metadata.get("execution_policy")
    expected_policy = {
        **EXECUTION_POLICY,
        "opencv_threads": profile["opencv_threads"],
        "seed": profile["seed"],
        "unconfirmed_returned": profile["algorithm"] == "botsort",
    }
    _object(policy, set(expected_policy) | {"blas_environment"}, "Tracker execution policy")
    for key, expected in expected_policy.items():
        if type(policy.get(key)) is not type(expected) or policy[key] != expected:
            raise ValueError(f"Pipeline tracker execution policy changed: {key}")
    environment = policy["blas_environment"]
    _object(
        environment,
        {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"},
        "Tracker BLAS environment",
    )
    for value in environment.values():
        if value is not None and (not isinstance(value, str) or len(value) > 2000):
            raise ValueError("Historical BLAS environment must be text or null")
