"""Bounded, deterministic COCO archives made only from frozen dataset snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tempfile
import warnings
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from iris.annotations import MAX_BOXES, TAXONOMY, _coordinates
from iris.datasets import CLASS_MAPPING, MAX_FRAMES, SCHEMA_VERSION, _canonical
from iris.media import _pixel_hash
from iris.store import DEFAULT_PROJECT_ID, Store

PROTOCOL = "iris-coco-export-v1"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SPLIT_ORDER = ("train", "val", "test")
_ID = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

README = """IRIS frozen dataset: COCO bounding boxes

Each split has <split>/annotations.json and <split>/images/<frame-id>.png.
Image file_name values are relative to the split directory. Set the consumer's
image root to train/, val/ or test/ accordingly. An empty test split is retained.
Categories are COCO person=1 and car=3, not IRIS training-head IDs person=1/car=2.
Boxes are [x, y, width, height] in original image pixels, with positive area;
width=x2-x1 and height=y2-y1, without inclusive-pixel adjustments or rounding.
No crowd, ignore, segmentation or keypoint labels are inferred. Validated empty
images are explicit negative examples and remain in the corresponding split.

The frozen release's train/val/test and scene-group assignments are unchanged.
Distinct scene groups do not prove independence. Validation is not a test split.
iris-manifest.json contains the exact original frozen manifest. Its image_path
values describe the original workspace; use export.json for archive paths and
COCO IDs. export.json records SHA-256 and sizes for every other archive member;
it omits its own hash. Image file hashes and RGB pixel hashes have different roles.

The manifest retains reviewer names, notes, annotation decisions and recorded
source provenance, including source URLs, license descriptions and attribution
where present. This metadata is not redacted or a guarantee of redistribution
rights. Preserve the original terms and attribution when sharing these files.
No source video, model checkpoint, workspace database or configuration file is
included. This archive is a dataset export, not a complete workspace backup.
It is not directly accepted by IRIS's current one-JSON, one-scene-group importer.
"""


class ExportLimitError(ValueError):
    """A valid release exceeds the deliberately bounded local export limits."""


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_bounded(path: Path, limit: int, description: str) -> bytes:
    try:
        with path.open("rb") as source:
            if path.stat().st_size > limit:
                raise ExportLimitError(f"{description} exceeds the export size limit")
            raw = source.read(limit + 1)
    except OSError as exc:
        raise ValueError(f"{description} is missing or unreadable") from exc
    if len(raw) > limit:
        raise ExportLimitError(f"{description} exceeds the export size limit")
    return raw


def _identifier(value, description: str):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"Frozen {description} must be an IRIS identifier")


def _hash(value, description: str):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"Frozen {description} must be a SHA-256 hash")


def _text(value, description: str, limit: int):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"Frozen {description} is invalid")


def _validate(manifest: dict, row: dict):
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest.get("id") != row["id"]
        or manifest.get("taxonomy") != TAXONOMY
        or manifest.get("class_mapping") != CLASS_MAPPING
    ):
        raise ValueError("Frozen manifest has an unsupported format or taxonomy")
    if manifest.get("project_id", DEFAULT_PROJECT_ID) != row.get("project_id", DEFAULT_PROJECT_ID):
        raise ValueError("Frozen manifest belongs to a different project")
    _text(manifest.get("name"), "dataset name", 160)
    frames, splits = manifest.get("frames"), manifest.get("splits")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"Frozen dataset must contain 1 to {MAX_FRAMES} frames")
    if not isinstance(splits, dict) or any(
        not isinstance(group, str) or not group.strip() or split not in SPLIT_ORDER
        for group, split in splits.items()
    ):
        raise ValueError("Frozen scene-group splits are invalid")
    seen_ids, seen_pixels, groups, used_splits = set(), set(), set(), set()
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Frozen frames must be objects")
        identifier = frame.get("frame_id")
        _identifier(identifier, "frame ID")
        if not isinstance(frame.get("image_path"), str) or not frame["image_path"]:
            raise ValueError("Frozen image path must be a nonempty relative path")
        _hash(frame.get("sha256"), "pixel hash")
        _hash(frame.get("image_file_sha256"), "image file hash")
        if identifier in seen_ids or frame["sha256"] in seen_pixels:
            raise ValueError("Frozen dataset contains duplicate frame IDs or image pixels")
        seen_ids.add(identifier)
        seen_pixels.add(frame["sha256"])
        for dimension in ("width", "height"):
            if type(frame.get(dimension)) is not int or frame[dimension] <= 0:
                raise ValueError("Frozen image dimensions must be positive integers")
        if frame["width"] * frame["height"] > MAX_IMAGE_PIXELS:
            raise ExportLimitError("Frozen image exceeds the 20 megapixel export limit")
        group, split = frame.get("scene_group"), frame.get("split")
        if (
            not isinstance(group, str)
            or not group.strip()
            or split not in SPLIT_ORDER
            or splits.get(group) != split
        ):
            raise ValueError("Frozen frame and scene-group splits disagree")
        groups.add(group)
        used_splits.add(split)
        annotation, boxes = frame.get("annotation"), frame.get("boxes")
        if (
            not isinstance(annotation, dict)
            or annotation.get("status") != "validated"
            or annotation.get("taxonomy_id") != TAXONOMY["id"]
            or annotation.get("frame_id") != identifier
            or annotation.get("frame_sha256") != frame["sha256"]
            or annotation.get("id") != frame.get("annotation_revision_id")
            or type(frame.get("revision")) is not int
            or frame["revision"] <= 0
            or type(annotation.get("revision")) is not int
            or annotation["revision"] != frame["revision"]
            or annotation.get("boxes") != boxes
        ):
            raise ValueError("Frozen annotation must match its validated revision and image")
        _identifier(frame.get("annotation_revision_id"), "annotation revision ID")
        _text(annotation.get("reviewer"), "reviewer", 120)
        if not isinstance(boxes, list) or len(boxes) > MAX_BOXES:
            raise ValueError("Frozen boxes exceed the supported annotation limit")
        box_ids = set()
        for box in boxes:
            if not isinstance(box, dict):
                raise ValueError("Frozen boxes must be objects")
            _text(box.get("id"), "box ID", 128)
            if box["id"] in box_ids or box.get("label") not in CLASS_MAPPING:
                raise ValueError("Frozen boxes have duplicate IDs or unsupported labels")
            box_ids.add(box["id"])
            for flag in ("iscrowd", "ignore"):
                if flag in box and (type(box[flag]) not in (int, bool) or box[flag] != 0):
                    raise ValueError("Frozen crowd or ignore boxes are unsupported")
            x1, y1, x2, y2 = _coordinates(box.get("box"), frame)
            area = (x2 - x1) * (y2 - y1)
            if not math.isfinite(area) or area <= 0:
                raise ValueError("Frozen box area must be finite and positive")
    if groups != set(splits) or not {"train", "val"} <= used_splits:
        raise ValueError("Frozen dataset requires matching groups and nonempty train and val")


def _snapshot(store: Store, dataset_id: str) -> tuple[dict, bytes, str]:
    row = store.get("dataset_versions", dataset_id)
    if row is None:
        raise KeyError(dataset_id)
    _identifier(row["id"], "dataset ID")
    if not isinstance(row.get("path"), str) or not row["path"]:
        raise ValueError("Frozen manifest path must be a nonempty relative path")
    version_dir = store.artifact_path(f"datasets/{row['id']}")
    path = store.artifact_path(row["path"])
    if path != version_dir / "manifest.json":
        raise ValueError("Frozen manifest is outside its version directory")
    raw = _read_bounded(path, MAX_MANIFEST_BYTES, "Frozen manifest")
    digest = _digest(raw)
    if digest != row["manifest_sha256"]:
        raise ValueError("Frozen manifest no longer matches its recorded hash")
    try:
        manifest = json.loads(raw)
        # This also rejects non-finite numbers in otherwise unused provenance.
        _canonical(manifest)
        _validate(manifest, row)
    except (KeyError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("Frozen manifest has invalid data") from exc
    for frame in manifest["frames"]:
        expected = version_dir / "images" / f"{frame['frame_id']}.png"
        if store.artifact_path(frame["image_path"]) != expected:
            raise ValueError("Frozen image is outside its version directory")
    return manifest, raw, digest


def _image_bytes(store: Store, frame: dict) -> bytes:
    raw = _read_bounded(store.artifact_path(frame["image_path"]), MAX_IMAGE_BYTES, "Frozen image")
    if _digest(raw) != frame["image_file_sha256"]:
        raise ValueError("Frozen image no longer matches its recorded file hash")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if (
                    source.format != "PNG"
                    or getattr(source, "n_frames", 1) != 1
                    or source.size != (frame["width"], frame["height"])
                ):
                    raise ValueError("Frozen image format or dimensions do not match")
                with source.convert("RGB") as image:
                    image.load()
                    if _pixel_hash(image) != frame["sha256"]:
                        raise ValueError("Frozen image no longer matches its recorded pixel hash")
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Frozen image is invalid or unreadable") from exc
    return raw


class _Archive:
    def __init__(self, archive: zipfile.ZipFile):
        self.archive = archive
        self.files = {}
        # Fixed ASCII paths, ZIP_STORED, no comments, extras or ZIP64: exact size.
        self.size = 22

    def add(self, name: str, raw: bytes):
        self.size += len(raw) + 76 + 2 * len(name.encode("ascii"))
        if self.size > MAX_ARCHIVE_BYTES:
            raise ExportLimitError("COCO archive exceeds the 256 MiB export limit")
        info = zipfile.ZipInfo(name, _ZIP_DATE)
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        self.archive.writestr(info, raw)
        self.files[name] = {"sha256": _digest(raw), "size_bytes": len(raw)}


def build_coco_export(store: Store, dataset_id: str) -> Path:
    """Return a completed temporary ZIP; the caller must remove it after delivery.

    All bytes come from the immutable snapshot, including reviewer/source metadata.
    Nothing is read from current annotations, original media, credentials or models.
    """
    manifest, raw_manifest, manifest_sha256 = _snapshot(store, dataset_id)
    categories = [{"id": item["coco_id"], "name": item["id"]} for item in TAXONOMY["classes"]]
    category_ids = {item["name"]: item["id"] for item in categories}
    documents = {
        split: {
            "info": {
                "description": manifest["name"],
                "version": PROTOCOL,
                "iris_dataset_id": dataset_id,
                "iris_taxonomy_id": TAXONOMY["id"],
                "split": split,
            },
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        for split in SPLIT_ORDER
    }
    frames = sorted(manifest["frames"], key=lambda frame: frame["frame_id"])
    mappings, annotation_id = [], 0
    for image_id, frame in enumerate(frames, 1):
        split = frame["split"]
        filename = f"images/{frame['frame_id']}.png"
        documents[split]["images"].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": frame["width"],
                "height": frame["height"],
                "iris_frame_id": frame["frame_id"],
                "iris_scene_group": frame["scene_group"],
                "split": split,
            }
        )
        mappings.append(
            {
                "coco_image_id": image_id,
                "iris_frame_id": frame["frame_id"],
                "split": split,
                "archive_path": f"{split}/{filename}",
                "image_file_sha256": frame["image_file_sha256"],
                "pixel_sha256": frame["sha256"],
                "annotation_revision_id": frame["annotation_revision_id"],
            }
        )
        for box in sorted(frame["boxes"], key=lambda box: box["id"]):
            annotation_id += 1
            x1, y1, x2, y2 = _coordinates(box["box"], frame)
            width, height = x2 - x1, y2 - y1
            documents[split]["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[box["label"]],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "iris_box_id": box["id"],
                }
            )
    export_dir = store.artifact_path("exports")
    export_dir.mkdir(exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"coco-{dataset_id}-", suffix=".zip", dir=export_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with zipfile.ZipFile(temporary, "w", allowZip64=False) as archive:
                output = _Archive(archive)
                for split in SPLIT_ORDER:
                    output.add(f"{split}/annotations.json", _canonical(documents[split]))
                    for frame in frames:
                        if frame["split"] == split:
                            output.add(
                                f"{split}/images/{frame['frame_id']}.png",
                                _image_bytes(store, frame),
                            )
                output.add("iris-manifest.json", raw_manifest)
                output.add("README.txt", README.encode())
                metadata = {
                    "protocol": PROTOCOL,
                    "dataset_id": dataset_id,
                    "taxonomy_id": TAXONOMY["id"],
                    "categories": categories,
                    "box_format": "xywh_pixels",
                    "manifest": {"path": "iris-manifest.json", "sha256": manifest_sha256},
                    "splits": {
                        split: {
                            "annotations_path": f"{split}/annotations.json",
                            "image_count": len(documents[split]["images"]),
                            "annotation_count": len(documents[split]["annotations"]),
                        }
                        for split in SPLIT_ORDER
                    },
                    "images": mappings,
                    "files": dict(output.files),
                }
                output.add("export.json", _canonical(metadata))
            if temporary.tell() != output.size:
                raise ValueError("COCO archive size does not match its recorded layout")
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
