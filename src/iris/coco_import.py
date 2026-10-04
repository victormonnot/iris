"""Bounded COCO bounding-box imports, with explicit mapping and human review.

A ZIP contains exactly one JSON document and the images named in that document.
Image paths resolve from the archive root or the JSON document's directory; two
different matches are rejected. Nothing in the document is fetched or executed.
"""

from __future__ import annotations

import json
import math
import shutil
import stat
import warnings
import zipfile
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

from iris.media import _file_hash, _perceptual_hash, _pixel_hash
from iris.store import DEFAULT_PROJECT_ID, Store, _decode, _encode, new_id, now
from iris.taxonomies import TAXONOMY, current_taxonomy, get_taxonomy

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_IMAGES = 100
MAX_IMAGE_PIXELS = 20_000_000
MAX_TOTAL_PIXELS = 100_000_000
MAX_ANNOTATIONS_PER_IMAGE = 500
MAX_ARCHIVE_ENTRIES = 512
SPLITS = {"train", "val", "test"}
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "BMP", "TIFF")


def _identifier(value, description: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"{description} must be a nonnegative integer no larger than 2^53 - 1.")
    return value


def _number(value, description: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{description} must be a finite number.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{description} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{description} must be a finite number.")
    return result


def _text(value, description: str, limit: int, *, multiline: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{description} must be nonempty text, at most {limit} characters.")
    allowed_controls = "\n\r\t" if multiline else ""
    if any(ord(character) < 32 and character not in allowed_controls for character in value):
        raise ValueError(f"{description} contains unsupported control characters.")
    return value.strip()


def _member_path(value: str, *, directory: bool = False) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError("Archive and image paths must be nonempty local relative paths.")
    path = value[:-1] if directory and value.endswith("/") else value
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or ":" in path
        or any(ord(character) < 32 for character in path)
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("Unsafe archive path: use local relative paths without traversal or URLs.")
    return path


def _archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    entries = archive.infolist()
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        raise ValueError(f"The archive contains more than {MAX_ARCHIVE_ENTRIES} entries.")
    members, seen = {}, set()
    total_bytes = 0
    for entry in entries:
        original_name = entry.orig_filename
        name = _member_path(original_name, directory=entry.is_dir())
        if name in seen:
            raise ValueError("The archive contains duplicate or conflicting paths.")
        seen.add(name)
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("Archive symlinks and special files are unsupported.")
        if (mode == stat.S_IFDIR and not entry.is_dir()) or (
            mode == stat.S_IFREG and entry.is_dir()
        ):
            raise ValueError("The archive contains conflicting file and directory types.")
        if entry.flag_bits & 1:
            raise ValueError("Encrypted archives are unsupported.")
        if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ValueError("Use a stored or Deflate ZIP archive.")
        total_bytes += entry.file_size
        if total_bytes > MAX_EXPANDED_BYTES:
            raise ValueError("The expanded archive exceeds the 256 MiB limit.")
        if not entry.is_dir():
            members[name] = entry
    for name in members:
        if any(parent.as_posix() in members for parent in PurePosixPath(name).parents):
            raise ValueError("The archive contains conflicting file and directory paths.")
    return members


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}.")


def _read_document(archive: zipfile.ZipFile, members: dict) -> tuple[str, dict]:
    names = [name for name in members if PurePosixPath(name).suffix.lower() == ".json"]
    if len(names) != 1:
        raise ValueError("Include exactly one COCO JSON document per archive.")
    filename = names[0]
    if members[filename].file_size > MAX_JSON_BYTES:
        raise ValueError("The COCO JSON document exceeds the 16 MiB limit.")
    with archive.open(members[filename]) as source:
        raw = source.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("The COCO JSON document exceeds the 16 MiB limit.")
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_invalid_constant,
            parse_float=lambda value: _number(float(value), "JSON number"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The COCO JSON document is invalid or excessively nested.") from exc
    if not isinstance(document, dict):
        raise ValueError("The COCO JSON document must be an object.")
    for field in ("images", "annotations", "categories"):
        if not isinstance(document.get(field), list):
            raise ValueError(f"COCO {field} must be an array.")
    if not 1 <= len(document["images"]) <= MAX_IMAGES:
        raise ValueError(f"Include between 1 and {MAX_IMAGES} images per import.")
    if not isinstance(document.get("info", {}), dict):
        raise ValueError("COCO info must be an object.")
    if not isinstance(document.get("licenses", []), list):
        raise ValueError("COCO licenses must be an array.")
    return filename, document


def _embedded_split(document: dict) -> str | None:
    values = set()
    for record in (document, document.get("info", {}), *document["images"]):
        if not isinstance(record, dict):
            raise ValueError("Every COCO image must be an object.")
        for key in ("split", "source_split", "subset"):
            value = record.get(key)
            if value is not None:
                if not isinstance(value, str) or value not in SPLITS:
                    raise ValueError("Embedded splits must be train, val, or test.")
                values.add(value)
    if len(values) > 1:
        raise ValueError("Mixed source splits are unsupported; import each split separately.")
    return next(iter(values), None)


def _not_ignored(record: dict, description: str):
    for flag in ("iscrowd", "ignore"):
        value = record.get(flag, 0)
        if type(value) not in (int, bool) or value not in (0, 1):
            raise ValueError(f"{description} {flag} must be 0 or 1.")
        if value:
            raise ValueError(
                "Crowd and ignore regions are unsupported. Import a subset without these "
                "images; removing only the flagged annotations would change evaluation semantics."
            )


def _structure(document: dict) -> tuple[dict, dict, dict]:
    categories, images, annotations = {}, {}, {}
    for category in document["categories"]:
        if not isinstance(category, dict):
            raise ValueError("Every COCO category must be an object.")
        category_id = _identifier(category.get("id"), "Category ID")
        _text(category.get("name"), "Category name", 256)
        if category_id in categories:
            raise ValueError("Duplicate COCO category ID.")
        categories[category_id] = category
    total_pixels = 0
    for image in document["images"]:
        image_id = _identifier(image.get("id"), "Image ID")
        if image_id in images:
            raise ValueError("Duplicate COCO image ID.")
        _not_ignored(image, "Image")
        width = _identifier(image.get("width"), "Image width")
        height = _identifier(image.get("height"), "Image height")
        if not width or not height or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions must be positive and at most 20 megapixels.")
        total_pixels += width * height
        if total_pixels > MAX_TOTAL_PIXELS:
            raise ValueError("The import exceeds the 100 megapixel total limit.")
        _member_path(image.get("file_name"))
        images[image_id] = image
        annotations[image_id] = []
    seen = set()
    for annotation in document["annotations"]:
        if not isinstance(annotation, dict):
            raise ValueError("Every COCO annotation must be an object.")
        annotation_id = _identifier(annotation.get("id"), "Annotation ID")
        if annotation_id in seen:
            raise ValueError("Duplicate COCO annotation ID.")
        seen.add(annotation_id)
        image_id = _identifier(annotation.get("image_id"), "Annotation image ID")
        category_id = _identifier(annotation.get("category_id"), "Annotation category ID")
        if image_id not in images or category_id not in categories:
            raise ValueError("An annotation references an unknown image or category.")
        _not_ignored(annotation, "Annotation")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Each COCO bbox must contain [x, y, width, height].")
        x, y, width, height = [_number(value, "Bounding-box coordinate") for value in bbox]
        image = images[image_id]
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > image["width"]
            or y + height > image["height"]
        ):
            raise ValueError("COCO bounding boxes must be positive and inside the image.")
        if "area" in annotation and _number(annotation["area"], "Annotation area") < 0:
            raise ValueError("Annotation area cannot be negative.")
        annotations[image_id].append(annotation)
        if len(annotations[image_id]) > MAX_ANNOTATIONS_PER_IMAGE:
            raise ValueError("An image has more than 500 annotations.")
    return categories, images, annotations


def _resolve_image(filename: str, json_filename: str, members: dict) -> str:
    filename = _member_path(filename)
    candidates = {filename, (PurePosixPath(json_filename).parent / filename).as_posix()}
    matches = candidates & members.keys()
    if len(matches) != 1:
        raise ValueError(
            f"Image {filename!r} is missing or ambiguous. Paths must resolve uniquely "
            "from the archive root or the COCO JSON directory."
        )
    return matches.pop()


def _copy_bounded(source, destination: Path, limit: int):
    total = 0
    with destination.open("xb") as output:
        while chunk := source.read(min(1024 * 1024, limit - total + 1)):
            total += len(chunk)
            if total > limit:
                raise ValueError("The source exceeds its import size limit.")
            output.write(chunk)
    return total


def _normalize_image(source: Path, output: Path, expected: dict) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # EPS and other document plugins can launch external renderers. Decode
            # only ordinary raster formats; extensions are not trusted.
            with Image.open(source, formats=IMAGE_FORMATS) as original:
                if original.size != (expected["width"], expected["height"]):
                    raise ValueError("COCO dimensions do not match the actual image dimensions.")
                if original.width * original.height > MAX_IMAGE_PIXELS:
                    raise ValueError("The image exceeds the 20 megapixel limit.")
                if getattr(original, "n_frames", 1) != 1:
                    raise ValueError("Animated images are unsupported for COCO import.")
                if original.getexif().get(274, 1) != 1:
                    raise ValueError(
                        "EXIF-rotated images are unsupported: normalize pixels and annotation "
                        "coordinates before importing."
                    )
                image = original.convert("RGB")
                image.load()
                image.save(output, format="PNG")
                return {
                    "format": original.format,
                    "sha256": _pixel_hash(image),
                    "perceptual_hash": _perceptual_hash(image),
                    "png_sha256": _file_hash(output),
                }
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(
            "A COCO image is unsupported, corrupt, unreadable, or oversized. "
            "Use PNG, JPEG, WebP, BMP, or TIFF raster images."
        ) from exc


def preview_import(
    store: Store, source: Path, filename: str, *, project_id: str = DEFAULT_PROJECT_ID
) -> dict:
    """Validate and retain a local archive without creating reviewable frames yet."""
    if store.get("projects", project_id) is None:
        raise ValueError("Project does not exist")
    taxonomy = current_taxonomy(store, project_id)
    source = Path(source)
    if not source.is_file() or not 0 < source.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ValueError("Provide a nonempty ZIP archive of at most 64 MiB.")
    filename = _text(str(filename).replace("\\", "/").rsplit("/", 1)[-1], "Filename", 255)
    import_id = new_id()
    directory = store.artifact_path(f"imports/{import_id}")
    directory.mkdir(parents=True)
    archive_path = directory / "source.zip"
    try:
        with source.open("rb") as original:
            _copy_bounded(original, archive_path, MAX_ARCHIVE_BYTES)
        with zipfile.ZipFile(archive_path) as archive:
            members = _archive_members(archive)
            json_filename, document = _read_document(archive, members)
            source_split = _embedded_split(document)
            categories, images, annotations = _structure(document)
            normalized, resolved_names, source_hashes = [], set(), set()
            counts = Counter(annotation["category_id"] for annotation in document["annotations"])
            for image_id, coco_image in images.items():
                resolved = _resolve_image(coco_image["file_name"], json_filename, members)
                if resolved in resolved_names:
                    raise ValueError("Multiple COCO images reference the same archive file.")
                resolved_names.add(resolved)
                image_key = new_id()
                original_path = directory / f"{image_key}.source"
                png_path = directory / f"{image_key}.png"
                with archive.open(members[resolved]) as original:
                    size = _copy_bounded(original, original_path, MAX_EXPANDED_BYTES)
                source_hash = _file_hash(original_path)
                if source_hash in source_hashes:
                    raise ValueError("The archive contains byte-identical duplicate images.")
                source_hashes.add(source_hash)
                properties = _normalize_image(original_path, png_path, coco_image)
                normalized.append(
                    {
                        "id": str(image_id),
                        "coco_image_id": image_id,
                        "filename": coco_image["file_name"],
                        "width": coco_image["width"],
                        "height": coco_image["height"],
                        "annotation_count": len(annotations[image_id]),
                        "coco_image": coco_image,
                        "original_annotations": annotations[image_id],
                        "source_path": original_path.relative_to(store.root).as_posix(),
                        "source_sha256": source_hash,
                        "size_bytes": size,
                        "path": png_path.relative_to(store.root).as_posix(),
                        **properties,
                    }
                )
            notices = [
                "Imported labels are proposals. Every image, including images without boxes, "
                "requires human review before dataset publication.",
                "One import creates one scene group. Keep related captures in the same group; "
                "the importer cannot establish independence between scenes.",
                "COCO segmentation is preserved as source metadata; only bounding boxes are used.",
            ]
            if len({image["sha256"] for image in normalized}) < len(normalized):
                notices.append(
                    "Some images have identical normalized pixels. A dataset version cannot "
                    "include duplicate pixels; deselect duplicates before publication."
                )
            summary = {
                "taxonomy": taxonomy,
                "filename": filename,
                "json_filename": json_filename,
                "image_count": len(images),
                "annotation_count": len(document["annotations"]),
                "categories": [
                    {"id": key, "name": category["name"], "count": counts[key]}
                    for key, category in categories.items()
                ],
                "original_categories": list(categories.values()),
                "images": normalized,
                "info": document.get("info", {}),
                "licenses": document.get("licenses", []),
                "source_split": source_split,
                "warnings": notices,
            }
        store.insert(
            "dataset_imports",
            {
                "id": import_id,
                "project_id": project_id,
                "path": archive_path.relative_to(store.root).as_posix(),
                "sha256": _file_hash(archive_path),
                "summary": summary,
                "metadata": {},
                "result": None,
                "created_at": now(),
            },
        )
    except BaseException as exc:
        shutil.rmtree(directory, ignore_errors=True)
        if isinstance(exc, (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, zlib.error)):
            raise ValueError("The ZIP archive is invalid or corrupt.") from exc
        raise
    return import_detail(store, import_id)


def import_detail(store: Store, import_id: str) -> dict:
    row = store.get("dataset_imports", import_id)
    if row is None:
        raise KeyError(import_id)
    summary = row["summary"]
    categories = {category["id"]: category["name"] for category in summary["categories"]}
    public_images = []
    for image in summary["images"]:
        boxes = []
        for annotation in image["original_annotations"]:
            x, y, width, height = annotation["bbox"]
            boxes.append(
                {
                    "annotation_id": annotation["id"],
                    "category_id": annotation["category_id"],
                    "category_name": categories[annotation["category_id"]],
                    "box": [x, y, x + width, y + height],
                }
            )
        public_images.append(
            {
                **{
                    key: image[key]
                    for key in (
                        "id",
                        "coco_image_id",
                        "filename",
                        "width",
                        "height",
                        "annotation_count",
                    )
                },
                "boxes": boxes,
                "image_url": f"/api/dataset-imports/{import_id}/images/{image['id']}",
            }
        )
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "taxonomy": _import_taxonomy(store, row),
        "sha256": row["sha256"],
        "created_at": row["created_at"],
        "status": "imported" if row["result"] is not None else "preview",
        **{
            key: summary[key]
            for key in (
                "filename",
                "image_count",
                "annotation_count",
                "categories",
                "info",
                "licenses",
                "warnings",
                "source_split",
            )
        },
        "images": public_images,
        "config": row["metadata"].get("config"),
        "result": row["result"],
    }


def preview_image_path(store: Store, import_id: str, image_id: str) -> Path:
    row = store.get("dataset_imports", import_id)
    if row is None:
        raise KeyError(import_id)
    image = next((item for item in row["summary"]["images"] if item["id"] == image_id), None)
    if image is None:
        raise KeyError(image_id)
    path = store.artifact_path(image["path"])
    if not path.is_file() or _file_hash(path) != image["png_sha256"]:
        raise RuntimeError("The imported preview image is missing or has changed.")
    return path


def _import_taxonomy(store: Store, row: dict) -> dict:
    """A preview pins its class definitions, including explicit negative images."""
    snapshot = row["summary"].get("taxonomy", TAXONOMY)
    taxonomy = get_taxonomy(store, snapshot["id"], row["project_id"])
    if snapshot != taxonomy:
        raise ValueError("The import's saved class definitions have changed.")
    return taxonomy


def _config(
    summary: dict,
    *,
    name,
    scene_group,
    source_url,
    license_name,
    attribution,
    source_split,
    category_mapping,
    taxonomy=TAXONOMY,
) -> dict:
    config = {
        "name": _text(name, "Import name", 160),
        "scene_group": _text(scene_group, "Scene group", 160),
        "source_url": _text(source_url, "Source URL", 2000),
        "license_name": _text(license_name, "License", 500),
        "attribution": _text(attribution, "Attribution", 2000, multiline=True),
    }
    url = urlsplit(config["source_url"])
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or any(character.isspace() for character in config["source_url"])
    ):
        raise ValueError(
            "Source URL must be an HTTP(S) URL without credentials; it is metadata only."
        )
    if source_split is not None and (
        not isinstance(source_split, str) or source_split not in SPLITS
    ):
        raise ValueError("Source split must be train, val, test, or unspecified.")
    if summary["source_split"] is not None and source_split != summary["source_split"]:
        raise ValueError("Explicitly preserve the split recorded in the COCO document.")
    expected = {str(category["id"]) for category in summary["categories"]}
    targets = {item["id"] for item in taxonomy["classes"]} | {"exclude"}
    if (
        not isinstance(category_mapping, dict)
        or set(category_mapping) != expected
        or any(
            not isinstance(value, str) or value not in targets
            for value in category_mapping.values()
        )
    ):
        raise ValueError("Map every source category explicitly to a saved target class or exclude.")
    return {**config, "source_split": source_split, "category_mapping": category_mapping.copy()}


def _insert(conn, table: str, values: dict):
    encoded = _encode(values)
    conn.execute(
        f"INSERT INTO {table} ({','.join(encoded)}) VALUES ({','.join('?' for _ in encoded)})",
        list(encoded.values()),
    )


def _verify_artifacts(store: Store, row: dict):
    path = store.artifact_path(row["path"])
    if not path.is_file() or _file_hash(path) != row["sha256"]:
        raise RuntimeError("The source archive is missing or has changed since preview.")
    for image in row["summary"]["images"]:
        original = store.artifact_path(image["source_path"])
        if not original.is_file() or _file_hash(original) != image["source_sha256"]:
            raise RuntimeError("An original image is missing or has changed since preview.")
        normalized = store.artifact_path(image["path"])
        if not normalized.is_file() or _file_hash(normalized) != image["png_sha256"]:
            raise RuntimeError("A normalized image is missing or has changed since preview.")
        with Image.open(normalized) as decoded:
            if (
                decoded.mode != "RGB"
                or decoded.size != (image["width"], image["height"])
                or _pixel_hash(decoded) != image["sha256"]
            ):
                raise RuntimeError("Normalized image pixels do not match the preview.")


def commit_import(
    store: Store,
    import_id: str,
    *,
    name,
    scene_group,
    source_url,
    license_name,
    attribution,
    source_split=None,
    category_mapping: dict[str, str],
) -> dict:
    """Create one session and pending proposals atomically; never validate annotations."""
    # Imported split reservations and frozen dataset reservations share one policy.
    from iris.datasets import _reservations

    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = _decode(
            conn.execute("SELECT * FROM dataset_imports WHERE id=?", (import_id,)).fetchone()
        )
        if row is None:
            raise KeyError(import_id)
        summary = row["summary"]
        taxonomy = _import_taxonomy(store, row)
        config = _config(
            summary,
            name=name,
            scene_group=scene_group,
            source_url=source_url,
            license_name=license_name,
            attribution=attribution,
            source_split=source_split,
            category_mapping=category_mapping,
            taxonomy=taxonomy,
        )
        if row["result"] is not None:
            if row["metadata"].get("config") != config:
                raise ValueError("This import is already committed with a different configuration.")
            return row["result"]
        _verify_artifacts(store, row)
        groups, pixels = _reservations(store, conn, row["project_id"])
        assignments = {
            source_split,
            groups.get(config["scene_group"]),
            *(pixels.get(image["sha256"]) for image in summary["images"]),
        } - {None}
        if len(assignments) > 1:
            raise ValueError(
                "The source split conflicts with an existing group or pixel reservation."
            )
        session_id, created_at = new_id(), now()
        _insert(
            conn,
            "sessions",
            {
                "id": session_id,
                "project_id": row["project_id"],
                "name": config["name"],
                "scene_group": config["scene_group"],
                "created_at": created_at,
            },
        )
        frame_ids, proposal_count, excluded = [], 0, 0
        categories = {category["id"]: category for category in summary["original_categories"]}
        for image in summary["images"]:
            asset_id, frame_id = new_id(), new_id()
            provenance = {
                "id": import_id,
                "taxonomy": taxonomy,
                "archive_sha256": row["sha256"],
                "archive_filename": summary["filename"],
                "json_filename": summary["json_filename"],
                **{
                    key: config[key]
                    for key in (
                        "source_url",
                        "license_name",
                        "attribution",
                        "source_split",
                        "category_mapping",
                    )
                },
                "coco_image": image["coco_image"],
                "original_annotations": image["original_annotations"],
                "original_categories": summary["original_categories"],
            }
            _insert(
                conn,
                "assets",
                {
                    "id": asset_id,
                    "session_id": session_id,
                    "filename": image["filename"],
                    "kind": "image",
                    "sha256": image["source_sha256"],
                    "size_bytes": image["size_bytes"],
                    "path": image["source_path"],
                    "metadata": {
                        "format": image["format"],
                        "width": image["width"],
                        "height": image["height"],
                        "dataset_import": provenance,
                    },
                    "created_at": created_at,
                },
            )
            _insert(
                conn,
                "frames",
                {
                    "id": frame_id,
                    "taxonomy_id": taxonomy["id"],
                    "session_id": session_id,
                    "asset_id": asset_id,
                    "frame_index": None,
                    "timestamp_seconds": None,
                    "width": image["width"],
                    "height": image["height"],
                    "sha256": image["sha256"],
                    "perceptual_hash": image["perceptual_hash"],
                    "path": image["path"],
                    "selected": True,
                    "extraction": {"method": "coco_import", "dataset_import": provenance},
                    "created_at": created_at,
                },
            )
            frame_ids.append(frame_id)
            for annotation in image["original_annotations"]:
                label = config["category_mapping"][str(annotation["category_id"])]
                if label == "exclude":
                    excluded += 1
                    continue
                x, y, width, height = annotation["bbox"]
                _insert(
                    conn,
                    "annotation_suggestions",
                    {
                        "id": new_id(),
                        "frame_id": frame_id,
                        "job_id": None,
                        "kind": "imported",
                        "label": label,
                        "box": [x, y, x + width, y + height],
                        "metadata": {
                            "import_id": import_id,
                            "archive_sha256": row["sha256"],
                            "source_url": config["source_url"],
                            "license_name": config["license_name"],
                            "attribution": config["attribution"],
                            "source_split": source_split,
                            "source_category": categories[annotation["category_id"]],
                            "source_annotation": annotation,
                            "source_image_id": image["coco_image_id"],
                            "mapping_target": label,
                            "target_taxonomy": taxonomy["id"],
                        },
                        "created_at": created_at,
                    },
                )
                proposal_count += 1
        result = {
            "session_id": session_id,
            "frame_ids": frame_ids,
            "proposal_count": proposal_count,
            "excluded_annotation_count": excluded,
        }
        conn.execute(
            "UPDATE dataset_imports SET metadata=?,result=? WHERE id=?",
            (json.dumps({"config": config}, allow_nan=False), json.dumps(result), import_id),
        )
    return result
