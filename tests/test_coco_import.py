"""COCO archive validation and transactional imports; all images are fixtures."""

import io
import json
import stat
import struct
import zipfile
from copy import deepcopy

import pytest
from PIL import EpsImagePlugin, Image

from iris import coco_import
from iris.annotations import get_annotation
from iris.coco_import import commit_import, import_detail, preview_image_path, preview_import
from iris.datasets import _reservations, dataset_candidates
from iris.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "workspace")


def image_bytes(color=(40, 80, 120), *, orientation=1):
    output = io.BytesIO()
    image = Image.new("RGB", (32, 24), color)
    exif = Image.Exif()
    exif[274] = orientation
    image.save(output, format="PNG", exif=exif)
    return output.getvalue()


def document():
    return {
        "info": {"description": "Synthetic import fixture; not flight data"},
        "licenses": [{"id": 1, "name": "Fixture only", "url": "https://example.test/license"}],
        "categories": [{"id": 4, "name": "pedestrian"}, {"id": 9, "name": "vehicle"}],
        "images": [
            {"id": 1, "file_name": "images/one.png", "width": 32, "height": 24, "license": 1},
            {"id": 2, "file_name": "images/two.png", "width": 32, "height": 24, "license": 1},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 4, "bbox": [2, 3, 10, 15], "iscrowd": 0},
            {
                "id": 2,
                "image_id": 1,
                "category_id": 9,
                "bbox": [17, 12, 12, 8],
                "segmentation": [[17, 12, 29, 12, 29, 20]],
                "area": 48,
            },
        ],
    }


def archive(tmp_path, doc=None, *, files=None, json_name="annotations.json", extra=(), raw=None):
    path = tmp_path / "fixture.zip"
    if files is None:
        files = {"images/one.png": image_bytes(), "images/two.png": image_bytes((20, 30, 50))}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr(json_name, raw if raw is not None else json.dumps(doc or document()))
        for name, data in files.items():
            zipped.writestr(name, data)
        for name, data in extra:
            zipped.writestr(name, data)
    return path


def config(**changes):
    return {
        "name": "Synthetic imported data",
        "scene_group": "fixture-scene",
        "source_url": "https://example.test/fixture",
        "license_name": "Fixture only",
        "attribution": "Automated fixture generator",
        "source_split": None,
        "category_mapping": {"4": "person", "9": "car"},
        **changes,
    }


def preview(store, tmp_path, doc=None, **kwargs):
    return preview_import(store, archive(tmp_path, doc, **kwargs), "fixture.zip")


def test_preview_preserves_original_metadata_and_creates_no_session(store, tmp_path):
    result = preview(store, tmp_path)
    assert result["status"] == "preview"
    assert result["image_count"] == result["annotation_count"] == 2
    assert result["categories"] == [
        {"id": 4, "name": "pedestrian", "count": 1},
        {"id": 9, "name": "vehicle", "count": 1},
    ]
    assert result["images"][0]["boxes"][0] == {
        "annotation_id": 1,
        "category_id": 4,
        "category_name": "pedestrian",
        "box": [2, 3, 12, 18],
    }
    assert result["images"][1]["annotation_count"] == 0
    assert result["info"] == document()["info"]
    assert result["licenses"] == document()["licenses"]
    assert result["result"] is result["config"] is None
    assert "path" not in result["images"][0]
    assert "source_path" not in result["images"][0]
    assert store.list("sessions") == store.list("annotation_revisions") == []
    row = store.get("dataset_imports", result["id"])
    assert store.artifact_path(row["path"]).read_bytes() == (tmp_path / "fixture.zip").read_bytes()
    with Image.open(preview_image_path(store, result["id"], "1")) as image:
        assert image.mode == "RGB" and image.size == (32, 24)


def test_commit_creates_pending_proposals_and_retains_negative_image(store, tmp_path):
    imported = preview(store, tmp_path)
    result = commit_import(store, imported["id"], **config())
    assert result["proposal_count"] == 2 and result["excluded_annotation_count"] == 0
    assert len(result["frame_ids"]) == 2
    session = store.get("sessions", result["session_id"])
    assert session["scene_group"] == "fixture-scene"
    assets = store.list("assets", session_id=session["id"])
    original = next(
        asset["metadata"]["dataset_import"]
        for asset in assets
        if asset["metadata"]["dataset_import"]["coco_image"]["id"] == 1
    )
    assert original["archive_sha256"] == imported["sha256"]
    assert original["original_annotations"] == document()["annotations"]
    assert original["original_categories"] == document()["categories"]
    assert original["source_url"] == config()["source_url"]
    assert original["category_mapping"] == config()["category_mapping"]
    for frame_id in result["frame_ids"]:
        frame = store.get("frames", frame_id)
        assert frame["selected"] is True
        assert frame["extraction"]["dataset_import"]["id"] == imported["id"]
        annotation = get_annotation(store, frame_id)
        assert annotation["revision"] == 0
        assert annotation["reviewer"] == ""
        assert all(suggestion["kind"] == "imported" for suggestion in annotation["suggestions"])
        assert all(suggestion["state"] == "pending" for suggestion in annotation["suggestions"])
    assert store.list("annotation_revisions") == []
    assert dataset_candidates(store)["excluded"]["unannotated"] == 2
    detail = import_detail(store, imported["id"])
    assert detail["status"] == "imported"
    assert detail["config"] == config()
    assert detail["result"] == result


def test_exclusion_is_explicit_and_preserves_excluded_source_labels(store, tmp_path):
    imported = preview(store, tmp_path)
    result = commit_import(
        store, imported["id"], **config(category_mapping={"4": "person", "9": "exclude"})
    )
    assert result["proposal_count"] == result["excluded_annotation_count"] == 1
    suggestions = store.list("annotation_suggestions")
    assert [suggestion["label"] for suggestion in suggestions] == ["person"]
    asset = store.get("assets", store.get("frames", result["frame_ids"][0])["asset_id"])
    assert asset["metadata"]["dataset_import"]["original_annotations"] == document()["annotations"]


def test_idempotency_and_restart_do_not_duplicate_session_or_proposals(store, tmp_path):
    imported = preview(store, tmp_path)
    result = commit_import(store, imported["id"], **config())
    reopened = Store(store.root)
    assert commit_import(reopened, imported["id"], **config()) == result
    assert len(store.list("sessions")) == 1
    assert len(store.list("annotation_suggestions")) == 2
    with pytest.raises(ValueError, match="different configuration"):
        commit_import(store, imported["id"], **config(name="Another name"))


@pytest.mark.parametrize(
    "changes",
    [
        {"category_mapping": {"4": "person"}},
        {"category_mapping": {"4": "person", "9": "truck"}},
        {"category_mapping": {4: "person", 9: "car"}},
        {"category_mapping": {"4": "person", "9": "car", "77": "exclude"}},
        {"source_url": "file:///tmp/example"},
        {"source_url": "https://username:password@example.test"},
        {"source_split": "validation"},
        {"scene_group": ""},
        {"license_name": " "},
        {"attribution": "\n"},
    ],
)
def test_invalid_config_never_creates_records(store, tmp_path, changes):
    imported = preview(store, tmp_path)
    with pytest.raises(ValueError):
        commit_import(store, imported["id"], **config(**changes))
    assert store.list("sessions") == store.list("assets") == store.list("frames") == []
    assert import_detail(store, imported["id"])["status"] == "preview"


@pytest.mark.parametrize(
    "field,value",
    [
        ("bbox", [2, 3, 0, 10]),
        ("bbox", [2, 3, -1, 10]),
        ("bbox", [-1, 0, 10, 10]),
        ("bbox", [30, 0, 10, 10]),
        ("bbox", [0, 20, 10, 10]),
        ("bbox", [False, 0, 10, 10]),
        ("bbox", [0, 0, "10", 10]),
        ("bbox", [0, 0, float("inf"), 10]),
        ("bbox", [0, 0, 10]),
        ("iscrowd", 1),
        ("ignore", 1),
        ("iscrowd", "0"),
        ("image_id", 44),
        ("category_id", 99),
        ("id", True),
        ("area", -1),
    ],
)
def test_invalid_annotation_rejects_whole_archive_and_cleans_preview(store, tmp_path, field, value):
    doc = document()
    doc["annotations"][0][field] = value
    with pytest.raises(ValueError):
        preview(store, tmp_path, doc)
    assert store.list("dataset_imports") == []
    assert list((store.root / "imports").iterdir()) == []


@pytest.mark.parametrize("collection", ["images", "annotations", "categories"])
def test_duplicate_ids_rejected(store, tmp_path, collection):
    doc = document()
    doc[collection].append(deepcopy(doc[collection][0]))
    with pytest.raises(ValueError, match="Duplicate COCO"):
        preview(store, tmp_path, doc)


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.png",
        "/tmp/escape.png",
        "images/../escape.png",
        "C:/escape.png",
        "images\\escape.png",
        "https://host/image.png",
        "images//escape.png",
        "./images/escape.png",
    ],
)
def test_unsafe_archive_entries_rejected_even_if_unreferenced(store, tmp_path, unsafe):
    with pytest.raises(ValueError, match="Unsafe archive path"):
        preview(store, tmp_path, extra=[(unsafe, b"unreferenced")])


def test_symlink_and_file_directory_conflicts_rejected(store, tmp_path):
    link = zipfile.ZipInfo("link.png")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="symlinks"):
        preview(store, tmp_path, extra=[(link, b"/etc/passwd")])
    with pytest.raises(ValueError, match="conflicting file and directory"):
        preview(store, tmp_path, extra=[("images", b"bad directory")])


def test_duplicate_zip_paths_rejected(store, tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        path = archive(tmp_path, extra=[("images/one.png", b"duplicate")])
    with pytest.raises(ValueError, match="duplicate"):
        preview_import(store, path, "fixture.zip")


def test_exactly_one_json_and_unique_image_path_resolution(store, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        preview(store, tmp_path, extra=[("other.json", b"{}")])
    result = preview(
        store,
        tmp_path,
        json_name="dataset/annotations.json",
        files={
            "dataset/images/one.png": image_bytes(),
            "dataset/images/two.png": image_bytes((90, 100, 50)),
        },
    )
    assert result["image_count"] == 2
    with pytest.raises(ValueError, match="ambiguous"):
        preview(
            store,
            tmp_path,
            json_name="dataset/annotations.json",
            extra=[
                ("dataset/images/one.png", image_bytes((11, 30, 50))),
            ],
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda doc: doc["images"][0].update(width=33), "dimensions"),
        (lambda doc: doc["images"][0].update(width=True), "integer"),
        (lambda doc: doc["images"][0].update(width=20_000, height=20_000), "20 megapixels"),
        (lambda doc: doc["images"][0].update(file_name="missing.png"), "missing"),
        (lambda doc: doc["images"][0].update(file_name="http://example.test/image.png"), "Unsafe"),
        (lambda doc: doc["images"][0].update(ignore=1), "Crowd and ignore"),
    ],
)
def test_invalid_image_metadata_rejected(store, tmp_path, mutation, error):
    doc = document()
    mutation(doc)
    with pytest.raises(ValueError, match=error):
        preview(store, tmp_path, doc)


def test_corrupt_and_rotated_images_rejected(store, tmp_path):
    for content, message in [
        (b"not a real image", "corrupt"),
        (image_bytes(orientation=6), "EXIF"),
    ]:
        with pytest.raises(ValueError, match=message):
            preview(
                store,
                tmp_path,
                files={
                    "images/one.png": content,
                    "images/two.png": image_bytes((10, 20, 30)),
                },
            )


def test_document_images_never_launch_external_renderers(store, tmp_path, monkeypatch):
    def forbidden_renderer(*args, **kwargs):
        pytest.fail("The COCO importer must not invoke a PostScript renderer")

    monkeypatch.setattr(EpsImagePlugin, "Ghostscript", forbidden_renderer)
    postscript = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 32 24\n%%EndComments\nshowpage\n"
    with pytest.raises(ValueError, match="unsupported"):
        preview(
            store,
            tmp_path,
            files={"images/one.png": postscript, "images/two.png": image_bytes((11, 22, 33))},
        )


def test_byte_duplicate_images_rejected_but_external_negative_allowed(store, tmp_path):
    with pytest.raises(ValueError, match="byte-identical"):
        preview(
            store,
            tmp_path,
            files={"images/one.png": image_bytes(), "images/two.png": image_bytes()},
        )
    doc = document()
    doc["annotations"], doc["categories"] = [], []
    imported = preview(store, tmp_path, doc)
    result = commit_import(store, imported["id"], **config(category_mapping={}))
    assert result["proposal_count"] == 0
    assert store.list("annotation_revisions") == []


def test_duplicate_json_keys_and_nonfinite_numbers_rejected(store, tmp_path):
    with pytest.raises(ValueError, match="Duplicate JSON"):
        preview(store, tmp_path, raw='{"images": [], "images": []}')
    with pytest.raises(ValueError, match="Non-finite"):
        preview(store, tmp_path, raw='{"value": NaN}')


def test_embedded_split_must_be_consistent_and_explicitly_preserved(store, tmp_path):
    doc = document()
    doc["info"]["split"] = "val"
    imported = preview(store, tmp_path, doc)
    assert imported["source_split"] == "val"
    for value in (None, "train"):
        with pytest.raises(ValueError, match="preserve the split"):
            commit_import(store, imported["id"], **config(source_split=value))
    result = commit_import(store, imported["id"], **config(source_split="val"))
    with store.connect() as conn:
        groups, pixels = _reservations(store, conn)
    assert groups["fixture-scene"] == "val"
    assert pixels[store.get("frames", result["frame_ids"][0])["sha256"]] == "val"
    doc["images"][0]["split"] = "train"
    with pytest.raises(ValueError, match="Mixed source splits"):
        preview(store, tmp_path, doc)


def test_source_split_conflicts_rejected_by_group_and_pixels(store, tmp_path):
    first = preview(store, tmp_path)
    commit_import(store, first["id"], **config(source_split="test"))
    second = preview(store, tmp_path)
    with pytest.raises(ValueError, match="reservation"):
        commit_import(store, second["id"], **config(source_split="train", scene_group="new-group"))
    third = preview(
        store,
        tmp_path,
        files={
            "images/one.png": image_bytes((101, 23, 54)),
            "images/two.png": image_bytes((231, 102, 73)),
        },
    )
    with pytest.raises(ValueError, match="reservation"):
        commit_import(store, third["id"], **config(source_split="val"))
    assert len(store.list("sessions")) == 1


def test_unknown_source_split_still_preserves_known_group_and_pixel_reservations(store, tmp_path):
    training = preview(store, tmp_path)
    commit_import(store, training["id"], **config(source_split="train", scene_group="training"))
    validation = preview(
        store,
        tmp_path,
        files={
            "images/one.png": image_bytes((112, 123, 154)),
            "images/two.png": image_bytes((231, 102, 73)),
        },
    )
    commit_import(store, validation["id"], **config(source_split="val", scene_group="validation"))
    repeated_training = preview(store, tmp_path)
    with pytest.raises(ValueError, match="reservation"):
        commit_import(
            store, repeated_training["id"], **config(source_split=None, scene_group="validation")
        )
    mixed_pixels = preview(
        store,
        tmp_path,
        files={
            "images/one.png": image_bytes(),
            "images/two.png": image_bytes((231, 102, 73)),
        },
    )
    with pytest.raises(ValueError, match="reservation"):
        commit_import(store, mixed_pixels["id"], **config(source_split=None, scene_group="new"))
    assert len(store.list("sessions")) == 2


def test_multiline_attribution_is_preserved(store, tmp_path):
    imported = preview(store, tmp_path)
    attribution = "Fixture author\nSource: automated fixture\t2026"
    commit_import(store, imported["id"], **config(attribution=attribution))
    assert import_detail(store, imported["id"])["config"]["attribution"] == attribution


@pytest.mark.parametrize("collection", ["images", "annotations", "categories"])
def test_ids_beyond_javascript_exact_integer_range_are_rejected(store, tmp_path, collection):
    doc = document()
    doc[collection][0]["id"] = 2**53
    with pytest.raises(ValueError, match=r"2\^53"):
        preview(store, tmp_path, doc)


def test_corrupt_deflate_is_reported_as_invalid_archive(store, tmp_path):
    path = archive(tmp_path)
    data = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as zipped:
        offset = zipped.getinfo("annotations.json").header_offset
    filename_length, extra_length = struct.unpack_from("<HH", data, offset + 26)
    data[offset + 30 + filename_length + extra_length] = 0b00000111
    path.write_bytes(data)
    with pytest.raises(ValueError, match="ZIP archive is invalid"):
        preview_import(store, path, "fixture.zip")
    assert store.list("dataset_imports") == []
    assert list((store.root / "imports").iterdir()) == []


@pytest.mark.parametrize("artifact", ["archive", "original", "normalized"])
def test_tampered_artifacts_cannot_be_committed(store, tmp_path, artifact):
    imported = preview(store, tmp_path)
    row = store.get("dataset_imports", imported["id"])
    path = {
        "archive": row["path"],
        "original": row["summary"]["images"][0]["source_path"],
        "normalized": row["summary"]["images"][0]["path"],
    }[artifact]
    store.artifact_path(path).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        commit_import(store, imported["id"], **config())
    assert store.list("sessions") == []
    if artifact == "normalized":
        with pytest.raises(RuntimeError):
            preview_image_path(store, imported["id"], "1")


def test_failed_commit_rolls_back_whole_session_and_can_retry(store, tmp_path, monkeypatch):
    imported = preview(store, tmp_path)
    insert = coco_import._insert

    def fail_on_proposal(conn, table, values):
        if table == "annotation_suggestions":
            raise RuntimeError("Injected fixture failure after session, asset, and frame inserts")
        insert(conn, table, values)

    monkeypatch.setattr(coco_import, "_insert", fail_on_proposal)
    with pytest.raises(RuntimeError, match="Injected fixture failure"):
        commit_import(store, imported["id"], **config())
    for table in ("sessions", "assets", "frames", "annotation_suggestions"):
        assert store.list(table) == []
    assert import_detail(store, imported["id"])["status"] == "preview"
    monkeypatch.setattr(coco_import, "_insert", insert)
    assert commit_import(store, imported["id"], **config())["proposal_count"] == 2


@pytest.mark.parametrize(
    "limit,value,error",
    [
        ("MAX_ARCHIVE_BYTES", 10, "64 MiB"),
        ("MAX_EXPANDED_BYTES", 10, "expanded archive"),
        ("MAX_JSON_BYTES", 10, "JSON document exceeds"),
        ("MAX_IMAGES", 1, "between 1 and 1"),
        ("MAX_TOTAL_PIXELS", 1000, "total limit"),
        ("MAX_ANNOTATIONS_PER_IMAGE", 1, "more than 500"),
    ],
)
def test_resource_limits_reject_without_records(store, tmp_path, monkeypatch, limit, value, error):
    monkeypatch.setattr(coco_import, limit, value)
    with pytest.raises(ValueError, match=error):
        preview(store, tmp_path)
    assert store.list("dataset_imports") == []


def test_missing_ids_are_not_found(store):
    with pytest.raises(KeyError):
        import_detail(store, "missing")
    with pytest.raises(KeyError):
        preview_image_path(store, "missing", "1")
    with pytest.raises(KeyError):
        commit_import(store, "missing", **config())
