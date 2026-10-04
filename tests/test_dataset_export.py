"""Portable COCO exports of synthetic, explicitly reviewed dataset fixtures."""

import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image
from pycocotools.coco import COCO

from iris import dataset_export
from iris.annotations import save_annotation
from iris.dataset_export import ExportLimitError, build_coco_export
from iris.datasets import _canonical, create_dataset
from iris.media import import_asset
from iris.store import Store, new_id, now


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def release(tmp_path):
    store = Store(tmp_path / "workspace")
    frames = []
    for index, split in enumerate(("train", "val", "test"), 1):
        session = store.insert(
            "sessions",
            {
                "id": new_id(),
                "name": f"Synthetic {split}",
                "scene_group": f"scene-{split}",
                "created_at": now(),
            },
        )
        path = tmp_path / f"{split}.png"
        Image.new("RGB", (80, 60), (index * 30, 40, 80)).save(path)
        asset = import_asset(store, session["id"], path, path.name)
        if split == "train":
            store.update(
                "assets",
                asset["id"],
                {
                    "metadata": {
                        **asset["metadata"],
                        "dataset_import": {
                            "source_url": "https://example.org/synthetic",
                            "license_name": "Synthetic test license",
                            "attribution": "Fixture author",
                            "source_split": split,
                            "original_annotations": [],
                        },
                    }
                },
            )
        frame = store.list("frames", asset_id=asset["id"])[0]
        store.update("frames", frame["id"], {"selected": True})
        boxes = (
            [
                {
                    "id": "box-z",
                    "label": "person" if split == "train" else "car",
                    "box": [0.25, 1.5, 79.75, 59.25],
                },
                {"id": "box-a", "label": "car", "box": [0, 0, 80, 60]},
            ]
            if split != "test"
            else []
        )
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=boxes,
            decisions={},
            status="validated",
            reviewer="Synthetic export reviewer",
            notes="Fixture labels, no human review of real data",
        )
        frames.append(frame)
    dataset = create_dataset(
        store,
        name="Synthetic export",
        frame_ids=[frame["id"] for frame in reversed(frames)],
        splits={f"scene-{split}": split for split in ("train", "val", "test")},
    )
    return store, dataset, frames


def _rewrite_manifest(store, dataset, mutate):
    path = store.artifact_path(dataset["path"])
    manifest = json.loads(path.read_bytes())
    mutate(manifest)
    raw = _canonical(manifest)
    path.write_bytes(raw)
    store.update("dataset_versions", dataset["id"], {"manifest_sha256": _sha256(raw)})
    return manifest


def _members(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_coco_geometry_ids_negatives_provenance_and_hashes(release):
    store, dataset, _ = release
    original = store.artifact_path(dataset["path"]).read_bytes()
    path = build_coco_export(store, dataset["id"])
    assert path.parent == store.root / "exports"
    members = _members(path)
    assert members["iris-manifest.json"] == original
    metadata = json.loads(members["export.json"])
    assert metadata["protocol"] == "iris-coco-export-v1"
    assert metadata["manifest"] == {
        "path": "iris-manifest.json",
        "sha256": dataset["manifest_sha256"],
    }
    expected_names = {
        "train/annotations.json",
        "val/annotations.json",
        "test/annotations.json",
        "iris-manifest.json",
        "export.json",
        "README.txt",
    }
    expected_names.update(
        f"{frame['split']}/images/{frame['frame_id']}.png"
        for frame in dataset["manifest"]["frames"]
    )
    assert set(members) == expected_names
    assert set(metadata["files"]) == set(members) - {"export.json"}
    for name, recorded in metadata["files"].items():
        assert recorded == {"sha256": _sha256(members[name]), "size_bytes": len(members[name])}
    train = json.loads(members["train/annotations.json"])
    val = json.loads(members["val/annotations.json"])
    test = json.loads(members["test/annotations.json"])
    assert train["categories"] == [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
    assert train["categories"] == val["categories"] == test["categories"]
    assert len(test["images"]) == 1
    assert test["annotations"] == []
    assert train["annotations"][0]["category_id"] == 3
    fractional = train["annotations"][1]
    assert fractional["category_id"] == 1
    assert fractional["bbox"] == [0.25, 1.5, 79.5, 57.75]
    assert fractional["area"] == 79.5 * 57.75
    assert fractional["iscrowd"] == 0
    assert fractional["iris_box_id"] == "box-z"
    assert train["annotations"][0]["bbox"] == [0, 0, 80, 60]
    images = sorted(metadata["images"], key=lambda image: image["iris_frame_id"])
    assert [image["coco_image_id"] for image in images] == [1, 2, 3]
    for split, document in (("train", train), ("val", val), ("test", test)):
        assert document["info"]["split"] == split
        for image in document["images"]:
            assert image["split"] == split
            assert f"{split}/{image['file_name']}" in members
    frozen = json.loads(members["iris-manifest.json"])
    source = next(frame for frame in frozen["frames"] if frame["split"] == "train")
    assert source["source"]["metadata"]["dataset_import"]["attribution"] == "Fixture author"
    assert source["annotation"]["reviewer"] == "Synthetic export reviewer"
    assert source["annotation"]["notes"] == "Fixture labels, no human review of real data"
    assert b"not redacted" in members["README.txt"]
    path.unlink()


def test_reference_coco_consumer_can_read_all_splits(release, tmp_path):
    store, dataset, _ = release
    archive = build_coco_export(store, dataset["id"])
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as contents:
        contents.extractall(unpacked)
    for split in ("train", "val", "test"):
        coco = COCO(str(unpacked / split / "annotations.json"))
        assert set(coco.getCatIds()) == {1, 3}
        assert len(coco.getImgIds()) == 1
        for image in coco.loadImgs(coco.getImgIds()):
            with Image.open(unpacked / split / image["file_name"]) as pixels:
                assert pixels.size == (image["width"], image["height"])
        assert len(coco.getAnnIds()) == (0 if split == "test" else 2)


def test_deterministic_across_repeated_exports_and_restart(release):
    store, dataset, _ = release
    first = build_coco_export(store, dataset["id"])
    second = build_coco_export(store, dataset["id"])
    restarted = build_coco_export(Store(store.root), dataset["id"])
    assert first != second != restarted
    assert first.read_bytes() == second.read_bytes() == restarted.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.testzip() is None
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_STORED


def test_live_edits_and_removed_originals_leave_export_unchanged(release):
    store, dataset, frames = release
    original = build_coco_export(store, dataset["id"]).read_bytes()
    frame = frames[0]
    save_annotation(
        store,
        frame["id"],
        expected_revision=1,
        boxes=[],
        decisions={},
        status="draft",
        reviewer="Later fixture editor",
    )
    store.update("frames", frame["id"], {"selected": False})
    store.artifact_path(frame["path"]).unlink()
    store.artifact_path(store.get("assets", frame["asset_id"])["path"]).unlink(missing_ok=True)
    store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frame["id"],
            "kind": "detector",
            "label": "person",
            "box": [1, 1, 2, 2],
            "metadata": {},
            "created_at": now(),
        },
    )
    assert build_coco_export(store, dataset["id"]).read_bytes() == original


def test_export_preserves_empty_optional_test_split(release):
    store, _, frames = release
    dataset = create_dataset(
        store,
        name="No test fixtures",
        frame_ids=[frame["id"] for frame in frames[:2]],
        splits={"scene-train": "train", "scene-val": "val"},
    )
    members = _members(build_coco_export(store, dataset["id"]))
    test = json.loads(members["test/annotations.json"])
    assert test["images"] == test["annotations"] == []
    assert not any(name.startswith("test/images/") for name in members)


def test_no_database_mutation_or_unrelated_workspace_files(release):
    store, dataset, _ = release
    for relative in (".env", "models/secret.pt", "uploads/private.mp4", "configuration.json"):
        path = store.root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("NEVER_EXPORT_SENTINEL")
    with store.connect() as connection:
        before = "\n".join(connection.iterdump())
    members = _members(build_coco_export(store, dataset["id"]))
    with store.connect() as connection:
        assert "\n".join(connection.iterdump()) == before
    assert not any(b"NEVER_EXPORT_SENTINEL" in value for value in members.values())


def test_missing_dataset_is_not_found(release):
    store, _, _ = release
    with pytest.raises(KeyError):
        build_coco_export(store, "absent")
    assert not (store.root / "exports").exists()


def test_foreign_project_is_rejected_even_with_rehashed_manifest(release):
    store, dataset, _ = release
    _rewrite_manifest(store, dataset, lambda doc: doc.update(project_id="another-project"))
    with pytest.raises(ValueError, match="different project"):
        build_coco_export(store, dataset["id"])
    assert list((store.root / "exports").glob("*")) == []


def test_legacy_default_project_manifest_exports_without_rewriting(release):
    store, dataset, _ = release
    _rewrite_manifest(store, dataset, lambda doc: doc.pop("project_id"))
    original = store.artifact_path(dataset["path"]).read_bytes()
    members = _members(build_coco_export(store, dataset["id"]))
    assert members["iris-manifest.json"] == original
    assert store.artifact_path(dataset["path"]).read_bytes() == original


@pytest.mark.parametrize("target", ["manifest", "image"])
@pytest.mark.parametrize("operation", ["remove", "change"])
def test_missing_or_tampered_snapshot_is_rejected_and_cleaned(release, target, operation):
    store, dataset, _ = release
    path = store.artifact_path(
        dataset["path"] if target == "manifest" else dataset["manifest"]["frames"][0]["image_path"]
    )
    if operation == "remove":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="missing|recorded"):
        build_coco_export(store, dataset["id"])
    assert list((store.root / "exports").glob("*")) == []


@pytest.mark.parametrize("replacement", ["../secret.png", "/tmp/secret.png", "uploads/other.png"])
def test_escaped_image_paths_are_rejected_even_with_rehashed_manifest(release, replacement):
    store, dataset, _ = release
    _rewrite_manifest(store, dataset, lambda doc: doc["frames"][0].update(image_path=replacement))
    with pytest.raises(ValueError, match="outside"):
        build_coco_export(store, dataset["id"])
    assert list((store.root / "exports").glob("*")) == []


def test_image_symlink_to_another_directory_is_rejected(release):
    store, dataset, frames = release
    copied = store.artifact_path(dataset["manifest"]["frames"][0]["image_path"])
    copied.unlink()
    copied.symlink_to(store.artifact_path(frames[0]["path"]))
    with pytest.raises(ValueError, match="outside"):
        build_coco_export(store, dataset["id"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc.update(taxonomy={}),
        lambda doc: doc.update(class_mapping={"person": 1, "car": 3}),
        lambda doc: doc.update(schema_version=True),
        lambda doc: doc.update(frames=[]),
        lambda doc: doc["frames"].append(doc["frames"][0]),
        lambda doc: doc["frames"][0].update(frame_id="../bad"),
        lambda doc: doc["frames"][0].pop("image_path"),
        lambda doc: doc["frames"][0].update(image_path=None),
        lambda doc: doc["frames"][0].update(image_path={"path": "invalid"}),
        lambda doc: doc["frames"][0].update(image_path=""),
        lambda doc: doc["frames"][0].update(width=True),
        lambda doc: doc["frames"][0].update(width=0),
        lambda doc: doc["frames"][0].update(split="validation"),
        lambda doc: doc["frames"][0].update(scene_group="unassigned"),
        lambda doc: doc["frames"][0].update(revision=0),
        lambda doc: doc["frames"][0].update(revision=True),
        lambda doc: doc["frames"][0]["annotation"].update(status="draft"),
        lambda doc: doc["frames"][0]["annotation"].update(reviewer=""),
        lambda doc: doc["frames"][0]["annotation"].update(frame_sha256="bad"),
        lambda doc: doc["frames"][0]["annotation"].update(boxes=[{}]),
        lambda doc: doc["splits"].update(extra="train"),
        lambda doc: doc.update(splits=[]),
    ],
)
def test_invalid_frozen_contract_is_rejected(release, mutation):
    store, dataset, _ = release
    _rewrite_manifest(store, dataset, mutation)
    with pytest.raises(ValueError):
        build_coco_export(store, dataset["id"])
    assert list((store.root / "exports").glob("*")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda box: box.update(box=[0, 0, 100, 60]),
        lambda box: box.update(box=[0, 0, 0, 60]),
        lambda box: box.update(box=[False, 0, 20, 60]),
        lambda box: box.update(box=[0, 0, 20]),
        lambda box: box.update(label="truck"),
        lambda box: box.update(id=""),
        lambda box: box.update(iscrowd=1),
        lambda box: box.update(ignore=True),
    ],
)
def test_invalid_box_geometry_and_semantics_are_rejected(release, mutation):
    store, dataset, _ = release

    def change(doc):
        frame = next(frame for frame in doc["frames"] if frame["boxes"])
        mutation(frame["boxes"][0])
        frame["annotation"]["boxes"] = frame["boxes"]

    _rewrite_manifest(store, dataset, change)
    with pytest.raises(ValueError):
        build_coco_export(store, dataset["id"])


@pytest.mark.parametrize("mismatch", ["pixels", "dimensions", "format"])
def test_verified_file_hash_does_not_bypass_pixel_dimensions_or_format(release, mismatch):
    store, dataset, _ = release
    frame = dataset["manifest"]["frames"][0]
    path = store.artifact_path(frame["image_path"])
    if mismatch == "pixels":
        Image.new("RGB", (80, 60), "orange").save(path, format="PNG")
    elif mismatch == "dimensions":
        Image.new("RGB", (81, 60), "orange").save(path, format="PNG")
    else:
        Image.new("RGB", (80, 60), "orange").save(path, format="JPEG")
    _rewrite_manifest(
        store,
        dataset,
        lambda doc: doc["frames"][0].update(image_file_sha256=_sha256(path.read_bytes())),
    )
    with pytest.raises(ValueError, match="pixel|dimensions|format"):
        build_coco_export(store, dataset["id"])
    assert list((store.root / "exports").glob("*")) == []


@pytest.mark.parametrize("limit", ["MAX_MANIFEST_BYTES", "MAX_IMAGE_BYTES", "MAX_ARCHIVE_BYTES"])
def test_size_limits_reject_and_remove_partial_archive(release, monkeypatch, limit):
    store, dataset, _ = release
    monkeypatch.setattr(dataset_export, limit, 1)
    with pytest.raises(ExportLimitError):
        build_coco_export(store, dataset["id"])
    assert list((store.root / "exports").glob("*")) == []


def test_pixel_limit_rejects_before_decoding(release, monkeypatch):
    store, dataset, _ = release
    monkeypatch.setattr(dataset_export, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ExportLimitError, match="megapixel"):
        build_coco_export(store, dataset["id"])


def test_archive_limit_accounts_for_complete_zip_headers(release, monkeypatch):
    store, dataset, _ = release
    first = build_coco_export(store, dataset["id"])
    size = first.stat().st_size
    first.unlink()
    monkeypatch.setattr(dataset_export, "MAX_ARCHIVE_BYTES", size)
    exact = build_coco_export(store, dataset["id"])
    assert exact.stat().st_size == size
    exact.unlink()
    monkeypatch.setattr(dataset_export, "MAX_ARCHIVE_BYTES", size - 1)
    with pytest.raises(ExportLimitError):
        build_coco_export(store, dataset["id"])
    assert list((store.root / "exports").iterdir()) == []


def test_failure_midway_through_writing_cleans_partial_archive(release, monkeypatch):
    store, dataset, _ = release
    original = zipfile.ZipFile.writestr
    count = 0

    def interrupted(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("Synthetic full disk")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", interrupted)
    with pytest.raises(OSError, match="full disk"):
        build_coco_export(store, dataset["id"])
    assert list((store.root / "exports").iterdir()) == []


def test_source_mutation_after_verification_cannot_change_written_bytes(release, monkeypatch):
    store, dataset, _ = release
    original = dataset_export._image_bytes
    expected = {}

    def verified_then_mutate(store, frame):
        raw = original(store, frame)
        expected[f"{frame['split']}/images/{frame['frame_id']}.png"] = raw
        store.artifact_path(frame["image_path"]).write_bytes(b"Mutated after verified read")
        return raw

    monkeypatch.setattr(dataset_export, "_image_bytes", verified_then_mutate)
    members = _members(build_coco_export(store, dataset["id"]))
    for name, raw in expected.items():
        assert members[name] == raw


def test_manifest_mutation_after_read_does_not_replace_archived_snapshot(release, monkeypatch):
    store, dataset, _ = release
    original = dataset_export._read_bounded
    manifest_bytes = store.artifact_path(dataset["path"]).read_bytes()

    def read_then_mutate(path, limit, description):
        raw = original(path, limit, description)
        if description == "Frozen manifest":
            path.write_bytes(b"{}")
        return raw

    monkeypatch.setattr(dataset_export, "_read_bounded", read_then_mutate)
    members = _members(build_coco_export(store, dataset["id"]))
    assert members["iris-manifest.json"] == manifest_bytes


def test_png_decode_uses_verified_bytes_not_source_file(release, monkeypatch):
    store, dataset, _ = release
    original = Image.open
    decoded = []

    def tracked(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        decoded.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(Image, "open", tracked)
    build_coco_export(store, dataset["id"])
    assert len(decoded) == 3
