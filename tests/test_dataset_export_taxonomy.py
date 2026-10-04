"""Generic COCO archives use only frozen definitions, mappings, images and revisions."""

import hashlib
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pycocotools.coco import COCO
from test_dataset_export import _members, _rewrite_manifest

from iris.annotations import save_annotation
from iris.app import create_app
from iris.dataset_export import GENERIC_README, build_coco_export
from iris.datasets import create_dataset
from iris.media import import_asset
from iris.store import DEFAULT_PROJECT_ID, SCHEMA_VERSION, Store, new_id, now
from iris.taxonomies import TAXONOMY, publish_taxonomy
from iris.workspace_archive import create_archive
from iris.workspace_restore import inspect_archive, restore_archive

CLASSES = [
    {"id": "helmet", "name": "Safety helmet", "definition": "A visible protective helmet."},
    {
        "id": "vehicle",
        "name": "Passenger vehicle",
        "definition": "A passenger car or SUV.",
        "coco_id": 3,
    },
    {
        "id": "marker",
        "name": "Inspection marker",
        "definition": "A toothbrush-shaped inspection marker.",
        "coco_id": 90,
    },
]


@pytest.fixture
def generic_release(tmp_path):
    store = Store(tmp_path / "workspace")
    taxonomy = publish_taxonomy(
        store, DEFAULT_PROJECT_ID, expected_taxonomy_id=TAXONOMY["id"], classes=CLASSES
    )
    frames = []
    for index, split in enumerate(("train", "val")):
        session = store.insert(
            "sessions",
            {"id": new_id(), "name": split, "scene_group": split, "created_at": now()},
        )
        source = tmp_path / f"{split}.png"
        Image.new("RGB", (80, 60), (40 + index, 70, 90)).save(source)
        asset = import_asset(store, session["id"], source, source.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        store.update("frames", frame["id"], {"selected": True})
        boxes = (
            [
                {"id": "helmet-box", "label": "helmet", "box": [0.25, 1.5, 20.75, 30.25]},
                {"id": "vehicle-box", "label": "vehicle", "box": [0, 0, 80, 60]},
            ]
            if split == "train"
            else []
        )
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            taxonomy_id=taxonomy["id"],
            boxes=boxes,
            decisions={},
            status="validated",
            reviewer="Synthetic reviewer",
            notes="Fixture only; no real detector run",
        )
        frames.append(frame)
    dataset = create_dataset(
        store,
        name="Generic fixture",
        frame_ids=[frame["id"] for frame in frames],
        splits={"train": "train", "val": "val"},
        taxonomy_id=taxonomy["id"],
    )
    return store, dataset, frames, taxonomy


def test_generic_export_download_is_complete_and_leaves_no_temporary_zip(generic_release):
    store, dataset, _, taxonomy = generic_release
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as client:
        response = client.get(f"/api/datasets/{dataset['id']}/export/coco")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        metadata = json.loads(archive.read("export.json"))
        assert metadata["taxonomy"] == taxonomy
        assert metadata["protocol"] == "iris-coco-export-v2"
    assert not list((store.root / "exports").glob("*.zip"))


def test_generic_coco_consumer_reads_three_classes_negatives_and_empty_test(
    generic_release, tmp_path
):
    store, dataset, frames, taxonomy = generic_release
    original = store.artifact_path(dataset["path"]).read_bytes()
    archive = build_coco_export(store, dataset["id"])
    members = _members(archive)
    assert members["iris-manifest.json"] == original
    assert members["README.txt"] == GENERIC_README.encode()
    metadata = json.loads(members["export.json"])
    assert metadata["protocol"] == "iris-coco-export-v2"
    assert metadata["taxonomy"] == taxonomy
    assert metadata["taxonomy_id"] == taxonomy["id"]
    assert (
        metadata["class_mapping"]
        == metadata["coco_mapping"]
        == {"helmet": 1, "vehicle": 2, "marker": 3}
    )
    assert metadata["taxonomy"]["classes"][1]["coco_id"] == 3
    assert metadata["coco_mapping"]["vehicle"] == 2
    assert metadata["taxonomy"]["classes"][2]["coco_id"] == 90
    assert metadata["coco_mapping"]["marker"] == 3
    assert set(metadata["files"]) == set(members) - {"export.json"}
    for name, item in metadata["files"].items():
        assert item == {
            "sha256": hashlib.sha256(members[name]).hexdigest(),
            "size_bytes": len(members[name]),
        }
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as contents:
        contents.extractall(unpacked)
    for split in ("train", "val", "test"):
        coco = COCO(str(unpacked / split / "annotations.json"))
        assert coco.dataset["categories"] == [
            {"id": 1, "name": "helmet"},
            {"id": 2, "name": "vehicle"},
            {"id": 3, "name": "marker"},
        ]
        assert len(coco.getImgIds()) == (0 if split == "test" else 1)
        assert len(coco.getAnnIds()) == (2 if split == "train" else 0)
        assert coco.getAnnIds(catIds=[3]) == []
        for image in coco.loadImgs(coco.getImgIds()):
            with Image.open(unpacked / split / image["file_name"]) as pixels:
                assert pixels.size == (80, 60)
    train = json.loads(members["train/annotations.json"])
    helmet = next(box for box in train["annotations"] if box["category_id"] == 1)
    assert helmet["bbox"] == [0.25, 1.5, 20.5, 28.75]
    assert helmet["area"] == 20.5 * 28.75
    val = json.loads(members["val/annotations.json"])
    assert val["images"][0]["iris_frame_id"] == frames[1]["id"]
    assert not any(name.startswith("test/images/") for name in members)


def test_generic_export_stays_identical_after_live_edits_and_registry_removal(generic_release):
    store, dataset, frames, taxonomy = generic_release
    before = build_coco_export(store, dataset["id"]).read_bytes()
    publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=taxonomy["id"],
        classes=[{**CLASSES[0], "definition": "Only worn helmets."}, *CLASSES[1:]],
    )
    save_annotation(
        store, frames[0]["id"], expected_revision=1, boxes=[], decisions={}, status="draft"
    )
    for frame in frames:
        store.artifact_path(frame["path"]).unlink()
        store.artifact_path(store.get("assets", frame["asset_id"])["path"]).unlink(missing_ok=True)
    # An export is standalone even if the live registry is unavailable. Deliberate
    # database damage is restricted to this temporary fixture; no Store migration.
    with store.connect() as connection:
        connection.execute("DELETE FROM taxonomy_versions")
        database = "\n".join(connection.iterdump())
    assert build_coco_export(store, dataset["id"]).read_bytes() == before
    with store.connect() as connection:
        assert "\n".join(connection.iterdump()) == database


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["coco_mapping"].update(vehicle=3),
        lambda manifest: manifest["coco_mapping"].update(helmet=True),
        lambda manifest: manifest["class_mapping"].update(marker=90),
        lambda manifest: manifest["coco_mapping"].pop("marker"),
        lambda manifest: manifest["taxonomy"]["classes"][0].update(definition=""),
        lambda manifest: manifest["frames"][0]["annotation"].update(taxonomy_id=TAXONOMY["id"]),
        lambda manifest: manifest["frames"][0]["annotation"].update(revision=2),
        lambda manifest: manifest["frames"][1]["annotation"].update(
            taxonomy_id="taxonomy-" + "a" * 32
        ),
    ],
)
def test_generic_export_rejects_rehashed_mapping_or_revision_corruption(generic_release, mutation):
    store, dataset, _, _ = generic_release
    _rewrite_manifest(store, dataset, mutation)
    with pytest.raises(ValueError):
        build_coco_export(store, dataset["id"])
    assert list((store.root / "exports").glob("*")) == []


def test_generic_workspace_backup_restores_exact_manifest_images_and_export(
    generic_release, tmp_path
):
    store, dataset, _, taxonomy = generic_release
    original_manifest = store.artifact_path(dataset["path"]).read_bytes()
    original_export = build_coco_export(store, dataset["id"]).read_bytes()
    publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=taxonomy["id"],
        classes=[{**CLASSES[0], "name": "Head protection"}, *CLASSES[1:]],
    )
    saved = create_archive(store.root, tmp_path / "workspace.zip")
    assert saved["manifest"]["schema_version"] == SCHEMA_VERSION
    checked = inspect_archive(saved["path"])
    target = tmp_path / "restored"
    restore_archive(saved["path"], target, expected_archive_sha256=checked["archive_sha256"])
    assert (target / dataset["path"]).read_bytes() == original_manifest
    for frame in dataset["manifest"]["frames"]:
        assert (target / frame["image_path"]).read_bytes() == store.artifact_path(
            frame["image_path"]
        ).read_bytes()
    reopened = Store(target)
    assert build_coco_export(reopened, dataset["id"]).read_bytes() == original_export
