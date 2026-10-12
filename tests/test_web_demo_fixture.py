"""Public demo data stays reproducible, attributed, and separate from human review."""

import copy
import hashlib
import importlib.util
import io
import json
import socket
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from iris.app import create_app

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "web-demo"
SOURCE = ROOT / "examples" / "street-scenes"


def load_example_module(name):
    spec = importlib.util.spec_from_file_location(f"iris_web_demo_{name}", EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(EXAMPLE))
        spec.loader.exec_module(module)
    return module


def assert_same_files(actual, expected):
    assert actual.keys() == expected.keys()
    changed = [name for name in actual if actual[name] != expected[name]]
    assert not changed, f"Rebuild changes these public files: {changed}"


@pytest.fixture(scope="module")
def generated():
    builder = load_example_module("build")

    def no_network(*args, **kwargs):
        pytest.fail("Rebuilding public demo fixtures must not access the network")

    def no_inference(*args, **kwargs):
        pytest.fail("Rebuilding public demo fixtures must replay saved predictions")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", no_network)
        patch.setattr("iris.inference.TorchvisionDetector", no_inference)
        first = builder.build()
        second = builder.build()
    assert_same_files(first, second)
    return first


@pytest.fixture(scope="module")
def fixture(generated):
    return json.loads(generated["fixture.json"])


def test_rebuild_matches_every_checked_in_file(generated):
    stored = {
        path.relative_to(EXAMPLE / "seed").as_posix(): path.read_bytes()
        for path in (EXAMPLE / "seed").rglob("*")
        if path.is_file()
    }
    assert_same_files(generated, stored)
    assert "fixture.json" in generated
    assert "CREDITS.txt" in generated
    for name, data in generated.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts
        assert isinstance(data, bytes)


def test_seed_contains_only_the_public_project_and_resolved_media(fixture, generated):
    assert fixture["format"] == "iris-web-demo-v1"
    assert fixture["fixture_version"] == 1
    assert fixture["project_id"] == "web-demo"
    assert fixture["session_id"] == "street-scenes"
    responses = fixture["responses"]
    (project,) = responses["/api/projects"]
    assert project == responses["/api/projects/web-demo"]
    assert project["id"] == fixture["project_id"]
    (session,) = responses["/api/sessions"]
    assert session == responses["/api/sessions/street-scenes"]
    assert session["project_id"] == project["id"]
    assert session["id"] == fixture["session_id"]
    frames = responses["/api/sessions/street-scenes/frames"]
    assets = responses["/api/sessions/street-scenes/assets"]
    assert {frame["id"] for frame in frames} == {"frame-6723", "frame-32941"}
    assert {frame["asset_id"] for frame in frames} == {asset["id"] for asset in assets}
    for frame in frames:
        assert frame["session_id"] == session["id"]
        assert frame["selected"] is True
        assert frame["duplicate_count"] == 0
        assert f"/api/frames/{frame['id']}/annotation" in responses
        resource = f"/api/frames/{frame['id']}/image"
        assert fixture["resources"][resource] in generated
    for route, name in fixture["resources"].items():
        assert route.startswith("/api/")
        assert name.startswith("media/")
        assert name in generated


def test_demo_pixels_match_licensed_photos_and_original_hashes(fixture, generated):
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    frames = {
        frame["id"]: frame for frame in fixture["responses"]["/api/sessions/street-scenes/frames"]
    }
    with zipfile.ZipFile(SOURCE / "street-scenes.zip") as archive:
        for source in provenance["images"]:
            image_id = source["coco_image_id"]
            original_bytes = archive.read(source["archive_path"])
            assert hashlib.sha256(original_bytes).hexdigest() == source["sha256"]
            original_resource = fixture["resources"][f"/api/assets/asset-{image_id}/media"]
            assert generated[original_resource] == original_bytes
            with Image.open(io.BytesIO(original_bytes)) as original:
                pixels = ImageOps.exif_transpose(original).convert("RGB")
            resource = fixture["resources"][f"/api/frames/frame-{image_id}/image"]
            with Image.open(io.BytesIO(generated[resource])) as image:
                assert image.size == (source["width"], source["height"])
                assert image.convert("RGB").tobytes() == pixels.tobytes()
                digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
                digest.update(image.convert("RGB").tobytes())
                assert digest.hexdigest() == frames[f"frame-{image_id}"]["sha256"]
    credits = generated["CREDITS.txt"].decode()
    for source in provenance["images"]:
        assert source["author"] in credits
        assert source["photo_url"] in credits
        assert source["license_url"] in credits
    assert provenance["annotation_source"]["license_url"] in credits


def test_provenance_hashes_cover_each_generated_file(generated):
    provenance = json.loads(generated["provenance.json"])
    assert set(provenance["files"]) == set(generated) - {"provenance.json"}
    for name, record in provenance["files"].items():
        assert record == {
            "sha256": hashlib.sha256(generated[name]).hexdigest(),
            "size_bytes": len(generated[name]),
        }
    assert (
        provenance["source_archive_sha256"]
        == hashlib.sha256((SOURCE / "street-scenes.zip").read_bytes()).hexdigest()
    )
    assert (
        provenance["recording_sha256"]
        == hashlib.sha256((EXAMPLE / "recording.json").read_bytes()).hexdigest()
    )


def test_build_output_refuses_to_overwrite_different_files(generated, tmp_path):
    builder = load_example_module("build")
    output = tmp_path / "output"
    builder.write_output(generated, output)
    builder.write_output(generated, output)
    existing = output / "fixture.json"
    existing.write_text("Existing content to preserve")
    with pytest.raises(FileExistsError):
        builder.write_output(generated, output)
    assert existing.read_text() == "Existing content to preserve"


def test_seed_does_not_publish_machine_paths_or_provider_secrets(generated):
    documents = [json.loads(data) for name, data in generated.items() if name.endswith(".json")]
    documents.append(json.loads((EXAMPLE / "recording.json").read_text()))
    text = json.dumps(documents)
    for private_fragment in ("/home/", "/tmp/", "file://", "127.0.0.1", "localhost", ".iris/"):
        assert private_fragment not in text

    def check(value):
        if isinstance(value, dict):
            assert not {
                "path",
                "source_path",
                "api_key",
                "access_token",
                "authorization",
                "provider_credentials",
            }.intersection(value)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(documents)


def test_imported_proposals_preserve_sources_without_claiming_human_review(fixture):
    source = json.loads((SOURCE / "annotations.json").read_text())
    originals = {item["id"]: item for item in source["annotations"]}
    all_proposals = []
    all_originals = []
    for frame in fixture["responses"]["/api/sessions/street-scenes/frames"]:
        annotation = fixture["responses"][f"/api/frames/{frame['id']}/annotation"]
        assert annotation["frame"]["id"] == frame["id"]
        assert annotation["revision"] == 0
        assert annotation["status"] == "unannotated"
        assert annotation["reviewer"] == annotation["notes"] == ""
        assert annotation["boxes"] == annotation["history"] == []
        assert annotation["decisions"] == {}
        imported = frame["extraction"]["dataset_import"]
        assert imported["source_split"] == "val"
        all_originals.extend(imported["original_annotations"])
        for proposal in annotation["suggestions"]:
            assert proposal["kind"] == "imported"
            assert proposal["state"] == "pending"
            assert proposal["frame_id"] == frame["id"]
            original = proposal["metadata"]["source_annotation"]
            assert original == originals[original["id"]]
            assert proposal["label"] == {1: "person", 3: "car"}[original["category_id"]]
            x, y, width, height = original["bbox"]
            assert proposal["box"] == [x, y, x + width, y + height]
            all_proposals.append(proposal)
    assert len(all_proposals) == 11
    assert sorted(all_originals, key=lambda item: item["id"]) == sorted(
        source["annotations"], key=lambda item: item["id"]
    )
    assert len(all_originals) == 19


def test_saved_comparison_preserves_recorded_predictions_and_timings(fixture):
    recording = load_example_module("recording").load_recording()
    (summary,) = fixture["responses"]["/api/sessions/street-scenes/comparisons"]
    detail = fixture["responses"][f"/api/comparisons/{summary['id']}"]
    assert summary["id"] == recording["comparison"]["id"] == fixture["comparison_id"]
    assert detail["job"] == recording["job"]
    assert detail["runs"] == recording["runs"]
    assert detail["model_ids"] == recording["comparison"]["model_ids"]
    assert detail["created_at"] == recording["comparison"]["created_at"]
    assert detail["config"]["protocol"] == recording["comparison"]["config"]["protocol"]
    assert detail["config"]["device"] == "cpu"
    mapped_frames = {
        frame["frame_id"]: f"frame-{frame['coco_image_id']}" for frame in recording["frames"]
    }
    expected = {
        row["id"]: {**row, "frame_id": mapped_frames[row["frame_id"]]}
        for row in recording["predictions"]
    }
    assert {row["id"]: row for row in detail["predictions"]} == expected
    assert len(detail["predictions"]) == 4
    assert sum(len(row["detections"]) for row in detail["predictions"]) == 400
    assert set(detail["replay"]["still_frame_ids"]) == set(mapped_frames.values())
    assert detail["replay"]["continuous_inference"] is False
    for frame_id in mapped_frames.values():
        annotation = fixture["responses"][f"/api/frames/{frame_id}/annotation"]
        sources = annotation["prediction_sources"]
        assert {row["id"] for row in sources} == {
            row["id"] for row in expected.values() if row["frame_id"] == frame_id
        }
        assert {row["comparison_id"] for row in sources} == {detail["id"]}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("predictions", 0, "detections", 0, "score"), float("nan")),
        (("predictions", 0, "detections", 0, "box", 2), 100000),
        (("predictions", 0, "run_id"), "missing-run"),
        (("runs", 0, "metadata", "weight_sha256"), "0" * 64),
        (("frames", 0, "frame_pixel_sha256"), "0" * 64),
        (("frames", 0, "source_sha256"), "0" * 64),
        (("runs", 0, "metadata", "local_path"), "/private/model.pth"),
    ],
)
def test_changed_recording_is_rejected_before_any_rows_are_inserted(tmp_path, path, value):
    builder = load_example_module("build")
    recorder = load_example_module("recording")
    recording = recorder.load_recording()
    target = recording
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with TestClient(
        create_app(tmp_path / "workspace", run_jobs=False), base_url="http://127.0.0.1"
    ) as client:
        imported = builder.import_source(client)
        store = client.app.state.store
        frames = {
            row["extraction"]["dataset_import"]["coco_image"]["id"]: row
            for row in imported["frames"]
        }
        with pytest.raises(ValueError):
            recorder.restore_recording(store, recording, imported["session"]["id"], frames)
        for table in ("jobs", "comparisons", "runs", "predictions", "annotation_revisions"):
            assert store.list(table) == []


def test_public_projection_drops_unknown_fields_at_every_depth(fixture):
    contract = load_example_module("public_contract")
    source = fixture["responses"]["/api/frames/frame-6723/annotation"]
    poisoned = copy.deepcopy(source)
    sentinel = "PRIVATE_LOCAL_WORKSPACE_SENTINEL"

    def add_private_fields(value):
        if isinstance(value, dict):
            for child in value.values():
                add_private_fields(child)
            value["local_path"] = sentinel
            value["provider_credentials"] = {"token": sentinel}
        elif isinstance(value, list):
            for child in value:
                add_private_fields(child)

    # Empty containers represent intentionally absent data and are checked separately.
    def restore_empty_containers(original, altered):
        if isinstance(original, dict):
            for key, value in original.items():
                if value == {} or value == []:
                    altered[key] = value
                else:
                    restore_empty_containers(value, altered[key])
        elif isinstance(original, list):
            for value, changed in zip(original, altered, strict=True):
                restore_empty_containers(value, changed)

    add_private_fields(poisoned)
    restore_empty_containers(source, poisoned)
    projected = contract.project(poisoned, contract.ANNOTATION)
    assert projected == source
    assert sentinel not in json.dumps(projected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", 1),
        ("revision", False),
        ("status", "validated"),
        ("reviewer", "Private reviewer"),
        ("notes", "Private notes"),
        ("history", [{"reviewer": "Private reviewer"}]),
        ("boxes", [{"label": "person", "box": [1, 2, 3, 4]}]),
        ("decisions", {"imported": "accepted"}),
    ],
)
def test_public_projection_rejects_reviewed_or_private_annotation_state(fixture, field, value):
    contract = load_example_module("public_contract")
    annotation = copy.deepcopy(fixture["responses"]["/api/frames/frame-6723/annotation"])
    annotation[field] = value
    with pytest.raises(ValueError):
        contract.project(annotation, contract.ANNOTATION)
