"""Frozen independent reference and operating-point quality, without model weights."""

from copy import deepcopy

import pytest
from PIL import Image

from iris import benchmark, benchmark_runs, inference
from iris.annotations import adopt_taxonomy, save_annotation
from iris.benchmark import (
    BenchmarkConflict,
    benchmark_candidates,
    benchmark_detail,
    create_benchmark,
    create_benchmark_config,
    load_benchmark_manifest,
    lock_benchmark,
    preview_benchmark,
    preview_benchmark_config,
    score_benchmark_outputs,
    score_proposals,
)
from iris.jobs import JobManager
from iris.media import import_asset
from iris.projects import create_project
from iris.store import Store, new_id, now
from iris.taxonomies import TAXONOMY, publish_taxonomy

MODEL = "ssdlite320_mobilenet_v3_large"
DETECTION = dict(label_id=1, label="person", box=[2, 3, 30, 35], score=0.85)
TIMING = dict(preprocess_ms=1, inference_ms=2, postprocess_ms=1, total_ms=4)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    frames, roles = [], {}
    for i in range(4):
        role = "tuning" if i < 2 else "evaluation"
        group = f"scene-{i}"
        session = store.insert(
            "sessions", dict(id=new_id(), name=group, scene_group=group, created_at=now())
        )
        source = tmp_path / f"{i}.png"
        Image.new("RGB", (80, 60), (i * 50, 30, 50)).save(source)
        asset = import_asset(store, session["id"], source, source.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        store.update("frames", frame["id"], {"selected": True})
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=[dict(id="human-box", label="person", box=[2, 3, 30, 35])],
            decisions={},
            status="validated",
            reviewer="Independent fixture author",
        )
        frames.append(frame)
        roles[group] = role
    model = {**inference.get_spec(MODEL), "status": "ready", "weight_sha256": "a" * 64}
    for module in (benchmark, benchmark_runs):
        monkeypatch.setattr(module, "catalog", lambda _: [deepcopy(model)])
    return store, JobManager(store), frames, roles, model


def settings(workspace, **changes):
    return dict(
        frame_ids=[frame["id"] for frame in workspace[2]],
        roles=workspace[3],
        reviewer="Independent fixture author",
        independence_notes="Reference drawn before candidate invocation: withheld sentinel.",
        independent_reference=True,
        **changes,
    )


def freeze(workspace, **changes):
    values = settings(workspace, **changes)
    preview = preview_benchmark(workspace[0], **values)
    return create_benchmark(
        workspace[0],
        name="Independent fixture",
        expected_fingerprint=preview["fingerprint"],
        **values,
    )


def config(workspace, reference, **changes):
    values = dict(model_id=MODEL, **changes)
    preview = preview_benchmark_config(workspace[0], reference["id"], **values)
    return create_benchmark_config(
        workspace[0],
        reference["id"],
        name="Local control",
        expected_fingerprint=preview["fingerprint"],
        **values,
    )


def test_preview_read_only_and_stale_reference_never_publishes(workspace):
    store = workspace[0]
    before = store.list("annotation_revisions")
    preview = preview_benchmark(store, **settings(workspace))
    assert preview["summary"]["role_counts"] == {"tuning": 2, "evaluation": 2}
    assert store.list("annotation_revisions") == before
    assert not store.list("benchmarks") and not (store.root / "benchmarks").exists()
    save_annotation(
        store,
        workspace[2][0]["id"],
        expected_revision=1,
        boxes=[],
        decisions={},
        status="draft",
        reviewer="",
    )
    with pytest.raises((ValueError, BenchmarkConflict)):
        create_benchmark(
            store,
            name="Stale",
            expected_fingerprint=preview["fingerprint"],
            **settings(workspace),
        )
    assert not store.list("benchmarks")
    assert not list((store.root / "benchmarks").iterdir())


def test_copied_reference_survives_live_revisions_and_source_image_edits(workspace):
    store = workspace[0]
    reference = freeze(workspace)
    raw = store.artifact_path(reference["path"]).read_bytes()
    frame = workspace[2][0]
    save_annotation(
        store,
        frame["id"],
        expected_revision=1,
        boxes=[],
        decisions={},
        status="validated",
        reviewer="Later reviewer",
    )
    Image.new("RGB", (80, 60), "white").save(store.artifact_path(frame["path"]))
    assert (
        load_benchmark_manifest(store, reference["id"], verify_images=True) == reference["manifest"]
    )
    assert store.artifact_path(reference["path"]).read_bytes() == raw
    assert reference["manifest"]["frames"][0]["boxes"]


def test_explicit_independence_declaration_and_both_roles_required(workspace):
    values = settings(workspace)
    values["independent_reference"] = False
    with pytest.raises(ValueError, match="declare"):
        preview_benchmark(workspace[0], **values)
    values["independent_reference"] = True
    values["roles"] = dict.fromkeys(values["roles"], "tuning")
    with pytest.raises(ValueError, match="Both tuning"):
        preview_benchmark(workspace[0], **values)


def test_known_model_derived_reference_excluded_despite_human_declaration(workspace):
    store = workspace[0]
    revision = store.list("annotation_revisions", frame_id=workspace[2][0]["id"])[0]
    boxes = deepcopy(revision["boxes"])
    boxes[0]["source"] = dict(kind="detector", suggestion_id="historical-model-proposal")
    store.update("annotation_revisions", revision["id"], {"boxes": boxes})
    assert benchmark_candidates(store)["excluded"]["model_derived_reference"] == 1
    with pytest.raises(ValueError, match="Model-derived"):
        preview_benchmark(store, **settings(workspace))


def test_pending_suggestion_reference_is_ineligible(workspace):
    store = workspace[0]
    store.insert(
        "annotation_suggestions",
        dict(
            id=new_id(),
            frame_id=workspace[2][0]["id"],
            label="person",
            box=[2, 3, 30, 35],
            kind="detector",
            metadata={},
            created_at=now(),
        ),
    )
    with pytest.raises(ValueError, match="pending"):
        preview_benchmark(store, **settings(workspace))


def test_original_video_and_frozen_benchmark_roles_cannot_be_recycled(workspace):
    store = workspace[0]
    for frame in (workspace[2][0], workspace[2][2]):
        store.update("assets", frame["asset_id"], {"kind": "video", "sha256": "f" * 64})
    with pytest.raises(ValueError, match="original video"):
        preview_benchmark(store, **settings(workspace))
    store.update("assets", workspace[2][2]["asset_id"], {"sha256": "e" * 64})
    freeze(workspace)
    values = settings(workspace)
    values["roles"] = {
        group: "evaluation" if role == "tuning" else "tuning"
        for group, role in values["roles"].items()
    }
    with pytest.raises(ValueError, match="reserved"):
        preview_benchmark(store, **values)


def test_custom_reference_freezes_namespace_and_official_partial_coverage(workspace):
    store = workspace[0]
    taxonomy = publish_taxonomy(
        store,
        "default",
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[
            dict(id="helmet", name="Helmet", definition="A helmet"),
            dict(id="vehicle", name="Vehicle", definition="A car", coco_id=3),
        ],
    )
    for frame in workspace[2]:
        save_annotation(store, frame["id"], expected_revision=1, boxes=[], decisions={})
        adopted = adopt_taxonomy(
            store,
            frame["id"],
            expected_revision=2,
            expected_taxonomy_id=TAXONOMY["id"],
            target_taxonomy_id=taxonomy["id"],
        )
        save_annotation(
            store,
            frame["id"],
            expected_revision=adopted["revision"],
            boxes=[dict(id="custom", label="helmet", box=[2, 3, 30, 35])],
            decisions={},
            status="validated",
            reviewer="Independent fixture author",
        )
    reference = freeze(workspace, taxonomy_id=taxonomy["id"])
    assert reference["manifest"]["output_mapping"] == {"helmet": 1, "vehicle": 2}
    candidate = config(workspace, reference)
    contract = candidate["config"]["proposal_contract"]
    assert contract["label_mapping"] == {"3": "vehicle"}
    assert contract["unsupported_class_ids"] == ["helmet"]

    from test_benchmark_runs import Detector

    class VehicleDetector(Detector):
        def predict(self, image):
            output = super().predict(image)
            output["detections"][0].update(label_id=3, label="car")
            return output

    preview = benchmark_runs.preview_benchmark_trial(
        store, reference["id"], config_id=candidate["id"], role="tuning"
    )
    trial = benchmark_runs.create_benchmark_trial(
        store,
        workspace[1],
        reference["id"],
        config_id=candidate["id"],
        role="tuning",
        expected_fingerprint=preview["fingerprint"],
    )
    result = benchmark_runs.run_benchmark_trial(
        store, trial["id"], lambda *_: None, lambda: False, detector_factory=VehicleDetector
    )
    assert result["quality"]["metrics"]["summary"]["class_conflicts"] == 2
    assert result["quality"]["metrics"]["per_class"]["helmet"]["fn"] == 2
    for output in store.list("benchmark_outputs"):
        proposal = output["result"]["proposals"][0]
        assert proposal["label"] == "vehicle" and proposal["original_label_id"] == 3


def test_global_pixel_reservation_and_project_scoped_scene_names(workspace, tmp_path):
    store = workspace[0]
    freeze(workspace)
    project = create_project(store, name="Separate experiment")
    new_frames = []
    # Reuse project A scene names with different images: names alone must not reserve B.
    for index, group in enumerate(("scene-0", "scene-2")):
        session = store.insert(
            "sessions",
            dict(
                id=new_id(),
                project_id=project["id"],
                name=group,
                scene_group=group,
                created_at=now(),
            ),
        )
        source = tmp_path / f"separate-{index}.png"
        Image.new("RGB", (80, 60), (220, 20, index * 80)).save(source)
        asset = import_asset(store, session["id"], source, source.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        store.update("frames", frame["id"], {"selected": True})
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=[],
            decisions={},
            status="validated",
            reviewer="Separate human author",
        )
        new_frames.append(frame)
    values = settings(workspace)
    values.update(
        frame_ids=[frame["id"] for frame in new_frames],
        roles={"scene-0": "evaluation", "scene-2": "tuning"},
    )
    assert (
        preview_benchmark(store, project_id=project["id"], **values)["summary"]["frame_count"] == 2
    )
    original = workspace[2][0]
    # Same original pixels in another project remain reserved globally.
    copied = import_asset(
        store, new_frames[0]["session_id"], store.artifact_path(original["path"]), "copied.png"
    )
    copied_frame = store.list("frames", asset_id=copied["id"])[0]
    store.update("frames", copied_frame["id"], {"selected": True})
    save_annotation(
        store,
        copied_frame["id"],
        expected_revision=0,
        boxes=[],
        decisions={},
        status="validated",
        reviewer="Separate human author",
    )
    values["frame_ids"][0] = copied_frame["id"]
    with pytest.raises(ValueError, match="reserved"):
        preview_benchmark(store, project_id=project["id"], **values)
    candidates = benchmark_candidates(store, project["id"])
    copied_candidate = next(
        frame
        for group in candidates["groups"]
        for frame in group["frames"]
        if frame["frame_id"] == copied_frame["id"]
    )
    assert copied_candidate["allowed_roles"] == ["tuning"]
    assert all(
        frame["session_id"] in {row["session_id"] for row in new_frames}
        for group in candidates["groups"]
        for frame in group["frames"]
    )


def test_lock_explicit_cas_and_no_future_adapters(workspace):
    store = workspace[0]
    reference = freeze(workspace)
    stale = reference["lock_fingerprint"]
    for approach in ("multimodal", "segmentation", "combined"):
        with pytest.raises(ValueError, match="unavailable"):
            preview_benchmark_config(store, reference["id"], model_id=MODEL, approach=approach)
    config(workspace, reference)
    with pytest.raises(BenchmarkConflict, match="changed"):
        lock_benchmark(store, reference["id"], expected_fingerprint=stale)
    detail = benchmark_detail(store, reference["id"])
    assert any("no complete tuning" in warning for warning in detail["warnings"])
    locked = lock_benchmark(store, reference["id"], expected_fingerprint=detail["lock_fingerprint"])
    assert locked["status"] == "locked"
    with pytest.raises(BenchmarkConflict, match="locked"):
        preview_benchmark_config(store, reference["id"], model_id=MODEL)


def test_quality_one_to_one_class_conflicts_and_failures_not_empty(workspace):
    manifest = freeze(workspace)["manifest"]
    frame = manifest["frames"][0]
    exact = dict(label="person", box=[2, 3, 30, 35])
    quality = score_proposals(frame, [exact, exact], manifest["taxonomy"])
    assert (quality["tp"], quality["fp"], quality["fn"]) == (1, 1, 0)
    wrong = score_proposals(frame, [{**exact, "label": "car"}], manifest["taxonomy"])
    assert (wrong["tp"], wrong["fp"], wrong["fn"], wrong["class_conflicts"]) == (0, 1, 1, 1)
    role = [row for row in manifest["frames"] if row["role"] == "tuning"]
    outputs = [
        dict(frame_id=row["frame_id"], result={"proposals": [exact]}, error=None) for row in role
    ]
    measured = score_benchmark_outputs(manifest, "tuning", outputs)
    assert measured["metrics"]["summary"]["matched_iou_mean"] == 1
    assert measured["metrics"]["summary"]["tp"] == 2
    outputs[0].update(error="Prediction failed", result=None)
    incomplete = score_benchmark_outputs(manifest, "tuning", outputs)
    assert not incomplete["complete"] and incomplete["metrics"] is None
