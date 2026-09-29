"""Evaluation lifecycle checks use frozen synthetic images and a fake detector."""

from copy import deepcopy

import pytest
from PIL import Image

from iris import evaluation, models
from iris.annotations import save_annotation
from iris.datasets import create_dataset
from iris.jobs import JobManager
from iris.media import import_asset
from iris.store import Store, new_id, now

MODEL_IDS = ["ssdlite320_mobilenet_v3_large", "fasterrcnn_mobilenet_v3_large_320_fpn"]
TIMING = {"preprocess_ms": 1, "inference_ms": 2, "postprocess_ms": 1, "total_ms": 4}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    store = Store(tmp_path / "workspace")
    frames = []
    for color, split in enumerate(("train", "val", "val", "test"), 1):
        session = store.insert(
            "sessions",
            {
                "id": new_id(),
                "name": f"Synthetic {split}",
                "scene_group": split,
                "created_at": now(),
            },
        )
        path = tmp_path / f"source-{color}.png"
        Image.new("RGB", (40, 30), (color, 20, 50)).save(path)
        asset = import_asset(store, session["id"], path, path.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        store.update("frames", frame["id"], {"selected": True})
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=[{"id": "box", "label": "person", "box": [2, 3, 14, 25]}],
            decisions={},
            status="validated",
            reviewer="Synthetic fixture reviewer",
        )
        frames.append(frame)
    dataset = create_dataset(
        store,
        name="Synthetic frozen release",
        frame_ids=[frame["id"] for frame in frames],
        splits={"train": "train", "val": "val", "test": "test"},
    )
    available = [
        {**models.get_spec(identifier), "status": "ready", "weight_sha256": str(index) * 64}
        for index, identifier in enumerate(MODEL_IDS, 1)
    ]
    monkeypatch.setattr(evaluation, "catalog", lambda _root: available)
    return store, dataset, frames, available


def queue(workspace, **changes):
    store, dataset, _, _ = workspace
    return evaluation.create_evaluation(
        store,
        JobManager(store),
        **{
            "name": "Synthetic held-out evaluation",
            "dataset_id": dataset["id"],
            "model_ids": MODEL_IDS[:1],
            **changes,
        },
    )


def fixture_detector(calls=None, *, fail_at=None, digest=None):
    calls = [] if calls is None else calls

    class SyntheticDetector:
        def __init__(self, root, model_id, *, device):
            self.model_id = model_id
            self.metadata = {
                "model_id": model_id,
                "device": device,
                "runtime": "synthetic fixture only",
                "weight_sha256": digest or str(MODEL_IDS.index(model_id) + 1) * 64,
            }
            calls.append(("load", model_id))

        def warmup(self, image):
            calls.append(("warmup", self.model_id, image.getpixel((0, 0))))

        def predict(self, image):
            if fail_at is not None and sum(item[0] == "predict" for item in calls) == fail_at:
                raise RuntimeError("Synthetic detector failure")
            calls.append(("predict", self.model_id, image.getpixel((0, 0))))
            return {
                "input_size": list(image.size),
                "timing": TIMING,
                "detections": [
                    {"label_id": 1, "label": "person", "score": 0.9, "box": [2, 3, 14, 25]},
                    {"label_id": 3, "label": "car", "score": 0.02, "box": [20, 10, 30, 20]},
                ],
            }

    return SyntheticDetector


def run(store, row, **kwargs):
    return evaluation.run_evaluation(
        store,
        row["id"],
        lambda *_args: None,
        kwargs.pop("cancelled", lambda: False),
        detector_factory=kwargs.pop("detector_factory", fixture_detector()),
        **kwargs,
    )


def promote(store, row, **changes):
    return evaluation.promote_reference(
        store,
        **{
            "evaluation_id": row["id"],
            "model_id": MODEL_IDS[0],
            "reviewer": "Fixture reviewer",
            "notes": "Synthetic selection exercise",
            **changes,
        },
    )


def test_queue_freezes_complete_split_hashes_settings_and_atomic_job(workspace):
    store, dataset, frames, available = workspace
    row = queue(workspace, model_ids=MODEL_IDS, confidence_threshold=0.6)
    assert row["job"]["kind"] == "evaluate"
    assert row["job"]["params"] == {"evaluation_id": row["id"]}
    assert row["config"]["dataset_manifest_sha256"] == dataset["manifest_sha256"]
    assert row["config"]["frame_ids"] == [frame["id"] for frame in frames[1:3]]
    assert row["config"]["model_hashes"] == {
        model["id"]: model["weight_sha256"] for model in available
    }
    assert row["config"]["confidence_threshold"] == 0.6
    assert row["config"]["warnings"]
    assert (
        evaluation.evaluation_summary(Store(store.root), store.get("evaluations", row["id"])) == row
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"split": "train"},
        {"split": "other"},
        {"model_ids": []},
        {"model_ids": [MODEL_IDS[0]] * 2},
        {"model_ids": ["missing"]},
        {"dataset_id": "missing"},
        {"name": " "},
        {"device": "magic"},
        {"confidence_threshold": float("nan")},
        {"confidence_threshold": True},
        {"confidence_threshold": -0.01},
        {"iou_threshold": 1.1},
        {"iou_threshold": 0},
        {"validation_evaluation_id": "irrelevant"},
        {"split": "test"},
    ],
)
def test_invalid_evaluation_never_queues(workspace, changes):
    with pytest.raises(ValueError):
        queue(workspace, **changes)
    assert workspace[0].list("jobs") == []


def test_uses_frozen_holdout_pixels_and_labels_with_complete_raw_predictions(workspace):
    store, dataset, frames, _ = workspace
    row = queue(workspace, model_ids=MODEL_IDS)
    # Sources, selections and newer labels are irrelevant to this frozen release.
    for frame in frames:
        store.update("frames", frame["id"], {"selected": False})
        save_annotation(store, frame["id"], expected_revision=1, boxes=[], decisions={})
        store.artifact_path(frame["path"]).unlink()
    for frame in dataset["manifest"]["frames"]:
        if frame["split"] != "val":
            store.artifact_path(frame["image_path"]).unlink()
    calls = []
    result = run(store, row, detector_factory=fixture_detector(calls))
    assert result["models_completed"] == 2
    assert result["predictions_created"] == 4
    assert not result["cancelled"]
    assert sum(call[0] == "warmup" for call in calls) == 2
    assert [call[2][0] for call in calls if call[0] == "predict"] == [2, 3, 2, 3]
    detail = evaluation.evaluation_detail(store, row["id"])
    assert detail["job"]["status"] == "succeeded"
    assert detail["job"]["result"] == result
    assert all(model["metrics"]["summary"]["tp"] == 2 for model in detail["models"])
    assert all(
        model["metrics"]["summary"]["map50"] == pytest.approx(1) for model in detail["models"]
    )
    assert all(len(prediction["detections"]) == 2 for prediction in detail["predictions"])
    assert all("decode_ms" in prediction["timing"] for prediction in detail["predictions"])
    assert all(
        frame["boxes"] and frame["image_url"].endswith("/image") for frame in detail["frames"]
    )
    JobManager(store).cancel(row["job_id"])
    assert store.get("jobs", row["job_id"])["status"] == "succeeded"
    with pytest.raises(ValueError, match="immutable"):
        run(store, row)


@pytest.mark.parametrize("changed", ["checkpoint", "dataset", "image", "runtime_hash"])
def test_worker_rejects_input_changes(workspace, changed):
    store, dataset, _, available = workspace
    row = queue(workspace)
    factory = fixture_detector()
    if changed == "checkpoint":
        available[0]["weight_sha256"] = "f" * 64
    elif changed == "dataset":
        store.update("dataset_versions", dataset["id"], {"manifest_sha256": "f" * 64})
    elif changed == "image":
        frame = next(frame for frame in dataset["manifest"]["frames"] if frame["split"] == "val")
        store.artifact_path(frame["image_path"]).write_bytes(b"tampered fixture")
    else:
        factory = fixture_detector(digest="f" * 64)
    with pytest.raises(ValueError, match="changed|hash"):
        run(store, row, detector_factory=factory)
    assert not store.list("evaluation_predictions")
    assert all(model["metrics"] is None for model in store.list("evaluation_models"))


def test_cancelled_partial_model_has_no_metrics_and_cannot_be_selected(workspace):
    store = workspace[0]
    row = queue(workspace)
    result = run(store, row, cancelled=lambda: bool(store.list("evaluation_predictions")))
    assert result["cancelled"]
    assert result["predictions_created"] == 1
    detail = evaluation.evaluation_detail(store, row["id"])
    assert len(detail["predictions"]) == 1
    assert detail["models"][0]["metrics"] is None
    store.update("jobs", row["job_id"], {"status": "cancelled"})
    with pytest.raises(ValueError, match="finish successfully"):
        promote(store, row)


def test_second_model_failure_retains_first_scores_but_prevents_selection(workspace):
    store = workspace[0]
    row = queue(workspace, model_ids=MODEL_IDS)
    with pytest.raises(RuntimeError, match="Synthetic detector failure"):
        run(store, row, detector_factory=fixture_detector(fail_at=2))
    detail = evaluation.evaluation_detail(store, row["id"])
    assert detail["models"][0]["metrics"] is not None
    assert detail["models"][1]["metrics"] is None
    store.update("jobs", row["job_id"], {"status": "failed"})
    with pytest.raises(ValueError, match="finish successfully"):
        promote(store, row)


def trained_entry(parent, frames, **overrides):
    return {
        **deepcopy(parent),
        "id": "trained_fixture",
        "origin": "trained",
        "taxonomy_id": "iris-objects-v1",
        "parent_model_id": parent["id"],
        "classes": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
        "provenance": {
            "training_scene_groups": ["train"],
            "training_frame_hashes": [frames[0]["sha256"]],
            "parent_weight_sha256": parent["weight_sha256"],
        },
        **overrides,
    }


@pytest.mark.parametrize(
    "overlap", ["group", "pixels", "ancestor", "missing", "taxonomy", "origin", "cycle"]
)
def test_rejects_training_leaks_and_unknown_provenance(workspace, overlap):
    _, _, frames, available = workspace
    trained = trained_entry(available[0], frames)
    available.append(trained)
    if overlap == "group":
        trained["provenance"]["training_scene_groups"] = ["val"]
    elif overlap == "pixels":
        trained["provenance"]["training_frame_hashes"] = [frames[1]["sha256"]]
    elif overlap == "ancestor":
        child = trained_entry(trained, frames, id="trained_child")
        available.append(child)
        trained["provenance"]["training_scene_groups"] = ["val"]
        trained = child
    elif overlap == "missing":
        trained["provenance"] = {}
    elif overlap == "taxonomy":
        trained["classes"][1]["id"] = 2
    elif overlap == "origin":
        trained["origin"] = "untracked"
    else:
        trained["parent_model_id"] = trained["id"]
    with pytest.raises(ValueError, match="train|provenance|taxonomy|cycle"):
        queue(workspace, model_ids=[trained["id"]])
    assert workspace[0].list("jobs") == []


def test_accepts_disjoint_trained_checkpoint_and_rechecks_lineage_at_execution(workspace):
    store, _, frames, available = workspace
    trained = trained_entry(available[0], frames)
    available.append(trained)
    row = queue(workspace, model_ids=[trained["id"]])
    trained["provenance"]["training_scene_groups"] = ["other-train"]
    with pytest.raises(ValueError, match="provenance changed"):
        run(store, row)


def test_test_audit_retains_validation_settings_and_cannot_select_reference(workspace):
    store, _, _, _ = workspace
    validation = queue(workspace, model_ids=MODEL_IDS, confidence_threshold=0.65)
    run(store, validation)
    test = queue(
        workspace,
        split="test",
        model_ids=MODEL_IDS,
        confidence_threshold=0.65,
        validation_evaluation_id=validation["id"],
    )
    result = run(store, test)
    assert result["predictions_created"] == 2
    assert evaluation.evaluation_detail(store, test["id"])["frames"][0]["split"] == "test"
    with pytest.raises(ValueError, match="never a test"):
        promote(store, test)
    for changes in (
        {"confidence_threshold": 0.5},
        {"iou_threshold": 0.6},
        {"model_ids": list(reversed(MODEL_IDS))},
        {"device": "cuda"},
    ):
        with pytest.raises(ValueError, match="same dataset|retain validation"):
            queue(
                workspace,
                **{
                    "split": "test",
                    "model_ids": MODEL_IDS,
                    "confidence_threshold": 0.65,
                    "validation_evaluation_id": validation["id"],
                    **changes,
                },
            )


def test_test_audit_rejects_unfinished_validation_and_changed_checkpoint(workspace):
    store, _, _, available = workspace
    validation = queue(workspace)
    with pytest.raises(ValueError, match="finish successfully"):
        queue(workspace, split="test", validation_evaluation_id=validation["id"])
    run(store, validation)
    available[0]["weight_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="retain validation"):
        queue(workspace, split="test", validation_evaluation_id=validation["id"])


def test_reference_history_explicit_reason_and_compare_and_swap(workspace):
    store = workspace[0]
    row = queue(workspace, model_ids=MODEL_IDS)
    run(store, row)
    assert evaluation.reference_history(store) == {"current": None, "history": []}
    with pytest.raises(ValueError, match="reason"):
        promote(store, row, notes=" ")
    first = promote(store, row)
    assert first["metadata"]["metrics"]["summary"]["map50"] == pytest.approx(1)
    with pytest.raises(evaluation.ReferenceConflict, match="Reference changed"):
        promote(store, row, model_id=MODEL_IDS[1])
    second = promote(store, row, model_id=MODEL_IDS[1], expected_previous_id=first["id"])
    saved = evaluation.reference_history(Store(store.root))
    assert saved == {"current": second, "history": [second, first]}
    assert second["metadata"]["previous_reference_id"] == first["id"]


def test_reference_refuses_modified_or_missing_checkpoint(workspace):
    store, _, _, available = workspace
    row = queue(workspace)
    run(store, row)
    available[0]["weight_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="changed"):
        promote(store, row)
    available[0]["status"] = "missing_weights"
    with pytest.raises(RuntimeError, match="not ready"):
        promote(store, row)
    assert evaluation.reference_history(store)["current"] is None


def test_metrics_publication_honors_cancellation_requested_during_scoring(workspace, monkeypatch):
    store = workspace[0]
    row = queue(workspace)
    original = evaluation.evaluate_predictions

    def score_and_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        JobManager(store).cancel(row["job_id"])
        return result

    monkeypatch.setattr(evaluation, "evaluate_predictions", score_and_cancel)
    result = run(store, row)
    assert result["cancelled"]
    assert evaluation.evaluation_detail(store, row["id"])["models"][0]["metrics"] is None
    assert store.get("jobs", row["job_id"])["status"] == "cancelled"


def test_unknown_parent_or_changed_ancestor_digest_cannot_claim_independence(workspace):
    _, _, frames, available = workspace
    trained = trained_entry(available[0], frames, parent_model_id="unknown-parent")
    available.append(trained)
    with pytest.raises(ValueError, match="unknown ancestor"):
        queue(workspace, model_ids=[trained["id"]])
    trained["parent_model_id"] = available[0]["id"]
    trained["provenance"]["parent_weight_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="parent digest"):
        queue(workspace, model_ids=[trained["id"]])


def test_cancel_before_loading_does_not_publish_model_or_scores(workspace):
    store = workspace[0]
    row = queue(workspace)
    calls = []
    result = run(store, row, cancelled=lambda: True, detector_factory=fixture_detector(calls))
    assert result["cancelled"] and result["predictions_created"] == 0
    assert calls == []
    assert store.list("evaluation_models") == []
    assert store.list("evaluation_predictions") == []


def test_second_frozen_image_tamper_preserves_partial_predictions_without_scores(workspace):
    store, dataset, _, _ = workspace
    row = queue(workspace)
    frames = [frame for frame in dataset["manifest"]["frames"] if frame["split"] == "val"]
    store.artifact_path(frames[1]["image_path"]).write_bytes(b"tampered second fixture")
    with pytest.raises(ValueError, match="hash"):
        run(store, row)
    detail = evaluation.evaluation_detail(store, row["id"])
    assert len(detail["predictions"]) == 1
    assert detail["models"][0]["metrics"] is None


def test_worker_refuses_queued_metrics_protocol_change(workspace, monkeypatch):
    store = workspace[0]
    row = queue(workspace)
    original = evaluation.get_protocol
    monkeypatch.setattr(
        evaluation, "get_protocol", lambda **kwargs: {**original(**kwargs), "changed": True}
    )
    with pytest.raises(ValueError, match="protocol changed"):
        run(store, row)
    assert store.list("evaluation_models") == []


def test_test_audit_rejects_different_dataset_even_with_identical_images(workspace):
    store, dataset, frames, _ = workspace
    validation = queue(workspace)
    run(store, validation)
    second = create_dataset(
        store,
        name="Second synthetic frozen release",
        parent_id=dataset["id"],
        frame_ids=[frame["id"] for frame in frames],
        splits={"train": "train", "val": "val", "test": "test"},
    )
    with pytest.raises(ValueError, match="same dataset"):
        queue(
            workspace,
            dataset_id=second["id"],
            split="test",
            validation_evaluation_id=validation["id"],
        )
