"""Frozen full/tiled evaluation lifecycle with explicit synthetic detector outputs."""

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
OBJECTS = [
    {"id": "person", "label": "person", "box": [210, 20, 230, 100]},
    {"id": "car", "label": "car", "box": [500, 40, 550, 100]},
]


@pytest.fixture
def tiled_workspace(tmp_path, monkeypatch):
    store = Store(tmp_path / "workspace")
    frames = []
    for index, split in enumerate(("train", "val", "val", "test"), 1):
        session = store.insert(
            "sessions",
            {
                "id": new_id(),
                "name": f"Synthetic {index}",
                "scene_group": split,
                "created_at": now(),
            },
        )
        path = tmp_path / f"synthetic-{index}.png"
        image = Image.new("RGB", (640, 128))
        # Encode each crop's source x coordinate and fixture identity in its pixels.
        image.putdata([(x % 256, index, x // 256) for _y in range(128) for x in range(640)])
        image.save(path)
        image.close()
        asset = import_asset(store, session["id"], path, path.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        store.update("frames", frame["id"], {"selected": True})
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=deepcopy(OBJECTS),
            decisions={},
            status="validated",
            reviewer="Synthetic fixture reviewer",
        )
        frames.append(frame)
    dataset = create_dataset(
        store,
        name="Synthetic tiled evaluation release",
        frame_ids=[row["id"] for row in frames],
        splits={"train": "train", "val": "val", "test": "test"},
    )
    available = [
        {**models.get_spec(identifier), "status": "ready", "weight_sha256": str(index) * 64}
        for index, identifier in enumerate(MODEL_IDS, 1)
    ]
    monkeypatch.setattr(evaluation, "catalog", lambda _root: available)
    return store, dataset, frames, available


def settings(workspace, **overrides):
    return {
        "name": "Synthetic full versus tiles",
        "dataset_id": workspace[1]["id"],
        "model_ids": MODEL_IDS[:1],
        "inference_mode": "paired",
        "tile_size": 256,
        "overlap": 0.25,
        **overrides,
    }


def queue(workspace, **overrides):
    return evaluation.create_evaluation(
        workspace[0], JobManager(workspace[0]), **settings(workspace, **overrides)
    )


def detector_factory(calls, *, fail_after=None, after_predict=None):
    class SyntheticDetector:
        def __init__(self, root, model_id, *, device):
            self.metadata = {
                "model_id": model_id,
                "device": device,
                "runtime": "synthetic fixture",
                "weight_sha256": str(MODEL_IDS.index(model_id) + 1) * 64,
            }
            calls.append(("load", model_id))

        def warmup(self, image):
            calls.append(("warmup", image.size))

        def predict(self, image):
            completed = sum(call[0] == "predict" for call in calls)
            if fail_after is not None and completed == fail_after:
                raise RuntimeError("Synthetic tile failure")
            red, _green, blue = image.getpixel((0, 0))
            origin_x = red + blue * 256
            calls.append(("predict", image.size, origin_x))
            if image.width == 640:
                detections = [
                    {"label": "person", "label_id": 1, "box": OBJECTS[0]["box"], "score": 0.9},
                    {"label": "car", "label_id": 3, "box": [50, 40, 100, 100], "score": 0.8},
                ]
            else:
                detections = [
                    {
                        "label": target["label"],
                        "label_id": {"person": 1, "car": 3}[target["label"]],
                        "box": [
                            target["box"][0] - origin_x,
                            target["box"][1],
                            target["box"][2] - origin_x,
                            target["box"][3],
                        ],
                        "score": 0.9 - origin_x / 10000,
                    }
                    for target in OBJECTS
                    if origin_x <= target["box"][0] < target["box"][2] <= origin_x + image.width
                ]
            if after_predict is not None:
                after_predict(image)
            return {
                "input_size": list(image.size),
                "detections": detections,
                "timing": {
                    "preprocess_ms": 1,
                    "inference_ms": 2,
                    "postprocess_ms": 1,
                    "total_ms": 4,
                },
            }

    return SyntheticDetector


def run(workspace, row, *, calls=None, cancelled=lambda: False, **factory_options):
    return evaluation.run_evaluation(
        workspace[0],
        row["id"],
        lambda *_args: None,
        cancelled,
        detector_factory=detector_factory([] if calls is None else calls, **factory_options),
    )


def promote(workspace, row, **overrides):
    return evaluation.promote_reference(
        workspace[0],
        evaluation_id=row["id"],
        model_id=MODEL_IDS[0],
        reviewer="Fixture reviewer",
        notes="Synthetic pipeline comparison only",
        **overrides,
    )


def test_preview_is_read_only_and_counts_warmups_and_complete_frozen_split(tiled_workspace):
    store, _dataset, frames, _available = tiled_workspace
    plan = evaluation.preview_evaluation(store, **settings(tiled_workspace))
    assert plan["forward_passes"] == 8
    assert plan["warmup_passes"] == 2
    assert plan["total_forward_passes"] == 10
    assert plan["limits"]["max_forward_passes"] == 4096
    assert [item["frame_id"] for item in plan["tiles"]] == [item["id"] for item in frames[1:3]]
    assert all(item["tile_count"] == 3 for item in plan["tiles"])
    assert store.list("jobs") == store.list("evaluations") == store.list("evaluation_models") == []
    row = queue(tiled_workspace)
    assert row["config"]["work"] == {
        key: value for key, value in plan.items() if key not in {"lanes", "inference"}
    }
    assert row["lanes"] == [
        {"model_id": MODEL_IDS[0], "variant": variant, "evaluation_model_id": None}
        for variant in ("full", "tiled")
    ]


def test_same_checkpoint_scores_merged_source_boxes_and_records_cost(tiled_workspace):
    store = tiled_workspace[0]
    row = queue(tiled_workspace)
    calls = []
    result = run(tiled_workspace, row, calls=calls)
    assert result["models_total"] == result["models_completed"] == 1
    assert result["runs_total"] == result["runs_completed"] == 2
    assert result["predictions_created"] == 4
    assert not result["cancelled"]
    assert sum(call[0] == "load" for call in calls) == 1
    assert [call[1] for call in calls if call[0] == "warmup"] == [(640, 128), (256, 128)]
    detail = evaluation.evaluation_detail(store, row["id"])
    assert detail["job"]["status"] == "succeeded"
    assert detail["job"]["result"] == result
    assert detail["model_ids"] == MODEL_IDS[:1]
    by_variant = {model["variant"]: model for model in detail["models"]}
    full, tiled = (by_variant[variant] for variant in ("full", "tiled"))
    assert full["metrics"]["summary"]["map"] == pytest.approx(0.5)
    assert tiled["metrics"]["summary"]["map"] == pytest.approx(1)
    assert (
        full["metrics"]["summary"]["tp"],
        full["metrics"]["summary"]["fp"],
        full["metrics"]["summary"]["fn"],
    ) == (2, 2, 2)
    assert (
        tiled["metrics"]["summary"]["tp"],
        tiled["metrics"]["summary"]["fp"],
        tiled["metrics"]["summary"]["fn"],
    ) == (4, 0, 0)
    assert full["metrics"]["protocol"] == tiled["metrics"]["protocol"] == row["config"]["protocol"]
    for prediction in detail["predictions"]:
        is_tiled = prediction["evaluation_model_id"] == tiled["id"]
        assert prediction["input_size"] == [640, 128]
        assert prediction["model_id"] == MODEL_IDS[0]
        assert prediction["timing"]["forward_passes"] == (3 if is_tiled else 1)
        assert prediction["timing"]["inference_ms"] == (6 if is_tiled else 2)
        if is_tiled:
            assert [item["box"] for item in prediction["detections"]] == [
                target["box"] for target in OBJECTS
            ]
            assert len(prediction["metadata"]["tiles"]) == 3
            assert prediction["timing"]["raw_detection_count"] == 3
            assert prediction["timing"]["kept_detection_count"] == 2
            assert all(value >= 0 for value in prediction["timing"].values())
    assert [lane["evaluation_model_id"] for lane in detail["lanes"]] == [full["id"], tiled["id"]]


@pytest.mark.parametrize(
    "changes",
    [
        {"inference_mode": "invalid"},
        {"inference_mode": "paired", "model_ids": MODEL_IDS},
        {"tile_size": 127},
        {"tile_size": True},
        {"overlap": 0.6},
        {"overlap": float("nan")},
    ],
)
def test_invalid_pipeline_settings_never_queue(tiled_workspace, changes):
    with pytest.raises(ValueError):
        queue(tiled_workspace, **changes)
    assert tiled_workspace[0].list("jobs") == []


def test_tiled_two_checkpoints_keep_distinct_run_metrics(tiled_workspace):
    row = queue(tiled_workspace, inference_mode="tiled", model_ids=MODEL_IDS)
    result = run(tiled_workspace, row)
    assert result["models_completed"] == result["runs_completed"] == 2
    detail = evaluation.evaluation_detail(tiled_workspace[0], row["id"])
    assert {model["model_id"] for model in detail["models"]} == set(MODEL_IDS)
    assert all(model["variant"] == "tiled" for model in detail["models"])
    assert all(model["metrics"]["summary"]["map"] == pytest.approx(1) for model in detail["models"])


def test_cancellation_within_first_tile_discards_incomplete_image(tiled_workspace):
    row = queue(tiled_workspace, inference_mode="tiled")
    stopped = False

    def cancel_after_tile(_image):
        nonlocal stopped
        stopped = True

    result = run(tiled_workspace, row, cancelled=lambda: stopped, after_predict=cancel_after_tile)
    assert result["cancelled"] and result["predictions_created"] == 0
    assert result["runs_completed"] == 0
    assert tiled_workspace[0].list("evaluation_predictions") == []
    assert all(model["metrics"] is None for model in tiled_workspace[0].list("evaluation_models"))


def test_second_variant_failure_keeps_full_scores_but_blocks_reference_and_audit(tiled_workspace):
    row = queue(tiled_workspace)
    with pytest.raises(RuntimeError, match="Synthetic tile failure"):
        run(tiled_workspace, row, fail_after=3)
    store = tiled_workspace[0]
    detail = evaluation.evaluation_detail(store, row["id"])
    assert len(detail["predictions"]) == 2
    assert detail["models"][0]["metrics"] is not None
    assert detail["models"][1]["metrics"] is None
    store.update("jobs", row["job_id"], {"status": "failed"})
    with pytest.raises(ValueError, match="finish successfully"):
        promote(tiled_workspace, row, variant="full")
    with pytest.raises(ValueError, match="finish successfully"):
        queue(tiled_workspace, split="test", validation_evaluation_id=row["id"])


def test_test_audit_keeps_pipeline_but_plans_its_own_frozen_split(tiled_workspace):
    validation = queue(tiled_workspace)
    run(tiled_workspace, validation)
    test = queue(tiled_workspace, split="test", validation_evaluation_id=validation["id"])
    assert test["config"]["inference"] == validation["config"]["inference"]
    assert test["config"]["lanes"] == validation["config"]["lanes"]
    assert test["config"]["work"]["total_forward_passes"] == 6
    assert test["config"]["work"] != validation["config"]["work"]
    assert run(tiled_workspace, test)["predictions_created"] == 2
    with pytest.raises(ValueError, match="never a test"):
        promote(tiled_workspace, test, variant="tiled")
    for changes in (
        {"inference_mode": "full"},
        {"inference_mode": "tiled"},
        {"tile_size": 320},
        {"overlap": 0.1},
    ):
        with pytest.raises(ValueError, match="retain validation"):
            queue(
                tiled_workspace, split="test", validation_evaluation_id=validation["id"], **changes
            )


def test_reference_requires_variant_and_saves_exact_pipeline(tiled_workspace):
    row = queue(tiled_workspace)
    run(tiled_workspace, row)
    with pytest.raises(ValueError, match="explicit inference variant"):
        promote(tiled_workspace, row)
    selected = promote(tiled_workspace, row, variant="tiled")
    detail = evaluation.evaluation_detail(tiled_workspace[0], row["id"])
    tiled = next(model for model in detail["models"] if model["variant"] == "tiled")
    assert selected["model_id"] == MODEL_IDS[0]
    assert selected["metadata"]["evaluation_model_id"] == tiled["id"]
    assert selected["metadata"]["variant"] == "tiled"
    assert selected["metadata"]["inference"] == tiled["metadata"]["inference"]
    assert selected["metadata"]["inference"]["merge_iou"] == 0.5
    assert selected["metadata"]["inference"]["max_detections"] == 300
    assert selected["metadata"]["device"] == "cpu"
    assert selected["metadata"]["timing_protocol"] == tiled["metadata"]["timing_protocol"]
    full = promote(tiled_workspace, row, variant="full", expected_previous_id=selected["id"])
    assert full["metadata"]["inference"] == {"variant": "full"}


@pytest.mark.parametrize(
    "change", ["algorithm", "merge_iou", "max_detections", "work", "lanes", "timing"]
)
def test_worker_rejects_changed_frozen_pipeline_before_loading(tiled_workspace, change):
    store = tiled_workspace[0]
    row = queue(tiled_workspace)
    config = deepcopy(row["config"])
    if change == "algorithm":
        config["inference"]["algorithm"] = "untracked"
    elif change in {"merge_iou", "max_detections"}:
        config["inference"]["tiling"][change] = 0.6 if change == "merge_iou" else 100
    elif change == "work":
        config["work"]["forward_passes"] += 1
    elif change == "lanes":
        config["lanes"].reverse()
    else:
        config["timing_protocol"]["version"] = "untracked"
    store.update("evaluations", row["id"], {"config": config})
    calls = []
    with pytest.raises(ValueError, match="protocol|plan|variants"):
        run(tiled_workspace, row, calls=calls)
    assert calls == []
    assert store.list("evaluation_models") == []


@pytest.mark.parametrize(
    "change",
    [
        "variant",
        "tile_size",
        "tile_boxes",
        "metrics_protocol",
        "weight_sha256",
        "lineage",
        "model_id",
    ],
)
def test_inconsistent_saved_run_cannot_become_reference_or_validation_evidence(
    tiled_workspace, change
):
    store = tiled_workspace[0]
    row = queue(tiled_workspace)
    run(tiled_workspace, row)
    tiled = next(model for model in store.list("evaluation_models") if model["variant"] == "tiled")
    metadata, metrics = deepcopy(tiled["metadata"]), deepcopy(tiled["metrics"])
    if change == "metrics_protocol":
        metrics["protocol"]["confidence_threshold"] = 0.7
    elif change == "tile_boxes":
        metadata["inference"]["tile_boxes"] = {}
    elif change in {"weight_sha256", "lineage", "model_id"}:
        metadata[change] = "inconsistent"
    else:
        metadata["inference"][change] = "full" if change == "variant" else 320
    store.update("evaluation_models", tiled["id"], {"metadata": metadata, "metrics": metrics})
    with pytest.raises(ValueError, match="inference settings"):
        promote(tiled_workspace, row, variant="tiled")
    with pytest.raises(ValueError, match="inference settings"):
        queue(tiled_workspace, split="test", validation_evaluation_id=row["id"])


def test_legacy_full_evaluation_still_reads_audits_and_promotes(tiled_workspace):
    store = tiled_workspace[0]
    row = queue(tiled_workspace, inference_mode="full")
    run(tiled_workspace, row)
    config = deepcopy(row["config"])
    for key in ("inference", "lanes", "work"):
        config.pop(key)
    store.update("evaluations", row["id"], {"config": config})
    model = store.list("evaluation_models")[0]
    metadata = deepcopy(model["metadata"])
    metadata.pop("inference")
    metadata.pop("timing_protocol")
    store.update("evaluation_models", model["id"], {"metadata": metadata})
    detail = evaluation.evaluation_detail(store, row["id"])
    assert detail["lanes"] == [
        {"model_id": MODEL_IDS[0], "variant": "full", "evaluation_model_id": model["id"]}
    ]
    reference = promote(tiled_workspace, row)
    assert reference["metadata"]["inference"] == {"variant": "full"}
    test = queue(
        tiled_workspace, inference_mode="full", split="test", validation_evaluation_id=row["id"]
    )
    assert run(tiled_workspace, test)["runs_completed"] == 1


def test_tiled_still_rejects_checkpoint_training_overlap(tiled_workspace):
    _store, _dataset, frames, available = tiled_workspace
    available.append(
        {
            **deepcopy(available[0]),
            "id": "trained_fixture",
            "origin": "trained",
            "taxonomy_id": "iris-objects-v1",
            "parent_model_id": MODEL_IDS[0],
            "provenance": {
                "training_scene_groups": ["val"],
                "training_frame_hashes": [frames[0]["sha256"]],
                "parent_weight_sha256": available[0]["weight_sha256"],
            },
        }
    )
    with pytest.raises(ValueError, match="used to train"):
        queue(tiled_workspace, model_ids=["trained_fixture"])


def test_evaluation_work_limit_preserves_full_split_capacity_and_bounds_tiled_cost():
    frames = [{"frame_id": str(index), "width": 640, "height": 128} for index in range(1000)]
    full_lanes = [{"model_id": identifier, "variant": "full"} for identifier in MODEL_IDS]
    assert (
        evaluation._evaluation_work(frames, full_lanes, {"mode": "full"})["total_forward_passes"]
        == 2002
    )
    tiled_lanes = [{"model_id": MODEL_IDS[0], "variant": "tiled"}]
    inference = {
        "mode": "tiled",
        "algorithm": "iris-tiling-v1",
        "tiling": {"tile_size": 128, "overlap": 0.0, "merge_iou": 0.5, "max_detections": 300},
    }
    assert (
        evaluation._evaluation_work(frames[:819], tiled_lanes, inference)["total_forward_passes"]
        == 4096
    )
    with pytest.raises(ValueError, match="limit is 4096"):
        evaluation._evaluation_work(frames[:820], tiled_lanes, inference)


def test_tiled_worker_scores_more_than_one_hundred_merged_detections(tiled_workspace):
    row = queue(tiled_workspace, inference_mode="tiled")

    class CrowdedSyntheticDetector(detector_factory([])):
        def predict(self, image):
            result = super().predict(image)
            result["detections"] = [
                {
                    "label_id": 1,
                    "label": "person",
                    "score": 0.9,
                    "box": [20 + index * 2, 3, 21 + index * 2, 8],
                }
                for index in range(60)
            ]
            return result

    evaluation.run_evaluation(
        tiled_workspace[0],
        row["id"],
        lambda *_: None,
        lambda: False,
        detector_factory=CrowdedSyntheticDetector,
    )
    detail = evaluation.evaluation_detail(tiled_workspace[0], row["id"])
    assert all(len(prediction["detections"]) == 180 for prediction in detail["predictions"])
    summary = detail["models"][0]["metrics"]["summary"]
    assert summary["fp"] == summary["native_prediction_count"] == 360
    assert summary["fn"] == 4
    assert detail["models"][0]["metrics"]["protocol"]["max_saved_detections_per_image"] == 300


def test_cancel_during_second_variant_scoring_does_not_publish_final_metrics(
    tiled_workspace, monkeypatch
):
    store = tiled_workspace[0]
    row = queue(tiled_workspace)
    original = evaluation.evaluate_predictions
    calls = 0

    def score_and_cancel(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            JobManager(store).cancel(row["job_id"])
        return result

    monkeypatch.setattr(evaluation, "evaluate_predictions", score_and_cancel)
    result = run(tiled_workspace, row)
    detail = evaluation.evaluation_detail(store, row["id"])
    assert result["cancelled"] and result["runs_completed"] == 1
    assert detail["job"]["status"] == "cancelled"
    assert detail["models"][0]["metrics"] is not None
    assert detail["models"][1]["metrics"] is None
    with pytest.raises(ValueError, match="finish successfully"):
        promote(tiled_workspace, row, variant="full")
