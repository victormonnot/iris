"""Read-only regression inspection using real metric calculations on synthetic labels."""

from copy import deepcopy

import pytest
from test_datasets import add_frame

from iris import evaluation_analysis, metrics
from iris.annotations import save_annotation
from iris.datasets import create_dataset
from iris.evaluation import evaluation_detail
from iris.evaluation_analysis import analyze_evaluation
from iris.metrics import evaluate_predictions, get_protocol
from iris.store import Store, new_id, now

MODELS = ["fixture-baseline", "fixture-candidate"]
PERSON_0 = [2, 2, 12, 22]
PERSON_1 = [22, 2, 32, 22]
CAR = [40, 30, 70, 50]
EXTRA = [1, 40, 12, 55]


def detection(label, box, score=0.9, **extra):
    return {
        "label": label,
        "label_id": {"person": 1, "car": 3, "bicycle": 2}[label],
        "box": box,
        "score": score,
        **extra,
    }


def record_evaluation(store, dataset, outputs, model_ids=MODELS):
    identifier, job_id = new_id(), new_id()
    frames = [frame for frame in dataset["manifest"]["frames"] if frame["split"] == "val"]
    protocol = get_protocol(0.5, 0.5)
    config = {
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "frame_ids": [frame["frame_id"] for frame in frames],
        "frame_hashes": {frame["frame_id"]: frame["sha256"] for frame in frames},
        "model_hashes": {model_id: str(index + 1) * 64 for index, model_id in enumerate(model_ids)},
        "model_names": {model_id: f"Synthetic {model_id}" for model_id in model_ids},
        "model_lineages": {model_id: [{"origin": "fixture"}] for model_id in model_ids},
        "confidence_threshold": 0.5,
        "iou_threshold": 0.5,
        "taxonomy_id": "iris-objects-v1",
        "class_mapping": {"person": 1, "car": 3},
        "protocol": protocol,
        "device": "cpu",
        "warnings": ["Synthetic fixture; no real detector result"],
    }
    store.insert(
        "jobs",
        {
            "id": job_id,
            "kind": "evaluate",
            "status": "succeeded",
            "params": {"evaluation_id": identifier},
            "created_at": now(),
        },
    )
    store.insert(
        "evaluations",
        {
            "id": identifier,
            "name": "Synthetic error changes",
            "dataset_id": dataset["id"],
            "split": "val",
            "model_ids": model_ids,
            "config": config,
            "job_id": job_id,
            "created_at": now(),
        },
    )
    for model_id in model_ids:
        model_row_id = new_id()
        predictions = [
            {
                "id": new_id(),
                "evaluation_id": identifier,
                "evaluation_model_id": model_row_id,
                "model_id": model_id,
                "frame_id": frame["frame_id"],
                "detections": outputs[model_id][index],
                "input_size": [frame["width"], frame["height"]],
                "timing": {},
                "created_at": now(),
            }
            for index, frame in enumerate(frames)
        ]
        store.insert(
            "evaluation_models",
            {
                "id": model_row_id,
                "evaluation_id": identifier,
                "model_id": model_id,
                "metadata": {
                    "model_id": model_id,
                    "weight_sha256": config["model_hashes"][model_id],
                    "model_name": config["model_names"][model_id],
                    "protocol": protocol,
                    "lineage": config["model_lineages"][model_id],
                    "device": "cpu",
                },
                "metrics": evaluate_predictions(frames, predictions),
                "created_at": now(),
            },
        )
        for prediction in predictions:
            store.insert("evaluation_predictions", prediction)
    return evaluation_detail(store, identifier)


@pytest.fixture
def saved(tmp_path):
    store = Store(tmp_path / "workspace")
    frames = [
        add_frame(store, tmp_path, group="train", color=(10, 20, 30)),
        add_frame(store, tmp_path, group="val", color=(20, 30, 40)),
        add_frame(
            store,
            tmp_path,
            group="val",
            color=(30, 40, 50),
            boxes=[
                {"id": "person-0", "label": "person", "box": PERSON_0},
                {"id": "person-1", "label": "person", "box": PERSON_1},
                {"id": "car", "label": "car", "box": CAR},
            ],
        ),
    ]
    dataset = create_dataset(
        store,
        name="Synthetic analysis release",
        frame_ids=[frame["id"] for frame in frames],
        splits={"train": "train", "val": "val"},
    )
    outputs = {
        MODELS[0]: [
            [detection("person", EXTRA)],
            [detection("person", PERSON_0), detection("car", CAR), detection("person", EXTRA)],
        ],
        MODELS[1]: [
            [detection("car", EXTRA)],
            [
                detection("bicycle", EXTRA, 0.99),
                detection("person", PERSON_0, 0.2),
                detection("person", PERSON_1),
                detection("car", CAR, native_label_id=2),
                detection("car", EXTRA),
                detection("car", [25, 40, 35, 55]),
            ],
        ],
    }
    detail = record_evaluation(store, dataset, outputs)
    return store, detail, dataset, outputs


def test_new_miss_and_recovery_on_same_frame_despite_unchanged_fn_count(saved):
    store, detail, _, _ = saved
    result = analyze_evaluation(store, detail["id"])
    assert result["protocol"] == "iris-error-analysis-v1"
    assert result["comparison"] == {
        "baseline_model_id": MODELS[0],
        "candidate_model_id": MODELS[1],
    }
    assert result["models"] == [
        {"id": identifier, "name": f"Synthetic {identifier}"} for identifier in MODELS
    ]
    frame = result["frames"][1]
    assert frame["counts"]["all"]["models"][MODELS[0]] == {"tp": 2, "fp": 1, "fn": 1}
    assert frame["counts"]["all"]["models"][MODELS[1]] == {"tp": 2, "fp": 2, "fn": 1}
    assert frame["counts"]["all"]["changes"] == {
        "new_misses": 1,
        "recovered": 1,
        "fp_delta": 1,
        "new_miss_indices": [0],
        "recovered_indices": [1],
    }
    assert frame["counts"]["person"]["changes"]["fp_delta"] == -1
    assert frame["counts"]["car"]["changes"]["fp_delta"] == 2
    assert result["summary"]["all"] == {
        "frame_count": 2,
        "ground_truth_count": 3,
        "models": {
            MODELS[0]: {"tp": 2, "fp": 2, "fn": 1, "error_frames": 2},
            MODELS[1]: {"tp": 2, "fp": 3, "fn": 1, "error_frames": 2},
        },
        "changes": {"new_misses": 1, "recovered": 1, "fp_delta": 1},
    }
    assert result["summary"]["person"]["changes"]["fp_delta"] == -2
    assert result["summary"]["car"]["changes"]["fp_delta"] == 3


def test_source_order_negatives_and_original_detection_indices_are_preserved(saved):
    store, detail, _, _ = saved
    result = analyze_evaluation(store, detail["id"])
    assert [frame["frame_id"] for frame in result["frames"]] == detail["config"]["frame_ids"]
    assert [frame["position"] for frame in result["frames"]] == [1, 2]
    negative = result["frames"][0]
    assert negative["counts"]["all"]["ground_truth_count"] == 0
    assert negative["counts"]["all"]["models"][MODELS[0]] == {"tp": 0, "fp": 1, "fn": 0}
    assert negative["counts"]["car"]["models"][MODELS[1]] == {"tp": 0, "fp": 1, "fn": 0}
    assert result["frames"][1]["counts"]["car"]["models"][MODELS[1]]["tp"] == 1
    candidate = next(model for model in detail["models"] if model["model_id"] == MODELS[1])
    assert candidate["metrics"]["frames"][1]["false_positives"] == [4, 5]
    assert any("outside person/car" in warning for warning in result["warnings"])


def test_single_model_has_no_comparison_or_change_claims(saved):
    store, _, dataset, outputs = saved
    detail = record_evaluation(store, dataset, outputs, model_ids=MODELS[:1])
    result = analyze_evaluation(store, detail["id"])
    assert result["comparison"] is None
    assert all(stats["changes"] is None for stats in result["summary"].values())
    assert all(
        stats["changes"] is None for frame in result["frames"] for stats in frame["counts"].values()
    )


def test_configured_model_order_controls_baseline_not_database_row_order(saved):
    store, detail, _, _ = saved
    store.update("evaluations", detail["id"], {"model_ids": list(reversed(MODELS))})
    result = analyze_evaluation(store, detail["id"])
    assert result["comparison"]["baseline_model_id"] == MODELS[1]
    changes = result["frames"][1]["counts"]["all"]["changes"]
    assert changes["new_miss_indices"] == [1]
    assert changes["recovered_indices"] == [0]
    assert changes["fp_delta"] == -1


def test_class_confusion_is_false_negative_and_other_class_false_positive(saved):
    store, _, dataset, outputs = saved
    outputs[MODELS[1]][1] = [detection("car", PERSON_0)]
    detail = record_evaluation(store, dataset, outputs)
    result = analyze_evaluation(store, detail["id"])
    frame = result["frames"][1]["counts"]
    assert frame["person"]["models"][MODELS[1]] == {"tp": 0, "fp": 0, "fn": 2}
    assert frame["car"]["models"][MODELS[1]] == {"tp": 0, "fp": 1, "fn": 1}
    assert frame["all"]["changes"]["new_miss_indices"] == [0, 2]


def test_read_only_and_restart_ignore_live_edits_missing_pixels_and_models(saved, monkeypatch):
    store, detail, dataset, _ = saved
    before = analyze_evaluation(store, detail["id"])
    for frame in detail["frames"]:
        save_annotation(store, frame["frame_id"], expected_revision=1, boxes=[], decisions={})
        store.update("frames", frame["frame_id"], {"selected": False})
    for frame in dataset["manifest"]["frames"]:
        store.artifact_path(frame["image_path"]).unlink()
        store.artifact_path(store.get("frames", frame["frame_id"])["path"]).unlink()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Analysis must not recompute scores or match predictions")

    monkeypatch.setattr(metrics, "evaluate_predictions", forbidden)
    monkeypatch.setattr(metrics, "_average_precision", forbidden)
    monkeypatch.setattr(store, "insert", forbidden)
    monkeypatch.setattr(store, "update", forbidden)
    with store.connect() as connection:
        dump = "\n".join(connection.iterdump())
    assert analyze_evaluation(store, detail["id"]) == before
    with store.connect() as connection:
        assert "\n".join(connection.iterdump()) == dump
    assert analyze_evaluation(Store(store.root), detail["id"]) == before


def test_saved_engine_versions_remain_readable_after_dependency_upgrade(saved, monkeypatch):
    store, detail, _, _ = saved
    actual = evaluation_analysis.get_protocol

    def upgraded(*args):
        protocol = actual(*args)
        return {**protocol, "engine_version": "future", "numpy_version": "future"}

    monkeypatch.setattr(evaluation_analysis, "get_protocol", upgraded)
    assert analyze_evaluation(store, detail["id"])["summary"]["all"]["ground_truth_count"] == 3


def test_missing_evaluation_is_not_found(saved):
    with pytest.raises(KeyError):
        analyze_evaluation(saved[0], "absent")


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled", "interrupted"])
def test_partial_evaluation_cannot_publish_analysis(saved, status):
    store, detail, _, _ = saved
    store.update("jobs", detail["job_id"], {"status": status})
    with pytest.raises(ValueError, match="finish successfully"):
        analyze_evaluation(store, detail["id"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda detail: detail["models"][0].update(metrics=None),
        lambda detail: detail["models"].pop(),
        lambda detail: detail["predictions"].pop(),
        lambda detail: detail["config"]["frame_ids"].reverse(),
        lambda detail: detail["config"]["frame_hashes"].update(extra="0" * 64),
        lambda detail: detail["config"].update(class_mapping={"person": 1, "car": 2}),
        lambda detail: detail["models"][0]["metadata"].update(weight_sha256="f" * 64),
        lambda detail: detail["models"][0]["metadata"].update(model_id=MODELS[1]),
        lambda detail: detail["predictions"][0].update(evaluation_model_id="wrong"),
        lambda detail: detail["predictions"][0].update(input_size=[80, 61]),
        lambda detail: detail["models"][0]["metrics"]["frames"][0].update(false_positives=[99]),
        lambda detail: detail["models"][0]["metrics"]["frames"][1].update(false_negatives=[True]),
        lambda detail: detail["models"][0]["metrics"]["frames"][1].update(false_negatives=[1, 1]),
        lambda detail: detail["models"][0]["metrics"]["frames"][1].update(tp=99),
        lambda detail: detail["models"][0]["metrics"]["summary"].update(fp=99),
        lambda detail: detail["models"][0]["metrics"]["per_class"][0].update(support=99),
        lambda detail: detail["models"][0]["metrics"]["frames"][1]["matches"][0].update(iou=0.1),
        lambda detail: detail["models"][0]["metrics"]["frames"][1]["matches"][0].update(
            label="car"
        ),
        lambda detail: detail["models"][0]["metrics"]["frames"][1]["matches"][0].update(
            ground_truth_index=1
        ),
    ],
)
def test_inconsistent_saved_results_fail_closed(saved, monkeypatch, mutation):
    store, detail, _, _ = saved
    damaged = deepcopy(detail)
    mutation(damaged)
    monkeypatch.setattr(evaluation_analysis, "evaluation_detail", lambda *_args: damaged)
    with pytest.raises(ValueError):
        analyze_evaluation(store, detail["id"])


def test_native_id_two_is_not_a_car_without_canonical_remapping(saved, monkeypatch):
    store, detail, _, _ = saved
    damaged = deepcopy(detail)
    prediction = next(row for row in damaged["predictions"] if len(row["detections"]) == 6)
    prediction["detections"][3]["label_id"] = 2
    monkeypatch.setattr(evaluation_analysis, "evaluation_detail", lambda *_args: damaged)
    with pytest.raises(ValueError, match="canonical COCO"):
        analyze_evaluation(store, detail["id"])


def test_low_confidence_or_nonproject_source_indices_cannot_be_counted(saved, monkeypatch):
    store, detail, _, _ = saved
    for source_index in (0, 1):
        damaged = deepcopy(detail)
        model = next(model for model in damaged["models"] if model["model_id"] == MODELS[1])
        model["metrics"]["frames"][1]["false_positives"][0] = source_index
        monkeypatch.setattr(
            evaluation_analysis, "evaluation_detail", lambda *_args, result=damaged: result
        )
        with pytest.raises(ValueError, match="retained source"):
            analyze_evaluation(store, detail["id"])


def test_protocol_threshold_must_match_saved_configuration(saved, monkeypatch):
    store, detail, _, _ = saved
    damaged = deepcopy(detail)
    damaged["config"]["confidence_threshold"] = 0.6
    monkeypatch.setattr(evaluation_analysis, "evaluation_detail", lambda *_args: damaged)
    with pytest.raises(ValueError, match="protocol"):
        analyze_evaluation(store, detail["id"])


def test_recorded_resolved_cuda_device_is_compatible_without_loading_gpu(saved, monkeypatch):
    store, detail, _, _ = saved
    damaged = deepcopy(detail)
    damaged["config"]["device"] = "cuda"
    for model in damaged["models"]:
        model["metadata"]["device"] = "cuda:0"
    monkeypatch.setattr(evaluation_analysis, "evaluation_detail", lambda *_args: damaged)
    assert analyze_evaluation(store, detail["id"])["summary"]["all"]["ground_truth_count"] == 3
    for unsupported in ("cpu", "cuda:-1", "cuda:any", "mps"):
        damaged["models"][0]["metadata"]["device"] = unsupported
        with pytest.raises(ValueError, match="inconsistent"):
            analyze_evaluation(store, detail["id"])
