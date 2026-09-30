"""Saved error comparisons distinguish inference variants of the same checkpoint."""

from copy import deepcopy

import pytest
from test_evaluation_analysis import MODELS, record_evaluation, saved

from iris import evaluation_analysis, metrics
from iris.evaluation import _timing_protocol, evaluation_detail
from iris.evaluation_analysis import analyze_evaluation
from iris.metrics import evaluate_predictions, get_protocol
from iris.store import Store
from iris.tiling import tile_boxes, validate_tiling_config

legacy_saved = saved


def versioned_evaluation(saved, mode="paired"):
    store, detail, dataset, outputs = saved
    config = deepcopy(detail["config"])
    model_ids = MODELS[:1] if mode == "paired" else detail["model_ids"]
    tiling = validate_tiling_config(tile_size=128, overlap=0.2)
    inference = {"mode": mode}
    if mode != "full":
        inference.update(algorithm="iris-tiling-v1", tiling=tiling)
    config.update(
        inference=inference,
        lanes=[
            {"model_id": identifier, "variant": variant}
            for identifier in model_ids
            for variant in (["full", "tiled"] if mode == "paired" else [mode])
        ],
        protocol=get_protocol(max_detections=100 if mode == "full" else 300),
        timing_protocol=_timing_protocol("full" if mode == "full" else "tiled"),
    )
    for key in ("model_names", "model_hashes", "model_lineages"):
        config[key] = {identifier: config[key][identifier] for identifier in model_ids}
    store.update("evaluations", detail["id"], {"config": config, "model_ids": model_ids})
    for model in detail["models"]:
        identifier = model["model_id"] if mode != "paired" else model_ids[0]
        variant = (
            ("full" if model["model_id"] == MODELS[0] else "tiled") if mode == "paired" else mode
        )
        recorded = {"variant": variant}
        if variant == "tiled":
            recorded.update(
                algorithm="iris-tiling-v1",
                **tiling,
                tile_boxes={
                    frame["frame_id"]: tile_boxes(frame["width"], frame["height"], tiling)
                    for frame in detail["frames"]
                },
            )
        predictions = []
        for row in detail["predictions"]:
            if row["evaluation_model_id"] != model["id"]:
                continue
            prediction = deepcopy(row)
            prediction.update(model_id=identifier, metadata={})
            if variant == "tiled":
                prediction["metadata"]["tiles"] = [
                    {
                        "tile_index": 0,
                        "box": recorded["tile_boxes"][row["frame_id"]][0],
                        "input_size": row["input_size"],
                        "detections": deepcopy(row["detections"]),
                        "timing": {
                            "preprocess_ms": 0.0,
                            "inference_ms": 0.0,
                            "postprocess_ms": 0.0,
                            "total_ms": 0.0,
                        },
                    }
                ]
                prediction["detections"] = [
                    {**item, "tile_index": 0} for item in prediction["detections"]
                ]
            store.update(
                "evaluation_predictions",
                row["id"],
                {key: prediction[key] for key in ("model_id", "metadata", "detections")},
            )
            predictions.append(prediction)
        store.update(
            "evaluation_models",
            model["id"],
            {
                "model_id": identifier,
                "variant": variant,
                "metadata": {
                    **model["metadata"],
                    "model_id": identifier,
                    "model_name": config["model_names"][identifier],
                    "weight_sha256": config["model_hashes"][identifier],
                    "lineage": config["model_lineages"][identifier],
                    "protocol": config["protocol"],
                    "inference": recorded,
                    "timing_protocol": _timing_protocol(variant),
                },
                "metrics": evaluate_predictions(
                    detail["frames"], predictions, max_detections=100 if mode == "full" else 300
                ),
            },
        )
    return store, evaluation_detail(store, detail["id"]), dataset, outputs


@pytest.fixture
def paired_saved(legacy_saved):
    return versioned_evaluation(legacy_saved)


def test_same_checkpoint_variants_keep_separate_runs_and_error_changes(paired_saved):
    store, detail, _, _ = paired_saved
    result = analyze_evaluation(store, detail["id"])
    baseline, candidate = [lane["evaluation_model_id"] for lane in detail["lanes"]]
    assert result["protocol"] == "iris-error-analysis-v2"
    assert "models" not in result
    assert result["runs"] == [
        {
            "id": baseline,
            "model_id": MODELS[0],
            "variant": "full",
            "name": f"Synthetic {MODELS[0]} · Full image",
        },
        {
            "id": candidate,
            "model_id": MODELS[0],
            "variant": "tiled",
            "name": f"Synthetic {MODELS[0]} · Tiled",
        },
    ]
    assert result["comparison"] == {"baseline_run_id": baseline, "candidate_run_id": candidate}
    assert result["summary"]["all"] == {
        "frame_count": 2,
        "ground_truth_count": 3,
        "runs": {
            baseline: {"tp": 2, "fp": 2, "fn": 1, "error_frames": 2},
            candidate: {"tp": 2, "fp": 3, "fn": 1, "error_frames": 2},
        },
        "changes": {"new_misses": 1, "recovered": 1, "fp_delta": 1},
    }
    frame = result["frames"][1]["counts"]
    assert frame["all"]["changes"] == {
        "new_misses": 1,
        "recovered": 1,
        "fp_delta": 1,
        "new_miss_indices": [0],
        "recovered_indices": [1],
    }
    assert frame["person"]["changes"]["fp_delta"] == -1
    assert frame["car"]["changes"]["fp_delta"] == 2
    negative = result["frames"][0]["counts"]
    assert negative["all"]["ground_truth_count"] == 0
    assert negative["all"]["runs"][candidate] == {"tp": 0, "fp": 1, "fn": 0}


@pytest.mark.parametrize("mode", ["full", "tiled"])
def test_two_checkpoint_runs_use_versioned_identities(legacy_saved, mode):
    store, detail, _, _ = versioned_evaluation(legacy_saved, mode)
    result = analyze_evaluation(store, detail["id"])
    assert [run["model_id"] for run in result["runs"]] == MODELS
    assert all(run["variant"] == mode for run in result["runs"])
    assert set(result["summary"]["all"]["runs"]) == {run["id"] for run in result["runs"]}
    assert result["summary"]["all"]["changes"] == {"new_misses": 1, "recovered": 1, "fp_delta": 1}


def test_database_row_order_does_not_reverse_full_and_tiled(paired_saved, monkeypatch):
    store, detail, _, _ = paired_saved
    before = analyze_evaluation(store, detail["id"])
    reordered = deepcopy(detail)
    reordered["models"].reverse()
    reordered["predictions"].reverse()
    monkeypatch.setattr(evaluation_analysis, "evaluation_detail", lambda *_: reordered)
    assert analyze_evaluation(store, detail["id"]) == before


@pytest.mark.parametrize("mode", ["full", "tiled"])
def test_one_saved_run_has_no_comparison_or_improvement_claim(legacy_saved, mode):
    store, _, dataset, outputs = legacy_saved
    detail = record_evaluation(store, dataset, outputs, model_ids=MODELS[:1])
    store, detail, _, _ = versioned_evaluation((store, detail, dataset, outputs), mode)
    result = analyze_evaluation(store, detail["id"])
    assert len(result["runs"]) == 1
    assert result["comparison"] is None
    assert all(stats["changes"] is None for stats in result["summary"].values())


def test_explicit_empty_predictions_count_misses_and_keep_negative_frames(paired_saved):
    store, detail, _, _ = paired_saved
    for model in detail["models"]:
        predictions = []
        for row in detail["predictions"]:
            if row["evaluation_model_id"] != model["id"]:
                continue
            row["detections"] = []
            for tile in row["metadata"].get("tiles", []):
                tile["detections"] = []
            store.update(
                "evaluation_predictions",
                row["id"],
                {
                    "detections": [],
                    "metadata": row["metadata"],
                },
            )
            predictions.append(row)
        store.update(
            "evaluation_models",
            model["id"],
            {
                "metrics": evaluate_predictions(detail["frames"], predictions, max_detections=300),
            },
        )
    result = analyze_evaluation(store, detail["id"])
    assert result["summary"]["all"]["changes"] == {
        "new_misses": 0,
        "recovered": 0,
        "fp_delta": 0,
    }
    assert all(
        stats == {"tp": 0, "fp": 0, "fn": 3, "error_frames": 1}
        for stats in result["summary"]["all"]["runs"].values()
    )
    assert result["frames"][0]["counts"]["all"]["ground_truth_count"] == 0


def test_analysis_reads_saved_results_without_pixels_detectors_or_new_metrics(
    paired_saved, monkeypatch
):
    store, detail, dataset, _ = paired_saved
    expected = analyze_evaluation(store, detail["id"])
    for frame in dataset["manifest"]["frames"]:
        store.artifact_path(frame["image_path"]).unlink()
        store.artifact_path(store.get("frames", frame["frame_id"])["path"]).unlink()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Saved error analysis must not rerun an evaluation")

    monkeypatch.setattr(metrics, "evaluate_predictions", forbidden)
    monkeypatch.setattr(metrics, "_average_precision", forbidden)
    monkeypatch.setattr("iris.evaluation.TorchvisionDetector", forbidden)
    monkeypatch.setattr(store, "insert", forbidden)
    monkeypatch.setattr(store, "update", forbidden)
    with store.connect() as connection:
        before = "\n".join(connection.iterdump())
    assert analyze_evaluation(store, detail["id"]) == expected
    with store.connect() as connection:
        assert "\n".join(connection.iterdump()) == before
    assert analyze_evaluation(Store(store.root), detail["id"]) == expected


@pytest.mark.parametrize("status", ["queued", "running", "cancelled", "failed", "interrupted"])
def test_incomplete_runs_do_not_publish_error_changes(paired_saved, status):
    store, detail, _, _ = paired_saved
    store.update("jobs", detail["job_id"], {"status": status})
    with pytest.raises(ValueError, match="finish successfully"):
        analyze_evaluation(store, detail["id"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda detail: detail["models"].pop(),
        lambda detail: detail["predictions"].pop(),
        lambda detail: detail["predictions"].append(deepcopy(detail["predictions"][0])),
        lambda detail: detail["predictions"][0].update(evaluation_model_id="orphan"),
        lambda detail: detail["predictions"][0].update(model_id="wrong-checkpoint"),
        lambda detail: detail["models"][0].update(variant="unknown"),
        lambda detail: detail["models"][0]["metadata"].update(inference={"variant": "tiled"}),
        lambda detail: detail["models"][0]["metadata"].update(weight_sha256="f" * 64),
        lambda detail: detail["models"][0]["metadata"].update(lineage=[]),
        lambda detail: detail["models"][0]["metadata"].update(timing_protocol={}),
        lambda detail: detail["config"]["lanes"].reverse(),
        lambda detail: detail["config"]["inference"].update(mode="full"),
        lambda detail: detail["config"]["inference"].update(algorithm="unrecognized"),
        lambda detail: detail["config"]["inference"]["tiling"].update(overlap=0.5),
        lambda detail: detail["lanes"][0].update(evaluation_model_id="other-run"),
        lambda detail: detail["config"]["protocol"].update(id="coco-bbox-iris-v1"),
        lambda detail: detail["config"].update(timing_protocol={}),
    ],
)
def test_inconsistent_run_identities_fail_closed(paired_saved, monkeypatch, mutation):
    store, detail, _, _ = paired_saved
    damaged = deepcopy(detail)
    mutation(damaged)
    monkeypatch.setattr(evaluation_analysis, "evaluation_detail", lambda *_: damaged)
    with pytest.raises(ValueError):
        analyze_evaluation(store, detail["id"])


def test_predictions_cannot_be_swapped_between_variants_of_one_checkpoint(
    paired_saved, monkeypatch
):
    store, detail, _, _ = paired_saved
    damaged = deepcopy(detail)
    left, right = [lane["evaluation_model_id"] for lane in damaged["lanes"]]
    for prediction in damaged["predictions"]:
        prediction["evaluation_model_id"] = (
            right if prediction["evaluation_model_id"] == left else left
        )
    monkeypatch.setattr(evaluation_analysis, "evaluation_detail", lambda *_: damaged)
    with pytest.raises(ValueError, match="tiled|crop"):
        analyze_evaluation(store, detail["id"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda prediction: prediction.update(metadata={}),
        lambda prediction: prediction["metadata"].update(tiles=[]),
        lambda prediction: prediction["metadata"]["tiles"][0].update(tile_index=True),
        lambda prediction: prediction["metadata"]["tiles"][0].update(box=[1, 0, 80, 60]),
        lambda prediction: prediction["metadata"]["tiles"][0].update(input_size=[79, 60]),
        lambda prediction: prediction["detections"][0].update(tile_index=1),
        lambda prediction: prediction["detections"][0].update(box=[3, 3, 9, 9]),
    ],
)
def test_merged_predictions_must_retain_their_actual_crop_source(
    paired_saved, monkeypatch, mutation
):
    store, detail, _, _ = paired_saved
    damaged = deepcopy(detail)
    tiled = next(
        lane["evaluation_model_id"] for lane in detail["lanes"] if lane["variant"] == "tiled"
    )
    prediction = next(row for row in damaged["predictions"] if row["evaluation_model_id"] == tiled)
    mutation(prediction)
    monkeypatch.setattr(evaluation_analysis, "evaluation_detail", lambda *_: damaged)
    with pytest.raises(ValueError):
        analyze_evaluation(store, detail["id"])
