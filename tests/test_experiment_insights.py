"""Pure report analysis of synthetic saved counts; no detector or metric execution."""

from copy import deepcopy

import pytest

from iris.experiment_insights import build_insights


def fixture(*, single=False, aggregate="all"):
    runtime = {
        "device": "cpu",
        "hardware": "Fixture CPU",
        "platform": "Fixture OS",
        "torch_version": "2.10.0+cpu",
        "torchvision_version": "0.25.0+cpu",
        "precision": "float32",
        "threads": 4,
        "interop_threads": 1,
        "timing_protocol": {
            "version": "torchvision-forward-v1",
            "batch_size": 1,
            "warmup_in_timings": False,
            "total_ms": "Decode and detector call; excludes warmup and weights",
            "decode_ms": "Read normalized PNG and verify pixels",
        },
    }
    lanes = [
        {
            "id": identifier,
            "variant": "full",
            "metrics": {
                "summary": {
                    "map": 0.25 + position * 0.125,
                    "map50": None if position == 0 else 0.7,
                    "tp": 1 + position,
                    "fp": 2 - position,
                    "fn": 2 - position,
                    "frame_count": 2,
                }
            },
            "timing": {"frame_count": 2, "measured_frame_count": 2, "mean_total_ms": 20},
            "runtime": deepcopy(runtime),
        }
        for position, identifier in enumerate(
            ("baseline",) if single else ("baseline", "candidate")
        )
    ]
    snapshot = {"lanes": lanes, "error_analysis": {"aggregate_filter": aggregate}}
    available = []
    detail = {"frames": []}
    for position, identifier in enumerate(("first", "second")):
        runs = {
            lane["id"]: {"tp": 0 if position else 1, "fp": 1, "fn": 0 if position else 2}
            for lane in lanes
        }
        counts = {
            "ground_truth_count": 0 if position else 3,
            "runs": runs,
            "changes": None
            if single
            else {
                "new_misses": 0 if position else 1,
                "recovered": 0 if position else 1,
                "fp_delta": 0,
                "new_miss_indices": [] if position else [1],
                "recovered_indices": [] if position else [2],
            },
        }
        available.append(
            {
                "frame_id": identifier,
                "scene_group": "Scene A",
                "counts": {aggregate: counts, "widget": deepcopy(counts)},
            }
        )
        detail["frames"].append(
            {
                "frame_id": identifier,
                "source": {
                    "asset_id": f"asset-{identifier}",
                    "kind": "image",
                    "filename": f"{identifier}.png",
                    "timestamp_seconds": None,
                },
            }
        )
    return snapshot, available, detail


def test_frozen_counts_are_aggregated_without_mutating_or_reinterpreting_ap():
    inputs = fixture(aggregate="__all__")
    original = deepcopy(inputs)
    result = build_insights(*inputs)
    assert inputs == original
    assert result["protocol"] == "iris-experiment-insights-v1"
    assert result["quality_delta"]["map"] == 0.125
    assert result["quality_delta"]["map50"] is None
    assert result["quality_delta"]["precision"] is None
    assert result["quality_delta"]["fp"] == -1
    assert result["quality_delta"]["frame_count"] == 0
    scene = result["scenes"][0]
    assert scene["frame_count"] == 2
    assert scene["negative_frame_count"] == 1
    assert scene["counts"]["__all__"] == {
        "frame_count": 2,
        "ground_truth_count": 3,
        "runs": {
            run_id: {"tp": 1, "fp": 2, "fn": 2, "error_frames": 2}
            for run_id in ("baseline", "candidate")
        },
        "changes": {"new_misses": 1, "recovered": 1, "fp_delta": 0},
    }
    assert "ap" not in scene["counts"]["widget"]
    assert result["frame_changes"] == {"first": "mixed", "second": "unchanged"}


@pytest.mark.parametrize(
    ("new_misses", "recovered", "fp_delta", "expected"),
    [
        (0, 0, 0, "unchanged"),
        (1, 0, 0, "regressed"),
        (0, 1, 0, "improved"),
        (0, 0, 1, "regressed"),
        (0, 0, -1, "improved"),
        (1, 1, 0, "mixed"),
        (1, 0, -2, "mixed"),
        (0, 1, 2, "mixed"),
    ],
)
def test_changes_keep_recoveries_and_regressions_separate(
    new_misses, recovered, fp_delta, expected
):
    inputs = fixture()
    inputs[1][0]["counts"]["all"]["changes"].update(
        new_misses=new_misses, recovered=recovered, fp_delta=fp_delta
    )
    assert build_insights(*inputs)["frame_changes"]["first"] == expected


def test_single_pipeline_has_no_deltas_or_invented_baseline():
    result = build_insights(*fixture(single=True))
    assert result["quality_delta"] is None
    assert set(result["frame_changes"].values()) == {"single"}
    assert result["scenes"][0]["counts"]["all"]["changes"] is None
    assert not result["timing"]["comparable"]
    assert "two evaluated" in result["timing"]["reasons"][0]


def test_suggestions_are_bounded_diverse_unique_and_deterministic():
    snapshot, frames, detail = fixture()
    original_frame = deepcopy(frames[0])
    original_source = deepcopy(detail["frames"][0])
    frames.clear()
    detail["frames"].clear()
    for index in range(20):
        frame = deepcopy(original_frame)
        frame.update(frame_id=f"frame-{index}", scene_group=f"Scene {index % 4}")
        frame["counts"]["all"]["changes"].update(
            new_misses=1 if index % 3 in (0, 1) else 0,
            recovered=1 if index % 3 in (0, 2) else 0,
        )
        frames.append(frame)
        detail["frames"].append({**original_source, "frame_id": frame["frame_id"]})
    result = build_insights(snapshot, frames, detail)
    suggestions = result["suggested_examples"]
    identifiers = [item["frame_id"] for item in suggestions]
    assert len(identifiers) == len(set(identifiers)) == 6
    assert [result["frame_changes"][item] for item in identifiers[:3]] == [
        "mixed",
        "regressed",
        "improved",
    ]
    assert (
        len({frames[int(item.removeprefix("frame-"))]["scene_group"] for item in identifiers}) == 4
    )
    assert build_insights(snapshot, frames, detail)["suggested_examples"] == suggestions


def test_unchanged_negative_false_positives_are_suggested_but_empty_correct_images_are_not():
    inputs = fixture()
    result = build_insights(*inputs)
    assert result["suggested_examples"][1] == {
        "frame_id": "second",
        "reason": "False positives on a reviewed image with no labeled objects",
    }
    for run in inputs[1][1]["counts"]["all"]["runs"].values():
        run["fp"] = 0
    assert len(build_insights(*inputs)["suggested_examples"]) == 1


def test_source_kinds_preserve_unknowns_and_do_not_infer_video_from_timestamp():
    inputs = fixture()
    first, second = (frame["source"] for frame in inputs[2]["frames"])
    first["timestamp_seconds"] = 5
    second.pop("kind")
    second["timestamp_seconds"] = 10
    sampling = build_insights(*inputs)["sampling"]
    assert sampling["video_sources"] == []
    assert sampling["still_image_count"] == sampling["unknown_source_count"] == 1
    assert sampling["continuous_inference"] is False


def test_video_sources_group_original_identity_and_sanitize_filenames_without_claiming_coverage():
    inputs = fixture()
    for index, frame in enumerate(inputs[2]["frames"]):
        frame["source"].update(
            kind="video",
            asset_id="same-source",
            sha256="a" * 64,
            filename=r"C:\private\flight.mp4",
            timestamp_seconds=42 - 40 * index,
        )
    sampling = build_insights(*inputs)["sampling"]
    assert sampling["video_sources"] == [
        {
            "source_id": "same-source",
            "filename": "flight.mp4",
            "frame_count": 2,
            "timestamps_available": 2,
            "first_timestamp_seconds": 2,
            "last_timestamp_seconds": 42,
            "timestamps_approximate": True,
        }
    ]
    assert "do not establish interval coverage" in sampling["warning"]
    assert sampling["still_image_count"] == sampling["unknown_source_count"] == 0


@pytest.mark.parametrize("timestamp", [None, -1, float("nan"), float("inf"), True, "5"])
def test_missing_or_invalid_timestamps_remain_unavailable(timestamp):
    inputs = fixture()
    inputs[2]["frames"][0]["source"].update(kind="video", timestamp_seconds=timestamp)
    video = build_insights(*inputs)["sampling"]["video_sources"][0]
    assert video["timestamps_available"] == 0
    assert video["first_timestamp_seconds"] is video["last_timestamp_seconds"] is None


def test_unidentified_videos_are_not_merged_by_shared_filename():
    inputs = fixture()
    for frame in inputs[2]["frames"]:
        frame["source"] = {"kind": "video", "filename": "same.mp4"}
    assert len(build_insights(*inputs)["sampling"]["video_sources"]) == 2


def test_same_recorded_runtime_allows_whole_pipeline_comparison_including_full_vs_tiled():
    inputs = fixture()
    assert build_insights(*inputs)["timing"]["comparable"]
    candidate = inputs[0]["lanes"][1]
    candidate["variant"] = "tiled"
    candidate["runtime"]["timing_protocol"].update(
        version="torchvision-tiled-v1", total_ms="Decode, verification, crops, passes and merge"
    )
    result = build_insights(*inputs)["timing"]
    assert result["comparable"]
    assert "tiled runs" in result["scope"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hardware", "Other CPU"),
        ("device", "cuda:0"),
        ("torch_version", "unknown"),
        ("precision", "float16"),
        ("threads", 8),
        ("interop_threads", 2),
    ],
)
def test_different_recorded_hardware_or_runtime_blocks_timing_comparison(field, value):
    inputs = fixture()
    inputs[0]["lanes"][1]["runtime"][field] = value
    assert not build_insights(*inputs)["timing"]["comparable"]


def test_missing_runtime_or_partial_timings_are_not_a_comparable_pair():
    inputs = fixture()
    for lane in inputs[0]["lanes"]:
        lane["runtime"].pop("interop_threads")
    inputs[0]["lanes"][1]["timing"]["measured_frame_count"] = 1
    result = build_insights(*inputs)["timing"]
    assert not result["comparable"]
    assert len(result["reasons"]) == 2


def test_unknown_or_mismatched_protocols_are_not_comparable():
    inputs = fixture()
    inputs[0]["lanes"][1]["runtime"]["timing_protocol"]["total_ms"] = "Includes loading weights"
    assert not build_insights(*inputs)["timing"]["comparable"]
    for lane in inputs[0]["lanes"]:
        lane["runtime"]["timing_protocol"]["version"] = "unknown"
    assert not build_insights(*inputs)["timing"]["comparable"]


def test_cuda_metadata_must_be_present_and_equal():
    inputs = fixture()
    for lane in inputs[0]["lanes"]:
        lane["runtime"]["device"] = "cuda:0"
    assert not build_insights(*inputs)["timing"]["comparable"]
    for lane in inputs[0]["lanes"]:
        lane["runtime"]["cuda"] = {
            "runtime": "12.8",
            "cudnn": 90000,
            "index": 0,
            "name": "Fixture GPU",
            "capability": [8, 9],
            "total_memory": 8 * 1024**3,
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "cudnn_benchmark": False,
        }
    assert build_insights(*inputs)["timing"]["comparable"]
    inputs[0]["lanes"][1]["runtime"]["cuda"]["name"] = "different-gpu"
    assert not build_insights(*inputs)["timing"]["comparable"]


@pytest.mark.parametrize("mutation", ["empty", "too_many", "duplicate", "reordered"])
def test_frame_identity_and_bounds_are_checked(mutation):
    snapshot, frames, detail = fixture()
    if mutation == "empty":
        frames.clear()
    elif mutation == "too_many":
        frames *= 501
    elif mutation == "duplicate":
        frames[1]["frame_id"] = frames[0]["frame_id"]
    else:
        frames.reverse()
    with pytest.raises(ValueError):
        build_insights(snapshot, frames, detail)
