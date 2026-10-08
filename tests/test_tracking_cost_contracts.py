"""Portable cost reports recompute scheduling and summaries from bounded raw evidence."""

import math
import subprocess
import sys
from copy import deepcopy

import pytest
from test_tracking_cost_runtime import cost_report as cost_report
from test_tracking_cost_runtime import example, fake_factories

from iris import tracking_cost_contracts as contracts
from iris import tracking_cost_runtime as runtime


def request(**changes):
    return {
        "lane_index": 0,
        "device": "cpu",
        "repeats": 1,
        "policy": "offline_all",
        "cadence_fps": None,
        **changes,
    }


def set_path(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


@pytest.mark.parametrize(
    "changes",
    [
        {"lane_index": True},
        {"lane_index": -1},
        {"lane_index": 2},
        {"device": "auto"},
        {"device": "cuda:0"},
        {"device": None},
        {"repeats": True},
        {"repeats": 0},
        {"repeats": 6},
        {"repeats": 1.0},
        {"policy": "live"},
        {"cadence_fps": 30},
        {"policy": "simulated_latest"},
        {"policy": "simulated_latest", "cadence_fps": True},
        {"policy": "simulated_latest", "cadence_fps": "30"},
        {"policy": "simulated_latest", "cadence_fps": 0.01},
        {"policy": "simulated_latest", "cadence_fps": 241},
        {"policy": "simulated_latest", "cadence_fps": math.nan},
        {"policy": "simulated_latest", "cadence_fps": 10**400},
        {"undeclared": "field"},
    ],
)
def test_request_rejects_implicit_device_clock_or_unbounded_work(changes):
    with pytest.raises(ValueError):
        contracts.validate_cost_request(request(**changes))


def test_cadence_normalization_and_nearest_rank_p95_are_explicit():
    assert (
        contracts.validate_cost_request(request(policy="simulated_latest", cadence_fps=30))[
            "cadence_fps"
        ]
        == 30.0
    )
    stats = contracts.distribution(list(range(1, 21)))
    assert stats == {
        "count": 20,
        "min": 1,
        "median": 10.5,
        "p95": 19,
        "max": 20,
        "mean": 10.5,
        "total": 210,
    }
    assert contracts.distribution([]) is None


def test_scheduler_exact_arrival_boundary_keeps_latest_and_records_only_available_drops():
    settings = request(policy="simulated_latest", cadence_fps=100)
    selected, dropped, start = contracts.next_frame([0, 1, 2, 3], 1, 20, settings)
    assert (selected, dropped, start) == (2, [1], 20)
    assert contracts.next_frame([0, 3, 8], 1, 20, settings) == (1, [], 30)
    assert contracts.next_frame([0, 3, 8], 1, 100, settings) == (2, [3], 100)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("complete",), False),
        (("schema",), "future-v1"),
        (("request", "lane_index"), 1),
        (("request", "device"), "cuda"),
        (("source", "cache_fingerprint"), "3" * 64),
        (("source", "lane_index"), False),
        (("profile", "seed"), 123),
        (("detector_config", "min_score"), 0.2),
        (("execution", "started_at"), "2026-10-08T12:00:00"),
        (("execution", "host", "logical_cpus"), True),
        (("execution", "host", "affinity_cpus"), [1, 1]),
        (("execution", "tracker_metadata", "execution_policy"), []),
        (("execution", "tracker_metadata", "execution_policy", "seed"), 999),
        (("execution", "tracker_metadata", "execution_policy", "device"), "cuda"),
        (("execution", "tracker_metadata", "execution_policy", "native_time_step"), True),
        (("execution", "pipeline_sources", "tracking_cost_runtime.py"), "bad-hash"),
        (("execution", "warmup", "tracker_reset"), False),
        (("passes", 0, "frames", 0, "outputs_sha256"), "bad-hash"),
        (("passes", 0, "frames", 0, "frame_index"), 1),
        (("passes", 0, "frames", 0, "timestamp_seconds"), 7.0),
        (("passes", 0, "frames", 0, "input_size"), [10, 20]),
        (("passes", 0, "frames", 0, "timing", "pipeline_ms"), 1.0),
        (("passes", 0, "frames", 0, "timing", "detector_inference_ms"), 100.0),
        (("passes", 0, "frames", 0, "timing", "tracker_adapter_ms"), 1.0),
        (("passes", 0, "frames", 0, "timing", "tracking_image_ms"), 1.0),
        (("passes", 0, "frames", 0, "work", "observation_count"), 5),
        (("passes", 0, "frames", 0, "work", "forward_passes"), 2),
        (("passes", 0, "frames", 0, "work", "tile_count"), 1),
        (("passes", 0, "frames", 0, "schedule"), {}),
        (("passes", 0, "dropped_frame_indices"), [1]),
        (("passes", 0, "pass_index"), True),
        (("passes", 0, "wall_ms"), 1.0),
        (("passes", 0, "memory", "cuda"), {"device": "cuda:0"}),
        (("passes", 0, "memory", "rss_start_bytes"), -1),
        (("passes", 0, "memory", "rss_sampled_peak_bytes"), 1),
        (("summary", "processed_frames"), 99),
        (("summary", "stages_ms", "pipeline_ms", "p95"), 0.0),
        (("protocol", "scheduling"), "real capture"),
        (("limitations",), []),
    ],
)
def test_changed_sources_timings_memory_and_summaries_are_rejected(cost_report, path, replacement):
    comparison, original = cost_report
    report = deepcopy(original)
    set_path(report, path, replacement)
    with pytest.raises(ValueError):
        contracts.validate_cost_report(report, comparison)


def test_truncated_repetition_cannot_be_published_even_with_recomputed_summary(cost_report):
    comparison, report = deepcopy(cost_report)
    report["passes"][0]["frames"].pop()
    report["summary"] = contracts.summarize(report)
    with pytest.raises(ValueError, match="pending"):
        contracts.validate_cost_report(report, comparison)


def test_simulation_report_schedule_and_drop_partition_cannot_be_rewritten(tmp_path, monkeypatch):
    store, comparison = example(tmp_path, indices=(0, 1, 2, 3, 4))
    factories, _, _, _, _ = fake_factories(monkeypatch, comparison)
    original = runtime.measure_tracking_cost(
        store, comparison, policy="simulated_latest", cadence_fps=100, **factories
    )
    for field, value in (
        ("arrival_ms", 99),
        ("start_ms", 26),
        ("finish_ms", 60),
        ("queue_delay_ms", 0),
        ("latency_ms", 0),
    ):
        report = deepcopy(original)
        report["passes"][0]["frames"][1]["schedule"][field] = value
        report["summary"] = contracts.summarize(report)
        with pytest.raises(ValueError, match="schedule"):
            contracts.validate_cost_report(report, comparison)
    report = deepcopy(original)
    report["passes"][0]["dropped_frame_indices"] = [3, 1]
    with pytest.raises(ValueError, match="dropped"):
        contracts.validate_cost_report(report, comparison)


def test_fast_short_clip_output_cadence_does_not_gain_a_free_initial_frame(tmp_path, monkeypatch):
    store, comparison = example(tmp_path)
    factories, _, _, _, _ = fake_factories(monkeypatch, comparison, detector_ms=5, tracker_ms=5)
    report = runtime.measure_tracking_cost(
        store, comparison, policy="simulated_latest", cadence_fps=30, **factories
    )
    assert report["summary"]["service_fps"] == pytest.approx(100)
    assert report["summary"]["simulated_output_fps"] == pytest.approx(30)


def test_one_output_has_no_output_cadence_interval(tmp_path, monkeypatch):
    store, comparison = example(tmp_path, indices=(0,))
    factories, _, _, _, _ = fake_factories(monkeypatch, comparison)
    report = runtime.measure_tracking_cost(
        store, comparison, policy="simulated_latest", cadence_fps=30, **factories
    )
    assert report["summary"]["simulated_output_fps"] is None


def test_gpu_peak_tampering_is_rejected_even_with_recomputed_summary(tmp_path, monkeypatch):
    store, comparison = example(tmp_path)
    factories, _, _, _, _ = fake_factories(monkeypatch, comparison)
    original = runtime.measure_tracking_cost(store, comparison, device="cuda", **factories)
    for key, value in (
        ("device", "cuda:1"),
        ("allocated_peak_bytes", 10),
        ("reserved_peak_bytes", 25),
        ("reserved_start_bytes", 10),
    ):
        report = deepcopy(original)
        report["passes"][0]["memory"]["cuda"][key] = value
        report["summary"] = contracts.summarize(report)
        with pytest.raises(ValueError, match="CUDA"):
            contracts.validate_cost_report(report, comparison)


def test_finite_bounded_json_and_complete_schema_are_mandatory(cost_report, monkeypatch):
    comparison, report = cost_report
    for bad in (None, [], {**report, "unknown": "field"}, {**report, "summary": math.nan}):
        with pytest.raises(ValueError):
            contracts.validate_cost_report(bad, comparison)
    monkeypatch.setattr(contracts, "MAX_REPORT_BYTES", 10)
    with pytest.raises(ValueError, match="size limit"):
        contracts.validate_cost_report(report, comparison)


def test_cost_contracts_import_without_optional_tracking_or_ml_modules():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from iris.tracking_cost_contracts import cost_status; cost_status(); "
            "assert not {'torch', 'torchvision', 'scipy', 'lap', 'cython_bbox'} & set(sys.modules)",
        ],
        check=True,
    )


def test_known_rss_sample_cannot_have_unknown_peak(cost_report):
    comparison, report = deepcopy(cost_report)
    report["passes"][0]["memory"]["rss_sampled_peak_bytes"] = None
    report["summary"] = contracts.summarize(report)
    with pytest.raises(ValueError, match="RSS boundary"):
        contracts.validate_cost_report(report, comparison)


def test_tiled_fresh_pipeline_and_work_counts_match_actual_geometry(tmp_path, monkeypatch):
    from iris import tiling

    store, comparison = example(tmp_path, inference_mode="tiled")
    factories, clock, _, _, _ = fake_factories(monkeypatch, comparison)
    monkeypatch.setattr(tiling, "time", clock)
    original = runtime.measure_tracking_cost(store, comparison, **factories)
    assert original["passes"][0]["frames"][0]["work"]["tile_count"] == 1
    for key in ("forward_passes", "tile_count"):
        report = deepcopy(original)
        report["passes"][0]["frames"][0]["work"][key] = 2
        report["summary"] = contracts.summarize(report)
        with pytest.raises(ValueError, match="crop geometry"):
            contracts.validate_cost_report(report, comparison)
