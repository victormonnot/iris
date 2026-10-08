"""Fresh pipeline boundaries and virtual arrival scheduling without optional ML."""

import hashlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image
from test_temporal_contracts import sequence
from test_temporal_detector import frozen_config, runtime_metadata

from iris import temporal_detector
from iris import tracking_cost_contracts as contracts
from iris import tracking_cost_runtime as runtime
from iris.media import _pixel_hash
from iris.tracking_contracts import FRAME_SCHEMA, make_profile


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def perf_counter(self):
        return self.seconds

    def advance(self, milliseconds):
        self.seconds += milliseconds / 1000


class Store:
    def __init__(self, root, records):
        self.root, self.records = root, records

    def get(self, table, identifier):
        assert table == "frames"
        return deepcopy(self.records.get(identifier))

    def artifact_path(self, path):
        return self.root / path


def example(tmp_path, *, indices=(0, 1, 2), gmc=False, inference_mode="full"):
    manifest, records = sequence(), {}
    template = manifest["frames"][0]
    manifest["frames"] = []
    manifest["clip"] = {"start_frame": indices[0], "end_frame": indices[-1]}
    manifest["gaps"] = [
        {"start_frame": a + 1, "end_frame": b - 1, "reason": "skipped"}
        for a, b in zip(indices, indices[1:], strict=False)
        if b != a + 1
    ]
    for index in indices:
        path = tmp_path / f"frame-{index}.png"
        with Image.new("RGB", (100, 80), (index, 2, 3)) as image:
            image.save(path)
            pixel_hash = _pixel_hash(image)
        frame = {
            **template,
            "frame_id": f"frame-{index}",
            "frame_index": index,
            "timestamp_seconds": index / 10,
            "sha256": pixel_hash,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        manifest["frames"].append(frame)
        records[frame["frame_id"]] = {
            "id": frame["frame_id"],
            "path": path.name,
            "width": 100,
            "height": 80,
            "sha256": pixel_hash,
        }
    config = frozen_config(inference_mode=inference_mode)
    manifest = contracts.validate_sequence_manifest(manifest)
    cache = {
        "id": "cache",
        "fingerprint": "1" * 64,
        "result_sha256": "2" * 64,
        "config": {"detector": config},
    }
    comparison = {
        "id": "comparison",
        "report": {
            "sequence": manifest,
            "lanes": [
                {
                    "name": name,
                    "report": {
                        "profile": make_profile(
                            algorithm,
                            class_ids=[1, 3],
                            gmc_method="sparseOptFlow" if gmc and index else "none",
                        ),
                        "cache": deepcopy(cache),
                    },
                }
                for index, (name, algorithm) in enumerate(
                    (("ByteTrack", "bytetrack"), ("BoT-SORT", "botsort"))
                )
            ],
        },
    }
    return Store(tmp_path, records), comparison


def fake_factories(monkeypatch, comparison, *, detector_ms=20, tracker_ms=5, mutate=None):
    clock, calls, trackers, detectors = Clock(), [], [], []
    monkeypatch.setattr(runtime, "time", clock)
    original = comparison["report"]["lanes"][0]["report"]["cache"]["config"]["detector"]
    monkeypatch.setattr(
        runtime,
        "_prepare_detector",
        lambda root, frozen, device: {**deepcopy(frozen), "device": device},
    )
    monkeypatch.setattr(temporal_detector, "_runtime", lambda *_: deepcopy(original["runtime"]))
    monkeypatch.setattr(runtime, "_rss_bytes", lambda: 100)
    monkeypatch.setattr(runtime, "_process_peak_bytes", lambda: 150)

    class Detector:
        def __init__(self, root, config):
            self.metadata = runtime_metadata(config)
            self.device = self.metadata["device"]
            self.torch = SimpleNamespace(cuda=FakeCuda())
            detectors.append(self)

        def predict(self, image):
            marker = image.getpixel((0, 0))[0]
            calls.append(("detector", marker))
            clock.advance(detector_ms)
            result = {
                "input_size": list(image.size),
                "detections": [
                    {"label_id": 1, "label": "person", "score": 0.8, "box": [3, 4, 21, 41]},
                    {"label_id": 3, "label": "car", "score": 0.5, "box": [30, 5, 65, 50]},
                ],
                "timing": {
                    "preprocess_ms": 0.0,
                    "inference_ms": float(detector_ms),
                    "postprocess_ms": 0.0,
                    "total_ms": float(detector_ms),
                },
            }
            if mutate:
                mutate(result)
            return result

    class FakeCuda:
        def synchronize(self, device):
            calls.append(("cuda_sync", str(device)))

        def reset_peak_memory_stats(self, device):
            calls.append(("cuda_reset", str(device)))

        def memory_allocated(self, device):
            return 20

        def memory_reserved(self, device):
            return 50

        def max_memory_allocated(self, device):
            return 30

        def max_memory_reserved(self, device):
            return 60

    class Tracker:
        def __init__(self, profile):
            self.profile = profile
            self.metadata = {
                "schema": "iris-tracker-runtime-v1",
                "algorithm": profile["algorithm"],
                "execution_policy": {
                    "device": "cpu",
                    "learned_reid": False,
                    "opencv_threads": profile["opencv_threads"],
                    "seed": profile["seed"],
                    "native_buffer_unit": "available_frame_updates",
                    "native_time_step": 1,
                    "skipped_source_frames": "no_synthetic_updates",
                },
            }
            trackers.append(self)

        def reset(self, sequence_id):
            self.sequence_id, self.index = sequence_id, 0
            calls.append(("reset", sequence_id))

        def verify_runtime(self):
            calls.append(("verify_tracker",))

        def update(self, frame, *, image=None):
            self.index += 1
            calls.append(("tracker", frame["frame_index"], self.index, image is not None))
            clock.advance(tracker_ms)
            gmc = self.profile["gmc_method"]
            if gmc != "none":
                assert image.shape == (80, 100, 3)
                assert image[0, 0].tolist() == [3, 2, frame["frame_index"]]
            return {
                "schema": FRAME_SCHEMA,
                "sequence_id": self.sequence_id,
                **{
                    key: frame[key]
                    for key in ("frame_id", "frame_index", "timestamp_seconds", "input_size")
                },
                "update_index": self.index,
                "observations": [
                    {
                        **row,
                        "track_id": row["detection_index"] + 1,
                        "confirmed": True,
                        "estimated_box": row["box"],
                    }
                    for row in frame["detections"]
                ],
                "unassigned": [],
                "predictions": [],
                "gmc": {
                    "method": gmc,
                    "status": "disabled" if gmc == "none" else "initialized",
                    "matrix": None if gmc == "none" else [[1, 0, 0], [0, 1, 0]],
                    "downscale": self.profile["gmc_downscale"],
                },
                "timing": {
                    "gmc_ms": 0.0,
                    "association_ms": float(tracker_ms),
                    "total_ms": float(tracker_ms),
                },
            }

    return (
        {"detector_factory": Detector, "tracker_factory": Tracker},
        clock,
        calls,
        detectors,
        trackers,
    )


def test_fresh_detection_and_tracker_share_one_measured_loop_with_reset_warmup(
    tmp_path, monkeypatch
):
    store, comparison = example(tmp_path)
    factories, clock, calls, detectors, trackers = fake_factories(monkeypatch, comparison)
    progress = []

    def notify(*args):
        progress.append(args)
        clock.advance(100)  # Progress bookkeeping is not per-frame service time.

    report = runtime.measure_tracking_cost(
        store, comparison, repeats=2, progress=notify, **factories
    )
    assert report["summary"]["sample_count"] == 6
    assert report["summary"]["stages_ms"]["pipeline_ms"]["median"] == pytest.approx(25)
    assert report["summary"]["service_fps"] == pytest.approx(40)
    assert report["summary"]["wall_ms"] == pytest.approx(750)
    assert report["summary"]["memory"]["cuda_allocated_peak_bytes"] is None
    assert len(detectors) == len(trackers) == 1
    assert [call[1] for call in calls if call[0] == "detector"] == [0, 0, 1, 2, 0, 1, 2]
    assert [call[2] for call in calls if call[0] == "tracker"] == [1, 1, 2, 3, 1, 2, 3]
    assert len([call for call in calls if call[0] == "reset"]) == 3
    assert not any(call[0].startswith("cuda") for call in calls)
    assert progress[-1][0] == 1


def test_simulation_executes_only_latest_available_arrivals_and_drains_last(tmp_path, monkeypatch):
    store, comparison = example(tmp_path, indices=(0, 1, 2, 3, 4))
    factories, _, calls, _, _ = fake_factories(
        monkeypatch, comparison, detector_ms=20, tracker_ms=5
    )
    report = runtime.measure_tracking_cost(
        store, comparison, policy="simulated_latest", cadence_fps=100, **factories
    )
    run = report["passes"][0]
    assert [frame["frame_index"] for frame in run["frames"]] == [0, 2, 4]
    assert run["dropped_frame_indices"] == [1, 3]
    assert [call[1] for call in calls if call[0] == "detector"] == [0, 0, 2, 4]
    assert [call[1] for call in calls if call[0] == "tracker"] == [0, 0, 2, 4]
    assert run["frames"][-1]["schedule"]["finish_ms"] == pytest.approx(75)
    assert report["summary"]["simulated_output_fps"] == pytest.approx(40)


def test_sparse_source_indices_remain_gaps_not_compressed_arrivals_or_drops(tmp_path, monkeypatch):
    store, comparison = example(tmp_path, indices=(10, 12, 15))
    factories, _, _, _, _ = fake_factories(monkeypatch, comparison)
    report = runtime.measure_tracking_cost(
        store, comparison, policy="simulated_latest", cadence_fps=10, **factories
    )
    assert [frame["schedule"]["arrival_ms"] for frame in report["passes"][0]["frames"]] == [
        0,
        200,
        500,
    ]
    assert report["summary"]["source_gap_frames_per_pass"] == 3
    assert report["summary"]["dropped_frames"] == 0
    assert report["summary"]["simulated_output_fps"] == pytest.approx(4)


def test_cuda_instrumentation_uses_actual_index_and_resets_each_postwarmup_pass(
    tmp_path, monkeypatch
):
    store, comparison = example(tmp_path)
    factories, _, calls, _, _ = fake_factories(monkeypatch, comparison)
    report = runtime.measure_tracking_cost(store, comparison, device="cuda", repeats=2, **factories)
    assert [call for call in calls if call[0] == "cuda_reset"] == [("cuda_reset", "cuda:0")] * 2
    assert report["summary"]["memory"]["cuda_allocated_peak_bytes"] == 30
    assert report["summary"]["memory"]["cuda_reserved_peak_bytes"] == 60
    assert report["execution"]["detector_signature"]["device"] == "cuda:0"
    assert calls.index(("cuda_reset", "cuda:0")) > calls.index(("tracker", 0, 1, False))


def test_botsort_gmc_reuses_same_source_pixels_in_bgr_order(tmp_path, monkeypatch):
    store, comparison = example(tmp_path, gmc=True)
    factories, _, calls, _, _ = fake_factories(monkeypatch, comparison)
    report = runtime.measure_tracking_cost(store, comparison, lane_index=1, **factories)
    assert all(call[3] for call in calls if call[0] == "tracker")
    assert report["profile"]["gmc_method"] == "sparseOptFlow"


def test_changed_png_is_rejected_before_detection_without_fallback(tmp_path, monkeypatch):
    store, comparison = example(tmp_path)
    factories, _, calls, _, _ = fake_factories(monkeypatch, comparison)
    (tmp_path / "frame-0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="PNG bytes"):
        runtime.measure_tracking_cost(store, comparison, **factories)
    assert not any(call[0] == "detector" for call in calls)


def test_cancelled_run_never_returns_a_partial_success(tmp_path, monkeypatch):
    store, comparison = example(tmp_path)
    factories, _, calls, _, _ = fake_factories(monkeypatch, comparison)
    with pytest.raises(runtime.TrackingCostCancelled):
        runtime.measure_tracking_cost(
            store,
            comparison,
            cancelled=lambda: sum(call[0] == "detector" for call in calls) >= 3,
            **factories,
        )


def test_detector_and_tracker_actual_runtime_changes_are_rejected(tmp_path, monkeypatch):
    store, comparison = example(tmp_path)
    factories, _, _, detectors, _ = fake_factories(monkeypatch, comparison)

    def progress(fraction, message):
        if fraction == 1:
            detectors[0].metadata["threads"] += 1

    with pytest.raises(ValueError, match="Detector runtime changed"):
        runtime.measure_tracking_cost(store, comparison, progress=progress, **factories)


def test_real_elapsed_outer_tracker_call_includes_validation_not_just_native_timing(
    tmp_path, monkeypatch
):
    store, comparison = example(tmp_path)
    factories, clock, _, _, _ = fake_factories(monkeypatch, comparison)
    base = factories["tracker_factory"]

    class ExtraValidation(base):
        def update(self, frame, **options):
            result = super().update(frame, **options)
            clock.advance(7)
            return result

    factories["tracker_factory"] = ExtraValidation
    report = runtime.measure_tracking_cost(store, comparison, **factories)
    summary = report["summary"]["stages_ms"]
    assert summary["tracker_adapter_ms"]["median"] == 5
    assert summary["tracker_call_ms"]["median"] == pytest.approx(12)
    assert summary["pipeline_ms"]["median"] == pytest.approx(32)


@pytest.fixture
def cost_report(tmp_path, monkeypatch):
    store, comparison = example(tmp_path)
    factories, _, _, _, _ = fake_factories(monkeypatch, comparison)
    report = runtime.measure_tracking_cost(store, comparison, **factories)
    return comparison, report


def test_report_is_portable_data_not_live_runtime_or_file_verification(cost_report, monkeypatch):
    comparison, report = cost_report
    monkeypatch.setattr(
        temporal_detector, "_runtime", lambda *_: pytest.fail("Live runtime accessed")
    )
    assert contracts.validate_cost_report(report, comparison) == report
