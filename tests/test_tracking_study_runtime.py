"""Independent request, replay-integrity, budget and aggregate study checks."""

import json
from copy import deepcopy

import pytest
from test_temporal_detection_api import client as client
from test_temporal_identities import comparison as comparison
from test_tracking_comparisons import completed as completed
from test_tracking_metrics import example, gt, observed
from test_tracking_quality import reviewed as reviewed
from test_tracking_replay import SyntheticTracker, forbidden, synthetic

from iris import temporal, tracking_studies
from iris import tracking_study_contracts as contracts
from iris import tracking_study_runtime as runtime
from iris.tracking_contracts import make_profile, profile_hash, semantic_frame
from iris.tracking_study_replay import EXECUTION_POLICY


def synthetic_metadata(profile):
    from iris.tracking import _provenance

    return {
        "schema": "iris-tracker-runtime-v1",
        "algorithm": profile["algorithm"],
        "provenance": _provenance(profile["algorithm"]),
        "python": "3.12.0",
        "platform": "Synthetic fixture; not evidence of native tracking execution",
        "packages": {
            name: "synthetic"
            for name in ("scipy", "lap", "cython_bbox", "numpy", "opencv-python-headless")
        },
        "execution_policy": {
            **EXECUTION_POLICY,
            "opencv_threads": profile["opencv_threads"],
            "seed": profile["seed"],
            "unconfirmed_returned": profile["algorithm"] == "botsort",
            "blas_environment": dict.fromkeys(
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            ),
        },
    }


@pytest.fixture(autouse=True)
def explicit_synthetic_metadata(monkeypatch):
    original = SyntheticTracker.__init__

    def initialize(self, profile, **kwargs):
        original(self, profile, **kwargs)
        self.metadata = synthetic_metadata(profile)

    monkeypatch.setattr(SyntheticTracker, "__init__", initialize)


@pytest.fixture
def bundle(client, comparison, reviewed):
    store = client.app.state.store
    dataset = temporal.create_temporal_dataset(
        store,
        name="Development fixture",
        entries=[
            {
                "sequence_id": comparison["sequence_id"],
                "split": "train",
                "reference_id": reviewed["id"],
            }
        ],
    )
    baseline = comparison["report"]["lanes"][0]["report"]["profile"]
    request = {
        "name": "Bounded fixture",
        "dataset_id": dataset["id"],
        "sources": [{"sequence_id": comparison["sequence_id"], "comparison_id": comparison["id"]}],
        "baseline": {"name": "Baseline", "profile": baseline},
        "candidates": [{"name": "Longer buffer", "profile": {**baseline, "buffer_updates": 60}}],
        "class_mapping": {"1": "person", "3": None},
        "iou_threshold": 0.5,
        "repeats": 2,
        "max_updates": 1000,
        "max_seconds": 120,
    }
    with store.connect() as conn:
        return tracking_studies._resolve(conn, request, store=store, project_id="default")


def test_runtime_replays_only_caches_and_counts_quality_once(client, bundle, monkeypatch):
    calls, trackers = synthetic(monkeypatch)
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    progress = []
    report = runtime.run_study(
        client.app.state.store, bundle, progress=lambda *args: progress.append(args)
    )
    assert len(trackers) == 4
    assert (
        len([call for call in calls if call[0] == "update"]) == bundle["budget"]["required_updates"]
    )
    assert contracts.validate_report(bundle, report) == report
    assert report["summary"]["decision"]["applied"] is False
    assert report["summary"]["decision"]["reason"] == "development_only"
    assert report["summary"]["splits"]["val"] is None
    summary = report["summary"]["splits"]["train"]
    assert summary["evaluated_frames"] == 3
    for profile in summary["profiles"]:
        assert profile["timing"]["tracker_ms"]["count"] == 6
        assert profile["counts"]["ground_truth"] == 2
        assert profile["repeatability"] == "observed_match"
    assert report["summary"]["comparisons"][1]["by_split"]["train"] == "no_gain"
    assert progress[-1][0] == 1
    assert all(first[0] <= second[0] for first, second in zip(progress, progress[1:], strict=False))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["request"].update(max_seconds=42),
        lambda r: r["sources"][0].update(cache_fingerprint="0" * 64),
        lambda r: r["dataset"].update(
            reserved_test_entries=[] if r["dataset"]["reserved_test_entries"] else [{}]
        ),
        lambda r: r["runs"][0].update(split="test"),
        lambda r: r["runs"][0]["replays"][0]["profile"].update(buffer_updates=2),
        lambda r: r["runs"][0]["replays"][0]["passes"][0].update(semantic_sha256="0" * 64),
        lambda r: r["runs"][0]["replays"][0]["passes"][0]["metadata"]["execution_policy"].update(
            device="cuda"
        ),
        lambda r: r["runs"][0]["replays"][0]["passes"][0]["frames"][0]["timing"].update(
            total_ms=float("inf")
        ),
        lambda r: r["runs"][0]["replays"][0]["passes"][0]["image_reads"][0].update(image_read_ms=5),
        lambda r: r["runs"][0]["replays"][0]["repeatability"].update(status="not_checked"),
        lambda r: r["summary"]["splits"]["train"]["profiles"][0]["counts"].update(
            false_negatives=99
        ),
        lambda r: r["summary"]["splits"]["train"]["profiles"][0]["timing"]["tracker_ms"].update(
            p95=999
        ),
        lambda r: r["summary"]["decision"].update(applied=True),
        lambda r: r["request"].update(iou_threshold=True),
        lambda r: r["sources"][0].update(reference_revision=True),
        lambda r: r["runs"][0]["replays"][0]["profile"].update(class_ids=[True, 3]),
    ],
)
def test_historical_validation_rejects_changed_evidence(client, bundle, mutation):
    report = runtime.run_study(client.app.state.store, bundle)
    mutation(report)
    with pytest.raises((ValueError, TypeError)):
        contracts.validate_report(bundle, report)


def test_rehashed_invalid_observation_is_not_accepted(client, bundle):
    report = runtime.run_study(client.app.state.store, bundle)
    replay = report["runs"][0]["replays"][0]
    run = replay["passes"][0]
    run["frames"][0]["observations"][0]["box"][0] += 1
    run["semantic_sha256"] = temporal._digest(
        {
            "profile_sha256": replay["profile_sha256"],
            "cache_fingerprint": replay["cache"]["fingerprint"],
            "sequence_sha256": replay["cache"]["config"]["sequence_sha256"],
            "frames": [semantic_frame(frame) for frame in run["frames"]],
        }
    )
    replay["repeatability"]["semantic_sha256"][0] = run["semantic_sha256"]
    with pytest.raises(ValueError):
        contracts.validate_report(bundle, report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(packages=[]),
        lambda m: m["packages"].update(numpy=None),
        lambda m: m["provenance"].update(commit="not-a-commit"),
        lambda m: m["provenance"]["adapter_sha256"].update({"tracking.py": "bad"}),
        lambda m: m["provenance"]["files"]["basetrack.py"].update(upstream_sha256=None),
        lambda m: m["execution_policy"].update(native_second_pass_match_threshold=0.2),
        lambda m: m["execution_policy"].update(blas_environment={}),
        lambda m: m.update(platform="Changed host"),
    ],
)
def test_rehashed_invalid_or_different_runtime_rejected(client, bundle, mutate):
    report = runtime.run_study(client.app.state.store, bundle)
    for run in report["runs"][0]["replays"][1]["passes"]:
        mutate(run["metadata"])
        run["runtime_sha256"] = temporal._digest(run["metadata"])
    with pytest.raises(ValueError):
        contracts.validate_report(bundle, report)


def test_stage_timings_cannot_exceed_their_parent(client, bundle):
    report = runtime.run_study(client.app.state.store, bundle)
    report["runs"][0]["replays"][0]["passes"][0]["frames"][0]["timing"].update(
        gmc_ms=5, association_ms=2, total_ms=3
    )
    with pytest.raises(ValueError, match="Nested tracker"):
        contracts.validate_report(bundle, report)


def test_cancellation_deadline_and_size_limit_publish_nothing(client, bundle, monkeypatch):
    calls, trackers = synthetic(monkeypatch)
    with pytest.raises(runtime.TrackingStudyCancelled):
        runtime.run_study(client.app.state.store, bundle, cancelled=lambda: True)
    assert not calls and not trackers
    ticks = iter([0, 121])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
    with pytest.raises(runtime.TrackingStudyBudgetExceeded):
        runtime.run_study(client.app.state.store, bundle)
    assert not calls and not trackers
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 0)
    monkeypatch.setattr(runtime, "MAX_REPORT_BYTES", 1)
    with pytest.raises(ValueError, match="size budget"):
        runtime.run_study(client.app.state.store, bundle)
    assert len(trackers) == 2  # Stops after the first completed two-pass profile.


def test_deadline_after_scoring_is_checked(client, bundle, monkeypatch):
    clock = [0]
    original = runtime.summarize
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])

    def slow_summary(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 121
        return result

    monkeypatch.setattr(runtime, "summarize", slow_summary)
    with pytest.raises(runtime.TrackingStudyBudgetExceeded):
        runtime.run_study(client.app.state.store, bundle)


def test_semantic_mismatch_withholds_quality_conclusion(client, bundle, monkeypatch):
    synthetic(monkeypatch, mismatch=True)
    report = runtime.run_study(client.app.state.store, bundle)
    assert report["summary"]["comparisons"][1]["by_split"]["train"] == "unstable"
    assert report["summary"]["decision"]["applied"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("repeats", True),
        ("repeats", 0),
        ("repeats", 4),
        ("max_seconds", float("nan")),
        ("max_seconds", 0),
        ("max_seconds", 601),
        ("max_updates", 0),
        ("max_updates", 20001),
        ("max_updates", 2.5),
        ("iou_threshold", 0),
        ("iou_threshold", float("inf")),
        ("class_mapping", {"1": None, "3": None}),
        ("class_mapping", {"1": "person"}),
    ],
)
def test_request_bounds(bundle, field, value):
    request = deepcopy(bundle["request"])
    request[field] = value
    with pytest.raises(ValueError):
        contracts.canonicalize_request(request)


def test_distinct_profiles_and_fixed_controls(bundle):
    request = deepcopy(bundle["request"])
    request["candidates"][0]["profile"] = deepcopy(request["baseline"]["profile"])
    with pytest.raises(ValueError, match="distinct canonical"):
        contracts.canonicalize_request(request)
    request["candidates"][0]["profile"]["seed"] = 1
    with pytest.raises(ValueError, match="execution controls"):
        contracts.canonicalize_request(request)


@pytest.mark.parametrize("algorithm", ["bytetrack", "botsort"])
@pytest.mark.parametrize("high", [0.11, 0.5, 0.9])
def test_suggestions_are_small_valid_and_reproducible(algorithm, high):
    baseline = make_profile(
        algorithm,
        class_ids=[1],
        high_threshold=high,
        new_track_threshold=high + 0.1 if algorithm == "bytetrack" else high,
        seed=42,
        opencv_threads=2,
    )
    first = contracts.suggestions(baseline)
    assert first == contracts.suggestions(baseline)
    assert 1 <= len(first["candidates"]) <= 7
    hashes = {profile_hash(baseline)}
    for candidate in first["candidates"]:
        digest = profile_hash(candidate["profile"])
        assert digest not in hashes
        hashes.add(digest)
        assert candidate["profile"]["seed"] == 42
        assert candidate["profile"]["opencv_threads"] == 2


def test_update_budget_counts_every_profile_source_and_repeat(bundle):
    assert bundle["budget"]["required_updates"] == 2 * 3 * 2
    request = {**bundle["request"], "max_updates": 11}
    with pytest.raises(ValueError, match="12 updates"):
        contracts.budget_for(request, bundle["sources"])


def test_null_identity_and_splits_are_not_pooled():
    # Two independent sequences deliberately reuse the same local track/GT IDs.
    dense, ref1 = example([[gt()], [gt()]], [[observed()], [observed()]])
    sparse, ref2 = example([[gt()], [gt()]], [[observed()], []], indices=[0, 2])
    baseline = make_profile("bytetrack", class_ids=[1, 3])
    candidate = {**baseline, "buffer_updates": 60}
    named = [{"name": "Baseline", "profile": baseline}, {"name": "Candidate", "profile": candidate}]
    sources, runs = [], []
    for index, (paired, reference, split) in enumerate(
        [(dense, ref1, "train"), (sparse, ref2, "val")]
    ):
        manifest = paired["report"]["sequence"]
        sources.append(
            {
                "entry": {"split": split},
                "sequence": {"id": f"seq{index}", "manifest": manifest},
                "reference": reference,
                "comparison": paired,
            }
        )
        replays = [deepcopy(lane["report"]) for lane in paired["report"]["lanes"]]
        for profile_index, replay in enumerate(replays):
            replay.update(
                profile=named[profile_index]["profile"],
                profile_sha256=profile_hash(named[profile_index]["profile"]),
                repeatability={"status": "not_checked"},
            )
            for frame in replay["passes"][0]["frames"]:
                frame["timing"] = {"total_ms": 2, "gmc_ms": 0, "association_ms": 1}
            replay["passes"][0].update(
                image_reads=[{"image_read_ms": 0}, {"image_read_ms": 0}],
                timing={"adapter_setup_ms": 1, "replay_ms": 10},
            )
        runs.append({"replays": replays})
    inputs = {
        "request": {
            "baseline": named[0],
            "candidates": named[1:],
            "class_mapping": {"1": "person", "3": None},
            "iou_threshold": 0.5,
        },
        "sources": sources,
    }
    summary = contracts.summarize(inputs, runs)
    train = summary["splits"]["train"]["profiles"][0]
    val = summary["splits"]["val"]["profiles"][0]
    assert train["identity"]["idf1"] == 1
    assert train["counts"]["ground_truth"] == val["counts"]["ground_truth"] == 2
    assert val["identity"]["available"] is False
    assert val["identity"]["idf1"] is None
    assert val["counts"]["false_negatives"] == 1
    assert summary["decision"]["reason"] == "no_verified_gain"
    json.dumps(summary, allow_nan=False)

    # A gain on one source cannot qualify a whole split whose other source has
    # no human-complete frame. Keep descriptive counts and withhold comparison.
    sources[1]["entry"]["split"] = "train"
    for frame in sources[1]["reference"]["payload"]["frames"]:
        frame["coverage"] = "partial"
    runs[0]["replays"][0]["passes"][0]["frames"][1]["observations"] = []
    summary = contracts.summarize(inputs, runs)
    assert summary["splits"]["train"]["profiles"][0]["counts"]["false_negatives"] == 1
    assert summary["splits"]["train"]["profiles"][1]["counts"]["false_negatives"] == 0
    assert summary["comparisons"][1]["by_split"]["train"] == "insufficient"
