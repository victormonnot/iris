"""Bounded portable input/report contracts using a real pipeline with fake inference."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image
from test_pipeline_runtime import FakeDetector, FakeTracker
from test_pipeline_runtime import manifest as controlled_manifest

from iris import pipeline_bundle_contracts as bundles
from iris import pipeline_runner as runner
from iris.pipeline_runtime import Pipeline


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    manifest = controlled_manifest()
    directory = tmp_path / "bundle"
    directory.mkdir()

    def checked(*_args, **_kwargs):
        return {
            "manifest": deepcopy(manifest),
            "manifest_sha256": bundles.digest(manifest),
            "files_verified": 1,
        }

    monkeypatch.setattr(bundles, "validate_directory", checked)
    return Pipeline(directory, detector_factory=FakeDetector, tracker_factory=FakeTracker)


def make_input(tmp_path, *, clock="provided", exif=False, count=3):
    directory = tmp_path / "input"
    directory.mkdir(exist_ok=True)
    rows = []
    for index in range(count):
        image = Image.new("RGB", (100, 80) if exif else (100, 100), (20 + index, 40, 60))
        if exif:
            orientation = image.getexif()
            orientation[274] = 6
            path = directory / f"image-{index}.jpg"
            image.save(path, exif=orientation)
        else:
            path = directory / f"image-{index}.png"
            image.save(path)
        rows.append(
            {
                "frame_id": f"source-{index}",
                "frame_index": index * 2,
                "timestamp_seconds": None if clock == "unknown" else index / 10,
                "path": path.name,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "input_size": {"width": 80 if exif else 100, "height": 100},
            }
        )
    rows[0]["select_detection_index"] = 2
    if count > 1:
        rows[-1]["release"] = True
    value = {
        "schema": "iris-pipeline-input-v1",
        "sequence_id": "local-sequence",
        "clock_kind": clock,
        "frames": rows,
    }
    path = directory / "frames.json"
    path.write_text(json.dumps(value))
    return path, value


def run_input(pipeline, tmp_path, **kwargs):
    path, value = make_input(tmp_path, **kwargs)
    return runner.run_frames(pipeline, path, bundle_directory=tmp_path / "bundle"), path, value


def save_report(path, report, *, rehash=False):
    value = deepcopy(report)
    if rehash:
        value["semantic_sha256"] = bundles.digest(runner._semantic(value))
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("clock", ["provided", "nominal_fps", "unknown"])
def test_frames_preserve_source_gaps_clock_and_explicit_selection_events(pipeline, tmp_path, clock):
    report, path, value = run_input(pipeline, tmp_path, clock=clock)
    assert report["complete"] is True
    assert report["schema"] == runner.RUN_SCHEMA
    assert report["bundle_manifest_sha256"] == bundles.digest(pipeline.manifest)
    assert report["runtime"] == pipeline.metadata
    assert report["source"]["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [row["frame_index"] for row in report["frames"]] == [0, 2, 4]
    assert [row["update_index"] for row in report["frames"]] == [1, 2, 3]
    assert [row["selection"]["state"] for row in report["frames"]] == [
        "observed",
        "recovering",
        "released",
    ]
    assert report["frames"][0]["selection"]["selected"]["detection_index"] == 2
    assert report["input_frames"][0]["select_detection_index"] == 2
    assert report["input_frames"][-1]["release"] is True
    assert [row["timestamp_seconds"] for row in report["frames"]] == [
        row["timestamp_seconds"] for row in value["frames"]
    ]
    assert report["semantic_sha256"] == bundles.digest(runner._semantic(report))


def test_exif_orientation_is_applied_before_pixels_dimensions_and_pipeline_coordinates(
    pipeline, tmp_path
):
    report, path, value = run_input(pipeline, tmp_path, exif=True)
    assert all(row["input_size"] == [80, 100] for row in report["frames"])
    decoded = runner._image((path.parent / value["frames"][0]["path"]).read_bytes())
    assert decoded.mode == "RGB" and decoded.getexif().get(274, 1) == 1
    assert report["input_frames"][0]["pixel_sha256"] == runner._pixel_hash(decoded)


@pytest.mark.parametrize(
    "change", ["sha", "size", "animated", "oversized_file", "oversized_pixels"]
)
def test_bad_image_facts_do_not_produce_a_report(pipeline, tmp_path, monkeypatch, change):
    path, value = make_input(tmp_path, count=1)
    row = value["frames"][0]
    if change == "sha":
        row["file_sha256"] = "a" * 64
    elif change == "size":
        row["input_size"]["width"] = 99
    elif change == "animated":
        target = path.parent / "animated.gif"
        Image.new("RGB", (100, 100), "red").save(
            target,
            save_all=True,
            append_images=[Image.new("RGB", (100, 100), "blue")],
            duration=100,
            loop=0,
        )
        row.update(path=target.name, file_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    elif change == "oversized_file":
        monkeypatch.setattr(runner, "MAX_IMAGE", 1)
    else:
        monkeypatch.setattr(runner, "MAX_PIXELS", 100)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.run_frames(pipeline, path, bundle_directory=tmp_path / "bundle")
    assert pipeline._detector.calls == 0


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_id",
        "duplicate_index",
        "reverse_time",
        "unknown_time",
        "bool_index",
        "bool_time",
        "select_release",
        "bool_select",
        "missing_key",
        "extra_key",
        "escape",
        "absolute",
        "backslash",
        "symlink",
    ],
)
def test_complete_manifest_clock_events_and_paths_validate_before_any_updates(
    pipeline, tmp_path, change
):
    path, value = make_input(tmp_path)
    row = value["frames"][1]
    if change == "duplicate_id":
        row["frame_id"] = value["frames"][0]["frame_id"]
    elif change == "duplicate_index":
        row["frame_index"] = 0
    elif change == "reverse_time":
        row["timestamp_seconds"] = 0
    elif change == "unknown_time":
        value["clock_kind"] = "unknown"
    elif change == "bool_index":
        row["frame_index"] = True
    elif change == "bool_time":
        row["timestamp_seconds"] = True
    elif change == "select_release":
        row.update(select_detection_index=2, release=True)
    elif change == "bool_select":
        row["select_detection_index"] = True
    elif change == "missing_key":
        row.pop("file_sha256")
    elif change == "extra_key":
        row["unexpected"] = 4
    elif change == "escape":
        row["path"] = "../outside.png"
    elif change == "absolute":
        row["path"] = str(path.parent / row["path"])
    elif change == "backslash":
        row["path"] = "child\\image.png"
    else:
        target = path.parent / row["path"]
        saved = target.with_suffix(".saved.png")
        target.rename(saved)
        target.symlink_to(saved)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.run_frames(pipeline, path, bundle_directory=tmp_path / "bundle")
    assert pipeline._detector.calls == 0


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_tokens_are_rejected_even_in_unknown_clock_inputs(
    pipeline, tmp_path, constant
):
    path, _ = make_input(tmp_path, clock="unknown")
    path.write_text(
        path.read_text().replace('"timestamp_seconds": null', '"timestamp_seconds": ' + constant, 1)
    )
    with pytest.raises(ValueError):
        runner.run_frames(pipeline, path, bundle_directory=tmp_path / "bundle")
    assert pipeline._detector.calls == 0


def test_duplicate_json_keys_and_inputs_inside_bundle_are_rejected(pipeline, tmp_path):
    path, _ = make_input(tmp_path)
    path.write_text(
        path.read_text().replace(
            '"clock_kind": "provided"', '"clock_kind": "provided", "clock_kind": "unknown"'
        )
    )
    with pytest.raises(ValueError, match="Duplicate"):
        runner.frame_input(path, tmp_path / "bundle")
    inside = tmp_path / "bundle" / "input.json"
    inside.write_text("{}")
    with pytest.raises(ValueError, match="outside"):
        runner.frame_input(inside, tmp_path / "bundle")


def test_report_publication_is_atomic_does_not_overwrite_and_removes_temporary_files(
    pipeline, tmp_path, monkeypatch
):
    report, _, _ = run_input(pipeline, tmp_path)
    output = tmp_path / "report.json"
    runner.write_report(output, report)
    assert json.loads(output.read_text()) == report
    original = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        runner.write_report(output, {"changed": True})
    assert output.read_bytes() == original and not list(tmp_path.glob(".pipeline-report-*"))
    racing = tmp_path / "racing.json"

    def racing_link(_source, destination):
        Path(destination).write_text("other writer")
        raise FileExistsError("Another writer won")

    monkeypatch.setattr(runner.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        runner.write_report(racing, report)
    assert racing.read_text() == "other writer" and not list(tmp_path.glob(".pipeline-report-*"))


def test_failed_or_oversized_serialization_cannot_leave_partial_report(
    pipeline, tmp_path, monkeypatch
):
    report, _, _ = run_input(pipeline, tmp_path)
    target = tmp_path / "report.json"
    monkeypatch.setattr(runner, "MAX_JSON", 20)
    with pytest.raises(ValueError):
        runner.write_report(target, report)
    assert not target.exists() and not list(tmp_path.glob(".pipeline-report-*"))


def test_comparison_excludes_timing_and_runtime_but_preserves_exact_input_and_output(
    pipeline, tmp_path
):
    report, _, _ = run_input(pipeline, tmp_path)
    changed = deepcopy(report)
    changed["runtime"]["detector"]["device"] = "cuda:0"
    changed["bundle_manifest_sha256"] = "f" * 64
    changed["runtime"]["bundle_manifest_sha256"] = "f" * 64
    for frame in changed["frames"]:
        frame["detector"]["timing"] = {key: 4.5 for key in frame["detector"]["timing"]}
        frame["tracking"]["timing"] = {key: 2.0 for key in frame["tracking"]["timing"]}
    reference = save_report(tmp_path / "reference.json", report)
    actual = save_report(tmp_path / "actual.json", changed)
    comparison = runner.compare_reports(reference, actual)
    assert comparison["status"] == "exact_match"
    assert comparison["reference_sha256"] != comparison["actual_sha256"]
    assert comparison["tolerance"] == 0 and comparison["mismatched_update_indices"] == []
    changed["frames"][1]["detector"]["detections"][0]["score"] = 0.8
    save_report(actual, changed, rehash=True)
    mismatch = runner.compare_reports(reference, actual)
    assert mismatch["status"] == "mismatch" and mismatch["mismatched_update_indices"] == [1]


def test_changed_pixels_or_explicit_events_prevent_an_exact_match(pipeline, tmp_path):
    report, _, _ = run_input(pipeline, tmp_path)
    changed = deepcopy(report)
    changed["input_frames"][1]["pixel_sha256"] = "e" * 64
    reference = save_report(tmp_path / "reference.json", report)
    actual = save_report(tmp_path / "actual.json", changed, rehash=True)
    assert runner.compare_reports(reference, actual)["status"] == "mismatch"
    path, value = make_input(tmp_path)
    for row in value["frames"]:
        row.pop("select_detection_index", None)
        row.pop("release", None)
    path.write_text(json.dumps(value))
    changed = runner.run_frames(pipeline, path, bundle_directory=tmp_path / "bundle")
    save_report(actual, changed)
    assert runner.compare_reports(reference, actual)["status"] == "mismatch"


@pytest.mark.parametrize(
    "change",
    [
        "stale_hash",
        "incomplete",
        "missing_detector",
        "wrong_size",
        "wrong_frame_identity",
        "bool_track_id",
        "extra_top_level",
    ],
)
def test_falsified_or_rehashed_malformed_reports_do_not_become_valid_parity_evidence(
    pipeline, tmp_path, change
):
    report, _, _ = run_input(pipeline, tmp_path)
    changed = deepcopy(report)
    if change == "stale_hash":
        changed["frames"][1]["detector"]["detections"][0]["score"] = 0.7
    elif change == "incomplete":
        changed["complete"] = False
    elif change == "missing_detector":
        changed["frames"][0]["detector"] = {}
    elif change == "wrong_size":
        changed["frames"][0]["detector"]["input_size"] = [1, 2]
    elif change == "wrong_frame_identity":
        changed["frames"][0]["frame_id"] = "unrelated"
    elif change == "bool_track_id":
        changed["frames"][0]["tracking"]["observations"][0]["track_id"] = True
    else:
        changed["qualification"] = "production-ready"
    reference = save_report(tmp_path / "reference.json", report)
    actual = save_report(tmp_path / "actual.json", changed, rehash=change != "stale_hash")
    with pytest.raises(ValueError):
        runner.compare_reports(reference, actual)


def test_numeric_type_changes_do_not_equal_exact_semantics_even_when_python_values_compare_equal(
    pipeline, tmp_path
):
    report, _, _ = run_input(pipeline, tmp_path)
    changed = deepcopy(report)
    # Integer and floating-point coordinates are each valid, but exact means the
    # canonical saved representation too; comparisons must not use Python ==.
    changed["frames"][1]["detector"]["detections"][0]["box"][0] = float(
        changed["frames"][1]["detector"]["detections"][0]["box"][0]
    )
    reference = save_report(tmp_path / "reference.json", report)
    actual = save_report(tmp_path / "actual.json", changed, rehash=True)
    assert runner.compare_reports(reference, actual)["status"] == "mismatch"


@pytest.fixture
def video_file(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path / "local.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
    assert writer.isOpened()
    try:
        for index in range(4):
            writer.write(np.full((100, 100, 3), (index * 30, 40, 120), dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("clock", ["nominal_fps", "unknown"])
def test_short_local_video_records_declared_clock_events_and_bounded_coverage(
    pipeline, tmp_path, video_file, clock
):
    events = tmp_path / "events.json"
    events.write_text(
        json.dumps(
            {
                "schema": "iris-pipeline-events-v1",
                "events": [
                    {"frame_index": 0, "select_detection_index": 2},
                    {"frame_index": 2, "release": True},
                ],
            }
        )
    )
    report = runner.run_video(
        pipeline,
        video_file,
        bundle_directory=tmp_path / "bundle",
        max_frames=3,
        clock=clock,
        events=events,
    )
    assert len(report["frames"]) == 3 and report["source"]["coverage"] == "bounded_prefix"
    assert report["source"]["nominal_fps"] == 10
    assert report["source"]["events_sha256"] == hashlib.sha256(events.read_bytes()).hexdigest()
    assert report["input_frames"][2]["timestamp_seconds"] == (
        0.2 if clock == "nominal_fps" else None
    )
    assert report["frames"][-1]["selection"]["state"] == "released"
    complete = runner.run_video(
        pipeline, video_file, bundle_directory=tmp_path / "bundle", max_frames=8, clock=clock
    )
    assert (
        len(complete["frames"]) == 4 and complete["source"]["coverage"] == "decoder_end_of_stream"
    )


@pytest.mark.parametrize(
    "events",
    [
        [{"frame_index": 5, "release": True}],
        [{"frame_index": 0, "select_detection_index": 2}, {"frame_index": 0, "release": True}],
        [{"frame_index": 1, "select_detection_index": 2, "release": True}],
    ],
)
def test_video_rejects_unconsumed_duplicate_or_contradictory_events(
    pipeline, tmp_path, video_file, events
):
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"schema": "iris-pipeline-events-v1", "events": events}))
    with pytest.raises(ValueError):
        runner.run_video(
            pipeline, video_file, bundle_directory=tmp_path / "bundle", max_frames=8, events=path
        )


@pytest.mark.parametrize("limit", [0, -1, 10_001, True])
def test_video_limits_are_explicit_bounded_integers(pipeline, tmp_path, video_file, limit):
    with pytest.raises(ValueError):
        runner.run_video(
            pipeline, video_file, bundle_directory=tmp_path / "bundle", max_frames=limit
        )
    assert pipeline._detector.calls == 0


def test_text_playlists_disguised_as_video_never_reach_decoder(pipeline, tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "playlist.mp4"
    path.write_text("#EXTM3U\nhttps://example.invalid/movie.mp4\n")
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda *_: pytest.fail("A playlist reached the decoder")
    )
    with pytest.raises(ValueError, match="binary video container"):
        runner.run_video(pipeline, path, bundle_directory=tmp_path / "bundle")


@pytest.mark.parametrize(
    "change",
    [
        "wrong_pixel_hash",
        "wrong_update",
        "wrong_clock",
        "wrong_runtime_bundle",
        "false_quality",
        "selection_state",
        "selection_policy",
        "timing_bool",
        "detector_score",
        "unaccounted_detection",
        "invented_prediction",
        "changed_observation",
    ],
)
def test_rehashed_reports_preserve_actual_detection_and_selection_relations(
    pipeline, tmp_path, change
):
    report, _, _ = run_input(pipeline, tmp_path)
    changed = deepcopy(report)
    frame = changed["frames"][1]
    if change == "wrong_pixel_hash":
        changed["input_frames"][0]["pixel_sha256"] = "not-a-hash"
    elif change == "wrong_update":
        frame["update_index"] = 8
    elif change == "wrong_clock":
        frame["tracking"]["timestamp_seconds"] = 0.11
    elif change == "wrong_runtime_bundle":
        changed["runtime"]["bundle_manifest_sha256"] = "a" * 64
    elif change == "false_quality":
        changed["runtime"]["independent_quality"] = "qualified"
    elif change == "selection_state":
        frame["selection"]["state"] = "observed"
    elif change == "selection_policy":
        changed["pipeline_contract"]["selection_policy"]["min_score"] = 0.8
    elif change == "timing_bool":
        frame["detector"]["timing"]["inference_ms"] = True
    elif change == "detector_score":
        frame["detector"]["detections"][0]["score"] = 1.1
    elif change == "unaccounted_detection":
        frame["tracking"]["observations"] = []
    elif change == "changed_observation":
        frame["tracking"]["observations"][0]["box"][0] += 1
    else:
        frame["tracking"]["predictions"] = [
            {
                "track_id": 99,
                "label_id": 1,
                "label": "person",
                "box": [10, 10, 20, 30],
                "confirmed": True,
                "last_observed_frame_id": "source-0",
                "last_observed_frame_index": 0,
                "last_observed_timestamp_seconds": 0.0,
                "last_observed_update_index": 1,
                "age_updates": 1,
                "age_seconds": 0.1,
            }
        ]
    reference = save_report(tmp_path / "reference.json", report)
    actual = save_report(tmp_path / "actual.json", changed, rehash=True)
    with pytest.raises(ValueError):
        runner.compare_reports(reference, actual)


def test_reports_can_exceed_one_thousand_frames_within_the_declared_budget(pipeline):
    pipeline.reset("long-sequence", clock_kind="unknown")
    report = runner._report(
        pipeline,
        {"kind": "frames", "input_sha256": "a" * 64, "coverage": "all_declared_frames"},
        "unknown",
    )
    image = Image.new("RGB", (100, 100))
    for index in range(1001):
        runner._append(
            report,
            pipeline,
            image,
            frame_id=f"frame-{index}",
            frame_index=index,
            timestamp_seconds=None,
            event={"select_detection_index": None, "release": False},
        )
    report = runner._finish(report)
    runner._validate_report(report)
    assert len(report["frames"]) == 1001
    assert runner.digest(runner._semantic(report)) == report["semantic_sha256"]
    assert runner.canonical({"same": [1, 2.0, None, False]}) == bundles.canonical(
        {"same": [1, 2.0, None, False]}
    )
    with pytest.raises(ValueError):
        runner.canonical([None] * 10_001)


def test_runtime_change_before_completion_prevents_publishable_run(pipeline, tmp_path, monkeypatch):
    original = pipeline.update

    def changed_runtime(*args, **kwargs):
        result = original(*args, **kwargs)
        pipeline._detector.changed = True
        return result

    monkeypatch.setattr(pipeline, "update", changed_runtime)
    path, _ = make_input(tmp_path, count=1)
    with pytest.raises(RuntimeError, match="runtime changed"):
        runner.run_frames(pipeline, path, bundle_directory=tmp_path / "bundle")
