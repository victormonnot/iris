"""Persisted video review tests use synthetic AVI pixels and explicit provider fixtures."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest
from test_media import make_video

from iris.assistance_provider import ProviderResponseError
from iris.jobs import JobManager
from iris.media import import_asset
from iris.store import Store, new_id, now
from iris.video_review_provider import PROMPT_VERSION
from iris.video_reviews import (
    get_review,
    list_reviews,
    prepare_review,
    preview_passage_extraction,
    queue_review,
    read_review_images,
    run_video_review,
    validate_passage_extraction,
)

READY = {
    "provider": "ollama",
    "model": "fixture-vision",
    "endpoint": "http://127.0.0.1:11434",
    "status": "ready",
    "version": "fixture-runtime-v1",
    "model_digest": "a" * 64,
}
REMOTE = {
    "provider": "alibaba",
    "model": "qwen3-vl-32b-instruct",
    "endpoint": "https://fixture.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
    "status": "ready",
}
RESPONSE = {
    "summary": "Synthetic sampled images only; events between samples are unknown.",
    "passages": [
        {
            "start_sample_id": "s2",
            "end_sample_id": "s3",
            "reason": "A visible fixture color changes.",
            "uncertainty": "high",
        },
        {
            "start_sample_id": "s5",
            "end_sample_id": "s5",
            "reason": "Another sampled color.",
            "uncertainty": "medium",
        },
    ],
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    store = Store(tmp_path / "workspace")
    session = store.insert(
        "sessions",
        {"id": new_id(), "name": "Fixture", "scene_group": "synthetic", "created_at": now()},
    )
    asset = import_asset(
        store,
        session["id"],
        make_video(tmp_path / "fixture.avi", [(index * 5, 30, 150) for index in range(40)]),
        "fixture.avi",
    )
    monkeypatch.setattr(
        "iris.video_reviews._provider_status",
        lambda provider, *args, **kwargs: deepcopy(REMOTE if provider == "alibaba" else READY),
    )
    return store, asset, JobManager(store)


class FixtureReviewer:
    calls = []

    def __init__(self, config):
        self.metadata = deepcopy(REMOTE if config["provider"] == "alibaba" else READY)

    def review(self, images, samples, instructions):
        self.calls.append((images, samples, instructions))
        return {
            **deepcopy(RESPONSE),
            "metadata": self.metadata,
            "prompt": "Fixture prompt",
            "raw_response": {"fixture": True, "content": deepcopy(RESPONSE)},
        }


def prepare(workspace, **kwargs):
    store, asset, _ = workspace
    return prepare_review(store, asset["id"], sample_count=6, **kwargs)


def start(workspace, preview, **kwargs):
    store, _, jobs = workspace
    job = queue_review(store, jobs, preview["id"], **kwargs)
    store.update("jobs", job["id"], {"status": "running", "started_at": now()})
    return job


def complete(workspace, **kwargs):
    preview = prepare(workspace, **kwargs)
    start(workspace, preview)
    run_video_review(workspace[0], preview["id"], lambda *_: None, lambda: False, FixtureReviewer)
    return get_review(workspace[0], preview["id"])


def test_prepare_decodes_exact_bounded_storyboard_without_creating_frames(workspace):
    store, asset, _ = workspace
    preview = prepare(workspace, instructions="  Review visible changes.  ")
    assert preview["status"] == "preview" and preview["job"] is None
    assert len(preview["images"]) == 6
    assert preview["config"]["source_sha256"] == asset["sha256"]
    assert preview["config"]["source_metadata"] == asset["metadata"]
    assert preview["config"]["prompt_version"] == PROMPT_VERSION
    assert preview["config"]["instructions"] == "Review visible changes."
    assert [sample["frame_index"] for sample in preview["images"]] == [0, 8, 16, 23, 31, 39]
    assert all(
        "path" not in item and item["url"].startswith("/api/video-reviews/")
        for item in preview["images"]
    )
    raw = store.get("video_reviews", preview["id"])
    assert len(read_review_images(store, raw)) == 6
    assert list_reviews(store, asset["id"]) == [preview]
    assert (
        not store.list("jobs")
        and not store.list("frames")
        and not store.list("annotation_revisions")
    )
    reopened = Store(store.root)
    assert get_review(reopened, preview["id"]) == preview


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_count": 1},
        {"sample_count": 13},
        {"sample_count": True},
        {"instructions": "a" * 2001},
        {"instructions": None},
        {"start_seconds": -1},
        {"end_seconds": 0},
    ],
)
def test_invalid_prepare_never_writes(workspace, kwargs):
    store, asset, _ = workspace
    with pytest.raises(ValueError):
        prepare_review(store, asset["id"], **kwargs)
    assert not store.list("video_reviews")
    assert not list(store.root.glob("video_reviews/*"))


def test_short_range_can_contain_one_actual_sample(workspace):
    preview = prepare(workspace, start_seconds=0, end_seconds=0.1)
    assert preview["config"]["plan"]["planned_count"] == 1
    assert preview["images"][0]["frame_index"] == 0


def test_missing_model_blocks_before_decoding(workspace, monkeypatch):
    monkeypatch.setattr(
        "iris.video_reviews._provider_status",
        lambda *_: {"status": "missing", "reason": "Install a fixture model"},
    )
    with pytest.raises(RuntimeError, match="fixture model"):
        prepare(workspace)
    assert not list(workspace[0].root.glob("video_reviews/*"))


def test_failed_decoding_cleans_images_and_record(workspace, monkeypatch):
    original = __import__("iris.video_reviews", fromlist=["cv2"]).cv2.VideoCapture

    class BrokenCapture:
        def __init__(self, path):
            self.real = original(path)
            self.count = 0

        def isOpened(self):
            return True

        def set(self, *args):
            return self.real.set(*args)

        def read(self):
            self.count += 1
            return (False, None) if self.count == 3 else self.real.read()

        def release(self):
            self.real.release()

    monkeypatch.setattr("iris.video_reviews.cv2.VideoCapture", BrokenCapture)
    with pytest.raises(ValueError, match="Cannot decode"):
        prepare(workspace)
    assert not workspace[0].list("video_reviews")
    assert not list(workspace[0].root.glob("video_reviews/*"))


def test_source_replacement_blocks_prepare_and_queue(workspace):
    store, asset, jobs = workspace
    preview = prepare(workspace)
    source = store.root / asset["path"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed since import"):
        queue_review(store, jobs, preview["id"])
    with pytest.raises(ValueError, match="changed since import"):
        prepare(workspace)
    assert not store.list("jobs")


@pytest.mark.parametrize("mutation", ["bytes", "count", "config", "path", "size", "dimensions"])
def test_tampered_preview_is_rejected_before_generation(workspace, mutation):
    store, _, jobs = workspace
    preview = prepare(workspace)
    raw = store.get("video_reviews", preview["id"])
    if mutation == "bytes":
        path = store.root / raw["images"][0]["path"]
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    elif mutation == "count":
        raw["images"].pop()
    elif mutation == "config":
        raw["config"]["instructions"] = "changed"
    elif mutation == "path":
        raw["images"][0]["path"] = "../outside.jpg"
    elif mutation == "size":
        raw["images"][0]["size_bytes"] = 100_000_000
    else:
        raw["images"][0]["width"] = 513
    store.update("video_reviews", preview["id"], {"config": raw["config"], "images": raw["images"]})
    with pytest.raises(ValueError):
        queue_review(store, jobs, preview["id"])
    assert not store.list("jobs")


def test_expiry_is_visible_and_blocks_queue(workspace):
    store, _, jobs = workspace
    preview = prepare(workspace)
    store.update(
        "video_reviews",
        preview["id"],
        {"expires_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()},
    )
    assert get_review(store, preview["id"])["status"] == "expired"
    with pytest.raises(ValueError, match="expired"):
        queue_review(store, jobs, preview["id"])


@pytest.mark.parametrize("changed", ["model_digest", "version", "model", "endpoint"])
def test_provider_identity_change_blocks_queue(workspace, monkeypatch, changed):
    preview = prepare(workspace)
    status = {**READY, changed: "changed"}
    monkeypatch.setattr("iris.video_reviews._provider_status", lambda *args, **kwargs: status)
    with pytest.raises(ValueError, match="provider changed"):
        queue_review(workspace[0], workspace[2], preview["id"])


def test_queue_is_once_only_and_only_one_review_per_asset(workspace):
    store, _, jobs = workspace
    first, second = prepare(workspace), prepare(workspace)
    job = queue_review(store, jobs, first["id"])
    assert job["kind"] == "video_review"
    with pytest.raises(ValueError, match="already been used"):
        queue_review(store, jobs, first["id"])
    with pytest.raises(RuntimeError, match="already queued"):
        queue_review(store, jobs, second["id"])
    jobs.cancel(job["id"])
    queue_review(store, jobs, second["id"])
    assert get_review(store, first["id"])["status"] == "cancelled"


def test_concurrent_queue_consumes_only_once(workspace, monkeypatch):
    import iris.video_reviews as service

    store, _, jobs = workspace
    preview = prepare(workspace)
    barrier = Barrier(2, timeout=10)
    original = service._check_provider

    def synchronized(*args):
        original(*args)
        barrier.wait()

    monkeypatch.setattr(service, "_check_provider", synchronized)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue_review, store, jobs, preview["id"]) for _ in range(2)]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except ValueError:
                pass
    assert len(results) == 1 and len(store.list("jobs")) == 1


@pytest.mark.parametrize(
    "approval",
    [
        {},
        {"allow_external": False},
        {"allow_external": True},
        {"allow_external": True, "max_cost_usd": 0},
        {"allow_external": True, "max_cost_usd": float("nan")},
        {"allow_external": True, "max_cost_usd": True},
    ],
)
def test_external_review_requires_explicit_exact_budget(workspace, approval):
    preview = prepare(workspace, provider="alibaba")
    assert preview["config"]["model_digest"] is None
    assert preview["config"]["model_version"] is None
    with pytest.raises(ValueError):
        queue_review(workspace[0], workspace[2], preview["id"], **approval)
    assert not workspace[0].list("jobs")


@pytest.mark.parametrize("approval", [{"allow_external": True}, {"max_cost_usd": 1}])
def test_local_review_rejects_external_approval_fields(workspace, approval):
    preview = prepare(workspace)
    with pytest.raises(ValueError, match="only valid"):
        queue_review(workspace[0], workspace[2], preview["id"], **approval)


def test_external_approved_exact_bytes_use_fixture_only(workspace):
    store, _, _ = workspace
    preview = prepare(workspace, provider="alibaba")
    budget = preview["config"]["estimated_cost"]["upper_bound_usd"]
    start(workspace, preview, allow_external=True, max_cost_usd=budget)
    record = store.get("video_reviews", preview["id"])
    assert record["config"] == preview["config"]
    assert record["metadata"]["consent"]["image_hashes"] == [
        sample["sha256"] for sample in preview["images"]
    ]
    run_video_review(store, preview["id"], lambda *_: None, lambda: False, FixtureReviewer)
    assert FixtureReviewer.calls[-1][0] == read_review_images(
        store, store.get("video_reviews", preview["id"])
    )
    assert get_review(store, preview["id"])["status"] == "succeeded"


def test_worker_publication_is_atomic_and_retains_provenance(workspace):
    store, _, _ = workspace
    preview = complete(workspace)
    assert preview["status"] == "succeeded"
    assert preview["prompt"] == "Fixture prompt"
    assert preview["raw_response"]["fixture"] is True
    assert preview["metadata"]["attempted_at"]
    assert preview["result"] == preview["job"]["result"]["result"]
    first, last = preview["result"]["passages"]
    assert first == {
        **RESPONSE["passages"][0],
        "id": "p1",
        "start_frame_index": 8,
        "end_frame_index": 16,
        "start_seconds": 2.0,
        "end_seconds": 4.25,
    }
    assert last["start_seconds"] == 7.75 and last["end_seconds"] == 8
    assert (
        not store.list("frames")
        and not store.list("annotation_revisions")
        and not store.list("annotation_suggestions")
    )
    assert get_review(Store(store.root), preview["id"]) == preview


def test_cancel_before_generation_never_constructs_reviewer(workspace):
    preview = prepare(workspace)
    start(workspace, preview)

    def forbidden(**_):
        pytest.fail("Cancelled review must never instantiate provider")

    result = run_video_review(workspace[0], preview["id"], lambda *_: None, lambda: True, forbidden)
    assert result["cancelled"]
    assert "attempted_at" not in get_review(workspace[0], preview["id"])["metadata"]


def test_cancel_after_generation_preserves_raw_but_no_result(workspace):
    store, _, jobs = workspace
    preview = prepare(workspace)
    job = start(workspace, preview)

    class Cancel(FixtureReviewer):
        def review(self, *args):
            result = super().review(*args)
            jobs.cancel(job["id"])
            return result

    result = run_video_review(store, preview["id"], lambda *_: None, lambda: False, Cancel)
    saved = get_review(store, preview["id"])
    assert result["cancelled"] and saved["result"] is None
    assert saved["raw_response"]["fixture"]
    assert not store.list("frames")


def test_provider_error_retains_raw_and_prevents_replay(workspace):
    store, _, _ = workspace
    preview = prepare(workspace)
    start(workspace, preview)

    class Broken(FixtureReviewer):
        def review(self, *args):
            raise ProviderResponseError(
                "Fixture invalid response",
                raw_response={"fixture": "invalid"},
                prompt="Failure prompt",
                metadata=self.metadata,
            )

    with pytest.raises(ProviderResponseError, match="invalid response"):
        run_video_review(store, preview["id"], lambda *_: None, lambda: False, Broken)
    saved = get_review(store, preview["id"])
    assert saved["raw_response"] == {"fixture": "invalid"}
    assert saved["prompt"] == "Failure prompt" and saved["result"] is None
    with pytest.raises(ValueError, match="already attempted"):
        run_video_review(store, preview["id"], lambda *_: None, lambda: False, FixtureReviewer)


def test_service_revalidates_provider_result(workspace):
    store, _, _ = workspace
    preview = prepare(workspace)
    start(workspace, preview)

    class Invented(FixtureReviewer):
        def review(self, *args):
            response = super().review(*args)
            response["passages"][0]["start_sample_id"] = "s99"
            return response

    with pytest.raises(ValueError, match="supplied sample"):
        run_video_review(store, preview["id"], lambda *_: None, lambda: False, Invented)
    saved = get_review(store, preview["id"])
    assert saved["result"] is None and saved["raw_response"]


def test_source_changed_during_review_blocks_publication(workspace):
    store, asset, _ = workspace
    preview = prepare(workspace)
    start(workspace, preview)

    class Changed(FixtureReviewer):
        def review(self, *args):
            result = super().review(*args)
            store.update("assets", asset["id"], {"metadata": {**asset["metadata"], "fps": 3.0}})
            return result

    with pytest.raises(ValueError, match="timing metadata changed"):
        run_video_review(store, preview["id"], lambda *_: None, lambda: False, Changed)
    saved = get_review(store, preview["id"])
    assert saved["result"] is None and saved["raw_response"]


def test_changed_consent_and_pricing_block_external_worker(workspace, monkeypatch):
    store, _, _ = workspace
    preview = prepare(workspace, provider="alibaba")
    start(workspace, preview, allow_external=True, max_cost_usd=1)
    monkeypatch.setattr("iris.video_reviews._estimate", lambda *_: {"upper_bound_usd": 2})
    with pytest.raises(ValueError, match="pricing changed"):
        run_video_review(store, preview["id"], lambda *_: None, lambda: False, FixtureReviewer)
    assert "attempted_at" not in get_review(store, preview["id"])["metadata"]


def test_worker_checks_source_before_generation(workspace):
    store, asset, _ = workspace
    preview = prepare(workspace)
    start(workspace, preview)
    path = store.root / asset["path"]
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        run_video_review(store, preview["id"], lambda *_: None, lambda: False, FixtureReviewer)
    assert "attempted_at" not in get_review(store, preview["id"])["metadata"]


def test_extraction_union_and_coverage_are_deterministic_without_writes(workspace):
    store, asset, _ = workspace
    preview = complete(workspace)
    plan = preview_passage_extraction(
        store,
        preview["id"],
        passage_ids=["p2", "p1"],
        frames_per_passage=3,
        context_seconds=0,
        coverage_frames=4,
    )
    assert plan["passage_ids"] == ["p1", "p2"]
    assert [position["frame_index"] for position in plan["positions"]] == [
        0,
        8,
        12,
        13,
        16,
        26,
        31,
        39,
    ]
    assert plan["planned_count"] == 8 and plan["max_frames"] == 10
    assert plan["source_sha256"] == asset["sha256"]
    assert plan["sampling_mode"] == "passages"
    assert plan["algorithm"] == "iris-video-passages-v1"
    assert validate_passage_extraction(store, plan) == plan
    assert not store.list("frames")
    assert len(store.list("jobs")) == 1


def test_context_clips_to_reviewed_range_and_single_sample_is_nonempty(workspace):
    store, _, _ = workspace
    preview = complete(workspace, start_seconds=1, end_seconds=9)
    plan = preview_passage_extraction(
        store,
        preview["id"],
        passage_ids=["p1", "p2"],
        context_seconds=30,
        frames_per_passage=50,
        coverage_frames=32,
    )
    assert plan["planned_count"] == 32
    assert all(
        segment["start_seconds"] == 1 and segment["end_seconds"] == 9 for segment in plan["ranges"]
    )
    assert plan["positions"][0]["frame_index"] == 4 and plan["positions"][-1]["frame_index"] == 35


@pytest.mark.parametrize(
    "kwargs",
    [
        {"passage_ids": []},
        {"passage_ids": ["p1", "p1"]},
        {"passage_ids": ["absent"]},
        {"passage_ids": "p1"},
        {"passage_ids": [True]},
        {"frames_per_passage": 0},
        {"frames_per_passage": 51},
        {"frames_per_passage": True},
        {"coverage_frames": -1},
        {"coverage_frames": 33},
        {"coverage_frames": False},
        {"context_seconds": -1},
        {"context_seconds": 31},
        {"context_seconds": float("nan")},
        {"context_seconds": True},
    ],
)
def test_invalid_passage_selection_is_read_only(workspace, kwargs):
    store, _, _ = workspace
    preview = complete(workspace)
    options = {"passage_ids": ["p1"], **kwargs}
    with pytest.raises(ValueError):
        preview_passage_extraction(store, preview["id"], **options)
    assert not store.list("frames") and len(store.list("jobs")) == 1


def test_unfinished_and_tampered_results_cannot_drive_extraction(workspace):
    store, _, _ = workspace
    preview = prepare(workspace)
    with pytest.raises(ValueError, match="successfully completed"):
        preview_passage_extraction(store, preview["id"], passage_ids=["p1"])
    start(workspace, preview)
    run_video_review(store, preview["id"], lambda *_: None, lambda: False, FixtureReviewer)
    result = get_review(store, preview["id"])["result"]
    result["passages"][0]["start_seconds"] = 0
    store.update("video_reviews", preview["id"], {"result": result})
    with pytest.raises(ValueError, match="passages changed"):
        preview_passage_extraction(store, preview["id"], passage_ids=["p1"])


def test_frozen_extraction_plan_rejects_changed_indices(workspace):
    preview = complete(workspace)
    plan = preview_passage_extraction(workspace[0], preview["id"], passage_ids=["p1"])
    plan["positions"][0]["frame_index"] += 1
    with pytest.raises(ValueError, match="plan changed"):
        validate_passage_extraction(workspace[0], plan)


def test_empty_model_proposals_are_successful_but_require_manual_sampling(workspace):
    preview = prepare(workspace)
    start(workspace, preview)

    class Empty(FixtureReviewer):
        def review(self, *args):
            result = super().review(*args)
            result["passages"] = []
            return result

    run_video_review(workspace[0], preview["id"], lambda *_: None, lambda: False, Empty)
    assert get_review(workspace[0], preview["id"])["result"]["passages"] == []
    with pytest.raises(ValueError, match="does not belong"):
        preview_passage_extraction(workspace[0], preview["id"], passage_ids=["p1"])


def test_schema_10_upgrade_preserves_every_existing_row(workspace):
    store, _, _ = workspace
    snapshot = {table: store.list(table) for table in store.columns if table != "video_reviews"}
    with store.connect() as conn:
        conn.execute("DROP TABLE video_reviews")
        conn.execute("PRAGMA user_version=10")
    reopened = Store(store.root)
    assert {
        table: reopened.list(table) for table in reopened.columns if table != "video_reviews"
    } == snapshot
    with reopened.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 11
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert reopened.list("video_reviews") == []


def test_environment_default_model_is_used_without_explicit_selection(monkeypatch):
    from iris.video_reviews import _provider_status

    monkeypatch.setenv("IRIS_OLLAMA_MODEL", "fixture-custom-model")
    captured = []
    monkeypatch.setattr(
        "iris.video_reviews.provider_status", lambda config: captured.append(config) or READY
    )
    _provider_status("ollama")
    assert captured[0]["model"] == "fixture-custom-model"


@pytest.mark.parametrize("change", ["consent", "expiry", "source"])
def test_changes_during_reviewer_construction_prevent_attempt(workspace, change):
    store, asset, _ = workspace
    preview = prepare(workspace, provider="alibaba")
    start(workspace, preview, allow_external=True, max_cost_usd=1)

    class Changed(FixtureReviewer):
        def __init__(self, config):
            super().__init__(config)
            if change == "consent":
                metadata = store.get("video_reviews", preview["id"])["metadata"]
                metadata["consent"]["allow_external"] = False
                store.update("video_reviews", preview["id"], {"metadata": metadata})
            elif change == "expiry":
                store.update(
                    "video_reviews",
                    preview["id"],
                    {"expires_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()},
                )
            else:
                store.update("assets", asset["id"], {"metadata": {**asset["metadata"], "fps": 2.0}})

        def review(self, *args):
            pytest.fail("Changed input must not trigger generation")

    with pytest.raises(ValueError):
        run_video_review(store, preview["id"], lambda *_: None, lambda: False, Changed)
    assert "attempted_at" not in get_review(store, preview["id"])["metadata"]


def test_source_change_during_queue_provider_probe_is_rejected(workspace, monkeypatch):
    store, asset, jobs = workspace
    preview = prepare(workspace)

    def changed(*args, **kwargs):
        store.update("assets", asset["id"], {"metadata": {**asset["metadata"], "fps": 2.0}})
        return deepcopy(READY)

    monkeypatch.setattr("iris.video_reviews._provider_status", changed)
    with pytest.raises(ValueError, match="timing metadata changed"):
        queue_review(store, jobs, preview["id"])
    assert not store.list("jobs")


def test_expiry_during_queue_probe_is_rejected(workspace, monkeypatch):
    store, _, jobs = workspace
    preview = prepare(workspace)

    def changed(*args, **kwargs):
        store.update(
            "video_reviews",
            preview["id"],
            {"expires_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()},
        )
        return deepcopy(READY)

    monkeypatch.setattr("iris.video_reviews._provider_status", changed)
    with pytest.raises(ValueError, match="expired"):
        queue_review(store, jobs, preview["id"])
    assert not store.list("jobs")


def test_final_progress_cancellation_preserves_no_output(workspace):
    store, _, jobs = workspace
    preview = prepare(workspace)
    job = start(workspace, preview)

    class Forbidden(FixtureReviewer):
        def review(self, *args):
            pytest.fail("Cancelled after progress must not generate")

    result = run_video_review(
        store, preview["id"], lambda *_: jobs.cancel(job["id"]), lambda: False, Forbidden
    )
    assert result["cancelled"]
    saved = get_review(store, preview["id"])
    assert saved["result"] is None and saved["raw_response"] is None


def test_model_identity_change_at_construction_prevents_attempt(workspace):
    preview = prepare(workspace)
    start(workspace, preview)

    class Changed(FixtureReviewer):
        def __init__(self, config):
            super().__init__(config)
            self.metadata["model_digest"] = "b" * 64

    with pytest.raises(ValueError, match="model changed before generation"):
        run_video_review(workspace[0], preview["id"], lambda *_: None, lambda: False, Changed)
    assert "attempted_at" not in get_review(workspace[0], preview["id"])["metadata"]


def test_point_passage_near_video_boundary_keeps_observed_anchor(workspace):
    store, _, _ = workspace
    preview = prepare(workspace)
    start(workspace, preview)

    class Point(FixtureReviewer):
        def review(self, *args):
            result = super().review(*args)
            result["passages"] = [
                {**RESPONSE["passages"][0], "start_sample_id": "s1", "end_sample_id": "s1"}
            ]
            return result

    run_video_review(store, preview["id"], lambda *_: None, lambda: False, Point)
    plan = preview_passage_extraction(
        store,
        preview["id"],
        passage_ids=["p1"],
        frames_per_passage=1,
        context_seconds=2,
        coverage_frames=0,
    )
    assert plan["ranges"] == [{"passage_id": "p1", "start_seconds": 0.0, "end_seconds": 2.25}]
    assert plan["positions"] == [{"frame_index": 0, "timestamp_seconds": 0.0}]


def test_two_frame_passage_keeps_both_observed_anchors_with_context(workspace):
    preview = complete(workspace)
    plan = preview_passage_extraction(
        workspace[0],
        preview["id"],
        passage_ids=["p1"],
        frames_per_passage=2,
        context_seconds=2,
        coverage_frames=0,
    )
    assert [position["frame_index"] for position in plan["positions"]] == [8, 16]
    assert plan["ranges"][0]["start_seconds"] == 0
    assert plan["ranges"][0]["end_seconds"] == 6.25


def test_one_frame_passage_chooses_an_observed_sample_not_unknown_midpoint(workspace):
    preview = complete(workspace)
    plan = preview_passage_extraction(
        workspace[0],
        preview["id"],
        passage_ids=["p1"],
        frames_per_passage=1,
        context_seconds=30,
        coverage_frames=0,
    )
    assert plan["positions"] == [{"frame_index": 8, "timestamp_seconds": 2.0}]


def test_larger_budget_keeps_anchors_and_sampled_context(workspace):
    preview = complete(workspace)
    plan = preview_passage_extraction(
        workspace[0],
        preview["id"],
        passage_ids=["p1"],
        frames_per_passage=6,
        context_seconds=2,
        coverage_frames=0,
    )
    indices = [position["frame_index"] for position in plan["positions"]]
    assert len(indices) == 6
    assert {0, 8, 16, 24} <= set(indices)
    assert len(indices) == len(set(indices))
