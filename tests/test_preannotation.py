"""Local detector orchestration uses generated pixels and explicit synthetic detectors."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
from PIL import Image

from iris import inference, preannotation_contracts
from iris.annotations import (
    add_detector_suggestions,
    adopt_taxonomy,
    get_annotation,
    save_annotation,
)
from iris.dataset_manifest import taxonomy_mappings
from iris.jobs import JobManager
from iris.media import import_asset
from iris.preannotation import (
    PreannotationConflict,
    create_preannotation,
    list_preannotations,
    preannotation_detail,
    preview_preannotation,
)
from iris.store import Store, new_id, now
from iris.taxonomies import TAXONOMY, publish_taxonomy

MODEL_ID = "ssdlite320_mobilenet_v3_large"
TIMING = {"preprocess_ms": 1, "inference_ms": 2, "postprocess_ms": 1, "total_ms": 4}
DETECTION = {"label_id": 1, "label": "person", "score": 0.85, "box": [2, 3, 30, 35]}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    session = store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": "Synthetic images",
            "scene_group": "preannotation",
            "created_at": now(),
        },
    )
    frames = []
    for index in range(2):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (80, 60), (index * 50, 30, 50)).save(path)
        asset = import_asset(store, session["id"], path, path.name)
        frames.append(store.list("frames", asset_id=asset["id"])[0])
    model = {**inference.get_spec(MODEL_ID), "status": "ready", "weight_sha256": "a" * 64}
    monkeypatch.setattr(inference, "catalog", lambda _: [deepcopy(model)])
    return store, JobManager(store), session, frames, model


def settings(workspace, **changes):
    return {
        "frame_ids": [frame["id"] for frame in workspace[3]],
        "model_id": workspace[4]["id"],
        "threshold": 0.5,
        **changes,
    }


def queue(workspace, **changes):
    store, jobs, session, _, _ = workspace
    options = settings(workspace, **changes)
    preview = preview_preannotation(store, session["id"], **options)
    return create_preannotation(
        store,
        jobs,
        session["id"],
        name="Synthetic detector proposals",
        expected_fingerprint=preview["fingerprint"],
        **options,
    )


def factory(workspace, *, detections=None, on_predict=None, outputs=None):
    model = workspace[4]

    class Detector:
        def __init__(self, *args, **kwargs):
            self.metadata = {"weight_sha256": model["weight_sha256"]}
            self.calls = 0

        def warmup(self, image):
            pass

        def predict(self, image):
            self.calls += 1
            if on_predict:
                on_predict(self.calls)
            if outputs is not None:
                return deepcopy(outputs)
            return {
                "input_size": list(image.size),
                "timing": dict(TIMING),
                "detections": deepcopy([DETECTION] if detections is None else detections),
            }

    return Detector


def run(workspace, comparison, **options):
    store = workspace[0]
    return inference.run_comparison(
        store,
        comparison["id"],
        options.pop("progress", lambda *_: None),
        options.pop("cancelled", lambda: False),
        detector_factory=factory(workspace, **options),
    )


def test_preview_is_read_only_and_covers_bounded_work_without_annotations(workspace):
    store, _, session, _, _ = workspace
    before = {table: store.list(table) for table in store.columns}
    preview = preview_preannotation(store, session["id"], **settings(workspace))
    assert preview["eligible_count"] == 2 and preview["excluded_count"] == 0
    assert preview["work"]["total_forward_passes"] == 3
    assert all(
        row["base_revision"] == 0 and row["coverage"]["coverage_complete"]
        for row in preview["frames"]
    )
    assert preview["source"]["provider"] == "local_detector"
    assert {table: store.list(table) for table in before} == before


@pytest.mark.parametrize("mode", ["full", "tiled"])
def test_outputs_become_pending_proposals_without_human_revision(workspace, mode):
    store, _, _, frames, _ = workspace
    comparison = queue(workspace, inference_mode=mode, tile_size=128)
    frozen = deepcopy(store.get("comparisons", comparison["id"]))
    result = run(workspace, comparison)
    assert result["predictions_created"] == 2
    assert store.list("annotation_revisions") == []
    suggestions = store.list("annotation_suggestions")
    assert len(suggestions) == 2
    for suggestion in suggestions:
        metadata = suggestion["metadata"]
        assert metadata["preannotation_id"] == comparison["id"]
        assert metadata["target_taxonomy"] == TAXONOMY["id"]
        assert metadata["base_revision"] == 0
        assert metadata["geometry"]["coordinates"]["format"] == "xyxy"
        (saved_detection,) = store.get("predictions", metadata["prediction_id"])["detections"]
        assert {key: saved_detection[key] for key in DETECTION} == DETECTION
        annotation = get_annotation(store, suggestion["frame_id"])
        assert annotation["revision"] == 0 and annotation["boxes"] == []
        assert annotation["suggestions"][0]["state"] == "pending"
    detail = preannotation_detail(store, comparison["id"])
    assert detail["counts"]["pending_review"] == 2 and detail["counts"]["proposals"] == 2
    assert all(row["state"] == "pending_review" for row in detail["frames"])
    assert store.get("comparisons", comparison["id"]) == frozen
    # Importing the same saved source through the old manual action stays idempotent.
    prediction = store.list("predictions", frame_id=frames[0]["id"])[0]
    add_detector_suggestions(
        store, frames[0]["id"], prediction_id=prediction["id"], expected_revision=0
    )
    assert len(store.list("annotation_suggestions")) == 2
    with pytest.raises(ValueError, match="immutable"):
        run(workspace, comparison)


def test_confirmation_receipt_survives_response_loss_and_explicit_new_preview(
    workspace, monkeypatch
):
    store, jobs, session, _, _ = workspace
    options = settings(workspace)
    preview = preview_preannotation(store, session["id"], **options)
    first = create_preannotation(
        store,
        jobs,
        session["id"],
        name="First name",
        expected_fingerprint=preview["fingerprint"],
        **options,
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            inference, "catalog", lambda _: pytest.fail("Receipt must not recheck models")
        )
        again = create_preannotation(
            store,
            jobs,
            session["id"],
            name="Renamed after preview",
            expected_fingerprint=preview["fingerprint"],
            **options,
        )
    assert again["id"] == first["id"] and again["name"] == "First name"
    assert len(store.list("jobs")) == 1
    jobs.cancel(first["job_id"])
    fresh = preview_preannotation(store, session["id"], **options)
    assert fresh["fingerprint"] != preview["fingerprint"]
    second = create_preannotation(
        store,
        jobs,
        session["id"],
        name="Explicit new request",
        expected_fingerprint=fresh["fingerprint"],
        **options,
    )
    assert second["id"] != first["id"]
    assert len(list_preannotations(store, session["id"])) == 2


def test_concurrent_confirmations_share_one_receipt(workspace):
    store, _, session, _, _ = workspace
    options = settings(workspace)
    preview = preview_preannotation(store, session["id"], **options)
    barrier = Barrier(2)

    def confirm(_):
        barrier.wait(5)
        return create_preannotation(
            store,
            JobManager(store),
            session["id"],
            name="Concurrent",
            expected_fingerprint=preview["fingerprint"],
            **options,
        )["id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        identifiers = list(pool.map(confirm, range(2)))
    assert len(set(identifiers)) == 1 and len(store.list("jobs")) == 1


@pytest.mark.parametrize("change", ["revision", "weights", "pixels"])
def test_preview_changes_require_confirmation_again(workspace, change):
    store, jobs, session, frames, model = workspace
    options = settings(workspace)
    preview = preview_preannotation(store, session["id"], **options)
    if change == "revision":
        save_annotation(store, frames[0]["id"], expected_revision=0, boxes=[], decisions={})
    elif change == "weights":
        model["weight_sha256"] = "b" * 64
    else:
        Image.new("RGB", (80, 60), "white").save(store.artifact_path(frames[0]["path"]))
    with pytest.raises(PreannotationConflict, match="changed"):
        create_preannotation(
            store,
            jobs,
            session["id"],
            name="Stale",
            expected_fingerprint=preview["fingerprint"],
            **options,
        )
    assert store.list("comparisons") == [] and store.list("jobs") == []


@pytest.mark.parametrize("when", ["before", "during"])
def test_human_edits_keep_raw_predictions_and_stop_proposal_publication(workspace, when):
    store, _, _, frames, _ = workspace
    comparison = queue(workspace)

    def edit():
        return save_annotation(
            store,
            frames[0]["id"],
            expected_revision=0,
            boxes=[],
            decisions={},
            status="validated",
            reviewer="Human fixture",
        )

    if when == "before":
        edit()
    run(
        workspace,
        comparison,
        on_predict=lambda count: edit() if when == "during" and count == 1 else None,
    )
    detail = preannotation_detail(store, comparison["id"])
    assert detail["counts"]["conflict"] == detail["counts"]["pending_review"] == 1
    assert detail["counts"]["predictions"] == 2
    assert store.list("annotation_suggestions", frame_id=frames[0]["id"]) == []
    annotation = get_annotation(store, frames[0]["id"])
    assert annotation["revision"] == 1 and annotation["status"] == "validated"
    assert annotation["boxes"] == []


def test_midrun_cancellation_retains_saved_prefix(workspace):
    store, jobs, _, frames, _ = workspace
    comparison = queue(workspace)

    def progress(_value, message):
        if message.startswith("Saved 1"):
            jobs.cancel(comparison["job_id"])

    result = run(
        workspace,
        comparison,
        progress=progress,
        cancelled=lambda: store.get("jobs", comparison["job_id"])["cancel_requested"],
    )
    assert result["cancelled"] and result["predictions_created"] == 1
    assert len(store.list("annotation_suggestions")) == 1
    assert store.list("annotation_suggestions", frame_id=frames[1]["id"]) == []
    detail = preannotation_detail(store, comparison["id"])
    assert detail["counts"]["pending_review"] == detail["counts"]["cancelled"] == 1


def test_cancel_after_forward_preserves_raw_without_proposals(workspace):
    store, jobs, _, _, _ = workspace
    comparison = queue(workspace)
    result = run(
        workspace,
        comparison,
        on_predict=lambda _: jobs.cancel(comparison["job_id"]),
        cancelled=lambda: store.get("jobs", comparison["job_id"])["cancel_requested"],
    )
    assert result["cancelled"] and result["predictions_created"] == 1
    assert len(store.list("predictions")) == 1 and store.list("annotation_suggestions") == []
    assert preannotation_detail(store, comparison["id"])["frames"][0]["state"] == "cancelled"


def test_empty_result_never_creates_negative_annotation(workspace):
    store = workspace[0]
    comparison = queue(workspace)
    run(workspace, comparison, detections=[])
    assert preannotation_detail(store, comparison["id"])["counts"]["no_proposals"] == 2
    assert store.list("annotation_suggestions") == store.list("annotation_revisions") == []


@pytest.mark.parametrize(
    "detection",
    [
        {**DETECTION, "label_id": 3},
        {**DETECTION, "score": -0.1},
        {**DETECTION, "box": [-1, 0, 8, 8]},
        {**DETECTION, "label_id": True},
        {**DETECTION, "score": float("nan")},
    ],
)
def test_invalid_detector_output_is_saved_as_evidence_without_proposals(workspace, detection):
    store = workspace[0]
    comparison = queue(workspace)
    run(workspace, comparison, detections=[detection])
    detail = preannotation_detail(store, comparison["id"])
    assert detail["counts"]["invalid_output"] == 2
    assert len(detail["predictions"]) == 2
    assert all(
        "raw_output" in row["metadata"] or row["metadata"].get("raw_output_not_json")
        for row in detail["predictions"]
    )
    assert store.list("annotation_suggestions") == store.list("annotation_revisions") == []


def test_excess_proposals_are_visible_errors_not_silent_truncation(workspace):
    store = workspace[0]
    comparison = queue(workspace)
    run(workspace, comparison, detections=[deepcopy(DETECTION) for _ in range(101)])
    detail = preannotation_detail(store, comparison["id"])
    assert detail["counts"]["invalid_output"] == 2
    assert all("100" in frame["reason"] for frame in detail["frames"])
    assert all(len(row["detections"]) == 101 for row in detail["predictions"])
    assert store.list("annotation_suggestions") == []


def custom_classes(workspace):
    store, _, _, frames, _ = workspace
    taxonomy = publish_taxonomy(
        store,
        "default",
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[
            {"id": "helmet", "name": "Helmet", "definition": "A protective helmet."},
            {"id": "vehicle", "name": "Vehicle", "definition": "A passenger car.", "coco_id": 3},
        ],
    )
    for frame in frames:
        adopt_taxonomy(
            store,
            frame["id"],
            expected_revision=0,
            expected_taxonomy_id=TAXONOMY["id"],
            target_taxonomy_id=taxonomy["id"],
        )
    return taxonomy


def test_official_detector_uses_explicit_custom_mapping_and_warns_uncovered_classes(workspace):
    store, _, session, _, _ = workspace
    custom_classes(workspace)
    preview = preview_preannotation(store, session["id"], **settings(workspace))
    assert preview["eligible_count"] == 2
    assert preview["frames"][0]["coverage"]["unsupported_class_ids"] == ["helmet"]
    comparison = queue(workspace)
    run(workspace, comparison, detections=[DETECTION, {**DETECTION, "label_id": 3, "label": "car"}])
    assert {row["label"] for row in store.list("annotation_suggestions")} == {"vehicle"}
    detail = preannotation_detail(store, comparison["id"])
    assert all(frame["unmapped_count"] == 1 for frame in detail["frames"])


def test_trained_custom_output_ids_keep_their_frozen_meaning(workspace, monkeypatch):
    store, _, _, _, model = workspace
    taxonomy = custom_classes(workspace)
    internal, output = taxonomy_mappings(taxonomy)
    model.update(
        origin="trained",
        taxonomy=taxonomy,
        taxonomy_id=taxonomy["id"],
        class_mapping=internal,
        output_class_mapping=output,
    )
    monkeypatch.setattr(inference, "get_spec", lambda *_: deepcopy(model))
    monkeypatch.setattr(preannotation_contracts, "get_spec", lambda *_: deepcopy(model))
    comparison = queue(workspace)
    run(
        workspace,
        comparison,
        detections=[{**DETECTION, "label": "helmet", "label_id": 1, "taxonomy_id": taxonomy["id"]}],
    )
    assert {row["label"] for row in store.list("annotation_suggestions")} == {"helmet"}
    assert all(
        row["metadata"]["target_taxonomy"] == taxonomy["id"]
        for row in store.list("annotation_suggestions")
    )


def test_class_adoption_during_detection_is_an_explicit_conflict(workspace):
    store = workspace[0]
    comparison = queue(workspace)
    run(
        workspace,
        comparison,
        on_predict=lambda count: custom_classes(workspace) if count == 1 else None,
    )
    detail = preannotation_detail(store, comparison["id"])
    assert detail["counts"]["conflict"] == 2 and len(detail["predictions"]) == 2
    assert store.list("annotation_suggestions") == []


def test_crash_after_raw_save_keeps_inspectable_unpublished_output(workspace, monkeypatch):
    store, jobs, _, _, _ = workspace
    comparison = queue(workspace)
    monkeypatch.setattr(
        preannotation_contracts,
        "normalize_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit("Fixture crash")),
    )
    with pytest.raises(SystemExit):
        run(workspace, comparison)
    jobs._interrupt_unfinished()
    detail = preannotation_detail(store, comparison["id"])
    assert detail["counts"]["raw_saved"] == detail["counts"]["interrupted"] == 1
    assert len(detail["predictions"]) == 1 and store.list("annotation_suggestions") == []
    with pytest.raises(ValueError, match="immutable"):
        run(workspace, comparison)


@pytest.mark.parametrize(
    "options",
    [
        {"frame_ids": []},
        {"frame_ids": ["same"] * 26},
        {"threshold": True},
        {"threshold": float("inf")},
        {"threshold": -0.1},
        {"inference_mode": "paired"},
    ],
)
def test_invalid_or_unbounded_requests_are_rejected(workspace, options):
    store, _, session, _, _ = workspace
    with pytest.raises(ValueError):
        preview_preannotation(store, session["id"], **settings(workspace, **options))
    assert store.list("jobs") == []


@pytest.mark.parametrize("corruption", ["kind", "comparison", "terminal"])
def test_invalid_job_provenance_stops_before_loading_detector(workspace, corruption):
    store = workspace[0]
    comparison = queue(workspace)
    change = (
        {"kind": "assist"}
        if corruption == "kind"
        else (
            {"params": {"comparison_id": "foreign-comparison"}}
            if corruption == "comparison"
            else {"status": "interrupted"}
        )
    )
    store.update("jobs", comparison["job_id"], change)
    with pytest.raises(ValueError, match="provenance|terminal"):
        inference.run_comparison(
            store,
            comparison["id"],
            lambda *_: None,
            lambda: False,
            detector_factory=lambda *_args, **_kwargs: pytest.fail("Do not load a detector"),
        )
    assert (
        store.list("runs")
        == store.list("predictions")
        == store.list("annotation_suggestions")
        == []
    )


def test_pixels_changed_after_forward_keep_raw_but_publish_no_proposal(workspace):
    store, _, _, frames, _ = workspace
    comparison = queue(workspace)

    def change_pixels(count):
        if count == 1:
            Image.new("RGB", (80, 60), "white").save(store.artifact_path(frames[0]["path"]))

    run(workspace, comparison, on_predict=change_pixels)
    detail = preannotation_detail(store, comparison["id"])
    assert detail["counts"]["conflict"] == detail["counts"]["pending_review"] == 1
    assert detail["counts"]["predictions"] == 2
    assert store.list("annotation_suggestions", frame_id=frames[0]["id"]) == []


def test_partial_summary_survives_later_detector_failure(workspace):
    store = workspace[0]
    comparison = queue(workspace)

    def fail_second(count):
        if count == 2:
            raise RuntimeError("Synthetic later frame inference failure")

    with pytest.raises(RuntimeError, match="later frame"):
        run(workspace, comparison, on_predict=fail_second)
    result = store.get("jobs", comparison["job_id"])["result"]
    assert result["predictions_created"] == 1
    assert (
        result["preannotation"]["frames_ready"]
        == result["preannotation"]["suggestions_created"]
        == 1
    )
    assert result["preannotation"]["frames_issues"] == 0
    assert len(store.list("predictions")) == len(store.list("annotation_suggestions")) == 1


def test_unmapped_classes_are_excluded_instead_of_guessed(workspace):
    store, _, session, frames, _ = workspace
    taxonomy = publish_taxonomy(
        store,
        "default",
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[{"id": "helmet", "name": "Helmet", "definition": "A protective helmet."}],
    )
    for frame in frames:
        adopt_taxonomy(
            store,
            frame["id"],
            expected_revision=0,
            expected_taxonomy_id=TAXONOMY["id"],
            target_taxonomy_id=taxonomy["id"],
        )
    preview = preview_preannotation(store, session["id"], **settings(workspace))
    assert preview["eligible_count"] == 0 and preview["excluded_count"] == 2
    assert preview["work"]["total_forward_passes"] == 0
    assert all("covers none" in row["reason"] for row in preview["frames"])
    with pytest.raises(ValueError, match="No images"):
        queue(workspace)
    assert store.list("jobs") == []
