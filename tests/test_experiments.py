"""Frozen reports built from saved synthetic evaluation results, without inference."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
from test_evaluation_analysis import MODELS, record_evaluation
from test_evaluation_analysis import saved as evaluation_fixture
from test_tiled_evaluation_analysis import versioned_evaluation

from iris import experiments
from iris.experiments import (
    ExperimentConflict,
    create_experiment,
    experiment_detail,
    list_experiments,
    preview_experiment,
    read_experiment_image,
    update_experiment,
)
from iris.store import Store, new_id, now

saved = evaluation_fixture


def create(saved, **fields):
    store, detail, _, _ = saved
    return create_experiment(
        store, evaluation_id=detail["id"], title="Synthetic experience", **fields
    )


def test_preview_is_read_only_and_normalizes_legacy_run_ids(saved, monkeypatch):
    store, detail, _, _ = saved
    before = {table: store.list(table) for table in store.columns}
    monkeypatch.setattr("iris.models.catalog", lambda *_: pytest.fail("No model catalog"))
    monkeypatch.setattr(
        "iris.metrics.evaluate_predictions", lambda *_: pytest.fail("No new metrics")
    )
    result = preview_experiment(store, detail["id"])
    assert result["snapshot"]["examples"] == []
    assert len(result["available_examples"]) == 2
    lanes = result["snapshot"]["lanes"]
    assert [lane["model_id"] for lane in lanes] == MODELS
    assert list(result["snapshot"]["error_analysis"]["summary"]["all"]["runs"]) == [
        lane["id"] for lane in lanes
    ]
    assert result["snapshot"]["error_analysis"]["comparison"] == {
        "baseline_run_id": lanes[0]["id"],
        "candidate_run_id": lanes[1]["id"],
    }
    assert result["snapshot"]["error_analysis"]["summary"]["all"]["changes"] == {
        "new_misses": 1,
        "recovered": 1,
        "fp_delta": 1,
    }
    assert all(lane["timing"]["mean_total_ms"] is None for lane in lanes)
    assert all(lane["training_status"] == "unavailable" for lane in lanes)
    assert {table: store.list(table) for table in store.columns} == before
    assert not (store.root / "reports").exists()


def test_creation_copies_chosen_examples_with_original_indexed_evidence(saved):
    store, detail, _, _ = saved
    chosen = detail["frames"][1]["frame_id"]
    report = create(
        saved,
        objective="Why this comparison?",
        conclusion="Fixture only.",
        example_frame_ids=[chosen],
    )
    assert report["revision"] == 1
    assert len(report["images"]) == 1 and "path" not in report["images"][0]
    example = report["snapshot"]["examples"][0]
    assert example["ground_truth"] == [
        {key: box[key] for key in ("id", "label", "box")} for box in detail["frames"][1]["boxes"]
    ]
    candidate = example["lanes"][1]
    source = next(
        row
        for row in detail["predictions"]
        if row["frame_id"] == chosen and row["model_id"] == MODELS[1]
    )
    assert candidate["detections"] == source["detections"]
    assert candidate["detections"][0]["label"] == "bicycle"
    assert candidate["detections"][1]["score"] == 0.2
    assert example["counts"]["all"]["changes"]["recovered_indices"] == [1]
    content = read_experiment_image(store, report["id"], chosen)
    assert content.startswith(b"\xff\xd8")
    assert hashlib.sha256(content).hexdigest() == example["image_sha256"]
    assert experiment_detail(Store(store.root), report["id"]) == report
    assert list_experiments(store)[0]["example_count"] == 1


def test_no_examples_requires_no_source_pixels_and_writes_no_image_directory(saved):
    store, _, dataset, _ = saved
    for frame in dataset["manifest"]["frames"]:
        store.artifact_path(frame["image_path"]).unlink()
    report = create(saved)
    assert report["images"] == report["snapshot"]["examples"] == []
    assert not (store.root / "reports").exists()


def test_saved_report_survives_missing_dataset_images_manifest_and_models(saved, monkeypatch):
    store, detail, dataset, _ = saved
    chosen = detail["frames"][0]["frame_id"]
    report = create(saved, example_frame_ids=[chosen])
    bytes_before = read_experiment_image(store, report["id"], chosen)
    for frame in dataset["manifest"]["frames"]:
        store.artifact_path(frame["image_path"]).unlink()
    store.artifact_path(dataset["path"]).unlink()
    monkeypatch.setattr(
        experiments, "evaluation_detail", lambda *_: pytest.fail("Saved report is self contained")
    )
    monkeypatch.setattr(
        experiments, "load_manifest", lambda *_: pytest.fail("Saved report is self contained")
    )
    assert experiment_detail(Store(store.root), report["id"]) == report
    assert read_experiment_image(store, report["id"], chosen) == bytes_before
    assert list_experiments(store)[0]["id"] == report["id"]


def test_paired_variants_keep_two_lanes_of_one_checkpoint(saved):
    paired = versioned_evaluation(saved)
    report = create(paired, example_frame_ids=[paired[1]["frames"][0]["frame_id"]])
    lanes = report["snapshot"]["lanes"]
    assert len({lane["model_id"] for lane in lanes}) == 1
    assert [lane["variant"] for lane in lanes] == ["full", "tiled"]
    assert len({lane["id"] for lane in lanes}) == 2
    assert report["snapshot"]["evaluation"]["config"]["inference"]["tiling"]["tile_size"] == 128
    assert report["snapshot"]["examples"][0]["lanes"][1]["variant"] == "tiled"


def test_single_lane_has_no_comparison_or_improvement_claim(saved):
    store, _, dataset, outputs = saved
    detail = record_evaluation(store, dataset, outputs, model_ids=MODELS[:1])
    report = create((store, detail, dataset, outputs))
    assert report["snapshot"]["error_analysis"]["comparison"] is None
    assert all(
        item["changes"] is None for item in report["snapshot"]["error_analysis"]["summary"].values()
    )


def test_test_split_label_and_validation_link_are_preserved(saved, monkeypatch):
    store, detail, _, _ = saved
    value = deepcopy(detail)
    value["split"] = "test"
    value["config"]["validation_evaluation_id"] = "earlier-validation"
    for frame in value["frames"]:
        frame["split"] = "test"
    monkeypatch.setattr(experiments, "evaluation_detail", lambda *_: deepcopy(value))
    report = create(saved)
    assert report["snapshot"]["evaluation"]["split"] == "test"
    assert (
        report["snapshot"]["evaluation"]["config"]["validation_evaluation_id"]
        == "earlier-validation"
    )
    assert not store.list("model_references")


@pytest.mark.parametrize(
    "fields",
    [
        {"title": ""},
        {"title": "a" * 161},
        {"title": None},
        {"objective": "a" * 4001},
        {"objective": 1},
        {"conclusion": []},
        {"example_frame_ids": ["unknown"]},
        {"example_frame_ids": ["x"] * 7},
        {"example_frame_ids": "bad"},
        {"example_frame_ids": ["../secret"]},
    ],
)
def test_invalid_creation_is_clean(saved, fields):
    store, detail, _, _ = saved
    with pytest.raises(ValueError):
        create_experiment(store, **{"evaluation_id": detail["id"], "title": "Fixture", **fields})
    assert not store.list("experiment_reports")
    assert not list(store.root.glob("reports/*"))


def test_duplicate_or_training_split_examples_are_rejected(saved):
    store, detail, dataset, _ = saved
    selected = detail["frames"][0]["frame_id"]
    train = next(
        frame["frame_id"] for frame in dataset["manifest"]["frames"] if frame["split"] == "train"
    )
    for ids in ([selected, selected], [train]):
        with pytest.raises(ValueError):
            create(saved, example_frame_ids=ids)
    assert not store.list("experiment_reports")


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled", "interrupted"])
def test_incomplete_evaluation_cannot_be_reported(saved, status):
    store, detail, _, _ = saved
    store.update("jobs", detail["job_id"], {"status": status})
    with pytest.raises(ValueError, match="finish successfully"):
        create(saved)
    assert not store.list("experiment_reports")


@pytest.mark.parametrize(
    "mutation", ["metric_nan", "metric_out_of_range", "prediction_mismatch", "timing_nan"]
)
def test_invalid_saved_evidence_fails_closed(saved, monkeypatch, mutation):
    detail = deepcopy(saved[1])
    if mutation == "metric_nan":
        detail["models"][0]["metrics"]["summary"]["map"] = float("nan")
    elif mutation == "metric_out_of_range":
        detail["models"][0]["metrics"]["per_class"][0]["ap"] = 2
    elif mutation == "prediction_mismatch":
        detail["predictions"][0]["input_size"] = [1, 1]
    else:
        detail["predictions"][0]["timing"]["total_ms"] = float("nan")
    monkeypatch.setattr(experiments, "evaluation_detail", lambda *_: deepcopy(detail))
    with pytest.raises(ValueError):
        create(saved)
    assert not saved[0].list("experiment_reports")


def test_private_nested_metadata_and_annotation_notes_are_omitted(saved, monkeypatch):
    detail = deepcopy(saved[1])
    secret = "/home/private/API_SECRET_DO_NOT_EXPORT"
    detail["config"]["secret_path"] = secret
    for frame in detail["frames"]:
        frame["source"]["metadata"]["private"] = {"raw_response": secret}
        frame["source"]["filename"] = "/private/nested/fixture.png"
        frame["annotation"]["notes"] = secret
        frame["annotation"]["reviewer"] = secret
    for model in detail["models"]:
        model["metadata"].update(
            path=secret, raw_response={"private": secret}, prompt=secret, api_key=secret
        )
        model["metadata"]["input_transform"] = {"color": "RGB", "private_path": secret}
    monkeypatch.setattr(experiments, "evaluation_detail", lambda *_: deepcopy(detail))
    report = create(saved, example_frame_ids=[detail["frames"][1]["frame_id"]])
    assert secret not in json.dumps(report)
    assert "/private/nested" not in json.dumps(report)
    assert report["snapshot"]["examples"][0]["source"]["filename"] == "fixture.png"


def test_unexpected_nested_values_in_allowlisted_scalars_are_rejected(saved, monkeypatch):
    detail = deepcopy(saved[1])
    detail["models"][0]["metadata"]["hardware"] = {"private_path": "do-not-export"}
    monkeypatch.setattr(experiments, "evaluation_detail", lambda *_: detail)
    with pytest.raises(ValueError, match="unexpected nested"):
        create(saved)


def test_safe_dataset_attribution_is_snapshotted_without_original_annotations(saved, monkeypatch):
    detail = deepcopy(saved[1])
    credit = {
        "source_url": "https://example.org/public",
        "license_name": "Fixture license",
        "attribution": "Synthetic author",
    }
    for frame in detail["frames"]:
        frame["source"]["metadata"]["dataset_import"] = {
            **credit,
            "original_annotations": ["private"],
            "archive_filename": "/private/archive.zip",
        }
    monkeypatch.setattr(experiments, "evaluation_detail", lambda *_: deepcopy(detail))
    report = create(saved, example_frame_ids=[detail["frames"][0]["frame_id"]])
    assert report["snapshot"]["dataset"]["sources"] == [credit]
    assert report["snapshot"]["examples"][0]["source"]["attribution"] == credit
    assert "original_annotations" not in json.dumps(report)
    assert "/private/archive" not in json.dumps(report)


def test_reference_decisions_are_historical_and_do_not_follow_later_changes(saved):
    store, detail, _, _ = saved
    original = store.insert(
        "model_references",
        {
            "id": new_id(),
            "evaluation_id": detail["id"],
            "model_id": MODELS[0],
            "reviewer": "Fixture reviewer",
            "notes": "Human fixture decision",
            "metadata": {
                "variant": "full",
                "evaluation_model_id": detail["models"][0]["id"],
                "model_name": "Saved model",
                "secret": "do-not-export",
            },
            "created_at": now(),
        },
    )
    report = create(saved)
    historical = report["snapshot"]["reference_decisions"]
    assert historical["historical"] is True and historical["current_reference_id"] == original["id"]
    store.insert(
        "model_references", {**original, "id": new_id(), "model_id": MODELS[1], "created_at": now()}
    )
    assert experiment_detail(store, report["id"])["snapshot"]["reference_decisions"] == historical
    assert "do-not-export" not in json.dumps(report)


def test_editorial_updates_do_not_change_frozen_snapshot_or_images(saved):
    store, detail, _, _ = saved
    report = create(saved, example_frame_ids=[detail["frames"][0]["frame_id"]])
    edited = update_experiment(
        store,
        report["id"],
        expected_revision=1,
        title="Updated title",
        objective="New question",
        conclusion="Human interpretation",
    )
    assert edited["revision"] == 2
    assert edited["snapshot"] == report["snapshot"]
    assert edited["snapshot_sha256"] == report["snapshot_sha256"]
    assert edited["images"] == report["images"]
    with pytest.raises(ExperimentConflict):
        update_experiment(
            store, report["id"], expected_revision=1, title="Stale", objective="", conclusion=""
        )
    assert experiment_detail(store, report["id"])["title"] == "Updated title"


def test_concurrent_edit_accepts_only_one_revision(saved):
    store = saved[0]
    report = create(saved)
    barrier = Barrier(2)

    def update(title):
        barrier.wait()
        try:
            return update_experiment(
                store, report["id"], expected_revision=1, title=title, objective="", conclusion=""
            )
        except ExperimentConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, ["First", "Second"]))
    assert sum(value is not None for value in results) == 1
    assert experiment_detail(store, report["id"])["revision"] == 2


@pytest.mark.parametrize("mutation", ["snapshot", "sha", "image_path", "image_id", "image_size"])
def test_persisted_tampering_is_detected_without_reading_outside_paths(saved, mutation):
    store, detail, _, _ = saved
    report = create(saved, example_frame_ids=[detail["frames"][0]["frame_id"]])
    raw = store.get("experiment_reports", report["id"])
    if mutation == "snapshot":
        raw["snapshot"]["lanes"][0]["metrics"]["summary"]["map"] = 0.123
    elif mutation == "sha":
        raw["snapshot_sha256"] = "a" * 64
    elif mutation == "image_path":
        raw["images"][0]["path"] = "../outside.jpg"
    elif mutation == "image_id":
        raw["images"][0]["frame_id"] = "other"
    else:
        raw["images"][0]["size_bytes"] = 10_000_000
    store.update(
        "experiment_reports",
        report["id"],
        {key: raw[key] for key in ("snapshot", "snapshot_sha256", "images")},
    )
    with pytest.raises(ValueError):
        experiment_detail(store, report["id"])


@pytest.mark.parametrize("mutation", ["missing", "bytes"])
def test_copied_jpeg_integrity_is_checked_on_read(saved, mutation):
    store, detail, _, _ = saved
    frame_id = detail["frames"][0]["frame_id"]
    report = create(saved, example_frame_ids=[frame_id])
    raw = store.get("experiment_reports", report["id"])
    path = store.artifact_path(raw["images"][0]["path"])
    if mutation == "missing":
        path.unlink()
    else:
        content = path.read_bytes()
        path.write_bytes(content[:-1] + b"x")
    with pytest.raises(ValueError):
        read_experiment_image(store, report["id"], frame_id)


def test_failed_second_image_cleans_first_copy_and_database_record(saved):
    store, detail, dataset, _ = saved
    chosen = [frame["frame_id"] for frame in detail["frames"]]
    missing = next(
        frame for frame in dataset["manifest"]["frames"] if frame["frame_id"] == chosen[1]
    )
    store.artifact_path(missing["image_path"]).unlink()
    with pytest.raises((ValueError, OSError)):
        create(saved, example_frame_ids=chosen)
    assert not store.list("experiment_reports") and not list(store.root.glob("reports/*"))


def test_reports_directory_cannot_escape_workspace_via_symlink(saved, tmp_path):
    store, detail, _, _ = saved
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "reports").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        create(saved, example_frame_ids=[detail["frames"][0]["frame_id"]])
    assert not list(outside.iterdir()) and not store.list("experiment_reports")


def test_changed_evaluation_during_image_copy_aborts_cleanly(saved, monkeypatch):
    store, detail, _, _ = saved
    original = experiments._read_training_image

    def changed(*args):
        image = original(*args)
        store.update("evaluations", detail["id"], {"name": "Changed title during copy"})
        return image

    monkeypatch.setattr(experiments, "_read_training_image", changed)
    with pytest.raises(ValueError, match="changed while"):
        create(saved, example_frame_ids=[detail["frames"][0]["frame_id"]])
    assert not store.list("experiment_reports") and not list(store.root.glob("reports/*"))


def test_missing_id_errors_are_explicit(saved):
    store = saved[0]
    for function in (preview_experiment, experiment_detail):
        with pytest.raises(KeyError):
            function(store, "unknown")
    report = create(saved)
    with pytest.raises(KeyError):
        read_experiment_image(store, report["id"], "not-selected")


def test_schema_11_upgrade_preserves_all_existing_rows(saved):
    store = saved[0]
    before = {table: store.list(table) for table in store.columns if table != "experiment_reports"}
    with store.connect() as conn:
        conn.execute("DROP TABLE experiment_reports")
        conn.execute("PRAGMA user_version=11")
    reopened = Store(store.root)
    assert {
        table: reopened.list(table) for table in reopened.columns if table != "experiment_reports"
    } == before
    with reopened.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 12
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    assert not list_experiments(reopened)


def _record_training(
    store, dataset, model_id, digest, parent_id="fixture-official", parent_hash="f" * 64
):
    job_id, training_id = new_id(), new_id()
    config = {
        "scope": "full_model",
        "scope_version": 1,
        "trainable_modules": ["backbone", "rpn", "roi_heads"],
        "steps": 2,
        "learning_rate": 0.001,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "parent_weight_sha256": parent_hash,
        "raw_prompt": "PRIVATE_PROMPT",
    }
    metadata = {
        "config": config,
        "trainable_parameters": 10,
        "frozen_parameters": 0,
        "total_parameters": 10,
        "head_weights_changed": True,
        "frozen_parameters_unchanged": True,
        "frozen_batchnorm_buffers_unchanged": True,
        "private_path": "PRIVATE_PATH",
    }
    store.insert(
        "jobs",
        {
            "id": job_id,
            "kind": "train",
            "status": "succeeded",
            "params": {"training_id": training_id},
            "created_at": now(),
        },
    )
    store.insert(
        "training_runs",
        {
            "id": training_id,
            "name": "Synthetic trained fixture",
            "dataset_id": dataset["id"],
            "parent_model_id": parent_id,
            "config": config,
            "metadata": metadata,
            "history": [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.5}],
            "checkpoint_id": model_id,
            "job_id": job_id,
            "created_at": now(),
        },
    )
    store.insert(
        "trained_models",
        {
            "id": model_id,
            "name": "Synthetic model",
            "training_id": training_id,
            "parent_model_id": parent_id,
            "architecture": "fixture",
            "path": "PRIVATE_CHECKPOINT_PATH",
            "weight_sha256": digest,
            "metadata": metadata,
            "created_at": now(),
        },
    )
    return training_id


def _replace_lineage(store, detail, model_id, lineage):
    config = deepcopy(detail["config"])
    config["model_lineages"][model_id] = lineage
    store.update("evaluations", detail["id"], {"config": config})
    for row in detail["models"]:
        if row["model_id"] == model_id:
            store.update(
                "evaluation_models",
                row["id"],
                {"metadata": {**row["metadata"], "lineage": lineage}},
            )


def test_trained_run_configuration_and_counts_are_copied_without_private_provenance(saved):
    store, detail, dataset, _ = saved
    model_id = MODELS[0]
    identifier = _record_training(
        store, dataset, model_id, detail["config"]["model_hashes"][model_id]
    )
    _replace_lineage(
        store,
        detail,
        model_id,
        [
            {
                "model_id": model_id,
                "origin": "trained",
                "parent_model_id": "fixture-official",
                "parent_weight_sha256": "f" * 64,
                "training_scene_groups": ["train"],
                "training_frame_hashes": ["a" * 64],
            },
            {"model_id": "fixture-official", "origin": "official"},
        ],
    )
    report = create(saved)
    lane = report["snapshot"]["lanes"][0]
    assert lane["training_status"] == "recorded"
    assert lane["training"]["id"] == identifier
    assert lane["training"]["config"]["scope"] == "full_model"
    assert lane["training"]["metadata"]["frozen_parameters"] == 0
    assert lane["training"]["history_summary"] == {
        "steps_completed": 2,
        "first_loss": 2.0,
        "last_loss": 1.5,
    }
    assert lane["lineage"][0]["training_frame_count"] == 1
    assert lane["lineage"][0]["training"] == lane["training"]
    assert "PRIVATE_" not in json.dumps(report)


def test_known_official_origin_is_pretrained_without_claiming_unknown_training(saved):
    store, detail, _, _ = saved
    _replace_lineage(store, detail, MODELS[0], [{"model_id": MODELS[0], "origin": "official"}])
    report = create(saved)
    assert report["snapshot"]["lanes"][0]["training_status"] == "pretrained"
    assert report["snapshot"]["lanes"][0]["training"] is None
    assert report["snapshot"]["lanes"][1]["training_status"] == "unavailable"


def test_missing_trained_record_is_explicitly_unavailable(saved):
    store, detail, _, _ = saved
    _replace_lineage(
        store,
        detail,
        MODELS[0],
        [
            {"model_id": MODELS[0], "origin": "trained", "parent_model_id": "fixture-official"},
            {"model_id": "fixture-official", "origin": "official"},
        ],
    )
    report = create(saved)
    assert report["snapshot"]["lanes"][0]["training_status"] == "unavailable"
    assert report["snapshot"]["lanes"][0]["training"] is None


def test_changed_ancestor_checkpoint_hash_is_not_attached_as_recorded_lineage(saved):
    store, detail, dataset, _ = saved
    _record_training(store, dataset, "ancestor", "d" * 64)
    _replace_lineage(
        store,
        detail,
        MODELS[0],
        [
            {
                "model_id": MODELS[0],
                "origin": "trained",
                "parent_model_id": "ancestor",
                "parent_weight_sha256": "e" * 64,
            },
            {
                "model_id": "ancestor",
                "origin": "trained",
                "parent_model_id": "fixture-official",
                "parent_weight_sha256": "f" * 64,
            },
            {"model_id": "fixture-official", "origin": "official"},
        ],
    )
    with pytest.raises(ValueError, match="lineage.*hash"):
        create(saved)
    assert not store.list("experiment_reports")


def test_changed_prediction_scores_with_identical_counts_abort_creation(saved, monkeypatch):
    store, detail, _, _ = saved
    source = next(row for row in detail["predictions"] if row["model_id"] == MODELS[0])
    original = experiments._read_training_image

    def mutate(*args):
        image = original(*args)
        detections = deepcopy(source["detections"])
        detections[0]["score"] = 0.88
        store.update("evaluation_predictions", source["id"], {"detections": detections})
        return image

    monkeypatch.setattr(experiments, "_read_training_image", mutate)
    with pytest.raises(ValueError, match="changed while"):
        create(saved, example_frame_ids=[detail["frames"][0]["frame_id"]])
    assert not store.list("experiment_reports") and not list(store.root.glob("reports/*"))


def test_post_commit_response_failure_preserves_registered_report_images(saved, monkeypatch):
    store, detail, _, _ = saved

    def broken(*_):
        raise RuntimeError("Synthetic response failure after commit")

    monkeypatch.setattr(experiments, "experiment_detail", broken)
    with pytest.raises(RuntimeError, match="after commit"):
        create(saved, example_frame_ids=[detail["frames"][0]["frame_id"]])
    (record,) = store.list("experiment_reports")
    assert store.artifact_path(record["images"][0]["path"]).is_file()


@pytest.mark.parametrize("invalid", [None, [], "invalid", 4])
def test_malformed_image_metadata_is_a_value_error(saved, invalid):
    store, detail, _, _ = saved
    report = create(saved, example_frame_ids=[detail["frames"][0]["frame_id"]])
    store.update("experiment_reports", report["id"], {"images": [invalid]})
    with pytest.raises(ValueError, match="metadata.*objects"):
        experiment_detail(store, report["id"])


def test_export_jpeg_strips_inherited_image_metadata(saved, monkeypatch):
    from PIL import Image

    store, detail, _, _ = saved
    original = experiments._read_training_image

    def with_metadata(*args):
        image = original(*args)
        exif = Image.Exif()
        exif[315] = "PRIVATE_EXIF_AUTHOR"
        image.info.update(
            exif=exif.tobytes(), comment=b"PRIVATE_COMMENT", icc_profile=b"PRIVATE_PROFILE"
        )
        return image

    monkeypatch.setattr(experiments, "_read_training_image", with_metadata)
    selected = detail["frames"][0]["frame_id"]
    report = create(saved, example_frame_ids=[selected])
    content = read_experiment_image(store, report["id"], selected)
    assert b"PRIVATE_" not in content


def test_metric_timing_mean_requires_every_frame_to_be_measured(saved):
    store, detail, _, _ = saved
    for model in detail["models"]:
        rows = [row for row in detail["predictions"] if row["evaluation_model_id"] == model["id"]]
        store.update("evaluation_predictions", rows[0]["id"], {"timing": {"total_ms": 5.0}})
    report = create(saved)
    assert report["snapshot"]["lanes"][0]["timing"]["measured_frame_count"] == 1
    assert report["snapshot"]["lanes"][0]["timing"]["mean_total_ms"] is None
