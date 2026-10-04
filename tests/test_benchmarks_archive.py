"""Archive benchmark reference/candidate separation without running a provider."""

import hashlib
import json
import sqlite3
import zipfile
from copy import deepcopy

import pytest
import test_workspace_archive as archive_fixtures
import test_workspace_restore as restore_fixtures

from iris.annotations import adopt_taxonomy, save_annotation
from iris.benchmark import SCORING
from iris.dataset_manifest import taxonomy_mappings
from iris.preannotation_contracts import build_contract
from iris.projects import create_project
from iris.store import DEFAULT_PROJECT_ID, SCHEMA_VERSION, TABLES, Store, new_id, now
from iris.taxonomies import TAXONOMY, publish_taxonomy
from iris.workspace_archive import ArchiveError, create_archive, preview_workspace
from iris.workspace_restore import inspect_archive, restore_archive


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def records(store):
    with store.connect() as connection:
        return {
            table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY id")]
            for table in TABLES
        }


@pytest.fixture(params=["builtin", "custom"])
def benchmark_workspace(tmp_path, request):
    store = archive_fixtures.workspace.__wrapped__(tmp_path)
    taxonomy = deepcopy(TAXONOMY)
    if request.param == "custom":
        taxonomy = publish_taxonomy(
            store,
            DEFAULT_PROJECT_ID,
            expected_taxonomy_id=TAXONOMY["id"],
            classes=[
                {"id": "helmet", "name": "Helmet", "definition": "A helmet."},
                {"id": "vehicle", "name": "Vehicle", "definition": "A car.", "coco_id": 3},
            ],
        )
    identifier = new_id()
    frames = []
    for index, frame in enumerate(store.list("frames")):
        session = store.get("sessions", frame["session_id"])
        asset = store.get("assets", frame["asset_id"])
        revision = 1
        if request.param == "custom":
            adopted = adopt_taxonomy(
                store,
                frame["id"],
                expected_revision=1,
                expected_taxonomy_id=TAXONOMY["id"],
                target_taxonomy_id=taxonomy["id"],
            )
            revision = adopted["revision"]
        save_annotation(
            store,
            frame["id"],
            expected_revision=revision,
            boxes=[
                {
                    "id": "human-box",
                    "label": taxonomy["classes"][0]["id"],
                    "box": [1, 2, 8, 10],
                }
            ]
            if index == 0
            else [],
            decisions={},
            status="validated",
            reviewer="Independent human",
        )
        annotation = max(
            store.list("annotation_revisions", frame_id=frame["id"]),
            key=lambda item: item["revision"],
        )
        path = f"benchmarks/{identifier}/images/{frame['id']}.png"
        image = archive_fixtures._put(store, path, store.artifact_path(frame["path"]).read_bytes())
        frames.append(
            {
                "frame_id": frame["id"],
                "session_id": session["id"],
                "session_name": session["name"],
                "scene_group": session["scene_group"],
                "role": "tuning" if index == 0 else "evaluation",
                "sha256": frame["sha256"],
                "width": frame["width"],
                "height": frame["height"],
                "perceptual_hash": frame["perceptual_hash"],
                "image_path": path,
                "image_file_sha256": image["sha256"],
                "annotation_revision_id": annotation["id"],
                "revision": annotation["revision"],
                "boxes": deepcopy(annotation["boxes"]),
                "annotation": annotation,
                "source": {
                    "asset_id": asset["id"],
                    "filename": asset["filename"],
                    "kind": asset["kind"],
                    "sha256": asset["sha256"],
                    "metadata": asset["metadata"],
                    "frame_index": frame["frame_index"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "extraction": frame["extraction"],
                },
            }
        )
    class_mapping, output_mapping = taxonomy_mappings(taxonomy)
    manifest = {
        "schema_version": 1,
        "protocol": "iris-preannotation-benchmark-v1",
        "id": identifier,
        "project_id": DEFAULT_PROJECT_ID,
        "name": "Independent reference fixture",
        "created_at": now(),
        "taxonomy": deepcopy(taxonomy),
        "class_mapping": class_mapping,
        "output_mapping": output_mapping,
        "reference": {
            "reviewer": "Independent human",
            "independent_reference": True,
            "independence_notes": "Drawn without any candidate output.",
        },
        "roles": {frame["scene_group"]: frame["role"] for frame in frames},
        "frames": frames,
    }
    document = archive_fixtures._put(
        store, f"benchmarks/{identifier}/manifest.json", canonical(manifest)
    )
    benchmark = store.insert(
        "benchmarks",
        {
            "id": identifier,
            "name": manifest["name"],
            "project_id": DEFAULT_PROJECT_ID,
            "path": document["path"],
            "manifest_sha256": document["sha256"],
            "summary": {"frames": 2},
            "status": "locked",
            "locked_at": now(),
            "created_at": now(),
        },
    )
    model_id = "ssdlite320_mobilenet_v3_large"
    config_payload = {
        "protocol": "iris-preannotation-benchmark-v1",
        "approach": "local_detector",
        "model_id": model_id,
        "model_name": "Fixture detector",
        "weight_sha256": "a" * 64,
        "proposal_contract": build_contract(store, model_id, taxonomy),
        "threshold": 0.5,
        "device": "cpu",
        "inference": {"mode": "full"},
        "scoring": SCORING,
        "reference_manifest_sha256": document["sha256"],
        "lineage": {},
    }
    config = store.insert(
        "benchmark_configs",
        {
            "id": new_id(),
            "benchmark_id": identifier,
            "name": "Fixture detector configuration",
            "approach": "local_detector",
            "config": config_payload,
            "fingerprint": digest(canonical(config_payload)),
            "created_at": now(),
        },
    )
    trial_id = new_id()
    job = store.insert(
        "jobs",
        {
            "id": new_id(),
            "kind": "benchmark",
            "status": "succeeded",
            "params": {"trial_id": trial_id},
            "created_at": now(),
        },
    )
    trial = store.insert(
        "benchmark_trials",
        {
            "id": trial_id,
            "benchmark_id": identifier,
            "config_id": config["id"],
            "split": "evaluation",
            "config": {
                "protocol": "iris-preannotation-benchmark-v1",
                "fingerprint": "b" * 64,
                "source_config_fingerprint": config["fingerprint"],
                "benchmark_manifest_sha256": document["sha256"],
                "role": "evaluation",
                "frame_ids": [frames[1]["frame_id"]],
                "candidate_config": config_payload,
                "work": {},
                "warnings": [],
            },
            "job_id": job["id"],
            "created_at": now(),
        },
    )
    output = store.insert(
        "benchmark_outputs",
        {
            "id": new_id(),
            "trial_id": trial_id,
            "frame_id": frames[1]["frame_id"],
            "raw_response": {"fixture": "raw independent candidate", "boxes": []},
            "result": {"proposals": []},
            "metadata": {"reference_sent": False},
            "created_at": now(),
        },
    )
    correction = store.insert(
        "benchmark_corrections",
        {
            "id": new_id(),
            "output_id": output["id"],
            "revision": 1,
            "status": "reviewed",
            "boxes": [],
            "decisions": {},
            "reviewer": "Correction reviewer",
            "notes": "Human negative",
            "timing": {"elapsed_ms": 1534.25, "complete": True},
            "created_at": now(),
        },
    )
    timer = store.insert(
        "benchmark_timers",
        {
            "id": new_id(),
            "output_id": output["id"],
            "reviewer": "Correction reviewer",
            "state": "paused",
            "revision": 2,
            "elapsed_ms": 1534.25,
            "segments": [{"elapsed_ms": 1534.25}],
            "metadata": {"stop_reason": "reviewed"},
            "created_at": now(),
            "updated_at": now(),
        },
    )
    return store, benchmark, manifest, config, trial, output, correction, timer


def rewrite_manifest(store, benchmark, manifest):
    raw = canonical(manifest)
    store.artifact_path(benchmark["path"]).write_bytes(raw)
    store.update("benchmarks", benchmark["id"], {"manifest_sha256": digest(raw)})


@pytest.mark.parametrize("timer_state", ["paused", "running"])
def test_benchmark_archive_retains_reference_raw_corrections_and_measured_timing(
    benchmark_workspace, tmp_path, monkeypatch, timer_state
):
    store, benchmark, manifest, *_, timer = benchmark_workspace
    store.update("benchmark_timers", timer["id"], {"state": timer_state})
    frozen = store.artifact_path(benchmark["path"]).read_bytes()
    original_reference = deepcopy(manifest["frames"])
    # Later human edits and class publications must not rewrite the benchmark reference.
    frame = manifest["frames"][0]
    save_annotation(
        store,
        frame["frame_id"],
        expected_revision=frame["revision"],
        boxes=[],
        decisions={},
        status="draft",
        reviewer="Later editor",
    )
    publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=manifest["taxonomy"]["id"],
        classes=[
            {**item, "definition": "Changed later definition."}
            for item in manifest["taxonomy"]["classes"]
        ],
    )
    expected = records(store)
    saved = create_archive(store.root, tmp_path / "benchmark.zip")
    assert saved["manifest"]["schema_version"] == SCHEMA_VERSION
    assert saved["manifest"]["counts"]["benchmark_corrections"] == 1

    def forbidden(*_args, **_kwargs):
        pytest.fail("Restoring a benchmark must not initialize or migrate Store")

    target = tmp_path / "restored"
    with monkeypatch.context() as context:
        context.setattr(Store, "__init__", forbidden)
        checked = inspect_archive(saved["path"])
        restore_archive(saved["path"], target, expected_archive_sha256=checked["archive_sha256"])
    assert (target / benchmark["path"]).read_bytes() == frozen
    assert json.loads(frozen)["frames"] == original_reference
    with zipfile.ZipFile(saved["path"]) as archive:
        for item in saved["manifest"]["files"]:
            assert (target / item["path"]).read_bytes() == archive.read(item["path"])
    assert records(Store(target)) == expected
    assert Store(target).get("benchmark_timers", timer["id"])["state"] == timer_state


@pytest.mark.parametrize(
    "damage",
    [
        "manifest_hash",
        "image_hash",
        "manifest_id",
        "foreign_project",
        "class_mapping",
        "taxonomy_definition",
        "image_path",
        "reference_revision",
        "frozen_boxes",
        "roles",
        "trial_config_owner",
        "config_hash",
        "config_contract",
        "trial_candidate_config",
        "trial_frame_ids",
        "timer_elapsed",
        "job_owner",
        "output_partition",
    ],
)
def test_benchmark_archive_rejects_broken_frozen_reference_or_trial_relations(
    benchmark_workspace, tmp_path, damage
):
    store, benchmark, manifest, config, trial, output, *_ = benchmark_workspace
    if damage == "manifest_hash":
        store.artifact_path(benchmark["path"]).write_bytes(b"{}")
    elif damage == "image_hash":
        store.artifact_path(manifest["frames"][0]["image_path"]).write_bytes(
            b"changed copied image"
        )
    elif damage == "trial_config_owner":
        other_id = new_id()
        other_manifest = deepcopy(manifest)
        other_manifest["id"] = other_id
        for frame in other_manifest["frames"]:
            old_path = frame["image_path"]
            frame["image_path"] = f"benchmarks/{other_id}/images/{frame['frame_id']}.png"
            archive_fixtures._put(
                store, frame["image_path"], store.artifact_path(old_path).read_bytes()
            )
        document = archive_fixtures._put(
            store, f"benchmarks/{other_id}/manifest.json", canonical(other_manifest)
        )
        store.insert(
            "benchmarks",
            {
                **benchmark,
                "id": other_id,
                "path": document["path"],
                "manifest_sha256": document["sha256"],
            },
        )
        other_config = {**config["config"], "reference_manifest_sha256": document["sha256"]}
        store.update(
            "benchmark_configs",
            config["id"],
            {
                "benchmark_id": other_id,
                "config": other_config,
                "fingerprint": digest(canonical(other_config)),
            },
        )
    elif damage == "config_hash":
        store.update("benchmark_configs", config["id"], {"fingerprint": "f" * 64})
    elif damage == "config_contract":
        config["config"]["proposal_contract"]["label_mapping"] = {"99": "unknown"}
        store.update(
            "benchmark_configs",
            config["id"],
            {"config": config["config"], "fingerprint": digest(canonical(config["config"]))},
        )
    elif damage == "trial_candidate_config":
        trial["config"]["candidate_config"]["threshold"] = 0.01
        store.update("benchmark_trials", trial["id"], {"config": trial["config"]})
    elif damage == "trial_frame_ids":
        trial["config"]["frame_ids"] = [manifest["frames"][0]["frame_id"]]
        store.update("benchmark_trials", trial["id"], {"config": trial["config"]})
    elif damage == "timer_elapsed":
        with store.connect() as connection:
            connection.execute("UPDATE benchmark_timers SET elapsed_ms=?", (float("inf"),))
    elif damage == "job_owner":
        store.update("jobs", trial["job_id"], {"params": {"trial_id": "another-trial"}})
    elif damage == "output_partition":
        store.update(
            "benchmark_outputs", output["id"], {"frame_id": manifest["frames"][0]["frame_id"]}
        )
    else:
        if damage == "manifest_id":
            manifest["id"] = "different"
        elif damage == "foreign_project":
            other = create_project(store, name="Foreign owner")
            manifest["project_id"] = other["id"]
            store.update("benchmarks", benchmark["id"], {"project_id": other["id"]})
        elif damage == "class_mapping":
            manifest["class_mapping"][manifest["taxonomy"]["classes"][0]["id"]] = 99
        elif damage == "taxonomy_definition":
            manifest["taxonomy"]["classes"][0]["definition"] = "Different frozen meaning"
        elif damage == "image_path":
            manifest["frames"][0]["image_path"] = manifest["frames"][1]["image_path"]
        elif damage == "reference_revision":
            first, second = manifest["frames"]
            first["annotation_revision_id"] = second["annotation_revision_id"]
            first["annotation"] = second["annotation"]
        elif damage == "frozen_boxes":
            manifest["frames"][0]["boxes"] = [
                {"id": "fake", "label": "person", "box": [1, 1, 4, 4]}
            ]
        else:
            manifest["frames"][1]["scene_group"] = manifest["frames"][0]["scene_group"]
        rewrite_manifest(store, benchmark, manifest)
    with pytest.raises(ArchiveError):
        create_archive(store.root, tmp_path / "broken.zip")
    assert not (tmp_path / "broken.zip").exists()


def test_rehashed_archive_still_rejects_cross_partition_candidate_output(
    benchmark_workspace, tmp_path
):
    store, _, manifest, _, _, output, *_ = benchmark_workspace
    store.update("benchmark_outputs", output["id"], {"frame_id": manifest["frames"][0]["frame_id"]})
    # ZIP checksums and database foreign keys are valid: the frozen role is what fails.
    archive = restore_fixtures._write_archive(
        tmp_path / "rehashed.zip", restore_fixtures._payload(store)
    )
    with pytest.raises(ArchiveError, match="partition"):
        inspect_archive(archive)
    target = tmp_path / "not-restored"
    with pytest.raises(ArchiveError, match="partition"):
        restore_archive(archive, target, expected_archive_sha256=digest(archive.read_bytes()))
    assert not target.exists()


@pytest.mark.parametrize(
    "table,column,value",
    [
        ("benchmark_outputs", "raw_response", '{"confidence":NaN}'),
        ("benchmark_timers", "segments", '[{"elapsed_ms":Infinity}]'),
    ],
)
def test_benchmark_archive_rejects_nonfinite_saved_evidence(
    benchmark_workspace, table, column, value
):
    store, *_ = benchmark_workspace
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(f"UPDATE {table} SET {column}=?", (value,))
    assert not preview_workspace(store.root)["can_create"]
