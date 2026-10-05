"""Bounded report selection and exact links to saved, simulated export evidence."""

from copy import deepcopy

import pytest
from test_experiment_deployments import evidence as evidence
from test_experiment_deployments import saved_measurement
from test_model_exports import fixture_workspace

from iris import experiment_deployments as deployments
from iris import experiments
from iris import export_runner as runner
from iris import model_exports as exports
from iris.store import now


def prepared(workspace):
    snapshot, _, detail = experiments._prepare(workspace[0], workspace[2]["id"], now())
    return snapshot, detail


def coherent_evidence(export, item, change):
    """Rehash a simulated declaration so rejection must concern its evaluation link."""
    export, item = deepcopy(export), deepcopy(item)
    change(export["config"])
    manifest = exports._manifest(export["config"], export["id"], export["created_at"])
    export.update(manifest=manifest, manifest_sha256=exports._digest(manifest))
    payload = item["payload"]
    payload["manifest_sha256"] = export["manifest_sha256"]
    frames = {frame["frame_id"]: frame for frame in export["config"]["reference"]["frames"]}
    for sample in payload["samples"]:
        frame = frames[sample["frame_id"]]
        sample["input_size"] = deepcopy(frame["input_size"])
        sample["detections"] = deepcopy(frame["detections"])
    item["fingerprint"] = exports._digest(payload)
    item["summary"] = runner.validate_measurement(manifest, export["config"]["reference"], payload)
    assert item["summary"]["parity_passed"]
    return export, item


def no_read(*_args, **_kwargs):
    pytest.fail("An empty or explicitly selected measurement list must not scan saved evidence")


def test_empty_selection_does_not_read_the_database(evidence, monkeypatch):
    workspace, _, _ = evidence
    store = workspace[0]
    for name in ("get", "list", "connect"):
        monkeypatch.setattr(store, name, no_read)
    assert deployments.available_measurements(store, {}, {}, selected_ids=[]) == []


def test_explicit_selection_reads_only_the_selected_ids_in_requested_order(evidence, monkeypatch):
    workspace, first_export, first = evidence
    second_export, second = saved_measurement(workspace, target="cuda")
    snapshot, detail = prepared(workspace)
    store = workspace[0]
    original_get = store.get
    reads = []

    def tracked_get(table, identifier):
        reads.append((table, identifier))
        return original_get(table, identifier)

    monkeypatch.setattr(store, "list", no_read)
    monkeypatch.setattr(store, "get", tracked_get)
    rows = deployments.available_measurements(
        store, snapshot, detail, selected_ids=[second["id"], first["id"]]
    )
    assert [row["id"] for row in rows] == [second["id"], first["id"]]
    assert reads == [
        ("model_export_measurements", second["id"]),
        ("model_exports", second_export["id"]),
        ("model_export_measurements", first["id"]),
        ("model_exports", first_export["id"]),
    ]


def test_preview_cap_limits_payload_reads_before_decoding(evidence, monkeypatch):
    workspace, _, first = evidence
    _, second = saved_measurement(workspace, target="cuda")
    snapshot, detail = prepared(workspace)
    store = workspace[0]
    newest, older = sorted(
        (first, second), key=lambda item: (item["created_at"], item["id"]), reverse=True
    )
    # The excluded payload is deliberately undecodable. A post-decode list slice fails this test.
    with store.connect() as connection:
        connection.execute(
            "UPDATE model_export_measurements SET payload=? WHERE id=?",
            ("not valid JSON", older["id"]),
        )
    assert deployments.MAX_AVAILABLE_MEASUREMENTS == 100
    monkeypatch.setattr(deployments, "MAX_AVAILABLE_MEASUREMENTS", 1)
    monkeypatch.setattr(store, "list", no_read)
    original_get = store.get
    reads = []

    def tracked_get(table, identifier):
        reads.append((table, identifier))
        return original_get(table, identifier)

    monkeypatch.setattr(store, "get", tracked_get)
    rows = deployments.available_measurements(store, snapshot, detail)
    assert [row["id"] for row in rows] == [newest["id"]]
    assert reads == [
        ("model_export_measurements", newest["id"]),
        ("model_exports", newest["export_id"]),
    ]


@pytest.mark.parametrize("recorded_device,claimed_device", [("cpu", "cuda:0"), ("cuda:0", "cpu")])
def test_reference_device_must_match_the_saved_evaluation(
    evidence, recorded_device, claimed_device
):
    workspace, export, item = evidence
    snapshot, detail = prepared(workspace)
    metadata = detail["models"][0]["metadata"]
    metadata["device"] = recorded_device

    def change(config):
        config["format"] = "iris-model-export-plan-v2"
        config["profile"] = runner.native_profile("cpu")
        config["source"]["reference_device"] = claimed_device
        config["evaluation_metadata_sha256"] = exports._digest(metadata)

    export, item = coherent_evidence(export, item, change)
    with pytest.raises(ValueError, match="evaluated checkpoint and lane"):
        deployments._measurement(export, item, snapshot, detail)


def test_legacy_cpu_reference_device_default_is_preserved(evidence):
    workspace, export, item = evidence
    snapshot, detail = prepared(workspace)
    assert "reference_device" not in export["manifest"]["source"]
    result = deployments._measurement(export, item, snapshot, detail)
    assert result["source"]["reference_device"] == "cpu"


def test_saved_runtime_metadata_drift_rejects_an_otherwise_valid_measurement(evidence):
    workspace, export, item = evidence
    snapshot, detail = prepared(workspace)
    detail["models"][0]["metadata"]["hardware"] = "Different recorded hardware"
    assert runner.validate_measurement(
        export["manifest"], export["config"]["reference"], item["payload"]
    )
    with pytest.raises(ValueError, match="evaluated checkpoint and lane"):
        deployments._measurement(export, item, snapshot, detail)


def test_architecture_must_match_the_evaluated_lane_even_with_consistent_export_profile(evidence):
    workspace, export, item = evidence
    snapshot, detail = prepared(workspace)

    def change(config):
        config["format"] = "iris-model-export-plan-v3"
        config["profile"] = runner.ssdlite_profile("cpu")
        config["model"]["architecture"] = runner.SSDLITE_ARCHITECTURE
        config["source"]["reference_device"] = "cpu"

    export, item = coherent_evidence(export, item, change)
    with pytest.raises(ValueError, match="evaluated checkpoint and lane"):
        deployments._measurement(export, item, snapshot, detail)


def test_same_class_ids_with_different_frozen_definitions_are_not_the_same_contract(tmp_path):
    workspace = fixture_workspace(tmp_path, custom=True)
    export, item = saved_measurement(workspace)
    snapshot, detail = prepared(workspace)

    def change(config):
        config["model"]["class_contract"]["taxonomy"]["classes"][0]["definition"] = (
            "A different class definition with the same numeric slot."
        )

    export, item = coherent_evidence(export, item, change)
    with pytest.raises(ValueError, match="evaluated checkpoint and lane"):
        deployments._measurement(export, item, snapshot, detail)


@pytest.mark.parametrize("field", ["image_hash", "detections", "input_size"])
def test_consistent_reference_and_payload_cannot_replace_the_original_saved_evidence(
    evidence, field
):
    workspace, export, item = evidence
    snapshot, detail = prepared(workspace)

    def change(config):
        frame = config["reference"]["frames"][0]
        if field == "image_hash":
            frame["sha256"] = "0" * 64
            next(item for item in config["frames"] if item["frame_id"] == frame["frame_id"])[
                "image_file_sha256"
            ] = frame["sha256"]
        elif field == "detections":
            frame["detections"][0]["score"] -= 0.05
        else:
            frame["input_size"][0] += 1

    export, item = coherent_evidence(export, item, change)
    with pytest.raises(ValueError, match="reference differs from the saved evaluated images"):
        deployments._measurement(export, item, snapshot, detail)


def test_invalid_link_is_hidden_from_preview_and_rejected_when_explicitly_selected(evidence):
    workspace, export, item = evidence
    store = workspace[0]
    snapshot, detail = prepared(workspace)

    def change(config):
        config["evaluation_metadata_sha256"] = "0" * 64

    export, item = coherent_evidence(export, item, change)
    store.update("model_exports", export["id"], {"config": export["config"]})
    assert deployments.available_measurements(store, snapshot, detail) == []
    with pytest.raises(ValueError, match="Selected standalone measurement"):
        deployments.available_measurements(store, snapshot, detail, selected_ids=[item["id"]])
