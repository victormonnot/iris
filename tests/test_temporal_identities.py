"""Identity proposals remain uncertain evidence until explicit, atomic human review."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
from test_temporal_api import (
    publish_reference,
    publish_sequence,
    reference_payload,
    sequence_source,
)
from test_temporal_detection_api import client as client
from test_temporal_detection_api import create
from test_temporal_detections import execute
from test_tracking_comparisons import completed as completed
from test_tracking_comparisons import launch, run_worker
from test_tracking_replay import SyntheticTracker, forbidden, synthetic

from iris import temporal
from iris.store import Store
from iris.taxonomies import TAXONOMY, publish_taxonomy
from iris.temporal_contracts import REFERENCE_SCHEMA_V2
from iris.temporal_identities import save_identity_edits
from iris.workspace_archive import ArchiveError, _inventory, create_archive, validate_database
from iris.workspace_restore import inspect_archive, restore_archive


@pytest.fixture
def comparison(client, completed, monkeypatch):
    synthetic(monkeypatch)
    original = SyntheticTracker.update

    def observed_and_predicted(self, frame, **kwargs):
        result = original(self, frame, **kwargs)
        if self.position == 1:
            self.first_observation = deepcopy(result)
            for obj in result["observations"]:
                obj["estimated_box"] = [40, 30, 70, 55]
        if self.position == 2:
            first = self.first_observation
            obj = first["observations"][0]
            result["predictions"] = [
                {
                    **{key: obj[key] for key in ("track_id", "label_id", "label", "confirmed")},
                    "box": [40, 30, 70, 55],
                    "last_observed_frame_id": first["frame_id"],
                    "last_observed_frame_index": first["frame_index"],
                    "last_observed_timestamp_seconds": first["timestamp_seconds"],
                    "last_observed_update_index": first["update_index"],
                    "age_updates": 1,
                    "age_seconds": result["timestamp_seconds"] - first["timestamp_seconds"],
                }
            ]
        return result

    monkeypatch.setattr(SyntheticTracker, "update", observed_and_predicted)
    record = launch(client, completed)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    return {**record, "report": job["result"]}


def propose(client, comparison, **overrides):
    return client.post(
        f"/api/temporal/sequences/{comparison['sequence_id']}/identity-proposals",
        json={
            "comparison_id": comparison["id"],
            "lane_index": 0,
            "class_mapping": {"1": "person", "3": None},
            **overrides,
        },
    )


def edit(client, payload, *, revision=0, reviewer="Human editor", reviewed_frames=None, **extra):
    return client.post(
        f"/api/temporal/sequences/{payload['sequence_id']}/identity-edits",
        json={
            "expected_revision": revision,
            "payload": payload,
            "reviewer": reviewer,
            "reviewed_frames": reviewed_frames or [],
            **extra,
        },
    )


def test_proposal_is_read_only_measured_uncertain_and_ids_are_independent(
    client, comparison, monkeypatch
):
    store = client.app.state.store
    before = {table: store.list(table) for table in ("jobs", "temporal_detection_frames")}
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    response = propose(client, comparison)
    assert response.status_code == 200, response.text
    draft = response.json()
    payload = draft["payload"]
    assert payload["schema"] == REFERENCE_SCHEMA_V2
    assert payload["provenance"]["author"] == ""
    assert payload["identities"][0]["id"].startswith("ref_")
    assert payload["provenance"]["origin"]["track_mapping"] == {"3": payload["identities"][0]["id"]}
    assert payload["frames"][0]["objects"][0]["box"] == [3, 4, 21, 41]
    assert payload["frames"][1]["objects"] == []  # prediction is not an observation
    assert all(frame["coverage"] == "unreviewed" for frame in payload["frames"])
    assert all(
        obj["certainty"] == "uncertain" for frame in payload["frames"] for obj in frame["objects"]
    )
    assert draft["proposal_summary"] == {
        "seeded_objects": 2,
        "seeded_identities": 1,
        "skipped_observations": 0,
        "excluded_predictions": 1,
    }
    assert draft["summary"]["human_reviewed_frames"] == 0
    assert draft["summary"]["dense_human_reference"] is False
    second = propose(client, comparison, lane_index=1).json()["payload"]
    assert second["identities"] != payload["identities"]
    assert store.list("temporal_references") == []
    assert {table: store.list(table) for table in before} == before


@pytest.mark.parametrize(
    "mapping",
    [
        {"1": "person"},
        {"1": "person", "2": None},
        {"1": 0, "3": None},
        {"1": "unknown-label", "3": None},
        {"01": "person", "3": None},
    ],
)
def test_proposal_mapping_is_explicit_and_uses_frozen_string_ids(client, comparison, mapping):
    assert propose(client, comparison, class_mapping=mapping).status_code in (409, 422)
    assert client.app.state.store.list("temporal_references") == []


def test_mapping_is_not_an_index_conversion_and_all_skip_is_unknown(client, comparison):
    payload = propose(client, comparison, class_mapping={"1": "car", "3": None}).json()["payload"]
    assert payload["identities"][0]["label"] == "car"
    assert payload["frames"][0]["objects"][0]["label"] == "car"
    skipped = propose(client, comparison, class_mapping={"1": None, "3": None}).json()
    assert skipped["payload"]["identities"] == []
    assert skipped["summary"]["unreviewed_frames"] == 3
    assert skipped["summary"]["complete_frames"] == 0
    assert skipped["proposal_summary"]["skipped_observations"] == 2


def test_seed_and_edits_follow_frozen_custom_taxonomy_after_project_changes(
    client, tmp_path, monkeypatch
):
    store = client.app.state.store
    classes = [{"id": "walker", "name": "Walker", "definition": "Observed human", "coco_id": 1}]
    taxonomy = publish_taxonomy(
        store, "default", expected_taxonomy_id=TAXONOMY["id"], classes=classes
    )
    sequence = publish_sequence(client, sequence_source(client, tmp_path))
    cache = create(client, sequence)
    execute(store, cache)
    monkeypatch.setattr("iris.tracking.tracking_status", lambda: {"available": True})
    synthetic(monkeypatch)
    comparison = launch(client, cache)
    assert run_worker(client, comparison, monkeypatch)["status"] == "succeeded"
    publish_taxonomy(
        store,
        "default",
        expected_taxonomy_id=taxonomy["id"],
        classes=[*classes, {"id": "cart", "name": "Cart", "definition": "A shopping cart"}],
    )
    response = propose(client, comparison, class_mapping={"1": "walker", "3": None})
    assert response.status_code == 200, response.text
    draft = response.json()["payload"]
    assert draft["taxonomy_id"] == taxonomy["id"]
    assert draft["identities"][0]["label"] == "walker"
    assert edit(client, draft).status_code == 201
    assert propose(client, comparison, class_mapping={"1": "cart", "3": None}).status_code == 409


def test_editor_never_accepts_automatic_review_and_requires_explicit_certainty(client, comparison):
    payload = propose(client, comparison).json()["payload"]
    for frame in payload["frames"]:
        frame["coverage"] = "complete"
        frame["review"] = {"status": "human_reviewed", "reviewer": "Forged"}
    saved = edit(client, payload)
    assert saved.status_code == 201, saved.text
    record = saved.json()
    assert record["summary"]["human_reviewed_frames"] == 0
    assert record["payload"]["provenance"]["author"] == "Human editor"
    review = [{"frame_index": 0, "coverage": "complete"}]
    rejected = edit(client, record["payload"], revision=1, reviewed_frames=review)
    assert rejected.status_code == 409 and "uncertain" in rejected.text
    assert len(client.app.state.store.list("temporal_references")) == 1
    payload = record["payload"]
    payload["frames"][0]["objects"][0]["certainty"] = "certain"
    accepted = edit(client, payload, revision=1, reviewed_frames=review).json()
    assert accepted["summary"]["human_complete_frames"] == 1
    assert accepted["summary"]["unreviewed_frames"] == 2
    assert accepted["payload"]["frames"][1]["coverage"] == "unreviewed"


@pytest.mark.parametrize(
    "change", ["box", "certainty", "identity", "label", "delete", "visibility"]
)
def test_changed_evidence_invalidates_review_and_preserves_other_frames(client, tmp_path, change):
    sequence = publish_sequence(client, sequence_source(client, tmp_path))
    payload = reference_payload(sequence)
    payload["frames"].append({**deepcopy(payload["frames"][0]), "frame_index": 2})
    first = edit(
        client,
        payload,
        reviewed_frames=[{"frame_index": i, "coverage": "complete"} for i in (0, 2)],
    ).json()
    draft = deepcopy(first["payload"])
    obj = draft["frames"][0]["objects"][0]
    if change == "box":
        obj["box"][0] += 1
    elif change == "certainty":
        obj["certainty"] = "uncertain"
    elif change == "identity":
        draft["identities"].append({"id": "new-human-id", "label": "person"})
        obj["identity_id"] = "new-human-id"
    elif change == "label":
        draft["identities"].append({"id": "new-human-id", "label": "car"})
        obj["identity_id"], obj["label"] = "new-human-id", "car"
    elif change == "visibility":
        obj["visibility"] = "occluded"
        obj["box"] = None
    else:
        draft["frames"][0]["objects"] = []
    saved = edit(client, draft, revision=1, reviewer="Second editor")
    assert saved.status_code == 201, saved.text
    second = saved.json()
    assert second["payload"]["frames"][0]["coverage"] == "unreviewed"
    assert second["payload"]["frames"][0]["review"] == {"status": "unreviewed", "reviewer": ""}
    assert second["payload"]["frames"][1] == first["payload"]["frames"][1]
    assert second["summary"]["human_reviewed_frames"] == 1
    assert temporal.reference_detail(client.app.state.store, first["id"]) == first


def test_legacy_reference_keeps_hash_and_reviews_when_upgraded_and_cas_is_atomic(client, tmp_path):
    sequence = publish_sequence(client, sequence_source(client, tmp_path))
    old = publish_reference(client, sequence)
    saved = edit(client, old["payload"], revision=1)
    assert saved.status_code == 201, saved.text
    record = saved.json()
    assert record["payload"]["schema"] == REFERENCE_SCHEMA_V2
    assert record["payload"]["provenance"] == {"author": "Human editor", "origin": None}
    assert record["payload"]["frames"] == old["payload"]["frames"]
    assert temporal.reference_detail(client.app.state.store, old["id"]) == old
    assert edit(client, record["payload"], revision=1).status_code == 409
    assert len(client.app.state.store.list("temporal_references")) == 2


def test_simultaneous_editor_saves_commit_exactly_one_revision(client, tmp_path):
    sequence = publish_sequence(client, sequence_source(client, tmp_path))
    first = edit(client, reference_payload(sequence)).json()
    barrier = Barrier(2)

    def save(author):
        barrier.wait(timeout=5)
        try:
            return save_identity_edits(
                client.app.state.store,
                sequence["id"],
                payload=first["payload"],
                expected_revision=1,
                reviewer=author,
                reviewed_frames=[],
            )["revision"]
        except temporal.TemporalConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(save, ["First user", "Second user"]))
    assert sorted(map(str, outcomes)) == ["2", "conflict"]
    assert len(client.app.state.store.list("temporal_references")) == 2


def test_object_order_and_notes_preserve_previous_review_author(client, tmp_path):
    sequence = publish_sequence(client, sequence_source(client, tmp_path))
    payload = reference_payload(sequence)
    payload["identities"].append({"id": "observed-b", "label": "person"})
    payload["frames"][0]["objects"].append(
        {
            **deepcopy(payload["frames"][0]["objects"][0]),
            "identity_id": "observed-b",
            "box": [40, 10, 60, 50],
        }
    )
    first = edit(
        client,
        payload,
        reviewer="Original reviewer",
        reviewed_frames=[{"frame_index": 0, "coverage": "complete"}],
    ).json()
    draft = deepcopy(first["payload"])
    draft["frames"][0]["objects"].reverse()
    draft["identities"].reverse()
    draft["notes"] = "Reordered display only"
    draft["frames"][0]["review"]["reviewer"] = "Forged replacement"
    saved = edit(client, draft, revision=1, reviewer="New author").json()
    assert saved["payload"]["provenance"]["author"] == "New author"
    assert saved["payload"]["frames"][0]["review"] == {
        "status": "human_reviewed",
        "reviewer": "Original reviewer",
    }


def test_omitted_and_empty_frames_only_become_negatives_after_explicit_review(client, tmp_path):
    sequence = publish_sequence(client, sequence_source(client, tmp_path))
    first = edit(client, reference_payload(sequence)).json()
    draft = deepcopy(first["payload"])
    draft["frames"].append(
        {
            "frame_index": 1,
            "coverage": "complete",
            "review": {"status": "human_reviewed", "reviewer": "Client claim"},
            "objects": [],
        }
    )
    second = edit(client, draft, revision=1).json()
    assert second["summary"]["unreviewed_frames"] == 3
    assert second["summary"]["omitted_frames"] == 1
    assert second["summary"]["complete_frames"] == 0
    third = edit(
        client,
        second["payload"],
        revision=2,
        reviewed_frames=[{"frame_index": 1, "coverage": "complete"}],
    ).json()
    assert third["summary"]["human_complete_frames"] == 1
    assert third["summary"]["unreviewed_frames"] == 2


def test_seed_snapshot_is_immutable_and_survives_deleted_live_identities(client, comparison):
    draft = propose(client, comparison).json()["payload"]
    first = edit(client, draft).json()
    changed = deepcopy(first["payload"])
    changed["provenance"]["origin"]["track_mapping"]["3"] = "ref_changed"
    assert edit(client, changed, revision=1).status_code == 409
    changed = deepcopy(first["payload"])
    changed["identities"] = []
    for frame in changed["frames"]:
        frame["objects"] = []
    second = edit(client, changed, revision=1)
    assert second.status_code == 201, second.text
    assert second.json()["payload"]["provenance"]["origin"] == draft["provenance"]["origin"]


@pytest.mark.parametrize(
    "field", ["semantic_sha256", "result_sha256", "cache_fingerprint", "profile_sha256"]
)
def test_generic_save_also_checks_seed_hashes(client, comparison, field):
    draft = propose(client, comparison).json()["payload"]
    draft["provenance"]["author"] = "Human"
    draft["provenance"]["origin"][field] = "0" * 64
    response = client.post(
        f"/api/temporal/sequences/{comparison['sequence_id']}/references",
        json={"expected_revision": 0, "payload": draft},
    )
    assert response.status_code == 409
    assert client.app.state.store.list("temporal_references") == []


def test_proposal_requires_exact_sequence_success_and_project(client, comparison, tmp_path):
    foreign = publish_sequence(client, sequence_source(client, tmp_path, group="other"))
    body = {
        "comparison_id": comparison["id"],
        "lane_index": 0,
        "class_mapping": {"1": "person", "3": None},
    }
    response = client.post(f"/api/temporal/sequences/{foreign['id']}/identity-proposals", json=body)
    assert response.status_code == 409 and "exact sequence" in response.text
    project = client.post("/api/projects", json={"name": "Other project"}).json()["id"]
    foreign = publish_sequence(
        client, sequence_source(client, tmp_path, project_id=project, group="foreign"), project
    )
    endpoint = f"/api/temporal/sequences/{foreign['id']}/identity-proposals"
    assert client.post(endpoint, params={"project_id": project}, json=body).status_code == 404
    assert edit(client, reference_payload(foreign)).status_code == 404
    client.app.state.store.update("jobs", comparison["id"], {"status": "running", "result": None})
    assert propose(client, comparison).status_code == 409


@pytest.mark.parametrize(
    "override",
    [
        {"reviewer": " "},
        {"reviewer": "x" * 201},
        {"expected_revision": True},
        {"reviewed_frames": [{"frame_index": 99, "coverage": "complete"}]},
        {"reviewed_frames": [{"frame_index": 0, "coverage": "complete"}] * 2},
    ],
)
def test_edit_rejects_invalid_review_body_without_writes(client, tmp_path, override):
    sequence = publish_sequence(client, sequence_source(client, tmp_path))
    response = edit(client, reference_payload(sequence), **override)
    assert response.status_code in (409, 422)
    assert client.app.state.store.list("temporal_references") == []


def test_archive_round_trips_v2_and_revalidates_seed_snapshot(
    client, comparison, tmp_path, monkeypatch
):
    draft = propose(client, comparison).json()["payload"]
    saved = edit(client, draft).json()
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    store = client.app.state.store
    archive = create_archive(store.root, tmp_path / "identities.zip")
    inspected = inspect_archive(archive["path"])
    restored = tmp_path / "restored"
    result = restore_archive(
        archive["path"], restored, expected_archive_sha256=inspected["archive_sha256"]
    )
    assert result["verified"]
    assert temporal.reference_detail(Store(restored), saved["id"]) == saved
    changed = deepcopy(saved["payload"])
    changed["provenance"]["origin"]["semantic_sha256"] = "0" * 64
    with store.connect() as conn:
        conn.execute(
            "UPDATE temporal_references SET payload=?,payload_sha256=? WHERE id=?",
            (json.dumps(changed), temporal._digest(changed), saved["id"]),
        )
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="temporal records"):
        validate_database(store.root, inventory)
