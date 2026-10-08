"""Read-only identity proposals and explicit human edits of immutable references.

Tracker associations propose identities; only measured detector boxes are seeded.
Seed provenance is a historical snapshot and survives later identity edits.
"""

import json
from copy import deepcopy

from iris.store import new_id
from iris.temporal_contracts import (
    MAX_SEQUENCE_FRAMES,
    REFERENCE_SCHEMA,
    REFERENCE_SCHEMA_V2,
    _choice,
    _integer,
    _items,
    _object,
    _text,
    reference_summary,
    validate_reference,
    validate_reference_provenance,
)


def _source(conn, sequence, comparison_id, lane_index):
    from iris.tracking_comparisons import _checked_job

    _text(comparison_id, "Seed comparison ID")
    _integer(lane_index, "Seed lane index", maximum=1)
    job, cache, source = _checked_job(conn, comparison_id)
    if source["id"] != sequence["id"] or source["project_id"] != sequence["project_id"]:
        raise ValueError("Identity proposal must belong to the exact sequence and project")
    if job["status"] != "succeeded":
        raise ValueError("Identity proposals require a successful frozen tracking comparison")
    lane = job["result"]["lanes"][lane_index]["report"]
    return job, cache, lane


def _class_mapping(mapping, lane, sequence):
    expected = {str(value) for value in lane["profile"]["class_ids"]}
    labels = {item["id"] for item in sequence["manifest"]["taxonomy"]["classes"]}
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise ValueError("Map every native lane class explicitly to a frozen taxonomy ID or null")
    for value in mapping.values():
        if value is not None:
            _choice(value, labels, "Identity proposal taxonomy ID")
    return deepcopy(mapping)


def _observed_tracks(lane, mapping):
    return {
        str(observation["track_id"])
        for frame in lane["passes"][0]["frames"]
        for observation in frame["observations"]
        if mapping[str(observation["label_id"])] is not None
    }


def validate_origin(conn, payload, sequence):
    """Recheck saved seed hashes and mappings without loading detector/tracker runtimes."""
    if payload["schema"] == REFERENCE_SCHEMA:
        return
    provenance = payload["provenance"]
    _text(provenance["author"], "Reference author", maximum=200)
    origin = provenance["origin"]
    if origin is None:
        return
    job, cache, lane = _source(conn, sequence, origin["comparison_id"], origin["lane_index"])
    if any(
        origin[key] != value
        for key, value in {
            "cache_id": cache["id"],
            "cache_fingerprint": cache["fingerprint"],
            "result_sha256": job["params"]["result_sha256"],
            "profile_sha256": lane["profile_sha256"],
            "semantic_sha256": lane["passes"][0]["semantic_sha256"],
        }.items()
    ):
        raise ValueError("Reference seed provenance does not match its frozen tracking source")
    mapping = _class_mapping(origin["class_mapping"], lane, sequence)
    if set(origin["track_mapping"]) != _observed_tracks(lane, mapping):
        raise ValueError("Reference seed track mapping does not match its original observations")


def identity_proposal(store, sequence_id, *, comparison_id, lane_index, class_mapping):
    """Build a detached draft. Never write references, jobs, cache rows or model outputs."""
    from iris.temporal import _row, _sequence_record

    with store.connect() as conn:
        conn.execute("BEGIN")
        sequence = _sequence_record(conn, _row(conn, "temporal_sequences", sequence_id))
        job, cache, lane = _source(conn, sequence, comparison_id, lane_index)
        mapping = _class_mapping(class_mapping, lane, sequence)
        tracks = {
            native_id: f"ref_{new_id()}"
            for native_id in sorted(_observed_tracks(lane, mapping), key=int)
        }
        origin = {
            "comparison_id": job["id"],
            "lane_index": lane_index,
            "cache_id": cache["id"],
            "cache_fingerprint": cache["fingerprint"],
            "result_sha256": job["params"]["result_sha256"],
            "profile_sha256": lane["profile_sha256"],
            "semantic_sha256": lane["passes"][0]["semantic_sha256"],
            "class_mapping": mapping,
            "track_mapping": tracks,
        }
        identities, frames, skipped, predictions = {}, [], 0, 0
        for source in lane["passes"][0]["frames"]:
            objects = []
            predictions += len(source["predictions"])
            for observation in source["observations"]:
                label = mapping[str(observation["label_id"])]
                if label is None:
                    skipped += 1
                    continue
                identity_id = tracks[str(observation["track_id"])]
                identities[identity_id] = label
                objects.append(
                    {
                        "identity_id": identity_id,
                        "label": label,
                        "box": deepcopy(observation["box"]),
                        "visibility": "visible",
                        "certainty": "uncertain",
                    }
                )
            frames.append(
                {
                    "frame_index": source["frame_index"],
                    "coverage": "unreviewed",
                    "review": {"status": "unreviewed", "reviewer": ""},
                    "objects": objects,
                }
            )
        manifest = sequence["manifest"]
        payload = validate_reference(
            {
                "schema": REFERENCE_SCHEMA_V2,
                "sequence_id": sequence_id,
                "sequence_sha256": sequence["manifest_sha256"],
                "taxonomy_id": manifest["taxonomy"]["id"],
                "identities": [
                    {"id": identifier, "label": label} for identifier, label in identities.items()
                ],
                "frames": frames,
                "notes": "",
                "provenance": {"author": "", "origin": origin},
            },
            manifest,
        )
        summary = reference_summary(payload, manifest)
        return {
            "payload": payload,
            "summary": summary,
            "proposal_summary": {
                "seeded_objects": summary["object_count"],
                "seeded_identities": summary["identity_count"],
                "skipped_observations": skipped,
                "excluded_predictions": predictions,
            },
        }


def _frame_content(frame):
    # Object order has no evidentiary meaning. Each object's class, identity,
    # measured geometry, visibility and certainty does.
    return sorted(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for obj in frame["objects"]
    )


def prepare_edit(payload, previous, sequence, *, reviewer, reviewed_frames):
    """Reset changed evidence, preserve unchanged reviews, then apply explicit reviews."""
    reviewer = _text(reviewer, "Identity editor reviewer", maximum=200)
    reviewed_frames = _items(
        reviewed_frames, "Explicitly reviewed frames", maximum=MAX_SEQUENCE_FRAMES
    )
    draft = deepcopy(payload)
    if not isinstance(draft, dict):
        raise ValueError("Identity editor payload must be a reference document")
    # Do not accept client-provided review claims. Check their shape before replacing
    # them so malformed fields still fail the strict reference contract.
    frames = _items(draft.get("frames"), "Reference frames", maximum=MAX_SEQUENCE_FRAMES)
    for frame in frames:
        _object(frame, {"frame_index", "coverage", "review", "objects"}, "Reference frame")
        _choice(frame["coverage"], {"complete", "partial", "unreviewed"}, "Reference coverage")
        review = _object(frame["review"], {"status", "reviewer"}, "Reference review")
        _choice(
            review["status"],
            {"unreviewed", "assistant_reviewed", "human_reviewed"},
            "Review status",
        )
        _text(review["reviewer"], "Reviewer", maximum=200, empty=True)
        frame["coverage"] = "unreviewed"
        frame["review"] = {"status": "unreviewed", "reviewer": ""}
    draft = validate_reference(draft, sequence["manifest"])
    if draft["schema"] == REFERENCE_SCHEMA:
        draft["schema"] = REFERENCE_SCHEMA_V2
        draft["provenance"] = {"author": "", "origin": None}
    if previous is not None:
        old_origin = previous.get("provenance", {}).get("origin")
        if draft["provenance"]["origin"] != old_origin:
            raise ValueError("The saved reference seed origin is immutable during identity edits")
    draft["provenance"]["author"] = reviewer
    validate_reference_provenance(
        draft["provenance"], {item["id"] for item in sequence["manifest"]["taxonomy"]["classes"]}
    )
    old_frames = {frame["frame_index"]: frame for frame in previous["frames"]} if previous else {}
    new_frames = {frame["frame_index"]: frame for frame in draft["frames"]}
    for index, frame in new_frames.items():
        old = old_frames.get(index)
        if old is not None and _frame_content(old) == _frame_content(frame):
            frame["coverage"] = old["coverage"]
            frame["review"] = deepcopy(old["review"])
    seen = set()
    for review in reviewed_frames:
        _object(review, {"frame_index", "coverage"}, "Explicit frame review")
        index = _integer(review["frame_index"], "Reviewed frame index")
        coverage = _choice(review["coverage"], {"complete", "partial"}, "Reviewed coverage")
        if index in seen or index not in new_frames:
            raise ValueError("Explicit reviews require distinct frames present in the reference")
        seen.add(index)
        new_frames[index]["coverage"] = coverage
        new_frames[index]["review"] = {"status": "human_reviewed", "reviewer": reviewer}
    return validate_reference(draft, sequence["manifest"])


def save_identity_edits(
    store, sequence_id, *, payload, expected_revision, reviewer, reviewed_frames
):
    from iris.temporal import save_reference

    return save_reference(
        store,
        sequence_id,
        payload=payload,
        expected_revision=expected_revision,
        _identity_review={"reviewer": reviewer, "reviewed_frames": reviewed_frames},
    )
