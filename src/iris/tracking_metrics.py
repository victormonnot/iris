"""Bounded, deterministic quality measurements on frozen temporal evidence.

No tracker, ML library, file access or workspace mutation is needed. The service
validates the frozen comparison; this module validates evaluation configuration
and the reference contract, and never treats unknown frames as negative examples.
"""

import math
from copy import deepcopy

from iris.temporal_contracts import sequence_hash, validate_reference, validate_sequence_manifest

REPORT_SCHEMA = "iris-tracking-quality-v1"
MAX_FRAMES = 500
MAX_FRAME_OBJECTS = 512
MAX_IDENTITY_COUNT = 512
MAX_ASSIGNMENT_WORK = 25_000_000

_POLICY = {
    "name": "iris-observed-identity-v1",
    "observation_policy": (
        "confirmed measured observations only; no predictions or unassigned boxes"
    ),
    "matching_policy": (
        "class-aware IoU threshold; previous-frame correspondence first, "
        "then maximum cardinality, then IoU; deterministic sorted identity ties"
    ),
    "continuity_policy": (
        "events within consecutive evaluable source frames; boxed-reference absence resets its "
        "continuity; observed misses preserve last association; "
        "excluded or missing frames reset all"
    ),
    "idf1_policy": (
        "global one-to-one identities using all threshold-compatible overlaps; "
        "only a fully dense human-complete geometrically evaluable clip"
    ),
}


def quality_status():
    """Public deterministic protocol and limits, without optional runtime imports."""
    return {
        "schema": REPORT_SCHEMA,
        "protocol": {**_POLICY, "default_iou_threshold": 0.5},
        "limits": {
            "max_frames": MAX_FRAMES,
            "max_frame_objects": MAX_FRAME_OBJECTS,
            "max_identity_count": MAX_IDENTITY_COUNT,
            "max_assignment_work": MAX_ASSIGNMENT_WORK,
        },
    }


class _Budget:
    def __init__(self):
        self.work = 0

    def charge(self, rows, columns, *, assignment=False):
        work = rows * columns * (min(rows, columns) if assignment else 1)
        if self.work + work > MAX_ASSIGNMENT_WORK:
            raise ValueError(
                "Tracking quality exceeds the bounded matching workload; "
                "prepare a shorter sequence or an explicit smaller class scope"
            )
        self.work += work


def _maximum_assignment(weights, budget=None):
    """Maximum-weight rectangular assignment, deterministic including equal costs.

    Hungarian shortest augmenting paths, O(min(n,m)^2 * max(n,m)), using only
    Python numbers. Zero-weight matches are discarded by callers. A rectangular
    assignment is sufficient for nonnegative weights: unmatched items have zero
    profit, so forced zero assignments do not affect the optimum.
    """
    rows = len(weights)
    columns = len(weights[0]) if rows else 0
    if not rows or not columns:
        return []
    if any(len(row) != columns for row in weights):
        raise ValueError("Assignment weights must be rectangular")
    if budget is not None:
        budget.charge(rows, columns, assignment=True)
    transposed = rows > columns
    matrix = [list(row) for row in zip(*weights, strict=True)] if transposed else weights
    row_count, column_count = len(matrix), len(matrix[0])
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    assigned_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        assigned_row[0] = row
        current_column = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = assigned_row[current_column]
            delta, next_column = math.inf, 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                cost = (
                    -matrix[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if cost < minimum[column]:
                    minimum[column] = cost
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta, next_column = minimum[column], column
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[assigned_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if not assigned_row[current_column]:
                break
        while current_column:
            previous_column = predecessor[current_column]
            assigned_row[current_column] = assigned_row[previous_column]
            current_column = previous_column
    result = [(row - 1, column - 1) for column, row in enumerate(assigned_row[1:], 1) if row]
    if transposed:
        result = [(column, row) for row, column in result]
    return sorted(result)


def _iou(first, second):
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    if not intersection:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _configuration(lanes, sequence, class_mapping, iou_threshold):
    try:
        valid_threshold = (
            type(iou_threshold) in (int, float)
            and math.isfinite(iou_threshold)
            and 0 < iou_threshold <= 1
        )
    except OverflowError:
        valid_threshold = False
    if not valid_threshold:
        raise ValueError("Tracking quality IoU threshold must be finite and in (0, 1]")
    expected = {str(value) for value in lanes[0]["report"]["profile"]["class_ids"]}
    if any(
        {str(value) for value in lane["report"]["profile"]["class_ids"]} != expected
        for lane in lanes
    ):
        raise ValueError("Tracking quality requires the same native classes in both lanes")
    if not isinstance(class_mapping, dict) or set(class_mapping) != expected:
        raise ValueError("Map every native comparison class to a taxonomy ID or null explicitly")
    labels = {entry["id"] for entry in sequence["taxonomy"]["classes"]}
    if any(
        value is not None and (not isinstance(value, str) or value not in labels)
        for value in class_mapping.values()
    ):
        raise ValueError("Tracking quality class mapping must use the frozen sequence taxonomy")
    if not any(value is not None for value in class_mapping.values()):
        raise ValueError("Tracking quality requires at least one included class")
    return (
        {key: class_mapping[key] for key in sorted(class_mapping, key=int)},
        float(iou_threshold),
    )


def _coverage(sequence, payload, labels):
    reference_frames = {frame["frame_index"]: frame for frame in payload["frames"]}
    evaluated, excluded, ground_truth = [], [], {}
    for source in sequence["frames"]:
        index = source["frame_index"]
        frame = reference_frames.get(index)
        reasons = []
        if frame is None:
            reasons.append("missing_reference_frame")
        else:
            if frame["review"]["status"] != "human_reviewed":
                reasons.append("not_human_reviewed")
            if frame["coverage"] != "complete":
                reasons.append("incomplete_frame_review")
            objects = [obj for obj in frame["objects"] if obj["label"] in labels]
            if any(
                obj["certainty"] != "certain"
                or obj["identity_id"] is None
                or obj["visibility"] == "unknown"
                or (obj["visibility"] == "occluded" and obj["box"] is None)
                for obj in objects
            ):
                reasons.append("unlocalized_or_uncertain_reference")
        if reasons:
            excluded.append({"frame_index": index, "reasons": reasons})
            continue
        evaluated.append(index)
        ground_truth[index] = sorted(
            (obj for obj in objects if obj["visibility"] in {"visible", "occluded"}),
            key=lambda obj: obj["identity_id"],
        )
    source_count = sequence["clip"]["end_frame"] - sequence["clip"]["start_frame"] + 1
    coverage = {
        "available_frames": len(sequence["frames"]),
        "source_frames": source_count,
        "evaluated_frames": len(evaluated),
        "excluded_frames": len(excluded),
        "evaluated_frame_indices": evaluated,
        "excluded": excluded,
        "evaluated_transitions": sum(
            second == first + 1 for first, second in zip(evaluated, evaluated[1:], strict=False)
        ),
        "dense": len(evaluated) == source_count,
        "scope_labels": sorted(labels),
        "reference_origin": deepcopy(payload.get("provenance", {}).get("origin")),
    }
    return coverage, ground_truth


def _event(kind, identity, track_id, *, previous_track_id=None, previous_reference_identity=None):
    return {
        "kind": kind,
        "reference_identity": identity,
        "track_id": track_id,
        "previous_track_id": previous_track_id,
        "previous_reference_identity": previous_reference_identity,
    }


def _identity_score(coverage, counts, potential, gt_ids, track_ids, budget):
    reason = None
    if not coverage["evaluated_frames"]:
        reason = "no_evaluated_frames"
    elif not coverage["dense"]:
        reason = "incomplete_reference_coverage"
    elif not counts["ground_truth"]:
        reason = "no_ground_truth_identity_detections"
    if reason:
        return {
            "available": False,
            "reason": reason,
            "idtp": None,
            "idfp": None,
            "idfn": None,
            "idf1": None,
        }
    if len(gt_ids) + len(track_ids) > MAX_IDENTITY_COUNT:
        raise ValueError(
            f"Tracking quality accepts at most {MAX_IDENTITY_COUNT} combined reference and "
            "tracker identities per lane; prepare a shorter sequence or smaller class scope"
        )
    rows, columns = sorted(gt_ids), sorted(track_ids)
    weights = [[potential.get((identity, track), 0) for track in columns] for identity in rows]
    idtp = sum(weights[row][column] for row, column in _maximum_assignment(weights, budget))
    idfn, idfp = counts["ground_truth"] - idtp, counts["observations"] - idtp
    return {
        "available": True,
        "reason": None,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "idf1": 2 * idtp / (2 * idtp + idfn + idfp),
    }


def _lane_quality(lane, coverage, ground_truth, mapping, threshold, budget):
    counts = dict.fromkeys(
        (
            "ground_truth",
            "observations",
            "true_positives",
            "false_positives",
            "false_negatives",
            "identity_switches",
            "fragments",
            "identity_transfers",
            "class_confusions",
            "excluded_predictions",
            "excluded_unconfirmed",
            "excluded_unassigned",
        ),
        0,
    )
    exclusions = {item["frame_index"]: item["reasons"] for item in coverage["excluded"]}
    frames, potential, gt_ids, track_ids = [], {}, set(), set()
    previous_index, previous_matches, continuity, tracker_history = None, {}, {}, {}
    for source in lane["report"]["passes"][0]["frames"]:
        index = source["frame_index"]
        for key, counter in (
            ("predictions", "excluded_predictions"),
            ("unassigned", "excluded_unassigned"),
        ):
            counts[counter] += sum(mapping[str(obj["label_id"])] is not None for obj in source[key])
        scoped = [
            obj for obj in source["observations"] if mapping[str(obj["label_id"])] is not None
        ]
        counts["excluded_unconfirmed"] += sum(not obj["confirmed"] for obj in scoped)
        observations = sorted(
            (obj for obj in scoped if obj["confirmed"]), key=lambda obj: obj["track_id"]
        )
        frame = {
            "frame_index": index,
            "evaluated": index in ground_truth,
            "reasons": exclusions.get(index, []),
            "matches": [],
            "false_negatives": [],
            "false_positives": [],
            "events": [],
            "class_confusions": [],
        }
        frames.append(frame)
        if index not in ground_truth:
            previous_index, previous_matches, continuity, tracker_history = None, {}, {}, {}
            continue
        if previous_index is None or index != previous_index + 1:
            previous_matches, continuity, tracker_history = {}, {}, {}
        previous_index = index
        objects = ground_truth[index]
        if len(objects) + len(observations) > MAX_FRAME_OBJECTS:
            raise ValueError(
                f"Tracking quality accepts at most {MAX_FRAME_OBJECTS} combined reference and "
                "observed boxes per evaluated frame; choose a smaller explicit class scope"
            )
        current_ids = {obj["identity_id"] for obj in objects}
        continuity = {
            identity: state for identity, state in continuity.items() if identity in current_ids
        }
        gt_ids.update(current_ids)
        track_ids.update(obj["track_id"] for obj in observations)
        counts["ground_truth"] += len(objects)
        counts["observations"] += len(observations)
        budget.charge(len(objects), len(observations))
        similarities = [[_iou(obj["box"], pred["box"]) for pred in observations] for obj in objects]
        count = min(len(objects), len(observations))
        cardinality_bonus, continuity_bonus = count + 1, (count + 1) ** 2
        weights = []
        for row, obj in enumerate(objects):
            values = []
            identity = obj["identity_id"]
            for column, pred in enumerate(observations):
                similarity = similarities[row][column]
                valid = similarity >= threshold and obj["label"] == mapping[str(pred["label_id"])]
                if valid:
                    pair = (identity, pred["track_id"])
                    potential[pair] = potential.get(pair, 0) + 1
                    values.append(
                        cardinality_bonus
                        + similarity
                        + continuity_bonus * (previous_matches.get(identity) == pred["track_id"])
                    )
                else:
                    values.append(0)
            weights.append(values)
        matches = [
            (row, column)
            for row, column in _maximum_assignment(weights, budget)
            if weights[row][column] > 0
        ]
        matched_rows, matched_columns = (
            {row for row, _ in matches},
            {column for _, column in matches},
        )
        counts["true_positives"] += len(matches)
        counts["false_negatives"] += len(objects) - len(matches)
        counts["false_positives"] += len(observations) - len(matches)
        next_matches = {}
        for row, column in matches:
            identity, track_id = objects[row]["identity_id"], observations[column]["track_id"]
            frame["matches"].append(
                {
                    "reference_identity": identity,
                    "track_id": track_id,
                    "iou": similarities[row][column],
                }
            )
            state = continuity.get(identity, {"track_id": None, "missed": False})
            if state["track_id"] is not None and state["track_id"] != track_id:
                counts["identity_switches"] += 1
                frame["events"].append(
                    _event(
                        "identity_switch", identity, track_id, previous_track_id=state["track_id"]
                    )
                )
            if state["missed"]:
                counts["fragments"] += 1
                frame["events"].append(
                    _event("fragment", identity, track_id, previous_track_id=state["track_id"])
                )
            former = tracker_history.get(track_id)
            if former is not None and former != identity:
                counts["identity_transfers"] += 1
                frame["events"].append(
                    _event(
                        "identity_transfer", identity, track_id, previous_reference_identity=former
                    )
                )
            continuity[identity] = {"track_id": track_id, "missed": False}
            tracker_history[track_id] = identity
            next_matches[identity] = track_id
        for row, obj in enumerate(objects):
            if row not in matched_rows:
                identity = obj["identity_id"]
                frame["false_negatives"].append(identity)
                if identity in continuity:
                    continuity[identity]["missed"] = True
        frame["false_positives"] = [
            obj["track_id"]
            for column, obj in enumerate(observations)
            if column not in matched_columns
        ]
        previous_matches = next_matches
        unmatched_rows = [row for row in range(len(objects)) if row not in matched_rows]
        unmatched_columns = [
            column for column in range(len(observations)) if column not in matched_columns
        ]
        wrong_class = [
            [
                cardinality_bonus + similarities[row][column]
                if similarities[row][column] >= threshold
                and objects[row]["label"] != mapping[str(observations[column]["label_id"])]
                else 0
                for column in unmatched_columns
            ]
            for row in unmatched_rows
        ]
        for row, column in _maximum_assignment(wrong_class, budget):
            if wrong_class[row][column] <= 0:
                continue
            gt_index, pred_index = unmatched_rows[row], unmatched_columns[column]
            obj, pred = objects[gt_index], observations[pred_index]
            frame["class_confusions"].append(
                {
                    "reference_identity": obj["identity_id"],
                    "track_id": pred["track_id"],
                    "reference_label": obj["label"],
                    "observed_label": mapping[str(pred["label_id"])],
                    "iou": similarities[gt_index][pred_index],
                }
            )
        counts["class_confusions"] += len(frame["class_confusions"])
    identity = _identity_score(coverage, counts, potential, gt_ids, track_ids, budget)
    return {
        "name": lane["name"],
        "profile_sha256": lane["report"]["profile_sha256"],
        "counts": counts,
        "precision": counts["true_positives"] / counts["observations"]
        if counts["observations"]
        else None,
        "recall": counts["true_positives"] / counts["ground_truth"]
        if counts["ground_truth"]
        else None,
        "identity": identity,
        "frames": frames,
    }


def evaluate_quality(comparison, reference, *, class_mapping, iou_threshold=0.5):
    """Evaluate a checked T4 comparison against exactly one immutable T1/T5 revision."""
    report = comparison["report"]
    sequence = validate_sequence_manifest(report["sequence"])
    payload = validate_reference(reference["payload"], sequence)
    lanes = report["lanes"]
    if len(lanes) != 2 or not 1 <= len(sequence["frames"]) <= MAX_FRAMES:
        raise ValueError(f"Tracking quality requires two lanes and 1–{MAX_FRAMES} source frames")
    indices = [frame["frame_index"] for frame in sequence["frames"]]
    if any(
        [frame["frame_index"] for frame in lane["report"]["passes"][0]["frames"]] != indices
        for lane in lanes
    ):
        raise ValueError("Tracking quality requires both complete frozen source frame sequences")
    mapping, threshold = _configuration(lanes, sequence, class_mapping, iou_threshold)
    labels = {label for label in mapping.values() if label is not None}
    coverage, ground_truth = _coverage(sequence, payload, labels)
    budget = _Budget()
    results = [
        _lane_quality(lane, coverage, ground_truth, mapping, threshold, budget) for lane in lanes
    ]
    cache = lanes[0]["report"]["cache"]
    return {
        "schema": REPORT_SCHEMA,
        "protocol": {**_POLICY, "iou_threshold": threshold, "class_mapping": mapping},
        "source": {
            "comparison_id": comparison["id"],
            "sequence_id": sequence["id"],
            "sequence_sha256": sequence_hash(sequence),
            "reference_id": reference["id"],
            "reference_revision": reference["revision"],
            "reference_sha256": reference["payload_sha256"],
            "cache_fingerprint": cache["fingerprint"],
            "result_sha256": cache["result_sha256"],
            "lane_profile_sha256": [lane["report"]["profile_sha256"] for lane in lanes],
            "lane_semantic_sha256": [
                lane["report"]["passes"][0]["semantic_sha256"] for lane in lanes
            ],
        },
        "coverage": coverage,
        "lanes": results,
        "limitations": [
            "This is one frozen sequence and reference revision, "
            "not an independent ranking or qualification.",
            "Human review is declared by the annotator; "
            "IRIS cannot authenticate annotation quality.",
            "Assisted reference seeds may share tracker boxes and identities; "
            "human review does not prove independent reference construction.",
            "Only human-complete geometrically evaluable frames count. "
            "Partial, assistant, unknown and missing frames are not negatives.",
            "Switches and fragments concern continuous boxed-reference presence; source gaps, "
            "excluded frames and reference absence reset continuity. These are IRIS diagnostics, "
            "not official CLEAR benchmark scores.",
            "Identity transfers and spatial class confusions are diagnostics, "
            "not additional false-positive or false-negative penalties.",
            "Only confirmed measured observations are scored; estimated, predicted, unconfirmed "
            "and unassigned boxes do not establish observed identity matches.",
            "Excluded prediction, unconfirmed and unassigned counts cover all available frames "
            "in the explicit class scope; other counts cover evaluated frames only.",
            "IDF1 requires a dense fully evaluated source clip and global identity assignment; "
            "sparse point scores do not establish continuity through unknown intervals.",
        ],
    }
