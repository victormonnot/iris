"""Read-only, bounded partition proposals over a saved version of reviewed classes."""

import hashlib
import math
import re

from iris.datasets import MAX_FRAMES, _canonical, dataset_candidates
from iris.store import DEFAULT_PROJECT_ID, Store

PROTOCOL = "iris-dataset-plan-v1"
DEFAULT_RATIOS = {"train": 0.8, "val": 0.2, "test": 0.0}
MAX_SIMILAR_PAIRS = 100
SIMILAR_DISTANCE = 4
_DHASH = re.compile(r"[0-9a-fA-F]{16}\Z")


def _ratios(value: dict | None) -> dict:
    value = DEFAULT_RATIOS if value is None else value
    if (
        not isinstance(value, dict)
        or set(value) != set(DEFAULT_RATIOS)
        or any(
            type(v) not in (int, float) or not 0 <= v <= 1 or not math.isfinite(v)
            for v in value.values()
        )
        or not math.isclose(sum(value.values()), 1.0, rel_tol=0, abs_tol=1e-9)
        or value["train"] <= 0
        or value["val"] <= 0
    ):
        raise ValueError(
            "Ratios must be train/val/test fractions totaling 1, with positive train and val"
        )
    return {split: float(value[split]) for split in DEFAULT_RATIOS}


def _counts(frames: list[dict], classes: list[str]) -> dict:
    return {
        "frame_count": len(frames),
        "negative_count": sum(frame["negative"] for frame in frames),
        "class_counts": {
            label: sum(frame["class_counts"][label] for frame in frames) for label in classes
        },
    }


def _components(groups: list[dict]) -> list[list[dict]]:
    """Scene groups joined by the same original video or exact pixels are indivisible."""
    parents = {group["scene_group"]: group["scene_group"] for group in groups}

    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    identities = {}
    for group in groups:
        name = group["scene_group"]
        for frame in group["frames"]:
            keys = [("pixels", frame["sha256"])]
            if frame["video_sha256"]:
                keys.append(("video", frame["video_sha256"]))
            for key in keys:
                if key in identities:
                    parents[find(name)] = find(identities[key])
                else:
                    identities[key] = name
    components = {}
    for group in groups:
        components.setdefault(find(group["scene_group"]), []).append(group)
    return list(components.values())


def _near_pairs(frames: list[dict], splits: dict) -> tuple[list[dict], int, int]:
    """At most 499500 comparisons and 100 exposed pairs; a signal, never a proof."""
    hashes = {
        frame["id"]: int(frame["perceptual_hash"], 16)
        for frame in frames
        if isinstance(frame.get("perceptual_hash"), str)
        and _DHASH.fullmatch(frame["perceptual_hash"])
    }
    count, invalid, pairs = 0, len(frames) - len(hashes), []
    # Cross-partition pairs deserve first inspection. Keep only the best bounded
    # list rather than retaining up to half a million large dictionaries.
    for index, frame in enumerate(frames):
        if frame["id"] not in hashes:
            continue
        for other in frames[index + 1 :]:
            if other["id"] not in hashes or frame["sha256"] == other["sha256"]:
                continue
            distance = (hashes[frame["id"]] ^ hashes[other["id"]]).bit_count()
            if distance > SIMILAR_DISTANCE:
                continue
            count += 1
            assignments = [splits.get(frame["scene_group"]), splits.get(other["scene_group"])]
            cross = None not in assignments and assignments[0] != assignments[1]
            rank = (not cross, distance, frame["id"], other["id"])
            pair = {
                "frame_ids": [frame["id"], other["id"]],
                "scene_groups": [frame["scene_group"], other["scene_group"]],
                "splits": assignments,
                "distance": distance,
                "cross_split": cross,
            }
            if len(pairs) < MAX_SIMILAR_PAIRS or rank < pairs[-1][0]:
                pairs.append((rank, pair))
                pairs.sort(key=lambda item: item[0])
                del pairs[MAX_SIMILAR_PAIRS:]
    return [pair for _, pair in pairs], count, invalid


def preview_dataset_plan(
    store: Store,
    project_id: str = DEFAULT_PROJECT_ID,
    taxonomy_id: str | None = None,
    ratios: dict | None = None,
    seed: int = 0,
) -> dict:
    """Propose group assignments without changing selection, annotations or releases.

    The fingerprint identifies this preview's inputs, not an authorization to
    freeze them later. The regular dataset publication transaction rechecks all
    eligibility, revision tokens, original-video identities and reservations.
    """
    ratios = _ratios(ratios)
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("Seed must be an integer between 0 and 2147483647")
    candidates = dataset_candidates(store, project_id, taxonomy_id)
    groups = candidates["groups"]
    frames = sorted(
        [
            {**frame, "scene_group": group["scene_group"]}
            for group in groups
            for frame in group["frames"]
        ],
        key=lambda frame: frame["id"],
    )
    if len(frames) > MAX_FRAMES:
        raise ValueError(
            f"Partition planning supports at most {MAX_FRAMES} eligible frames; "
            "refine the selection first"
        )
    classes = [item["id"] for item in candidates["taxonomy"]["classes"]]
    total = _counts(frames, classes)
    blockers, warnings = [], list(candidates["warnings"])
    if not frames:
        blockers.append(
            {
                "code": "no_candidates",
                "message": (
                    "Select and validate images in this class version before planning partitions."
                ),
            }
        )
    duplicates = {}
    for frame in frames:
        duplicates.setdefault(frame["sha256"], []).append(frame)
    exact_duplicates = [
        {
            "sha256": digest,
            "frame_ids": [f["id"] for f in copies],
            "scene_groups": sorted({f["scene_group"] for f in copies}),
        }
        for digest, copies in sorted(duplicates.items())
        if len(copies) > 1
    ]
    if exact_duplicates:
        blockers.append(
            {
                "code": "exact_duplicates",
                "message": (
                    "Exact duplicate pixels are selected. Explicitly deselect redundant copies "
                    "before freezing; this proposal does not choose a copy for you."
                ),
                "frame_ids": [
                    identifier
                    for duplicate in exact_duplicates
                    for identifier in duplicate["frame_ids"]
                ],
            }
        )
    components = []
    related = {}
    for linked in _components(groups):
        names = sorted(group["scene_group"] for group in linked)
        related.update({name: names for name in names})
        component_frames = [frame for group in linked for frame in group["frames"]]
        reservations = {group["reserved_split"] for group in linked if group["reserved_split"]}
        for frame in component_frames:
            if frame["reserved_split"]:
                reservations.add(frame["reserved_split"])
            reservations.update(frame["video_reserved_splits"])
        components.append(
            {
                "groups": names,
                "counts": _counts(component_frames, classes),
                "reservations": reservations,
            }
        )
    actual = {
        split: {"frame_count": 0, "negative_count": 0, "class_counts": dict.fromkeys(classes, 0)}
        for split in ratios
    }
    splits = {}

    def assign(component, split):
        splits.update(dict.fromkeys(component["groups"], split))
        actual[split]["frame_count"] += component["counts"]["frame_count"]
        actual[split]["negative_count"] += component["counts"]["negative_count"]
        for label in classes:
            actual[split]["class_counts"][label] += component["counts"]["class_counts"][label]

    free = []
    for component in components:
        reservations = component["reservations"]
        if len(reservations) > 1:
            blockers.append(
                {
                    "code": "conflicting_reservations",
                    "message": (
                        "Linked scene groups, pixels or an original video have incompatible "
                        "reserved splits. Review their provenance; the planner cannot move "
                        "a reservation."
                    ),
                    "scene_groups": component["groups"],
                    "reserved_splits": sorted(reservations),
                }
            )
        elif reservations:
            assign(component, next(iter(reservations)))
        else:
            free.append(component)
    free.sort(
        key=lambda c: (
            -c["counts"]["frame_count"],
            hashlib.sha256(_canonical({"seed": seed, "groups": c["groups"]})).hexdigest(),
        )
    )
    active = [split for split, ratio in ratios.items() if ratio > 0]

    def score(component, split):
        # A deterministic greedy approximation, not a stratification guarantee.
        # Preserve frame ratios first; class/negative counts provide a weaker tie
        # breaker so a rare class never overrides an original-video boundary.
        target = total["frame_count"] * ratios[split]
        before = actual[split]["frame_count"]
        after = before + component["counts"]["frame_count"]
        result = ((after - target) ** 2 - (before - target) ** 2) / max(target, 1)
        for label in classes:
            target = total["class_counts"][label] * ratios[split]
            before = actual[split]["class_counts"][label]
            after = before + component["counts"]["class_counts"][label]
            result += (
                0.2
                * ((after - target) ** 2 - (before - target) ** 2)
                / max(target, 1)
                / len(classes)
            )
        target = total["negative_count"] * ratios[split]
        before = actual[split]["negative_count"]
        after = before + component["counts"]["negative_count"]
        return result + 0.1 * ((after - target) ** 2 - (before - target) ** 2) / max(target, 1)

    for index, component in enumerate(free):
        empty = [split for split in active if not actual[split]["frame_count"]]
        choices = empty if len(free) - index <= len(empty) else active
        chosen = min(choices, key=lambda split: (score(component, split), active.index(split)))
        assign(component, chosen)
    for split in ratios:
        if not actual[split]["frame_count"] and ratios[split] > 0:
            message = (
                f"No {split} images: indivisible groups and reservations cannot populate "
                "every requested partition."
            )
            warnings.append(message)
            if split in {"train", "val"}:
                blockers.append(
                    {"code": "empty_required_split", "message": message, "split": split}
                )
        if actual[split]["frame_count"] and ratios[split] == 0:
            warnings.append(
                f"{split} contains reserved images despite its requested zero ratio; "
                "reservations take precedence."
            )
        if actual[split]["frame_count"]:
            missing = [label for label in classes if actual[split]["class_counts"][label] == 0]
            if missing:
                warnings.append(
                    f"No reviewed objects for these classes in {split}: {', '.join(missing)}."
                )
    if not actual["test"]["frame_count"]:
        warnings.append(
            "No test split: validation data must not be reported as an independent test."
        )
    if sum(actual["train"]["class_counts"].values()) == 0:
        warnings.append(
            "The proposed training split has no positive objects; training requires "
            "at least one positive annotation."
        )
    warnings.append(
        "Ratios are approximate because complete linked groups and existing reservations "
        "take precedence. Class coverage and visual independence are not guaranteed."
    )
    similar_pairs, near_total, invalid_hashes = _near_pairs(frames, splits)
    if near_total:
        warnings.append(
            f"{near_total} pairs have dHash distance at most {SIMILAR_DISTANCE}; showing up to "
            f"{MAX_SIMILAR_PAIRS}, prioritizing cross-split pairs. Inspect the images: this "
            "heuristic can miss related images and flag unrelated images."
        )
    if invalid_hashes:
        warnings.append(
            f"Perceptual similarity could not be checked for {invalid_hashes} images "
            "with unavailable hashes."
        )
    fingerprint = hashlib.sha256(
        _canonical(
            {
                "protocol": PROTOCOL,
                "project_id": project_id,
                "taxonomy": candidates["taxonomy"],
                "ratios": ratios,
                "seed": seed,
                "groups": groups,
            }
        )
    ).hexdigest()
    return {
        "protocol": PROTOCOL,
        "project_id": project_id,
        "taxonomy": candidates["taxonomy"],
        "taxonomy_id": candidates["taxonomy"]["id"],
        "ratios": ratios,
        "seed": seed,
        "splits": splits,
        "frame_ids": [frame["id"] for frame in frames],
        "expected_revisions": {frame["id"]: frame["annotation_revision_id"] for frame in frames},
        "fingerprint": fingerprint,
        "groups": [
            {
                "scene_group": group["scene_group"],
                "split": splits.get(group["scene_group"]),
                "count": group["count"],
                **{
                    key: value
                    for key, value in _counts(group["frames"], classes).items()
                    if key != "frame_count"
                },
                "reserved_split": group["reserved_split"],
                "related_groups": related[group["scene_group"]],
                "frame_ids": [frame["id"] for frame in group["frames"]],
            }
            for group in groups
        ],
        "summary": {
            **total,
            "box_count": sum(total["class_counts"].values()),
            "split_counts": {split: actual[split]["frame_count"] for split in ratios},
            "split_class_counts": {split: actual[split]["class_counts"] for split in ratios},
            "split_negative_counts": {split: actual[split]["negative_count"] for split in ratios},
            "target_frame_counts": {
                split: total["frame_count"] * ratios[split] for split in ratios
            },
            "group_count": len(groups),
            "linked_group_count": len(components),
        },
        "excluded": candidates["excluded"],
        "warnings": warnings,
        "blockers": blockers,
        "can_freeze": not blockers,
        "exact_duplicates": exact_duplicates,
        "similar_pairs": similar_pairs,
        "similar_pairs_total": near_total,
        "similar_pairs_truncated": near_total > len(similar_pairs),
        "similarity": {
            "method": "dhash64",
            "max_distance": SIMILAR_DISTANCE,
            "pair_limit": MAX_SIMILAR_PAIRS,
        },
    }
