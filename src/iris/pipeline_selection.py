"""Finite selected-object state shared by offline diagnostics and portable inference.

Only current measured observations are candidates. Predictions never renew memory.
Geometry and track numbers provide association evidence, not physical identity.
The machine retains one accepted anchor, one pending observation and one index.
"""

import math
from copy import deepcopy

from .tracking_selection_contracts import validate_policy


def _iou(first, second):
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )
    area1 = (first[2] - first[0]) * (first[3] - first[1])
    area2 = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area1 + area2 - intersection) if intersection else 0.0


def _geometry(anchor, candidate):
    width, height = anchor[2] - anchor[0], anchor[3] - anchor[1]
    width2, height2 = candidate[2] - candidate[0], candidate[3] - candidate[1]
    center = math.hypot(
        (candidate[0] + candidate[2] - anchor[0] - anchor[2]) / 2,
        (candidate[1] + candidate[3] - anchor[1] - anchor[3]) / 2,
    ) / math.hypot(width, height)
    ratio = max(width * height, width2 * height2) / min(width * height, width2 * height2)
    return {"iou": _iou(anchor, candidate), "center_distance": center, "area_ratio": ratio}


def _candidate(anchor, observation, policy):
    geometry = _geometry(anchor["box"], observation["box"])
    reasons = []
    if not observation["confirmed"]:
        reasons.append("unconfirmed")
    if observation["label_id"] != anchor["label_id"]:
        reasons.append("different_class")
    if observation["score"] < policy["min_score"]:
        reasons.append("low_score")
    if geometry["iou"] < policy["min_iou"]:
        reasons.append("insufficient_overlap")
    if geometry["center_distance"] > policy["max_center_distance"]:
        reasons.append("center_distance")
    if geometry["area_ratio"] > policy["max_area_ratio"]:
        reasons.append("area_ratio")
    return {
        "detection_index": observation["detection_index"],
        "track_id": observation["track_id"],
        "eligible": not reasons,
        "reasons": reasons,
        **geometry,
    }


def _anchor(frame, observation):
    return {
        "frame_id": frame["frame_id"],
        "frame_index": frame["frame_index"],
        "timestamp_seconds": frame["timestamp_seconds"],
        "update_index": frame["update_index"],
        "observation": deepcopy(observation),
    }


class SelectionState:
    """One explicit logical selection at a time, independent of ground truth.

    Input frames must have passed validate_tracking_frame. Callers own sequencing
    and poisoning after failed updates. Repeated selections require a preceding
    release or expiration; a new selection starts a new local logical number.
    """

    def __init__(self, policy, mode="guarded_geometry"):
        if mode not in {"guarded_geometry", "track_id_only"}:
            raise ValueError("Unsupported selected-object mode")
        self._policy = validate_policy(policy)
        self._mode = mode
        self.reset()

    def reset(self):
        self._last = self._pending = self._previous = None
        self._state = "idle"
        self._epoch = 0

    def update(self, frame, *, select_detection_index=None, release=False):
        if type(release) is not bool:
            raise ValueError("Release must be an explicit boolean")
        if select_detection_index is not None:
            if type(select_detection_index) is not int or select_detection_index < 0:
                raise ValueError("Selection requires a nonnegative native detection index")
            if release:
                raise ValueError("Select and release cannot occur in the same update")
            if self._state not in {"idle", "released", "expired"}:
                raise ValueError("Release the active selection before selecting another object")
            choices = [
                item
                for item in frame["observations"]
                if item["detection_index"] == select_detection_index
            ]
            if (
                len(choices) != 1
                or not choices[0]["confirmed"]
                or choices[0]["score"] < self._policy["min_score"]
            ):
                raise ValueError("Select a current confirmed measured observation above min_score")
        elif release and self._state == "idle":
            raise ValueError("There is no selected object to release")
        policy, mode = self._policy, self._mode
        last, pending, previous, state = self._last, self._pending, self._previous, self._state
        if select_detection_index is not None:
            self._epoch += 1
        gap = previous is not None and frame["frame_index"] != previous["frame_index"] + 1
        age = {"updates": None, "seconds": None}
        if last is not None:
            age["updates"] = frame["update_index"] - last["update_index"]
            if frame["timestamp_seconds"] is not None and last["timestamp_seconds"] is not None:
                age["seconds"] = frame["timestamp_seconds"] - last["timestamp_seconds"]
        selected, event, checks = None, None, []
        reason = "before_selection"
        if select_detection_index is not None:
            selected = next(
                item
                for item in frame["observations"]
                if item["detection_index"] == select_detection_index
            )
            state, reason, event = "observed", "explicit_selection", "selected"
            pending = None
        elif release:
            state, reason, event = "released", "explicit_release", "released"
            pending = None
        elif state == "released":
            reason = "selection_released"
        elif state == "expired":
            reason = "selection_expired"
        elif last is not None:
            expired_updates = age["updates"] > policy["max_lost_updates"]
            expired_seconds = (
                policy["max_lost_seconds"] is not None
                and age["seconds"] is not None
                and age["seconds"] > policy["max_lost_seconds"] + 1e-12
            )
            if expired_updates or expired_seconds:
                state, reason, event, pending = (
                    "expired",
                    "seconds_limit" if expired_seconds else "update_limit",
                    "expired",
                    None,
                )
            else:
                candidates = []
                for observation in frame["observations"]:
                    if mode == "guarded_geometry":
                        check = _candidate(last["observation"], observation, policy)
                    else:
                        reasons = []
                        for condition, text in (
                            (not observation["confirmed"], "unconfirmed"),
                            (
                                observation["label_id"] != last["observation"]["label_id"],
                                "different_class",
                            ),
                            (
                                observation["track_id"] != last["observation"]["track_id"],
                                "different_track_id",
                            ),
                            (observation["score"] < policy["min_score"], "low_score"),
                        ):
                            if condition:
                                reasons.append(text)
                        check = {
                            "detection_index": observation["detection_index"],
                            "track_id": observation["track_id"],
                            "eligible": not reasons,
                            "reasons": reasons,
                            "iou": None,
                            "center_distance": None,
                            "area_ratio": None,
                        }
                    checks.append(check)
                    if check["eligible"]:
                        candidates.append(observation)
                if not candidates:
                    state, reason, pending = "lost", "no_compatible_observation", None
                elif len(candidates) > 1:
                    state, reason, pending = "ambiguous", "multiple_compatible_observations", None
                else:
                    candidate = candidates[0]
                    continuous = (
                        state in {"observed", "recovered"}
                        and not gap
                        and candidate["track_id"] == last["observation"]["track_id"]
                    )
                    if mode == "track_id_only" or continuous:
                        selected = candidate
                        event = None if continuous else "recovered"
                        state = "observed" if continuous else "recovered"
                        reason, pending = (
                            "continued_observation" if continuous else "same_track_returned",
                            None,
                        )
                    else:
                        coherent = (
                            pending is not None
                            and not gap
                            and candidate["track_id"] == pending["observation"]["track_id"]
                            and _candidate(pending["observation"], candidate, policy)["eligible"]
                        )
                        pending = {
                            "observation": deepcopy(candidate),
                            "count": pending["count"] + 1 if coherent else 1,
                        }
                        if pending["count"] >= policy["recovery_confirmation_updates"]:
                            selected = candidate
                            state, reason, event, pending = (
                                "recovered",
                                "unique_candidate_confirmed",
                                "recovered",
                                None,
                            )
                        else:
                            state, reason = "recovering", "candidate_needs_confirmation"
        if selected is not None:
            last = _anchor(frame, selected)
            age = {"updates": 0, "seconds": 0.0 if frame["timestamp_seconds"] is not None else None}
        output = {
            **{
                key: deepcopy(frame[key])
                for key in (
                    "frame_id",
                    "frame_index",
                    "timestamp_seconds",
                    "input_size",
                    "update_index",
                )
            },
            "state": state,
            "reason": reason,
            "event": event,
            "logical_object_id": f"selection-{self._epoch}" if last is not None else None,
            "selected": deepcopy(selected),
            "last_observed": deepcopy(last),
            "age": age,
            "pending": None
            if pending is None
            else {
                "track_id": pending["observation"]["track_id"],
                "observations": pending["count"],
                "required": policy["recovery_confirmation_updates"],
            },
            "candidates": checks,
            "source_gap": gap,
        }
        self._last, self._pending, self._state = last, pending, state
        # Do not retain prior frame images, detections or growing frame history.
        self._previous = {"frame_index": frame["frame_index"]}
        return output
