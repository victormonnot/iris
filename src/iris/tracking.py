"""Class-isolated native trackers over saved detections, independent of the workspace.

Only construction imports optional tracking dependencies. Updates consume actual
available frames; timestamps describe evidence, not the native Kalman time step.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from iris.tracking_contracts import (
    validate_profile,
    validate_tracking_frame,
    validate_update_input,
)

_NATIVE_LOCK = threading.RLock()
OPTIONAL_PACKAGES = {"scipy": "1.17.1", "lap": "0.5.12", "cython_bbox": "0.1.5"}


class TrackingError(RuntimeError):
    """Native state is no longer reusable; reset before another sequence replay."""


def tracking_status():
    """Inspect installed package metadata without importing the optional runtimes."""
    packages = {}
    for name, required in OPTIONAL_PACKAGES.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        packages[name] = {
            "required": required,
            "installed": installed,
            "ready": installed == required,
        }
    return {
        "available": all(item["ready"] for item in packages.values()),
        "packages": packages,
        "installation": "Install the optional tracking extra in the environment running IRIS",
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(algorithm):
    directory = Path(__file__).parent
    vendor = directory / "_vendor"
    manifest = json.loads((vendor / "tracking-provenance.json").read_text())
    record = manifest[algorithm]
    for name, entry in record["files"].items():
        if _sha(vendor / algorithm / name) != entry["vendored_sha256"]:
            raise TrackingError("Bundled tracker source changed; restore the pinned implementation")
    for filename, key in (
        ("LICENSE", "license_sha256"),
        ("adaptations.patch", "adaptations_sha256"),
    ):
        if _sha(vendor / algorithm / filename) != record[key]:
            raise TrackingError("Bundled tracker provenance changed")
    return {
        **record,
        "manifest_sha256": _sha(vendor / "tracking-provenance.json"),
        "adapter_sha256": {
            name: _sha(directory / name) for name in ("tracking.py", "tracking_contracts.py")
        },
    }


class _FixedWarp:
    """One camera estimate per sequence update, shared by independent class trackers."""

    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, image, detections=None):
        return self.matrix


class TrackingAdapter:
    def __init__(self, profile):
        self._profile = validate_profile(profile)
        status = tracking_status()
        if not status["available"]:
            raise ImportError(
                status["installation"] + ": scipy, lap and cython_bbox versions must match"
            )
        provenance = _provenance(self._profile["algorithm"])
        # Import errors remain explicit; no approximate replacement is used.
        import cv2
        import numpy as np

        if self._profile["algorithm"] == "bytetrack":
            from iris._vendor.bytetrack.basetrack import BaseTrack, TrackState
            from iris._vendor.bytetrack.byte_tracker import BYTETracker

            self._constructor = BYTETracker
        else:
            from iris._vendor.botsort.basetrack import BaseTrack, TrackState
            from iris._vendor.botsort.bot_sort import BoTSORT
            from iris._vendor.botsort.gmc import GMC

            self._constructor = BoTSORT
            self._gmc_constructor = GMC
        self._np, self._cv2 = np, cv2
        self._base, self._states = BaseTrack, TrackState
        self._guard = threading.RLock()
        self._sequence_id = None
        self._metadata = {
            "schema": "iris-tracker-runtime-v1",
            "algorithm": self._profile["algorithm"],
            "provenance": provenance,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (*OPTIONAL_PACKAGES, "numpy", "opencv-python-headless")
            },
            "execution_policy": {
                "device": "cpu",
                "learned_reid": False,
                "class_association": "independent_native_instance_per_class",
                "native_frame_rate_argument": 30,
                "native_buffer_unit": "available_frame_updates",
                "native_time_step": 1,
                "skipped_source_frames": "no_synthetic_updates",
                "native_high_and_low_comparison": "strict_greater_than",
                "native_second_pass_match_threshold": 0.5,
                "native_unconfirmed_match_threshold": 0.7,
                "native_duplicate_iou_distance": 0.15,
                "native_expiry_order": "association_before_lost_cleanup_not_a_hard_ttl",
                "native_iou_geometry": "cython_bbox_inclusive_pixel_overlap",
                "unconfirmed_returned": self._profile["algorithm"] == "botsort",
                "gmc_scope": "one_sequence_estimate_shared_across_classes",
                "gmc_failure": "raise_and_require_reset_no_silent_fallback",
                "gmc_insufficient_matches": "native_identity_transform_reported",
                "opencv_threads": self._profile["opencv_threads"],
                "opencv_rng": "seed_plus_update_index_before_each_update_no_rng_restore_api",
                "seed": self._profile["seed"],
                "blas_environment": {
                    name: os.environ.get(name)
                    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                },
                "repeatability": "measure_semantic_hashes_on_repeated_fresh_replays",
            },
        }

    @property
    def profile(self):
        return deepcopy(self._profile)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def verify_runtime(self):
        """Refuse to certify a replay after its installed implementation changed."""
        with self._guard:
            packages = {
                name: importlib.metadata.version(name) for name in self._metadata["packages"]
            }
            environment = {
                name: os.environ.get(name)
                for name in self._metadata["execution_policy"]["blas_environment"]
            }
            if (
                _provenance(self._profile["algorithm"]) != self._metadata["provenance"]
                or packages != self._metadata["packages"]
                or environment != self._metadata["execution_policy"]["blas_environment"]
            ):
                self._poisoned = True
                raise TrackingError("Tracking implementation or runtime changed during replay")

    def reset(self, sequence_id):
        if (
            not isinstance(sequence_id, str)
            or not sequence_id.strip()
            or len(sequence_id) > 160
            or any(ord(char) < 32 for char in sequence_id)
        ):
            raise ValueError("Tracking requires a bounded sequence identity")
        with self._guard, _NATIVE_LOCK:
            self.verify_runtime()
            profile = self._profile
            args = SimpleNamespace(
                track_thresh=profile["high_threshold"],
                track_high_thresh=profile["high_threshold"],
                track_low_thresh=profile["low_threshold"],
                new_track_thresh=profile["new_track_threshold"],
                track_buffer=profile["buffer_updates"],
                match_thresh=profile["match_threshold"],
                mot20=not profile["fuse_score"],
                proximity_thresh=0.5,  # Not used without learned ReID.
                appearance_thresh=0.25,  # Not used without learned ReID.
                with_reid=False,
                cmc_method="none",  # Shared GMC is evaluated once outside class association.
                name="iris-temporal",
                ablation=False,
            )
            saved_count = self._base._count
            try:
                self._native = {
                    label: self._constructor(args, frame_rate=30) for label in profile["class_ids"]
                }
            finally:
                self._base._count = saved_count
            self._gmc = (
                self._gmc_constructor(
                    method=profile["gmc_method"], downscale=profile["gmc_downscale"]
                )
                if profile["algorithm"] == "botsort"
                else None
            )
            self._sequence_id = sequence_id
            self._id_count = self._updates = 0
            self._last_frame = self._dimensions = None
            self._labels, self._last_observed = {}, {}
            self._seen_frame_ids = set()
            self._poisoned = False

    def _check_order(self, frame, image):
        if self._sequence_id is None:
            raise TrackingError("Reset the adapter with its sequence ID before the first update")
        if self._poisoned:
            raise TrackingError("This tracker failed; reset before replaying the sequence")
        if frame["frame_id"] in self._seen_frame_ids:
            raise ValueError("Source frame identities must be unique within a sequence")
        dimensions = tuple(frame["input_size"])
        if self._dimensions is not None and self._dimensions != dimensions:
            raise ValueError("Image dimensions changed; start a new sequence")
        if self._last_frame is not None:
            previous = self._last_frame
            if (
                frame["frame_id"] == previous["frame_id"]
                or frame["frame_index"] <= previous["frame_index"]
            ):
                raise ValueError("Tracking frames must follow strictly increasing source indices")
            current_at, previous_at = frame["timestamp_seconds"], previous["timestamp_seconds"]
            if (current_at is None) != (previous_at is None) or (
                current_at is not None and current_at <= previous_at
            ):
                raise ValueError(
                    "Known tracking timestamps must increase; clock availability cannot change"
                )
        labels = dict(self._labels)
        for detection in frame["detections"]:
            label_id, label = detection["label_id"], detection["label"]
            if label_id in labels and labels[label_id] != label:
                raise ValueError("Detector class names changed within this sequence")
            labels[label_id] = label
        if self._profile["gmc_method"] != "none":
            width, height = dimensions
            if (
                not isinstance(image, self._np.ndarray)
                or image.dtype != self._np.uint8
                or image.shape != (height, width, 3)
                or min(dimensions) < self._profile["gmc_downscale"]
            ):
                raise ValueError(
                    "Camera compensation needs original uint8 BGR pixels at source dimensions"
                )
        return labels

    def update(self, frame, *, image=None):
        with self._guard:
            started = time.perf_counter()
            source = validate_update_input(frame, self._profile)
            labels = self._check_order(source, image)
            with _NATIVE_LOCK:
                old_count, old_threads = self._base._count, self._cv2.getNumThreads()
                try:
                    self._cv2.setNumThreads(self._profile["opencv_threads"])
                    self._cv2.setRNGSeed((self._profile["seed"] + self._updates) % 2**31)
                    self._base._count = self._id_count
                    result = self._update(source, image, labels, started)
                    self._id_count = self._base._count
                    checked = validate_tracking_frame(result, source, self._profile)
                except Exception as exc:
                    self._poisoned = True
                    raise TrackingError(
                        f"{self._profile['algorithm']} failed on frame {source['frame_index']}; "
                        "reset before replaying the sequence"
                    ) from exc
                finally:
                    self._base._count = old_count
                    self._cv2.setNumThreads(old_threads)
            self._last_frame = source
            self._seen_frame_ids.add(source["frame_id"])
            self._dimensions = tuple(source["input_size"])
            self._labels = labels
            self._updates += 1
            return checked

    def _update(self, source, image, labels, started):
        np, profile = self._np, self._profile
        width, height = source["input_size"]
        camera_start = time.perf_counter()
        matrix = np.eye(2, 3)
        camera_status = "disabled"
        if profile["gmc_method"] != "none":
            matrix = self._gmc.apply(image)
            camera_status = self._gmc.last_status
            if (
                not isinstance(matrix, np.ndarray)
                or matrix.shape != (2, 3)
                or not np.isfinite(matrix).all()
            ):
                raise ValueError(
                    "Native camera compensation did not return a finite affine transform"
                )
        gmc_ms = (time.perf_counter() - camera_start) * 1000
        camera = {
            "method": profile["gmc_method"],
            "status": camera_status,
            "matrix": matrix.tolist() if camera_status != "disabled" else None,
            "downscale": profile["gmc_downscale"],
        }
        association_start = time.perf_counter()
        by_class = {label: [] for label in profile["class_ids"]}
        for detection in source["detections"]:
            by_class[detection["label_id"]].append(detection)
        observations, predictions, unassigned = [], [], []
        for label_id, native in self._native.items():
            detections = by_class[label_id]
            measurements = np.asarray(
                [[*row["box"], row["score"]] for row in detections], dtype=np.float64
            ).reshape((-1, 5))
            if profile["algorithm"] == "bytetrack":
                output = native.update(measurements, (height, width), (height, width))
            else:
                native.gmc = _FixedWarp(matrix)
                output = native.update(measurements, image)
            # Keep exact source evidence for both returned and unconfirmed observations.
            for track in native.tracked_stracks:
                if track.frame_id == native.frame_id:
                    self._detection(track, detections)
                    self._last_observed[int(track.track_id)] = {
                        "last_observed_frame_id": source["frame_id"],
                        "last_observed_frame_index": source["frame_index"],
                        "last_observed_timestamp_seconds": source["timestamp_seconds"],
                        "last_observed_update_index": self._updates + 1,
                    }
            assigned = set()
            for track in output:
                if track.frame_id != native.frame_id or track.state != self._states.Tracked:
                    raise ValueError(
                        "Native output tried to present a prediction as an observation"
                    )
                detection = self._detection(track, detections)
                index = detection["detection_index"]
                if index in assigned:
                    raise ValueError("Native association assigned one measurement more than once")
                assigned.add(index)
                observations.append(
                    {
                        **deepcopy(detection),
                        "track_id": int(track.track_id),
                        "confirmed": bool(track.is_activated),
                        "estimated_box": self._box(track),
                    }
                )
            pending = {
                self._detection(track, detections)["detection_index"]
                for track in native.tracked_stracks
                if track.frame_id == native.frame_id and not track.is_activated
            }
            for detection in detections:
                if detection["detection_index"] not in assigned:
                    unassigned.append(
                        {
                            **deepcopy(detection),
                            "reason": self._unassigned_reason(detection, pending),
                        }
                    )
            for track in native.lost_stracks:
                if track.state != self._states.Lost:
                    continue
                last = self._last_observed[int(track.track_id)]
                elapsed = (
                    None
                    if source["timestamp_seconds"] is None
                    else source["timestamp_seconds"] - last["last_observed_timestamp_seconds"]
                )
                predictions.append(
                    {
                        "track_id": int(track.track_id),
                        "label_id": label_id,
                        "label": labels[label_id],
                        "box": self._box(track),
                        "confirmed": bool(track.is_activated),
                        **last,
                        "age_updates": self._updates + 1 - last["last_observed_update_index"],
                        "age_seconds": elapsed,
                    }
                )
        association_ms = (time.perf_counter() - association_start) * 1000
        return {
            "schema": "iris-tracking-frame-v1",
            "sequence_id": self._sequence_id,
            **{
                key: deepcopy(source[key])
                for key in ("frame_id", "frame_index", "timestamp_seconds", "input_size")
            },
            "update_index": self._updates + 1,
            "observations": sorted(observations, key=lambda row: row["detection_index"]),
            "predictions": sorted(predictions, key=lambda row: row["track_id"]),
            "unassigned": sorted(unassigned, key=lambda row: row["detection_index"]),
            "gmc": camera,
            "timing": {
                "gmc_ms": gmc_ms,
                "association_ms": association_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        }

    @staticmethod
    def _detection(track, detections):
        index = track.detection_index
        if type(index) is not int or not 0 <= index < len(detections):
            raise ValueError("Native observation lost its exact input detection index")
        return detections[index]

    def _box(self, track):
        box = self._np.asarray(track.tlbr)
        if (
            box.shape != (4,)
            or not self._np.isfinite(box).all()
            or not (box[0] < box[2] and box[1] < box[3])
        ):
            raise ValueError("Native estimated box is not finite with positive area")
        return box.tolist()

    def _unassigned_reason(self, detection, pending):
        score, profile = detection["score"], self._profile
        if detection["detection_index"] in pending:
            return "native_unconfirmed"
        if score <= profile["low_threshold"]:
            return "below_low_threshold"
        if score == profile["high_threshold"]:
            return "strict_high_boundary"
        if score < profile["high_threshold"]:
            return "unmatched_low_confidence"
        if score < profile["new_track_threshold"]:
            return "below_birth_threshold"
        return "native_suppressed"


def make_tracker(profile):
    """Construct the explicitly requested native adapter; no weights or network."""
    return TrackingAdapter(profile)
