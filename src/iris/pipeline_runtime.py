"""Bounded standalone detector → native tracker → explicit selected-object pipeline.

No project database, radio or application control enters this interface. A failed
update poisons the sequence until reset, including failures before native state
changes. Upstream tracker history is bounded by a hard per-sequence update limit.
"""

from __future__ import annotations

import hashlib
import math
import sys
import threading
from copy import deepcopy
from pathlib import Path

from .pipeline_bundle_contracts import canonical
from .pipeline_selection import SelectionState
from .tracking_contracts import validate_profile, validate_tracking_frame, validate_update_input

FRAME_SCHEMA = "iris-pipeline-frame-v1"
MAX_UPDATES = 10_000
MAX_IMAGE_PIXELS = 64 * 1024**2
CLOCK_KINDS = {"provided", "nominal_fps", "unknown"}


class PipelineError(RuntimeError):
    """The current sequence cannot continue; an explicit reset is required."""


def _text(value, name):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 160
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} must be bounded nonempty text")
    value.encode("utf-8")
    return value


def _integer(value, name):
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"{name} must be a nonnegative JSON integer")
    return value


class Pipeline:
    """Run a verified v2 directory with explicit source clock and selection events.

    Factories are private test seams, never part of the portable CLI or manifest.
    Normal execution verifies that imported package code is the exact inspected
    directory, then separately records actual runtime and historical provenance.
    """

    def __init__(
        self,
        bundle_directory,
        manifest=None,
        device=None,
        detector_factory=None,
        tracker_factory=None,
    ):
        from .pipeline_bundle_contracts import validate_directory

        self._lock = threading.RLock()
        self._directory = Path(bundle_directory)
        verified = validate_directory(self._directory, expected_manifest=manifest)
        self._manifest = deepcopy(verified["manifest"])
        if self._manifest["format"] != "iris-pipeline-bundle-v2":
            raise ValueError("Pipeline execution requires a v2 bundle; v1 supports inspection only")
        self._manifest_sha256 = verified["manifest_sha256"]
        self._directory = self._directory.resolve()
        self._profile = validate_profile(self._manifest["tracker"]["profile"])
        self._testing = detector_factory is not None or tracker_factory is not None
        if not self._testing:
            self._verify_loaded_code()
        if detector_factory is None:
            from .pipeline_detector_runtime import Detector

            detector_factory = Detector
        if tracker_factory is None:
            from .tracking import make_tracker

            tracker_factory = make_tracker
        self._detector = detector_factory(self._directory, deepcopy(self._manifest), device=device)
        self._tracker = tracker_factory(deepcopy(self._profile))
        selection = self._manifest["selection"]
        self._selection = SelectionState(selection["policy"]) if selection is not None else None
        self._sequence_id = None
        self._clock_kind = None
        self._updates = 0
        self._last = None
        self._frame_ids = set()
        self._dimensions = None
        self._poisoned = False
        self._closed = False
        self.verify_runtime()

    def _verify_loaded_code(self):
        prefix = __package__ + "."
        for name, module in tuple(sys.modules.items()):
            if name != __package__ and not name.startswith(prefix):
                continue
            filename = getattr(module, "__file__", None)
            if not filename:
                raise PipelineError("Portable runtime module has no verifiable source file")
            source = Path(filename)
            if source.is_symlink() or not source.resolve().is_relative_to(self._directory):
                raise PipelineError("Portable runtime imported package code outside its bundle")
            relative = source.resolve().relative_to(self._directory).as_posix()
            identity = self._manifest["files"].get(relative)
            if (
                identity is None
                or not source.is_file()
                or source.stat().st_size != identity["size"]
                or hashlib.sha256(source.read_bytes()).hexdigest() != identity["sha256"]
            ):
                raise PipelineError("Imported runtime code differs from the manifest inventory")

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    @property
    def metadata(self):
        selection = self._manifest["selection"]
        value = {
            "schema": "iris-pipeline-runtime-v1",
            "bundle_manifest_sha256": self._manifest_sha256,
            "detector": self._detector.metadata,
            "tracker": self._tracker.metadata,
            "selection": None
            if selection is None
            else {
                "algorithm": selection["algorithm"],
                "policy_sha256": selection["policy_sha256"],
                "code_sha256": self._manifest["files"]["iris_bundle/pipeline_selection.py"][
                    "sha256"
                ],
                "identity_claim": "association_evidence_not_physical_identity",
            },
            "code_sha256": {
                path: entry["sha256"]
                for path, entry in self._manifest["files"].items()
                if path.startswith("iris_bundle/") and path.endswith(".py")
            },
            "limits": {"max_updates_per_reset": MAX_UPDATES, "max_image_pixels": MAX_IMAGE_PIXELS},
            "clock_kind": self._clock_kind,
            "independent_quality": "not_qualified",
        }
        canonical(value)
        return deepcopy(value)

    def verify_runtime(self):
        """Check the immutable package and loaded dependencies, without qualification."""
        from .pipeline_bundle_contracts import validate_directory

        with self._lock:
            try:
                if self._closed:
                    raise PipelineError("Pipeline is closed")
                verified = validate_directory(self._directory, expected_manifest=self._manifest)
                if verified["manifest_sha256"] != self._manifest_sha256:
                    raise PipelineError("Portable manifest changed during execution")
                if not self._testing:
                    self._verify_loaded_code()
                self._detector.verify_runtime()
                self._tracker.verify_runtime()
                canonical(self.metadata)
            except BaseException:
                self._poisoned = True
                raise

    def reset(self, sequence_id, clock_kind="provided"):
        with self._lock:
            try:
                _text(sequence_id, "Sequence ID")
                if not isinstance(clock_kind, str) or clock_kind not in CLOCK_KINDS:
                    raise ValueError("Clock kind must be provided, nominal_fps or unknown")
                self.verify_runtime()
                self._tracker.reset(sequence_id)
                if self._selection is not None:
                    self._selection.reset()
                self._sequence_id, self._clock_kind = sequence_id, clock_kind
                self._updates = 0
                self._last = self._dimensions = None
                self._frame_ids = set()
                self._poisoned = False
            except BaseException:
                self._poisoned = True
                raise

    def _source(self, image, frame_id, frame_index, timestamp_seconds, selected, release):
        from PIL import Image

        if self._closed:
            raise PipelineError("Pipeline is closed")
        if self._poisoned:
            raise PipelineError("Pipeline sequence failed; reset before another update")
        if self._sequence_id is None:
            raise PipelineError("Reset the pipeline before the first update")
        if self._updates >= MAX_UPDATES:
            raise PipelineError("Pipeline update limit reached; begin a new explicit sequence")
        _text(frame_id, "Frame ID")
        _integer(frame_index, "Source frame index")
        if frame_id in self._frame_ids:
            raise ValueError("Frame IDs cannot repeat within a sequence")
        if self._clock_kind == "unknown":
            if timestamp_seconds is not None:
                raise ValueError("Unknown source clocks require null timestamps")
        elif (
            type(timestamp_seconds) not in (int, float)
            or not math.isfinite(timestamp_seconds)
            or timestamp_seconds < 0
        ):
            raise ValueError("Known source clocks require finite nonnegative timestamps")
        if self._last is not None:
            if frame_index <= self._last["frame_index"]:
                raise ValueError("Source frame indices must strictly increase")
            if (
                timestamp_seconds is not None
                and timestamp_seconds <= self._last["timestamp_seconds"]
            ):
                raise ValueError("Source timestamps must strictly increase")
        if type(release) is not bool:
            raise ValueError("Release must be an explicit boolean")
        if selected is not None:
            _integer(selected, "Selected detection index")
            if release:
                raise ValueError("Select and release cannot occur in the same update")
        if (selected is not None or release) and self._selection is None:
            raise ValueError("This bundle does not contain selected-object settings")
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("Pipeline input must be an oriented RGB Pillow image")
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Apply EXIF orientation before supplying pipeline image coordinates")
        size = list(image.size)
        if min(size) <= 0 or size[0] * size[1] > MAX_IMAGE_PIXELS:
            raise ValueError("Pipeline image dimensions exceed the supported limit")
        if self._dimensions is not None and size != self._dimensions:
            raise ValueError("Image dimensions must stay constant until sequence reset")
        return {
            "frame_id": frame_id,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "input_size": size,
        }

    def _detections(self, value, source):
        if (
            not isinstance(value, dict)
            or set(value) != {"input_size", "detections", "native_detection_count", "timing"}
            or value["input_size"] != source["input_size"]
        ):
            raise ValueError("Detector returned an invalid or differently oriented frame")
        _integer(value["native_detection_count"], "Native detection count")
        if value["native_detection_count"] > 100:
            raise ValueError("Detector exceeded its frozen native output cap")
        timing = value["timing"]
        if (
            not isinstance(timing, dict)
            or set(timing) != {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"}
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in timing.values()
            )
        ):
            raise ValueError("Detector timings must be finite nonnegative measurements")
        entries = {
            row["output_id"]: row for row in self._manifest["detector"]["output_mapping"]["entries"]
        }
        # Validate every detector output, including classes not requested by the tracker.
        full_profile = {**self._profile, "class_ids": sorted(entries)}
        checked = validate_update_input({**source, "detections": value["detections"]}, full_profile)
        for row in checked["detections"]:
            if (
                row["detection_index"] >= value["native_detection_count"]
                or row["label"] != entries[row["label_id"]]["label"]
                or row["score"] < self._manifest["detector"]["config"]["min_score"]
            ):
                raise ValueError("Detector output violates its frozen mapping or native indices")
        classes = set(self._profile["class_ids"])
        return {
            **source,
            "detections": [row for row in checked["detections"] if row["label_id"] in classes],
        }

    def update(
        self,
        image,
        *,
        frame_id,
        frame_index,
        timestamp_seconds,
        select_detection_index=None,
        release=False,
    ):
        with self._lock:
            try:
                source = self._source(
                    image, frame_id, frame_index, timestamp_seconds, select_detection_index, release
                )
                detected = self._detector.predict(image)
                tracking_source = self._detections(detected, source)
                pixels = None
                if self._profile["gmc_method"] != "none":
                    import numpy as np

                    pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[:, :, ::-1])
                tracked = validate_tracking_frame(
                    self._tracker.update(tracking_source, image=pixels),
                    tracking_source,
                    self._profile,
                )
                if (
                    tracked["sequence_id"] != self._sequence_id
                    or tracked["update_index"] != self._updates + 1
                ):
                    raise ValueError("Tracker sequence or update count differs from pipeline state")
                selected = (
                    None
                    if self._selection is None
                    else self._selection.update(
                        tracked, select_detection_index=select_detection_index, release=release
                    )
                )
                output = {
                    "schema": FRAME_SCHEMA,
                    "sequence_id": self._sequence_id,
                    **source,
                    "update_index": self._updates + 1,
                    "detector": detected,
                    "tracking": tracked,
                    "selection": selected,
                }
                canonical(output)
                self._updates += 1
                self._frame_ids.add(frame_id)
                self._last = {"frame_index": frame_index, "timestamp_seconds": timestamp_seconds}
                self._dimensions = source["input_size"]
                return deepcopy(output)
            except BaseException:
                self._poisoned = True
                raise

    def close(self):
        with self._lock:
            self._closed = True
            self._poisoned = True
            # Detach model/native history; no caller-held result aliases internal state.
            self._detector = self._tracker = self._selection = None
