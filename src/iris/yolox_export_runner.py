"""Standalone, offline OpenCV CPU runner for the explicit IRIS YOLOX ONNX contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

FORMAT = "iris-yolox-onnx-v1"
MAX_FILE = 256 * 1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")),
    )


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Expected an ONNX manifest object")
    if manifest.get("format") != FORMAT or manifest.get("architecture") != "yolox_nano":
        raise ValueError("Unsupported ONNX bundle format or architecture")
    expected_input = {
        "name": "images",
        "shape": [1, 3, 416, 416],
        "dtype": "float32",
        "color": "BGR",
        "range": [0, 255],
        "letterbox": {"alignment": "top_left", "value": 114, "interpolation": "opencv_linear"},
    }
    classes = manifest.get("classes")
    if not isinstance(classes, list) or not 1 <= len(classes) <= 100:
        raise ValueError("Expected 1–100 explicit YOLOX classes")
    identifiers, categories = set(), set()
    for index, item in enumerate(classes):
        if not isinstance(item, dict):
            raise ValueError("Invalid ONNX class record")
        if (
            type(item.get("index")) is not int
            or item["index"] != index
            or not isinstance(item.get("id"), str)
            or not item["id"]
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or type(item.get("category_id")) is not int
            or item["category_id"] < 1
            or item["id"] in identifiers
            or item["category_id"] in categories
        ):
            raise ValueError("Invalid or duplicate ONNX class mapping")
        identifiers.add(item["id"])
        categories.add(item["category_id"])
    if manifest.get("input") != expected_input or manifest.get("output") != {
        "name": "output",
        "shape": [1, 3549, 5 + len(classes)],
        "encoding": "yolox_raw_grid",
        "strides": [8, 16, 32],
    }:
        raise ValueError("Unsupported preprocessing or raw-output contract")
    model = manifest.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("Invalid ONNX model identity")
    if (
        model.get("path") != "model.onnx"
        or type(model.get("size_bytes")) is not int
        or not 0 < model["size_bytes"] <= MAX_FILE
        or not re.fullmatch(r"[0-9a-f]{64}", model.get("sha256", ""))
    ):
        raise ValueError("Invalid ONNX file identity")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not 4 <= len(files) <= 13:
        raise ValueError("Invalid bundle file inventory")
    for path, identity in files.items():
        if not isinstance(path, str) or not isinstance(identity, dict):
            raise ValueError("Invalid file inventory entry")
        if (
            path.startswith("/")
            or "\\" in path
            or ".." in Path(path).parts
            or path != Path(path).as_posix()
            or path == "manifest.json"
            or type(identity.get("size")) is not int
            or not 0 < identity["size"] <= MAX_FILE
            or not re.fullmatch(r"[0-9a-f]{64}", identity.get("sha256", ""))
        ):
            raise ValueError("Invalid bundle file path or identity")
    if files.get("model.onnx") != {"size": model["size_bytes"], "sha256": model["sha256"]}:
        raise ValueError("ONNX file inventory differs from model identity")
    if not {"run.py", "requirements.txt", "README.md", "parity/reference.json"} <= files.keys():
        raise ValueError("Incomplete standalone ONNX bundle")
    return manifest


def validate_prediction(manifest, prediction):
    if not isinstance(prediction, dict):
        raise ValueError("Expected a prediction object")
    size, detections = prediction.get("input_size"), prediction.get("detections")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(value) is not int or not 1 <= value <= 64000000 for value in size)
        or size[0] * size[1] > 64000000
        or not isinstance(detections, list)
        or len(detections) > 100
    ):
        raise ValueError("Invalid prediction dimensions or detection count")
    classes = {item["index"] + 1: item for item in manifest["classes"]}
    for detection in detections:
        if not isinstance(detection, dict):
            raise ValueError("Invalid detection record")
        slot = detection.get("native_label_id")
        item = classes.get(slot) if type(slot) is int else None
        box, score = detection.get("box"), detection.get("score")
        if (
            item is None
            or detection.get("label") != item["id"]
            or type(detection.get("label_id")) is not int
            or detection["label_id"] != item["category_id"]
            or not isinstance(box, list)
            or len(box) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
            or not 0 <= box[0] < box[2] <= size[0]
            or not 0 <= box[1] < box[3] <= size[1]
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Detection differs from the class/geometry contract")
        taxonomy = manifest.get("taxonomy_id")
        if taxonomy and taxonomy != "iris-objects-v1" and detection.get("taxonomy_id") != taxonomy:
            raise ValueError("Detection taxonomy differs from the bundle")


def inspect_bundle(directory):
    directory = Path(directory).resolve()
    path = directory / "manifest.json"
    if path.is_symlink() or not 0 < path.stat().st_size <= 2 * 1024**2:
        raise ValueError("Invalid manifest file")
    manifest = validate_manifest(read_json(path.read_bytes()))
    for name, identity in manifest["files"].items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory)
            or not path.is_file()
            or path.stat().st_size != identity["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]
        ):
            raise ValueError(f"Bundle file changed: {name}")
    reference = read_json((directory / "parity/reference.json").read_bytes())
    validate_reference(manifest, reference)
    return manifest


def validate_reference(manifest, reference):
    if (
        not isinstance(reference, dict)
        or reference.get("model_id") != manifest["source"]["model_id"]
        or reference.get("weight_sha256") != manifest["source"]["checkpoint_sha256"]
        or not isinstance(reference.get("frames"), list)
        or not 1 <= len(reference["frames"]) <= 8
    ):
        raise ValueError("ONNX reference differs from checkpoint identity")
    seen = set()
    for frame in reference["frames"]:
        if not isinstance(frame, dict) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", frame.get("frame_id", "")
        ):
            raise ValueError("Invalid ONNX reference frame identifier")
        validate_prediction(manifest, frame)
        if (
            frame["frame_id"] in seen
            or frame.get("path") != f"parity/images/{frame['frame_id']}.png"
            or manifest["files"].get(frame["path"], {}).get("sha256") != frame.get("sha256")
        ):
            raise ValueError("ONNX reference image identity differs from inventory")
        seen.add(frame["frame_id"])


class Runner:
    def __init__(self, directory):
        import cv2
        import numpy as np

        self.directory = Path(directory)
        self.manifest = inspect_bundle(directory)
        self.cv2, self.np = cv2, np
        cv2.setNumThreads(2)
        self.net = cv2.dnn.readNetFromONNX(
            np.frombuffer((self.directory / "model.onnx").read_bytes(), dtype=np.uint8)
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        grids, strides = [], []
        for stride in (8, 16, 32):
            yy, xx = np.mgrid[: 416 // stride, : 416 // stride]
            grids.append(np.column_stack((xx.ravel(), yy.ravel())))
            strides.append(np.full((xx.size, 1), stride))
        self.grid, self.strides = np.concatenate(grids), np.concatenate(strides)

    def predict(self, path):
        from PIL import Image, ImageOps

        np, cv2 = self.np, self.cv2
        start = time.perf_counter()
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            if width * height > 64_000_000:
                raise ValueError("Image exceeds 64 million pixels")
            bgr = np.asarray(image)[:, :, ::-1]
        decoded = time.perf_counter()
        ratio = min(416 / height, 416 / width)
        tensor = np.full((416, 416, 3), 114, dtype=np.float32)
        resized = cv2.resize(
            bgr,
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor[: resized.shape[0], : resized.shape[1]] = resized
        self.net.setInput(np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]))
        raw = self.net.forward("output")
        if raw.shape != tuple(self.manifest["output"]["shape"]) or not np.isfinite(raw).all():
            raise ValueError("Invalid native ONNX output")
        native = raw[0].copy()
        native[:, :2] = (native[:, :2] + self.grid) * self.strides
        native[:, 2:4] = np.exp(native[:, 2:4]) * self.strides
        boxes = (
            np.column_stack(
                (native[:, :2] - native[:, 2:4] / 2, native[:, :2] + native[:, 2:4] / 2)
            )
            / ratio
        )
        if not np.isfinite(boxes).all():
            raise ValueError("Non-finite decoded ONNX boxes")
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        scores = native[:, 4:5] * native[:, 5:]
        best_class = scores.argmax(axis=1)
        candidates = []
        for item in self.manifest["classes"]:
            values = scores[:, item["index"]]
            eligible = np.flatnonzero(
                (values >= 0.001)
                & (best_class == item["index"])
                & (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
            )
            order = eligible[np.argsort(-values[eligible], kind="stable")]
            while order.size:
                first = int(order[0])
                candidates.append((float(values[first]), first, item))
                remaining = order[1:]
                low = np.maximum(boxes[first, :2], boxes[remaining, :2])
                high = np.minimum(boxes[first, 2:], boxes[remaining, 2:])
                overlap = np.maximum(0, high - low).prod(axis=1)
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                iou = overlap / np.maximum(area[first] + area[remaining] - overlap, 1e-12)
                order = remaining[iou <= 0.5]
        detections = []
        for score, index, item in sorted(candidates, key=lambda x: -x[0])[:100]:
            detection = {
                "label": item["id"],
                "label_id": item["category_id"],
                "native_label_id": item["index"] + 1,
                "box": [float(x) for x in boxes[index]],
                "score": score,
            }
            taxonomy_id = self.manifest.get("taxonomy_id")
            if taxonomy_id:
                detection["taxonomy_id"] = taxonomy_id
            detections.append(detection)
        finished = time.perf_counter()
        return {
            "input_size": [width, height],
            "detections": detections,
            "timing": {
                "total_ms": (finished - decoded) * 1000,
                "decode_ms": (decoded - start) * 1000,
            },
        }


def validate_measurement(manifest, reference, payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a measurement object")
    if (
        payload.get("format") != "iris-yolox-measurement-v1"
        or payload.get("manifest_sha256") != digest(manifest)
        or payload.get("device") != "cpu"
        or type(payload.get("repeats")) is not int
        or not 1 <= payload["repeats"] <= 10
    ):
        raise ValueError("Measurement does not match this CPU ONNX bundle")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or len(value) > 2048
        for key, value in environment.items()
    ):
        raise ValueError("Measurement environment must contain bounded text fields")
    expected = [
        (repeat, frame) for repeat in range(payload["repeats"]) for frame in reference["frames"]
    ]
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != len(expected):
        raise ValueError("Measurement sample inventory differs from references")
    passed, timings = True, []
    for sample, (repeat, frame) in zip(samples, expected, strict=True):
        if not isinstance(sample, dict):
            raise ValueError("Invalid measurement sample")
        prediction = sample.get("prediction", {})
        validate_prediction(manifest, prediction)
        if sample.get("frame_id") != frame["frame_id"] or sample.get("repeat") != repeat:
            raise ValueError("Measurement sample identity/order changed")
        for key in ("total_ms", "decode_ms"):
            value = prediction.get("timing", {}).get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid measurement timing")
        if not isinstance(prediction.get("detections"), list):
            raise ValueError("Invalid measurement predictions")
        canonical(prediction)
        passed &= all(prediction.get(key) == frame[key] for key in ("input_size", "detections"))
        timings.append(prediction["timing"]["total_ms"])
    return {
        "parity_passed": passed,
        "sample_count": len(samples),
        "device": "cpu",
        "mean_total_ms": sum(timings) / len(timings),
        "execution_verified": False,
        "evidence_kind": payload.get("evidence_kind", "declared_execution"),
        "parity_protocol": "Exact saved native PyTorch outputs versus OpenCV ONNX outputs",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("inspect")
    predict = actions.add_parser("predict")
    predict.add_argument("image", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    measure = actions.add_parser("measure")
    measure.add_argument("--repeats", type=int, default=3, choices=range(1, 11))
    measure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = inspect_bundle(args.bundle)
    if args.command == "inspect":
        print(json.dumps(manifest, indent=2))
        return 0
    if args.output.resolve().is_relative_to(args.bundle.resolve()):
        raise ValueError("Write results outside the immutable bundle")
    runner = Runner(args.bundle)
    if args.command == "predict":
        payload = runner.predict(args.image)
        code = 0
    else:
        reference = read_json((args.bundle / "parity/reference.json").read_bytes())
        runner.predict(args.bundle / reference["frames"][0]["path"])
        samples = [
            {
                "frame_id": frame["frame_id"],
                "repeat": repeat,
                "prediction": runner.predict(args.bundle / frame["path"]),
            }
            for repeat in range(args.repeats)
            for frame in reference["frames"]
        ]
        payload = {
            "format": "iris-yolox-measurement-v1",
            "manifest_sha256": digest(manifest),
            "device": "cpu",
            "repeats": args.repeats,
            "samples": samples,
            "environment": {
                "python": platform.python_version(),
                "opencv": runner.cv2.__version__,
                "platform": platform.platform(),
            },
            "evidence_kind": "declared_execution",
        }
        summary = validate_measurement(manifest, reference, payload)
        payload["summary"] = summary
        code = 0 if summary["parity_passed"] else 3
    with args.output.open("x") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        print(f"ONNX runner failed: {exc}", file=sys.stderr)
        sys.exit(2)
