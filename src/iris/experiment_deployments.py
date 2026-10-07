"""Project saved standalone measurements into self-contained report evidence."""

from __future__ import annotations

from copy import deepcopy
from statistics import median

from iris import model_exports
from iris.model_taxonomy import class_contract

PROTOCOL = "iris-experiment-deployments-v1"
MAX_MEASUREMENTS = 4
MAX_AVAILABLE_MEASUREMENTS = 100
CUDA_FIELDS = (
    "runtime",
    "cudnn",
    "index",
    "name",
    "capability",
    "total_memory",
    "tf32_matmul",
    "tf32_cudnn",
    "cudnn_benchmark",
)
LIMITATIONS = [
    "Imported measurements are declarations; IRIS has not authenticated execution on the target.",
    "Export total time excludes image decoding, model loading and warmup. IRIS evaluation "
    "total time includes decoding. No cross-context speedup is calculated.",
    "Parity covers only the selected saved reference images, not general model quality "
    "or continuous video performance.",
]


def snapshot(measurements=()):
    return {
        "protocol": PROTOCOL,
        "measurements": deepcopy(list(measurements)),
        "limitations": list(LIMITATIONS),
    }


def _measurement(export, item, report, detail):
    """Validate saved JSON and its precise evaluation lane without loading a detector."""
    manifest, config = export["manifest"], export["config"]
    source, model = manifest["source"], manifest["model"]
    yolox = manifest.get("format") == "iris-yolox-onnx-v1"
    if yolox:
        from iris.yolox_exports import manifest as onnx_manifest

        model = config["model"]  # Checkpoint identity remains distinct from the ONNX graph.
        expected_manifest = onnx_manifest(
            config, export["id"], export["created_at"], manifest["files"], manifest["validation"]
        )
    else:
        expected_manifest = model_exports._manifest(config, export["id"], export["created_at"])
    lane = next(lane for lane in report["lanes"] if lane["id"] == source["evaluation_model_id"])
    metadata = next(row["metadata"] for row in detail["models"] if row["id"] == lane["id"])
    if (
        not export["path"]
        or source["evaluation_id"] != report["evaluation"]["id"]
        or export["evaluation_id"] != source["evaluation_id"]
        or source["dataset_id"] != report["dataset"]["id"]
        or source["dataset_manifest_sha256"] != report["dataset"]["manifest_sha256"]
        or export["trained_model_id"] != lane["model_id"]
        or model["id"] != lane["model_id"]
        or model["sha256"] != lane["weight_sha256"]
        or model["architecture"] != metadata["architecture"]
        or model["class_contract"] != class_contract(metadata)
        or source.get("reference_device", "cpu") != metadata["device"]
        or config["evaluation_metadata_sha256"] != model_exports._digest(metadata)
        or lane["variant"] != "full"
        or manifest != expected_manifest
        or model_exports._digest(manifest) != export["manifest_sha256"]
        or item["export_id"] != export["id"]
    ):
        raise ValueError("Standalone measurement does not match this evaluated checkpoint and lane")
    reference = config["reference"]
    frames = {frame["frame_id"]: frame for frame in detail["frames"]}
    predictions = {
        row["frame_id"]: row
        for row in detail["predictions"]
        if row["evaluation_model_id"] == lane["id"]
    }
    for frame in reference["frames"]:
        saved, prediction = frames[frame["frame_id"]], predictions[frame["frame_id"]]
        if (
            frame["sha256"] != saved["image_file_sha256"]
            or frame["input_size"] != prediction["input_size"]
            or frame["detections"] != prediction["detections"]
        ):
            raise ValueError("Standalone reference differs from the saved evaluated images")
    payload = item["payload"]
    if (
        len(model_exports._canonical(payload)) > model_exports.MAX_MEASUREMENT_BYTES
        or model_exports._digest(payload) != item["fingerprint"]
    ):
        raise ValueError("Saved standalone measurement no longer matches its checksum")
    if yolox:
        from iris.yolox_export_runner import validate_measurement

        summary = validate_measurement(manifest, reference, payload)
    else:
        summary = model_exports._runtime().validate_measurement(manifest, reference, payload)
    if summary != item["summary"]:
        raise ValueError("Saved standalone measurement summary is inconsistent")
    if yolox:
        summary = _onnx_summary(summary, reference, payload)
    environment = {
        key: deepcopy(payload["environment"][key])
        for key in (
            "device",
            "processor",
            "platform",
            "machine",
            "torch",
            "torchvision",
            "python",
            "pillow",
            "cpu_count",
            "threads",
            "interop_threads",
            "precision",
            "batch_size",
            "opencv",
        )
        if key in payload["environment"]
    }
    if "cuda" in payload["environment"]:
        environment["cuda"] = {
            key: deepcopy(payload["environment"]["cuda"][key])
            for key in CUDA_FIELDS
            if key in payload["environment"]["cuda"]
        }
    return {
        "id": item["id"],
        "export_id": export["id"],
        "name": export["name"],
        "model_id": model["id"],
        "lane_id": lane["id"],
        "created_at": item["created_at"],
        "fingerprint": item["fingerprint"],
        "archive_sha256": export["archive_sha256"],
        "model_sha256": model["sha256"],
        "profile": {
            key: deepcopy(manifest["profile"][key])
            for key in ("id", "architecture", "device", "precision", "batch_size", "timing")
            if key in manifest["profile"]
        },
        "source": {
            **{
                key: source[key]
                for key in ("evaluation_id", "evaluation_model_id", "dataset_manifest_sha256")
            },
            "reference_device": source.get("reference_device", "cpu"),
            "frame_ids": [frame["frame_id"] for frame in reference["frames"]],
        },
        "environment": environment,
        "summary": {
            key: deepcopy(summary[key])
            for key in (
                "parity_passed",
                "frames",
                "repeats",
                "sample_count",
                "mismatched_samples",
                "timing_ms",
                "decode_ms",
                "load_ms",
                "warmup_ms",
                "execution_verified",
            )
        },
        "declaration": payload.get(
            "declaration", payload.get("evidence_kind", "declared_execution")
        ),
    }


def _onnx_summary(summary, reference, payload):
    """Project measured fields without inventing unrecorded stage/load durations."""

    def distribution(values):
        return {"min": min(values), "median": median(values), "max": max(values)}

    samples = payload["samples"]
    expected = reference["frames"] * payload["repeats"]
    mismatches = [
        {"frame_id": sample["frame_id"], "repeat": sample["repeat"]}
        for sample, frame in zip(samples, expected, strict=True)
        if any(sample["prediction"][key] != frame[key] for key in ("input_size", "detections"))
    ]
    return {
        **summary,
        "frames": len(reference["frames"]),
        "repeats": payload["repeats"],
        "mismatched_samples": mismatches,
        "timing_ms": {
            **{
                key: {"min": None, "median": None, "max": None}
                for key in ("preprocess_ms", "inference_ms", "postprocess_ms")
            },
            "total_ms": distribution(
                [item["prediction"]["timing"]["total_ms"] for item in samples]
            ),
        },
        "decode_ms": distribution([item["prediction"]["timing"]["decode_ms"] for item in samples]),
        "load_ms": None,
        "warmup_ms": None,
    }


def available_measurements(store, report, detail, *, selected_ids=None):
    """Only expose measurements belonging to the report's evaluation and full-image lane.

    Broken optional evidence cannot prevent a plain report. Explicitly selecting
    missing, unrelated or invalid evidence is always rejected. Bundle files and
    checkpoints are unnecessary: import already validated them, and the saved
    reference is checked against this evaluation again here.
    """
    result = []
    wanted = None if selected_ids is None else set(selected_ids)
    if wanted == set():
        return []
    if wanted is None:
        # Read identities first; never decode an unbounded collection of 8 MiB payloads.
        with store.connect() as conn:
            selected_ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT m.id FROM model_export_measurements m "
                    "JOIN model_exports e ON e.id=m.export_id WHERE e.evaluation_id=? "
                    "ORDER BY m.created_at DESC,m.id DESC LIMIT ?",
                    (report["evaluation"]["id"], MAX_AVAILABLE_MEASUREMENTS),
                )
            ]
    for identifier in selected_ids:
        try:
            item = store.get("model_export_measurements", identifier)
            export = store.get("model_exports", item["export_id"])
            result.append(_measurement(export, item, report, detail))
        except (KeyError, TypeError, ValueError, StopIteration, OverflowError) as exc:
            if wanted is not None:
                raise ValueError(
                    "Selected standalone measurement is unavailable or inconsistent"
                ) from exc
    if wanted is not None:
        indexed = {item["id"]: item for item in result}
        if wanted != indexed.keys():
            raise ValueError(
                "Selected measurements must belong to this evaluated checkpoint and lane"
            )
        return [indexed[identifier] for identifier in selected_ids]
    return result
