#!/usr/bin/env python3
"""Project the selected, existing FPV evaluations into a public recording.

Read an explicitly supplied workspace in SQLite read-only mode. No model runs,
weights, private notes, workstation paths or unrelated workspace rows are copied.
The checked-in result supports ordinary builds without access to that workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import zipfile
from pathlib import Path

from public_contract import project
from recording import (
    EVALUATION_IDS,
    MODEL_IDS,
    RECORDING,
    SOURCE_FRAMES,
    validate_recording,
)

SOURCE_URL = "https://github.com/victormonnot/iris/tree/main/examples/web-demo/source"
TERMS_URL = "https://github.com/victormonnot/iris/blob/main/examples/web-demo/source/CREDITS.txt"
SOURCE_CAPTURE = {
    "id": "ec843f1682954d93a3286069e9a98842",
    "created_at_utc": "2026-10-03T11:16:41.145464+00:00",
    "manifest_sha256": "5fdc619c4f700656250c868eab294155bc9e55c2752dc02e8bbe8b39270f2708",
    "date_basis": "Native camera recording manifest; capture start, not Iris import date.",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def collect(workspace):
    """Return only the explicit public sample; the caller chooses an absent output."""
    workspace = workspace.resolve(strict=True)
    connection = sqlite3.connect((workspace / "iris.sqlite3").as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    def row(table, identity):
        # Table names are internal constants, never command-line input.
        value = connection.execute(f"SELECT * FROM {table} WHERE id=?", (identity,)).fetchone()
        if value is None:
            raise ValueError(f"The selected {table} record is unavailable")
        result = dict(value)
        for key in (
            "config",
            "metadata",
            "detections",
            "timing",
            "input_size",
            "boxes",
            "extraction",
        ):
            if key in result:
                result[key] = json.loads(result[key])
        return result

    def artifact(relative, expected):
        path = (workspace / relative).resolve(strict=True)
        if not path.is_relative_to(workspace):
            raise ValueError("A source artifact escapes the supplied workspace")
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError("A source image differs from its saved hash")
        return data

    try:
        evaluations = [row("evaluations", eid) for eid in EVALUATION_IDS]
        if any(
            e["dataset_id"] != evaluations[0]["dataset_id"] or e["split"] != "val"
            for e in evaluations
        ):
            raise ValueError("The source evaluations do not share the frozen validation dataset")
        dataset = row("dataset_versions", evaluations[0]["dataset_id"])
        if any(
            e["config"]["dataset_manifest_sha256"] != dataset["manifest_sha256"]
            for e in evaluations
        ):
            raise ValueError("The evaluations do not share the recorded dataset digest")
        manifest = json.loads(artifact(dataset["path"], dataset["manifest_sha256"]))
        frozen_frames = {f["frame_id"]: f for f in manifest["frames"]}
        files, bindings, images, annotations, source_images, source_reviews = {}, [], [], [], [], []
        for source_index, fid in SOURCE_FRAMES.items():
            frame = row("frames", fid)
            asset = row("assets", frame["asset_id"])
            filename = f"images/{source_index}.jpg"
            data = artifact(asset["path"], asset["sha256"])
            files[filename] = data
            for evaluation in evaluations:
                if evaluation["config"]["frame_hashes"][fid] != frame["sha256"]:
                    raise ValueError("The evaluation used different source pixels")
            frozen = frozen_frames[fid]
            review = frozen["annotation"]
            if frozen["sha256"] != frame["sha256"] or frozen["split"] != "val":
                raise ValueError("The public frame differs from its frozen validation reference")
            if review["status"] != "validated" or review["frame_sha256"] != frame["sha256"]:
                raise ValueError("The source reference is not a matching human review")
            if len(review["boxes"]) != 1 or review["boxes"][0]["label"] != "person":
                raise ValueError("Unexpected human reference contents")
            box = review["boxes"][0]["box"]
            width, height = box[2] - box[0], box[3] - box[1]
            annotations.append(
                {
                    "id": source_index,
                    "image_id": source_index,
                    "category_id": 1,
                    "bbox": [box[0], box[1], width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            images.append(
                {
                    "id": source_index,
                    "license": 1,
                    "file_name": filename,
                    "coco_url": "",
                    "flickr_url": "",
                    "date_captured": SOURCE_CAPTURE["created_at_utc"][:10],
                    "width": frame["width"],
                    "height": frame["height"],
                }
            )
            bindings.append(
                {
                    "frame_id": fid,
                    "asset_id": asset["id"],
                    "coco_image_id": source_index,
                    "source_filename": filename,
                    "source_sha256": asset["sha256"],
                    "frame_pixel_sha256": frame["sha256"],
                    "width": frame["width"],
                    "height": frame["height"],
                }
            )
            source_images.append(
                {
                    "coco_image_id": source_index,
                    "archive_path": filename,
                    "source_frame_id": fid,
                    "source_frame_index": source_index,
                    "source_filename": Path(asset["filename"]).name,
                    "title": f"FPV courtyard recording — frame {source_index}",
                    "author": "Victor Monnot",
                    "author_url": "https://victormonnot.com/",
                    "photo_url": SOURCE_URL,
                    "license": "All rights reserved",
                    "license_url": TERMS_URL,
                    "sha256": asset["sha256"],
                    "size_bytes": len(data),
                    "width": frame["width"],
                    "height": frame["height"],
                }
            )
            source_reviews.append(
                {
                    "frame_id": fid,
                    "annotation_id": source_index,
                    "revision": review["revision"],
                    "created_at": review["created_at"],
                    "box": box,
                    "source_state": "human-reviewed",
                    "demo_initial_state": "pending imported proposal",
                }
            )
        credits = (
            "IRIS FPV DEMO — SOURCE AND USE TERMS\n\n"
            "Copyright © 2026 Victor Monnot. All rights reserved.\n"
            "Two original camera JPEG frames (1601 and 1681), recorded on 3 October\n"
            "2026, and their human reference annotations are published by their author\n"
            "for this Iris demonstration. They are not covered by the software MIT\n"
            "licence; no general redistribution or training-data licence is granted.\n"
            "Author: https://victormonnot.com/\n"
            f"Source: {SOURCE_URL}\nUse terms: {TERMS_URL}\n\n"
            "The selected development images were already used during model development.\n"
            "They are not an independent test or evidence of general FPV robustness.\n"
            "The original JPEG bytes, complete image dimensions and recorded detections\n"
            "are preserved. Public COCO image IDs equal source camera frame indices.\n"
        ).encode()
        files["CREDITS.txt"] = credits
        files["annotations.json"] = json_bytes(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
                "licenses": [
                    {"id": 1, "name": "All rights reserved — Victor Monnot", "url": TERMS_URL}
                ],
            }
        )
        source_provenance = {
            "format": "iris-fpv-demo-source-v1",
            "capture": SOURCE_CAPTURE,
            "images": source_images,
            "annotation_source": {
                "author": "Victor Monnot",
                "license": "All rights reserved",
                "license_url": TERMS_URL,
                "mirror": SOURCE_URL,
                "kind": "manual reference boxes",
                "reviews": source_reviews,
            },
            "default_import_mapping": {"1": "person"},
            "files": {
                name: {"sha256": digest(files[name]), "size_bytes": len(files[name])}
                for name in ("CREDITS.txt", "annotations.json")
            },
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in sorted(files.items()):
                entry = zipfile.ZipInfo(name, (2026, 10, 13, 0, 0, 0))
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, data)
        archive_bytes = buffer.getvalue()
        public_source = {
            **{name: data for name, data in files.items() if not name.startswith("images/")},
            "fpv-yolox.zip": archive_bytes,
            "provenance.json": json_bytes(source_provenance),
        }
        cid = "fpv-yolox-comparison"
        runs, predictions, models = [], [], []
        for index, evaluation in enumerate(evaluations):
            mid = MODEL_IDS[index]
            run_row = connection.execute(
                "SELECT id FROM evaluation_models WHERE evaluation_id=? AND model_id=?",
                (evaluation["id"], mid),
            ).fetchone()
            source_run = row("evaluation_models", run_row[0])
            meta = source_run["metadata"]
            run = {
                "id": source_run["id"],
                "comparison_id": cid,
                "model_id": mid,
                "variant": source_run["variant"],
                "created_at": source_run["created_at"],
                "metadata": {**meta, "source_evaluation_id": evaluation["id"]},
            }
            runs.append(run)
            for fid in SOURCE_FRAMES.values():
                pred_row = connection.execute(
                    (
                        "SELECT id FROM evaluation_predictions WHERE "
                        "evaluation_id=? AND frame_id=? AND model_id=?"
                    ),
                    (evaluation["id"], fid, mid),
                ).fetchone()
                prediction = row("evaluation_predictions", pred_row[0])
                predictions.append(
                    {
                        **prediction,
                        "comparison_id": cid,
                        "run_id": source_run["id"],
                        "metadata": {"source_evaluation_id": evaluation["id"]},
                    }
                )
            from iris.models import COCO_CATEGORIES

            models.append(
                {
                    "id": mid,
                    "name": "YOLOX-Nano · original"
                    if index == 0
                    else "YOLOX-Nano · trained in Iris",
                    "architecture": meta["architecture"],
                    "weights_name": meta["weights_name"],
                    "weight_sha256": meta["weight_sha256"],
                    "license_url": meta["license_url"],
                    "code_license_url": meta["license_url"],
                    "classes": (
                        [
                            {"id": i, "name": name}
                            for i, name in enumerate(COCO_CATEGORIES)
                            if i and name != "N/A"
                        ]
                        if index == 0
                        else [{"id": 1, "name": "person"}]
                    ),
                    "task": "object_detection",
                    "inference": True,
                    "training": True,
                    "training_summary": (
                        "Official pretrained COCO weights."
                        if index == 0
                        else (
                            "Person-only model trained in Iris for 800 steps; "
                            "partial backbone, clipping v2."
                        )
                    ),
                    "origin": "official" if index == 0 else "trained",
                    "parent_model_id": None if index == 0 else MODEL_IDS[0],
                }
            )
        first_date = evaluations[0]["created_at"]
        last_date = max(p["created_at"] for p in predictions)
        comparison = {
            "id": cid,
            "session_id": "fpv-yolox",
            "name": "YOLOX-Nano · original vs trained in Iris",
            "frame_ids": list(SOURCE_FRAMES.values()),
            "model_ids": list(MODEL_IDS),
            "created_at": first_date,
            "job_id": "fpv-yolox-saved-results",
            "config": {
                "device": "cuda:0",
                "warmup": 1,
                "taxonomy": "coco-2017-v1",
                "class_mapping": {"person": 1},
                "model_hashes": {m["id"]: m["weight_sha256"] for m in models},
                "protocol": evaluations[0]["config"]["timing_protocol"],
                "inference": {"mode": "full"},
                "lanes": [{"model_id": m, "variant": "full"} for m in MODEL_IDS],
                "model_class_contracts": {
                    model: evaluation["config"]["model_class_contracts"][model]
                    for model, evaluation in zip(MODEL_IDS, evaluations, strict=True)
                },
                "source_evaluation_ids": list(EVALUATION_IDS),
                "result_origin": "saved-evaluation-projection",
            },
        }
        job = {
            "id": comparison["job_id"],
            "kind": "infer",
            "status": "succeeded",
            "params": {"comparison_id": cid},
            "result": {
                "comparison_id": cid,
                "frames_total": 2,
                "models_total": 2,
                "predictions_created": 4,
                "cancelled": False,
            },
            "progress": 1.0,
            "message": "Saved evaluation results",
            "error": None,
            "created_at": first_date,
            "started_at": first_date,
            "finished_at": last_date,
            "cancel_requested": False,
        }
        provenance = {
            "result_origin": "saved-evaluation-projection",
            "source_evaluations": [
                {"id": e["id"], "model_id": m, "created_at": e["created_at"]}
                for e, m in zip(evaluations, MODEL_IDS, strict=True)
            ],
            "source_dataset_id": evaluations[0]["dataset_id"],
            "source_dataset_manifest_sha256": evaluations[0]["config"]["dataset_manifest_sha256"],
            "source_frame_ids": list(SOURCE_FRAMES.values()),
            "source_prediction_ids": [p["id"] for p in predictions],
            "source_training_id": "184a0b5356ca4425b621495cd386f410",
            "training_steps": 800,
            "new_inference": False,
            "network_used": False,
            "assembly": (
                "Comparison/job wrappers join two saved evaluations for "
                "display; no new joint comparison run occurred. Wrapper dates "
                "span the source evaluations, not an elapsed execution time."
            ),
            "model_filtering": (
                "All native saved detections are retained, including low "
                "scores, ignored classes and source taxonomy metadata. Display "
                "defaults are person class at confidence 0.35."
            ),
            "timing_scope": (
                "Unchanged per-image CUDA measurements from the original "
                "evaluations on RTX 4060; not browser or user-device "
                "performance, and not a new simultaneous comparison."
            ),
            "limitations": (
                "Two selected images from the reused 17-image development "
                "validation set. These examples do not establish generalization "
                "or independently measure model quality."
            ),
            "archive_sha256": digest(archive_bytes),
            "source_provenance_sha256": digest(public_source["provenance.json"]),
        }
        captured = project(
            {
                "format": "iris-demo-evaluation-recording-v1",
                "frames": bindings,
                "comparison": comparison,
                "job": job,
                "runs": runs,
                "predictions": predictions,
                "models": models,
                "provenance": provenance,
            },
            RECORDING,
        )
        validate_recording(captured)
        return {
            **{f"source/{name}": data for name, data in public_source.items()},
            "recording.json": json_bytes(captured),
        }
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, required=True, help="Existing source workspace (read only)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory; never overwritten"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    files = collect(args.workspace)
    for name, data in files.items():
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"Projected {len(files)} public files from saved results; no inference executed")


if __name__ == "__main__":
    main()
