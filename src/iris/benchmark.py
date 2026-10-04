"""Frozen human references and explicit tuning/evaluation protocols for preannotation."""

import hashlib
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path

from PIL import Image

from iris.annotations import _coordinates, _latest
from iris.dataset_manifest import taxonomy_mappings
from iris.datasets import (
    _canonical,
    _eligibility,
    _reservation_state,
    _source_video,
    dataset_candidates,
)
from iris.evaluation import PRETRAINING_WARNING, _model_lineage
from iris.inference import _load_verified_frame, _work_plan
from iris.media import _file_hash, _pixel_hash
from iris.metrics import _iou
from iris.models import catalog
from iris.preannotation_contracts import build_contract, validate_contract
from iris.projects import record_project
from iris.store import DEFAULT_PROJECT_ID, Store, _decode, new_id, now
from iris.taxonomies import get_taxonomy
from iris.tiling import validate_tiling_config

PROTOCOL = "iris-preannotation-benchmark-v1"
ROLES = {"tuning", "evaluation"}
MAX_ROLE_FRAMES = 25
MAX_CONFIGS = 8
SCORING = {
    "id": "iris-proposal-quality-v1",
    "iou_threshold": 0.5,
    "matching": "one-to-one, same class, proposal order, maximum IoU; ties use reference order",
    "class_conflicts": "overlapping unmatched boxes of different classes; also count as FP and FN",
    "scope": "proposals delivered after the frozen provider threshold; no AP or invented scores",
    "headline": "only a complete role with one successful output for every reference image",
}
WARNINGS = [
    "Human independence is declared, not proof of freedom from prior model influence "
    "or pretraining.",
    "Distinct scene names and source hashes do not prove that visually related scenes "
    "are independent.",
    "Repeated inspection of evaluation results makes the held-out set less independent.",
]


class BenchmarkConflict(RuntimeError):
    """A confirmation no longer describes the frozen inputs or protocol state."""


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def benchmark_approaches(root=None):
    from iris.benchmark_combined import provider_status as combined_status
    from iris.multimodal_provider import provider_status
    from iris.sam_provider import provider_status as sam_status

    multimodal = provider_status()
    segmentation = sam_status(root) if root is not None else None
    combined = combined_status(root) if root is not None else None
    return [
        {"id": "local_detector", "name": "Installed local detector control", "available": True},
        {
            "id": "multimodal",
            "name": "A — multimodal alone",
            "implemented": True,
            "available": multimodal["status"] == "ready",
            "reason": multimodal["reason"],
        },
        {
            "id": "segmentation",
            "name": "B — SAM alone",
            "implemented": True,
            "available": bool(segmentation and segmentation["status"] == "ready"),
            "reason": segmentation["reason"] if segmentation else "Check local SAM setup",
        },
        {
            "id": "combined",
            "name": "C — Astra + SAM 3",
            "implemented": True,
            "available": bool(combined and combined["status"] == "ready"),
            "reason": combined["reason"] if combined else "Check OpenAI and local SAM setup",
        },
    ]


def validate_benchmark_manifest(manifest: dict) -> dict:
    """Validate an immutable reference without consulting current annotations or classes."""
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("protocol") != PROTOCOL
    ):
        raise ValueError("Unsupported benchmark reference manifest")
    internal, output = taxonomy_mappings(manifest.get("taxonomy"))
    reference, frames = manifest.get("reference"), manifest.get("frames")
    if (
        manifest.get("class_mapping") != internal
        or manifest.get("output_mapping") != output
        or not isinstance(reference, dict)
        or reference.get("independent_reference") is not True
        or not isinstance(reference.get("reviewer"), str)
        or not reference["reviewer"].strip()
        or not isinstance(reference.get("independence_notes"), str)
        or not reference["independence_notes"].strip()
        or not isinstance(frames, list)
        or not 2 <= len(frames) <= MAX_ROLE_FRAMES * 2
    ):
        raise ValueError("The frozen benchmark reference or class mapping is inconsistent")
    roles, seen_ids, seen_pixels, videos = {}, set(), set(), {}
    counts = dict.fromkeys(sorted(ROLES), 0)
    for frame in frames:
        annotation = frame.get("annotation", {})
        identifier, role = frame.get("frame_id"), frame.get("role")
        if (
            not isinstance(identifier, str)
            or identifier in seen_ids
            or role not in ROLES
            or annotation.get("id") != frame.get("annotation_revision_id")
            or annotation.get("frame_id") != identifier
            or annotation.get("revision") != frame.get("revision")
            or annotation.get("frame_sha256") != frame.get("sha256")
            or annotation.get("taxonomy_id") != manifest["taxonomy"]["id"]
            or annotation.get("status") != "validated"
            or not annotation.get("reviewer")
            or annotation.get("boxes") != frame.get("boxes")
            or not isinstance(frame.get("boxes"), list)
            or frame["sha256"] in seen_pixels
        ):
            raise ValueError(
                "Frozen reference annotations or unique image identities are inconsistent"
            )
        seen_ids.add(identifier)
        seen_pixels.add(frame["sha256"])
        group = frame.get("scene_group")
        if not isinstance(group, str) or not group.strip() or roles.get(group, role) != role:
            raise ValueError("A benchmark scene cannot cross tuning and evaluation")
        roles[group] = role
        counts[role] += 1
        video = _source_video(frame.get("source", {}))
        if video and videos.get(video, role) != role:
            raise ValueError("An original video cannot cross benchmark roles")
        if video:
            videos[video] = role
        for box in frame["boxes"]:
            if box.get("label") not in internal or box.get("source", {}).get("kind") not in {
                "manual",
                "imported",
            }:
                raise ValueError(
                    "Benchmark references must use reviewed human boxes, not model-derived boxes"
                )
            _coordinates(box.get("box"), frame)
        expected = f"benchmarks/{manifest['id']}/images/{identifier}.png"
        if frame.get("image_path") != expected or not isinstance(
            frame.get("image_file_sha256"), str
        ):
            raise ValueError("A benchmark image is outside its frozen reference directory")
    if manifest.get("roles") != roles or any(
        not 1 <= count <= MAX_ROLE_FRAMES for count in counts.values()
    ):
        raise ValueError("A benchmark needs 1–25 images in each role and distinct complete scenes")
    return manifest


def load_benchmark_manifest(store: Store, benchmark_id: str, verify_images: bool = False) -> dict:
    row = store.get("benchmarks", benchmark_id)
    if row is None:
        raise KeyError(benchmark_id)
    if row["path"] != f"benchmarks/{benchmark_id}/manifest.json":
        raise ValueError("Benchmark manifest path differs from its identity")
    raw = store.artifact_path(row["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["manifest_sha256"]:
        raise ValueError("Benchmark reference manifest changed")
    manifest = validate_benchmark_manifest(json.loads(raw))
    if manifest["id"] != benchmark_id or manifest["project_id"] != row["project_id"]:
        raise ValueError("Benchmark reference ownership changed")
    if verify_images:
        for frame in manifest["frames"]:
            with open_benchmark_image(store, frame):
                pass
    return manifest


def open_benchmark_image(store, frame):
    path = store.artifact_path(frame["image_path"])
    if _file_hash(path) != frame["image_file_sha256"]:
        raise ValueError("Frozen benchmark image bytes changed")
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.load()
    if image.size != (frame["width"], frame["height"]) or _pixel_hash(image) != frame["sha256"]:
        image.close()
        raise ValueError("Frozen benchmark pixels changed")
    return image


def benchmark_frame(store, benchmark_id, frame_id):
    for frame in load_benchmark_manifest(store, benchmark_id)["frames"]:
        if frame["frame_id"] == frame_id:
            return frame
    raise KeyError(frame_id)


def _role_reservations(store, conn, project_id):
    groups, pixels, videos = _reservation_state(store, conn, project_id)

    def role(split):
        return "tuning" if split == "train" else "evaluation"

    groups = {key: {role(value)} for key, value in groups.items()}
    pixels = {key: {role(value)} for key, value in pixels.items()}
    videos = {key: {role(value) for value in values} for key, values in videos.items()}
    for row in conn.execute("SELECT id,project_id FROM benchmarks"):
        for frame in load_benchmark_manifest(store, row["id"])["frames"]:
            if row["project_id"] == project_id:
                groups.setdefault(frame["scene_group"], set()).add(frame["role"])
            pixels.setdefault(frame["sha256"], set()).add(frame["role"])
            video = _source_video(frame["source"])
            if video:
                videos.setdefault(video, set()).add(frame["role"])
    return groups, pixels, videos


def _allowed_roles(group, digest, video, reservations):
    groups, pixels, videos = reservations
    used = groups.get(group, set()) | pixels.get(digest, set()) | videos.get(video, set())
    return sorted(ROLES if not used else used if len(used) == 1 else set())


def benchmark_candidates(store: Store, project_id=DEFAULT_PROJECT_ID, taxonomy_id=None):
    candidates = dataset_candidates(store, project_id, taxonomy_id)
    groups = []
    excluded = {**candidates["excluded"], "model_derived_reference": 0}
    with store.connect() as conn:
        reservations = _role_reservations(store, conn, project_id)
        for group in candidates["groups"]:
            frames = []
            for frame in group["frames"]:
                revision = _latest(conn, frame["id"])
                if any(
                    box.get("source", {}).get("kind") not in {"manual", "imported"}
                    for box in revision["boxes"]
                ):
                    excluded["model_derived_reference"] += 1
                    continue
                frames.append(
                    {
                        **frame,
                        "frame_id": frame["id"],
                        "source_filename": frame["source"]["filename"],
                        "taxonomy_id": candidates["taxonomy"]["id"],
                        "allowed_roles": _allowed_roles(
                            group["scene_group"],
                            frame["sha256"],
                            frame["video_sha256"],
                            reservations,
                        ),
                    }
                )
            if frames:
                groups.append(
                    {
                        **group,
                        "frames": frames,
                        "count": len(frames),
                        "allowed_roles": sorted(
                            set.intersection(*(set(frame["allowed_roles"]) for frame in frames))
                        ),
                    }
                )
    return {
        **candidates,
        "groups": groups,
        "excluded": excluded,
        "warnings": [*candidates["warnings"], *WARNINGS],
    }


def _reference_preview(
    store,
    *,
    project_id,
    frame_ids,
    roles,
    reviewer,
    independence_notes,
    independent_reference,
    taxonomy_id=None,
):
    if (
        not isinstance(frame_ids, list)
        or not 2 <= len(frame_ids) <= 50
        or any(not isinstance(frame_id, str) or not frame_id for frame_id in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("Choose 2–50 distinct reference images, at most 25 per role")
    if (
        not isinstance(roles, dict)
        or any(
            not isinstance(group, str) or not group.strip() or role not in ROLES
            for group, role in roles.items()
        )
        or independent_reference is not True
        or not isinstance(reviewer, str)
        or not 1 <= len(reviewer.strip()) <= 120
        or not isinstance(independence_notes, str)
        or not 1 <= len(independence_notes.strip()) <= 4000
    ):
        raise ValueError(
            "Assign scene roles and explicitly declare a named independent human reference"
        )
    project = store.get("projects", project_id)
    if project is None:
        raise KeyError(project_id)
    taxonomy = get_taxonomy(store, taxonomy_id or project["taxonomy_id"], project_id)
    snapshots, classes = [], {item["id"] for item in taxonomy["classes"]}
    with store.connect() as conn:
        conn.execute("BEGIN")
        reservations = _role_reservations(store, conn, project_id)
        for frame_id in frame_ids:
            frame = _decode(conn.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone())
            if frame is None:
                raise ValueError("A selected reference image is missing")
            session = store.get("sessions", frame["session_id"])
            if session["project_id"] != project_id:
                raise ValueError("Every reference image must belong to this project")
            annotation, reason = _eligibility(conn, frame, taxonomy["id"])
            if reason or not annotation["reviewer"]:
                raise ValueError(
                    f"Reference image is not human validated and fully reviewed: {reason}"
                )
            if annotation["frame_sha256"] != frame["sha256"]:
                raise ValueError("Reference annotation no longer matches the selected pixels")
            for box in annotation["boxes"]:
                if (
                    box.get("source", {}).get("kind") not in {"manual", "imported"}
                    or box["label"] not in classes
                ):
                    raise ValueError(
                        "Model-derived boxes cannot serve as the independent reference"
                    )
                _coordinates(box["box"], frame)
            asset = store.get("assets", frame["asset_id"])
            role = roles.get(session["scene_group"])
            if role not in _allowed_roles(
                session["scene_group"], frame["sha256"], _source_video(asset), reservations
            ):
                raise ValueError(
                    "Scene, pixels or original video are reserved for another benchmark role"
                )
            with _load_verified_frame(store, frame, annotation["frame_sha256"]):
                pass
            snapshots.append(
                {
                    "frame_id": frame_id,
                    "session_id": session["id"],
                    "session_name": session["name"],
                    "scene_group": session["scene_group"],
                    "role": role,
                    "sha256": frame["sha256"],
                    "width": frame["width"],
                    "height": frame["height"],
                    "perceptual_hash": frame["perceptual_hash"],
                    "annotation_revision_id": annotation["id"],
                    "revision": annotation["revision"],
                    "boxes": annotation["boxes"],
                    "annotation": annotation,
                    "source": {
                        "asset_id": asset["id"],
                        "filename": asset["filename"],
                        "kind": asset["kind"],
                        "sha256": asset["sha256"],
                        "metadata": asset["metadata"],
                        "frame_index": frame["frame_index"],
                        "timestamp_seconds": frame["timestamp_seconds"],
                        "extraction": frame["extraction"],
                    },
                }
            )
    if set(roles) != {frame["scene_group"] for frame in snapshots}:
        raise ValueError("Role assignments must exactly match the selected scene groups")
    seen, videos = set(), {}
    counts = {role: sum(frame["role"] == role for frame in snapshots) for role in sorted(ROLES)}
    if any(not 1 <= count <= MAX_ROLE_FRAMES for count in counts.values()):
        raise ValueError("Both tuning and evaluation need 1–25 images from separate scenes")
    for frame in snapshots:
        if frame["sha256"] in seen:
            raise ValueError("Benchmark references cannot contain duplicate image pixels")
        seen.add(frame["sha256"])
        video = _source_video(frame["source"])
        if video and videos.get(video, frame["role"]) != frame["role"]:
            raise ValueError("Frames from one original video cannot cross benchmark roles")
        if video:
            videos[video] = frame["role"]
    reference = {
        "reviewer": reviewer.strip(),
        "independence_notes": independence_notes.strip(),
        "independent_reference": True,
    }
    summary = {
        "frame_count": len(snapshots),
        "role_counts": counts,
        "negative_count": sum(not frame["boxes"] for frame in snapshots),
        "class_counts": {
            label: sum(box["label"] == label for frame in snapshots for box in frame["boxes"])
            for label in classes
        },
    }
    value = {
        "project_id": project_id,
        "taxonomy": taxonomy,
        "frames": snapshots,
        "roles": roles,
        "reference": reference,
    }
    return {**value, "summary": summary, "warnings": WARNINGS, "fingerprint": _digest(value)}


def preview_benchmark(store: Store, *, project_id=DEFAULT_PROJECT_ID, **settings):
    return _reference_preview(store, project_id=project_id, **settings)


def create_benchmark(
    store: Store, *, name, expected_fingerprint, project_id=DEFAULT_PROJECT_ID, **settings
):
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
        raise ValueError("Benchmark name must contain 1–160 characters")
    identifier, created = new_id(), now()
    directory = Path("benchmarks") / identifier
    staging = store.artifact_path(f"benchmarks/.staging-{identifier}")
    published = False
    try:
        staging.mkdir(parents=True)
        (staging / "images").mkdir()
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            preview = _reference_preview(store, project_id=project_id, **settings)
            if preview["fingerprint"] != expected_fingerprint:
                raise BenchmarkConflict("Reference inputs changed; preview the benchmark again")
            internal, output = taxonomy_mappings(preview["taxonomy"])
            manifest = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "id": identifier,
                "project_id": project_id,
                "name": name.strip(),
                "created_at": created,
                "taxonomy": preview["taxonomy"],
                "class_mapping": internal,
                "output_mapping": output,
                "reference": preview["reference"],
                "roles": preview["roles"],
                "frames": deepcopy(preview["frames"]),
            }
            for frame in manifest["frames"]:
                copied = staging / "images" / f"{frame['frame_id']}.png"
                live = store.get("frames", frame["frame_id"])
                with _load_verified_frame(store, live, frame["sha256"]) as image:
                    image.save(copied, format="PNG")
                frame.update(
                    image_path=str(directory / "images" / copied.name),
                    image_file_sha256=_file_hash(copied),
                )
            validate_benchmark_manifest(manifest)
            content = _canonical(manifest)
            (staging / "manifest.json").write_bytes(content)
            staging.rename(store.artifact_path(str(directory)))
            conn.execute(
                "INSERT INTO benchmarks "
                "(id,project_id,name,path,manifest_sha256,summary,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    identifier,
                    project_id,
                    name.strip(),
                    str(directory / "manifest.json"),
                    hashlib.sha256(content).hexdigest(),
                    json.dumps({**preview["summary"], "warnings": WARNINGS}),
                    created,
                ),
            )
        published = True
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not published:
            shutil.rmtree(store.artifact_path(str(directory)), ignore_errors=True)
    return benchmark_detail(store, identifier)


def preview_benchmark_config(
    store,
    benchmark_id,
    *,
    model_id,
    threshold=0.5,
    device="cpu",
    inference_mode="full",
    tile_size=640,
    overlap=0.2,
    approach="local_detector",
    multimodal=None,
    segmentation=None,
    combined=None,
):
    row = store.get("benchmarks", benchmark_id)
    if row is None:
        raise KeyError(benchmark_id)
    if row["status"] != "tuning":
        raise BenchmarkConflict("This benchmark's configurations are already locked")
    if segmentation is not None and approach != "segmentation":
        raise ValueError("SAM settings require the segmentation approach")
    if multimodal is not None and approach != "multimodal":
        raise ValueError("Multimodal settings require the multimodal approach")
    if combined is not None and approach != "combined":
        raise ValueError("Combined settings require the combined approach")
    if approach == "combined":
        from iris.benchmark_combined import preview_config

        if (threshold, device, inference_mode, tile_size, overlap) != (
            0.5,
            "cpu",
            "full",
            640,
            0.2,
        ):
            raise ValueError("Use nested combined settings; detector settings do not apply")
        return preview_config(store, benchmark_id, model_id=model_id, combined=combined)
    if approach == "segmentation":
        from iris.benchmark_segmentation import preview_config

        if (threshold, device, inference_mode, tile_size, overlap) != (
            0.5,
            "cpu",
            "full",
            640,
            0.2,
        ):
            raise ValueError(
                "Use nested segmentation settings; detector settings do not apply to SAM"
            )
        return preview_config(store, benchmark_id, model_id=model_id, segmentation=segmentation)
    if approach == "multimodal":
        from iris.benchmark_multimodal import preview_config

        return preview_config(store, benchmark_id, model_id=model_id, multimodal=multimodal)
    if approach != "local_detector":
        raise ValueError(
            "This future approach is unavailable; "
            "choose a local detector, multimodal or SAM candidate"
        )
    if multimodal is not None:
        raise ValueError("Multimodal settings require the multimodal approach")
    if (
        type(threshold) not in {int, float}
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Confidence threshold must be between zero and one")
    if device not in {"cpu", "cuda"} or inference_mode not in {"full", "tiled"}:
        raise ValueError("Choose CPU/CUDA and full/tiled inference")
    manifest = load_benchmark_manifest(store, benchmark_id)
    models = {model["id"]: model for model in catalog(store.root)}
    model = models.get(model_id)
    if model is None or model["status"] != "ready" or not model.get("weight_sha256"):
        raise ValueError("This detector is not installed and ready")
    trained = store.get("trained_models", model_id)
    if trained and record_project(store, "trained_models", trained) != row["project_id"]:
        raise ValueError("The detector must belong to this benchmark's project")
    contract = build_contract(store, model_id, manifest["taxonomy"])
    if not contract["supported_class_ids"]:
        raise ValueError("The detector covers none of the benchmark's classes")
    lineage = _model_lineage(
        model, models, [frame for frame in manifest["frames"] if frame["role"] == "evaluation"]
    )
    tiling = validate_tiling_config(tile_size, overlap)
    inference = {"mode": inference_mode}
    if inference_mode == "tiled":
        inference.update(algorithm="iris-tiling-v1", tiling=tiling)
    config = {
        "protocol": PROTOCOL,
        "approach": approach,
        "model_id": model_id,
        "model_name": model["name"],
        "weight_sha256": model["weight_sha256"],
        "proposal_contract": contract,
        "threshold": float(threshold),
        "device": device,
        "inference": inference,
        "scoring": SCORING,
        "reference_manifest_sha256": row["manifest_sha256"],
        "lineage": lineage,
    }
    work = {
        role: _work_plan(
            [
                {**frame, "id": frame["frame_id"]}
                for frame in manifest["frames"]
                if frame["role"] == role
            ],
            [{"model_id": model_id, "variant": inference_mode}],
            inference,
        )
        for role in sorted(ROLES)
    }
    return {
        "benchmark_id": benchmark_id,
        "config": config,
        "fingerprint": _digest(config),
        "work": work,
        "warnings": [*contract["warnings"], PRETRAINING_WARNING],
    }


def create_benchmark_config(store, benchmark_id, *, name, expected_fingerprint, **settings):
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
        raise ValueError("Configuration name must contain 1–160 characters")
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = _decode(
            conn.execute(
                "SELECT * FROM benchmark_configs WHERE benchmark_id=? AND fingerprint=?",
                (benchmark_id, expected_fingerprint),
            ).fetchone()
        )
        if existing:
            return existing
        if (
            conn.execute(
                "SELECT COUNT(*) FROM benchmark_configs WHERE benchmark_id=?", (benchmark_id,)
            ).fetchone()[0]
            >= MAX_CONFIGS
        ):
            raise ValueError("A benchmark supports at most eight frozen configurations")
        preview = preview_benchmark_config(store, benchmark_id, **settings)
        if preview["fingerprint"] != expected_fingerprint:
            raise BenchmarkConflict("Detector configuration changed; preview it again")
        identifier = new_id()
        conn.execute(
            "INSERT INTO benchmark_configs "
            "(id,benchmark_id,name,approach,config,fingerprint,created_at) VALUES (?,?,?,?,?,?,?)",
            (
                identifier,
                benchmark_id,
                name.strip(),
                preview["config"]["approach"],
                json.dumps(preview["config"]),
                preview["fingerprint"],
                now(),
            ),
        )
    return store.get("benchmark_configs", identifier)


def validate_benchmark_config(row, benchmark, manifest):
    config = row["config"]
    if (
        row["benchmark_id"] != benchmark["id"]
        or row["approach"] not in {"local_detector", "multimodal", "segmentation", "combined"}
        or config.get("protocol") != PROTOCOL
        or config.get("approach") != row["approach"]
        or _digest(config) != row["fingerprint"]
        or config.get("reference_manifest_sha256") != benchmark["manifest_sha256"]
    ):
        raise ValueError("Frozen benchmark configuration provenance is inconsistent")
    if row["approach"] == "multimodal":
        from iris.benchmark_multimodal import validate_config

        return validate_config(config, manifest)
    if row["approach"] == "segmentation":
        from iris.benchmark_segmentation import validate_config

        return validate_config(config, manifest)
    if row["approach"] == "combined":
        from iris.benchmark_combined import validate_config

        return validate_config(config, manifest)
    if config.get("scoring") != SCORING:
        raise ValueError("Frozen benchmark configuration provenance is inconsistent")
    contract = validate_contract(config["proposal_contract"])
    if contract["taxonomy"] != manifest["taxonomy"] or contract["model_id"] != config["model_id"]:
        raise ValueError("Frozen benchmark detector classes are inconsistent")
    threshold, inference = config.get("threshold"), config.get("inference", {})
    if (
        type(threshold) not in {int, float}
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or config.get("device") not in {"cpu", "cuda"}
        or inference.get("mode") not in {"full", "tiled"}
        or not isinstance(config.get("weight_sha256"), str)
        or len(config["weight_sha256"]) != 64
        or any(value not in "0123456789abcdef" for value in config["weight_sha256"])
    ):
        raise ValueError("Frozen benchmark detector settings are invalid")
    if inference["mode"] == "tiled":
        tiling = inference.get("tiling", {})
        if (
            inference.get("algorithm") != "iris-tiling-v1"
            or validate_tiling_config(tiling.get("tile_size"), tiling.get("overlap")) != tiling
        ):
            raise ValueError("Frozen benchmark tiling settings are invalid")
    return config


def _lock_fingerprint(store, row):
    return _digest(
        {
            "id": row["id"],
            "manifest": row["manifest_sha256"],
            "status": row["status"],
            "configs": store.list("benchmark_configs", benchmark_id=row["id"]),
            "trials": [
                {"id": trial["id"], "status": store.get("jobs", trial["job_id"])["status"]}
                for trial in store.list("benchmark_trials", benchmark_id=row["id"])
            ],
        }
    )


def lock_benchmark(store, benchmark_id, *, expected_fingerprint):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = store.get("benchmarks", benchmark_id)
        if row is None:
            raise KeyError(benchmark_id)
        if row["status"] == "locked":
            return benchmark_detail(store, benchmark_id)
        if _lock_fingerprint(store, row) != expected_fingerprint:
            raise BenchmarkConflict(
                "Benchmark trials or configurations changed; inspect before locking"
            )
        if not store.list("benchmark_configs", benchmark_id=benchmark_id):
            raise ValueError("Freeze at least one candidate configuration before locking")
        if conn.execute(
            "SELECT 1 FROM benchmark_trials t JOIN jobs j ON j.id=t.job_id "
            "WHERE t.benchmark_id=? AND j.status IN ('queued','running')",
            (benchmark_id,),
        ).fetchone():
            raise BenchmarkConflict("Wait for active tuning trials before locking configurations")
        conn.execute(
            "UPDATE benchmarks SET status='locked',locked_at=? WHERE id=?", (now(), benchmark_id)
        )
    return benchmark_detail(store, benchmark_id)


def benchmark_detail(store, benchmark_id):
    row = store.get("benchmarks", benchmark_id)
    if row is None:
        raise KeyError(benchmark_id)
    from iris.benchmark_runs import benchmark_trial_detail

    configs = store.list("benchmark_configs", benchmark_id=benchmark_id)
    trials = [
        benchmark_trial_detail(store, trial["id"], include_outputs=False)
        for trial in store.list("benchmark_trials", benchmark_id=benchmark_id)
    ]
    tested = {
        trial["config_id"]
        for trial in trials
        if trial["split"] == "tuning" and trial["quality"]["complete"]
    }
    return {
        **row,
        "manifest": load_benchmark_manifest(store, benchmark_id),
        "configs": configs,
        "trials": trials,
        "warnings": [
            *WARNINGS,
            *[
                f"Configuration '{config['name']}' has no complete tuning result; "
                "locking it does not establish its suitability."
                for config in configs
                if config["id"] not in tested
            ],
        ],
        "lock_fingerprint": _lock_fingerprint(store, row),
        "approaches": benchmark_approaches(store.root),
    }


def list_benchmarks(store, project_id=DEFAULT_PROJECT_ID):
    return store.list("benchmarks", project_id=project_id)


def score_proposals(frame, proposals, taxonomy):
    """Operating-point geometry only; no confidence fabrication and no AP."""
    labels = {item["id"] for item in taxonomy["classes"]}
    per_class = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    matched, matches, false_positives = set(), [], []
    for index, proposal in enumerate(proposals):
        if proposal.get("label") not in labels:
            raise ValueError("A scored proposal has an unknown reference class")
        coordinates = _coordinates(proposal.get("box"), frame)
        choices = [
            (i, _iou(coordinates, target["box"]))
            for i, target in enumerate(frame["boxes"])
            if i not in matched and target["label"] == proposal["label"]
        ]
        choices = [choice for choice in choices if choice[1] >= SCORING["iou_threshold"]]
        if choices:
            chosen, overlap = max(choices, key=lambda choice: (choice[1], -choice[0]))
            matched.add(chosen)
            matches.append({"reference_index": chosen, "proposal_index": index, "iou": overlap})
            per_class[proposal["label"]]["tp"] += 1
        else:
            false_positives.append(index)
            per_class[proposal["label"]]["fp"] += 1
    false_negatives = [i for i in range(len(frame["boxes"])) if i not in matched]
    for index in false_negatives:
        per_class[frame["boxes"][index]["label"]]["fn"] += 1
    conflicts, used = [], set()
    for index in false_positives:
        proposal = proposals[index]
        choices = [
            (i, _iou(proposal["box"], frame["boxes"][i]["box"]))
            for i in false_negatives
            if i not in used and frame["boxes"][i]["label"] != proposal["label"]
        ]
        choices = [choice for choice in choices if choice[1] >= SCORING["iou_threshold"]]
        if choices:
            chosen, overlap = max(choices, key=lambda choice: (choice[1], -choice[0]))
            used.add(chosen)
            conflicts.append(
                {
                    "reference_index": chosen,
                    "proposal_index": index,
                    "iou": overlap,
                    "expected": frame["boxes"][chosen]["label"],
                    "predicted": proposal["label"],
                }
            )
    return {
        "frame_id": frame["frame_id"],
        "tp": len(matches),
        "fp": len(false_positives),
        "fn": len(false_negatives),
        "class_conflicts": len(conflicts),
        "matches": matches,
        "class_conflict_pairs": conflicts,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "per_class": per_class,
    }


def score_benchmark_outputs(manifest, role, outputs):
    frames = [frame for frame in manifest["frames"] if frame["role"] == role]
    by_frame = {output["frame_id"]: output for output in outputs}
    complete = (
        len(outputs) == len(frames)
        and len(by_frame) == len(frames)
        and all(
            frame["frame_id"] in by_frame
            and by_frame[frame["frame_id"]]["error"] is None
            and isinstance(by_frame[frame["frame_id"]]["result"], dict)
            for frame in frames
        )
    )
    if not complete:
        return {
            "complete": False,
            "protocol": SCORING,
            "metrics": None,
            "reason": "A complete successful output is required for every role image; "
            "failed images are not empty predictions",
        }
    details = [
        score_proposals(
            frame, by_frame[frame["frame_id"]]["result"]["proposals"], manifest["taxonomy"]
        )
        for frame in frames
    ]
    totals = {
        key: sum(detail[key] for detail in details) for key in ("tp", "fp", "fn", "class_conflicts")
    }
    matches = [match["iou"] for detail in details for match in detail["matches"]]
    totals.update(
        precision=totals["tp"] / (totals["tp"] + totals["fp"])
        if totals["tp"] + totals["fp"]
        else None,
        recall=totals["tp"] / (totals["tp"] + totals["fn"])
        if totals["tp"] + totals["fn"]
        else None,
        matched_iou_mean=sum(matches) / len(matches) if matches else None,
        frame_count=len(frames),
        reference_count=sum(len(frame["boxes"]) for frame in frames),
    )
    return {
        "complete": True,
        "protocol": SCORING,
        "metrics": {
            "summary": totals,
            "frames": details,
            "per_class": {
                item["id"]: {
                    key: sum(detail["per_class"][item["id"]][key] for detail in details)
                    for key in ("tp", "fp", "fn")
                }
                for item in manifest["taxonomy"]["classes"]
            },
        },
        "reason": None,
    }
