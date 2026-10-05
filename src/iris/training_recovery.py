"""Durable CPU/CUDA training states and explicit, immutable successor attempts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path

from iris.store import _decode, new_id, now
from iris.training_device import CUDA_PROTOCOL, device_label, normalize_device, resolve_device

PROTOCOL = "iris-training-state-v1"
MAX_STATE_BYTES = 512 * 1024**2
MAX_STEPS = 10000
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def base_config(config):
    return {key: value for key, value in config.items() if key not in {"request_id", "resume_from"}}


def durable(config):
    return config.get("checkpoint_protocol") in {PROTOCOL, CUDA_PROTOCOL}


def validate_config(config):
    if not durable(config):
        raise ValueError("This training predates durable optimizer checkpoints")
    device = normalize_device(config.get("device"))
    if config["checkpoint_protocol"] != (PROTOCOL if device == "cpu" else CUDA_PROTOCOL):
        raise ValueError("Training device differs from its recovery state protocol")
    if device != "cpu":
        identity = config.get("device_identity")
        if (
            config["device"] != device
            or not isinstance(identity, dict)
            or identity.get("device") != device
            or config.get("precision") != "float32"
            or config.get("deterministic_algorithms") is not False
        ):
            raise ValueError("Unsupported CUDA training configuration or device identity")
    steps, interval = config.get("steps"), config.get("checkpoint_interval")
    if (
        type(steps) is not int
        or not 1 <= steps <= MAX_STEPS
        or type(interval) is not int
        or not math.ceil(steps / 200) <= interval <= 1000
        or type(config.get("history_interval")) is not int
        or config.get("history_interval") != 10
        or type(config.get("batch_size")) is not int
        or config.get("batch_size") != 1
    ):
        raise ValueError("Unsupported durable training configuration")


def sampling(seed, frames, count):
    """Replay only sampler bookkeeping, never pixels or optimizer work."""
    randomizer, order, visits = random.Random(seed), [], []
    for _ in range(count):
        if not order:
            order = list(range(len(frames)))
            randomizer.shuffle(order)
        visits.append(frames[order.pop()]["frame_id"])
    sampler = json.loads(
        canonical({"random_state": randomizer.getstate(), "remaining_order": order})
    )
    return randomizer, order, visits, sampler


def file_identity(path):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_STATE_BYTES:
        raise ValueError("Training state is missing, unsafe or larger than 512 MiB")
    with path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    return checksum, path.stat().st_size


def _sync_checkpoint_directory(directory, root):
    # Persist the renamed file and newly created attempt directories before the
    # database references them. Directory descriptors are a POSIX capability.
    if hasattr(os, "O_DIRECTORY"):
        for path in (directory, directory.parent, root):
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _get(connection, table, identifier):
    return _decode(
        connection.execute(f"SELECT * FROM {table} WHERE id=?", (identifier,)).fetchone()
    )


def _frames(root, connection, training):
    from iris.datasets import load_manifest
    from iris.model_exports import _View
    from iris.model_taxonomy import class_contract, dataset_contract
    from iris.training import _scope_from_config

    dataset = _get(connection, "dataset_versions", training["dataset_id"])
    if (
        dataset is None
        or dataset["manifest_sha256"] != training["config"]["dataset_manifest_sha256"]
    ):
        raise ValueError("Training dataset identity changed")
    manifest = load_manifest(_View(root, connection), dataset["id"])
    if class_contract(training["config"]) != dataset_contract(manifest):
        raise ValueError("Training class definitions changed")
    _scope_from_config(training["config"])
    frames = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    if not frames or not any(frame["boxes"] for frame in frames):
        raise ValueError("The frozen training split needs positive examples")
    return frames


def _validate_history(history, frames, config, step):
    if not isinstance(history, list) or len(history) < step:
        raise ValueError("Checkpoint history prefix is missing")
    _, _, visits, sampler = sampling(config["seed"], frames, step)
    previous = 0.0
    for index, (item, frame_id) in enumerate(zip(history[:step], visits, strict=True), 1):
        if (
            not isinstance(item, dict)
            or item.get("step") != index
            or type(item.get("step")) is not int
            or item.get("frame_id") != frame_id
            or not isinstance(item.get("losses"), dict)
            or not item["losses"]
        ):
            raise ValueError("Checkpoint history or sampling order is inconsistent")
        for number in [item.get("loss"), item.get("elapsed_seconds"), *item["losses"].values()]:
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError("Checkpoint history has invalid losses or times")
        if item["elapsed_seconds"] < previous:
            raise ValueError("Checkpoint elapsed time moved backwards")
        previous = item["elapsed_seconds"]
    return sampler


def binding(training, history, runtime):
    config = training["config"]
    return {
        "protocol": config["checkpoint_protocol"],
        "config_sha256": digest(base_config(config)),
        "dataset_id": training["dataset_id"],
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "parent_model_id": training["parent_model_id"],
        "parent_weight_sha256": config["parent_weight_sha256"],
        "step": len(history),
        "history_sha256": digest(history),
        "elapsed_seconds": history[-1]["elapsed_seconds"],
        "runtime": runtime,
    }


def validate_training_checkpoint(row, *, connection, root):
    """Validate only recorded metadata; archives verify opaque state bytes separately."""
    training = _get(connection, "training_runs", row["training_id"])
    if training is None:
        raise ValueError("Training state has no owning run")
    validate_config(training["config"])
    step = row["step"]
    if type(step) is not int or not 1 <= step <= training["config"]["steps"]:
        raise ValueError("Training state step is outside its frozen plan")
    if row["path"] != f"training_checkpoints/{training['id']}/{row['id']}.pth":
        raise ValueError("Training state path is outside its attempt directory")
    metadata = row["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {"binding", "sampler"}:
        raise ValueError("Unsupported training checkpoint metadata")
    if not isinstance(metadata["binding"], dict):
        raise ValueError("Training state binding is malformed")
    runtime = metadata["binding"].get("runtime")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("Training state runtime identity is missing")
    frames = _frames(root, connection, training)
    sampler = _validate_history(training["history"], frames, training["config"], step)
    if (
        canonical(metadata["sampler"]) != canonical(sampler)
        or canonical(metadata["binding"])
        != canonical(binding(training, training["history"][:step], runtime))
        or not isinstance(row["state_sha256"], str)
        or not _SHA.fullmatch(row["state_sha256"])
        or type(row["size_bytes"]) is not int
        or not 0 < row["size_bytes"] <= MAX_STATE_BYTES
    ):
        raise ValueError("Training state differs from its configuration, sampler or history")


def validate_training_recoveries(connection, root):
    from iris.model_exports import _View
    from iris.projects import record_project

    consumed = set()
    view = _View(root, connection)
    runs = {row["id"]: row for row in view.list("training_runs")}
    for row in runs.values():
        resume = row["config"].get("resume_from")
        if resume is None:
            continue
        ancestor, visited = row, set()
        while ancestor and ancestor["config"].get("resume_from"):
            if ancestor["id"] in visited:
                raise ValueError("Training recovery lineage contains a cycle")
            visited.add(ancestor["id"])
            parent = ancestor["config"]["resume_from"]
            if not isinstance(parent, dict) or not isinstance(parent.get("training_id"), str):
                raise ValueError("Training recovery lineage is malformed")
            ancestor = runs.get(parent["training_id"])
        if not isinstance(resume, dict) or set(resume) != {
            "training_id",
            "checkpoint_id",
            "step",
            "state_sha256",
        }:
            raise ValueError("Invalid training recovery lineage")
        if (
            type(resume["step"]) is not int
            or not 1 <= resume["step"] <= MAX_STEPS
            or not isinstance(resume["checkpoint_id"], str)
        ):
            raise ValueError("Invalid training recovery checkpoint identity")
        source = view.get("training_runs", resume["training_id"])
        checkpoint = view.get("training_checkpoints", resume["checkpoint_id"])
        if source is None or checkpoint is None or source["id"] in consumed:
            raise ValueError("Training recovery source is missing or was consumed more than once")
        consumed.add(source["id"])
        source_job, job = view.get("jobs", source["job_id"]), view.get("jobs", row["job_id"])
        if (
            source["id"] == row["id"]
            or source["created_at"] > row["created_at"]
            or source_job["status"] not in {"failed", "cancelled", "interrupted"}
            or source_job["kind"] != "train"
            or source_job["params"].get("training_id") != source["id"]
            or job["kind"] != "train"
            or job["params"].get("training_id") != row["id"]
            or job["params"].get("recovery_of") != source["job_id"]
            or record_project(view, "training_runs", source)
            != record_project(view, "training_runs", row)
            or canonical(base_config(source["config"])) != canonical(base_config(row["config"]))
            or source["dataset_id"] != row["dataset_id"]
            or source["parent_model_id"] != row["parent_model_id"]
            or checkpoint["step"] != resume["step"]
            or checkpoint["state_sha256"] != resume["state_sha256"]
            or canonical(row["history"][: resume["step"]])
            != canonical(source["history"][: resume["step"]])
        ):
            raise ValueError("Training recovery differs from its frozen source attempt")
        allowed = {source["id"]}
        inherited = source["config"].get("resume_from")
        if inherited:
            parent_checkpoint = view.get("training_checkpoints", inherited["checkpoint_id"])
            if parent_checkpoint:
                allowed.add(parent_checkpoint["training_id"])
        if checkpoint["training_id"] not in allowed:
            raise ValueError("Training recovery uses an unrelated checkpoint")
        if checkpoint["training_id"] != source["id"] and (
            not inherited or inherited["checkpoint_id"] != checkpoint["id"]
        ):
            raise ValueError("Inherited training state identity is inconsistent")
        validate_training_checkpoint(checkpoint, connection=connection, root=root)


def _checkpoint(store, training):
    checkpoints = store.list("training_checkpoints", training_id=training["id"])
    if checkpoints:
        return max(checkpoints, key=lambda item: item["step"])
    inherited = training["config"].get("resume_from")
    return store.get("training_checkpoints", inherited["checkpoint_id"]) if inherited else None


def recovery_summary(store, training):
    supported = durable(training["config"])
    checkpoint = _checkpoint(store, training) if supported else None
    job = store.get("jobs", training["job_id"])
    children = [
        row
        for row in store.list("training_runs")
        if row["config"].get("resume_from", {}).get("training_id") == training["id"]
    ]
    can_resume = bool(
        checkpoint
        and job["status"] in {"failed", "cancelled", "interrupted"}
        and not training["checkpoint_id"]
        and not children
    )
    reason = (
        "An optimizer checkpoint is available. Preview before resuming."
        if can_resume
        else "This attempt already has a successor."
        if children
        else "This attempt has no durable optimizer state."
        if not checkpoint
        else "Stop the current attempt before preparing a resume."
        if job["status"] in {"queued", "running"}
        else "A completed model can start a new fine-tuning run."
    )
    step = checkpoint["step"] if checkpoint else 0
    return {
        "supported": supported,
        "can_resume": can_resume,
        "reason": reason,
        "checkpoint_id": checkpoint["id"] if checkpoint else None,
        "checkpoint_step": step,
        "recorded_steps": len(training["history"]),
        "recomputed_steps": max(0, len(training["history"]) - step),
        "existing_training_id": children[0]["id"] if children else None,
    }


def preview_resume(store, training_id):
    from iris.training import _check_holdouts, _manifest, _ready_parent

    training = store.get("training_runs", training_id)
    if training is None:
        raise KeyError(training_id)
    summary = recovery_summary(store, training)
    if not summary["can_resume"] and not summary["existing_training_id"]:
        raise ValueError(summary["reason"])
    validate_config(training["config"])
    checkpoint = _checkpoint(store, training)
    if training["config"]["device"] != "cpu":
        resolve_device(training["config"]["device"], expected=training["config"]["device_identity"])
    with store.connect() as connection:
        validate_training_recoveries(connection, store.root)
        validate_training_checkpoint(checkpoint, connection=connection, root=store.root)
    checksum, size = file_identity(store.artifact_path(checkpoint["path"]))
    if checksum != checkpoint["state_sha256"] or size != checkpoint["size_bytes"]:
        raise ValueError("Training checkpoint bytes changed; resume is unavailable")
    parent = _ready_parent(store, training["parent_model_id"])
    if parent["weight_sha256"] != training["config"]["parent_weight_sha256"]:
        raise ValueError("Parent checkpoint changed; resume is unavailable")
    from iris.training_architectures import FRCNN

    if parent["architecture"] != training["config"].get("architecture", FRCNN):
        raise ValueError("Parent architecture changed; resume is unavailable")
    _check_holdouts(_manifest(store, training["dataset_id"]), parent)
    # Verify every frozen training file at explicit preview, without opening hold-outs.
    for frame in _manifest(store, training["dataset_id"])["frames"]:
        if frame["split"] == "train":
            with store.artifact_path(frame["image_path"]).open("rb") as image:
                if hashlib.file_digest(image, "sha256").hexdigest() != frame["image_file_sha256"]:
                    raise ValueError("A frozen training image changed; resume is unavailable")
    plan = {
        "source_training_id": training_id,
        "checkpoint_id": checkpoint["id"],
        "checkpoint_step": checkpoint["step"],
        "state_sha256": checkpoint["state_sha256"],
        "target_steps": training["config"]["steps"],
        "remaining_steps": training["config"]["steps"] - checkpoint["step"],
        "recorded_steps": len(training["history"]),
        "recomputed_steps": summary["recomputed_steps"],
        "config": base_config(training["config"]),
        "checkpoint_metadata": checkpoint["metadata"],
    }
    return {
        **plan,
        "fingerprint": digest(plan),
        "existing_training_id": summary["existing_training_id"],
        "warnings": [
            "Resume creates a new attempt with unchanged data, classes and settings.",
            "Work recorded after the durable checkpoint will be recomputed; the old "
            "history stays intact.",
            "Optimizer, random generators and sampler state are restored on the original "
            "training device. Runtime compatibility is checked before training continues.",
            *(
                [
                    "CUDA operations may be nondeterministic; "
                    "exact repeated results are not guaranteed."
                ]
                if training["config"]["device"] != "cpu"
                else []
            ),
            "No real resume or model-quality claim is established by a preview.",
        ],
    }


def resume_training(store, jobs, training_id, *, expected_fingerprint):
    from iris.training import training_detail

    with jobs.guard, store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        preview = preview_resume(store, training_id)
        if preview["fingerprint"] != expected_fingerprint:
            raise ValueError("Resume evidence changed; inspect a fresh preview")
        if preview["existing_training_id"]:
            identifier = preview["existing_training_id"]
        else:
            source = store.get("training_runs", training_id)
            identifier, job_id, created_at = new_id(), new_id(), now()
            config = {
                **base_config(source["config"]),
                "resume_from": {
                    "training_id": training_id,
                    "checkpoint_id": preview["checkpoint_id"],
                    "step": preview["checkpoint_step"],
                    "state_sha256": preview["state_sha256"],
                },
            }
            connection.execute(
                "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    "train",
                    "queued",
                    json.dumps({"training_id": identifier, "recovery_of": source["job_id"]}),
                    f"Waiting to resume the saved {device_label(config['device'])} optimizer state",
                    created_at,
                ),
            )
            connection.execute(
                "INSERT INTO training_runs "
                "(id,name,dataset_id,parent_model_id,config,history,job_id,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    source["name"],
                    source["dataset_id"],
                    source["parent_model_id"],
                    canonical(config).decode(),
                    canonical(source["history"][: preview["checkpoint_step"]]).decode(),
                    job_id,
                    created_at,
                ),
            )
    return training_detail(store, identifier)


def claim_attempt(store, training):
    token = new_id()
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        job = _get(connection, "jobs", training["job_id"])
        if (
            job["status"] not in {"queued", "running"}
            or job["cancel_requested"]
            or job["params"].get("training_claim")
        ):
            raise ValueError("This training attempt is stopped or already claimed")
        connection.execute(
            "UPDATE jobs SET params=? WHERE id=?",
            (canonical({**job["params"], "training_claim": token}).decode(), job["id"]),
        )
    return token


def save_history(store, training, history, token):
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        job = _get(connection, "jobs", training["job_id"])
        if (
            job["status"] not in {"queued", "running"}
            or job["params"].get("training_claim") != token
        ):
            return False
        connection.execute(
            "UPDATE training_runs SET history=? WHERE id=?",
            (canonical(history).decode(), training["id"]),
        )
    return True


def save_metadata(store, training, metadata, token):
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        job = _get(connection, "jobs", training["job_id"])
        if (
            job["status"] not in {"queued", "running"}
            or job["params"].get("training_claim") != token
        ):
            return False
        connection.execute(
            "UPDATE training_runs SET metadata=? WHERE id=?",
            (canonical(metadata).decode(), training["id"]),
        )
    return True


def save_checkpoint(store, training, trainer, history, sampler, token):
    """Publish state and its exact history prefix together, retaining the latest two."""
    runtime = trainer.resume_runtime()
    metadata = {"binding": binding(training, history, runtime), "sampler": sampler}
    directory = store.artifact_path(f"training_checkpoints/{training['id']}")
    directory.mkdir(parents=True, exist_ok=True)
    identifier = new_id()
    target = directory / f"{identifier}.pth"
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
    published, removed = False, []
    try:
        trainer.write_resume_state(temporary, binding=metadata["binding"], sampler=sampler)
        checksum, size = file_identity(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        with store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = _get(connection, "jobs", training["job_id"])
            if (
                job["status"] not in {"queued", "running"}
                or job["params"].get("training_claim") != token
            ):
                return None
            temporary.replace(target)
            _sync_checkpoint_directory(directory, store.root)
            connection.execute(
                "INSERT INTO training_checkpoints "
                "(id,training_id,step,path,state_sha256,size_bytes,metadata,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    training["id"],
                    len(history),
                    str(target.relative_to(store.root)),
                    checksum,
                    size,
                    canonical(metadata).decode(),
                    now(),
                ),
            )
            connection.execute(
                "UPDATE training_runs SET history=? WHERE id=?",
                (canonical(history).decode(), training["id"]),
            )
            stale = connection.execute(
                "SELECT id,path FROM training_checkpoints WHERE training_id=? "
                "ORDER BY step DESC LIMIT -1 OFFSET 2",
                (training["id"],),
            ).fetchall()
            for old in stale:
                if connection.execute(
                    "SELECT 1 FROM training_runs WHERE "
                    "json_extract(config,'$.resume_from.checkpoint_id')=?",
                    (old["id"],),
                ).fetchone():
                    continue
                connection.execute("DELETE FROM training_checkpoints WHERE id=?", (old["id"],))
                removed.append(store.artifact_path(old["path"]))
            connection.commit()
            published = True
        for old in removed:
            old.unlink(missing_ok=True)
        return store.get("training_checkpoints", identifier)
    finally:
        temporary.unlink(missing_ok=True)
        if not published:
            target.unlink(missing_ok=True)
