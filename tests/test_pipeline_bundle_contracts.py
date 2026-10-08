"""Synthetic portable-pipeline contracts and hostile archive inputs; no ML execution."""

import hashlib
import math
import os
import stat
import struct
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from test_models_taxonomy import checkpoint_row
from test_temporal_detector import frozen_config
from test_tracking_study_runtime import synthetic_metadata
from test_training_taxonomy import contract as custom_contract

from iris import models, temporal_detector
from iris import pipeline_bundle_contracts as bundles
from iris import pipeline_detector_contracts as detectors
from iris.model_taxonomy import class_contract
from iris.tracking_contracts import make_profile, profile_hash
from iris.tracking_selection_contracts import DEFAULT_POLICY, validate_policy
from iris.training_architectures import FRCNN, SSDLITE, YOLOX

contract = custom_contract


def synthetic_bundle(*, architecture=SSDLITE, algorithm="bytetrack", selection=False):
    checkpoint = b"synthetic checkpoint bytes, deliberately never deserialized"
    config = frozen_config(architecture)
    config["weight_sha256"] = hashlib.sha256(checkpoint).hexdigest()
    profile = make_profile(algorithm, class_ids=[1, 3])
    metadata = synthetic_metadata(profile)
    files = {
        name: ("Synthetic resource: " + name).encode()
        for name in bundles.required_paths(architecture)
    }
    files["detector/model.pth"] = checkpoint
    files["iris_bundle/__init__.py"] = b""
    files["licenses/tracker-LICENSE"] = (
        Path(models.__file__).parent / "_vendor" / algorithm / "LICENSE"
    ).read_bytes()
    # Inspection must not import or execute the bundled inspector or source modules.
    files["inspect.py"] = b"raise AssertionError('untrusted bundled code executed')\n"
    manifest = {
        "format": bundles.FORMAT,
        "id": "bundle-1",
        "name": "Synthetic portable contract",
        "created_at": "2026-10-08T12:00:00+00:00",
        "producer": {"name": "IRIS", "version": "0.55.0"},
        "detector": bundles.detector_contract(config, "cpu", checkpoint_size=len(checkpoint)),
        "tracker": {
            "profile": profile,
            "profile_sha256": profile_hash(profile),
            "runtime": metadata,
            "runtime_sha256": bundles.digest(metadata),
        },
        "selection": None,
        "interface": deepcopy(bundles.INTERFACE),
        "validation": deepcopy(bundles.VALIDATION),
        "source": {
            "descriptor": {
                "kind": "comparison",
                "job_id": "comparison-1",
                "sequence_id": "sequence-1",
                "profile_sha256": profile_hash(profile),
            },
            **{
                key: "a" * 64
                for key in (
                    "source_report_sha256",
                    "sequence_sha256",
                    "cache_fingerprint",
                    "result_sha256",
                    "replay_sha256",
                    "first_pass_semantic_sha256",
                )
            },
            "inherited_dataset": None,
            "repeatability": "not_checked",
        },
        "licenses": bundles.license_contract(config),
        "files": {
            path: {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            for path, raw in files.items()
        },
    }
    if selection:
        policy = validate_policy(DEFAULT_POLICY)
        manifest["selection"] = {
            "algorithm": "guarded_geometry",
            "policy": policy,
            "policy_sha256": bundles.digest(policy),
            "source_job_id": "selection-1",
            "source_report_sha256": "b" * 64,
        }
    return manifest, files


def write_bundle(
    path, manifest, files, *, manifest_raw=None, compression=zipfile.ZIP_STORED, extras=()
):
    with zipfile.ZipFile(path, "w", compression=compression, allowZip64=False) as archive:
        archive.writestr(
            "manifest.json", bundles.canonical(manifest) if manifest_raw is None else manifest_raw
        )
        for name, raw in files.items():
            archive.writestr(name, raw)
        for name, raw in extras:
            archive.writestr(name, raw)
    return path


def test_bundle_roundtrip_fingerprints_and_detached_validation_without_code_execution(tmp_path):
    manifest, files = synthetic_bundle(selection=True)
    original = deepcopy(manifest)
    checked = bundles.validate_manifest(manifest)
    checked["detector"]["target_device"] = "cuda"
    assert manifest == original
    path = write_bundle(tmp_path / "pipeline.zip", manifest, files)
    result = bundles.inspect_bundle(path, expected_manifest=manifest)
    assert result == {
        "manifest": manifest,
        "manifest_sha256": bundles.digest(manifest),
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_bytes": path.stat().st_size,
    }
    assert result["manifest"]["validation"]["checkpoint_payload"] == "not_deserialized"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("architecture", [FRCNN, SSDLITE, YOLOX])
@pytest.mark.parametrize("origin", ["official", "trained_builtin", "trained_custom"])
def test_detector_recipe_snapshot_parity_and_unambiguous_class_indices(
    architecture, origin, contract
):
    spec = None
    if origin != "official":
        row = checkpoint_row(contract if origin == "trained_custom" else class_contract({}))
        row["architecture"] = architecture
        spec = models._trained_spec(row)
    config = frozen_config(architecture, spec=spec)
    assert detectors.validate_detector_config(config) == temporal_detector.validate_detector_config(
        config
    )
    output = bundles.detector_contract(config, "cuda", checkpoint_size=100)
    assert output["config"]["device"] == "cpu" and output["target_device"] == "cuda"
    mapping = output["output_mapping"]
    first = mapping["entries"][0]
    assert first["internal_index"] == (0 if architecture == YOLOX else 1)
    assert first["native_label_id"] == 1
    assert first["output_id"] == 1
    if origin == "official":
        assert mapping["head_slots"] == (80 if architecture == YOLOX else 91)
        # COCO ID 12 is absent, but head indices differ by architecture.
        at_gap = next(row for row in mapping["entries"] if row["output_id"] == 13)
        assert at_gap["internal_index"] == (11 if architecture == YOLOX else 13)
        assert at_gap["native_label_id"] == (12 if architecture == YOLOX else 13)
    elif origin == "trained_builtin":
        assert mapping["entries"][1] == {
            "internal_index": 1 if architecture == YOLOX else 2,
            "native_label_id": 2,
            "output_id": 3,
            "label": "car",
        }
    else:
        assert [row["output_id"] for row in mapping["entries"]] == [1, 2, 3, 4]
        assert mapping["head_slots"] == (4 if architecture == YOLOX else 5)
    assert output["checkpoint"]["encoding"] == (
        "pytorch_model_envelope"
        if architecture == YOLOX and origin == "official"
        else "pytorch_state_dict"
    )


@pytest.mark.parametrize("architecture", [FRCNN, SSDLITE, YOLOX])
def test_native_detector_semantic_mutations_reject_in_both_validators(architecture):
    base = frozen_config(architecture)
    mutations = [
        lambda x: x["preprocessing"].update(color="WRONG"),
        lambda x: x["native_filtering"].update(score_threshold=0.2),
        lambda x: x["output_policy"].update(complete_above_score_floor=True),
        lambda x: x["classes"][0].update(name="someone"),
        lambda x: x["runtime"]["source_sha256"].update(unexpected="a" * 64),
        lambda x: x.update(min_score=True),
        lambda x: x.update(inference={"mode": "full", "extra": True}),
    ]
    for change in mutations:
        value = deepcopy(base)
        change(value)
        for validate in (
            detectors.validate_detector_config,
            temporal_detector.validate_detector_config,
        ):
            with pytest.raises(ValueError):
                validate(value)


def test_tiled_recipe_is_explicitly_unsupported_not_silently_reinterpreted():
    config = frozen_config(inference_mode="tiled")
    assert temporal_detector.validate_detector_config(config)["inference"]["mode"] == "tiled"
    with pytest.raises(ValueError):
        bundles.detector_contract(config, "cpu", checkpoint_size=10)


@pytest.mark.parametrize(
    "mutation",
    [
        "weights",
        "encoding",
        "mapping",
        "profile",
        "profile_class",
        "score_floor",
        "tracker_runtime",
        "tracker_hash",
        "selection_hash",
        "selection_algorithm",
        "interface",
        "source_hash",
        "source_profile",
        "reserved_test",
        "license_hash",
        "license_rights",
        "extra_file",
        "missing_file",
        "qualified",
        "parity",
        "live_id",
        "foreign_field",
    ],
)
def test_rehashed_semantic_or_inventory_tampering_is_rejected(mutation):
    manifest, _ = synthetic_bundle(selection=True)
    if mutation == "weights":
        manifest["detector"]["checkpoint"]["sha256"] = "e" * 64
    elif mutation == "encoding":
        manifest["detector"]["checkpoint"]["encoding"] = "pytorch_model_envelope"
    elif mutation == "mapping":
        manifest["detector"]["output_mapping"]["entries"][1]["output_id"] = 99
    elif mutation == "profile":
        manifest["tracker"]["profile"]["buffer_updates"] += 1
    elif mutation == "profile_class":
        manifest["tracker"]["profile"]["class_ids"] = [12]
        manifest["tracker"]["profile_sha256"] = profile_hash(manifest["tracker"]["profile"])
        manifest["source"]["descriptor"]["profile_sha256"] = manifest["tracker"]["profile_sha256"]
    elif mutation == "score_floor":
        manifest["detector"]["config"]["min_score"] = 0.3
    elif mutation == "tracker_runtime":
        manifest["tracker"]["runtime"]["execution_policy"]["seed"] += 1
        manifest["tracker"]["runtime_sha256"] = bundles.digest(manifest["tracker"]["runtime"])
    elif mutation == "tracker_hash":
        manifest["tracker"]["runtime_sha256"] = "a" * 64
    elif mutation == "selection_hash":
        manifest["selection"]["policy"]["max_lost_updates"] += 1
    elif mutation == "selection_algorithm":
        manifest["selection"]["algorithm"] = "track_id_only"
    elif mutation == "interface":
        manifest["interface"]["coordinates"] = "normalized_xywh"
    elif mutation == "source_hash":
        manifest["source"]["replay_sha256"] = "not-a-hash"
    elif mutation == "source_profile":
        manifest["source"]["descriptor"]["profile_sha256"] = "a" * 64
    elif mutation == "reserved_test":
        manifest["source"]["inherited_dataset"] = {
            "dataset_id": "d1",
            "manifest_sha256": "a" * 64,
            "split": "test",
        }
    elif mutation == "license_hash":
        manifest["files"]["licenses/tracker-LICENSE"]["sha256"] = "a" * 64
    elif mutation == "license_rights":
        manifest["licenses"]["detector"]["weights_rights"] = "MIT"
    elif mutation == "extra_file":
        manifest["files"]["private/frame.png"] = {"sha256": "a" * 64, "size": 100}
    elif mutation == "missing_file":
        del manifest["files"]["inspect.py"]
    elif mutation == "qualified":
        manifest["validation"]["status"] = "qualified"
    elif mutation == "parity":
        manifest["validation"]["pipeline_parity"] = "passed"
    elif mutation == "live_id":
        manifest["selection"]["track_id"] = 17
    else:
        manifest["unexpected"] = 1
    with pytest.raises(ValueError):
        bundles.validate_manifest(manifest)


@pytest.mark.parametrize("algorithm", ["bytetrack", "botsort"])
def test_tracker_provenance_license_and_optional_policy_are_preserved(algorithm, tmp_path):
    manifest, files = synthetic_bundle(architecture=YOLOX, algorithm=algorithm, selection=True)
    result = bundles.inspect_bundle(write_bundle(tmp_path / "portable.zip", manifest, files))
    assert result["manifest"]["licenses"]["detector"]["code_license"] == "Apache-2.0"
    assert "licenses/detector-NOTICE" in result["manifest"]["files"]
    assert result["manifest"]["tracker"]["runtime"]["algorithm"] == algorithm
    assert result["manifest"]["selection"]["source_job_id"] == "selection-1"


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/absolute",
        "C:/drive",
        "back\\slash",
        "empty//part",
        "dot/./file",
        "manifest.json\x00suffix",
        "MANIFEST.JSON",
    ],
)
def test_archive_paths_and_case_collisions_are_rejected_without_extraction(tmp_path, path):
    manifest, files = synthetic_bundle()
    zip_path = write_bundle(tmp_path / "unsafe.zip", manifest, files, extras=[(path, b"untrusted")])
    with pytest.raises(ValueError):
        bundles.inspect_bundle(zip_path)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["unsafe.zip"]


def test_duplicate_archive_names_unlisted_files_and_file_changes_are_rejected(tmp_path):
    manifest, files = synthetic_bundle()
    with pytest.warns(UserWarning, match="Duplicate name"):
        path = write_bundle(
            tmp_path / "duplicate.zip",
            manifest,
            files,
            extras=[("manifest.json", bundles.canonical(manifest))],
        )
    with pytest.raises(ValueError, match="duplicate"):
        bundles.inspect_bundle(path)
    path = write_bundle(
        tmp_path / "extra.zip", manifest, files, extras=[("harmless.txt", b"extra")]
    )
    with pytest.raises(ValueError, match="inventory"):
        bundles.inspect_bundle(path)
    changed = {**files, "detector/model.pth": b"x" * len(files["detector/model.pth"])}
    path = write_bundle(tmp_path / "changed.zip", manifest, changed)
    with pytest.raises(ValueError, match="fingerprint"):
        bundles.inspect_bundle(path)


@pytest.mark.parametrize(
    "mode",
    [
        "compressed",
        "symlink",
        "fifo",
        "directory",
        "extra",
        "encrypted",
        "zip64",
        "archive_comment",
        "prefix",
    ],
)
def test_unsupported_zip_features_are_rejected(tmp_path, mode):
    manifest, files = synthetic_bundle()
    path = tmp_path / "features.zip"
    if mode == "compressed":
        write_bundle(path, manifest, files, compression=zipfile.ZIP_DEFLATED)
    elif mode in {"symlink", "fifo", "directory", "extra", "zip64"}:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("manifest.json" + ("/" if mode == "directory" else ""))
            info.create_system = 3
            if mode in {"symlink", "fifo"}:
                info.external_attr = (stat.S_IFLNK if mode == "symlink" else stat.S_IFIFO) << 16
            if mode == "extra":
                info.extra = b"\x99\x00\x00\x00"
            if mode == "zip64":
                with archive.open(info, "w", force_zip64=True) as target:
                    target.write(bundles.canonical(manifest))
            else:
                archive.writestr(info, bundles.canonical(manifest))
            for name, raw in files.items():
                archive.writestr(name, raw)
    else:
        write_bundle(path, manifest, files)
        if mode == "archive_comment":
            with zipfile.ZipFile(path, "a") as archive:
                archive.comment = b"comment"
        else:
            raw = bytearray(path.read_bytes())
            if mode == "encrypted":
                struct.pack_into("<H", raw, 6, 1)
                central = raw.index(b"PK\x01\x02")
                struct.pack_into("<H", raw, central + 8, 1)
            else:
                raw = b"MZ executable prefix" + raw
            path.write_bytes(raw)
    with pytest.raises(ValueError):
        bundles.inspect_bundle(path)


def test_duplicate_nonfinite_deep_and_oversized_json_rejected_before_use():
    for raw in (
        b'{"format":1,"format":2}',
        b'{"bad":NaN}',
        b'{"bad":Infinity}',
        b'{"bad":1e999}',
        rb'"\ud800"',
        b"[" * 50 + b"0" + b"]" * 50,
    ):
        with pytest.raises(ValueError):
            bundles.read_json(raw)
    with pytest.raises(ValueError):
        bundles.read_json(b" " * (bundles.MAX_JSON_BYTES + 1))
    for value in (math.nan, math.inf, 10**1000):
        with pytest.raises(ValueError):
            bundles.canonical({"number": value})


def test_central_directory_bounds_checked_before_zipfile_parses_members(tmp_path, monkeypatch):
    manifest, files = synthetic_bundle()
    path = write_bundle(tmp_path / "central.zip", manifest, files)
    raw = bytearray(path.read_bytes())
    struct.pack_into("<H", raw, len(raw) - 22 + 8, 65535)
    struct.pack_into("<H", raw, len(raw) - 22 + 10, 65535)
    path.write_bytes(raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("ZipFile constructed before directory bound")

    monkeypatch.setattr(bundles.zipfile, "ZipFile", forbidden)
    with pytest.raises(ValueError, match="central directory"):
        bundles.inspect_bundle(path)


def test_archive_symlink_fifo_expected_manifest_and_checkpoint_cancellation(tmp_path):
    manifest, files = synthetic_bundle()
    path = write_bundle(tmp_path / "source.zip", manifest, files)
    linked = tmp_path / "link.zip"
    linked.symlink_to(path)
    with pytest.raises(ValueError, match="symbolic"):
        bundles.inspect_bundle(linked)
    fifo = tmp_path / "fifo.zip"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        bundles.inspect_bundle(fifo)
    different = {**manifest, "name": "different name"}
    with pytest.raises(ValueError, match="saved export"):
        bundles.inspect_bundle(path, different)
    calls = []

    def cancelled():
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("cancelled on purpose")

    with pytest.raises(RuntimeError, match="cancelled on purpose"):
        bundles.inspect_bundle(path, checkpoint=cancelled)


def test_portable_validator_uses_only_stdlib_and_copied_contracts(tmp_path):
    manifest, files = synthetic_bundle(selection=True)
    path = write_bundle(tmp_path / "source.zip", manifest, files)
    package = tmp_path / "iris_bundle"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in (
        "pipeline_bundle_contracts.py",
        "pipeline_detector_contracts.py",
        "tracking_contracts.py",
        "tracking_selection_contracts.py",
    ):
        (package / name).write_bytes((Path(models.__file__).parent / name).read_bytes())
    code = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'iris','torch','torchvision','numpy','cv2','PIL','scipy','lap'}:
        raise AssertionError('Optional/IRIS import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from iris_bundle.pipeline_bundle_contracts import inspect_bundle
assert inspect_bundle(sys.argv[2])['manifest']['validation']['pipeline_execution']=='not_run'
"""
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(tmp_path), str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("kind", ["comparison", "study"])
def test_source_kind_and_inherited_dataset_must_agree(kind):
    manifest, _ = synthetic_bundle()
    manifest["source"]["descriptor"]["kind"] = kind
    inherited = {"dataset_id": "dataset-1", "manifest_sha256": "a" * 64, "split": "train"}
    manifest["source"]["inherited_dataset"] = inherited if kind == "comparison" else None
    with pytest.raises(ValueError, match="inherited dataset"):
        bundles.validate_manifest(manifest)
    manifest["source"]["inherited_dataset"] = inherited if kind == "study" else None
    assert bundles.validate_manifest(manifest) == manifest


def test_custom_definition_normalization_and_multiline_text_match_frozen_detector(contract):
    spec = models._trained_spec(checkpoint_row(contract))
    config = frozen_config(spec=spec)
    for field in ("name", "definition"):
        changed = deepcopy(config)
        changed["class_contract"]["taxonomy"]["classes"][0][field] += " "
        for validate in (
            detectors.validate_detector_config,
            temporal_detector.validate_detector_config,
        ):
            with pytest.raises(ValueError, match="normalized"):
                validate(changed)
    config["class_contract"]["taxonomy"]["classes"][0]["definition"] = (
        "Visible person.\nExclude occluded extent."
    )
    assert detectors.validate_detector_config(config) == temporal_detector.validate_detector_config(
        config
    )
