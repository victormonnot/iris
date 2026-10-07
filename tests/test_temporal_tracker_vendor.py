"""Upstream provenance, minimal optional imports and unchanged native semantics."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import iris

VENDOR = Path(iris.__file__).parent / "_vendor"
MANIFEST = json.loads((VENDOR / "tracking-provenance.json").read_text())


def reverse_patch(adapted, hunks):
    """Reconstruct official bytes from the complete documented unified patch."""
    lines = adapted.splitlines(keepends=True)
    original, position = [], 0
    for header, changes in hunks:
        match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n", header)
        assert match
        old_start, old_count, new_start, new_count = (
            int(value) if value is not None else 1 for value in match.groups()
        )
        start = new_start - 1
        original.extend(lines[position:start])
        position = start
        assert len(original) == old_start - 1
        old_used, new_used = 0, 0
        for line in changes:
            assert line[0] in "+- "
            if line[0] != "-":
                assert lines[position] == line[1:]
                position += 1
                new_used += 1
            if line[0] != "+":
                original.append(line[1:])
                old_used += 1
        assert (old_used, new_used) == (old_count, new_count)
    original.extend(lines[position:])
    return "".join(original).encode()


@pytest.mark.parametrize("name", MANIFEST)
def test_official_sources_licenses_and_complete_adaptations(name):
    record = MANIFEST[name]
    root = VENDOR / name
    assert record["license"] == "MIT"
    assert hashlib.sha256((root / "LICENSE").read_bytes()).hexdigest() == record["license_sha256"]
    patch = (root / "adaptations.patch").read_bytes()
    assert hashlib.sha256(patch).hexdigest() == record["adaptations_sha256"]
    sections, filename = {}, None
    for line in patch.decode().splitlines(keepends=True):
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            filename = Path(line[4:].strip()).name
            sections[filename] = []
        elif line.startswith("@@ "):
            sections[filename].append((line, []))
        else:
            sections[filename][-1][1].append(line)
    for filename, info in record["files"].items():
        source = (root / filename).read_bytes()
        assert hashlib.sha256(source).hexdigest() == info["vendored_sha256"]
        if info["path"] is None:
            assert info["upstream_sha256"] is None
            continue
        reconstructed = reverse_patch(source.decode(), sections[filename])
        assert hashlib.sha256(reconstructed).hexdigest() == info["upstream_sha256"]
    assert set(sections) == {
        filename for filename, info in record["files"].items() if info["path"] is not None
    }


def test_optional_native_imports_do_not_load_model_or_gui_libraries():
    pytest.importorskip("lap", reason="optional tracking runtime not installed")
    pytest.importorskip("cython_bbox", reason="optional tracking runtime not installed")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import iris._vendor.bytetrack.byte_tracker; "
            "import iris._vendor.botsort.bot_sort; "
            "assert not any(n.split('.')[0] in "
            "{'torch', 'torchvision', 'matplotlib', 'fast_reid', 'argos'} for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_gmc_reports_disabled_and_initialization_without_changing_warp():
    from iris._vendor.botsort.gmc import GMC

    disabled = GMC(method="none")
    assert disabled.last_status == "uninitialized"
    np.testing.assert_array_equal(disabled.apply(None), np.eye(2, 3))
    assert disabled.last_status == "disabled"
    enabled = GMC(method="sparseOptFlow")
    image = np.random.default_rng(0).integers(0, 256, (80, 100, 3), dtype=np.uint8)
    np.testing.assert_array_equal(enabled.apply(image), np.eye(2, 3))
    assert enabled.last_status == "initialized"
    warp = enabled.apply(image)
    assert enabled.last_status == "estimated"
    np.testing.assert_allclose(warp, np.eye(2, 3), atol=1e-6)


def test_gmc_reports_native_identity_on_insufficient_matches(monkeypatch):
    from iris._vendor.botsort.gmc import GMC

    tracker = GMC(method="sparseOptFlow")
    tracker.initializedFirstFrame = True
    tracker.prevFrame = np.zeros((40, 50), dtype=np.uint8)
    tracker.prevKeyPoints = np.zeros((4, 1, 2), dtype=np.float32)
    monkeypatch.setattr(
        cv2,
        "calcOpticalFlowPyrLK",
        lambda *args: (np.zeros((4, 1, 2)), np.ones((4, 1)), None),
    )
    warp = tracker.apply(np.zeros((80, 100, 3), dtype=np.uint8))
    np.testing.assert_array_equal(warp, np.eye(2, 3))
    assert tracker.last_status == "identity_insufficient_matches"


def test_gmc_preserves_native_error_on_featureless_optical_flow():
    from iris._vendor.botsort.gmc import GMC

    tracker = GMC(method="sparseOptFlow")
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    tracker.apply(image)
    with pytest.raises(cv2.error):
        tracker.apply(image)


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_native_empty_update_and_exact_detector_index_survive_reactivation(name):
    pytest.importorskip("lap", reason="optional tracking runtime not installed")
    pytest.importorskip("cython_bbox", reason="optional tracking runtime not installed")
    if name == "bytetrack":
        from iris._vendor.bytetrack.basetrack import BaseTrack
        from iris._vendor.bytetrack.byte_tracker import BYTETracker

        BaseTrack._count = 0
        native = BYTETracker(
            SimpleNamespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False),
            frame_rate=30,
        )

        def update(rows):
            return native.update(rows, (100, 100), (100, 100))

    else:
        from iris._vendor.botsort.bot_sort import BoTSORT

        native = BoTSORT(
            SimpleNamespace(
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=0.8,
                mot20=False,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=False,
                cmc_method="none",
                name="iris-native-test",
                ablation=False,
            ),
            frame_rate=30,
        )

        def update(rows):
            return native.update(rows, None)

    rows = np.array([[50, 50, 80, 90, 0.05], [10, 10, 30, 45, 0.9]], dtype=np.float64)
    initial = update(rows.copy())
    assert len(initial) == 1
    assert initial[0].detection_index == 1
    identity = initial[0].track_id
    assert update(np.empty((0, 5), dtype=np.float64)) == []
    assert native.frame_id == 2
    recovered = update(rows.copy())
    assert len(recovered) == 1
    assert recovered[0].track_id == identity
    assert recovered[0].detection_index == 1
    assert recovered[0].frame_id == 3
