# Native tracking over saved detections

IRIS 0.48 adds local **ByteTrack** and **BoT-SORT without learned ReID** adapters.
Both consume the same complete [temporal detector cache](temporal-detections.md)
without running its detector again. A Python interface and command-line replay
produce explicit observations, predictions and unassigned detections. The
adapters do not depend on ARGOS or its target-selection or control rules.

This is the T3 execution foundation. A Studio tracking comparator, temporal
identity editor, tracking-quality metrics and tracker-profile export remain
separate planned steps. Running a tracker does not validate its identities or
establish that it is better than another tracker.

## Installation and readiness

The optional `tracking` extra pins `scipy==1.17.1`, `lap==0.5.12` and
`cython_bbox==0.1.5`. Existing IRIS dependencies supply NumPy and headless OpenCV.
The replay needs no Torch, detector weights, GPU, API key or network connection.
Installing packages can require network access; `cython_bbox` may require a C
compiler where a compatible wheel is unavailable.

```sh
uv sync --locked --inexact --extra tracking
uv run --extra tracking iris tracking status
```

Readiness checks installed package versions without importing optional tracker
runtimes or opening a workspace. Successful installation is not proof that every
input sequence will execute. Keep `--extra tracking` on subsequent `uv` commands;
if the same environment also runs local detectors, keep both `--extra ml` and
`--extra tracking`. The inexact installation preserves other installed extras. For an existing CUDA
environment, add these packages without replacing its Torch installation:

```sh
uv pip install --python /path/to/cuda-env/bin/python \
  scipy==1.17.1 lap==0.5.12 cython_bbox==0.1.5
```

## Replay a complete detector cache

Use an existing cache ID from the temporal API. The CLI opens its workspace in
read-only mode, does not migrate it, and creates no jobs or database rows.

```sh
uv run --extra tracking iris tracking replay \
  --data-dir /path/to/iris-data \
  --cache-id CACHE_ID \
  --tracker bytetrack \
  --class-id 1 \
  --repeats 2 \
  --output /path/to/new-bytetrack-report.json
```

Choose `--tracker botsort` for BoT-SORT with sparse optical-flow camera motion
compensation (GMC). Its replay also needs the original extracted PNGs; their
frozen file and pixel hashes are checked before providing BGR pixels to GMC.
It does not decode the source video or use detector-resized images.

`--class-id` can be repeated. IDs are the frozen detector's native classes, not
workspace taxonomy IDs: official COCO person is `1`, while trained detectors
carry their own class contract. Omitting the option selects all cached detector
classes. The adapter maintains independent native tracker state per class and
unique output IDs across those classes; two different classes cannot associate.

Replay retains all stored scores for the selected classes, even below the
tracker's low threshold, and preserves original `detection_index` values. Every
selected detection must appear exactly once as an observation or an unassigned
row. Excluded classes are listed separately. A saved score floor higher than
the requested low threshold is rejected: explicitly calculate another detector
cache at a sufficiently low supported floor, or explicitly raise the tracker's
low threshold. T2 floors cannot be lower than `0.001`. Native filtering, NMS and detection
caps remain irreversible; a low-floor cache is not raw detector proposals.

Only a complete cache can run. A fresh adapter is reset for each replay pass.
The default is two passes; `--repeats` accepts one through five. Reports publish
atomically only after every pass succeeds and never replace an existing file.
Cancellation, missing required pixels or a native failure leaves no report
claiming complete results. No partial replay is silently resumed.

## Profiles and the standalone interface

`iris-tracker-profile-v1` records every supported execution option. Defaults are
starting settings, not an optimized profile for a particular scene or camera.

| Option | ByteTrack | BoT-SORT |
| --- | --- | --- |
| High confidence threshold | `0.5` | `0.5` |
| Low confidence threshold | Fixed native `0.1` | `0.1` |
| New-track threshold | Native high threshold plus `0.1` | `0.6` |
| First association threshold | `0.8` | `0.8` |
| Lost buffer | `30` analyzed updates | `30` analyzed updates |
| Confidence fusion | Enabled | Enabled |
| Camera motion compensation | `none` | `sparseOptFlow`, downscale `2` |
| Learned ReID | Disabled | Disabled |
| OpenCV execution | One thread, seed `0` | One thread, seed `0` |

The profile supports explicit threshold, buffer, confidence-fusion, OpenCV
thread/seed settings and `none` GMC for BoT-SORT. It rejects unknown options,
invalid classes and settings that contradict the native implementation, rather
than ignoring them. Native high/low comparisons are strict: a score exactly at
the high threshold enters neither association pass. The second-pass threshold
is fixed at `0.5` and the unconfirmed association threshold at `0.7`; these native
constants and other association semantics are included in runtime metadata.
Native IoU uses `cython_bbox` inclusive-pixel overlap geometry; IRIS does not
replace it with a different box-distance implementation.

To use BoT-SORT without reading source images, create a complete profile:

```python
import json
from pathlib import Path

from iris.tracking_contracts import make_profile

profile = make_profile("botsort", class_ids=[1, 3], gmc_method="none")
Path("botsort-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
```

Pass `--profile botsort-profile.json` instead of `--tracker` and `--class-id`.
A complete profile cannot be partially overridden by those CLI options.

Other Python projects can use the adapter directly with their own ordered
detections. No Store object is required:

```python
from iris.tracking import make_tracker
from iris.tracking_contracts import make_profile

tracker = make_tracker(make_profile("bytetrack", class_ids=[1]))
tracker.reset("my-independent-sequence")
result = tracker.update(
    {
        "frame_id": "frame-0001",
        "frame_index": 0,
        "timestamp_seconds": 0.0,
        "input_size": [640, 480],
        "detections": [
            {
                "detection_index": 7,
                "label_id": 1,
                "label": "person",
                "score": 0.8,
                "box": [100, 80, 140, 190],
            }
        ],
    }
)
```

Call `reset` for each new sequence. Within one sequence, source indices and known
timestamps must strictly increase, class names and dimensions must stay stable,
and clock availability cannot change. For BoT-SORT sparse optical flow, also pass
`image=<original BGR uint8 array>` matching the frame dimensions. ByteTrack and
BoT-SORT with `gmc_method="none"` need no image. Native errors invalidate that
adapter's state; reset before replaying. Failure never triggers another algorithm
or disables GMC automatically.

## What each result means

`iris-tracking-frame-v1` preserves source frame identity, dimensions and timestamp
alongside the analyzed `update_index`:

- **Observations** retain the exact original detection box, score, label and
  detection index, with an associated `track_id`, `confirmed` state and separate
  internally estimated geometry. Measured boxes are never replaced by predictions.
- **Predictions** are native lost-track geometry with the last observed source
  frame, update, timestamp and ages. They have no fresh detector score. They can
  leave the image and are not evidence that the object is currently visible.
- **Unassigned detections** retain the source row and an explicit reason:
  confidence boundary, unmatched low confidence, insufficient birth confidence,
  unconfirmed native output or native suppression. ByteTrack emits confirmed
  observations; BoT-SORT can also emit unconfirmed ones with `confirmed: false`.
- **GMC evidence** identifies disabled compensation, initialization, an estimated
  affine transform or the native identity transform for insufficient matches.
  Featureless images or failed native optical flow/estimation can raise an error;
  they are not reported as successful camera motion estimation.

Track IDs are produced identities scoped to the replayed sequence. They do not
become T1 reference identities, human validation or a selected application target.
An increasing ID alone is not a measured identity error.

## Time, gaps and repeated runs

The native Kalman filters advance once per analyzed frame. An analyzed empty
frame receives an empty update. Missing/unanalyzed source frames remain explicit
sequence gaps and receive no synthetic updates; the runner never invents empty
scenes for them. The report retains the complete sequence manifest and its
`nominal_fps`, `provided` or `unknown` clock provenance.

`buffer_updates` describes analyzed update counts, **not seconds**. A source gap
may therefore advance the timestamp substantially while advancing native state
only once. The source timestamp supplies prediction age in seconds when known;
it does not change the native unit Kalman time step. Lost-track cleanup occurs
after native association, so the configured buffer is not a strict upper bound
on later reactivation. Tests preserve this native behavior explicitly.

Reports retain profile and cache/result hashes, original frame payload and
execution hashes, source manifests, pinned upstream/license/source provenance,
package/runtime metadata, and each pass's outputs. Semantic hashes exclude
timings; `observed_match` or `observed_mismatch` compares actual repeated outputs
on these inputs and this runtime. One pass records `not_checked`. This is not a
guarantee of identical output on all hardware or of identity quality.
Package/source provenance is checked again after each pass; runtime drift within
or between passes prevents publication of a complete report.

Each frame records GMC, association and adapter-total times. PNG loading,
verification and BGR conversion are measured separately. Pass totals include
the replay loop; adapter construction/reset has its own measurement. Detector
inference, reading/validating the cache, end-of-pass runtime verification and
writing the report are excluded.
Do not add overlapping timings or present cache replay speed as end-to-end
camera throughput. Broader cost and hardware comparisons belong to later work.

## Upstream code, licensing and updates

IRIS vendors bounded portions of the official MIT implementations:

- [ByteTrack](https://github.com/FoundationVision/ByteTrack), revision
  `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, copyright Yifu Zhang.
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT), revision
  `251985436d6712aaf682aaaf5f71edb4987224bd`, copyright Nir Aharon.

Original license texts, upstream and bundled file hashes, and exact adaptation
patches are included in `src/iris/_vendor`. The patches isolate imports, retain
NumPy compatibility, remove unused/optional ML imports, and carry detection
indices and GMC status without replacing association or Kalman algorithms.
Tracker construction verifies the bundled source and license hashes.

Updating a tracker is an explicit software change: pin the new upstream commit,
verify licenses, regenerate source/patch provenance, run adapter contract/native
tests and repeated cache replays, and retain earlier reports for comparison.
There is no runtime download, automatic upstream upgrade or silent replacement
of an existing saved profile. Standalone JSON reports outside the workspace are
not automatically included in workspace backups.
