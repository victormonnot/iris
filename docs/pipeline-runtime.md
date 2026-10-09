# Standalone detector and tracking runtime

[Documentation](README.md)

IRIS 0.56 exports a complete native image/frame/video pipeline in each new
[v2 pipeline bundle](pipeline-bundles.md). The consumer runs the frozen detector,
ByteTrack or BoT-SORT and optional selected-object policy without an IRIS server,
workspace, database or ARGOS dependency. Opening or exporting a bundle never
installs dependencies or executes the model.

## Extract and prepare

Use installed IRIS to inspect an archive as data and extract it into a **new**
directory. Extraction checks the complete inventory and hashes before publishing
that directory; it never executes code from the archive or replaces a destination.
Hashes verify consistency, not the identity of the package author.

```sh
iris pipeline inspect pipeline.zip
iris pipeline extract pipeline.zip --to ./native-pipeline
```

The package contains its checkpoint, Python runtime, `README.md`, `requirements.txt`,
`inspect_bundle.py`, `run.py`, `example.py`, native tracker sources and licence notices.
Historical v1 archives remain inspectable and extractable but do not contain an
executable pipeline. Their saved manifests are not upgraded in place.

Prepare a separate Python 3.12 or 3.13 environment on Linux. Install matching
PyTorch 2.10.0 and Torchvision 0.25.0 builds for the frozen CPU or NVIDIA CUDA
family, then the exact packages in `requirements.txt`. The runtime records the
actual build versions, device and tracker provenance. It does not silently
substitute CPU when CUDA is unavailable. `--device cuda:1` selects a particular
GPU within a CUDA bundle's declared family.

```sh
python -B ./native-pipeline/run.py check-runtime
```

This explicitly validates package files, dependencies and native operator support,
then loads the local checkpoint with `weights_only=True` and strict state loading.
It performs no download and reports errors for missing or incompatible packages.
The first deployment matrix covers native float32 SSDLite, Faster R-CNN and
YOLOX-Nano, with their official and IRIS-trained class mappings. Tracking runs on
CPU. ONNX, TensorRT, tiled detectors and learned appearance ReID are not part of
this runtime. An arbitrary Jetson or other embedded board still needs its own
compatible dependency build and target measurements.

## Process a video

Keep inputs and outputs outside the extracted bundle:

```sh
python -B ./native-pipeline/run.py video /data/clip.mp4 \
  --max-frames 500 --output /data/clip-run.json
```

The runner accepts regular local AVI, MP4, MOV, MKV and WebM files. It applies
container orientation to decoded RGB images. It declares timestamps as source
indices divided by the container's **nominal FPS**; these are not certified
capture timestamps or a simulation of a live camera. Use `--clock unknown` to emit
null timestamps when that assumption is inappropriate. Variable frame-rate footage
requiring its actual timestamps should use the timestamped frame interface.

Every available decoded image creates one tracker update. The output reports
whether processing stopped at the requested limit or the decoder stopped yielding
frames. Decoder end-of-stream alone does not prove that a damaged video was fully
decoded. A limit processes an explicit prefix, not an undocumented sample.

For a bundle with a selection policy, `--events /data/events.json` supplies explicit
application decisions by zero-based decoded frame index:

```json
{
  "schema": "iris-pipeline-events-v1",
  "events": [
    {"frame_index": 0, "select_detection_index": 0},
    {"frame_index": 120, "release": true}
  ]
}
```

Selection addresses the current detector output's `detection_index`, which must
correspond to a confirmed measured tracker observation above the policy's score
gate. It is not a track number or an instruction to select the first person
forever. Invalid events fail explicitly. The runner never chooses a target itself.

## Supply timestamped images

The `frames` command accepts a JSON manifest with a declared clock and image hashes:

```json
{
  "schema": "iris-pipeline-input-v1",
  "sequence_id": "flight-01",
  "clock_kind": "provided",
  "frames": [
    {
      "frame_id": "frame-0000",
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "path": "frames/0000.png",
      "file_sha256": "REPLACE_WITH_THE_64_CHARACTER_SHA256_OF_THE_IMAGE_FILE",
      "input_size": {"width": 640, "height": 480},
      "select_detection_index": 0,
      "release": false
    }
  ]
}
```

```sh
python -B ./native-pipeline/run.py frames /data/input.json \
  --output /data/frame-run.json
```

Image paths are relative to the input JSON's directory and cannot escape it or
use symbolic links. Each file hash is checked before decoding. Dimensions describe
the RGB image after EXIF orientation. Input frames must retain unique IDs,
strictly increasing source indices and increasing finite timestamps. With
`clock_kind: "unknown"`, every timestamp must be null. Selection fields are
optional; omit them for detection/tracking alone.

## Integrate frame by frame

Place the extracted package on the consumer's Python import path. `example.py`
shows a minimal caller. Start the consumer with `python -B` or set
`PYTHONDONTWRITEBYTECODE=1` before importing the package: the extracted directory
rejects bytecode caches and any other undeclared files. An application can own
its camera or decoder and supply
oriented RGB Pillow images directly:

```python
from iris_bundle.pipeline_runtime import Pipeline

pipeline = Pipeline("/path/to/native-pipeline")
pipeline.reset("recording-01", clock_kind="provided")
result = pipeline.update(
    rgb_image,
    frame_id="frame-0000",
    frame_index=0,
    timestamp_seconds=0.0,
)
```

Continue in source order. The image shape must stay fixed within a sequence.
An explicitly skipped source index does not insert synthetic empty updates.
Provide `select_detection_index=...` on a chosen frame and `release=True` on a
later frame. A released or expired target can be followed by a new explicit
selection; its local logical selection number increments. Reset creates a new
sequence and clears track/selection state. A failed update requires an explicit
reset before further updates, including input validation failures.

The runtime uses the same native tracker adapters and selected-object transitions
as IRIS. BoT-SORT camera compensation receives the original oriented BGR image.
Predicted tracks stay separate from detector measurements, and ambiguity,
confirmation, loss, expiry and release retain their distinct states. See the
[tracker](tracking.md) and [selected-object](selected-object.md) contracts.

A pipeline is bounded to 10,000 updates per reset. Upstream tracker histories
retain entries during that interval; this is not an unlimited-duration streaming
memory guarantee. The CLI additionally bounds images to 32 MiB and 64 million
pixels and complete reports to 128 MiB. Break longer workloads into explicit
sequences, accepting the corresponding reset of track identities. The calling
application owns scheduling, stale frames, control decisions and device I/O.

## Inspect results and compare parity

Each `iris-pipeline-run-v1` report records:

- The bundle manifest hash, declared clock, actual runtime and local input provenance.
- Decoded pixel hashes, frame identity, dimensions and explicit selection events.
- All detector outputs with their stable indices, measured tracker observations,
  predictions, unassigned detections, camera compensation and optional selection state.
- A semantic hash that excludes measured wall-clock durations.

Results publish atomically to a new file only after the requested processing
succeeds. Existing files are never overwritten. Keep execution reports next to
an application's test evidence; the immutable bundle's `not_run` declarations
are historical packaging facts and are not silently promoted.

Compare an explicit reference run with a candidate run:

```sh
python -B ./native-pipeline/run.py compare /data/reference.json \
  /data/candidate.json --output /data/parity.json
```

Comparison needs no ML libraries. It validates both reports and compares source
pixels, events, clocks, detector outputs, tracker semantics and selection states
exactly. Timings, runtime hardware and image/container encoding are excluded;
numerical differences are not hidden behind a tolerance. A mismatch is saved with
its frame indices and returns exit code 2; malformed evidence fails without a
successful parity report. File hashes bind the compared evidence but do not
attest its author or establish that a purported reference came from IRIS.

Matching outputs on a frozen clip establishes execution parity on that clip.
It does not establish improved detection quality, reliable physical identity,
real-time camera latency, performance on a laptop or independent qualification.
CPU and CUDA numerical outputs may differ; compare each target against an explicit
IRIS reference on that same target and record cross-device differences separately.

For a new application, follow the [independent qualification protocol](pipeline-qualification.md)
to separate portability checks from reference quality and target-device evidence.
