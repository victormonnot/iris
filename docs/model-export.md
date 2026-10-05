# Portable trained-model export

IRIS exports a trained **Faster R-CNN MobileNetV3-Large 320 FPN** checkpoint with
a standalone CPU runner. The package contains the original PyTorch `state_dict`,
its frozen classes, the complete inference recipe, dependencies, saved reference
predictions, and the small set of reference images needed to check parity outside
IRIS. This is model export; dataset COCO export is a separate operation.

This first profile supports the builtin person/car head and custom heads trained
in IRIS, for all supported training scopes. It supports one full image per
forward pass. Tiled evaluation, official unmodified detectors, SAM, multimodal
providers, ONNX, GPU execution, quantization, and other architectures are outside
this profile.

## What is verified

The export preserves the checkpoint bytes and records the full SHA-256 and size
of every packaged file. Saved reference predictions must belong to that exact
checkpoint and its frozen class definitions. The reference is an existing IRIS
evaluation; packaging does not run a detector or manufacture new predictions.

The software tests cover manifest validation, class slots, coordinates, command
execution with simulated detector outputs, file integrity, and the measurement
protocol. A copied runner can inspect a bundle without IRIS or third-party
packages installed. **Real exported-model execution, numerical parity, and target
hardware performance are deferred until the planned real-model testing phase.**
Every new package therefore declares `real_execution: "not_run"`. Packaging and
successful hash inspection do not establish that a real checkpoint loads or
performs well on another machine.

## Package contents

```text
manifest.json
model.pth
run.py
requirements.txt
README.md
parity/reference.json
parity/images/<frame_id>.png
```

The manifest uses `iris-model-export-v1`, freezes the `PROFILE` in
`src/iris/export_runner.py`, and contains the evaluation, evaluation-model row,
dataset, dataset-manifest hash, and checkpoint identities. Its inventory excludes
`manifest.json` itself. The manifest hash is computed over sorted, compact UTF-8
JSON with no trailing newline. The reference file uses the same canonical
encoding. Image hashes cover the encoded PNG bytes, rather than IRIS's separate
RGB pixel hash.

Reference sets contain one to eight images, in a frozen order, and their complete
native predictions. The runner accepts checkpoint files up to 1 GiB, individual
images up to 32 MiB and 64 million pixels, and textual files up to 2 MiB. Bundle
paths are relative and cannot contain traversal or symbolic links. File hashes
detect changes; they do not authenticate an external publisher.

## Run outside IRIS

The profile requires Python 3.12 or 3.13, PyTorch 2.10.0, Torchvision 0.25.0, and
Pillow 12.3.0. Provision these dependencies on the target machine using the
appropriate PyTorch CPU distributions. The runner never installs dependencies,
downloads weights, contacts an API, or requires access to an IRIS workspace.
Inspection uses only Python's standard library:

```sh
python /path/to/export/run.py inspect
python /path/to/export/run.py predict /path/to/image.png --output /path/to/prediction.json
python /path/to/export/run.py measure --repeats 3 --output /path/to/measurement.json
```

The default bundle directory is the runner's own directory. To select another
directory, place `--bundle /path/to/export` before the command. Output files must
be new files outside the immutable bundle directory; existing files are never
replaced. Commands return exit code 0 on success, 2 on invalid input or runtime
failure, and 3 when a completed measurement finds a parity mismatch. A mismatch
still writes its full measurement report for inspection and import into IRIS.

## Frozen inference recipe

The runner builds `fasterrcnn_mobilenet_v3_large_320_fpn` with `weights=None` and
`weights_backbone=None`. It uses the frozen class count plus the background slot,
replaces backbone `BatchNorm2d` modules with `FrozenBatchNorm2d(eps=1e-5)`, and
loads the local checkpoint with `weights_only=True`, `map_location="cpu"`, and
`load_state_dict(strict=True)`. It then uses evaluation mode and float32 CPU
inference. Threads are capped at `min(4, os.cpu_count() or 1)`.

Images receive EXIF orientation correction, RGB conversion, and CHW float32
conversion divided by 255. Normalization, resizing, proposal filtering, NMS,
and restoration to original oriented coordinates are part of the Torchvision
forward. The frozen profile records mean `[0.485, 0.456, 0.406]`, standard
deviation `[0.229, 0.224, 0.225]`, short-edge size 320, maximum long edge 640, and
padding divisible by 32. There is no second external resize or normalization.

Final box score threshold is 0.001, box NMS IoU threshold is 0.5, and maximum
detections per image is 100. The inference RPN score threshold is 0.05, its NMS
IoU threshold is 0.7, and its pre/post-NMS limits are 150. The training-only RPN
threshold of 0.0 is not an inference setting.

Boxes use `xyxy` pixels relative to the oriented image, with exclusive right and
bottom edges. Native scores and detection order are preserved. The builtin head
maps native slot 1 to person/category 1 and slot 2 to car/category 3. Custom heads
use their own ordered slots and category IDs, even if a class carries an explicit
COCO mapping. Each prediction keeps `native_label_id`; custom predictions also
carry their frozen `taxonomy_id`. The runner preserves the legacy omission of
that per-prediction field for the builtin head.

These settings follow the existing IRIS adapter and the pinned upstream
[Torchvision builder](https://raw.githubusercontent.com/pytorch/vision/v0.25.0/torchvision/models/detection/faster_rcnn.py),
[transform](https://raw.githubusercontent.com/pytorch/vision/v0.25.0/torchvision/models/detection/transform.py),
and [FrozenBatchNorm implementation](https://raw.githubusercontent.com/pytorch/vision/v0.25.0/torchvision/ops/misc.py).
The [PyTorch 2.10 loading API](https://docs.pytorch.org/docs/2.10/generated/torch.load.html)
documents the restricted state-dictionary loading mode.

## Parity and target measurements

The measurement format is `iris-export-measurement-v1`. It binds the canonical
manifest hash, records the complete runtime and hardware declaration, loads the
model once, and performs one unmeasured warmup on the first reference image.
It then processes every reference image for each of one to ten repetitions,
in repetition-major order. Every sample contains its frame ID, repetition,
dimensions, complete predictions, and durations.

Parity requires identical dimensions, detection count and order, class identities,
boxes, and scores. Absolute and relative tolerances are zero. A legitimate numeric
or count mismatch produces a failed parity result; malformed evidence is rejected.
The checker never silently widens the tolerance, changes ordering, or filters
small scores to produce a passing result. Parity against saved predictions checks
reproduction, not annotation accuracy or generalization to new images.

Timing scopes are explicit:

- `load_ms`: runtime setup, model construction, checkpoint integrity check and loading.
- `warmup.duration_ms`: one full detector call; excluded from measured samples.
- `decode_ms`: file read, image hash verification and image decoding.
- `preprocess_ms`: orientation, RGB conversion and tensor creation.
- `inference_ms`: full Torchvision forward, including resize, proposals and NMS.
- `postprocess_ms`: CPU result conversion and output validation.
- `total_ms`: the three detector stages; excludes decode, load, warmup and JSON writing.

Bundle validation precedes the loading timer. Timing summaries use the recorded
minimum, median and maximum for each stage. Measurements include raw samples so
different hardware, versions, class sets, image sizes and negative examples can
be assessed without hiding variability. The `total_ms` used by IRIS evaluation
includes decode; compare matching timing scopes rather than those totals directly.

Imported reports are retained as **declared external evidence**. IRIS recomputes
parity and summaries from the samples and checks the frozen protocol. It cannot
authenticate execution or the claimed hardware of an imported JSON file.
`simulation` reports remain explicitly simulated, and `external_execution`
reports remain unverified declarations even when parity passes.
