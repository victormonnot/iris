# Portable trained-model export

[Documentation](README.md)

For the separate **YOLOX-Nano ONNX/OpenCV CPU** profile, see
[YOLOX ONNX bundles](yolox-onnx.md). That profile converts a graph and executes a
bounded numerical check. The native PyTorch profiles below retain their original
copy-only behavior and immutable manifests.

IRIS exports trained **Faster R-CNN MobileNetV3-Large 320 FPN** and
**SSDLite320 MobileNetV3-Large** checkpoints with a standalone CPU or NVIDIA CUDA
runner. The package contains the original PyTorch `state_dict`,
its frozen classes, the complete inference recipe, dependencies, saved reference
predictions, and the small set of reference images needed to check parity outside
IRIS. This is model export; dataset COCO export is a separate operation.

The native profiles support the builtin person/car head and custom heads trained
in IRIS, for all supported training scopes. It supports one full image per
forward pass. Tiled evaluation, official unmodified detectors, SAM, multimodal
providers, ONNX, TensorRT, quantization, and other architectures are outside
these profiles. Choose the export target independently of the training device:
CPU-to-CPU, CPU-to-CUDA, CUDA-to-CPU, and CUDA-to-CUDA use the same full checkpoint.
The saved reference evaluation may also have run on either CPU or CUDA.

## What is verified

The export preserves the checkpoint bytes and records the full SHA-256 and size
of every packaged file. Saved reference predictions must belong to that exact
checkpoint and its frozen class definitions. The reference is an existing IRIS
evaluation; packaging does not run a detector or manufacture new predictions.

The software tests cover manifest validation, class slots, coordinates, command
execution with simulated detector outputs, file integrity, and the measurement
protocol. A copied runner can inspect a bundle without IRIS or third-party
packages installed. Real acceptance trials also executed copied bundles in
separate CPU and CUDA environments with no IRIS installation and networking
disabled, on the same CPU/RTX 4060 host used for the pilot.

The trials covered both architectures after 40 light-scope training steps, each
trained on CPU and CUDA, with PyTorch 2.10.0, Torchvision 0.25.0 and Pillow 12.3.0.
Each of the four training-to-target paths used the same eight saved validation
images and three measurement repetitions, for eight original-reference exports:

| Training/reference device | Standalone target | Execution, both architectures | Exact parity |
| --- | --- | --- | --- |
| CPU | CPU | Completed | Passed |
| CPU | CUDA | Completed | Failed |
| CUDA | CPU | Completed | Failed |
| CUDA | CUDA | Completed | Passed |

The four cross-device failures affected all 24 samples per export. Native
detection counts were unchanged; the largest observed coordinate difference was
about 0.000214 pixels and the largest score difference was about 0.00000167.
True positives, false positives and misses at confidence 0.5 and IoU 0.5 were
unchanged on this subset. Predictions were identical across the three repetitions
on each target. These observations do not change the zero-tolerance parity result
or establish accuracy on an independent test set.

Four additional control exports used new IRIS references evaluated on the
respective target device, preserving the same checkpoints, images and strict
comparison. All four passed exact parity. The eight original exports and their
four failed measurements remain unchanged; new references do not turn a failed
cross-device comparison into a pass. In total, the 12 measurements contain
288 samples.

Timing measurements describe these models, images and runtimes on this host.
Separately sampled process resident memory and aggregate GPU memory use are
descriptive observations, not allocator peaks or a portable memory requirement.
Other hardware, embedded systems, longer-trained models and other training scopes
still need their own execution, parity and performance checks.

The later [R10 street-vehicle trial](acceptance-results.md#r10-completed-street-vehicle-workflow-weak-detector-quality)
also exported a two-class SSDLite car/bus head after 40 light-scope CUDA steps.
The copied runner used the same isolated standalone CUDA environment and compared
two saved validation images across three repetitions. All six samples passed
strict parity for the complete native outputs, despite weak detector quality at
the chosen confidence threshold. This extends the measured custom-class export
coverage on that host; it does not establish accuracy or compatibility elsewhere.

Every new package declares `real_execution: "not_run"` because packaging itself
does not run inference. Subsequent measurements are separate records and do not
rewrite that immutable manifest. Packaging and successful hash inspection alone
do not establish that a checkpoint loads or performs well on another machine.

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

Faster R-CNN CPU-reference/CPU-target exports use `iris-model-export-v1`, freezing the
`iris-torchvision-trained-cpu-v1` profile in
`src/iris/export_runner.py`, and contain the evaluation, evaluation-model row,
dataset, dataset-manifest hash, and checkpoint identities. Its inventory excludes
`manifest.json` itself. The manifest hash is computed over sorted, compact UTF-8
JSON with no trailing newline. The reference file uses the same canonical
encoding. Image hashes cover the encoded PNG bytes, rather than IRIS's separate
RGB pixel hash.

A Faster R-CNN CUDA target or CUDA reference uses `iris-model-export-v2` and the
`iris-torchvision-trained-native-v2` profile. Its `profile.device` freezes the
target family (`cpu` or `cuda`), and `source.reference_device` records the saved
evaluation device. Version 1 manifests and measurements remain unchanged and
readable. Existing package files are validated against their own frozen inventory,
without requiring that their runner match the current IRIS source code.

SSDLite uses `iris-model-export-v3` and the distinct
`iris-torchvision-ssdlite-native-v1` profile for both CPU and CUDA targets. It
records the target and reference devices just like version 2. Its builder,
normalization, fixed input size, and native filtering are bound to SSDLite;
substituting a Faster R-CNN architecture or recipe is rejected. The original
Faster R-CNN profile bytes and existing package contracts remain unchanged.

Reference sets contain one to eight images, in a frozen order, and their complete
native predictions. The runner accepts checkpoint files up to 1 GiB, individual
images up to 32 MiB and 64 million pixels, and textual files up to 2 MiB. Bundle
paths are relative and cannot contain traversal or symbolic links. File hashes
detect changes; they do not authenticate an external publisher.

## Run outside IRIS

The profile requires Python 3.12 or 3.13, PyTorch 2.10.0, Torchvision 0.25.0, and
Pillow 12.3.0. Provision these dependencies on the target machine using the
appropriate CPU or CUDA PyTorch and Torchvision distributions. The runner never installs dependencies,
downloads weights, contacts an API, or requires access to an IRIS workspace.
Inspection uses only Python's standard library:

```sh
python /path/to/export/run.py inspect
python /path/to/export/run.py check-runtime
python /path/to/export/run.py predict /path/to/image.png --output /path/to/prediction.json
python /path/to/export/run.py measure --repeats 3 --output /path/to/measurement.json
```

For a CUDA target, add `--device cuda:0` (or another visible index) to
`check-runtime`, `predict`, or `measure`. Omitting the option uses the current CUDA
device. The selected family must match the frozen target; there is no automatic
fallback to CPU. Create another export to measure the same model on a different
device family. `check-runtime` explicitly imports the installed packages and
queries the selected device, without constructing a detector or loading weights.
For CUDA it also requires registered Torchvision CUDA detection operators
(`nms` for SSDLite; `nms` and `roi_align` for Faster R-CNN) and a GPU architecture
supported by the installed PyTorch build.
Success only confirms the dependency/device probe; real inference still needs to
be tested.

An ARM or embedded GPU is a separate deployment environment. Jetson installations
need a compatible board, JetPack release, and vendor PyTorch/Torchvision builds;
the versions pinned by this profile may be unavailable on older boards. CUDA
support does not imply that every Jetson or GPU can run this package. Consult
[NVIDIA's Jetson installation guidance](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
when provisioning the target. IRIS does not create TensorRT engines or change the
target's dependencies automatically.

The default bundle directory is the runner's own directory. To select another
directory, place `--bundle /path/to/export` before the command. Output files must
be new files outside the immutable bundle directory; existing files are never
replaced. Commands return exit code 0 on success, 2 on invalid input or runtime
failure, and 3 when a completed measurement finds a parity mismatch. A mismatch
still writes its full measurement report for inspection and import into IRIS.

## Frozen inference recipe

Both builders use `weights=None`, `weights_backbone=None`, and the frozen class
count plus the background slot. The runner loads the local checkpoint with
`weights_only=True`, `map_location="cpu"`, and
`load_state_dict(strict=True)`. It then transfers the model to the selected CPU or
CUDA device and uses evaluation mode and float32 inference. Threads are capped at
`min(4, os.cpu_count() or 1)`. CUDA execution disables TF32 for matrix multiplication
and cuDNN and disables cuDNN benchmarking. The same checkpoint can therefore be
packaged for either device independently of its training origin.

Images receive EXIF orientation correction, RGB conversion, and CHW float32
conversion divided by 255. Normalization, resizing, proposal filtering, NMS,
and restoration to original oriented coordinates are part of the Torchvision
forward. There is no second external resize or normalization.

| Recipe | Faster R-CNN MobileNetV3 | SSDLite320 MobileNetV3 |
| --- | --- | --- |
| Builder | `fasterrcnn_mobilenet_v3_large_320_fpn` | `ssdlite320_mobilenet_v3_large` |
| Normalization layers | Backbone `FrozenBatchNorm2d`, epsilon `1e-5` | Builder `BatchNorm2d`, epsilon `0.001`, momentum `0.03` |
| Image mean | `[0.485, 0.456, 0.406]` | `[0.5, 0.5, 0.5]` |
| Image standard deviation | `[0.229, 0.224, 0.225]` | `[0.5, 0.5, 0.5]` |
| Resize | Short edge 320, maximum long edge 640 | Fixed 320 × 320 |
| Padding divisor | 32 | 1 |
| Candidate filtering | RPN settings below | Top 300 candidates per class |

SSDLite retains its ordinary BatchNorm layers and saved running statistics; the
Faster R-CNN frozen normalization conversion is never applied to it. Both use
evaluation mode at inference, regardless of the training scope.

Final box score threshold is 0.001, box NMS IoU threshold is 0.5, and maximum
detections per image is 100 for both architectures. Faster R-CNN's inference RPN
score threshold is 0.05, its NMS
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
[SSDLite builder](https://raw.githubusercontent.com/pytorch/vision/v0.25.0/torchvision/models/detection/ssdlite.py),
[SSD transform setup](https://raw.githubusercontent.com/pytorch/vision/v0.25.0/torchvision/models/detection/ssd.py),
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
Different CPU/CUDA kernels may produce numerical or ordering differences; these
remain explicit parity failures rather than receiving looser tolerances.
The checker never silently widens the tolerance, changes ordering, or filters
small scores to produce a passing result. Parity against saved predictions checks
reproduction, not annotation accuracy or generalization to new images.

Timing scopes are explicit:

- `load_ms`: runtime setup, model construction, checkpoint integrity check and loading.
- `warmup.duration_ms`: one full detector call; excluded from measured samples.
- `decode_ms`: file read, image hash verification and image decoding.
- `preprocess_ms`: orientation, RGB conversion, tensor creation and device transfer.
- `inference_ms`: full Torchvision forward, including resize, proposals and NMS.
- `postprocess_ms`: CPU result conversion and output validation.
- `total_ms`: the three detector stages; excludes decode, load, warmup and JSON writing.

Bundle validation precedes the loading timer. Timing summaries use the recorded
minimum, median and maximum for each stage. Measurements include raw samples so
different hardware, versions, class sets, image sizes and negative examples can
be assessed without hiding variability. The `total_ms` used by IRIS evaluation
includes decode; compare matching timing scopes rather than those totals directly.
CUDA timing synchronizes the selected device before prediction and after each
timed stage. Its measurement environment records the CUDA runtime, cuDNN version,
visible device index, GPU name, compute capability, total memory and precision
flags. These timing boundaries follow
[PyTorch's asynchronous CUDA execution guidance](https://docs.pytorch.org/docs/2.10/notes/cuda.html#asynchronous-execution).

Imported reports are retained as **declared external evidence**. IRIS recomputes
parity and summaries from the samples and checks the frozen protocol. It cannot
authenticate execution or the claimed hardware of an imported JSON file.
`simulation` reports remain explicitly simulated, and `external_execution`
reports remain unverified declarations even when parity passes.

To package a native detector together with a measured tracker profile and optional
selection settings, use [portable pipeline bundles](pipeline-bundles.md). That
format includes a standalone inspector; its tracking runtime and pipeline parity
checks are separate from the detector-only runners documented here.
