# Training and evaluating custom classes

[Documentation](README.md)

Use **Manage classes** to define stable class IDs and annotation rules, review the
images, then freeze one class version in **Dataset & training**. The release saves
the definitions, images, labels and split assignments used by every later run.
An optional official COCO category mapping is an explicit assertion that a class
has the same meaning as that source category. Names alone never create mappings.

## Parent checkpoint and training

The supported trainable architectures are Faster R-CNN MobileNetV3-Large 320 FPN,
SSDLite320 MobileNetV3-Large and YOLOX-Nano. Start from provisioned official weights
or an IRIS checkpoint with the exact same frozen class snapshot and mappings.
A trained parent from another class version is incompatible even if its labels
have the same names. Start from official weights for the new definitions;
existing datasets and checkpoints remain usable.

The two Torchvision models use N object classes plus background. With an official
parent, IRIS copies background and explicitly mapped COCO classifier rows.
Faster R-CNN also copies per-class box-regression rows; SSDLite preserves its
class-independent regression and maps classifier rows separately for every anchor.
YOLOX uses N foreground classes with no background class. It copies explicitly
mapped COCO classifier rows and retains the box and objectness heads. Classes
without mappings retain seeded initialization in all three architectures.
With a compatible trained parent, the learned head is retained. The chosen light,
partial or full training depth then determines which parameters can change.
Initialization and class mappings are recorded with the training settings. See
[trainable model choices](trainable-models.md) for architecture-specific depths,
frozen normalization and negative-image loss policies. The
[YOLOX training guide](yolox-training.md) describes its input format, loss and
gradient clipping recipe.

Only the frozen training split is read by the optimizer. At least one training
image must contain a positive annotation; validated negatives are also supported.
Inherited training groups and image hashes cannot appear in held-out evaluation
data. Use the plan preview before starting a bounded CPU or NVIDIA CUDA run. A successful run
records a checkpoint hash and full class definitions and returns the checkpoint
to the model catalog. Loss is not a quality measurement.

Runs support 1 to 10,000 optimizer steps on CPU or one selected NVIDIA GPU, using
float32 and batch size one. Training depth has the same meaning on either device.
The interface saves recovery state
every 50 steps by default, with the interval reviewed in the plan. A stopped
attempt with a saved state can be continued explicitly in a new attempt with
the same data, settings and runtime. Continuation preserves SGD momentum and
CPU RNG and, for CUDA, selected GPU RNG state; starting another fine-tuning run from a completed checkpoint uses a
new optimizer. See [longer training and continuation](long-training.md) for
the interval limits, storage policy, timing scope and legacy API behavior.

## Saved inference and annotation

Custom checkpoint outputs use the stable class ID as their label, with numeric
output IDs 1…N in the saved class order. Checkpoints using the original Person /
Car definitions retain output IDs person=1 and car=3. Saved `native_label_id`
values use IRIS's one-based foreground order, so these two classes have IDs 1 and 2.
Torchvision heads use those slots directly; YOLOX's network channels are zero-based
and its adapter adds one before serialization. Output IDs belong to that
checkpoint's namespace, not a universal category list.

Comparisons save per-model class contracts when a custom checkpoint is involved.
They support full-image and tiled inference and preserve their original labels
after the project changes. Predictions from a matching custom checkpoint can
be imported as pending annotation proposals without a COCO mapping. They require
human review and never overwrite validated boxes automatically. Different saved
class versions cannot silently become equivalent through a numeric ID collision.

## Evaluation and reports

Evaluate one or two compatible checkpoints on the same frozen validation split.
Trained checkpoints must match the complete class snapshot. An official baseline
requires a COCO mapping for every target class; a baseline missing one custom
class is rejected. For a class outside COCO, compare compatible trained versions
or evaluate one checkpoint on its own. Unmapped official predictions remain
recorded as explicitly ignored source categories.

Metrics, error filters, overlays and reports cover every saved class. Classes with
no reference objects have undefined AP and are excluded from macro AP; false
positives still count, including on negative images. Test audits reuse a completed
validation evaluation's checkpoints and settings. Reports retain definitions and
results independently of later project edits and can be exported as offline HTML.
See [the evaluation protocol](evaluation.md) and [experiment reports](experiments.md).

## Checkpoint portability and current limits

Checkpoints contain a tensor state dictionary, with architecture, head size,
class snapshot, input transform, hashes and training provenance recorded by IRIS.
Reload reconstructs the architecture and exact head before strict state loading.
This checks that custom heads can be saved and reloaded without relying on current
project definitions. The head contains N+1 class slots for Torchvision and N for
YOLOX. Export then depends on the architecture:

- [Native Torchvision bundles](model-export.md) contain the completed checkpoint,
  frozen classes, inference recipe, PyTorch CPU or CUDA runner and saved evaluation
  examples for external parity checks.
- [YOLOX ONNX bundles](yolox-onnx.md) convert the checkpoint and provide an OpenCV
  CPU runner. They check numerical agreement during conversion and record exact
  saved-prediction parity separately.

Real exports of the pilot's person-only Torchvision heads have run in separate
CPU and CUDA environments on the same host. Same-device exact parity passed; cross-device
exact parity failed despite unchanged detection counts at the measured operating
point. See the [measured export matrix](model-export.md#what-is-verified).
The later [street-vehicle acceptance](acceptance-results.md#r10-completed-street-vehicle-workflow-weak-detector-quality)
also exercised a two-class SSDLite car/bus head on CUDA: real training, reload,
evaluation and standalone export completed, with exact parity on six samples.
Its detector quality remained poor; workflow completion is not a quality gain.
Other class sets and physical target machines need their own checks. The native
Torchvision profiles do not provide ONNX or TensorRT conversion.
Use the PyTorch runner on an embedded target only when that target satisfies its
runtime and operator requirements. Internal optimizer recovery states cannot be
used as inference exports.

Completed checkpoint tensors are saved on CPU and can be loaded for inference on
CPU or CUDA independently of the training device. Choose the export target for
the destination machine, then check the target runtime and measure parity there.
An interrupted optimizer state requires its original training device and runtime;
it cannot be used to transfer an in-progress run between CPU and GPU. See
[compute targets](compute-targets.md) for setup and platform compatibility.

The existing scopes and step limits apply to custom classes. Multimodal
candidate review retains its original Person / Car scope. Direct detector
preannotation and disagreement review support compatible frozen custom classes.
Synthetic fixtures verify the software path, including state continuation. Real
40-step acceptance trials covered the two Torchvision architectures with light
training on CPU and light, partial and full training on an RTX 4060 using
PyTorch 2.10.0 / Torchvision 0.25.0 `cu128`. Completed checkpoints were reloaded in separate
evaluation workers and evaluated on human-reviewed images from a separate source
context. Light-scope runs also continued after cancellation and forced worker
termination on their original CPU or CUDA device and runtime.

These short trials do not establish general quality gains. Full training is not
necessarily better, and learning rates must be assessed for each architecture and
depth. For those Torchvision trials, longer runs, partial/full training on CPU,
partial/full-scope recovery, server restart or power-loss recovery and other
hardware remain untested. Custom-head exports beyond the measured person-only
and CUDA SSDLite car/bus cases need their own validation.
Cross-device execution and its exact-parity limits
are recorded in the export matrix above.

A later [YOLOX acceptance cycle](acceptance-results.md#yolox-nano-custom-detector-accepted-by-an-external-application)
completed 400-step head-only and 800-step partial-backbone CUDA runs, CPU reload
and ONNX export. It reused a small development validation set; it does not extend
the Torchvision recovery evidence or establish general quality gains. No model
weights or datasets are downloaded automatically.
