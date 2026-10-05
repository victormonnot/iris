# Training and evaluating custom classes

Use **Manage classes** to define stable class IDs and annotation rules, review the
images, then freeze one class version in **Dataset & training**. The release saves
the definitions, images, labels and split assignments used by every later run.
An optional official COCO category mapping is an explicit assertion that a class
has the same meaning as that source category. Names alone never create mappings.

## Parent checkpoint and training

The supported trainable architecture is Faster R-CNN MobileNetV3-Large 320 FPN.
Start from its provisioned official weights or an IRIS checkpoint with the exact
same frozen class snapshot and mappings. A trained parent from another class
version is incompatible even if its labels have the same names. Start from official
weights for the new definitions; existing datasets and checkpoints remain usable.

The prediction head has N object classes plus background. With an official parent,
IRIS copies background and the classifier and box-regression rows for explicitly
mapped COCO categories. Classes without mappings retain seeded initialization.
With a compatible trained parent, the learned head is retained. The chosen light,
partial or full training depth then determines which parameters can change.
Initialization and class mappings are recorded with the training settings.

Only the frozen training split is read by the optimizer. At least one training
image must contain a positive annotation; validated negatives are also supported.
Inherited training groups and image hashes cannot appear in held-out evaluation
data. Use the plan preview before starting a bounded CPU run. A successful run
records a checkpoint hash and full class definitions and returns the checkpoint
to the model catalog. Loss is not a quality measurement.

Runs support 1 to 10,000 CPU optimizer steps. The interface saves recovery state
every 50 steps by default, with the interval reviewed in the plan. A stopped
attempt with a saved state can be continued explicitly in a new attempt with
the same data, settings and runtime. Continuation preserves SGD momentum and
RNG state; starting another fine-tuning run from a completed checkpoint uses a
new optimizer. See [longer training and continuation](long-training.md) for
the interval limits, storage policy, timing scope and legacy API behavior.

## Saved inference and annotation

Custom checkpoint outputs use the stable class ID as their label, with numeric
output IDs 1…N in the saved class order. Native head IDs are also retained. The
original Person / Car checkpoints continue to output person=1 and car=3, with
native head slots 1 and 2. Output IDs belong to that checkpoint's namespace.

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

Checkpoints contain a tensor state dictionary, with architecture, N+1 head size,
class snapshot, input transform, hashes and training provenance recorded by IRIS.
Reload reconstructs the architecture and exact head before strict state loading.
This checks that custom heads can be saved and reloaded without relying on current
project definitions. The [standalone model export workflow](model-export.md)
packages a completed trained checkpoint, its frozen classes and inference recipe,
a CPU PyTorch runner, and saved evaluation examples for external parity checks.
Real exported-model execution and target performance remain to be tested. No
ONNX, TensorRT or embedded-runtime export is provided. Internal optimizer
recovery states cannot be used as inference exports.

The existing CPU scopes and step limits apply to custom classes. Multimodal
candidate review retains its original Person / Car scope. Direct detector
preannotation and disagreement review support compatible frozen custom classes.
Synthetic fixtures verify the software path, including state continuation, not
detector quality on real images. Real detector training and resume trials remain
deferred. No model weights or datasets are downloaded automatically.
