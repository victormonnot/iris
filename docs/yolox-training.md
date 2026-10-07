# YOLOX-Nano training

IRIS supports `yolox_nano` as an official COCO detector and as a parent for custom
project classes. Training, evaluation and recovery use the same frozen datasets,
durable jobs and class contracts as the other local detectors. No images leave the
machine and model loading never downloads weights implicitly.

The implementation vendors the minimal model and loss sources from
[YOLOX 0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/419778480ab6ec0590e5d3831b3afb3b46ab2aa3).
The Apache 2.0 license, original source hashes and modifications are retained in
`src/iris/_vendor/yolox/`. It does not install the historical YOLOX package or build
its optional CUDA extensions.

## Input and class contract

- Fixed `416 × 416`, BGR, float32 values in `[0, 255]`.
- OpenCV bilinear resizing preserves aspect ratio; padding is top-left aligned,
  value `114`. EXIF orientation precedes inference preprocessing.
- The official model has 80 foreground classes and **no background class**.
  Official compact indices are mapped to sparse COCO category IDs before IRIS
  returns detections.
- A custom model has exactly one output channel per frozen project class. The
  native network uses zero-based class indices; IRIS retains its one-based class
  mapping at the annotation and evaluation boundary.
- Only explicitly mapped COCO classes copy their official classification rows.
  New classes receive seeded initialization. Box and objectness heads are kept.
  Continuing a trained model requires the exact same frozen class definitions.

Native IRIS inference uses objectness multiplied by the highest class probability
per anchor, class-aware NMS at `0.5`, a `0.001` score floor and at most 100 boxes.
Evaluation confidence thresholds are applied subsequently. Export consumers must
record their own filtering policy when comparing deployment behavior.

## Bounded fine-tuning

All three depths are available on CPU and CUDA:

| Scope | Trainable modules |
| --- | --- |
| Prediction head only | Class, box and objectness projection layers |
| Partial backbone | Final CSP backbone stage and complete detection head |
| Full model | All backbone, feature pyramid and detection parameters |

Batch-normalization running statistics remain frozen for every scope. Batch size
is one, inputs stay at 416 pixels, and the worker uses native SimOTA assignment,
IoU, objectness and classification losses. Empty images contribute objectness
loss. Validation and test images are never used by training.

This adapter is a bounded fine-tuning workflow, **not a reproduction of the full
upstream COCO training recipe**: no mosaic, mixup, random resizing, EMA or learning
rate schedule is enabled. Increasing the step count alone does not establish a
quality gain. Compare against the official parent on held-out scenes before
deploying any checkpoint.

The versioned `iris-yolox-nano-training-v2` recipe clips finite gradients to a
global L2 norm of `10` before each SGD update and records the original norm plus
a clipping flag in the step history. This bounds unstable updates observed with
batch-one adaptation, including negative images whose objectness loss covers all
anchors. Nonfinite losses or gradients still stop the attempt; clipping never
turns an invalid update into a successful one. Earlier v1 attempts retain their
original recipe and cannot silently resume under v2.

Checkpoint-enabled jobs retain optimizer, model, sampler and random state using
IRIS's existing CPU/CUDA recovery protocol. Model weights are saved on CPU for
portable loading. GPU memory exhaustion stops the attempt without changing the
device automatically; published recovery states remain available.

## Portable inference

The native model supports raw-grid export with `decode_in_inference=False`.
For one fixed input `[1, 3, 416, 416]`, its output is
`[1, 3549, 5 + number_of_classes]` with strides `[8, 16, 32]`. These outputs still
need grid decoding, score combination, coordinate restoration and NMS.

Use the versioned ONNX export manifest to transport preprocessing, ordered class
semantics and checkpoint provenance. Compare deployment predictions and measured
latency before replacing an existing model. A successful export does not prove a
gain in accuracy or tracking continuity.
