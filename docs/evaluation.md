# Detection evaluation protocol

An IRIS evaluation measures one or two checkpoints on every image in a frozen
dataset's validation or test split. It is separate from the visual comparator,
which can inspect unannotated session frames. Both views save predictions and
measured timings; only the evaluation has a reviewed reference.

## Reference and class mapping

Releases contain copies of normalized PNGs, full validated annotation revisions,
reviewer names, source provenance and scene-group splits. Evaluation reads these
copies, not the current annotation editor. Image file and pixel hashes are checked
before inference. The manifest, selected checkpoint hashes and training ancestry
are frozen at queue time and rechecked in the worker.

The fixed taxonomy is `iris-objects-v1`: visible people, including riders, and
passenger cars, including SUVs/minivans. Cars exclude buses, trucks, motorcycles
and bicycles. Boxes enclose visible extents with continuous `xyxy` pixel
coordinates and exclusive right/bottom edges. The scoring adapter converts them
to COCO `xywh` without adding one pixel.

Official models use COCO IDs 1 (`person`) and 3 (`car`). IRIS-trained heads use
native IDs 1 and 2; their adapter records the mapping to COCO and preserves the
native ID. Other valid COCO classes remain in raw outputs but do not contribute
to person/car metrics. Their ignored count is recorded. Unknown or inconsistent
class identifiers are rejected. There are no crowd annotations or ignored regions
in this taxonomy; unsupported annotation flags are rejected rather than inferred.

## Scores and errors

AP is computed with the reference [COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
through pinned `pycocotools` 2.0.11. The saved protocol also identifies NumPy and
the IRIS protocol version. The application reports:

| Value | Definition |
| --- | --- |
| mAP | Mean interpolated AP over IoU thresholds 0.50, 0.55, …, 0.95 and classes with reference objects. |
| AP50 / AP75 | AP at IoU 0.50 / 0.75, averaged over classes with reference objects. |
| Per-class AP | The same calculations for person or car individually. |
| Precision | `TP / (TP + FP)` at the recorded confidence and IoU thresholds. |
| Recall | `TP / (TP + FN)` at those thresholds. |

COCO AP uses 101 recall points, all object areas, and the `maxDets=100` result
with configured limits `[1, 10, 100]`. It consumes all available native scores,
independently of the chosen precision/recall confidence threshold. The initial
detectors apply a native score floor of 0.001, NMS IoU 0.5 and a cap of 100
detections per image before IRIS receives their outputs. These limits and the
architecture's proposal settings are recorded per model. AP cannot recover
predictions discarded by those stages.

At the precision/recall operating point, detections with score greater than or
equal to the chosen confidence threshold are matched greedily, in descending
score order, to unmatched reference boxes of the same class. The best overlap
must be greater than or equal to the chosen IoU threshold. A reference object
can match only once; additional detections become false positives. An unmatched
reference object becomes a false negative. Micro totals aggregate these counts
across both classes and all images. The result retains matched indices and IoUs,
false-positive prediction indices and false-negative annotation indices so the
image viewer can inspect the exact errors counted.

Score ties preserve the original prediction order; AP additionally preserves
manifest image order. Equal IoUs choose the last eligible reference box in
manifest order. Stable ordering makes repeated evaluation of the same artifacts
reproducible; it does not remove ambiguity in difficult overlaps.

An absent reference class has undefined AP and is excluded from macro AP.
Recall is undefined when there are no reference objects; precision is undefined
when no predictions survive the threshold. These values are JSON `null` and
display as **N/A**, never as perfect scores. False positives on validated negative
images still count. An entirely negative split has no AP or recall; its false
positives remain useful evidence. Class support and prediction counts accompany
the scores. A small or unbalanced reference cannot support broad quality claims.

## Explore saved errors and paired changes

The error explorer uses completed evaluation results to find examples, without
running inference or calculating a new set of metrics. Choose all classes,
person or car, then filter missed objects, false positives or changes between
the two models. The table opens the same frozen image in the existing side-by-side
viewer. Class filtering retains the original annotation and prediction indices.

The first model in the saved evaluation is the **baseline**, and the second is
the **candidate**. These names indicate comparison direction, not training age
or a recommendation. A single-model evaluation supports error inspection but
has no paired changes.

| Change | Meaning at the saved confidence and IoU thresholds |
| --- | --- |
| New misses | Reference objects matched by the baseline but missed by the candidate. |
| Recovered objects | Reference objects missed by the baseline but matched by the candidate. |
| False-positive delta | Candidate false-positive count minus baseline false-positive count. |

Changes in misses compare the exact same frozen reference object indices.
For example, if the baseline misses person A and the candidate misses person B,
each model has one miss, but the comparison shows one recovered object and one
new miss. A wrong-class detection can contribute a missed object in one class
and a false positive in another. False-positive deltas compare counts; they do
not establish correspondence between individual predictions from two models.

The class summary covers the **whole saved split**. Table filters and sorting
only change the examples being browsed; they do not change AP, mAP, the operating
thresholds or the frozen release. Validated negative images remain in the table,
including those with false positives. A class with no reference objects keeps
its false positives and zero object count; the existing AP display stays N/A.
Matching still follows the saved evaluation protocol, including its tie rules.
Paired changes are inspection evidence, not proof of an overall quality gain.

`GET /api/evaluations/{evaluation_id}/analysis` provides the same read-only
analysis under protocol `iris-error-analysis-v1`. It checks completed coverage,
saved model/frame/protocol identities and the consistency of the recorded error
indices and counts. Missing or incompatible records make the analysis unavailable
(HTTP 409); they are never interpreted as empty predictions or perfect detections.
An unknown evaluation returns 404. The explorer reads saved artifacts, not model
weights, image pixels or current annotation edits. Image serving separately
checks frozen pixels when the viewer opens them.

Test-audit errors remain available for reporting. Inspecting them does not move
images into training or promote a model, and using them to make later choices
can compromise the test's independence. Use validation for model improvements.

## Validation, test and model choices

Dataset creation reserves scene groups and exact pixel hashes to one split across
releases. Evaluation rejects overlap with local training data for the chosen
checkpoint and its ancestors. A missing or incompatible local ancestry is an
error. These checks cannot prove independence from external pretraining corpora,
recognize all near duplicates, or detect incorrect human grouping.

Use validation for model and threshold choices. A final test audit links to a
successfully completed validation evaluation on the same dataset version. It
must reuse that run's ordered model selection, hashes, confidence/IoU settings,
device, protocol and ancestry. Neither a training split nor a retuned test request
is accepted. The interface cannot prevent a person from using previously viewed
test results to guide future choices; doing so weakens test independence.

A reference selection is an explicit, appended decision from a completed
validation comparison. It records the reviewer, reason, dataset, checkpoint hash,
metrics and previous reference. Concurrent stale decisions are rejected. Test
audits cannot directly select a reference. Selecting a reference is a local
record, not a deployment or a claim that the model is universally better.

## Timing and completion

Each checkpoint has one excluded warmup forward and batch size one. The saved
timings separate decoding and hash verification, tensor preparation, full
Torchvision forward, output conversion, and total processing. The forward
includes resizing, normalization, proposal filtering, NMS and rescaling.
Loading checkpoints and database writes are outside the measurement. Device,
hardware, runtime versions, thread count, precision, transforms and native
filters remain with each model's results. Displayed means summarize those
measured frame timings; they are not drone throughput benchmarks.

An incomplete model has no aggregate quality metrics. Finished model results
can survive another model's failure, but the evaluation remains incomplete and
cannot support reference selection or a test audit. Saved per-frame predictions
and logs survive cancellation, failure and restart. Retry by creating another
evaluation; results are not silently resumed or mixed.

## Verification scope

Controlled fixtures test perfect and missed detections, duplicate matches,
false positives, score ordering, empty images, absent classes, IoU boundaries,
mapping errors and COCO interpolation. Workflow tests cover immutable inputs,
training overlap, changed hashes, partial results, fixed test settings and
reference history. The opt-in CPU test trains a real checkpoint, evaluates it
against its parent, audits the test split and verifies persistence.

Error-explorer fixtures cover recovered and newly missed objects despite equal
miss totals, class-specific false positives, negative images, original source
indices, single-model and incomplete runs, saved-result integrity and reopening
the workspace. These checks use synthetic labels and predictions; the explorer
itself never runs a detector.

All generated test images and automated review records are explicitly synthetic.
They establish software behavior, not aerial detection accuracy. Real validation
requires representative recordings, independent scene groups and careful human
review. No bundled flight dataset or performance improvement is claimed.
