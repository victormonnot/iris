# Annotation review queue

The queue summarizes selected frames from the current session. It reads the
latest saved annotation revision and all currently recorded proposals in one
SQLite snapshot. It does not create annotations, select frames, change splits,
load checkpoints or contact an annotation provider.

## Progress and navigation

Each selected frame has one review state:

| State | Meaning |
| --- | --- |
| Unannotated | No saved revision and no pending proposals. An image without boxes still needs review. |
| Draft | The latest revision is a draft and all proposals have decisions. |
| Pending proposals | At least one proposal has no decision in the latest revision, including new proposals received after validation. |
| Validated | The latest revision is validated and every current proposal has a decision. This includes explicitly validated empty images. |

**Needs review** combines the first three states. Progress counts saved state;
unsaved edits remain local to the open editor. A new draft after validation no
longer counts as validated. The queue shows reserved splits from imported data
and frozen dataset versions, including reservations on exact image pixels.

Filters and ordering only affect navigation. The open frame remains available
when it falls outside the filter; unsaved edits are preserved. Changing frames
uses the editor's existing discard confirmation. **Validate and next** first
saves through the normal revision/conflict checks, then advances to a frame
still needing review in the active queue. A failed or conflicting save leaves
the current frame open. No proposal is accepted automatically.

## Optional detector disagreement

Select one saved, successful comparison from the same session with exactly two
distinct models and the canonical COCO taxonomy. The comparison must provide
both predictions for an image with compatible run identity, recorded image hash
and dimensions. Images outside the comparison, missing outputs, and incompatible
records are **unavailable**, not empty predictions.

Protocol `iris-review-disagreement-v1` operates on saved outputs:

1. Validate the saved detections and retain person/car COCO IDs 1 and 3 with
   confidence greater than or equal to the chosen threshold. Label text and
   trained models' native class IDs are not used for mapping.
2. Build candidate pairs of the same class whose intersection-over-union (IoU)
   is greater than or equal to the selected IoU threshold. Coordinates are
   continuous pixel `xyxy`, without adding one to widths or heights.
3. Find a maximum-cardinality one-to-one matching. Deterministic augmenting paths
   visit original left detection indices, with neighbors in descending IoU then
   right-index order. The goal is the most matched boxes, not maximum summed IoU.
4. Count unmatched detections on both sides. Their fraction of all retained
   detections is the disagreement value. It is independent of which model is
   presented on the left. Matching details retain original prediction indices.
5. Among unmatched boxes, match overlapping different-class pairs to expose
   possible class conflicts. They remain unmatched in the primary count.
   Ambiguous overlaps can have several valid pairings; conflict details describe
   the deterministic pairing for the recorded model order.

For example, two models that each return one person at different positions have
two unmatched detections out of two. Identical person boxes match. A person and
a car box at the same position are a class conflict. Two empty retained outputs
have **no detection signal**, with a null disagreement value rather than zero.
Scores are filtered independently; they are not treated as comparable calibrated
probabilities across architectures.

The API returns the protocol, thresholds, comparison identity, model identities
and source prediction IDs. The saved comparison remains the evidence; the queue
is a view of current annotation progress. Reading the queue checks recorded
hashes and dimensions without opening image pixels. Normal annotation saving
and dataset freezing retain their separate image-integrity checks.

## Interpretation and limits

Disagreement ordering surfaces cases to inspect. It is not accuracy, uncertainty,
precision/recall, or a prediction that annotation will improve a model. Both
models can agree on incorrect boxes or miss objects together. Predictions below
a detector's native score cutoff or removed by its NMS/cap cannot be recovered.
At most 100 native detections per model and image are supported, matching the
existing comparison protocol. Other COCO classes are outside this queue's scope.

Only an explicit human save validates an image. Selecting an ordering cannot
move validation/test imagery into training. Test-reserved rows retain their
source order rather than becoming a ranked training-data recommendation. Their
labels can still be reviewed normally. Use a frozen dataset and **Evaluation**
for actual quality measurements.
