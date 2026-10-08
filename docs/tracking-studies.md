# Bounded tracking profile studies

IRIS 0.53 compares an explicit baseline with a small set of tracker profiles on
frozen temporal data. The study reuses saved detector outputs and measures both
reference-based quality and tracker replay cost. It does not train a detector,
change application settings or automatically select a deployment profile.

## Freeze the data before trying profiles

Prepare a temporal dataset with the source sequences, exact reference revisions
and their roles. The study panel can create that dataset through the existing
[temporal dataset workflow](temporal-data.md). Saving the dataset and launching
a study are separate explicit actions.

- **Development (`train`)** is available for exploring settings. This role does
  not mean that the study trains a neural network.
- **Validation (`val`)** is also available for choosing settings. Repeatedly using
  it for profile comparisons makes it tuning evidence, not an independent test.
- **Reserved test (`test`)** is listed as withheld. T8 never replays its frames
  or calculates its tracking metrics.

The same source video, identical pixels and declared scene/take groups keep their
existing split reservations across temporal and image datasets. Two clips from
one recording cannot become independent development and validation sets merely
by assigning different sequence IDs. IRIS does not discover every near duplicate;
related takes must still be grouped correctly.

Every participating development or validation sequence needs a complete saved
tracking comparison and the reference pinned by the dataset. An incomplete
reference remains incomplete: selecting it does not manufacture ground truth.
A development-only study is useful for checking behavior but cannot establish a
validated improvement. New difficult recordings remain necessary for qualification.

## Choose a small, explicit comparison

Choose an existing comparison lane as the baseline. Review the suggested
alternatives or edit their settings before previewing the job. The study freezes
every selected profile; it does not silently explore an additional parameter grid.

The supported settings include confidence thresholds, track creation thresholds,
association thresholds, lost-track retention in **processed updates**, confidence
score fusion, and optional BoT-SORT camera compensation. Learned re-identification
is not enabled. Native constraints still apply: ByteTrack uses a fixed low
threshold of 0.1 and a creation threshold equal to its high threshold plus 0.1.
BoT-SORT requires low < high ≤ creation threshold.

All profiles use the same selected native classes and identical cached detector
outputs for each sequence. Sources must share the detector recipe; historical
device/runtime declarations can differ. A cache score floor above a profile's
low threshold is rejected. Lowering a tracker threshold cannot recover detections
that the detector already filtered out.

Preview shows the exact work before execution: baseline plus selected candidates,
participating sequences, repetitions, available-frame updates and the declared
wall-time limit. Limits are eight total profiles, four active sequences, 500
available frames per sequence, three repetitions, 20,000 total updates and 600
seconds. A request can set smaller work and time budgets.

The time limit is cooperative: IRIS checks it between work units and at publication.
It cannot promise to interrupt an individual native tracker call at an exact
millisecond. Cancellation, failure or budget exhaustion publishes no successful
partial study. Starting again creates a fresh attempt.

## Read quality and coverage together

Each profile starts a fresh tracker for each source and repetition. Quality is
evaluated on the first replay pass against the pinned human reference; repetitions
check observed repeatability and provide timing samples. They are not extra
ground-truth images. Differing repeated outputs withhold comparative conclusions.

The [T6 quality rules](tracking-quality.md) still apply: only eligible confirmed
observations and human-complete reference frames count. Unknown intervals,
unlocalized occlusions and sparse coverage remain visible. IDF1 stays unavailable
when its coverage requirements are not met. A reference seeded from a tracker
retains that provenance after review.

Development and validation results stay separate. Counts can be combined within
a role, with rates recomputed from their denominators. Numeric tracker IDs never
identify the same person across separate sequences. An unavailable identity score
must not turn into zero or a perfect score when aggregating sources. If any
participating sequence has no evaluable reference frames, that role keeps its
descriptive counts but cannot establish a comparative gain.

Compare missed and extra objects, identity changes and fragmentation alongside
the timing difference. A lower cost can accompany worse continuity; a higher
score on a tiny known passage does not establish broader robustness. The current
baseline remains unchanged. Ties, insufficient coverage and the absence of a
validated gain do not justify replacing it automatically.

## What the cost means

The study records native tracker-adapter, association and camera-compensation
timings from the replay, along with setup, source-image reads and replay wall time.
Association and compensation are nested parts of the adapter timer, not extra
time to add again. The native adapter timer excludes some final wrapper validation;
the replay wall timer includes image reads, copies and progress reporting.

These are **tracker replay measurements on cached detections**. No detector is
loaded or run, no fresh pipeline memory peak is measured, and no camera cadence
is simulated. Saved detector timings are not added to infer end-to-end latency.
Use the separate [T7 cost workflow](tracking-cost.md) for its explicitly supported
fresh detector/pipeline measurements. Its historical results do not automatically
describe a changed profile from this study.

Timing samples describe this execution environment and passage. Small differences
can reflect warm caches or other work on the machine. There is no automatic
weighted quality/speed score and no claim of portable or embedded performance.

## Saved evidence

The durable `tracking_study` job pins the dataset, references, comparisons, caches,
profiles, class mapping, overlap threshold and budget. A successful report retains
the complete replays, protocol, runtime provenance and computed results. Reading
or transferring a saved study checks those results without executing a tracker or
loading detector weights. Old T4 comparisons and T6/T7 reports remain unchanged.

Downloads are review evidence, not a standalone deployment bundle. Exporting a
detector/tracker package and applying it in another application remain later
steps. Changing a form or opening a saved report never launches another study.

All API requests use the owning `project_id` query parameter:

| Request | Purpose |
| --- | --- |
| `GET /api/temporal/tracking-study-status` | Read protocol, limits and optional tracker readiness |
| `GET /api/temporal/tracking-study-sources` | Inspect datasets, reference revisions and completed comparisons |
| `POST /api/temporal/tracking-studies/suggestions` | Prepare an explicit candidate list from a baseline profile |
| `POST /api/temporal/tracking-studies/preview` | Validate sources, profiles and budget without running trackers |
| `POST /api/temporal/tracking-studies` | Queue the exact previewed request and its `expected_fingerprint` |
| `GET /api/temporal/tracking-studies` | Read saved attempt history |
| `GET /api/temporal/tracking-studies/{id}` | Read the complete saved evidence |
| `GET /api/temporal/tracking-studies/{id}/report` | Download the complete review report |

A study can also be opened with `?project=PROJECT_ID&tracking_study=JOB_ID`.
The request pins `dataset_id`, source comparison IDs, baseline and candidate
profiles, the explicit native-to-reference `class_mapping`, `iou_threshold`,
`repeats`, `max_updates` and `max_seconds`. Each profile also has a display name.
