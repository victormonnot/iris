# Tracking quality against reviewed identities

IRIS 0.51 evaluates a saved two-lane tracking comparison against an explicit
revision from [Temporal identities](temporal-identities.md). The result measures
the saved confirmed observations on that reference. It does not run detection,
rerun either tracker, tune a profile, or measure deployment speed.

## Prepare and read a report

In **Tracking comparison**, open a completed comparison and its quality panel.
Choose a saved reference revision for the same sequence, map the selected native
detector classes to the sequence's frozen classes, and calculate a report.
An ignored class is outside the evaluation scope. A name match alone never
establishes a class mapping. The default box-overlap threshold is **IoU 0.5**;
the report freezes the chosen threshold and mapping.

Both lanes use the same reference, source frames, classes and protocol. Reports
retain the exact reference revision, sequence and cache checksums, tracker
profiles and output hashes. Later reference edits do not change a saved result.
Saved history can be reopened without running inference or creating another
report. A failed calculation publishes no partial result.

Read coverage before comparing numbers. A report can explain errors on a small
reviewed passage without establishing which tracker is better on other footage.
References initialized from a tracker remain identified as such after human
review. Review does not make that test blind or independent of its proposals.
No automatic winner or deployment qualification is produced.

## Which evidence counts

Only frames marked **human reviewed and complete** can contribute exhaustive
box counts. Unreviewed, assistant-reviewed, partial and omitted frames are unknown
evidence, not negative examples. The report lists excluded source indices and
reasons. Sparse references can therefore expose local errors without implying
full-clip coverage.

Within the selected classes, a certain identity with a visible or occluded
observed box is matchable. An explicitly out-of-view identity requires no box.
An occluded identity without a box makes the frame geometrically indeterminate:
IRIS excludes that frame instead of counting a potentially correct observation
as a false positive. It does not interpolate boxes into unknown intervals.

The tracker side includes **confirmed measured observations** only. Predicted
positions, Kalman-estimated boxes, unconfirmed tracks and unassigned detections
cannot earn matches. Their excluded counts are shown separately across the
available source frames. In particular, a detector observation that the tracker
has not confirmed can leave a reference object missed under this protocol.
Viewer confidence filters do not affect evaluation.

## Box and continuity measures

Frame matching is one-to-one and class aware, with IoU at least the chosen
threshold. Valid associations from the immediately preceding frame are preferred,
then match count and overlap. Ordering and tie resolution are deterministic.

- **Matched observations / true positives:** reference objects associated with
  an eligible tracker observation.
- **Missed objects / false negatives:** eligible reference boxes without a match.
- **Extra observations / false positives:** eligible tracker boxes without a match.
- **Precision:** matched observations divided by all eligible observations.
- **Recall:** matched observations divided by all eligible reference boxes.

A zero denominator produces an unavailable rate, not a perfect score.
Spatially overlapping unmatched boxes with different classes can be paired as
**class confusions**. Those pairs still contribute a miss and an extra observation;
the diagnostic does not add another penalty.

Continuity counters use consecutive scorable source frames:

- An **identity switch** occurs when a continuously box-visible reference identity
  receives a different tracker ID from its last matched ID. A larger number alone
  is not an error. Missed observations preserve that last-match memory.
- A **fragment** is a matched, then missed, then matched interval while the same
  reference identity remains continuously box-visible. Initial misses are not
  fragments; a return can be both a fragment and an identity switch.
- An **identity transfer** diagnoses one tracker ID becoming associated with a
  different reference identity within an evaluated contiguous interval.

Missing source frames and excluded references break event continuity. A reference
identity's declared absence also ends its box-visible interval. Sparse reports
count only events inside covered intervals and expose the number of evaluated
transitions; zero observed switches does not establish error-free tracking through
the unknown intervals.

These are **IRIS continuity diagnostics**, not an assertion of exact CLEAR-MOT
benchmark behavior. In particular, IRIS resets identity memory at unknown
intervals and reference absences. The official
[TrackEval CLEAR implementation](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py)
uses a different last-match policy across reference absences.

## Whole-clip identity score

**IDF1** is available only when every source frame in the clip is present,
human-complete and geometrically scorable, with reference identity detections.
An unavailable score includes its reason. Empty negative passages can expose
false positives but cannot establish identity quality.

IDF1 uses a separate **global one-to-one identity assignment** over the clip.
IRIS accumulates every class-compatible, threshold-qualified overlap between a
reference and tracker identity, rather than just the selected frame matches.
The assignment maximizes correctly identified detections (`IDTP`); `IDFN` and
`IDFP` account for the remaining reference and tracker detections.
`IDF1 = 2 × IDTP / (2 × IDTP + IDFN + IDFP)`.
This follows the global identity formulation in
[Ristani et al.](https://arxiv.org/abs/1609.01775) and
[TrackEval's identity metric](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py),
applied to IRIS's explicitly filtered observations and reviewed reference scope.
The official benchmark's treatment of empty data is not used as a quality claim.

## Persistence and API

SQLite schema 22 adds immutable `tracking_quality_reports`. Archives preserve and
validate reports against their pinned comparison and reference, including their
computed contents. Existing sequences, reference revisions and comparison jobs
remain unchanged. Reading and checking a quality report requires no model weights,
tracker installation, GPU or cloud API.

All requests use the usual project-scoped `project_id` query parameter:

| Request | Purpose |
| --- | --- |
| `GET /api/temporal/tracking-comparisons/{id}/quality-reports` | Read saved report history |
| `POST /api/temporal/tracking-comparisons/{id}/quality-reports` | Calculate and save a complete report |
| `GET /api/temporal/tracking-quality-reports/{id}` | Read a pinned report |

Creation accepts `reference_id`, `class_mapping` and `iou_threshold`. Reports use
`iris-tracking-quality-v1` and retain frame-level matches, misses and diagnostic
events. Computation is bounded; unsupported sizes fail explicitly rather than
silently sampling the reference or publishing incomplete counts.

The current limits are 500 available frames, 512 combined reference and observed
boxes per evaluated frame, and 512 combined reference/tracker identities per
lane for global IDF1. Both lanes share a deterministic matching-work budget of
25 million estimated operations; unusually crowded inputs can reach that budget
before the other limits. The saved JSON report is limited to 48 MiB.
`GET /api/temporal/tracking-quality-status` exposes the protocol and these limits.

See [tracking cost measurements](tracking-cost.md) for separate fresh execution
timings and cadence simulations; the saved comparison's quality scores do not
describe a new run with dropped frames.

Training, profile search, application target-lock
policies and qualification on independent recordings remain separate steps.
