# Tracking comparison in Studio

IRIS 0.49 adds a local, visual comparison of **ByteTrack** and **BoT-SORT without
learned ReID**. Both trackers consume the same complete detector cache. You can
inspect their boxes, identifiers and observed trails on synchronized source
frames, save the comparison and reopen it after a restart.

This is a visual inspection tool. The separate
[Temporal identities editor](temporal-identities.md) now handles identity
corrections and human review. This comparator does not compute identity accuracy,
tune tracker parameters, or qualify a tracker for a
deployment. Two tracks with the same number in different lanes need not represent
the same object. An increasing track number alone does not establish an error.

## Prepare and compare

1. In **Data intake**, import a video, extract frames and select a short passage
   from one source. Consecutive frames are more informative for tracking than
   widely spaced samples; extracting frames does not invent missing observations.
2. Open **Tracking comparison**. Save the selection as a temporal sequence, or
   choose an existing sequence. The sequence freezes its source frames, class
   version, clock and gaps. Ordinary image selections and annotations remain
   separate from temporal references.
3. Choose an existing complete detector cache, or explicitly calculate one with
   an installed local detector. CPU and CUDA refer to this detector execution;
   the optional native association implementations run on CPU. Reusing a cache
   does not load detector weights or run detector inference again.
4. Choose native detector classes and whether BoT-SORT uses camera motion
   compensation. Start a comparison. Both lanes use their documented native
   default profiles, with the same selected classes and saved detections.
5. Scrub or step through the synchronized views. Playback speed changes the
   viewer only; it does not rerun tracking or change the saved timing contract.

Install the optional [tracking dependencies](tracking.md#installation-and-readiness)
before launching a comparison. Browsing saved results does not require loading
those native runtimes. A Studio comparison accepts at most **500 available
frames** and publishes at most **48 MiB** of report JSON. Longer sequences remain
available through the underlying temporal services and command-line replay.
Use a shorter sequence if the Studio limit is exceeded.

## Read the evidence

- **Observed boxes** use the original detector observation, with its confidence.
  A native unconfirmed track is marked separately from a confirmed one.
- **Predicted boxes** are the tracker's estimated positions without a current
  matching detection. They have no fresh detection confidence. Prediction is
  not proof that the object remains visible or that its identity is correct.
- **Unassigned detections** remain available for inspection. A display filter
  does not alter the saved cache or either tracker's input.
- **Trails** connect observed positions only while observations and source
  frames are consecutive. They break across absent observations or source gaps;
  the viewer does not interpolate missing evidence.
- **Observation absence and return** describe a native track's output history.
  They are not human-confirmed losses or recoveries of the same real object.
  Crowded or ambiguous passages require inspection; this screen cannot certify
  which association is correct.

Source frame indices and available analyzed updates are different quantities.
An available frame with zero detections still updates the tracker. A missing or
skipped source frame does not generate a synthetic update. The default buffer
counts **30 analyzed updates**, not 30 source frames or a fixed wall-clock time;
native expiry is evaluated after association. See the
[native profile semantics](tracking.md#profiles-and-the-standalone-interface).

A nominal frame-rate clock is labelled as estimated timing. A provided clock
retains its declared provenance, which may describe receipt times rather than
camera exposure. Unknown timestamps remain unknown; playback in that case is
only an inspection cadence. Neither playback nor recorded detector timings
measure the complete performance of a deployed tracking application.

## Saved jobs and HTTP interface

Comparisons use the existing local worker queue. Cancellation, interruption or
a failed lane leaves an explicit unsuccessful job, without a complete comparison
report. Starting another comparison creates a fresh attempt from the beginning.
It does not resume hidden tracker state. Complete results are stored with the
job and included in workspace backup/restoration, without a new database schema.

Each comparison runs **one pass per lane**. The embedded replay reports explicitly
say that repeatability was not checked. Use the command-line replay's repeated
passes when that evidence is needed. Neither repeated matching outputs nor visual
smoothness establish tracking quality.

All endpoints below are project scoped using the existing `project_id` query
parameter:

| Request | Purpose |
| --- | --- |
| `GET /api/temporal/tracking-status` | Inspect optional runtime readiness |
| `POST /api/temporal/detection-caches/{cache_id}/tracking-comparisons` | Queue two native tracker lanes |
| `GET /api/temporal/detection-caches/{cache_id}/tracking-comparisons` | Read lightweight comparison history |
| `GET /api/temporal/tracking-comparisons/{job_id}` | Read status and the completed report |
| `GET /api/temporal/sequences/{sequence_id}/frames/{frame_id}/image` | Read the hash-verified frozen source image |

Creation accepts a name, a nonempty list of native `class_ids`, and
`gmc_method` (`sparseOptFlow` or `none`) for BoT-SORT. ByteTrack uses no camera
motion compensation. Both exact profiles and the cache hashes are frozen before
execution. A detector cache floor higher than the tracker low threshold is
rejected; changing the display filter cannot restore previously filtered scores.

The complete report uses `iris-tracking-comparison-v1`, with the frozen sequence
and one [T3 replay report](tracking.md) per lane. History and ordinary job polling
omit that heavy payload. The source-image endpoint verifies the saved file and
decoded pixel hashes; the viewer hides overlays if the matching image fails to
load. A saved result can also be opened using `?project=PROJECT_ID&tracking_comparison=JOB_ID`.

Use **Review identities** to open the separate
[human temporal identity editor](temporal-identities.md). It can prepare proposals
from this lane, while keeping reference identities, reviewer attribution and
sparse/dense coverage in the [versioned data contract](temporal-data.md).
Proposals do not become validated references automatically.
