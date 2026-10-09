# Importing and selecting useful examples

[Documentation](README.md)

Open a project and session in **Data intake**. Import several images or videos
with the file picker or drop them onto the import area. The queue names its target
session and reports each file separately: imported, already present, failed or
not yet processed. A file failure leaves successful imports available. Retry the
failed files explicitly after correcting the problem. Stopping the queue finishes
the active request and leaves the remaining files unprocessed; it does not delete
completed imports. This browser queue is not a background job that survives closing
the page. Each file is limited to 2 GiB and stays on the local machine.

Source bytes, checksums, filenames and session ownership are preserved. Reimporting
the same bytes into the same session reuses the source. A different session keeps
its own provenance even for an identical file. Images become unselected frames;
videos still require an explicit extraction. No import creates validated labels.

## Find frames worth reviewing

Filter the gallery by source, filename and review state. Review filters distinguish
unannotated images, drafts, pending proposals, validated positives and validated
negatives. A negative is a human-validated empty annotation with no pending
proposals. An unreviewed image with no boxes is not a negative.

Saved compatible predictions provide two optional review signals: target predictions
with confidence from 0.1 up to, but excluding, 0.5; and no saved target predictions.
These signals use existing outputs and do not run a detector. They do not measure
model error or prove the absence of an object. A partly mapped official detector
cannot establish a no-target signal for classes it does not support. Inspect these
images alongside representative positives and reviewed negatives.

Exact pixel matches and similar images are shown as review aids. Similarity uses a
64-bit difference hash with Hamming distance at most 6. This image-level heuristic
can flag unrelated flat images and miss related scenes; a different hash is not
proof of independence. Nothing is deleted or deselected automatically. Open the
suggested images, inspect their source and choose what to keep.

Selection is separate from annotation and validation. Bulk selection applies to
explicit frame IDs in one session, with at most 1,000 images per request. The server
compares their displayed selection states in one transaction. If another tab has
changed a state, the whole request fails and the gallery must be refreshed.

Open a frame to inspect its original source, frame index, approximate timestamp and
extraction settings. When browsing a source's video frames, chronological navigation
and the original-video player help recover the surrounding context. Video times
retain their recorded `frame_index / nominal_fps` basis; playback support depends
on the browser's codecs.

## Preview complete scene-group partitions

In **Dataset & training**, choose one saved class version and open the partition
assistant. Set target proportions and a seed, then preview. The default is 80%
training, 20% validation and no test split. The seed provides repeatable tie-breaking,
not proof of independent data. Train and validation proportions must be positive;
test can be zero. Complete groups often prevent exact proportions.

The assistant uses selected, fully validated images from that class version.
It keeps scene groups whole, links groups sharing the same original video bytes,
and respects existing scene-group, exact-pixel and video split reservations.
Reservations for exact pixels and original video bytes apply across the workspace;
the response does not expose another project's records. It reports class counts,
negative examples and missing coverage so the user can review representativeness.
Exact duplicate frames require a selection decision before a proposal can be
applied or a release can freeze. The partition preview shows up to 100 similar
pairs using a stricter dHash distance of at most 4, prioritizing pairs that would
cross splits. These warnings do not block publication or establish scene identity.

Review the proposed assignments and warnings, then apply them to the builder.
Applying a plan changes the form only. It does not select images, validate labels,
freeze a release or launch training. Apply rereads candidates and checks their IDs,
revision IDs and reservations before filling the menus. If candidates changed,
refresh and preview again. The final freeze rechecks reviews, reservations and
source identities.

Frames from the same original video cannot cross splits, including when that video
was imported under another name or session. This protection also applies across
releases. Historical releases remain readable; an old video already reserved in
conflicting splits is reported and cannot be used in a new release. No historical
assignment is silently changed.

Reencoded or trimmed copies may have different source hashes, and separate videos
may show the same scene. Group related sessions deliberately and inspect similar
images. Hash checks and suggested proportions cannot establish an independent test
set or guarantee that rare classes and difficult conditions are represented.

## Bounds and local API

Selection insights inspect at most 2,000 session frames, compare at most 1,000
perceptual hashes and return bounded duplicate/neighbor lists. Prediction lookup
is limited to recent saved sources. Responses report the limits and truncation;
an image without an analyzed signal is not classified as negative. Dataset planning
is bounded by the 1,000-image release limit and caps displayed similarity pairs.

- `GET /api/sessions/{id}/selection-insights` returns read-only review signals.
- `POST /api/sessions/{id}/selection` accepts `frame_ids`, `selected` and an exact
  `expected_selection` mapping; stale selections return HTTP 409.
- `POST /api/datasets/plan` accepts optional `taxonomy_id`, `seed` and `ratios`
  (`train`, `val`, `test`, summing to one). The response includes assignments,
  revision IDs, a fingerprint, warnings and blockers. It creates no saved plan.
- Media upload responses add `import_status: created | existing`. This describes
  the request result and is not written into the asset record.

All endpoints retain project scoping. No provider, API key, model runtime or network
service is required for this workflow.
