# Review identities through time

[Documentation](README.md)

IRIS 0.50 adds **Temporal identities** to Studio. It edits a sequence's reference
identities and observed boxes, separately from tracker outputs and ordinary image
annotations. This is the human review step after the
[tracking comparator](tracking-studio.md). The separate
[quality report](tracking-quality.md) evaluates saved outputs against an explicit
reviewed revision.

## Start a reference

Open a frozen sequence in Temporal identities, or follow **Review identities**
from a lane in Tracking comparison. The editor loads the latest saved reference,
or a blank draft if none exists. It uses the sequence's frozen classes and
verified source pixels, including its clock and missing-frame information.

For a new reference, choose a saved comparison and a lane, map its native classes
to the sequence's classes, then explicitly load its proposals. An ignored class
is not added to the reference. Official detector mappings can use declared COCO
IDs; trained detectors carry their own output-class mapping. A similar display
name does not establish a class mapping.

Only the lane's **measured detector observations** become proposals. Predicted
positions and Kalman-estimated boxes do not. Proposed identities receive separate
reference IDs, and every proposed observation starts uncertain and unreviewed.
An image with no proposal also starts unreviewed, not as a negative example.
Loading proposals performs no inference and does not save or validate anything.

The seed records the comparison, lane, cache and profile hashes, class mapping
and original track-to-reference mapping. That mapping describes the initial
proposals; it does not change when you later split or merge reference identities.
After the first save, the seed origin is preserved. Existing references can be
edited without replacing their recorded seed.

## Correct the passage

Move between available source frames, zoom in on distant objects, and edit the
reference boxes. You can add a missing object, remove a false proposal, assign
an observation to an existing identity, or create another identity.

- **Split** assigns observations from a chosen source-frame boundary onward to
  a new reference identity. Both sides must contain an observation.
- **Merge** combines two identities of the same class. If both appear in one
  frame, resolve that conflict first; the editor never silently drops one box.
- **Visibility** distinguishes visible, occluded, out of view and unknown.
  Visible objects need an observed box. Occluded objects may have no box;
  out-of-view and unknown objects have no invented position.
- **Certainty** records whether you can identify the object confidently.
  Uncertain evidence remains useful for review but cannot count as complete
  reference coverage.

These operations modify only the stated observations. They do not interpolate
boxes through missing images or copy a predicted location into a human reference.
Undo and redo operate on the unsaved draft. Switching away or reloading warns
when unsaved changes would be lost.

## Review and save

Enter the reviewer's name, then review each frame explicitly. A partial review
can retain doubts and uncertain objects. Complete review means that you have
checked the identities, boxes and all relevant objects on that frame. Confirming
an empty frame is an explicit human negative, not a detector-derived conclusion.

Saving appends a new immutable reference revision. It records the save author
and the reviewer of each explicitly reviewed frame. Unchanged frame reviews keep
their previous author. Changing an identity assignment, box, visibility or
certainty resets the affected frame's review; split and merge operations also
clear affected pending confirmations. Saving by itself never promotes proposals
to human-reviewed evidence.

The server enforces this behavior against the latest saved revision. Merely
submitting `human_reviewed` inside a draft payload does not validate a frame
through the editor endpoint. A stale `expected_revision` returns a conflict and
preserves the draft in the interface, so another tab's save is not overwritten.
As elsewhere in IRIS, reviewer names are declared local attribution, not an
authenticated proof of who performed the review.

The coverage summary distinguishes partial, unreviewed and human-complete
frames. A **dense human reference** requires every source frame in the clip to
be available and completely human reviewed. A sequence with missing source
frames remains sparse even when all available frames have been reviewed.
Coverage describes the saved evidence, not its accuracy or independence.

## Data and HTTP contracts

The editor appends `iris-temporal-reference-v2` payloads to the existing
`temporal_references` table. This payload version was introduced with IRIS 0.50
without changing the then-current SQLite schema 21; the current database schema
is 22. The payload retains all [version 1 reference fields](temporal-data.md#reference-identity-and-review)
and adds `provenance`:

- `author`: the latest save author's name.
- `origin`: null for a manual reference, or the frozen tracking seed with
  `comparison_id`, `lane_index`, `cache_id`, `cache_fingerprint`,
  `result_sha256`, `profile_sha256`, `semantic_sha256`, `class_mapping` and
  `track_mapping`.

Existing version 1 references and their hashes remain unchanged. Editing one
creates a version 2 revision; temporal datasets that pin an older reference
continue to use that exact revision. Workspace archives validate and retain both
versions and the saved seed links without loading a detector or tracker runtime.

All endpoints are project scoped using the normal `project_id` query parameter:

| Request | Purpose |
| --- | --- |
| `POST /api/temporal/sequences/{id}/identity-proposals` | Prepare an unsaved, unreviewed draft from one completed lane |
| `POST /api/temporal/sequences/{id}/identity-edits` | Save a new reference revision with explicit frame-review actions |
| `GET /api/temporal/sequences/{id}/references` | Read saved immutable revisions |

Proposal creation accepts `comparison_id`, `lane_index` (0 or 1), and
`class_mapping`. The mapping must include every native class selected in that
lane, with a frozen taxonomy ID or null for each. The response contains `payload`,
`summary` and `proposal_summary`; it writes no rows.

An edit accepts `expected_revision`, the whole reference `payload`, `reviewer`
and `reviewed_frames`. Each explicit review has a `frame_index` and `coverage`
(`complete` or `partial`). The returned reference contains its new revision,
payload checksum and computed summary. Invalid mappings, source mismatches and
review contradictions fail without saving partial changes.
