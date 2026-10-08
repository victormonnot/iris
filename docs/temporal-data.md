# Temporal sources and reference identities

IRIS 0.46 adds the data foundation for evaluating tracking. A **sequence** freezes
one video clip and its already extracted frames. A **reference revision** records
which real objects appear across those frames. A **temporal dataset version** pins
sequences, optional reference revisions and train/validation/test assignments.

These are local Python/JSON API services. The separate
[Studio tracking comparator](tracking-studio.md) can now create and inspect
sequences. These data contracts do not extract new frames, run trackers, compute
tracking scores or train a model.
Ordinary image annotations remain independent and unchanged. No optional ML
runtime, external provider or ARGOS integration is required.

The separate [temporal detector cache](temporal-detections.md) service can now
calculate and retain detections for these frozen frames. Its outputs remain
independent of reference identities and human review.

## Source sequence

`iris-temporal-sequence-v1` records:

- Sequence ID, project, name and optional parent sequence ID.
- Original video ID and SHA-256, source session and scene group, and a declared
  take group. Related recordings should share a group even when their files differ.
- The project's complete frozen class definition, independent of future edits.
- Inclusive source `start_frame` and `end_frame` bounds.
- Available frames in increasing source-index order: workspace frame ID, pixel
  SHA-256, PNG byte SHA-256, original dimensions and timestamp in seconds.
- Ordered, nonoverlapping gaps with `skipped`, `unavailable` or `unknown` reason.

Available frames and gaps must cover the clip exactly. The service fills omitted
positions with `unknown` gaps unless the caller supplies explicit gap evidence.
A missing frame is never treated as an empty detection or absent person. Frames
can change dimensions; consumers must use each frame's saved dimensions.

The service checks project/source ownership, source bytes and decoded image
pixels before publication. Documents are hashed using sorted compact UTF-8 JSON.
A new clip version creates a new row and retains its parent's original video
identity and take group. Existing rows cannot be updated through `Store.update`.
No sequence is created automatically from existing image annotations.

### Clocks

The clock contains `basis`, `fps` and a nonempty `provenance` description:

| Basis | Meaning | Timestamp rules |
| --- | --- | --- |
| `nominal_fps` | Estimated media position from source metadata | Exactly `frame_index / fps`; positive FPS matching the source |
| `provided` | Caller-declared timestamps, with their origin documented | Finite, nonnegative, strictly increasing seconds; `fps` is null |
| `unknown` | No usable temporal clock | Timestamps and `fps` are null |

Ordinary IRIS extraction uses nominal FPS. These times are not measured camera
capture times and may be approximate for variable-frame-rate media. Provided
timestamps can represent a sidecar or decoder timestamps, but validation does
not certify their measurement or alignment. All timestamps within one sequence
share one documented origin. T1 supplies neither runtime admission times nor a
simulated processing clock. Future tracker buffers must distinguish seconds
from numbers of analyzed updates.

## Reference identity and review

`iris-temporal-reference-v1` pins the sequence ID, manifest checksum and taxonomy
ID. Its identity catalog contains `{id, label}` pairs. IDs describe real objects
**within this sequence**; they are not tracker output numbers, image-box IDs or
cross-video person identities.

Each included frame records:

- `coverage`: `complete`, `partial` or `unreviewed`.
- `review`: `status` (`unreviewed`, `assistant_reviewed`, `human_reviewed`) and
  reviewer attribution. Reviewed frames require a nonempty reviewer.
- `objects`: reference identity (or null when uncertain), class, optional pixel
  `xyxy` box, visibility and certainty.

Visibility is `visible`, `occluded`, `out_of_view` or `unknown`. A visible object
requires a box inside the source image. An occluded object can omit its box;
an out-of-view or unknown object cannot have a fabricated position. Certain
objects require a known identity and known visibility. Uncertain observations
can retain a proposed identity or omit it. One identity appears at most once per
frame and keeps its declared class. Tracker IDs and predicted-position fields
are rejected in this reference format.

Complete coverage is an explicit reviewed claim with no uncertain objects. An
empty, completely reviewed frame can represent a negative; an omitted or
unreviewed frame is unknown. Assistant review never becomes human review
automatically. Attribution is caller-declared in this single-user application,
not authenticated evidence of who performed the work.

The summary exposes reviewed/omitted/uncertain counts. `dense_human_reference`
is true only when every source frame in the clip is available and completely
reviewed by a human. It describes declared coverage, not annotation accuracy,
dataset independence or permission to compute every tracking metric. T1 does
not implement metrics or fill positions between reviewed frames.

Saving requires `expected_revision` (0 for the first save). A stale save returns
a conflict. Successful saves append revision 1, 2, ... and retain old payloads.
A dataset pins one exact revision; later edits do not change its reference.

IRIS 0.50 adds `iris-temporal-reference-v2` for the
[Studio identity editor](temporal-identities.md). It retains the same identity,
geometry and review semantics and adds the save author and optional frozen
tracking-seed provenance. Existing version 1 payloads and their hashes remain
unchanged. The editor resets reviews on changed frames and applies human review
only through explicit frame-review actions.

## Temporal datasets and split protection

`iris-temporal-dataset-v1` contains distinct sequence entries with their checksums,
optional reference IDs/checksums and `train`, `val` or `test` assignments. All
entries must use one frozen taxonomy and belong to the dataset's project.
References may be absent or partial while preparing a dataset; their presence
alone does not qualify it for evaluation.

The versioned evaluation policy scopes identities to each sequence, restricts
assistant-reviewed evidence to diagnostics and excludes unreviewed, uncertain
and predicted evidence. Later metric implementations must
apply and disclose their additional matching and coverage requirements.

Reservations persist across temporal and ordinary image dataset versions:

- The same original video SHA-256 stays in one split, including different clips,
  reimports and copies in another project.
- Exact image pixels stay in one split throughout the workspace.
- Declared scene/take groups stay in one split within their project.
- Declared source splits from imported datasets are respected.

Reservations are checked in the same write transaction as publication. Parent
versions remain immutable. New file encodings and different group names do not
prove independence: T1 does not discover every near duplicate or related take.
Users must group these before freezing. The response includes this limitation.

## API

Every route accepts the ordinary `?project_id=...` scope and checks linked record
ownership. GET does not mutate data or run a model. POST returns 201 on success;
invalid input shapes return 422, missing/foreign records 404, and semantic or
revision conflicts 409.

| Route | Operation |
| --- | --- |
| `GET /api/temporal/sequences` | List project sequences with latest reference |
| `POST /api/temporal/sequences` | Freeze extracted source frames |
| `GET /api/temporal/sequences/{id}` | Read a frozen sequence |
| `GET /api/temporal/sequences/{id}/references` | Read revision history |
| `POST /api/temporal/sequences/{id}/references` | Save `{expected_revision, payload}` |
| `GET /api/temporal/references/{id}` | Read one exact reference revision |
| `GET /api/temporal/datasets` | List project temporal datasets |
| `POST /api/temporal/datasets` | Freeze sequence/reference/split assignments |
| `GET /api/temporal/datasets/{id}` | Read one temporal dataset version |

Example sequence request using existing IDs:

```json
{
  "name": "Courtyard crossing",
  "asset_id": "existing-video-id",
  "frame_ids": ["existing-frame-10", "existing-frame-12"],
  "take_group": "courtyard-take-a",
  "clip": {"start_frame": 10, "end_frame": 12},
  "gaps": [{"start_frame": 11, "end_frame": 11, "reason": "skipped"}]
}
```

The default clock uses the source nominal FPS. To provide timestamps, also pass
`clock: {basis: "provided", fps: null, provenance: "..."}` and a `timestamps`
object mapping **every selected frame ID** to its timestamp in seconds.

Example temporal dataset request:

```json
{
  "name": "Continuity development set",
  "entries": [
    {"sequence_id": "sequence-a", "reference_id": "reference-a-revision-1", "split": "train"},
    {"sequence_id": "sequence-b", "reference_id": null, "split": "val"}
  ],
  "notes": "Separate takes; validation identity review is still pending."
}
```

Limits: 10,000 available frames per sequence, clip span of 1,000,000 source
frames, 10,000 reference identities, 500 objects per frame, 100,000 objects per
reference revision, and 1,000 sequences per dataset. Names/groups use at most
160 characters and notes 4,000. Limits bound documents, not guaranteed runtime
memory or latency. These synchronous publication endpoints target prepared,
bounded clips; extraction and model jobs keep their existing workflows.

## Storage and recovery

SQLite schema 20 adds three tables without rewriting previous rows. JSON
documents remain in the database and use existing workspace media; no model or
frame copy is created. Workspace backups verify temporal document hashes,
ownership, revision/parent histories, splits and referenced PNG byte hashes.
Archive inspection and restoration also support unchanged schemas 12–19 and do
not instantiate a model or migrate saved bytes. Opening a restored workspace
performs the ordinary additive migration. See [backup and recovery](workspace-backup.md).
