# Reusable temporal detections

IRIS 0.47 adds durable detector caches for [frozen temporal sequences](temporal-data.md).
Calculate the available frames once, retain the original detector output order,
and reuse the saved results for stricter confidence or class selection. This is
the input foundation for later tracker comparisons; it assigns no track IDs,
computes no tracking metrics. The [Studio tracking comparator](tracking-studio.md)
now provides sequence/cache controls and replays these saved detections with
separate native tracker jobs.

The Python/JSON API supports installed official and trained Faster R-CNN,
SSDLite and YOLOX-Nano detectors, on CPU or NVIDIA CUDA, with full-image or tiled
inference. Preparation verifies local weights and runtime metadata without
constructing the model or downloading anything. The existing sequential worker
performs inference. CUDA execution never silently falls back to CPU.

## Frozen inputs and saved outputs

A cache configuration pins the sequence checksum and complete ordered list of
available frame IDs, detector weight checksum, class definitions, preprocessing,
native filtering, requested device, tiling settings, package versions and adapter
source hashes. An execution receipt additionally records the actual hardware,
device, thread settings and model metadata. Trained detector classes must match
the frozen sequence taxonomy; official COCO outputs retain their detector class
IDs and expose the mapping to sequence classes where available.

Each immutable frame result contains original-pixel `xyxy` boxes, class IDs and
names, scores, input dimensions, source pixel/file hashes, source frame index and
timestamp, execution identity, measured timings and inference work counts. Source
timestamps retain the sequence's declared clock; an unknown timestamp stays null.
The cache preserves T1 gaps and calculates only available frames. It neither
extracts missing frames nor invents observations for them.

An empty saved `detections` list is a completed detector result at the recorded
settings. It is not a human assertion that no object was present. A missing result
means the frame has not been published. Coverage is `empty`, `partial` or
`complete`, with completed/remaining counts; a result checksum is available only
when every available sequence frame has a saved output. Job status and coverage
remain separate: interruption after the final publication may leave complete
saved results under an interrupted attempt.

## Low confidence does not mean unfiltered proposals

The storage score floor defaults to `0.001`; accepted values are `0.001` through
`1`. Outputs have already passed the detector's native proposal selection,
confidence tests, class selection, non-maximum suppression and detection caps.
Full-image inference retains at most 100 native detections; tiled inference also
applies per-tile native limits and a final merged cap of 300.

A low storage floor preserves more of those surviving outputs. It cannot recover
candidates removed earlier or establish that every possible object above that
score is represented. The frozen recipe records native filtering and the output
policy explicitly, including this limitation.

A complete cache can be read with `min_score` at or above its saved floor and an
optional list of detector `class_ids`. Filtering performs no inference and never
changes the cache. It preserves each retained `detection_index`, including gaps
left after filtering. Those indices refer to the original detector output list
(the merged list for tiled inference), not temporal identities. Frame timings
and `native_detection_count` continue to describe the producing calculation.

Lowering the saved floor or changing weights, preprocessing, device, native NMS,
caps or tiling requires a new calculation. The API exposes full/tiled mode and
tile size/overlap; native filter and cap changes are not runtime overrides.

## API workflow

Pass `?project_id=<project-id>` on every request; omission selects `default`.
Sequence, trained-model, cache and job ownership are checked. Use an existing
sequence created through the [temporal data API](temporal-data.md#api).

| Method and path | Result |
| --- | --- |
| `POST /api/temporal/sequences/{id}/detection-caches/preview` | Frozen recipe, frame/forward-pass counts and any matching existing cache |
| `POST /api/temporal/sequences/{id}/detection-caches` | Queue a new cache or return the matching one |
| `GET /api/temporal/sequences/{id}/detection-caches` | Sequence cache history |
| `GET /api/temporal/detection-caches/{id}` | Recipe, coverage, attempts and class mapping |
| `GET /api/temporal/detection-caches/{id}/frames` | Complete saved results at the stored floor |
| `POST /api/temporal/detection-caches/{id}/read` | Complete results with optional stricter score/class filters |
| `POST /api/jobs/{id}/cancel` | Request cancellation while preserving committed frames |
| `GET /api/jobs/{id}/recovery` | Check remaining work and continuation compatibility |
| `POST /api/jobs/{id}/recover` | Queue an explicit linked continuation using its preview fingerprint |

Example preview body for an installed official detector:

```json
{
  "model_id": "ssdlite320_mobilenet_v3_large",
  "device": "cpu",
  "inference_mode": "full",
  "min_score": 0.001
}
```

Add `"name": "Courtyard baseline"` when creating the cache. Creation returns
HTTP 201 for a new cache and HTTP 200 with `reused: true` for the same frozen
recipe. Reusing creation still prepares and checks the requested local recipe;
reading an existing cache directly needs neither weights nor the ML runtime.
An existing partial cache is returned as partial and is not automatically resumed.
`force_new: true` explicitly requests a separate calculation even for the same
recipe, retaining the old cache and its results.

For tiled inference, use `"inference_mode": "tiled"`, `"tile_size": 640` and
`"overlap": 0.2`. Tile sizes range from 128 to 2048 pixels; overlap ranges from
0 to 0.5. Preview reports the total tile forward passes separately from one
warmup forward pass per attempt. Loading, warmup and GPU availability are checked
when the worker executes; a successful preview is not proof of runnable hardware.

Example read body for the official detector's person class:

```json
{"min_score": 0.25, "class_ids": [1]}
```

Read responses identify the original cache/result hashes, applied filter, source
sequence, producer jobs and stored frame hashes. A derived frame is not a newly
published cache row. Incomplete caches return a conflict instead of exposing a
misleading complete sequence or filling missing frames with empty detections.

## Cancellation and continuation

Each attempt is claimed once. A frame output and its progress checkpoint commit
atomically, so retained results always form an exact prefix of the ordered
available frames. Cancellation during inference discards the unfinished frame;
cancellation during tiled work publishes no partial tile result. A subsequent
attempt recalculates that unfinished frame and continues after the saved prefix.

For a failed, cancelled or interrupted attempt, inspect the recovery response and
submit `{"fingerprint": "<returned-fingerprint>"}` to its recovery endpoint.
The new job inherits the exact hashes of retained outputs. The original job and
its outcome stay in history, and one attempt cannot acquire two successors.
Reopening the server or restoring a workspace does not resume computation.
The existing task history lists temporal detector jobs and their saved frame
counts; its task drawer can check and confirm continuation for these jobs.

Continuation verifies source files, weights, classes, preprocessing and pinned
runtime sources/packages. After loading, the worker verifies the actual execution
signature against any retained results, including hardware and thread settings.
It refuses to combine incompatible environments. Drift requires a fresh cache;
old saved outputs remain readable. See [job recovery](job-recovery.md).

## Timing, persistence and limits

Per-frame measurements retain source verification/decode, model preprocessing,
inference and postprocessing, optional crop/merge work, cache filtering and a
frame total. Stage values can overlap; their sum is not a separate measurement.
The total excludes model loading, warmup and database publication. Each attempt
records its own model load and warmup separately. These are timings from the
producing run, not the speed of reading a cache, a tracker or a live video pipeline.

SQLite schema 21 adds two tables without modifying prior annotation or temporal
records. Workspace archives preserve empty, partial and complete caches and
validate configuration/output hashes, execution receipts and attempt lineage.
Inspection and restoration do not load model weights or import the ML runtime.
Source media and compatible runtime are required again for continuation, not
for reading completed saved results. See [workspace backup](workspace-backup.md).

The cache establishes reproducible detector inputs for later comparisons. It
does not establish detection quality, identity continuity, independent evaluation
data or performance on another computer. Sequence/reference data retain their
existing review and independence limitations.
