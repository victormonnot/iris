# DINO-X preannotation

The standalone [DINO-X → Astra review adapter](dinox-astra-review.md) can evaluate
saved detections in a separate experiment. It is not yet part of the Annotation
workflow described below.

## Prepare, run and review

In **Annotation → Cloud proposals · DINO-X**, configure a key with available
DeepDataSpace credits, then select the current image or up to 25 selected images
from one session. The key field is write-only; it is never saved in browser storage
or the workspace archive. The existing local detector workflow remains available.

Choose a native score threshold and, optionally, one phrase per saved class. Preview
the chosen images, saved-result reuse, remote tasks to poll and new request count.
Confirm external processing and enter a CNY allowance before starting. This bounds
the number of admitted requests using the saved list price, not a guaranteed future
invoice. An API key does not establish available quota. IRIS never buys credits.

Follow the saved batch, open a frame and accept, correct or reject its proposals in
the ordinary annotation editor. Add omissions manually. Save draft and validate
remain separate human actions; inference never validates annotations. Native outputs,
class mappings and proposal provenance remain inspectable after edits.

## Interrupted work and saved responses

Schema 19 stores batches and individual request receipts separately. Each request
is keyed by the image record, verified pixel identity, saved taxonomy and complete
provider profile. A later explicit preview can reuse its successful response for
zero new detection calls, even after human annotation edits. Stable suggestion IDs
preserve prior accept/reject decisions. Changing a model setting or class phrase
creates a distinct request with its own estimate.

The worker records intent before its single POST, then saves the returned task ID
before polling. Cancellation or a server restart preserves both. A new batch can
poll a known task without sending the image again. A submission with an unknown
outcome is blocked from automatic replay. A confirmed failed task or rejected
submission can be tried in a new explicitly previewed batch; retries never occur
inside the failed attempt. Processing stops at the first error, preserving earlier
results and untouched remaining images.

Raw output is persisted before normalization. Invalid saved output can be inspected
or revalidated locally without another API call. Publication checks the current
image, taxonomy and human annotation revision against the confirmed snapshot;
concurrent edits retain the remote response but prevent overwriting human work.
Cached proposals are checked against their native response before reuse.

Workspace backup and restore preserve these receipts, proposals and human revisions.
Older schema 12–18 archives remain supported. Restoring never authenticates or starts
a remote task; credentials stay outside the archive. The optional DINO-X annotation
workflow does not imply a DINO-X adapter in the independent Benchmark workspace.

## HTTP interface

- `GET /api/dinox/provider`: offline credential presence and price information.
- `PUT /api/dinox/key` / `DELETE /api/dinox/key`: update the local credential file.
- `POST /api/sessions/{id}/dinox-batches/preview`: freeze an offline cost/input preview.
- `POST /api/sessions/{id}/dinox-batches`: confirm its fingerprint, external processing and CNY cap.
- `GET /api/sessions/{id}/dinox-batches`: saved batch history.
- `GET /api/dinox-batches/{id}`: per-image state, requests, raw output and estimated cost.

Batch and job resources follow project ownership. Standard job cancellation applies.

## Provider contract

The optional `dinox` provider uses DINO-X-1.0 for box proposals. Local inference remains
the default. Image pixels and the selected text prompts leave the machine only when a
caller explicitly submits a prepared request; configuration and cost estimates are offline.

Each image requires one detection request, including all selected classes. Prompts are
joined with ` . `, following the [official DINO-X demonstration](https://github.com/IDEA-Research/DINO-X-API/blob/main/demo.py).
The default prompt for a class is its ID. Optional per-class prompts are converted to
lowercase with normalized whitespace. Every class must have exactly one distinct prompt;
periods, control characters and prompt-free syntax are rejected. Native response categories
must exactly match these frozen prompts; unknown categories fail the image.

The frozen profile contains the complete taxonomy, prompt-to-class mapping, model label,
box threshold (default `0.25`), IoU threshold (`0.8`), `bbox` target, coordinate contract,
input/output bounds, price and documentation sources. The adapter sends a metadata-free
RGB PNG at the original oriented dimensions. It does not resize. Native pixel `xyxy` boxes
are validated before clipping to the visible image; provenance retains the original box.
Invalid geometry, missing scores, excessive output or an absent objects array fail the
result. An explicit empty array is valid evidence of zero proposals and still requires
whole-image human review. Scores are not calibrated across providers. A hosted version
label does not pin remote weights or service behavior.

The documented [list price](https://algos.deepdataspace.com/en/price/README.md), checked on
2026-10-06, is **0.15 CNY per detection call**. Estimates use one request per image and
are not invoices, quota checks or guarantees about future pricing. No currency conversion
or account lookup occurs.

Transport uses direct TLS to `api.deepdataspace.com`, with `Token` authentication:
`POST /v2/task/dinox/detection` returns a task ID, and
`GET /v2/task_status/{task_id}` checks it. These paths and the idempotency header follow
the provider's [V2 task implementation](https://github.com/deepdataspace/dds-cloudapi-sdk/blob/main/dds_cloudapi_sdk/tasks/v2_task.py)
and [base task contract](https://github.com/deepdataspace/dds-cloudapi-sdk/blob/main/dds_cloudapi_sdk/tasks/base.py).
IRIS does not import that SDK. There are no SDK telemetry, redirects, proxy discovery,
background probes or automatic POST retries. `submit` makes at most one POST, and `poll`
makes one GET without a polling loop. The caller persists intent before dispatch and the
task ID before polling. An idempotency header is not treated as permission to resubmit an
ambiguous paid operation. Responses are bounded to 4 MiB; PNGs to 16 MiB and 25 million
pixels; proposals to the IRIS contract limit of 300. These are local safeguards.

## Server credentials

The server checks `DDS_CLOUD_API_TOKEN` in its process environment first, then the same
key in `~/.config/iris/cloud-credentials.json`. The credential never appears in provider
profiles, browser status responses or proposal provenance. Status reports configuration
presence only; it does not validate the account remotely.

The local configuration helpers update only this key and preserve other entries, including
`FAL_KEY`. They reject symlinks and insecure ownership/permissions, require a `0700`
credential directory and a `0600` file, and write atomically. Local-file changes apply to
the next request. An environment override remains effective until the server is restarted
with that variable changed or removed; deleting the disk key does not delete the override.
The HTTP layer must restrict credential mutations to trusted local requests.

Transport failures expose fixed error messages and sanitized evidence. The batch layer
must preserve uncertain submission outcomes and must not retry them automatically.
