# Jobs, partial results and explicit recovery

The project job history keeps processing requests, settings, progress, errors and
worker logs. Open a job's details to inspect its source, saved partial artifacts
and related attempts. Filter the history or load more entries to find older work.
Closing the browser does not cancel a server job. Stopping the IRIS server marks
unfinished jobs **interrupted**, including jobs that had not started. Reopening
IRIS does not automatically launch them again.

Cancellation preserves artifacts already saved. A progress value is not a success
indicator: inspect the terminal status and saved results. Provider computation may
continue after IRIS stops waiting, including on a local Ollama server.

## Continue a frozen extraction

New video extraction jobs freeze their sampling positions, original video hash,
class version and initial deduplication inputs when queued. Each completed position
records whether it produced an image or was skipped as existing, identical or
perceptually similar. These checkpoints survive a worker or server interruption.

For a failed, cancelled or interrupted extraction, choose **Check continuation**.
The preview verifies the original video, saved images and sampling inputs, then
shows completed and remaining positions. Confirm explicitly to create a linked
attempt for the remaining work. The earlier job and its outcome stay in history.
Repeated or concurrent confirmation cannot create two successors for one attempt.

If a crash happened after an image was saved but before its position was recorded,
IRIS recognizes that image's operation and plan provenance and retains it. Exact
and perceptual duplicate decisions already recorded are not recomputed. Progress
and result counts for a continuation include retained work; the job detail also
counts images saved by that particular attempt.

Changed source bytes, changed saved images or changed deduplication inputs prevent
continuation. Adding unrelated extraction results to that source can change those
inputs. The original class version remains pinned even if the project's current
definitions change. Start a separate extraction to choose a different sampling
plan or class version. Older jobs without a frozen plan also require a new run.

## Continue a temporal detector cache

Temporal detection jobs calculate a frozen sequence with one pinned detector
recipe. A complete image output and its checkpoint commit together, including
images with no retained detections. Saved outputs form an exact prefix of the
sequence's available frames. Cancellation or interruption leaves that prefix
intact; missing frames are not reported as empty predictions.

Use `GET /api/jobs/{job_id}/recovery` to inspect a failed, cancelled or interrupted
attempt, then `POST /api/jobs/{job_id}/recover` with its returned `fingerprint`.
Pass the owning `project_id` on both requests. Confirmation creates a linked job
for the remaining frames. It does not overwrite the earlier attempt or rerun its
saved images. Inspect an existing successor rather than submitting the same
continuation again.

Changed source bytes, weights, preprocessing or pinned runtime prevent continuation.
The worker also checks actual device, hardware and execution settings after loading
the detector; it refuses to append results from a different execution environment.
Create a fresh cache to intentionally change the recipe. Each attempt records its
own loading and warmup cost separately. Reading complete saved results needs no
weights or ML runtime. See [temporal detector caches](temporal-detections.md).

## Restart a tracking measurement

Visual tracking comparisons and [tracking cost measurements](tracking-cost.md)
publish complete reports only. A failed, cancelled or interrupted cost job does
not retain a successful partial timing report. Start a new measurement to obtain
fresh setup, warmup, tracker state and a complete set of repetitions. Previously
completed reports remain unchanged; viewing them does not execute the pipeline.

[Profile studies](tracking-studies.md) also publish complete reports only. Their
explicit frame-update budget is checked before launch, and their cooperative time
budget is checked during execution. A cancelled, failed or time-limited attempt
does not publish its tested subset as a completed comparison. A new launch keeps
the earlier attempt in history and starts every selected profile again.

## Prepare unfinished local batch images

A stopped local annotation batch can prepare a **new batch** for its failed,
cancelled and interrupted images that have no saved proposals. Successful images
and images with proposals remain available in the earlier batch and are excluded.
The new preview checks current saved annotations, source predictions, selection,
model availability and settings. Review its eligible and excluded images before
confirming new requests. This does not resume a provider's earlier computation.

The original batch and child records remain unchanged. The successor records its
parent and preview fingerprint; repeating a confirmation after a lost HTTP response
returns the same batch. If that successor later needs recovery, use its own details.
Batch recovery accepts local Ollama reviews only; it cannot enqueue API reviews.

## External calls with an uncertain outcome

Annotation and video reviews retain a durable dispatch receipt. A request can be
**not started**, **dispatching**, **response received**, or **outcome unknown**.
The provider, model, attempt time and available response identifiers help inspect
what happened. Receiving a response does not prove that it was valid or that the
job completed successfully. A stored transport-error envelope alone is not proof
that the provider responded.

If IRIS stops or loses the connection after dispatch, the provider may already
have processed and billed the request. IRIS preserves the uncertain state and
does not resend it automatically. It also prevents a second execution of the
same saved request. Older external records without sufficient dispatch evidence
are treated conservatively; no zero-cost claim is inferred from their failure.

Inspect the saved evidence and the provider's records before deciding on another
request. A new API review still requires a new image or storyboard preview and
explicit cost approval. Recovering a lost browser confirmation first looks for
the saved job receipt; it does not send the provider request again.

## Other jobs and limits

Training runs with durable checkpoints offer **Preview continuation** in their
training details. Confirmation creates one linked attempt with the same frozen
inputs, restoring the optimizer, CPU random state, selected CUDA random state
when applicable, and image order. Continuation retains the original training
device and runtime; a completed model can independently use CPU or CUDA. Work after the
latest saved state is recomputed; the old attempt stays unchanged. Starting from
a completed model instead initializes a new optimizer. Older runs without saved
optimizer state require a new run. See [training continuation](long-training.md).

Comparisons and evaluations retain their saved predictions and metrics.
**Prepare a new run** opens the relevant workspace without launching processing.
A new comparison or evaluation is a separate run. No generic retry button
silently reruns model or API work.

The upload queue remains a browser operation: completed uploads are preserved,
but closing the page does not retain unuploaded browser files for later processing.
Extraction checkpoints and dispatch receipts use existing SQLite records; schema
14 and earlier frozen datasets remain compatible with workspace backup and restore.
Training recovery adds schema 18; opening an older workspace adds its checkpoint
table without rewriting previous training records or model files.
Temporal detection caches add schema 21 without changing previous image annotations,
temporal references or saved datasets. Workspace restoration never resumes a cache
job automatically.
