# Local annotation batches

A batch prepares model review proposals for several selected frames in one
session. Each image has its own durable assistance record and processing job.
Saved human annotations remain unchanged until you explicitly review and save them.

## Prepare a batch

1. Select images in **Data intake**, then open **Annotation → Local batch review**.
2. Check the images to include, up to **25**. This checklist does not change dataset
   selection. Save or discard unsaved edits on a frame before including it.
3. Choose an installed local Ollama vision model. Availability checks do not load
   or download a model. An unavailable runtime or model blocks preparation with an
   explicit reason.
4. Choose the candidate source: **saved labels**, or a **completed comparison**
   and one of its detectors. For detector results, set a confidence threshold.
   A batch does not rerun that detector.
5. Preview eligibility, inspect skipped images and their reasons, then start the
   eligible reviews. Changing the selection, source, model or review settings
   requires a new preview.

Every eligible image must have **1–8 person/car candidate boxes**. Zero candidates,
too many candidates, missing predictions or image files, and an already active
assistance job make an image ineligible. Imported proposals are not saved labels:
review and save them first, or use saved detector predictions as the source.
Images without candidates are not treated as validated negatives.

The model examines the image and candidate crops. It can recommend keeping,
changing or rejecting a category, or express uncertainty. It does not generate new
box coordinates or guarantee that missed objects will be found. Inspect the whole
image during human review.

## Follow and review results

The batch history shows each image's status, progress, errors and number of saved
proposals. Images use the existing single local worker, so reviews run one at a
time alongside other queued processing jobs. A failed image does not discard other
results or prevent later images from running.

Open a frame from the batch results to review its proposals in the annotation
editor. You can accept, correct or reject them, then explicitly validate the frame.
New proposals do not overwrite unsaved editor changes or create human revisions.

## Cancel, restart and retry

**Cancel batch** cancels waiting images and requests cancellation of an active
review. Already saved proposals, raw responses and provenance remain available.
The local provider may continue computing briefly after the worker is stopped;
cancellation is not a guarantee of immediate GPU release.

Reopening IRIS retains batch history. Jobs that were active when the server stopped
become **interrupted**; they are not automatically restarted. A batch with mixed
terminal outcomes is shown as **partial**, with each outcome visible. Progress
alone does not indicate success: inspect image statuses and proposal counts.

For a stopped batch, prepare a new batch for unfinished images from its details.
The preview includes failed, cancelled and interrupted images with no saved
proposals; successful images and images with proposals remain in the earlier batch.
Current saved inputs are checked again, and creation requires explicit confirmation.
The new batch links to its parent. Repeating a confirmation returns that same batch.
You can also select images manually to prepare a separate batch. There is no
automatic retry or silent reuse of an earlier request. See [jobs and recovery](job-recovery.md).

## Provenance and local execution

The preview identifies the requested images and freezes their saved revisions,
candidate sources, image hashes, model digest and review settings. Queueing checks
these inputs again and creates all eligible child jobs in one transaction. If
inputs changed, prepare a new preview. Each child retains its candidates and model
configuration; the worker verifies image bytes and the installed model digest.

Batches accept local Ollama models only. The API rejects external provider,
endpoint, budget and consent fields. Single-image API review retains its separate
image preview and explicit consent workflow.

HTTP endpoints:

- `POST /api/sessions/{session_id}/assistance-batches/preview`
- `POST /api/sessions/{session_id}/assistance-batches`
- `GET /api/sessions/{session_id}/assistance-batches`
- `GET /api/assistance-batches/{batch_id}`
- `POST /api/assistance-batches/{batch_id}/cancel`

Creation requires the fingerprint returned by a matching preview. Invalid request
fields return `422`; stale inputs and unavailable models return `409`.
