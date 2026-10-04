# Detector preannotation

**Annotation → Generate proposals** creates new boxes directly from images,
without existing labels or an earlier comparison. It uses installed local
detectors through the existing worker. Manual drawing, editing and validation
remain available without model weights, a provider runtime or an API key.

## Prepare and review

1. Select images in Data intake. In Annotation, choose the current image or a
   checklist of up to 25 selected images from the same session.
2. Choose an installed detector, a proposal threshold and full-image or tiled
   inference. Save or discard unsaved edits before preparing these images.
3. Preview the images, eligible/excluded counts, class coverage and detector work.
   Official COCO models require explicit COCO mappings. Trained checkpoints require
   the exact saved class definitions. Classes without a mapping remain manual work.
4. Explicitly start the previewed run. Follow each image's state and open its
   proposals in the editor. Cancellation preserves saved results; interruption
   never automatically sends or runs a new request.
5. Accept, correct or reject proposals. Add missing objects manually. **Save draft**
   and **Validate frame** remain separate decisions. No proposals is not a validated
   negative image, and accepted proposals alone do not validate the whole image.

The proposal threshold filters saved native outputs; it does not bypass a
detector's own score cutoff, suppression or detection cap. Each image can publish
at most 100 proposals. An excess is reported with the raw output retained; use a
higher threshold in an explicitly prepared new run. Confidence scores are native
model outputs, not calibrated or comparable probabilities.

The review queue can focus on pending uncertain or low-score proposals and possible
omissions. These are inspection hints from saved evidence, not measured errors.
Both agreeing detectors can miss objects. See [review hints](review-queue.md).

## Saved evidence and conflicts

A run freezes the selected images and hashes, model and checkpoint hash, inference
settings, proposal threshold, annotation revisions, class definitions and mappings.
Predictions retain native categories, scores, coordinates, runtime metadata, timing
and any tiled-region evidence. Proposal metadata links to the source output and
its coordinate normalization. Human changes create separate immutable revisions;
raw predictions and original proposals are preserved.

Preview creates no job. Confirmation rechecks the frozen input fingerprint in a
transaction. Repeating the same confirmation returns the same saved receipt,
including after losing the HTTP response. Preparing a fresh preview after that
run creates a distinct request; it can run the detector again. A queued or running
request excludes its images from another preannotation request.

Raw output is saved before publishing proposals. If image pixels, saved class
version or annotation revision changed during processing, the image receives a
conflict and its raw output remains inspectable. It does not overwrite human work.
Invalid outputs returned by the adapter retain their raw evidence and error.
Non-JSON, non-finite or oversized outputs retain a bounded diagnostic summary
instead of an invalid or unbounded payload.
Decode, model or tile failures before a complete output is returned retain the
error and earlier complete images; partial tiled images are not published or
stored as complete predictions. A worker interrupted after
saving raw output may leave it without published proposals; a new run is explicit.
The run can therefore finish with images requiring attention; inspect per-image
states rather than assuming that job completion validates or labels every image.

## Provider capabilities

| Implemented adapter | New geometry | Classes | Processing |
| --- | --- | --- | --- |
| Local detector | New bounding boxes from original images | Frozen checkpoint definitions or explicit COCO mapping | Local installed weights |
| Ollama candidate reviewer | Keeps existing candidate boxes | Original Person / Car definitions | Local installed vision model |
| Alibaba candidate reviewer | Keeps existing candidate boxes | Original Person / Car definitions | External, separately previewed and approved |

Candidate reviewers require 1–8 boxes and may flag uncertainty or discuss omissions
in scene notes. They do not locate new objects or automatically publish new class
definitions. Existing external review still displays the exact outgoing images,
provider/model, operations and estimated cost before separate confirmation.

The capability protocol is `iris-preannotation-capabilities-v1`. Requests use
`iris-local-preannotation-v1`, their class mapping contract uses
`iris-detector-preannotation-v1`, and normalized box outputs use
`iris-box-proposals-v1`. Coordinates declare original oriented image dimensions,
pixel or normalized `xyxy`, and an explicit scale/offset into original pixels.
Unknown class IDs, mismatched definitions, non-finite values, out-of-bounds boxes
and unsupported transforms are rejected. Adapters preserve raw outputs before
normalization. Crop and mask adapters must supply their own documented conversion.

Tests use clearly identified synthetic multimodal, mask-derived and combined
examples to exercise this boundary. They do not establish real provider support,
quality or cost. No SAM, new multimodal generation API or combined pipeline is
available through this contract alone. Those integrations require separate work.

## API

- `GET /api/preannotation-providers`: implemented capabilities and project detector availability.
- `POST /api/sessions/{id}/preannotations/preview`: inspect a bounded request without executing it.
- `POST /api/sessions/{id}/preannotations`: confirm the request with `expected_fingerprint` and a name.
- `GET /api/sessions/{id}/preannotations`: reopen saved requests.
- `GET /api/preannotations/{id}`: inspect the job, frozen configuration, per-image states and raw outputs.

These routes are project-scoped. Data uses existing schema-14 comparison, prediction
and suggestion records, without a database migration or a new background service.
