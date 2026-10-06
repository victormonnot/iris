# Combined Astra + SAM 3 adapter

The separate [DINO-X → Astra review experiment](dinox-astra-review.md) reuses
completed detector boxes without a planning stage. It does not change or validate
the SAM pipeline described here.

Benchmark approach C uses a bounded, frozen three-stage protocol:

1. **Planning:** GPT-6 Astra sees the prepared image and the frozen class names,
   IDs and definitions. It returns exactly one short SAM phrase for every class.
   It cannot return boxes, masks, points, scores or candidate objects.
2. **Grounding:** local SAM 3 evaluates those phrases on the original image.
   Its native detector supplies all candidate geometry. This uses the same
   checkpoint, CUDA execution profile and box normalization as approach B.
3. **Review:** Astra sees the same prepared image, class definitions, generated
   phrases and an identified list of SAM candidates. Each candidate receives
   exactly one `accept`, `reject` or `relabel` decision. No coordinates are
   accepted in the review response.

The planner gives SAM short visual concepts instead of arbitrary instructions;
the local detector supplies geometry; the reviewer can reject or relabel those
existing candidates. This is a testable division of work, not evidence that C
outperforms either provider alone. It cannot recover objects that SAM missed.
There is no iterative search, repeated planning, automatic repair, model
fallback or extra provider call in v1.

All real trials remain deferred. Tests use synthetic SAM results and intercepted
HTTP responses. No weight download, installation, training or live API request
is needed to prepare or validate a configuration.

## Frozen configuration

`iris-combined-preannotation-v1` contains a complete taxonomy snapshot,
`openai_config` and `sam_config`, plus the exact planning/review instructions and
strict JSON schemas. The model identity is `gpt-6-astra+sam3`. The OpenAI child
reuses the existing explicit Responses settings, image encoding and pricing
profile; the SAM child reuses the pinned source/checkpoint/runtime identities.
See [the OpenAI adapter](openai-preannotation-adapter.md) and
[the SAM adapter](sam-preannotation-adapter.md).

Planning emits `prompts: [{class_id, text}]` in frozen class order. Missing,
repeated or unknown classes are errors. Phrases must be printable, nonempty and
at most 120 characters. SAM's provisioned tokenizer also checks its limit of
30 content tokens before grounding. A generated phrase that fails this check
fails the stage; the system does not request a replacement automatically.

Each image gets one image encoding in SAM and one text evaluation per class.
There are at most **two external calls per image**: planning and review. Review
is still required when SAM produces zero candidates, and its only valid result
then has an empty `decisions` list. This records that step explicitly without
treating missing or failed work as a successful empty result.

Review emits:

```json
{
  "decisions": [
    {
      "id": "sam3-0-7",
      "action": "accept",
      "label": "person",
      "reason": "Visible evidence supports the supplied class.",
      "uncertain": false
    }
  ]
}
```

Every SAM ID must occur exactly once. `accept` and `reject` preserve the input
class in `label`; `relabel` must choose another frozen class. Unknown IDs,
additional geometry, omitted IDs, duplicate decisions, invented classes and
invalid uncertainty flags fail validation. Decisions can be returned in any
order; saved results follow the original SAM order.

The final proposals keep the original SAM IDs and exact pixel coordinates.
Their score is **null**: combining a SAM score with an Astra decision does not
create a new calibrated confidence. The native SAM score, original class,
native-box provenance, review decision, returned Astra model and response ID
remain inspectable. Rejected candidates remain in the complete decision record.
No stage validates human annotations or changes the independent benchmark
reference.

## Preview, dynamic inputs and budget

The planning request is fully determined before execution. The review's
candidate list and phrases are unknown until planning and grounding finish.
The preview therefore displays an explicit **review template**, never a fake
completed request. It freezes the image identity, instructions, class snapshot,
response schema, settings and a 128 KiB maximum allowance for dynamic UTF-8
text. The allowance counts the generated phrases and candidate list after their
JSON escaping inside the prompt. Static text and schemas are bounded separately
to 64 KiB. Each actual review request is checked against those limits before
dispatch. There are no more than 100 candidates.

The same metadata-free RGB PNG encoding is used for both external requests.
SAM operates locally on original pixels. Review candidate coordinates are
normalized against the whole original image, so they remain meaningful on the
resized external image; final stored coordinates are never rescaled by Astra.

Each request has two distinct digests:

- `input_sha256` binds its safe image descriptor, prompt/schema, settings,
  protocol and stage. Archives can reconstruct and check this without uploading
  an image or loading SAM.
- `request_sha256` is the hash of the actual serialized HTTPS request, including
  the PNG data URL. It is recorded as sent evidence, not represented as
  independently recomputable from metadata without image bytes.

The review template has its own `template_sha256`. It deliberately has no
`request_sha256`: the actual review body does not yet exist. Archive validation
reconstructs the later review input from the recorded planning result and
canonical SAM proposals, rather than trusting a free-standing candidate list.

The admission estimate reserves the exact planning allowance plus a review
allowance with the full dynamic text limit and both maximum output limits.
It uses UTF-8 byte counts, framing/image allowances and the frozen cache-write
input rate. This is a conservative **planning estimate, not a guaranteed billing
cap or an invoice**. The protocol's two-call limit does not imply a fixed price.
Observed usage and its estimated cost are recorded per response; unavailable
usage or an unknown dispatch outcome must remain explicit.

The pinned Astra price profile uses USD 10 input, USD 1 cached input, USD 12.50
cache writes and USD 50 output per million tokens. The documented long-context
rate multipliers remain part of the estimate. Current source: [GPT-6 Astra
model and pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).

## Transport and provenance boundary

`CombinedOpenAI.request` accepts only a prepared request and two mandatory
callbacks. The durable `before_dispatch` callback runs immediately before its
single HTTPS POST; `after_response` receives raw response evidence and measured
metadata before any planning or review normalization. Callback conflicts remain
callback conflicts. They are never converted into a network failure or retried.
Complete HTTP error responses and malformed JSON retain response receipts;
incomplete transport remains distinguishable from a complete response.

Requests use Responses, `store: false`, no tools, no conversation state and no
background execution. `store: false` is not a claim of zero provider retention.
See [Responses structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[image input](https://developers.openai.com/api/docs/guides/images-vision) and
[OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data).

The benchmark orchestrator owns explicit consent, budget admission, stage
claims, cancellation, restart behavior, results and artifact publication.
The provider module receives no Store, reference boxes, correction notes,
benchmark filenames or human-review history. Only frozen class definitions,
image pixels and this pipeline's generated evidence cross its boundary.
Safe request records exclude PNG bytes, data URLs, credentials and arbitrary
provider metadata. The key comes from the existing server environment only;
neither configuration preparation nor construction makes a provider request.

Historical validators are pure. Changing the prompts, schemas, image policy,
model profile or algorithm requires a new version rather than rewriting v1.
The separate per-stage raw responses, identities and decisions are essential
to distinguish an invalid generation from an actual successful empty result.
