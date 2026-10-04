# OpenAI proposal adapter

`multimodal_provider.py` implements one explicitly approved Responses request to
`https://api.openai.com/v1/responses`, using `gpt-6-astra`. Configuration and image
preparation are offline. Server credentials come from `IRIS_OPENAI_API_KEY`, or
`OPENAI_API_KEY` when the IRIS variable is absent. They never enter saved settings,
request previews, response evidence or the browser. A ready configuration means
that a key is present; account access and model availability remain untested.

The request contains one cleaned RGB PNG, the class IDs/names/definitions, and a
fixed task prompt. It contains no human reference, annotation notes, prior
candidate boxes or correction records. The long edge is selectable among 512,
1024, 1536 and 2048 pixels, default 1536. `detail: original` preserves this prepared
image within the documented model limits. Returned normalized coordinates map to
the original image dimensions. The preview records both dimensions, hashes and
the transform. The model can still mislocalize objects; geometry validation is
not evidence of visual accuracy. [Official image documentation](https://developers.openai.com/api/docs/guides/images-vision)

Responses uses a strict JSON schema. IRIS then independently checks completion,
refusal, class IDs, geometry, proposal count and types. Every proposal has
`score: null`; no confidence score is invented. A refusal, invalid result or
incomplete response is an error, never a successful empty prediction.
The default output limit is 4096 tokens, configurable from 1024 to 8192, with a
maximum of 100 proposals. [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

`before_dispatch()` runs immediately before the single POST. A complete HTTP
response, including a rejected request or malformed JSON envelope, invokes
`after_response(raw, metadata)` before proposal validation. Callback failures
propagate unchanged. Network interruption without a complete response remains an
unknown outcome. There is no retry, redirect, proxy lookup, model fallback, tool
call or automatic response recovery. Raw evidence is bounded to 2 MiB and redacted.

Pricing was checked on 2026-10-04: Standard rates per million tokens are $10 input,
$1 cached input, $12.50 cache writes and $50 output. Above 272,000 input tokens,
the input rates double and output increases by 1.5. Reasoning tokens already belong
to the reported output total. IRIS records the requested alias, returned model,
response ID and reported usage. Calculated cost uses the frozen rates and is not
an invoice; missing usage means unknown cost. No immutable model-weight digest is
available. [Model and pricing](https://developers.openai.com/api/docs/models/gpt-6-astra)

The offline planning allowance combines UTF-8 prompt/schema bytes, 4096 framing
tokens, image patches and the full output limit, using the higher cache-write
input rate. Its `upper_bound_usd` field is an admission estimate, **not a guaranteed
billing cap**. Actual provider usage, pricing changes and taxes can differ.

The v1 prompt, encoding, schema and price profile are immutable protocol data.
Future price or request changes must introduce a new version and retain the v1
validator, so saved configurations and offline workspace archives remain readable.

Requests set `store: false`, `background: false`, `stream: false` and use no
conversation state. `store: false` does not establish zero retention for all
provider processing, including abuse monitoring and prompt caching.
[Provider data controls](https://developers.openai.com/api/docs/guides/your-data)

Protocol tests use synthetic transport fixtures only. They establish request and
failure handling, not model availability, localization quality or actual billing.
