# Review existing boxes with a vision model

[Documentation](README.md)

**Annotation → Multimodal review** asks a local or hosted Qwen model to check
existing person/car boxes. It can suggest a different class, rejection or
uncertainty. It does not draw new boxes or find every missing object.

This workflow requires the original `iris-objects-v1` Person / Car definitions
and **1–8 candidate boxes** from saved annotations or a saved comparison result.
For new box proposals or custom classes, see [detector preannotation](preannotation.md)
and [class compatibility](classes.md).

Each request includes one scene resized to at most 1024 pixels on its longest
edge, with candidate crops up to 320 pixels. Review the returned proposals in
the [annotation editor](annotation-editor.md), correct geometry, add missing
objects and validate the whole image yourself. A model response never validates
an image automatically.

## Local Qwen through Ollama

Install [Ollama](https://docs.ollama.com/linux) separately. Iris does not install
it or pull models automatically. The default
[Qwen3-VL 4B Instruct download](https://ollama.com/library/qwen3-vl:4b-instruct)
is about 3.3 GB, plus the runtime. Memory needs and execution speed depend on
your hardware; CPU review can be slow.

For a manually managed Ollama server, run this from the project directory in a
separate terminal. The example stores weights in the default ignored workspace:

```sh
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 \
  OLLAMA_MODELS="$PWD/.iris/ollama" ollama serve
```

If Ollama already runs as a service, configure that service instead of starting
a second server. Use the desired workspace path for `OLLAMA_MODELS`.
[Ollama's configuration guide](https://docs.ollama.com/faq) explains these variables
and service configuration.

In another terminal, explicitly download the model:

```sh
OLLAMA_HOST=127.0.0.1:11434 ollama pull qwen3-vl:4b-instruct
```

In Iris, refresh availability under **Multimodal review**, choose **Local**,
select the installed vision model and a saved candidate source, then request a
review. Images are sent to Ollama only when you launch a review.

The defaults can be set when starting Iris:

```sh
IRIS_OLLAMA_URL=http://127.0.0.1:11434 \
  IRIS_OLLAMA_MODEL=qwen3-vl:4b-instruct uv run iris
```

Keep `--extra ml` on `uv run` if you also use Iris's local detectors. Ollama itself
does not require the Iris ML extra. The adapter accepts loopback HTTP endpoints
and installed local vision models; it rejects cloud names and remote aliases,
ignores HTTP proxies and does not follow redirects.

Prompts, candidate provenance, raw responses, generation settings and the local
model digest are saved in the workspace. Invalid or incomplete responses fail
without publishing proposals. For several frames, see [local annotation batches](annotation-batches.md).

## Hosted Qwen through Alibaba Cloud

Choose **API** for **Qwen3-VL 32B Instruct** or **Qwen3-VL 235B-A22B Instruct**.
These requests send the previewed images and prompt to Alibaba Cloud Model Studio;
they do not require a local model download.

Create a workspace and API key separately, then configure the server environment:

```sh
export IRIS_DASHSCOPE_BASE_URL="https://YOUR_WORKSPACE_ID.eu-central-1.maas.aliyuncs.com/compatible-mode/v1"
# Supply IRIS_DASHSCOPE_API_KEY through your shell or secret manager.
.venv/bin/iris
```

Iris does not load `.env` files automatically. Its presets use a Frankfurt
workspace endpoint with **Global deployment scope**, so that endpoint alone does
not imply that inference stays in the EU. Check the provider's
[region and deployment documentation](https://www.alibabacloud.com/help/en/model-studio/regions)
when configuring the account.

**Configured** means settings are present, not that credentials have been tested.
Refreshing availability and preparing a preview do not contact Alibaba.

1. Select the API model and saved candidate source.
2. Generate a preview. Inspect the exact scene and crops, endpoint, review focus
   and conservative cost ceiling in USD.
3. Approve and launch that request. A preview expires after 30 minutes and can
   be used once; changing its selection or configuration requires a new preview.
4. Review the returned proposals and save your decisions in the annotation editor.

The ceiling uses maximum input tokens and a 1,024-token output limit at recorded
list prices, rather than predicting actual token use. Taxes and later price
changes are excluded; the provider bill remains authoritative. The saved request
includes the dated price source and reported usage when available.

Iris makes one attempt, with no automatic retry or provider fallback. Invalid,
incomplete or wrong-model responses create no labels. Cancelling stops the local
worker, but the provider may still finish and bill an already received request.
Another attempt requires a new preview and approval.

Outgoing image hashes, approved ceiling, provider, model, prompt, response and
usage remain traceable. A hosted model ID is not an immutable weight digest.
API keys stay in the server/worker environment and are not returned to the UI.

## Related workflows

[Video review](video-review.md) uses storyboards and event proposals with its own
limits. [DINO-X](dinox-preannotation.md) proposes new boxes, while the
[annotation benchmark](benchmark.md) compares methods against an independent
reviewed reference. See [recorded experiments](acceptance-results.md) for the
real runs and the limits of their quality measurements.
