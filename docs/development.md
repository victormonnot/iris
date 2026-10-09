# Development and verification

[Documentation](README.md)

Start with [setup](setup.md) and the [architecture and code map](architecture.md).
The browser uses plain HTML, CSS and JavaScript served by FastAPI. There is no
frontend build step. Node is only needed for the JavaScript tests, which use its
built-in test runner.

## Routine checks

Run from the repository root after `uv sync --locked`:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
node --test tests/js/*.test.cjs
```

Leave `IRIS_TEST_MODEL_DIR`, `IRIS_TEST_TRAINING` and `IRIS_TEST_OLLAMA` unset for
the ordinary suite. They opt into the real model checks described below.
Keep the appropriate extras on `uv run` when working on optional runtime code.

Tests use temporary workspaces, generated media and mocked external services.
They cover API and job behavior, annotation revisions, frozen datasets, model
contracts, exports and recovery. Browser helper tests cover geometry, appearance,
history and navigation. Test doubles are confined to tests and are not exposed
as application models.

These checks establish software behavior. Accuracy and speed on real recordings
need separate experiments; see [recorded results](acceptance-results.md) and
[pipeline qualification](pipeline-qualification.md). The repository includes a
small [attributed photo example](../examples/street-scenes/README.md), but no model weights.

## Optional checks with installed detectors

Explicitly install the ML runtime and both official Torchvision checkpoints
before opting in. `IRIS_TEST_MODEL_DIR` points to the **workspace containing the
models directory**, not to an individual checkpoint or the models directory itself:

```sh
IRIS_TEST_MODEL_DIR=/absolute/path/to/iris-data \
  uv run --extra ml pytest tests/test_models.py
```

The live adapter checks run SSDLite and Faster R-CNN on generated images and
compare their outputs with the official builders. They download nothing and
provide no real-image accuracy benchmark. Without the variable, those checks
are skipped. The test source is [test_models.py](../tests/test_models.py).

## Optional CPU training checks

Use already provisioned official weights:

```sh
IRIS_TEST_TRAINING=1 IRIS_TEST_MODEL_DIR=/absolute/path/to/iris-data \
  uv run --extra ml pytest tests/test_training_live.py
```

These bounded Torchvision tests perform real optimizer steps on generated data.
They check the three training depths, changed and frozen parameters, normalization
buffers, checkpoint reload and continuation. The worker workflow also evaluates
saved generations and checks persistence after restart. The three depths are
training scopes, not proof that all three supported detector architectures have
completed the same checks.

The tests download nothing. Their fixture review records are generated, not
human-reviewed field annotations. Without `IRIS_TEST_TRAINING=1`, they are skipped.
See [test_training_live.py](../tests/test_training_live.py) for the exact coverage.

## Optional local vision-language check

First provision and start [local Ollama](multimodal-review.md#local-qwen-through-ollama):

```sh
IRIS_TEST_OLLAMA=1 uv run pytest tests/test_assistance_provider.py -k real_ollama
```

The test uses `IRIS_OLLAMA_URL` and `IRIS_OLLAMA_MODEL` when set. It checks a real
response on a generated image, including output structure and provenance. It
does not measure semantic accuracy, download a model or start Ollama. An
unavailable configured provider fails the check; an unset opt-in skips it.

Hosted-provider tests use offline transports and fixtures. Passing them does
not establish paid-provider availability or annotation quality. Real submissions
remain explicit application operations with their own previews and receipts.

## API and saved contracts

The local server exposes its API schema at `/openapi.json`. API routes, workers
and portable runners share the contracts described in the feature guides.
When changing a saved format, class mapping or preprocessing recipe, check its
consumers and compatibility rules as well as the UI. The
[architecture guide](architecture.md) maps these modules and storage boundaries.
