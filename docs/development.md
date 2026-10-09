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

## Continuous integration

[The GitHub Actions workflow](../.github/workflows/ci.yml) runs on pull requests,
pushes to `main`, and manual dispatch. It checks:

- Ruff lint and formatting, plus the JavaScript helpers on Node 22.
- The full Python suite with the base installation on Python 3.12 and 3.13.
  Missing optional runtimes and live opt-ins appear as skips in the test summary.
- A selected set of native CPU tests on Python 3.12 with `ml` and `tracking`
  installed: model class mappings, training state and scopes, generated-weight
  optimization, native trackers and portable pipeline behavior. This job fails
  if any selected test is skipped.

Dependencies come from `uv.lock`; the ML extra uses CPU PyTorch packages.
Setup downloads software dependencies, but no model checkpoints. Tests generate
their own data and weights, leave live opt-ins disabled, and need no API secrets
or model service. The CPU job does not establish CUDA, pretrained-checkpoint,
ONNX conversion or real-image quality coverage, nor optional-runtime coverage on
Python 3.13. Those checks remain separate.

Actions are pinned to commit SHAs and uv to a fixed version. The workflow uses
[setup-uv's cache](https://github.com/astral-sh/setup-uv#usage) with separate keys
for the base installations and CPU extras. Job timeouts are 10 minutes for checks,
45 for each base suite and 20 for CPU fixtures; these are limits, not measured
GitHub runner durations. New runs cancel an older run for the same branch or PR.

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
