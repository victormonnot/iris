# SAM 3 image adapter

Benchmark approach B uses **SAM 3 image concept detection**, locally, with one
explicit short text phrase per frozen class. The independent human reference,
correction notes, candidate boxes and previous predictions are never model input.
The class definitions remain visible in the saved configuration, but SAM receives
the selected phrases, not those full definitions.

This integration has protocol and simulated-runtime tests. Real SAM execution,
weight provisioning and hardware measurements are deferred. IRIS never installs
SAM, downloads its weights or falls back to an external service automatically.

## Pinned model and sources

The `iris-sam3-preannotation-v1` profile freezes these public identities:

| Component | Identity |
| --- | --- |
| Meta source | `facebookresearch/sam3` at `2345a4ad109ac29c569da749c91d84f10dc08c40` |
| Hugging Face model | `facebook/sam3` at `3c879f39826c281e95690f02c7821c4de09afae7` |
| Checkpoint | `sam3.pt`, 3,450,062,241 bytes |
| Checkpoint SHA-256 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| Workspace location | `models/sam3/sam3.pt` |

The checkpoint identity was read from the model's public repository metadata,
without fetching the weight file. Sources: [pinned Meta repository](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40),
[Meta model card](https://huggingface.co/facebook/sam3/blob/3c879f39826c281e95690f02c7821c4de09afae7/README.md),
and [Hugging Face file metadata](https://huggingface.co/api/models/facebook/sam3?blobs=true).

SAM 3.1 is a different checkpoint/profile. Its March 2026 release introduces
Object Multiplex for shared-memory multi-object **video tracking**. The native
image builder still selects the SAM 3 checkpoint. This adapter therefore does
not claim to run SAM 3.1, or infer image-quality improvements from its video
benchmarks. See [Meta's SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/RELEASE_SAM3p1.md).

The actual license is the **SAM License**, despite an inconsistent MIT classifier
in package metadata. Checkpoint access requires accepting Meta's terms through
the gated model repository. Configurations and historical results can be backed
up and restored without installed weights. If `models/sam3/sam3.pt` is present
in the workspace, the archive includes it and verifies the pinned file size and
SHA-256. The separate runtime environment and credentials remain excluded. See the [pinned
license](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/LICENSE)
and [official access instructions](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/README.md#getting-started).

## Native API and geometry

The official image interface constructs `build_sam3_image_model(...)`, then uses
`Sam3Processor.set_image(image)` and `set_text_prompt(prompt, state)`. It returns
native detector boxes, scores and masks. Detector boxes are not bounding boxes
recomputed from mask pixels. Sources: [model builder](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model_builder.py)
and [image processor](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_image_processor.py).

IRIS's isolated worker follows the image/text detector path and disables the
segmentation head. It retains detector boxes and native query indices rather than
allocating masks at full image resolution. This is a boxes-only implementation of
that model path; it is not a claim of segmentation-mask evaluation. It must not
call the standard processor's complete grounding postprocess with segmentation
disabled, because that method requires `pred_masks`.

The frozen preprocessing is RGB, square 1008×1008 resize and mean/std 0.5.
Execution uses CUDA BF16 autocast, with no CPU or precision fallback. One image
embedding is reused for independent text phrases. The native score is
`sigmoid(detection_logit) * sigmoid(presence_logit)`. Filtering uses a strict
`score > threshold`, matching Meta's processor. These scores are not calibrated
probabilities and are not comparable with another provider's scores.

The worker returns `iris-sam3-native-boxes-v1` with:

```text
image: {width, height}
coordinates: {format: xyxy, space: normalized, image_size: [W,H],
              to_original: {scale: [W,H], offset: [0,0]}}
prompts: [{class_id, text, boxes, scores, native_indices, error}]
complete: boolean
metadata: runtime and measured timing evidence
```

There is one row for every configured phrase, including empty outputs. Missing
phrases, partial execution, mismatched geometry, non-finite values and invalid
native indices are errors. All returned geometry is validated, including boxes
below the cutoff. IRIS clips native normalized boxes to image bounds, scales to
original pixel coordinates, and preserves the original box and a `clipped` flag
in proposal provenance. A retained box with no remaining positive area is
rejected; a finite, properly ordered native box below the cutoff is filtered
even if it lies entirely outside the image.
Overlapping proposals for different classes remain separate: there is no hidden
cross-class suppression. More than 100 retained proposals fails the image
explicitly instead of silently truncating its objects.

## Phrase and resource limits

SAM's concept task uses short noun phrases, such as `safety helmet`, rather than
arbitrary instruction following. IRIS freezes one editable phrase of 1–120
printable characters per class, in taxonomy order, for 1–100 classes. The
model's text encoder actually uses 32 tokens including its start/end markers:
the runner refuses phrases exceeding **30 content tokens** before inference.
The tokenizer's standalone default of 77 must not be mistaken for the encoder's
limit. No phrase is silently truncated. See [the text encoder](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/text_encoder_ve.py)
and [tokenizer](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/tokenizer_ve.py).

Configuration preparation needs neither weights nor the tokenizer. Token-length
validation requires the provisioned local tokenizer; consequently a character-
valid frozen phrase can still fail that later check. Language and domain quality
remain unmeasured until the deferred trials. Choosing a phrase does not prove
that SAM implements the full human class definition.

The profile bounds each image to 16,777,216 pixels and its transport PNG to
16 MiB. Each phrase has at most 300 native detections and a final image has at
most 100 retained proposals. Native JSON evidence is bounded to 2 MiB. Model
initialization and a prediction each have a 300-second deadline. Timeouts and
cancellation preserve returned partial evidence when available; missing results
are never interpreted as successful empty detections.

## Future manual provisioning

The Meta package requires NumPy `<2`, whereas the IRIS application uses NumPy
`>=2`. Keep SAM in its **own Python virtual environment**, exposed to IRIS through the
server-side `IRIS_SAM_PYTHON` executable setting. Do not install it into the IRIS
environment. The v1 runtime profile pins PyTorch 2.10.0, torchvision 0.25.0 and
NumPy 1.26.4, requires Python 3.12 and CUDA ≥12.6, and requires GPU BF16 support.
The isolated worker currently requires **Linux**, including its parent-death
process guard. Python 3.13 is outside this profile: NumPy 1.26.4 targets older
Python versions. The installation must be **noneditable and directly from the
pinned Git URL**: the worker verifies the distribution's `direct_url.json`.
Installing a checkout with `pip install .` or `pip install -e .` does not provide
the required immutable VCS provenance and is rejected.
These choices are an explicit IRIS execution profile, not an assertion that all
other upstream combinations fail. Sources: [Meta prerequisites](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/README.md#prerequisites)
and [package dependencies](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/pyproject.toml).

The native builder has a CPU-looking signature, but this pinned source allocates
CUDA tensors during positional-encoding precomputation. IRIS therefore reports
CUDA as required instead of advertising an unverified CPU fallback. See
[positional encoding](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/position_encoding.py).
No reliable minimum VRAM guarantee is made from the checkpoint's file size.

When real trials are authorized later, provision the exact source checkout and
its packaged BPE tokenizer in that separate environment, accept checkpoint terms
and fetch the pinned checkpoint manually, then put the verified file at the
workspace-relative path above. The worker supplies an explicit checkpoint, uses
the tokenizer packaged with that pinned source, and disables Hugging Face
checkpoint loading. Runtime checks and execution are local; no Hugging Face
token is needed by IRIS for inference.

The following is a **future manual provisioning recipe**, not a step IRIS runs.
It downloads packages and source code; it has not been executed as part of this
brick. It requires Git, Python 3.12 with `venv`, and an appropriate NVIDIA driver.
Run it only when the deferred installation and real trials are authorized:

```sh
IRIS_SAM_ENV="$HOME/.local/share/iris/sam3-env"
python3.12 -m venv "$IRIS_SAM_ENV"
"$IRIS_SAM_ENV/bin/python" -m pip install --upgrade pip 'setuptools<82' wheel
"$IRIS_SAM_ENV/bin/python" -m pip install \
  'torch==2.10.0' 'torchvision==0.25.0' \
  --index-url https://download.pytorch.org/whl/cu128
"$IRIS_SAM_ENV/bin/python" -m pip install \
  'numpy==1.26.4' einops psutil 'pycocotools==2.0.11'
"$IRIS_SAM_ENV/bin/python" -m pip install --no-build-isolation \
  'sam3 @ git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40'
export IRIS_SAM_PYTHON="$IRIS_SAM_ENV/bin/python"
```

Preserve that absolute executable setting in the environment that starts the
IRIS server. Keep the interpreter's virtual-environment path rather than
resolving its symlink to the system Python. After accepting the model terms,
obtain the checkpoint manually at the Hugging Face revision listed above and
copy it as a regular file to the workspace location. No checkpoint-fetch command
is part of application startup.

The extra packages above address unconditional imports in the pinned source:
`einops` in [RoPE](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/sam/rope.py),
`pycocotools` in [COCO loaders](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/train/data/coco_json_loaders.py),
and `psutil` in [the video module imported by the shared builder](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_video_predictor.py).
The builder also imports `pkg_resources`; setuptools
[removed that module in version 82](https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html),
so this source pin needs the older setuptools runtime. These import dependencies
do not mean that IRIS runs video inference or trains SAM.

`provider_status(root, config)` reports local runtime/weight readiness without
loading the model or sending a request. It does not prove inference succeeded.
`Sam3Preannotator` loads once per trial, reports initialization timing separately,
and accepts only an image and cancellation callback in `predict`. The benchmark
orchestrator owns immutable previews, runtime identity revalidation, job state,
raw-output publication and independent quality scoring.

Frozen configuration validation never imports Torch or consults current runtime
state. Keep the v1 source/checkpoint/profile constants immutable; a new model,
precision or preprocessing profile requires a new protocol. This lets old
benchmarks and archives remain readable after runtime removal.
