# CPU, CUDA and embedded compute targets

Choose the training device for the machine doing the optimization, then choose
the inference or export target for the machine using the completed model. These
choices are independent:

| Training device | Completed-model inference target | Supported software path |
| --- | --- | --- |
| CPU | CPU | CPU training and CPU inference/export |
| NVIDIA CUDA GPU | CPU | GPU training, CPU checkpoint tensors, CPU inference/export |
| CPU | NVIDIA CUDA GPU | CPU training, CUDA inference/export |
| NVIDIA CUDA GPU | NVIDIA CUDA GPU | GPU training and CUDA inference/export, including another compatible GPU |

This table describes the implemented paths, not hardware certification. The
current software checks use synthetic data, tiny CPU modules and mocked CUDA
interfaces. Real detector training, CPU/GPU transfer, GPU continuation, exported
inference, numerical parity, quality and performance remain part of the planned
real-model acceptance tests. No CUDA environment was installed for those software
checks; the development environment remains CPU-only.

## Select execution devices

In **Dataset & training**, select CPU or an available NVIDIA GPU before preparing
the training plan. The list describes devices on the IRIS server, which may be a
different machine from the browser. The server checks the installed runtime,
visible GPU properties and Torchvision CUDA operator registrations without
loading detector weights. An available device does not establish that the
training scope will fit its available memory. An unavailable device includes the
reason; IRIS never silently turns a requested GPU run into CPU training.

Training supports one selected device, float32, batch size one and the existing
light, partial and full scopes. There is no automatic mixed precision, multi-GPU
training or hardware-driven adjustment of training settings. The API accepts
`device: "cpu"`, `"cuda"` (GPU 0), or `"cuda:<index>"` using the GPU numbering visible
to the IRIS process. A changed device or runtime requires a new preview. CUDA runs
always use durable recovery checkpoints, including runs of fewer than 200 steps.

Choose the inference device independently in model comparison and evaluation.
Final model weights are stored as CPU tensors and loaded onto the chosen device;
the saved training provenance does not override the inference selection. CUDA
timing synchronizes the selected device at measurement boundaries. This does not
make the measurements representative of an end-to-end embedded application.

## Set up a separate CUDA environment

The project's `ml` extra intentionally installs **CPU** wheels. Keep that managed
environment for CPU work. For a supported Linux/WSL machine with an NVIDIA driver,
create a separate environment with Python 3.12 or 3.13. These are explicit setup
commands to run when preparing that machine; selecting GPU in IRIS runs none of
them:

```sh
uv venv --python 3.12 .venv-cuda
uv pip install --python .venv-cuda/bin/python -e .
uv pip install --python .venv-cuda/bin/python torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv-cuda/bin/iris --data-dir /path/to/workspace
```

The `cu128` command is an example for a compatible driver and GPU architecture.
PyTorch also publishes CUDA 12.6 (`cu126`) and CUDA 13.0 (`cu130`) builds for these
pinned versions. Select the build using the
[official PyTorch 2.10 installation instructions](https://pytorch.org/get-started/previous-versions/#v2100)
and the requirements of the target driver and GPU; a CUDA wheel is not compatible
with every NVIDIA generation. IRIS checks whether the installed build supports
the visible GPU architecture and whether Torchvision supplies the detection
kernels it needs.

Start IRIS directly with `.venv-cuda/bin/iris` to preserve this environment. Do not
run `uv sync --extra ml` or `uv run --extra ml` against it: those commands select
the project's CPU runtime. Stop the previous server before starting another
server on the same workspace and port. Model weights are provisioned separately
through the existing explicit model download workflow; changing a runtime does
not download a checkpoint.

CPU execution remains available from a compatible CUDA environment. GPU execution
requires the CUDA runtime and a visible compatible GPU. Runtime checks accept the
pinned public Torch/Torchvision versions with platform build suffixes such as
`+cu128`; vendor prerelease builds are not interchangeable with those exact
version requirements.

## Continue an interrupted run

Recovery is more restrictive than completed-model inference. The optimizer state,
CPU RNG and selected CUDA GPU RNG belong to the original training attempt. A
continuation preserves its device index, configuration, data, runtime and GPU
identity, including UUID when available. It cannot change CPU to GPU or transfer
an unfinished run onto another GPU. Start a new fine-tuning run from a completed
model when intentionally choosing a different training device; this initializes
a new optimizer.

CPU states retain the original `iris-training-state-v1` protocol. CUDA uses
`iris-training-state-cuda-v1`; its tensors are serialized on CPU and restored onto
the selected device. CUDA runtime identity includes the Torch/CUDA/cuDNN builds,
GPU properties, TF32 and cuDNN settings and cuBLAS workspace configuration. RNG
restoration does not guarantee bitwise equality with uninterrupted GPU execution:
the detection pipeline can use nondeterministic CUDA operations. See
[longer training and explicit continuation](long-training.md) for the full
recovery and storage contract.

## Export and verify the destination

In **Model exports**, select a CPU or CUDA target independently of the training
device. A completed full-image evaluation supplies saved reference predictions
from CPU or CUDA. Exports preserve the original weights, classes, input/output
recipe and reference device. They never run inference while being packaged.

The standalone runner requires Python 3.12 or 3.13, PyTorch 2.10.0, Torchvision
0.25.0 and Pillow 12.3.0. Provision its dependencies for the destination first;
the bundle does not install them. Inspecting a bundle requires only Python's
standard library. A CUDA target can select a visible CUDA index:

```sh
python /path/to/export/run.py inspect
python /path/to/export/run.py check-runtime --device cuda:0
python /path/to/export/run.py predict /path/to/image.png --device cuda:0 --output /path/to/prediction.json
python /path/to/export/run.py measure --device cuda:0 --repeats 3 --output /path/to/measurement.json
```

For a CPU-target bundle, omit the device option or use `--device cpu`.
`check-runtime` examines the runtime and device without loading the model or
executing inference. A successful check is followed by explicit real inference
and measurement on the target; it is not a performance or compatibility
certification. The target device must match the bundle's CPU/CUDA profile.

Parity remains an exact comparison against the saved reference. Different devices
or runtimes can produce different boxes or scores; those differences stay visible
as failed parity, including when crossing from a CPU reference to a CUDA target.
Import the full measurement report into IRIS and assess it alongside independent
quality measurements. See [model export](model-export.md) for profile versions,
timing scopes and report interpretation.

## Embedded and ARM targets

A target must provide the runner's Python version, pinned Torch/Torchvision/Pillow
versions, matching native detection operators and sufficient memory. CPU and CUDA
are execution families, not declarations that every processor or board can run
the bundle. An embedded Linux CPU or compatible NVIDIA GPU target can use the
same weights when those requirements are satisfied.

For Jetson, the available vendor runtime depends on the board, JetPack release,
Python version and compatible PyTorch/Torchvision builds. Use NVIDIA's
[Jetson framework compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#compatibility)
to assess that combination. The desktop CUDA wheel example above is not a generic
Jetson installation command. A vendor wheel with a different or prerelease Torch
version does not satisfy this export profile merely because CUDA is available.

Older Jetson Nano devices illustrate why the board matters: NVIDIA identifies
JetPack 4.6.6 as their final JetPack release. That fact does not establish
compatibility with IRIS's modern pinned runtime; such targets need a separately
validated deployment solution.
([NVIDIA JetPack 4 end-of-life notice](https://forums.developer.nvidia.com/t/announcing-end-of-life-for-nvidia-jetpack-4-with-the-release-of-jetpack-4-6-6/314409))

This feature supplies native PyTorch CPU/CUDA execution and portable weights. It
does not convert models to ONNX or TensorRT, quantize them, certify Jetson hardware,
or promise a latency, memory budget or frame rate. Those are separate target
integration and measurement tasks.
