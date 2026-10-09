# Longer CPU/CUDA training and explicit continuation

[Documentation](README.md)

IRIS supports 1 to 10,000 optimizer steps for Faster R-CNN MobileNetV3-Large
320 FPN, SSDLite320 MobileNetV3-Large, YOLOX-Nano and their compatible trained
descendants.
Training uses CPU or one selected
NVIDIA CUDA device, float32, batch size one, and the selected light, partial or
full depth. The frozen
training split supplies every example; validation and test images remain
reserved for the separate evaluation workflow. Mixed precision, multi-GPU
training, automatic hyperparameter search and automatic model selection are not
provided. Training on a GPU does not restrict the completed model to GPU
inference: see [compute targets and CUDA setup](compute-targets.md).

The software checks use synthetic data, tiny CPU PyTorch modules and mocked CUDA
interfaces to exercise state persistence and continuation. Real 40-step acceptance
trials covered the two Torchvision architectures with light training on CPU and
all three depths on an RTX 4060 using PyTorch 2.10.0 / Torchvision 0.25.0 `cu128`. Completed inference
checkpoints were reloaded and evaluated in separate workers.

For light training on those two architectures and devices, the trials separately
cancelled or forcibly terminated the worker, then explicitly continued from a
saved state on the original device and runtime. The resulting weights matched
the corresponding uninterrupted run in these trials; this does not guarantee
bit-for-bit CUDA reproducibility elsewhere. For those Torchvision trials, longer
runs, partial/full training on CPU, partial/full-scope recovery, server restart
and power-loss recovery remain untested. The pilot does not establish general
quality gains or performance on other machines.

A later [YOLOX acceptance cycle](acceptance-results.md#yolox-nano-custom-detector-accepted-by-an-external-application)
completed 400-step head-only and 800-step partial-backbone CUDA runs. Those runs
do not establish YOLOX interruption or continuation parity. Its
[versioned training recipe](yolox-training.md) defines the loss and gradient
clipping policy; a stopped attempt cannot resume under a different recipe.

## Prepare a run

In **Dataset & training**, choose a frozen dataset and a compatible local parent,
then choose CPU or an available NVIDIA GPU and set the training depth, step count,
learning rate and seed. The device belongs to the IRIS server. CUDA requires a
compatible, explicitly provisioned PyTorch/Torchvision runtime and NVIDIA driver;
an unavailable device is reported instead of silently falling back to CPU.
Changing the device requires another preview. The default
step count remains 20. The interface enables recovery checkpoints with a default
interval of 50 steps. Preview the plan before starting; changing any setting
requires another preview.

The interval must be an integer from 1 to 1,000 and at least
`ceil(total_steps / 200)`. This bounds the plan to at most 200 periodic saves,
including its final step. For example, 10,000 steps require an interval of at
least 50. A checkpoint is saved at each interval and after the last optimizer
step, even when that last step falls between intervals. A cooperative cancellation
also attempts to save the latest completed step. A process killed during a step
can only recover from a state that was already published.

The preview reports training image count, image visits, complete passes, scope
and checkpoint policy. It does not predict wall-clock duration. Hardware, image size,
training depth, state validation and disk writes affect how long the run takes.
No model weights, runtime packages or datasets are downloaded by this workflow.

## What a recovery checkpoint contains

A recovery checkpoint preserves the complete model tensor state, SGD state and
momentum, exact optimizer parameter order and settings, CPU Torch RNG, image
sampler state and remaining shuffled order. It also retains module training
modes and the modules that have received nonzero gradients. Its binding records
the frozen configuration, dataset and parent identities, completed step,
history-prefix hash, recorded active time and runtime identity. CUDA states also
preserve the selected GPU's Torch RNG. Model and optimizer tensors are serialized
on CPU; restoration moves the optimizer state to its corresponding parameters'
device. CPU state files retain the `iris-training-state-v1` protocol; CUDA states
use `iris-training-state-cuda-v1`.

Loading reconstructs the original trainer and verifies the state before applying
it. The original parent-weight and buffer baselines remain available for final
frozen-layer verification. Restoration keeps the optimizer state and restores
the CPU and, for CUDA, selected GPU RNG after model and optimizer loading. Starting another fine-tuning
run from a completed model initializes a new optimizer; use explicit continuation
when the intent is to preserve an interrupted run's optimization state.

Recovery states are internal training artifacts. They do not appear as inference
models and cannot be selected for comparison, evaluation or portable model
export. Successful training still publishes a separate model checkpoint with
its own hash and provenance. That completed model can use the
[standalone model export workflow](model-export.md).

## Resume a stopped attempt

1. Open the run in **Training runs**. An attempt must be failed, cancelled or
   interrupted and have a durable recovery state. A running attempt must stop
   first; restarting the server never resumes it automatically.
2. Preview the continuation. IRIS verifies the saved state bytes, frozen
   dataset and training images, original parent and checkpoint history. The
   preview identifies the saved step, original target and recorded steps that
   will need to be recomputed.
3. Start the continuation explicitly. IRIS creates a new attempt and job,
   preserving the source attempt, logs and history. The source history prefix
   through the durable step is copied into the new attempt. Work recorded after
   that step is recomputed; a history entry by itself is not an optimizer state.
4. Follow the new attempt in run history. The original target step count, data,
   class definitions, depth, learning rate, seed and other optimizer settings
   stay unchanged. A state saved at the target step can finish publication
   without repeating optimization.

Each stopped attempt can have one successor. Repeating the same continuation
request returns that existing successor rather than creating another job. If
the successor also stops, it can have its own explicitly requested successor;
an inherited source state remains available even if that attempt stopped before
writing a newer checkpoint.

The worker requires the saved runtime identity: full Python, Torch and
Torchvision versions, host architecture, thread counts, execution device,
float32 and deterministic-algorithm setting. CUDA continuation additionally binds
the CUDA/cuDNN builds, selected GPU index, model and compute capability, UUID when
available, cuDNN options, TF32 flags and cuBLAS workspace setting. Moving a recovery
state from CPU to GPU, to another GPU index or to another runtime is not supported.
Completed inference weights remain portable across supported CPU/CUDA targets.
Runtime compatibility is checked before further
optimizer work. Changed state bytes, dataset or parent identity, configuration,
sampler or history prevent continuation. A preview does not run the model and
cannot establish that its later execution will succeed.

CPU training keeps deterministic algorithms enabled. CUDA training uses float32
with TF32 and cuDNN benchmarking disabled, but leaves deterministic algorithms
disabled because the detection pipeline includes operations without guaranteed
deterministic CUDA implementations. Restoring RNG and optimizer state does not
promise bit-for-bit equality with uninterrupted GPU training.

## History, timing and storage

Durable runs flush history every ten steps and whenever they publish a recovery
checkpoint. Logs and saved history remain inspectable after cancellation or
failure. An abrupt stop can lose work since the latest saved state, including
work whose loss already appears in the history.

The displayed active time comes from step history. It starts after model loading
and state restoration, includes image reads, optimizer work and prior active
history/state checks and writes, and excludes downtime between attempts. A step
timestamp is captured before saving that step's state, so the last timestamp
does not include its own checkpoint write or final model publication. Resumed
history starts from the durable prefix's recorded time; work discarded after
that prefix is not added to the successor's total. This is an observed progress
measure, not a forward-only benchmark or a promise about the remaining time.

IRIS retains the latest two published recovery states per attempt, each limited
to 512 MiB. Allow space for an additional state being written before an older
one is removed, plus model weights, frozen images and other artifacts. States
belonging to earlier attempts remain retained, and a state referenced by a
successor is preserved. The two-state policy is per attempt, not a total
workspace disk limit. Workspace backup includes registered recovery states and
their lineage; restoring a workspace does not restart training.

Training loss measures optimization on the training examples. It is not AP,
recall or evidence of generalization. Evaluate a completed model on independent,
reviewed validation data before selecting a reference model; reserve the test
split for the separate audit.

## API compatibility

`POST /api/trainings/preview` accepts `checkpoint_interval` with the training
settings and returns a `request_id` and `fingerprint`. Creating a durable run
with `POST /api/trainings` requires that request ID and the fingerprint as
`expected_fingerprint`, with the same inputs. This makes retries idempotent and
rejects changed plans. The optional `device` setting defaults to `cpu`; `cuda`
selects `cuda:0`, and `cuda:<index>` selects a particular visible GPU. Requests above
200 steps and every CUDA request are durable automatically; omitting their
interval selects 50. `GET /api/training/devices` reports available server devices
and the reason CUDA is unavailable when setup is incomplete.

For existing integrations, a CPU request of at most 200 steps that omits
`checkpoint_interval` keeps the legacy short-run behavior. It does not require
the new preview fields and does not save optimizer recovery state. Existing
short runs cannot acquire missing optimizer or RNG state retroactively. The
current interface supplies a checkpoint interval even for short runs.

`POST /api/trainings/{training_id}/resume-preview` prepares a stopped attempt's
continuation without loading a model. Send its `fingerprint` as
`expected_fingerprint` to `POST /api/trainings/{training_id}/resume` to create or
retrieve the successor. The continuation API does not accept replacement
training settings or a different target step count.
