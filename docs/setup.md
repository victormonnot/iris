# Set up Iris

[Documentation](README.md)

Iris runs on **Linux or WSL**, with **Python 3.12 or 3.13** and
[uv](https://docs.astral.sh/uv/getting-started/installation/).
The server uses Linux process handling and workspace locks; native Windows and
macOS are not supported by the current implementation.

## Start the application

```sh
git clone https://github.com/victormonnot/iris.git
cd iris
uv sync --locked
uv run iris
```

Open **http://127.0.0.1:8000**. Iris serves both the interface and the API;
there is no frontend build step. Node is only needed for JavaScript tests.
Stop the server with Ctrl+C.

Data import, manual annotation and saved-result inspection work without model
weights, a GPU or an API key. The [first-run guide](first-run.md) uses two included
photos to try the workflow. To use your own data, create a project and session,
then open **Data intake**.

## Choose a workspace

By default, Iris stores its database, copied source media, annotations, datasets,
weights and results in `.iris/` relative to the directory where you launch it.
This directory is excluded from Git. Use an explicit path when starting Iris
from different directories:

```sh
uv run iris --data-dir /absolute/path/to/iris-data
```

`IRIS_DATA_DIR` supplies the default when `--data-dir` is absent. CLI model
commands must use the same workspace as the server:

```sh
uv run iris --data-dir /absolute/path/to/iris-data models list
```

Only one server can open a workspace at a time. Projects share that workspace's
official weights, job queue and backup; see [projects](projects.md). Use
[workspace backup and restore](workspace-backup.md) to preserve or move saved
work. Opening an older workspace applies database migrations.

## Add local detectors

Stop the server before changing its installed dependencies. This installs the
CPU runtime and explicitly downloads the catalogued pretrained detectors:

```sh
uv sync --locked --extra ml
uv run --extra ml iris models download --all
uv run --extra ml iris
```

For a custom workspace, add the same `--data-dir` to both Iris commands. Keep
`--extra ml` on later `uv run` and `uv sync` commands to retain the optional
packages. Model availability checks do not download weights or run inference.
Download receipts preserve source URLs and hashes in the workspace.

Continue with [model comparison](model-comparison.md). For NVIDIA CUDA, use
[compute targets](compute-targets.md); selecting CUDA in the interface does not
install a compatible Torch build. [Tracking](tracking.md), [ONNX export](yolox-onnx.md)
and [multimodal review](multimodal-review.md) have their own optional dependencies.

## Port and remote access

Use a different local port if 8000 is occupied:

```sh
uv run iris --port 8001
```

The server binds to `127.0.0.1`. To use a Linux workstation from another computer,
start Iris on the workstation, then open an SSH tunnel from your own computer:

```sh
ssh -N -L 8000:127.0.0.1:8000 user@workstation
```

Open **http://127.0.0.1:8000** on your computer. Computation and workspace files
stay on the workstation. Iris is a single-user application without account
authentication; its server is designed for local access.

## Interface and configuration

The sidebar selects the project and workspace view. **Project jobs** shows work
in progress and saved job details. **Appearance** supports System, Light and Dark;
the preference stays in the browser. **Skip to workspace** and keyboard tab
navigation are available without a mouse.

Optional provider settings are read from the server's environment. Iris does
not automatically load `.env` files. See the specific provider guide before
configuring one. The built-in API schema is at `/openapi.json` on your server.
