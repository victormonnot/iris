import argparse
import os
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the local IRIS workbench")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("IRIS_DATA_DIR", ".iris")))
    args = parser.parse_args()
    os.environ["IRIS_DATA_DIR"] = str(args.data_dir.resolve())
    uvicorn.run("iris.app:create_app", factory=True, host="127.0.0.1", port=args.port)
