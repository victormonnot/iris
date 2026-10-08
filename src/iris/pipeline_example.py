"""Minimal timestamped integration. Run from a trusted extracted bundle.

python -B example.py /absolute/path/to/image.jpg
The calling application supplies the sequence clock and explicit selection events.
"""

import sys
from pathlib import Path

from iris_bundle.pipeline_runtime import Pipeline
from PIL import Image, ImageOps


def main():
    directory = Path(__file__).resolve().parent
    pipeline = Pipeline(directory)
    pipeline.reset("example-session", clock_kind="provided")
    with Image.open(sys.argv[1]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    result = pipeline.update(
        image,
        frame_id="frame-0",
        frame_index=0,
        timestamp_seconds=0.0,
        # Pass select_detection_index only after an explicit application choice.
    )
    print(result)
    # Repeat update() in source order; use null timestamps with clock_kind="unknown".
    # After any failure, begin a new explicitly reset sequence. Maximum 10000 updates.
    # Send only result["tracking"]["observations"] to measured-object consumers.
    # Predictions and lost/ambiguous/recovering selection states are not measurements.


if __name__ == "__main__":
    main()
