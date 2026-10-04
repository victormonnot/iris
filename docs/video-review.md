# Video passage review

Use **Suggest passages** beside an imported video to ask a vision-language model
which parts could be useful to inspect or annotate. The model receives a sparse
storyboard of timestamped images, **not the continuous video or its audio**.
It can miss short events, small objects and activity between sampled images.

## Workflow

1. Choose a time range, 2–12 samples (default 8), and a local or hosted model.
   Describe what you want to find, such as people, cars, camera changes,
   occlusions or difficult negative examples. A very short range may contain
   fewer frames than the requested sample count.
2. Prepare the storyboard. This decodes local frames and stores JPEG copies with
   a maximum edge of 512 pixels. Inspect every image and its timestamp. This
   step performs no generation and creates no dataset frames.
3. Start the review explicitly. Local processing uses the selected installed
   Ollama vision model. Hosted processing requires its configured account and
   an additional approval for the displayed images, instructions and maximum
   request cost. A prepared storyboard expires after 30 minutes and can be used
   for one request only. Changing settings requires a new preview.
4. Read the saved summary and up to six suggested passages. Each has observed
   boundary samples, an explanation and uncertainty reported by the model.
   This uncertainty is not a calibrated probability. No passages are checked
   automatically; an empty response means the model made no proposal.
5. Choose passages, the image budget per passage, and optional surrounding
   seconds. Keep additional full-range samples if useful (8 by default, 0–32
   allowed). This provides coverage outside the model's choices. Preview the
   combined positions, then explicitly start extraction.
6. Review the resulting images in the gallery. They remain unselected and have
   no automatic annotations. Normal selection, annotation and human validation
   still apply before dataset creation and training.

Extraction preserves observed passage anchors within the chosen budget. With a
single image per passage, it chooses a middle observed sample; with a budget of
at least two, it retains the observed start and end. Remaining positions cover
the expanded passage. Overlapping positions are merged. The combined budget is
at most 332 positions (six passages × 50 plus 32 coverage samples), and exact
pixel duplicates or already extracted positions can reduce the number added.
No claim that the chosen data will improve a detector is made before evaluation.

## Providers and data handling

The provider and model choices are shared with assisted annotation: installed
local Ollama vision models, and the two configured Alibaba Qwen3-VL Instruct
profiles. Downloads and fallbacks are never automatic. Missing local models or
API configuration are reported before queueing a request.

Only the displayed JPEG copies, their sample IDs and approximate timestamps,
the review prompt, and your instructions are sent. Source video files, audio
and source EXIF metadata are not sent. Images are transmitted using their exact
prepared bytes; their checksums are rechecked before the request. The original
video checksum and timing metadata are also bound to the review.

Hosted requests use the existing Frankfurt workspace endpoint and **Global**
deployment scope, which does not guarantee inference remains in the EU. The
approval records the images, settings and conservative list-price ceiling for
one request; reported token usage is saved when provided and is not an invoice.
The ceiling uses the documented maximum input and configured 1,024 output-token
limit. See the provider's [32B model profile](https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-32b-instruct),
[235B model profile](https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-235b-a22b-instruct)
and [multiple-image input documentation](https://www.alibabacloud.com/help/en/model-studio/vision).

## Persistence and limits

The local workspace keeps storyboard images, model identity, prompts, bounded
raw responses, errors and passage proposals across restarts. The local model
digest and Ollama version are verified; hosted providers supply a model ID,
without an immutable weights digest. Model text is treated as a proposal and
never executed. Unknown sample IDs, overlapping passages, malformed JSON or
truncated completions fail validation.

Jobs can be cancelled. Cancellation or a server interruption does not trigger
another model request. An already submitted request may still run at the
provider and incur its charge. A fresh preview and explicit action are required
to retry. Failed reviews never start extraction. Job details distinguish a received
response from an unknown delivery outcome. After a connection loss, the provider
may already have processed the request; IRIS does not resend it automatically.
See [jobs and recovery](job-recovery.md) for retained receipts and partial results.

Timestamps derive from frame index and nominal FPS. As with normal extraction,
some variable-rate recordings or container metadata can produce inaccurate
times or unreadable final positions. Use a shorter range or a constant-rate
copy if needed. Storyboards do not correct those source timing limitations.

The tests use generated videos and identified provider fixtures. A successful
protocol test establishes persistence, consent, parsing and extraction behavior;
it does not establish passage-selection quality on real flight footage.
