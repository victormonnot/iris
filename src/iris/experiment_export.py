"""Portable, bounded HTML reports rendered only from verified experiment snapshots."""

from __future__ import annotations

import base64
import math
from html import escape

from iris import experiments
from iris.store import Store

MAX_HTML_BYTES = 16 * 1024 * 1024
CSP = (
    "default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
    "base-uri 'none'; form-action 'none'; object-src 'none'"
)
STYLES = """
:root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont,
  "Segoe UI", sans-serif; color: #183b2c; background: #f5f2e9; }
* { box-sizing: border-box; }
body { margin: 0; padding: 48px 24px; line-height: 1.55; }
main { max-width: 1080px; margin: auto; }
header { border-top: 5px solid #214e39; padding-top: 26px; margin-bottom: 36px; }
.eyebrow { font-size: 12px; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
h1 { font-size: clamp(28px, 4vw, 46px); line-height: 1.16; margin: 14px 0 18px;
  letter-spacing: -.035em; overflow-wrap: anywhere; }
h2 { font-size: 23px; letter-spacing: -.02em; margin: 0 0 15px; }
h3 { font-size: 17px; margin: 0 0 10px; overflow-wrap: anywhere; }
p { margin: 10px 0; overflow-wrap: anywhere; }
.muted, .note { color: #627268; font-size: 13px; }
.tag { display: inline-block; border: 1px solid #c9d4c8; border-radius: 20px;
  padding: 4px 11px; font-size: 12px; margin-right: 5px; background: #eef2e8; }
section { margin: 0 0 30px; padding: 25px; background: #fffef9; border: 1px solid #dce1d5;
  border-radius: 12px; }
.columns { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 18px; }
.card { border: 1px solid #dce1d5; border-radius: 8px; padding: 17px; min-width: 0; }
.prose { white-space: pre-wrap; }
.empty { color: #77847b; font-style: italic; }
summary { cursor: pointer; font-weight: 650; padding: 8px 0; }
.data-list { margin: 0; font-size: 13px; }
.data-list dt { color: #627268; margin-top: 10px; }
.data-list dd { margin: 2px 0 8px; overflow-wrap: anywhere; }
.mono, code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 11px; overflow-wrap: anywhere; }
.table-wrap { overflow-x: auto; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; min-width: 640px; font-size: 13px; }
caption { text-align: left; color: #627268; margin-bottom: 8px; }
th, td { border-bottom: 1px solid #e2e5dc; padding: 10px 12px; text-align: right;
  vertical-align: top; overflow-wrap: anywhere; }
th { background: #edf2e8; font-weight: 650; }
th:first-child, td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: 0; }
.metric-results td:not(:first-child), .error-results td:not(:first-child),
.class-results td:nth-child(n+3), .class-results td:first-child { white-space: nowrap; }
.class-results td:nth-child(2) { min-width: 160px; max-width: 230px; overflow-wrap: normal; }
.class-results th { overflow-wrap: normal; }
.notice { padding: 13px 16px; border-left: 3px solid #b58638; background: #faf3e3;
  border-radius: 0 5px 5px 0; font-size: 13px; }
.facts { display: flex; gap: 10px; flex-wrap: wrap; }
.facts p { background: #edf2e8; border-radius: 7px; padding: 10px 15px; margin: 0; }
.facts strong { font-size: 23px; display: block; }
.example { padding: 20px 0; border-top: 1px solid #dce1d5; break-inside: avoid; }
.example:first-of-type { border-top: 0; }
.visual { background: #162d23; border-radius: 6px; overflow: hidden; }
svg { width: 100%; height: auto; display: block; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 12px; margin: 14px 0; }
.legend span::before { content: ""; display: inline-block; width: 21px; margin-right: 6px;
  vertical-align: middle; border-top: 2px solid; }
.legend .ground::before { border-top-style: dashed; }
.ground { color: #2879a8; } .missed { color: #b74243; }
.correct { color: #18744e; } .false-positive { color: #ad661b; }
ul { padding-left: 20px; }
li { margin: 6px 0; overflow-wrap: anywhere; }
footer { color: #627268; font-size: 12px; padding: 4px 0 20px; }
@media(max-width: 680px) { body { padding: 24px 12px; } section { padding: 18px; }
  .columns { grid-template-columns: 1fr; } th, td { padding: 8px; } }
@page { size: A4; margin: 14mm; }
@media print { :root, body { background: white; } body { padding: 0; font-size: 10pt; }
  main { max-width: none; } h1 { font-size: 26pt; } h2 { font-size: 16pt; }
  section { padding: 14px 0; border: 0; border-top: 1px solid #dce1d5; border-radius: 0; }
  header { margin-bottom: 18px; } h2,h3 { break-after: avoid; }
  .card, .facts, .notice, tr { break-inside: avoid; }
  .table-wrap { overflow: visible; } table { font-size: 9pt; min-width: 0; }
  th, td { padding: 7px 6px; }
  .class-results td:nth-child(2) { min-width: 0; max-width: 180px; }
  thead { display: table-header-group; } svg { max-height: 85mm; }
  details > summary { display: none; } details::details-content { content-visibility: visible; }
  details > :not(summary) { display: block !important; }
  * { print-color-adjust: exact; -webkit-print-color-adjust: exact; } }
"""


class ExperimentExportLimitError(ValueError):
    """A report exceeds the deliberately bounded HTML download size."""


ExportLimitError = ExperimentExportLimitError


class _Document:
    def __init__(self):
        self.parts = []
        self.size = 0

    def add(self, text):
        raw = text.encode("utf-8")
        self.size += len(raw)
        if self.size > MAX_HTML_BYTES:
            raise ExportLimitError("Experiment HTML exceeds the 16 MiB export limit")
        self.parts.append(raw)

    def finish(self):
        return b"".join(self.parts)


def _text(value):
    return "N/A" if value is None else str(value)


def _e(value):
    return escape(_text(value), quote=True)


def _number(value, *, minimum=None, maximum=None):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError("The saved report contains an invalid numeric value")
    return value


def _rate(value):
    return "N/A" if value is None else f"{_number(value, minimum=0, maximum=1) * 100:.1f}%"


def _count(value):
    if value is None:
        return "N/A"
    if type(value) is not int or value < 0:
        raise ValueError("The saved report contains an invalid count")
    return str(value)


def _milliseconds(value):
    return "N/A" if value is None else f"{_number(value, minimum=0):.2f} ms"


def _delta(first, second, kind):
    if first is None or second is None:
        return "N/A"
    _number(first)
    _number(second)
    difference = second - first
    if kind == "rate":
        return f"{difference * 100:+.1f} pp"
    if kind == "time":
        relative = f"{difference / first * 100:+.1f}%" if first > 0 else "relative N/A"
        return f"{difference:+.2f} ms ({relative})"
    return f"{difference:+d}"


def _lane_name(lane):
    variant = {"full": "Full image", "tiled": "Tiled image"}.get(lane["variant"])
    if variant is None:
        raise ValueError("The saved report contains an unsupported inference variant")
    # Snapshot names already include the saved inference variant.
    return lane["name"]


def _table(headers, rows, *, caption=None, kind=""):
    heading = "".join(f'<th scope="col">{_e(item)}</th>' for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_e(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    label = f"<caption>{_e(caption)}</caption>" if caption else ""
    return (
        '<div class="table-wrap"><table class="'
        + _e(kind)
        + '">'
        + label
        + "<thead><tr>"
        + heading
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _details(values):
    return (
        '<dl class="data-list">'
        + "".join(f"<dt>{_e(label)}</dt><dd>{_e(value)}</dd>" for label, value in values)
        + "</dl>"
    )


def _metrics(snapshot):
    lanes = snapshot["lanes"]
    paired = len(lanes) == 2
    headings = ["Metric", *[_lane_name(lane) for lane in lanes]]
    if paired:
        headings.append("Candidate − baseline")
    rows = []
    metrics = [
        ("COCO mAP · IoU 0.50–0.95", "map", "rate"),
        ("AP50", "map50", "rate"),
        ("AP75", "map75", "rate"),
        ("Precision", "precision", "rate"),
        ("Recall", "recall", "rate"),
        ("True positives", "tp", "count"),
        ("False positives", "fp", "count"),
        ("Missed objects", "fn", "count"),
        ("Mean total processing time", "mean_total_ms", "time"),
    ]
    if snapshot.get("version") == 2:
        metrics.insert(5, ("F1", "f1", "rate"))
    for label, key, kind in metrics:
        values = [
            (lane["timing"] if kind == "time" else lane["metrics"]["summary"]).get(key)
            for lane in lanes
        ]
        formatter = {"rate": _rate, "count": _count, "time": _milliseconds}[kind]
        row = [label, *[formatter(value) for value in values]]
        if paired:
            comparable = snapshot.get("insights", {}).get("timing", {}).get("comparable")
            row.append(
                "Not comparable"
                if kind == "time" and comparable is False
                else _delta(*values, kind)
            )
        rows.append(row)
    return _table(
        headings,
        rows,
        caption="Saved results from the same frozen evaluation",
        kind="metric-results",
    )


def _scene_results(document, snapshot):
    insights = snapshot.get("insights")
    if insights is None:
        return
    document.add("<section><h2>Results by scene</h2>")
    document.add(
        '<p class="note">Counts use the saved confidence and IoU operating point. '
        "These scene counts are not scene AP scores. Negative images contain no reviewed "
        "objects from the evaluated classes; their false positives still count.</p>"
    )
    names = _class_names(snapshot)
    aggregate = snapshot["error_analysis"].get("aggregate_filter", "all")
    for scene in insights["scenes"]:
        document.add(
            "<h3>"
            + _e(scene["scene_group"])
            + '</h3><p class="note">Evaluated images: '
            + _count(scene["frame_count"])
            + " · Negative images: "
            + _count(scene["negative_frame_count"])
            + "</p>"
        )
        rows, changes = [], []
        for label, counts in scene["counts"].items():
            name = "All classes" if label == aggregate else names.get(label, label)
            for lane in snapshot["lanes"]:
                run = counts["runs"][lane["id"]]
                rows.append(
                    [
                        name,
                        _lane_name(lane),
                        _count(counts.get("ground_truth_count")),
                        *[_count(run.get(key)) for key in ("tp", "fp", "fn", "error_frames")],
                    ]
                )
            delta = counts.get("changes")
            if delta is not None and len(snapshot["lanes"]) == 2:
                if type(delta["fp_delta"]) is not int:
                    raise ValueError("The saved scene false-positive change must be an integer")
                changes.append(
                    [
                        name,
                        _count(delta["recovered"]),
                        _count(delta["new_misses"]),
                        f"{delta['fp_delta']:+d}",
                    ]
                )
        document.add(
            _table(
                ["Class", "Run", "Labeled objects", "TP", "FP", "Missed", "Error images"],
                rows,
            )
        )
        if changes:
            document.add(
                _table(["Class", "Recovered by candidate", "Newly missed", "FP change"], changes)
            )
    if not insights["scenes"]:
        document.add('<p class="empty">No scene counts were recorded.</p>')
    document.add("</section>")


def _sampling(document, snapshot):
    insights = snapshot.get("insights")
    if insights is None:
        return
    sampling = insights["sampling"]
    document.add("<section><h2>Image and video sampling</h2>")
    document.add(
        '<p class="notice">This evaluation covers saved images and sampled video frames. '
        "It does not establish continuous video coverage, live throughput, tracking quality "
        "or performance on intervening frames.</p>"
    )
    document.add(
        _details(
            [
                ("Still images", _count(sampling["still_image_count"])),
                ("Images with unknown source type", _count(sampling["unknown_source_count"])),
            ]
        )
    )
    rows = []
    for source in sampling["video_sources"]:
        times = []
        for key in ("first_timestamp_seconds", "last_timestamp_seconds"):
            value = source.get(key)
            times.append("N/A" if value is None else f"{_number(value, minimum=0):.3f} s")
        rows.append(
            [
                source["filename"],
                source["source_id"],
                _count(source["frame_count"]),
                _count(source["timestamps_available"]),
                *times,
            ]
        )
    if rows:
        document.add(
            _table(
                ["Video", "Source ID", "Sampled frames", "Timed frames", "First", "Last"],
                rows,
                caption="Source timestamps are approximate, not a continuous measured duration.",
            )
        )
    else:
        document.add('<p class="empty">No video sources were recorded.</p>')
    if sampling.get("warning"):
        document.add('<p class="note">' + _e(sampling["warning"]) + "</p>")
    document.add("</section>")


def _timing_context(document, snapshot):
    timing = snapshot.get("insights", {}).get("timing")
    if timing is None:
        return
    if timing["comparable"] is False:
        document.add(
            '<p class="notice">Local timing differences are not compared because the saved '
            "execution contexts are not comparable.</p>"
        )
    if timing.get("reasons"):
        document.add("<ul>")
        for reason in timing["reasons"]:
            document.add("<li>" + _e(reason) + "</li>")
        document.add("</ul>")
    document.add('<p class="note">' + _e(timing["scope"]) + "</p>")


_CUDA_DETAIL_FIELDS = (
    ("GPU", "name"),
    ("CUDA runtime", "runtime"),
    ("cuDNN", "cudnn"),
    ("GPU index", "index"),
    ("Compute capability", "capability"),
    ("GPU memory (bytes)", "total_memory"),
    ("TF32 matrix multiplication", "tf32_matmul"),
    ("TF32 cuDNN", "tf32_cudnn"),
    ("cuDNN benchmarking", "cudnn_benchmark"),
)


def _deployment_environment(measurement):
    profile, environment = measurement["profile"], measurement["environment"]
    values = [("Profile", profile.get("id")), ("Architecture", profile.get("architecture"))]
    values.extend(
        (label, profile.get(key))
        for label, key in (
            ("Target device family", "device"),
            ("Precision", "precision"),
            ("Batch size", "batch_size"),
        )
    )
    values.extend(
        (label, environment.get(key))
        for label, key in (
            ("Measured device", "device"),
            ("Processor", "processor"),
            ("Platform", "platform"),
            ("Machine", "machine"),
            ("Python", "python"),
            ("PyTorch", "torch"),
            ("Torchvision", "torchvision"),
            ("Pillow", "pillow"),
            ("CPU threads", "threads"),
            ("Interop threads", "interop_threads"),
        )
    )
    cuda = environment.get("cuda")
    if cuda is not None:
        values.extend((label, cuda.get(key)) for label, key in _CUDA_DETAIL_FIELDS)
    return _details(values)


def _deployments(document, snapshot):
    deployments = snapshot.get("deployments")
    if deployments is None:
        return
    document.add("<section><h2>Declared target measurements</h2>")
    document.add(
        '<p class="notice">These imported measurements are producer declarations. '
        "Checksums and consistency checks do not independently prove execution or authenticity. "
        "Their timings are separate from IRIS evaluation timings; no cross-context speedup "
        "or automatic winner is inferred.</p>"
    )
    measurements = deployments["measurements"]
    if not measurements:
        document.add('<p class="empty">No target measurement was selected for this report.</p>')
    for measurement in measurements:
        summary = measurement["summary"]
        source = measurement["source"]
        if type(summary["parity_passed"]) is not bool:
            raise ValueError("The saved target parity result must be a boolean")
        document.add('<article class="example"><h3>' + _e(measurement["name"]) + "</h3>")
        declaration = measurement["declaration"]
        document.add(
            '<p class="notice">'
            + (
                "SIMULATION — synthetic fixture; this is not measured hardware performance."
                if declaration == "simulation"
                else "Producer declaration: " + _e(declaration)
            )
            + "</p>"
        )
        document.add(
            '<p class="notice">Exact parity: '
            + ("PASSED" if summary["parity_passed"] else "FAILED")
            + " · Mismatched samples: "
            + _count(len(summary["mismatched_samples"]))
            + ". Execution has not been independently verified.</p>"
        )
        document.add(
            _details(
                [
                    ("Measurement ID", measurement["id"]),
                    ("Export ID", measurement["export_id"]),
                    ("Checkpoint ID", measurement["model_id"]),
                    ("Report run ID", measurement["lane_id"]),
                    ("Imported", measurement["created_at"]),
                    ("Measurement payload SHA-256", measurement["fingerprint"]),
                    ("Export archive SHA-256", measurement["archive_sha256"]),
                    ("Checkpoint SHA-256", measurement["model_sha256"]),
                    ("Reference evaluation ID", source["evaluation_id"]),
                    ("Reference evaluation run ID", source["evaluation_model_id"]),
                    ("Reference dataset SHA-256", source["dataset_manifest_sha256"]),
                    ("Reference device", source["reference_device"]),
                    ("Reference frame IDs", ", ".join(source["frame_ids"])),
                    ("Measured frames", _count(summary["frames"])),
                    ("Repeats", _count(summary["repeats"])),
                    ("Measured samples", _count(summary["sample_count"])),
                ]
            )
        )
        document.add("<details open><summary>Declared target environment</summary>")
        document.add(_deployment_environment(measurement))
        document.add("</details>")
        rows = []
        for label, key in (
            ("Preprocessing", "preprocess_ms"),
            ("Model forward", "inference_ms"),
            ("Postprocessing", "postprocess_ms"),
            ("Total processing", "total_ms"),
        ):
            distribution = summary["timing_ms"][key]
            rows.append(
                [label, *[_milliseconds(distribution[k]) for k in ("min", "median", "max")]]
            )
        document.add(
            _table(["Target stage", "Minimum", "Median", "Maximum"], rows, kind="metric-results")
        )
        decode = summary["decode_ms"]
        document.add(
            _table(
                ["Separate cost", "Minimum", "Median", "Maximum"],
                [["Image decoding", *[_milliseconds(decode[k]) for k in ("min", "median", "max")]]],
            )
        )
        document.add(
            _details(
                [
                    ("Model loading (excluded from processing)", _milliseconds(summary["load_ms"])),
                    ("Warmup (excluded from processing)", _milliseconds(summary["warmup_ms"])),
                ]
            )
        )
        timing = measurement["profile"].get("timing", {})
        if isinstance(timing, dict):
            document.add(
                _details(
                    [
                        ("Target timing definition", timing.get("total_ms")),
                        ("Image decoding definition", timing.get("decode_ms")),
                        ("Model loading definition", timing.get("load_ms")),
                        ("Warmup passes", timing.get("warmup_passes")),
                        ("Target synchronization", timing.get("cuda_synchronization")),
                    ]
                )
            )
        document.add("</article>")
    if deployments.get("limitations"):
        document.add("<ul>")
        for limitation in deployments["limitations"]:
            document.add("<li>" + _e(limitation) + "</li>")
        document.add("</ul>")
    document.add("</section>")


def _class_names(snapshot):
    taxonomy = snapshot["evaluation"]["config"].get("taxonomy")
    if taxonomy:
        return {item["id"]: item["name"] for item in taxonomy["classes"]}
    return {"person": "person", "car": "car"}


def _class_metrics(snapshot):
    rows = []
    for label, name in _class_names(snapshot).items():
        for lane in snapshot["lanes"]:
            item = next(
                (item for item in lane["metrics"]["per_class"] if item["label"] == label), None
            )
            if item is None:
                raise ValueError("A saved report is missing a class result")
            rows.append(
                [
                    name if name == label else f"{name} ({label})",
                    _lane_name(lane),
                    _count(item.get("support")),
                    _rate(item.get("ap")),
                    _rate(item.get("ap50")),
                    _rate(item.get("precision")),
                    _rate(item.get("recall")),
                    _count(item.get("fp")),
                    _count(item.get("fn")),
                ]
            )
    return _table(
        ["Class", "Run", "Labeled objects", "AP", "AP50", "Precision", "Recall", "FP", "Missed"],
        rows,
        kind="class-results",
    )


def _error_summary(snapshot):
    analysis = snapshot["error_analysis"]
    if analysis.get("comparison") is None:
        return '<p class="note">One run: recovered objects and new misses are not compared.</p>'
    rows = []
    names = _class_names(snapshot)
    aggregate = analysis.get("aggregate_filter", "all")
    for label in analysis["summary"]:
        changes = analysis["summary"][label]["changes"]
        delta = changes["fp_delta"]
        if type(delta) is not int:
            raise ValueError("The saved false-positive change must be an integer")
        rows.append(
            [
                "All classes" if label == aggregate else names.get(label, label),
                _count(changes["recovered"]),
                _count(changes["new_misses"]),
                f"{delta:+d}",
            ]
        )
    return _table(
        ["Objects", "Recovered by candidate", "Newly missed by candidate", "False-positive change"],
        rows,
        kind="error-results",
    )


def _flow(snapshot):
    dataset = snapshot["dataset"]
    evaluation = snapshot["evaluation"]
    output = [
        '<div class="card"><h3>1 · Frozen dataset</h3>',
        _details(
            [
                ("Dataset", dataset["name"]),
                ("Dataset ID", dataset["id"]),
                ("Manifest SHA-256", dataset["manifest_sha256"]),
                ("Dataset images", _count(dataset.get("summary", {}).get("frame_count"))),
                ("Evaluated scene groups", len(dataset["source_groups"])),
                (
                    "Evaluation split",
                    "Final test audit" if evaluation["split"] == "test" else "Validation",
                ),
            ]
        ),
        "</div>",
    ]
    for number, lane in enumerate(snapshot["lanes"], 2):
        output.append(f'<div class="card"><h3>{number} · {_e(_lane_name(lane))}</h3>')
        values = [
            ("Checkpoint ID", lane["model_id"]),
            ("Checkpoint SHA-256", lane["weight_sha256"]),
            ("Evaluation run ID", lane["id"]),
        ]
        training = lane.get("training")
        if training is None:
            values.append(
                (
                    "Training history",
                    "Pretrained checkpoint"
                    if lane.get("training_status") == "pretrained"
                    else "Not available in this saved report",
                )
            )
        else:
            config = training["config"]
            history = training["history_summary"]
            values.extend(
                [
                    ("Training run", training["name"]),
                    ("Training run ID", training["id"]),
                    ("Training dataset", training["dataset_name"]),
                    ("Training dataset ID", training["dataset_id"]),
                    ("Training manifest SHA-256", training["dataset_manifest_sha256"]),
                    ("Parent checkpoint", training["parent_model_id"]),
                    ("Parent checkpoint SHA-256", training["parent_weight_sha256"]),
                    ("Training depth", config.get("scope", "prediction_head_only")),
                    ("Completed optimizer steps", _count(history["steps_completed"])),
                    ("Learning rate", config.get("learning_rate")),
                    ("Seed", config.get("seed")),
                ]
            )
        runtime = lane.get("runtime", {})
        values.extend(
            (label, runtime.get(key))
            for label, key in (
                ("Device", "device"),
                ("Hardware", "hardware"),
                ("Platform", "platform"),
                ("Architecture", "architecture"),
                ("Precision", "precision"),
                ("PyTorch", "torch_version"),
                ("Torchvision", "torchvision_version"),
                ("CPU threads", "threads"),
                ("Interop threads", "interop_threads"),
            )
        )
        cuda = runtime.get("cuda")
        if cuda is not None:
            values.extend((label, cuda.get(key)) for label, key in _CUDA_DETAIL_FIELDS)
        output.extend([_details(values), "</div>"])
    return '<div class="columns">' + "".join(output) + "</div>"


def _protocol(snapshot):
    evaluation = snapshot["evaluation"]
    config = evaluation["config"]
    analysis = snapshot["error_analysis"]
    protocol = config.get("protocol", {})
    inference = config.get("inference", {"mode": "full"})
    ious = protocol.get("ap_iou_thresholds")
    if ious is not None and (not isinstance(ious, list) or not ious):
        raise ValueError("Saved AP overlap thresholds are invalid")
    values = [
        ("Evaluation", evaluation["name"]),
        ("Evaluation ID", evaluation["id"]),
        ("Evaluation created", evaluation["created_at"]),
        ("Inference mode", inference.get("mode", "full")),
        ("Confidence threshold for precision/recall", _rate(analysis["confidence_threshold"])),
        ("IoU threshold for precision/recall", _rate(analysis["iou_threshold"])),
        ("Error analysis protocol", analysis["protocol"]),
        ("Metric protocol", protocol.get("id")),
        ("Metric engine", protocol.get("engine")),
        ("Metric engine version", protocol.get("engine_version")),
        ("NumPy version", protocol.get("numpy_version")),
        (
            "COCO maxDets",
            ", ".join(_count(value) for value in protocol.get("max_dets", [])) or None,
        ),
        (
            "AP IoU thresholds",
            ", ".join(f"{_number(value, minimum=0, maximum=1):.2f}" for value in ious)
            if ious
            else None,
        ),
    ]
    if inference.get("mode", "full") != "full":
        tiling = inference.get("tiling", {})
        values.extend(
            [
                ("Tile size", tiling.get("tile_size")),
                ("Tile overlap", _rate(tiling.get("overlap"))),
                ("Merge IoU", _rate(tiling.get("merge_iou"))),
            ]
        )
    output = [
        "<details open><summary>Recorded metric and timing definitions</summary>",
        _details(values),
    ]
    taxonomy = config.get("taxonomy")
    if taxonomy:
        output.append("<h3>Frozen class definitions</h3>")
        output.append(
            _details(
                [
                    (f"{item['name']} ({item['id']})", item["definition"])
                    for item in taxonomy["classes"]
                ]
            )
        )
    for lane in snapshot["lanes"]:
        timing = lane.get("runtime", {}).get("timing_protocol", {})
        output.append("<h3>" + _e(_lane_name(lane)) + " · Timing</h3>")
        output.append(
            _details(
                [
                    ("Timing protocol", timing.get("version", timing.get("id"))),
                    ("Total processing time includes / excludes", timing.get("total_ms")),
                    ("Image decoding", timing.get("decode_ms")),
                    ("Model forward time", timing.get("inference_ms")),
                    ("Warmup", timing.get("warmup_frames", timing.get("warmup"))),
                    ("Warmup iterations", timing.get("warmup_iterations")),
                    ("Warmup included in timings", timing.get("warmup_in_timings")),
                    ("Batch size", timing.get("batch_size")),
                    (
                        "Device synchronization",
                        timing.get("synchronize", timing.get("synchronization")),
                    ),
                ]
            )
        )
    return "".join([*output, "</details>"])


def _credits(snapshot):
    sources = snapshot["dataset"].get("sources", [])
    if not sources:
        return ""
    return (
        "<section><h2>Recorded data attribution</h2>"
        + _table(
            ["Source", "License", "Attribution"],
            [
                [source.get("source_url"), source.get("license_name"), source.get("attribution")]
                for source in sources
            ],
        )
        + "</section>"
    )


def _box_svg(box, label, color, dashed, width, height):
    coordinates = box.get("box")
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
        raise ValueError("A saved example contains invalid box coordinates")
    x1, y1, x2, y2 = [_number(value) for value in coordinates]
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("A saved example box falls outside the original image")
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    # SVG coordinates use source pixels: a fixed minimum would produce huge
    # labels when a small source image is expanded to a report card.
    font_size = width / 32
    gap = font_size / 5
    text_width = min(width - 2 * gap, len(label) * font_size * 0.62)
    text_x = min(max(gap, x1 + gap), width - text_width - gap)
    text_y = max(font_size + gap, y1 - gap)
    return (
        f'<rect x="{x1:g}" y="{y1:g}" width="{x2 - x1:g}" height="{y2 - y1:g}" '
        f'fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"'
        f"{dash}><title>{_e(label)}</title></rect>"
        f'<text x="{text_x:g}" y="{text_y:g}" font-size="{font_size:g}" fill="{color}" '
        f'stroke="#fffef9" stroke-width="{font_size / 9:g}" paint-order="stroke" '
        f'font-weight="700">{_e(label)}</text>'
    )


def _visual(example, lane, image_uri, threshold, class_names=None):
    class_names = class_names or {"person": "person", "car": "car"}
    width, height = example["width"], example["height"]
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("A saved example has invalid original image dimensions")
    errors = lane["errors"]
    missed = set(errors["false_negatives"])
    false_positives = set(errors["false_positives"])
    matched = {match["detection_index"] for match in errors["matches"]}
    result = [
        f'<div class="visual"><svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Saved labels and predictions">',
        f'<image href="{image_uri}" width="{width}" height="{height}" preserveAspectRatio="none"/>',
    ]
    for index, box in enumerate(example["ground_truth"]):
        absent = index in missed
        result.append(
            _box_svg(
                box,
                f"{'Missed' if absent else 'Label'}: {class_names.get(box['label'], box['label'])}",
                "#b74243" if absent else "#2879a8",
                True,
                width,
                height,
            )
        )
    for index, detection in enumerate(lane["detections"]):
        if (
            detection.get("ignored")
            or detection["label"] not in class_names
            or detection["score"] < threshold
        ):
            continue
        score = _rate(detection["score"])
        status, color = (
            ("FP", "#ad661b")
            if index in false_positives
            else (("TP", "#18744e") if index in matched else ("Prediction", "#627268"))
        )
        result.append(
            _box_svg(
                detection,
                f"{status}: {class_names[detection['label']]} {score}",
                color,
                False,
                width,
                height,
            )
        )
    return "".join([*result, "</svg></div>"])


def _examples(document, store, report, include_images):
    snapshot = report["snapshot"]
    examples = snapshot["examples"]
    if not isinstance(examples, list) or len(examples) > 6:
        raise ValueError("A saved experiment supports at most six selected examples")
    document.add("<section><h2>Selected examples</h2>")
    if not examples:
        document.add('<p class="empty">No examples were selected for this report.</p>')
    elif not include_images:
        document.add(
            '<p class="note">Images were excluded from this export. '
            "Only the selected examples' saved counts and identifiers appear below.</p>"
        )
    else:
        document.add(
            '<div class="legend"><span class="ground">Reviewed label · dashed</span>'
            '<span class="ground missed">Missed label · dashed</span>'
            '<span class="correct">True positive · solid</span>'
            '<span class="false-positive">False positive · solid</span></div>'
        )
    lanes = {lane["id"]: lane for lane in snapshot["lanes"]}
    threshold = snapshot["error_analysis"]["confidence_threshold"]
    for example in examples:
        document.add(
            '<article class="example"><h3>'
            + _e(example["source"]["filename"])
            + '</h3><p class="note">Frame '
            + _e(example["frame_id"])
            + " · Scene group "
            + _e(example["scene_group"])
            + "</p>"
        )
        seconds = example["source"].get("timestamp_seconds")
        if seconds is not None:
            document.add(f'<p class="note">Source time: {_number(seconds, minimum=0):.3f} s</p>')
        insights = snapshot.get("insights")
        if insights is not None:
            change = insights["frame_changes"].get(example["frame_id"])
            reason = next(
                (
                    item["reason"]
                    for item in insights["suggested_examples"]
                    if item["frame_id"] == example["frame_id"]
                ),
                None,
            )
            document.add(_details([("Saved frame change", change), ("Example suggestion", reason)]))
        uri = None
        if include_images:
            content = experiments.read_experiment_image(store, report["id"], example["frame_id"])
            # Reject oversized image embedding before allocating a base64 string.
            if (
                document.size + 4 * ((len(content) + 2) // 3) * len(example["lanes"])
                > MAX_HTML_BYTES
            ):
                raise ExportLimitError("Experiment images exceed the 16 MiB export limit")
            uri = "data:image/jpeg;base64," + base64.b64encode(content).decode("ascii")
        document.add('<div class="columns">')
        for lane in example["lanes"]:
            saved = lanes[lane["run_id"]]
            document.add('<div class="card"><h3>' + _e(_lane_name(saved)) + "</h3>")
            if uri is not None:
                document.add(_visual(example, lane, uri, threshold, _class_names(snapshot)))
            errors = lane["errors"]
            document.add(
                '<p class="note">'
                + " · ".join(
                    f"{label}: {_count(errors[key])}"
                    for label, key in (("TP", "tp"), ("FP", "fp"), ("Missed", "fn"))
                )
                + "</p></div>"
            )
        document.add("</div></article>")
    document.add("</section>")


def _references(snapshot):
    history = snapshot["reference_decisions"]
    output = [
        '<p class="note">Historical decisions captured with this report on '
        + _e(history["captured_at"])
        + ". They do not describe the workspace's current reference.</p>"
    ]
    decisions = history["decisions"]
    if not decisions:
        output.append(
            '<p class="empty">No reference decision was captured for this evaluation.</p>'
        )
    for item in decisions:
        output.extend(
            [
                '<article class="card">',
                _details(
                    [
                        ("Model", item.get("model_name", item["model_id"])),
                        ("Variant", item.get("variant", "full")),
                        ("Decision ID", item["id"]),
                        ("Recorded", item["created_at"]),
                        ("Reviewer", item["reviewer"]),
                    ]
                ),
                '<p class="prose">' + _e(item["notes"]) + "</p></article>",
            ]
        )
    return "".join(output)


def render_experiment_html(
    store: Store, report_id: str, *, include_images: bool = False, expected_revision: int
) -> bytes:
    """Export one verified report revision without models, inference or external resources."""
    if type(include_images) is not bool:
        raise ValueError("include_images must be a boolean")
    if type(expected_revision) is not int or expected_revision < 1:
        raise ValueError("Expected report revision must be a positive integer")
    report = experiments.experiment_detail(store, report_id)
    if report["revision"] != expected_revision:
        raise experiments.ExperimentConflict("The report changed; reload it before exporting")
    snapshot = report["snapshot"]
    lanes = snapshot["lanes"]
    if not isinstance(lanes, list) or not 1 <= len(lanes) <= 2:
        raise ValueError("A saved experiment requires one or two evaluated runs")
    evaluation = snapshot["evaluation"]
    if evaluation["split"] not in {"val", "test"}:
        raise ValueError("The report requires a validation evaluation or a final test audit")
    split = "Final test audit" if evaluation["split"] == "test" else "Validation"
    document = _Document()
    document.add(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="'
        + _e(CSP)
        + '"><title>'
        + _e(report["title"])
        + " · IRIS experiment</title><style>"
        + STYLES
        + '</style></head><body><main><header><div class="eyebrow">IRIS / Experiment report</div>'
        + "<h1>"
        + _e(report["title"])
        + '</h1><span class="tag">'
        + split
        + '</span><span class="tag">Revision '
        + str(expected_revision)
        + '</span><p class="muted">Evidence captured '
        + _e(snapshot["captured_at"])
        + " · Report updated "
        + _e(report["updated_at"])
        + "</p></header>"
    )
    document.add('<section><h2>Objective and conclusion</h2><div class="columns">')
    for label, key in (("Objective", "objective"), ("Author's conclusion", "conclusion")):
        value = report[key]
        document.add(f'<div><h3>{label}</h3><p class="prose{" empty" if not value else ""}">')
        document.add(_e(value) if value else "Not recorded.")
        document.add("</p></div>")
    document.add("</div></section>")
    if evaluation["split"] == "test":
        document.add(
            '<p class="notice">Final test audit: reporting evidence, not a basis for choosing '
            "or promoting a model. Preserve test independence.</p>"
        )
    document.add("<section><h2>Measured results</h2>")
    summary = lanes[0]["metrics"]["summary"]
    document.add(
        '<div class="facts"><p><strong>'
        + _count(summary.get("frame_count"))
        + "</strong>Evaluated images</p><p><strong>"
        + _count(summary.get("ground_truth_count"))
        + "</strong>Reviewed objects</p></div>"
    )
    if len(lanes) == 2:
        document.add(
            '<p class="note">The first run is the baseline; the second is the candidate. '
            "Changes are candidate minus baseline. Percentage metrics use percentage points "
            "(pp), not relative percent change. Comparable local time includes its relative "
            "change when the baseline is greater than zero.</p>"
        )
    document.add(_metrics(snapshot))
    _timing_context(document, snapshot)
    document.add(
        '<p class="note">N/A means undefined or unavailable, never a perfect score. '
        "Classes without labeled objects are excluded from macro AP. Precision, recall and "
        "error counts use the saved operating point; AP uses the saved native scores. "
        "Local timings are not a benchmark of the exported model on its target hardware."
        "</p></section>"
    )
    document.add("<section><h2>Results by class</h2>" + _class_metrics(snapshot) + "</section>")
    document.add("<section><h2>Recovered objects and new misses</h2>" + _error_summary(snapshot))
    document.add(
        '<p class="note">Recoveries and new misses refer to the same frozen ground-truth '
        "objects. False-positive change compares counts, not object identities.</p></section>"
    )
    _scene_results(document, snapshot)
    _sampling(document, snapshot)
    document.add(
        "<section><h2>Dataset, training and checkpoints</h2>" + _flow(snapshot) + "</section>"
    )
    document.add("<section><h2>Evaluation protocol</h2>" + _protocol(snapshot) + "</section>")
    _deployments(document, snapshot)
    _examples(document, store, report, include_images)
    document.add(
        "<section><h2>Recorded reference decisions</h2>" + _references(snapshot) + "</section>"
    )
    document.add(_credits(snapshot))
    warnings = list(snapshot["error_analysis"].get("warnings", []))
    for lane in lanes:
        warnings.extend(lane["metrics"].get("warnings", []))
    if warnings:
        document.add("<section><h2>Limits of this evidence</h2><ul>")
        for warning in dict.fromkeys(warnings):
            document.add("<li>" + _e(warning) + "</li>")
        document.add("</ul></section>")
    document.add(
        "<footer><p>Standalone local export · "
        + ("Selected images included" if include_images else "No image pixels included")
        + " · No inference, training or new metric computation was performed.</p>"
        + '<p class="mono">Report '
        + _e(report["id"])
        + " · Snapshot SHA-256 "
        + _e(report["snapshot_sha256"])
        + "</p></footer></main></body></html>"
    )
    latest = experiments.experiment_detail(store, report_id)
    if (
        latest["revision"] != expected_revision
        or latest["snapshot_sha256"] != report["snapshot_sha256"]
    ):
        raise experiments.ExperimentConflict("The report changed during export; reload it")
    return document.finish()
