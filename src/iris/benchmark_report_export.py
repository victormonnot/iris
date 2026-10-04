"""Standalone, script-free summaries of already frozen benchmark reports."""

import json
import math
from html import escape

EVIDENCE_LABELS = {
    "not_declared": "Evidence origin not declared",
    "simulation": "Simulation — software verification, not real model performance",
    "real_data": "Real data declared by the report author",
}


def export_json(report):
    return json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def _text(value):
    return escape(str(value), quote=True)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _percent(value):
    return f"{value * 100:.1f}%" if _number(value) else "Unavailable"


def _time(value):
    return f"{value / 1000:.3f} s" if _number(value) else "Unmeasured"


def _money(value):
    return f"USD {value:.6f}" if _number(value) else "Unknown"


def _table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f'<th scope="col">{_text(h)}</th>' for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(f"<td>{_text(v) if v is not None else 'Unavailable'}</td>" for v in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _paragraph(value, *, css=""):
    return f'<p class="{css}">{_text(value)}</p>'


def _cost(cost):
    if not cost.get("external"):
        return "Local execution cost not measured"
    if cost.get("usage_cost_usd") is not None:
        return _money(cost["usage_cost_usd"]) + " estimated from recorded usage"
    known = cost.get("known_usage_cost_usd")
    return (
        "Total unknown; known usage subtotal "
        + (_money(known) if known is not None else "unavailable")
        + f"; {cost.get('unknown_outcome_count', 0)} unknown delivery outcomes"
    )


def _trials(config):
    rows = []
    for trial in config["trials"]:
        quality = trial["quality"]
        summary = (quality.get("metrics") or {}).get("summary", {})
        corrections, latency = trial["corrections"], trial["latency"]
        coverage = trial["coverage"]
        rows.append(
            [
                trial["id"],
                trial["status"],
                f"{coverage['ready']}/{coverage['planned']} usable; "
                f"{coverage['failed']} failed; {coverage['missing']} missing",
                f"{summary.get('fp', 'N/A')} extra / {summary.get('fn', 'N/A')} missed",
                summary.get("class_conflicts", "N/A"),
                _percent(summary.get("precision")),
                _percent(summary.get("recall")),
                _percent(summary.get("matched_iou_mean")),
                f"{_time(latency.get('total_ms'))}; "
                f"{latency['measured_count']}/{latency['planned_count']} timed",
                _cost(trial["cost"]),
                f"{corrections['reviewed_count']}/{corrections['planned_count']} reviewed; "
                f"{_time(corrections.get('recorded_review_ms'))}; "
                f"{corrections['fully_timed_count']} fully timed",
            ]
        )
    return _table(
        [
            "Trial",
            "Job status",
            "Image coverage",
            "Box errors",
            "Class conflicts",
            "Precision",
            "Recall",
            "Matched IoU",
            "Processing time",
            "API usage cost estimate",
            "Saved human review",
        ],
        rows,
    )


def _trial_details(trial, frames):
    latency, cost, corrections = trial["latency"], trial["cost"], trial["corrections"]
    body = f"<h4>Trial {_text(trial['id'])}</h4>"
    body += _paragraph(f"Created {trial['created_at']} · {trial['status']}")
    if trial.get("error"):
        body += _paragraph(trial["error"], css="notice")
    if not trial["quality"]["complete"]:
        body += _paragraph(trial["quality"].get("reason") or "Quality measurement incomplete")
    body += _paragraph(latency["includes"])
    if latency.get("model_load_ms") is not None:
        inclusion = (
            "already included in processing time"
            if latency.get("model_load_included")
            else "measured separately from processing time"
        )
        body += _paragraph(f"Model loading: {_time(latency['model_load_ms'])}, {inclusion}.")
    body += _paragraph(latency["note"])
    if cost.get("external"):
        body += _paragraph(
            f"Planning allowance: {_money(cost.get('estimated_ceiling_microusd') / 1e6)} · "
            f"Approved budget: {_money(cost.get('budget_microusd') / 1e6)} · "
            f"Reserved allowance: {_money(cost.get('reserved_microusd') / 1e6)}"
        )
    body += _paragraph(cost["note"])
    changes = corrections["changes"]
    body += _paragraph(
        "Latest completed reviews: "
        + " · ".join(
            f"{changes.get(key, 0)} {key}" for key in ("accepted", "corrected", "rejected", "added")
        )
        + ". "
        + corrections["note"]
    )
    rows = []
    for frame in trial["frames"]:
        quality = frame.get("quality") or {}
        correction = frame.get("correction")
        review = (
            f"Revision {correction['revision']} · {correction['status']} · "
            f"{correction['reviewer']} · {_time(correction['timing'].get('elapsed_ms'))}"
            if correction
            else "Unreviewed"
        )
        rows.append(
            [
                frames[frame["frame_id"]]["source_filename"],
                frame["state"],
                frame.get("error") or "",
                quality.get("fp", "N/A"),
                quality.get("fn", "N/A"),
                quality.get("class_conflicts", "N/A"),
                _time(frame.get("elapsed_ms")),
                review,
            ]
        )
    body += _table(
        [
            "Image",
            "State",
            "Error",
            "Extra",
            "Missed",
            "Class conflicts",
            "Time",
            "Saved correction",
        ],
        rows,
    )
    body += "<details><summary>Recorded model and runtime identity</summary><pre>"
    body += _text(json.dumps(trial["identity"], ensure_ascii=False, indent=2)) + "</pre></details>"
    return body


def export_html(report):
    """Export saved values only; no files, network, model or current measurements."""
    snapshot = report["snapshot"]
    comparison = snapshot["comparison"]
    benchmark, reference = comparison["benchmark"], comparison["reference"]
    frames = {frame["frame_id"]: frame for frame in reference["frames"]}
    body = '<p class="eyebrow">IRIS · Preannotation comparison</p>'
    body += f"<h1>{_text(snapshot['title'])}</h1>"
    body += _paragraph(EVIDENCE_LABELS[snapshot["evidence_kind"]], css="notice")
    body += _paragraph(
        f"{benchmark['name']} · {comparison['role']} · {reference['image_count']} images · "
        f"{reference['scene_count']} scenes · {reference['object_count']} reference objects"
    )
    body += _paragraph(f"Report saved {report['created_at']}")
    if snapshot["objective"]:
        body += "<h2>Objective</h2>" + _paragraph(snapshot["objective"], css="prose")
    body += "<h2>Conditions and limits</h2>"
    body += "<ul>" + "".join(f"<li>{_text(w)}</li>" for w in comparison["warnings"]) + "</ul>"
    body += _paragraph(
        "All recorded trials for this role are included as of the report. "
        "Descriptive repeat ranges are not uncertainty estimates or independent datasets. "
        "Class conflicts are already counted among extra and missed boxes. "
        "No overall winner or comparable confidence score is inferred."
    )
    for config in comparison["configs"]:
        body += f"<section><h2>{_text(config['name'])}</h2>"
        body += _paragraph(f"{config['approach']} · {config['model_id']}")
        body += _paragraph(f"Frozen configuration: {config['fingerprint']}", css="checksum")
        repeatability = config["repeatability"]
        body += _paragraph(
            f"{repeatability['complete_count']}/{repeatability['trial_count']} complete trials · "
            f"{repeatability['incomplete_count']} excluded from quality ranges."
        )
        body += _paragraph(repeatability["note"])
        if repeatability["measured"]:
            body += _paragraph(
                f"Distinct ordered geometry results: {repeatability['distinct_geometry_count']}. "
                f"Runtime identities differ: {repeatability['mixed_runtime_identity']}. "
                f"Returned model identities differ: {repeatability['mixed_returned_models']}."
            )
            body += _table(
                ["Metric", "Measured repetitions", "Minimum", "Mean", "Maximum"],
                [
                    [name, values["count"], values["min"], values["mean"], values["max"]]
                    for name, values in repeatability["metrics"].items()
                ],
            )
        body += _trials(config) if config["trials"] else _paragraph("No trial for this role")
        for trial in config["trials"]:
            body += _trial_details(trial, frames)
        body += "</section>"
    if snapshot["conclusion"]:
        body += "<h2>Author interpretation</h2>" + _paragraph(snapshot["conclusion"], css="prose")
    body += "<h2>Traceability</h2>"
    body += _paragraph(f"Report {report['id']} · protocol {snapshot['protocol']}")
    body += _paragraph(f"Snapshot SHA-256: {report['snapshot_sha256']}", css="checksum")
    body += _paragraph(
        f"Reference manifest SHA-256: {benchmark['manifest_sha256']}", css="checksum"
    )
    body += _paragraph(
        "This standalone summary retains recorded values. Reopen the report in IRIS for "
        "the image comparison, or download its JSON for frozen configurations, boxes and revisions."
    )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
        f"<title>{_text(snapshot['title'])} · IRIS</title><style>"
        "*{box-sizing:border-box}body{margin:0;background:#f5f7f3;color:#183b3e;"
        "font:16px/1.55 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:40px 24px}"
        "h1{font-size:2.3rem;line-height:1.2}h2{margin-top:2rem}h4{margin-bottom:.5rem}"
        "section{border-top:1px solid #cad8d2;margin-top:2rem;padding-top:1rem}"
        ".eyebrow{color:#426e65}.notice{padding:14px;background:#e5edde;"
        "border-left:4px solid #54765c}"
        ".prose{white-space:pre-wrap}.checksum,td,pre{overflow-wrap:anywhere}"
        ".checksum{font-size:.85rem}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;"
        "background:white;font-size:.85rem}th,td{padding:10px;text-align:left;vertical-align:top;"
        "border-bottom:1px solid #dae4dd}th{background:#eaf0e9}pre{white-space:pre-wrap;"
        "font-size:.8rem;background:white;padding:12px}details{margin:12px 0}"
        "@media print{body{background:white}main{padding:0}.scroll{overflow:visible}"
        "table{font-size:8pt}th,td{padding:4px}section{break-before:auto}}"
        "</style></head><body><main>" + body + "</main></body></html>"
    )
    return document.encode("utf-8")
