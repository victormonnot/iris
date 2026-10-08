"""Bounded native-tracker replays, with no detector inference or automatic apply."""

import json
import time
from copy import deepcopy

from iris.tracking_replay import TrackingReplayCancelled, replay_detection_cache
from iris.tracking_study_contracts import (
    MAX_REPORT_BYTES,
    REPORT_SCHEMA,
    budget_for,
    source_bindings,
    summarize,
    validate_report,
)


class TrackingStudyCancelled(RuntimeError):
    """Explicit cancellation; no completed study can be published."""


class TrackingStudyBudgetExceeded(ValueError):
    """A cooperative wall deadline expired; no partial success is published."""


def run_study(store, bundle, progress=None, cancelled=None):
    request = bundle["request"]
    budget = budget_for(request, bundle["sources"])
    started = time.monotonic()

    def stop():
        if cancelled is not None and cancelled():
            raise TrackingStudyCancelled(
                "Tracking profile study cancelled; no complete report published"
            )
        if time.monotonic() - started >= request["max_seconds"]:
            raise TrackingStudyBudgetExceeded(
                "Tracking profile study reached its explicit time budget; "
                "no partial report published"
            )
        return False

    stop()
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "request": deepcopy(request),
        "fingerprint": bundle["fingerprint"],
        "dataset": {
            "id": bundle["dataset"]["id"],
            "manifest_sha256": bundle["dataset"]["manifest_sha256"],
            "reserved_test_entries": [
                deepcopy(entry)
                for entry in bundle["dataset"]["manifest"]["entries"]
                if entry["split"] == "test"
            ],
        },
        "sources": source_bindings(bundle),
        "runs": [],
        "summary": None,
    }
    profiles = [request["baseline"], *request["candidates"]]
    completed_updates = 0
    try:
        for source in bundle["sources"]:
            stop()
            run = {
                "sequence_id": source["sequence"]["id"],
                "split": source["entry"]["split"],
                "replays": [],
            }
            report["runs"].append(run)
            source_updates = len(source["sequence"]["manifest"]["frames"]) * request["repeats"]
            for named in profiles:
                stop()

                def update(
                    fraction,
                    message,
                    completed=completed_updates,
                    count=source_updates,
                    name=named["name"],
                ):
                    stop()
                    if progress is not None:
                        progress(
                            (completed + fraction * count) / budget["required_updates"] * 0.9,
                            f"{name}: {message}",
                        )

                replay = replay_detection_cache(
                    store,
                    source["cache"]["id"],
                    profile=named["profile"],
                    repeats=request["repeats"],
                    cancelled=stop,
                    progress=update,
                )
                stop()
                run["replays"].append(replay)
                completed_updates += source_updates
                if len(json.dumps(report, allow_nan=False).encode("utf-8")) > MAX_REPORT_BYTES:
                    raise ValueError("Tracking study report exceeded its portable size budget")
        stop()
        if progress is not None:
            progress(
                0.92, "Evaluating frozen human references; repeated replays do not multiply labels"
            )
        report["summary"] = summarize(bundle, report["runs"], checkpoint=stop)
        stop()
        validate_report(bundle, report, checkpoint=stop)
        stop()
        if progress is not None:
            progress(1.0, "Complete profile study; baseline retained and no profile applied")
        stop()
        return report
    except TrackingReplayCancelled as exc:
        stop()
        raise TrackingStudyCancelled("Tracking profile study cancelled") from exc
