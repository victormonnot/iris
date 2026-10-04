"""Reports capture reference decisions from their own project only."""

import json

import pytest
from test_datasets import add_frame
from test_evaluation_analysis import record_evaluation

from iris.datasets import create_dataset
from iris.experiments import create_experiment, experiment_detail, preview_experiment
from iris.projects import create_project
from iris.store import Store, new_id, now


@pytest.mark.parametrize("promotion_order", [(0, 1), (1, 0)])
def test_report_reference_snapshot_is_project_local_in_both_promotion_orders(
    tmp_path, promotion_order
):
    store = Store(tmp_path / "workspace")
    evaluations = []
    for index in range(2):
        project = create_project(store, name=f"Project {index}")
        frames = []
        for group, color in (("train", 10), ("val", 20)):
            frame = add_frame(store, tmp_path, group=group, color=(color + index, 30, 40))
            store.update("sessions", frame["session_id"], {"project_id": project["id"]})
            frames.append(frame)
        dataset = create_dataset(
            store,
            project_id=project["id"],
            name="Synthetic dataset",
            frame_ids=[frame["id"] for frame in frames],
            splits={"train": "train", "val": "val"},
        )
        evaluations.append(record_evaluation(store, dataset, {"fixture": [[]]}, ["fixture"]))
    references = {}
    for index in promotion_order:
        references[index] = store.insert(
            "model_references",
            {
                "id": new_id(),
                "evaluation_id": evaluations[index]["id"],
                "model_id": "fixture",
                "reviewer": f"Reviewer {index}",
                "notes": f"Project {index} decision",
                "metadata": {},
                "created_at": now(),
            },
        )
    for index, evaluation in enumerate(evaluations):
        preview = preview_experiment(store, evaluation["id"])
        snapshot = preview["snapshot"]["reference_decisions"]
        assert snapshot["current_reference_id"] == references[index]["id"]
        assert [row["id"] for row in snapshot["decisions"]] == [references[index]["id"]]
        report = create_experiment(store, evaluation_id=evaluation["id"], title="Synthetic report")
        assert report["snapshot"]["reference_decisions"] == {
            **snapshot,
            "captured_at": report["snapshot"]["reference_decisions"]["captured_at"],
        }
        assert references[1 - index]["id"] not in json.dumps(report)
        reopened = experiment_detail(Store(store.root), report["id"])
        assert reopened["snapshot"] == report["snapshot"]
