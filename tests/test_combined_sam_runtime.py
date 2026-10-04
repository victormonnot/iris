"""C-only dynamic class phrases without changing B's fixed-prompt contract."""

from copy import deepcopy

import pytest
from test_sam_runtime import CONFIG, FAKE_WORKER, identity
from test_sam_runtime import simulated as simulated
from test_sam_runtime import simulated_model as simulated_model

from iris import sam_runtime as runtime
from iris import sam_runtime_worker as worker


def test_default_worker_keeps_immutable_prompts(simulated_model):
    checkpoint, calls, _ = simulated_model
    model = worker._Runtime(CONFIG, str(checkpoint))
    with pytest.raises(ValueError, match="frozen"):
        model.set_prompts([{"class_id": "helmet", "text": "new helmet"}])
    assert model.prompts == CONFIG["prompts"] and len(calls) == 1


def test_combined_worker_reuses_model_and_checks_each_new_phrase(simulated_model):
    checkpoint, calls, tokenizer = simulated_model
    model = worker._Runtime({**CONFIG, "allow_dynamic_prompts": True}, str(checkpoint))
    prompts = [{"class_id": "helmet", "text": "blue hard hat"}]
    assert model.set_prompts(prompts) == {"prompts": prompts}
    assert model.prompts == prompts and len(calls) == 1
    with pytest.raises(ValueError, match="class order"):
        model.set_prompts([{"class_id": "new-class", "text": "helmet"}])
    tokenizer.encode = lambda _: list(range(31))
    with pytest.raises(ValueError, match="30 tokenizer"):
        model.set_prompts([{"class_id": "helmet", "text": "long tokenization"}])
    assert model.prompts == prompts


def test_default_parent_cannot_change_prompts(simulated):
    checkpoint = simulated()
    with runtime.SamRuntime(CONFIG, checkpoint) as model:
        with pytest.raises(runtime.SamRuntimeError, match="dynamic"):
            model.set_prompts(CONFIG["prompts"])


def test_combined_parent_requires_exact_acknowledgment(tmp_path, monkeypatch, simulated):
    checkpoint = simulated()
    source = FAKE_WORKER.replace(
        "    else:\n        if MODE in",
        "    elif req['op'] == 'set_prompts':\n"
        "        value = {'prompts': req['prompts']}\n"
        "    else:\n        if MODE in",
    )
    path = tmp_path / "dynamic_worker.py"
    path.write_text(f"MODE='normal'\nIDENTITY={identity()!r}\n" + source)
    monkeypatch.setattr(runtime, "_WORKER_PATH", path)
    with runtime.SamRuntime(CONFIG, checkpoint, allow_dynamic_prompts=True) as model:
        process = model.process
        prompts = [{"class_id": "helmet", "text": "blue safety helmet"}]
        model.set_prompts(prompts)
        assert model.prompts == prompts and model.process is process
        assert model.metadata["received_config"]["allow_dynamic_prompts"] is True
        wrong = deepcopy(prompts)
        wrong[0]["class_id"] = "outside-taxonomy"
        with pytest.raises(runtime.SamRuntimeError, match="class order"):
            model.set_prompts(wrong)
    path.write_text(
        f"MODE='normal'\nIDENTITY={identity()!r}\n"
        + source.replace("value = {'prompts': req['prompts']}", "value = {'prompts': []}")
    )
    with runtime.SamRuntime(CONFIG, checkpoint, allow_dynamic_prompts=True) as model:
        with pytest.raises(runtime.SamRuntimeError, match="acknowledge"):
            model.set_prompts(CONFIG["prompts"])
        assert model.process is None
