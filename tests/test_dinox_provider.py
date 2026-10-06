"""Offline DINO-X transport/normalization/credential fixtures; no paid calls."""

import base64
import io
import json
import os
import stat
import traceback
from copy import deepcopy

import pytest
from PIL import Image, PngImagePlugin

from iris import dinox_provider as provider
from iris.taxonomies import TAXONOMY

KEY = "synthetic-dinox-test-token"


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv(provider.KEY_ENV, raising=False)
    monkeypatch.setattr(provider, "CREDENTIAL_PATH", tmp_path / "config" / "iris" / "keys.json")


@pytest.fixture
def config():
    return provider.freeze_config(TAXONOMY)


@pytest.fixture
def png():
    return provider.encode_image(Image.new("RGB", (100, 80), "blue"))


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv(provider.KEY_ENV, KEY)
    state = {
        "requests": [],
        "connections": [],
        "closed": 0,
        "status": 200,
        "error": None,
        "response": {"code": 0, "data": {"task_uuid": "synthetic-task-1"}},
    }

    class Response:
        def __init__(self):
            self.status = state["status"]
            value = state["response"]
            self.stream = io.BytesIO(
                value if isinstance(value, bytes) else json.dumps(value).encode()
            )

        def read1(self, size):
            return self.stream.read(size)

    class Connection:
        sock = None

        def __init__(self, host, *, timeout):
            state["connections"].append((host, timeout))

        def request(self, method, path, *, body, headers):
            state["requests"].append(
                {"method": method, "path": path, "body": body, "headers": headers}
            )
            if state["error"]:
                raise state["error"]

        def getresponse(self):
            return Response()

        def close(self):
            state["closed"] += 1

    monkeypatch.setattr(provider.http.client, "HTTPSConnection", Connection)
    return state


def test_frozen_multiclass_config_offline_and_self_contained(config, transport):
    assert config["settings"] == {
        "model": "DINO-X-1.0",
        "prompt": {"type": "text", "text": "person . car"},
        "targets": ["bbox"],
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8,
    }
    assert config["label_mapping"] == {"person": "person", "car": "car"}
    assert provider.validate_frozen_config(json.loads(json.dumps(config))) == config
    assert provider.provider_status()["status"] == "ready"
    assert provider.provider_status()["connection_verified"] is False
    assert provider.provider_status()["key_source"] == "environment"
    assert provider.estimate(8)["total"] == 1.2
    assert provider.estimate(0)["total"] == 0
    assert transport["connections"] == []
    assert KEY not in json.dumps(config)
    assert str(provider.CREDENTIAL_PATH) not in json.dumps(config)
    config["taxonomy"]["classes"][0]["name"] = "changed"
    assert TAXONOMY["classes"][0]["name"] == "Person"


def test_custom_prompts_are_exact_canonical_and_complete():
    config = provider.freeze_config(TAXONOMY, {"person": " Construction Worker ", "car": "sedan"})
    assert config["settings"]["prompt"]["text"] == "construction worker . sedan"
    assert config["label_mapping"] == {"construction worker": "person", "sedan": "car"}
    result = provider.normalize(
        {
            "objects": [
                {"category": "construction worker", "bbox": [1, 2, 10, 20], "score": 0.8},
            ]
        },
        config,
        100,
        80,
    )
    assert result["proposals"][0]["label"] == "person"


@pytest.mark.parametrize(
    "prompts",
    [
        {},
        {"person": "person"},
        {"person": "person", "car": "person"},
        {"person": "person.car", "car": "car"},
        {"person": "person\n", "car": "car"},
        {"person": "<prompt_free>", "car": "car"},
        {"person": " ", "car": "car"},
        {"person": 5, "car": "car"},
    ],
)
def test_prompt_ambiguity_is_rejected(prompts):
    with pytest.raises(provider.DinoXConfigError):
        provider.freeze_config(TAXONOMY, prompts)


@pytest.mark.parametrize("threshold", [True, -0.1, 1.1, float("nan"), "0.25"])
def test_invalid_threshold(threshold):
    with pytest.raises(provider.DinoXConfigError):
        provider.freeze_config(TAXONOMY, threshold=threshold)


@pytest.mark.parametrize(
    "key,value",
    [
        ("endpoint", "https://example.com/receive"),
        ("model", "other"),
        ("native_coordinates", "xywh"),
        ("max_proposals", True),
        ("api_key", KEY),
        ("label_mapping", {"person": "car", "car": "person"}),
    ],
)
def test_frozen_config_rejects_tampering(config, key, value):
    config[key] = value
    with pytest.raises(provider.DinoXConfigError):
        provider.validate_frozen_config(config)


def test_encode_preserves_pixels_and_dimensions_removes_metadata():
    image = Image.new("RGB", (30, 20), (4, 55, 166))
    image.info.update(exif=b"source-private-path", icc_profile=b"private-profile")
    data = provider.encode_image(image)
    with Image.open(io.BytesIO(data)) as decoded:
        assert decoded.size == image.size
        assert decoded.tobytes() == image.tobytes()
        assert decoded.info == {}
    assert b"source-private-path" not in data


def test_one_submit_contains_only_expected_payload(config, png, transport):
    result = provider.submit(config, png, idempotency_key="intent-1")
    assert result["task_id"] == "synthetic-task-1"
    assert result["metadata"]["image_bytes"] == len(png)
    assert transport["connections"] == [(provider.HOST, provider.REQUEST_TIMEOUT)]
    assert len(transport["requests"]) == transport["closed"] == 1
    request = transport["requests"][0]
    assert request["path"] == "/v2/task/dinox/detection"
    assert request["headers"]["Token"] == KEY
    assert request["headers"]["Idempotency-Key"] == "intent-1"
    payload = json.loads(request["body"])
    assert payload == {
        **config["settings"],
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
    }
    assert not any(key in payload for key in ("taxonomy", "file", "project", "labels", "path"))
    assert KEY not in json.dumps(result)


def test_bad_png_and_image_bounds_never_dispatch(config, png, transport, monkeypatch):
    image = Image.new("RGB", (100, 80))
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("filename", "private-name")
    stream = io.BytesIO()
    image.save(stream, format="PNG", pnginfo=metadata)
    for invalid in (b"bad", b"\x89PNG\r\n\x1a\n", stream.getvalue()):
        with pytest.raises(provider.DinoXConfigError):
            provider.submit(config, invalid, idempotency_key="intent-1")
    monkeypatch.setattr(provider, "MAX_IMAGE_PIXELS", 2)
    with pytest.raises(provider.DinoXConfigError):
        provider.encode_image(image)
    assert transport["requests"] == []


@pytest.mark.parametrize("identifier", ["../bad", "x?token=secret", "a\r\nToken: injected", "", 5])
def test_invalid_ids_do_not_dispatch(identifier, transport, config, png):
    with pytest.raises(provider.DinoXConfigError):
        provider.poll(identifier)
    with pytest.raises(provider.DinoXConfigError):
        provider.submit(config, png, idempotency_key=identifier)
    assert transport["requests"] == []


@pytest.mark.parametrize(
    "status,expected",
    [
        ("waiting", "pending"),
        ("running", "pending"),
        ("processing", "pending"),
        ("failed", "failed"),
        ("success", "succeeded"),
    ],
)
def test_poll_is_one_get_and_recognizes_lifecycle(transport, status, expected):
    transport["response"] = {"code": 0, "data": {"status": status, "result": {"objects": []}}}
    result = provider.poll("task-1")
    assert result["status"] == expected
    assert transport["requests"][0]["body"] is None
    assert transport["requests"][0]["method"] == "GET"
    assert transport["requests"][0]["path"] == "/v2/task_status/task-1"
    assert len(transport["requests"]) == 1
    assert ("result" in result) is (status == "success")


@pytest.mark.parametrize(
    "response",
    [
        b"not json",
        b'{"code": 23, "code": 0, "data": {"status": "waiting"}}',
        b'{"code": 0,"data":NaN}',
        [],
        {"code": True, "data": {}},
        {"code": 0},
        {"code": 0, "data": {"status": "invented"}},
        {"code": 0, "data": {"status": "success", "result": None}},
    ],
)
def test_bad_responses_fail_closed(transport, response):
    transport["response"] = response
    with pytest.raises(provider.DinoXResponseError):
        provider.poll("task-1")
    assert len(transport["requests"]) == transport["closed"] == 1


def test_ambiguous_timeout_never_retries_or_exposes_secrets(config, png, transport):
    transport["error"] = TimeoutError("private-image-path " + KEY)
    with pytest.raises(provider.DinoXTransportError) as caught:
        provider.submit(config, png, idempotency_key="intent-1")
    error = caught.value
    assert error.outcome_unknown and error.dispatched
    formatted = "".join(traceback.format_exception(error))
    assert KEY not in formatted and "private-image-path" not in formatted
    assert len(transport["requests"]) == transport["closed"] == 1


@pytest.mark.parametrize(
    "http_status,unknown", [(401, False), (403, False), (429, True), (503, True), (302, True)]
)
def test_http_errors_are_safe_and_never_follow_or_retry(
    config, png, transport, http_status, unknown
):
    transport["status"] = http_status
    transport["response"] = {
        "code": 23,
        "msg": "private-response " + KEY,
        "Token": "other-secret",
        "image": "private-pixels",
    }
    with pytest.raises(provider.DinoXResponseError) as caught:
        provider.submit(config, png, idempotency_key="intent-1")
    error = caught.value
    assert error.outcome_unknown is unknown
    assert "private-response" not in str(error)
    assert KEY not in json.dumps(error.raw_response)
    assert error.raw_response["body"]["Token"] == "[redacted]"
    assert error.raw_response["body"]["image"] == "[redacted]"
    assert len(transport["requests"]) == 1


def test_response_limit(transport, monkeypatch):
    monkeypatch.setattr(provider, "MAX_RESPONSE_BYTES", 100)
    transport["response"] = b"x" * 101
    with pytest.raises(provider.DinoXResponseError):
        provider.poll("task-1")
    assert transport["closed"] == 1


def test_normalize_preserves_native_geometry_filters_and_accepts_empty(config):
    raw = {
        "objects": [
            {"category": "car", "bbox": [-2, 10, 105, 85], "score": 0.9},
            {"category": "person", "bbox": [1, 2, 4, 8], "score": 0.1},
        ]
    }
    original = deepcopy(raw)
    result = provider.normalize(raw, config, 100, 80)
    assert raw == original
    assert result["clipped_count"] == result["filtered_count"] == 1
    assert result["proposals"][0]["box"] == [0.0, 10.0, 100.0, 80.0]
    assert result["proposals"][0]["source"]["native_box"] == [-2, 10, 105, 85]
    assert result["proposals"][0]["source"]["native_coordinates"] == "xyxy_original_pixels"
    assert provider.normalize({"objects": []}, config, 100, 80)["proposals"] == []


@pytest.mark.parametrize(
    "item",
    [
        {"category": "unknown", "bbox": [1, 2, 4, 8], "score": 0.8},
        {"category": "Person", "bbox": [1, 2, 4, 8], "score": 0.8},
        {"category": "person", "bbox": [1, 2, 4, 8], "score": True},
        {"category": "person", "bbox": [1, 2, 4, 8], "score": 1.1},
        {"category": "person", "bbox": [1, 2, 4, 8], "score": float("nan")},
        {"category": "person", "bbox": [1, 2, 0, 8], "score": 0.8},
        {"category": "person", "bbox": [1, 2, float("inf"), 8], "score": 0.8},
        {"category": "person", "bbox": [101, 2, 104, 8], "score": 0.8},
        {"category": "person", "bbox": [True, 2, 4, 8], "score": 0.8},
    ],
)
def test_invalid_or_unknown_native_output_never_silently_dropped(config, item):
    with pytest.raises(provider.DinoXResponseError):
        provider.normalize({"objects": [item]}, config, 100, 80)


@pytest.mark.parametrize(
    "raw",
    [{}, {"objects": None}, {"objects": {}}, {"objects": [None] * (provider.MAX_OUTPUTS + 1)}],
)
def test_output_requires_bounded_explicit_array(config, raw):
    with pytest.raises(provider.DinoXResponseError):
        provider.normalize(raw, config, 100, 80)


def test_credentials_are_private_atomic_dynamic_and_preserve_other_keys():
    path = provider.CREDENTIAL_PATH
    assert provider.provider_status()["status"] == "missing_key"
    assert provider.update_key(KEY) is True
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    path.write_text(
        json.dumps({provider.KEY_ENV: KEY, "FAL_KEY": "synthetic-fal", "other": {"a": 1}})
    )
    assert provider.update_key("synthetic-replacement") is True
    assert json.loads(path.read_text()) == {
        provider.KEY_ENV: "synthetic-replacement",
        "FAL_KEY": "synthetic-fal",
        "other": {"a": 1},
    }
    assert provider.provider_status()["key_source"] == "file"
    assert provider.provider_status()["key_configured"] is True
    assert provider.clear_key() is True
    assert json.loads(path.read_text()) == {"FAL_KEY": "synthetic-fal", "other": {"a": 1}}
    assert provider.clear_key() is False
    assert list(path.parent.iterdir()) == [path]
    assert provider.provider_status()["status"] == "missing_key"


def test_environment_wins_and_clear_does_not_claim_to_clear_it(monkeypatch):
    provider.update_key("synthetic-file")
    monkeypatch.setenv(provider.KEY_ENV, KEY)
    assert provider._credential() == (KEY, "environment")
    provider.clear_key()
    assert provider.provider_status()["status"] == "ready"
    assert provider.provider_status()["environment_override"] is True
    monkeypatch.setenv(provider.KEY_ENV, "bad\nkey")
    assert provider.provider_status()["status"] == "invalid_key"


@pytest.mark.parametrize("invalid", [None, "", "bad key", "bad\r\nkey", "é", "x" * 1025])
def test_invalid_credential_never_written(invalid):
    with pytest.raises(provider.DinoXConfigError) as caught:
        provider.update_key(invalid)
    assert caught.value.dispatched is False
    assert not provider.CREDENTIAL_PATH.exists()


def test_symlink_credential_and_parent_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("do not alter")
    provider.CREDENTIAL_PATH.parent.mkdir(mode=0o700, parents=True)
    provider.CREDENTIAL_PATH.symlink_to(target)
    assert provider.provider_status()["status"] == "invalid_key"
    with pytest.raises(provider.DinoXConfigError):
        provider.update_key(KEY)
    with pytest.raises(provider.DinoXConfigError):
        provider.clear_key()
    assert target.read_text() == "do not alter"
    provider.CREDENTIAL_PATH.unlink()
    provider.CREDENTIAL_PATH.parent.rmdir()
    provider.CREDENTIAL_PATH.parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(provider.DinoXConfigError):
        provider.update_key(KEY)


def test_insecure_or_malformed_credential_store_is_not_overwritten():
    provider.update_key(KEY)
    path = provider.CREDENTIAL_PATH
    os.chmod(path, 0o644)
    with pytest.raises(provider.DinoXConfigError):
        provider.clear_key()
    os.chmod(path, 0o600)
    path.write_text("secret malformed input")
    with pytest.raises(provider.DinoXConfigError) as caught:
        provider.update_key("replacement")
    assert "secret malformed input" not in "".join(traceback.format_exception(caught.value))
    assert path.read_text() == "secret malformed input"


def test_failed_atomic_replacement_retains_previous_value(monkeypatch):
    provider.update_key(KEY)
    old = provider.CREDENTIAL_PATH.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("sensitive private failure")

    monkeypatch.setattr(provider.os, "replace", fail)
    with pytest.raises(provider.DinoXConfigError):
        provider.update_key("replacement")
    assert provider.CREDENTIAL_PATH.read_bytes() == old
    assert list(provider.CREDENTIAL_PATH.parent.iterdir()) == [provider.CREDENTIAL_PATH]


def test_ambiguous_json_credential_file_is_not_overwritten():
    provider.update_key(KEY)
    content = '{"FAL_KEY": "first", "FAL_KEY": "second"}'
    provider.CREDENTIAL_PATH.write_text(content)
    with pytest.raises(provider.DinoXConfigError):
        provider.update_key("replacement")
    assert provider.CREDENTIAL_PATH.read_text() == content
