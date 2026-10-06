"""Explicit DINO-X requests and offline, frozen box contracts.

This module never starts a batch, retries a POST, or accepts a configurable host.
The caller must durably record its intent before ``submit`` and its returned task
ID before polling. Catalog/configuration/credential operations do no network I/O.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import http.client
import io
import json
import math
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from PIL import Image

from iris.dataset_manifest import taxonomy_mappings
from iris.preannotation_contracts import MAX_OUTPUTS, OUTPUT_PROTOCOL, normalize_output

PROTOCOL = "iris-dinox-preannotation-v1"
MODEL = "DINO-X-1.0"
HOST = "api.deepdataspace.com"
API_PATH = "/v2/task/dinox/detection"
KEY_ENV = "DDS_CLOUD_API_TOKEN"
CREDENTIAL_PATH = Path.home() / ".config" / "iris" / "cloud-credentials.json"
MAX_CREDENTIAL_BYTES = 64 * 1024
MAX_PNG_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
REQUEST_TIMEOUT = 45.0
PRICE_CHECKED_AT = "2026-10-06"
PRICE_SOURCE = "https://algos.deepdataspace.com/en/price/README.md"
SOURCES = [
    "https://github.com/IDEA-Research/DINO-X-API/blob/main/demo.py",
    "https://github.com/deepdataspace/dds-cloudapi-sdk/blob/main/dds_cloudapi_sdk/tasks/base.py",
    "https://github.com/deepdataspace/dds-cloudapi-sdk/blob/main/dds_cloudapi_sdk/tasks/v2_task.py",
    PRICE_SOURCE,
]
REPRODUCIBILITY_WARNING = (
    "Hosted model version label; model weights and service implementation are not pinned."
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_MESSAGES = {
    "invalid_config": "DINO-X requires the unchanged frozen provider configuration.",
    "invalid_prompt": "Provide one distinct text prompt per class without periods or controls.",
    "missing_key": "The DINO-X credential is not configured.",
    "invalid_key": "The DINO-X credential is missing or invalid.",
    "credential_storage": "The local credential file could not be accessed securely.",
    "invalid_image": "DINO-X requires a bounded metadata-free RGB PNG image.",
    "invalid_identifier": "DINO-X requires a valid task or request identifier.",
    "transport": "DINO-X transport failed; the paid submission outcome may be unknown.",
    "http_status": "DINO-X returned an unsuccessful HTTP status.",
    "invalid_response": "DINO-X returned an invalid or oversized response.",
    "provider_error": "DINO-X reported an API error.",
    "invalid_output": "DINO-X output does not satisfy the frozen class and box contract.",
}


class DinoXError(RuntimeError):
    """Safe fixed message and optional sanitized evidence, never remote exception text."""

    def __init__(self, code, *, raw_response=None, dispatched=False, outcome_unknown=False):
        self.code = code if code in _MESSAGES else "invalid_response"
        self.raw_response = raw_response
        self.metadata = {}
        self.dispatched = dispatched
        self.outcome_unknown = outcome_unknown
        super().__init__(_MESSAGES[self.code])


class DinoXConfigError(DinoXError, ValueError):
    pass


class DinoXTransportError(DinoXError):
    def __init__(self, code):
        super().__init__(code, dispatched=True, outcome_unknown=True)


class DinoXResponseError(DinoXError, ValueError):
    def __init__(self, code, *, raw_response=None, dispatched=True, outcome_unknown=True):
        super().__init__(
            code, raw_response=raw_response, dispatched=dispatched, outcome_unknown=outcome_unknown
        )


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _parse_json(content):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Ambiguous JSON object")
            result[key] = value
        return result

    return json.loads(content, object_pairs_hook=unique_object)


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def estimate(count):
    """Frozen list-price estimate, not a bill or an account/quota query."""
    if type(count) is not int or not 0 <= count <= 1_000_000:
        raise DinoXConfigError("invalid_config")
    return {
        "currency": "CNY",
        "amount_per_request": 0.15,
        "request_count": count,
        "total": float(Decimal("0.15") * count),
        "pricing_source": PRICE_SOURCE,
        "price_checked_at": PRICE_CHECKED_AT,
        "basis": "One detection request per image, all classes together; estimate, not an invoice.",
    }


def freeze_config(taxonomy, class_prompts=None, threshold=0.25):
    """Snapshot class semantics and exact native-category mapping without credentials."""
    try:
        taxonomy_mappings(taxonomy)
    except (TypeError, ValueError, KeyError, OverflowError):
        raise DinoXConfigError("invalid_config") from None
    if not _number(threshold) or not 0 <= threshold <= 1:
        raise DinoXConfigError("invalid_config")
    identifiers = [item["id"] for item in taxonomy["classes"]]
    if class_prompts is None:
        class_prompts = {identifier: identifier for identifier in identifiers}
    if not isinstance(class_prompts, dict) or set(class_prompts) != set(identifiers):
        raise DinoXConfigError("invalid_prompt")
    prompts = {}
    for identifier in identifiers:
        value = class_prompts[identifier]
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 120
            or "." in value
            or any(not c.isprintable() for c in value)
            or not value.strip()
            or "<" in value
            or ">" in value
        ):
            raise DinoXConfigError("invalid_prompt")
        prompts[identifier] = " ".join(value.lower().split())
    if len(set(prompts.values())) != len(prompts):
        raise DinoXConfigError("invalid_prompt")
    prompt = " . ".join(prompts.values())
    if len(prompt) > 4096:
        raise DinoXConfigError("invalid_prompt")
    result = {
        "protocol": PROTOCOL,
        "provider": "dinox",
        "model": MODEL,
        "endpoint": f"https://{HOST}{API_PATH}",
        "operation": "propose_boxes",
        "taxonomy": deepcopy(taxonomy),
        "taxonomy_id": taxonomy["id"],
        "class_prompts": prompts,
        "label_mapping": {value: identifier for identifier, value in prompts.items()},
        "settings": {
            "model": MODEL,
            "prompt": {"type": "text", "text": prompt},
            "targets": ["bbox"],
            "bbox_threshold": float(threshold),
            "iou_threshold": 0.8,
        },
        "native_coordinates": "xyxy_original_pixels",
        "image_encoding": {
            "format": "PNG",
            "mode": "RGB",
            "source_metadata": "removed",
            "resized": False,
            "max_bytes": MAX_PNG_BYTES,
            "max_pixels": MAX_IMAGE_PIXELS,
        },
        "max_response_bytes": MAX_RESPONSE_BYTES,
        "max_proposals": MAX_OUTPUTS,
        "pricing": estimate(1),
        "sources": list(SOURCES),
        "reproducibility": REPRODUCIBILITY_WARNING,
    }
    try:
        _json(result)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DinoXConfigError("invalid_config") from None
    return result


def validate_frozen_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("settings"), dict):
        raise DinoXConfigError("invalid_config")
    expected = freeze_config(
        config.get("taxonomy"),
        config.get("class_prompts"),
        config["settings"].get("bbox_threshold"),
    )
    try:
        if _json(config) != _json(expected):
            raise DinoXConfigError("invalid_config")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DinoXConfigError("invalid_config") from None
    return expected


def _valid_key(value):
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 1024
        and all(33 <= ord(c) <= 126 for c in value)
    )


@contextmanager
def _credential_directory(*, create=False):
    """Walk with directory descriptors so symlinks cannot redirect credential I/O."""
    descriptor = None
    try:
        path = CREDENTIAL_PATH.parent
        if not path.is_absolute() or ".." in path.parts:
            raise DinoXConfigError("credential_storage")
        descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for index, part in enumerate(path.parts[1:]):
            if create and index >= len(path.parts) - 3:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise DinoXConfigError("credential_storage")
        fcntl.flock(descriptor, fcntl.LOCK_EX if create else fcntl.LOCK_SH)
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_credentials(descriptor):
    try:
        fd = os.open(
            CREDENTIAL_PATH.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
        )
    except FileNotFoundError:
        return {}
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_size > MAX_CREDENTIAL_BYTES
        ):
            raise DinoXConfigError("credential_storage")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            content = stream.read(MAX_CREDENTIAL_BYTES + 1)
        values = _parse_json(content)
        if not isinstance(values, dict) or len(content) > MAX_CREDENTIAL_BYTES:
            raise DinoXConfigError("credential_storage")
        _json(values)
        return values
    finally:
        os.close(fd)


def _disk_credentials():
    try:
        with _credential_directory() as descriptor:
            return _read_credentials(descriptor)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError, RecursionError):
        raise DinoXConfigError("credential_storage") from None


def _credential():
    value = os.environ.get(KEY_ENV)
    source = "environment" if value else "file"
    if not value:
        value = _disk_credentials().get(KEY_ENV)
    if value is None or value == "":
        raise DinoXConfigError("missing_key")
    if not _valid_key(value):
        raise DinoXConfigError("invalid_key")
    return value, source


def _write_key(value):
    temporary = None
    try:
        with _credential_directory(create=True) as descriptor:
            values = _read_credentials(descriptor)
            if value is None and KEY_ENV not in values:
                return False
            if value is None:
                del values[KEY_ENV]
            else:
                values[KEY_ENV] = value
            content = _json(values) + b"\n"
            if len(content) > MAX_CREDENTIAL_BYTES:
                raise DinoXConfigError("credential_storage")
            temporary = ".credentials-" + uuid.uuid4().hex
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            try:
                with os.fdopen(fd, "wb", closefd=False) as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(fd)
                os.replace(
                    temporary, CREDENTIAL_PATH.name, src_dir_fd=descriptor, dst_dir_fd=descriptor
                )
                temporary = None
                os.fsync(descriptor)
            finally:
                os.close(fd)
                if temporary is not None:
                    os.unlink(temporary, dir_fd=descriptor)
            return True
    except (OSError, ValueError, TypeError, RecursionError):
        raise DinoXConfigError("credential_storage") from None


def update_key(value):
    """Save only the DINO-X key, atomically; callers enforce local web access."""
    if not _valid_key(value):
        raise DinoXConfigError("invalid_key")
    return _write_key(value)


def clear_key():
    """Remove only the disk key; a process environment override remains effective."""
    return _write_key(None)


def provider_status():
    result = {
        "provider": "dinox",
        "model": MODEL,
        "local_only": False,
        "connection_verified": False,
        "configured": False,
        "key_configured": False,
        "key_source": None,
        "environment_override": bool(os.environ.get(KEY_ENV)),
        "key_env": KEY_ENV,
        "credential_updates": (
            "Local-file updates apply immediately; environment changes need restart."
        ),
        "estimate": estimate(1),
        "price": estimate(1),
    }
    try:
        _, source = _credential()
    except DinoXConfigError as exc:
        return {
            **result,
            "status": "missing_key" if exc.code == "missing_key" else "invalid_key",
            "reason": str(exc),
        }
    return {
        **result,
        "configured": True,
        "key_configured": True,
        "key_source": source,
        "status": "ready",
        "reason": "Configuration present; connectivity and credentials have not been tested.",
    }


def _size(width, height):
    return (
        type(width) is int
        and type(height) is int
        and width > 0
        and height > 0
        and width * height <= MAX_IMAGE_PIXELS
    )


def encode_image(image):
    """Encode original oriented pixels with no source metadata, file name or resize."""
    if not isinstance(image, Image.Image) or not _size(*image.size):
        raise DinoXConfigError("invalid_image")
    try:
        rgb = image.convert("RGB")
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        output = io.BytesIO()
        clean.save(output, format="PNG")
        data = output.getvalue()
        if len(data) > MAX_PNG_BYTES:
            raise DinoXConfigError("invalid_image")
        return data
    except (OSError, ValueError, TypeError):
        raise DinoXConfigError("invalid_image") from None


def _validate_png(png):
    if (
        not isinstance(png, bytes)
        or len(png) > MAX_PNG_BYTES
        or not png.startswith(b"\x89PNG\r\n\x1a\n")
    ):
        raise DinoXConfigError("invalid_image")
    try:
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.info or not _size(*image.size):
                raise DinoXConfigError("invalid_image")
            image.verify()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise DinoXConfigError("invalid_image") from None


def _redact(value, secret):
    if isinstance(value, str):
        value = value.replace(secret, "[redacted]") if secret else value
        return re.sub(r"data:image/[^;,\s]+;base64,[A-Za-z0-9+/=]+", "[redacted image]", value)
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {
            _redact(key, secret): (
                "[redacted]"
                if key.lower()
                in {
                    "token",
                    "authorization",
                    "api_key",
                    "apikey",
                    "access_token",
                    "dds_cloud_api_token",
                    "image",
                    "image_url",
                }
                else _redact(item, secret)
            )
            for key, item in value.items()
        }
    return value


def _request(method, path, key, *, payload=None, idempotency_key=None):
    headers = {"Token": key, "Accept": "application/json"}
    body = None
    if payload is not None:
        body = _json(payload)
        headers.update({"Content-Type": "application/json", "Idempotency-Key": idempotency_key})
    connection = None
    started = time.monotonic()
    try:
        connection = http.client.HTTPSConnection(HOST, timeout=REQUEST_TIMEOUT)
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        chunks, size = [], 0
        while True:
            remaining = REQUEST_TIMEOUT - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            chunk = response.read1(min(64 * 1024, MAX_RESPONSE_BYTES + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise DinoXResponseError("invalid_response")
            chunks.append(chunk)
        parsed = _parse_json(b"".join(chunks))
        if not isinstance(parsed, dict):
            raise DinoXResponseError("invalid_response")
        _json(parsed)  # reject non-finite JSON before preserving any evidence
        receipt = {
            "http_status": response.status,
            "body": _redact(parsed, key),
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }
        if not 200 <= response.status < 300:
            raise DinoXResponseError(
                "http_status",
                raw_response=receipt,
                outcome_unknown=response.status not in {401, 403},
            )
        if type(parsed.get("code")) is not int or parsed["code"] != 0:
            raise DinoXResponseError("provider_error", raw_response=receipt)
        if not isinstance(parsed.get("data"), dict):
            raise DinoXResponseError("invalid_response", raw_response=receipt)
        return receipt
    except DinoXError:
        raise
    except (OSError, http.client.HTTPException):
        raise DinoXTransportError("transport") from None
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise DinoXResponseError("invalid_response") from None
    finally:
        if connection is not None:
            try:
                connection.close()
            except (OSError, http.client.HTTPException):
                pass


def submit(config, png, *, idempotency_key):
    """Perform at most one POST; there is no retry, even on an ambiguous timeout."""
    config = validate_frozen_config(config)
    _validate_png(png)
    if not isinstance(idempotency_key, str) or not _IDENTIFIER.fullmatch(idempotency_key):
        raise DinoXConfigError("invalid_identifier")
    key, _ = _credential()
    payload = {
        **config["settings"],
        "image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
    }
    receipt = _request("POST", API_PATH, key, payload=payload, idempotency_key=idempotency_key)
    task_id = receipt["body"]["data"].get("task_uuid")
    if not isinstance(task_id, str) or not _IDENTIFIER.fullmatch(task_id):
        raise DinoXResponseError("invalid_response", raw_response=receipt)
    return {
        "task_id": task_id,
        "raw_response": receipt,
        "metadata": {
            "provider": "dinox",
            "model": MODEL,
            "image_sha256": hashlib.sha256(png).hexdigest(),
            "image_bytes": len(png),
            "idempotency_key": idempotency_key,
        },
    }


def poll(task_id):
    """Perform a single GET without waiting, sleeping or creating another task."""
    if not isinstance(task_id, str) or not _IDENTIFIER.fullmatch(task_id):
        raise DinoXConfigError("invalid_identifier")
    key, _ = _credential()
    receipt = _request("GET", "/v2/task_status/" + task_id, key)
    data = receipt["body"]["data"]
    status = data.get("status")
    if isinstance(status, str) and status in {"waiting", "running", "processing"}:
        return {"status": "pending", "raw_response": receipt}
    if status == "failed":
        return {"status": "failed", "raw_response": receipt, "error_code": "provider_failed"}
    if status == "success" and isinstance(data.get("result"), dict):
        return {"status": "succeeded", "raw_response": receipt, "result": data["result"]}
    raise DinoXResponseError("invalid_response", raw_response=receipt)


def normalize(raw_result, config, width, height):
    """Strictly map every native category; preserve geometry and clip visible bounds."""
    config = validate_frozen_config(config)
    try:
        if (
            not _size(width, height)
            or not isinstance(raw_result, dict)
            or len(_json(raw_result)) > MAX_RESPONSE_BYTES
        ):
            raise ValueError
        objects = raw_result.get("objects")
        if not isinstance(objects, list) or len(objects) > MAX_OUTPUTS:
            raise ValueError
        proposals, clipped_count, filtered_count = [], 0, 0
        for index, item in enumerate(objects):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("category"), str)
                or item["category"] not in config["label_mapping"]
            ):
                raise ValueError
            box, score = item.get("bbox"), item.get("score")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(_number(v) for v in box)
                or not _number(score)
                or not 0 <= score <= 1
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError
            clipped = [
                min(width if axis % 2 == 0 else height, max(0, value))
                for axis, value in enumerate(box)
            ]
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                raise ValueError
            if score < config["settings"]["bbox_threshold"]:
                filtered_count += 1
                continue
            changed = clipped != box
            clipped_count += changed
            proposals.append(
                {
                    "id": f"dinox-{index}",
                    "label": config["label_mapping"][item["category"]],
                    "box": clipped,
                    "score": float(score),
                    "source": {
                        "provider": "dinox",
                        "model": MODEL,
                        "native_index": index,
                        "prompt": item["category"],
                        "native_box": deepcopy(box),
                        "native_coordinates": config["native_coordinates"],
                        "clipped": changed,
                    },
                }
            )
        result = normalize_output(
            {
                "protocol": OUTPUT_PROTOCOL,
                "taxonomy_id": config["taxonomy_id"],
                "coordinates": {
                    "format": "xyxy",
                    "space": "original_pixels",
                    "image_size": [width, height],
                    "to_original": {"scale": [1, 1], "offset": [0, 0]},
                },
                "proposals": proposals,
            },
            config["taxonomy"],
            width=width,
            height=height,
        )
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise DinoXResponseError(
            "invalid_output", dispatched=False, outcome_unknown=False
        ) from None
    result.update(filtered_count=filtered_count, clipped_count=clipped_count)
    if clipped_count:
        result["warnings"].append(
            f"{clipped_count} boxes clipped to image bounds; native coordinates preserved."
        )
    result["warnings"].append(REPRODUCIBILITY_WARNING)
    return result
