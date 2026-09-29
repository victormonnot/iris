"""Explicit, single-request reviews through Alibaba Cloud Model Studio.

Configuration and catalog reads are offline. Only ``review`` sends image pixels;
the assistance service is responsible for obtaining consent for that exact request.
"""

from __future__ import annotations

import base64
import http.client
import json
import math
import os
import re
import time
from copy import deepcopy
from decimal import Decimal
from urllib.parse import urlsplit

from PIL import Image

from iris.assistance_provider import (
    MAX_RESPONSE_BYTES,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    ProviderResponseError,
    _encode,
    _json,
    _schema,
    _validate_candidates,
    _validate_review,
)

KEY_ENV = "IRIS_DASHSCOPE_API_KEY"
ENDPOINT_ENV = "IRIS_DASHSCOPE_BASE_URL"
DEFAULT_MODEL = "qwen3-vl-32b-instruct"
MAX_INPUT_TOKENS = 129024
MAX_OUTPUT_TOKENS = 1024
REVIEW_TIMEOUT = 180.0
PRICE_CHECKED_AT = "2026-09-29"
REGION = "Germany (Frankfurt)"
DEPLOYMENT_SCOPE = "Global"
REGIONS_SOURCE = "https://www.alibabacloud.com/help/en/model-studio/regions"
STRUCTURED_OUTPUT_SOURCE = (
    "https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output"
)
MODELS = {
    "qwen3-vl-32b-instruct": {
        "id": "alibaba-qwen3-vl-32b",
        "name": "Qwen3-VL 32B Instruct",
        "input_usd_per_million": "0.16",
        "output_usd_per_million": "0.64",
    },
    "qwen3-vl-235b-a22b-instruct": {
        "id": "alibaba-qwen3-vl-235b",
        "name": "Qwen3-VL 235B-A22B Instruct",
        "input_usd_per_million": "0.287",
        "output_usd_per_million": "1.147",
    },
}
SETTINGS = {
    "temperature": 0,
    "max_tokens": MAX_OUTPUT_TOKENS,
    "stream": False,
    # These open-weight VL models support JSON Object, not native JSON Schema.
    "response_format": {"type": "json_object"},
}
IMAGE_ENCODING = {
    "format": "jpeg",
    "quality": 85,
    "frame_long_edge": 1024,
    "crop_long_edge": 320,
    "source_metadata": "removed",
}


def _config(config=None):
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - {"provider", "endpoint", "model"}:
        raise ValueError("Only the provider, endpoint and model may be configured.")
    if config.get("provider", "alibaba") != "alibaba":
        raise ValueError("Unsupported hosted provider.")
    model = config.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError("Select one of the supported Qwen3-VL Instruct models.")
    endpoint = config.get("endpoint", os.environ.get(ENDPOINT_ENV, ""))
    if not endpoint:
        raise ValueError(f"Set {ENDPOINT_ENV} to your Frankfurt workspace base URL.")
    if not isinstance(endpoint, str) or not re.fullmatch(
        r"https://[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
        r"\.eu-central-1\.maas\.aliyuncs\.com/compatible-mode/v1",
        endpoint,
    ):
        raise ValueError(
            "The API base URL must be an HTTPS Frankfurt workspace endpoint ending "
            "in .eu-central-1.maas.aliyuncs.com/compatible-mode/v1."
        )
    return {"provider": "alibaba", "endpoint": endpoint, "model": model}


def _api_key():
    value = os.environ.get(KEY_ENV, "")
    if not value:
        raise ValueError(f"Set {KEY_ENV} in the IRIS server environment.")
    if len(value) > 512 or not value.isascii() or any(not 33 <= ord(c) <= 126 for c in value):
        raise ValueError(f"{KEY_ENV} must be a valid API key without whitespace.")
    return value


def conservative_estimate(config=None):
    """List-price ceiling for one accepted request; not a tokenizer prediction."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("Invalid hosted provider configuration.")
    model = (config or {}).get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError("Unsupported hosted model.")
    spec = MODELS[model]
    input_rate = Decimal(spec["input_usd_per_million"])
    output_rate = Decimal(spec["output_usd_per_million"])
    bound = (MAX_INPUT_TOKENS * input_rate + MAX_OUTPUT_TOKENS * output_rate) / 1_000_000
    return {
        "currency": "USD",
        "upper_bound_usd": float(bound),
        "estimate_label": "Conservative per-request list-price ceiling",
        "basis": (
            "Full documented maximum input plus the configured output limit; "
            "actual image and text token counts are unknown before inference. "
            "Excludes taxes and later price changes; no discount assumed."
        ),
        "input_token_bound": MAX_INPUT_TOKENS,
        "output_token_bound": MAX_OUTPUT_TOKENS,
        "input_usd_per_million": float(input_rate),
        "output_usd_per_million": float(output_rate),
        "pricing_source": f"https://www.alibabacloud.com/help/en/model-studio/{model}",
        "price_checked_at": PRICE_CHECKED_AT,
    }


def provider_status(config=None):
    """Configuration presence only: no DNS lookup, auth probe or paid generation."""
    result = {
        "provider": "alibaba",
        "endpoint": None,
        "model": DEFAULT_MODEL,
        "local_only": False,
        "region": REGION,
        "deployment_scope": DEPLOYMENT_SCOPE,
        "key_env": KEY_ENV,
        "endpoint_env": ENDPOINT_ENV,
        "connection_verified": False,
    }
    try:
        checked = _config(config)
    except ValueError as exc:
        supplied = config if isinstance(config, dict) else {}
        model = supplied.get("model")
        if isinstance(model, str) and model in MODELS:
            result["model"] = model
        missing = (config is None or isinstance(config, dict)) and not supplied.get(
            "endpoint", os.environ.get(ENDPOINT_ENV, "")
        )
        return {
            **result,
            "status": "missing_config" if missing else "invalid_config",
            "reason": str(exc),
        }
    result.update(checked)
    try:
        _api_key()
    except ValueError as exc:
        return {**result, "status": "missing_key", "reason": str(exc)}
    return {
        **result,
        "status": "ready",
        "reason": (
            "Configuration present; connectivity and credentials have not been tested. "
            "Frankfurt is the access region; inference uses the provider's Global scope."
        ),
    }


def catalog():
    result = []
    for model, spec in MODELS.items():
        estimate = conservative_estimate({"model": model})
        result.append(
            {
                **provider_status({"model": model}),
                "id": spec["id"],
                "name": spec["name"],
                "estimate": estimate,
                "pricing": estimate,
                "regions_source": REGIONS_SOURCE,
                "structured_output": "JSON Object with strict local validation",
                "structured_output_source": STRUCTURED_OUTPUT_SOURCE,
            }
        )
    return result


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
                if key.lower() in {"authorization", "api_key", "apikey", "access_token"}
                else _redact(item, secret)
            )
            for key, item in value.items()
        }
    return value


def _request(config, payload, key):
    """One direct HTTPS request. No proxy, redirect, implicit retry or fallback."""
    parsed = urlsplit(_config(config)["endpoint"])
    connection = http.client.HTTPSConnection(parsed.hostname, 443, timeout=REVIEW_TIMEOUT)
    deadline = time.monotonic() + REVIEW_TIMEOUT
    chunks = bytearray()
    status = None
    try:
        connection.request(
            "POST",
            parsed.path + "/chat/completions",
            body=json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Hosted review request timed out.")
        connection.sock.settimeout(remaining)
        response = connection.getresponse()
        status = response.status
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Hosted review request timed out.")
            if response.fp is not None:
                response.fp.raw._sock.settimeout(remaining)
            chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
            if len(chunks) > MAX_RESPONSE_BYTES:
                raise ValueError("Hosted review response exceeds the 1 MiB limit.")
        raw = _redact(_json(bytes(chunks).decode("utf-8")), key)
        if status != 200:
            raise ProviderResponseError(
                f"Alibaba Cloud returned HTTP {status}; no retry or redirect was attempted.",
                raw_response=raw,
            )
        if not isinstance(raw, dict) or raw.get("error"):
            raise ProviderResponseError(
                "Alibaba Cloud returned an invalid response.", raw_response=raw
            )
        return raw
    except ProviderResponseError:
        raise
    except (OSError, ValueError, RecursionError, http.client.HTTPException) as exc:
        raise ProviderResponseError(
            _redact(str(exc), key),
            raw_response={
                "http_status": status,
                "body": _redact(
                    bytes(chunks[:MAX_RESPONSE_BYTES]).decode("utf-8", errors="replace"), key
                ),
                "truncated": len(chunks) > MAX_RESPONSE_BYTES,
            },
        ) from None
    finally:
        connection.close()


def _images(image, candidates):
    images = [base64.b64decode(_encode(image, 1024))]
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["box"]
        crop = image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
        images.append(base64.b64decode(_encode(crop, 320)))
    return images


def _usage_metadata(raw, estimate):
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return {"usage": None, "usage_cost_usd": None}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if any(type(value) is not int or value < 0 for value in (prompt_tokens, completion_tokens)):
        return {"usage": deepcopy(usage), "usage_cost_usd": None}
    price = (
        prompt_tokens * Decimal(str(estimate["input_usd_per_million"]))
        + completion_tokens * Decimal(str(estimate["output_usd_per_million"]))
    ) / 1_000_000
    return {
        "usage": deepcopy(usage),
        "usage_cost_usd": float(price),
        "usage_cost_basis": "Reported tokens at the recorded list prices; not an invoice.",
        "usage_exceeds_bound": (
            prompt_tokens > MAX_INPUT_TOKENS or completion_tokens > MAX_OUTPUT_TOKENS
        ),
    }


class AlibabaReviewer:
    def __init__(self, config=None, expected_images=None):
        self.config = _config(config)
        self.expected_images = None if expected_images is None else tuple(expected_images)
        status = provider_status(self.config)
        self.metadata = {
            **status,
            "prompt_version": PROMPT_VERSION,
            "settings": deepcopy(SETTINGS),
            "image_encoding": deepcopy(IMAGE_ENCODING),
            "pricing": conservative_estimate(self.config),
            "model_identity": "Provider model ID; immutable weights digest is unavailable.",
            "structured_output": "JSON Object with strict local validation",
        }
        if status["status"] != "ready":
            raise ProviderResponseError(status["reason"], metadata=self.metadata)

    def review(self, image: Image.Image, candidates: list, instructions: str = "") -> dict:
        ids = _validate_candidates(image, candidates)
        if not isinstance(instructions, str) or len(instructions) > 2000:
            raise ValueError("Instructions must contain at most 2000 characters.")
        images = _images(image, candidates)
        if self.expected_images is not None:
            if tuple(images) != self.expected_images:
                raise ValueError("The images no longer match the approved preview.")
            images = self.expected_images
        user_prompt = (
            "Candidates in crop order: "
            + json.dumps(
                [{"candidate_id": c["id"], "proposed_label": c["label"]} for c in candidates],
                ensure_ascii=False,
            )
            + "\nAdditional context (JSON string): "
            + json.dumps(instructions, ensure_ascii=False)
            + "\nRequired JSON schema: "
            + json.dumps(_schema(ids), ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = json.dumps(messages, ensure_ascii=False)
        raw = None
        key = _api_key()
        try:
            raw = _request(
                self.config,
                {
                    "model": self.config["model"],
                    **deepcopy(SETTINGS),
                    "messages": [
                        messages[0],
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                *[
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/jpeg;base64,"
                                            + base64.b64encode(data).decode("ascii")
                                        },
                                    }
                                    for data in images
                                ],
                            ],
                        },
                    ],
                },
                key,
            )
            self.metadata.update(
                connection_verified=True,
                reason="The provider responded to this hosted review request.",
            )
            # Keep token usage even when the model returned unusable content.
            self.metadata.update(_usage_metadata(raw, self.metadata["pricing"]))
            self.metadata["returned_model"] = raw.get("model")
            self.metadata["request_id"] = raw.get("id")
            if raw.get("model") != self.config["model"]:
                raise ValueError("The provider returned an unexpected model.")
            choices = raw.get("choices")
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], dict)
            ):
                raise ValueError("The provider must return exactly one completion.")
            choice = choices[0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Generation was interrupted or truncated.")
            message = choice.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("Invalid hosted response content.")
            result = _validate_review(_json(message["content"]), ids)
            return {
                **result,
                "raw_response": raw,
                "prompt": _redact(prompt, key),
                "metadata": deepcopy(self.metadata),
            }
        except ProviderResponseError as exc:
            raise ProviderResponseError(
                _redact(str(exc), key),
                raw_response=_redact(raw if raw is not None else exc.raw_response, key),
                metadata=_redact(self.metadata, key),
                prompt=_redact(prompt, key),
            ) from None
        except (ValueError, TypeError) as exc:
            raise ProviderResponseError(
                _redact(str(exc), key),
                raw_response=_redact(raw, key),
                metadata=_redact(self.metadata, key),
                prompt=_redact(prompt, key),
            ) from None
