"""Available annotation providers without downloads or remote availability probes."""

from iris import assistance_provider as local
from iris import remote_provider as remote


def catalog() -> dict:
    default = local.ProviderConfig.from_env()
    models = [default.model]
    try:
        config = local._config(default)
        tags = local._request(config, "GET", "/api/tags").get("models", [])
        for row in tags[:32]:
            if isinstance(row, dict) and not local._remote(row):
                name = row.get("name")
                if isinstance(name, str) and name not in models:
                    models.append(name)
    except (ValueError, OSError, local.ProviderResponseError):
        pass
    local_models = []
    for name in models:
        status = local.provider_status({"endpoint": default.endpoint, "model": name})
        local_models.append({**status, "id": name, "label": name})
    remote_models = []
    for profile in remote.catalog():
        remote_models.append(
            {
                **profile,
                "id": profile["model"],
                "label": profile["name"],
                "status": "configured" if profile["status"] == "ready" else profile["status"],
            }
        )
    return {
        "default_provider": "ollama",
        "default_model": default.model,
        "providers": [
            {"id": "ollama", "name": "Local · Ollama", "local": True, "models": local_models},
            {
                "id": "alibaba",
                "name": "API · Alibaba Cloud Model Studio",
                "local": False,
                "models": remote_models,
            },
        ],
    }
