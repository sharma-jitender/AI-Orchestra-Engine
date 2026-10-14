from __future__ import annotations

import httpx

from app.config import get_settings
from app.registry import ModelSpec, RoutingConfig, load_models, load_routing


def provider_available(spec: ModelSpec) -> bool:
    settings = get_settings()
    if spec.provider == "groq":
        return bool(settings.groq_api_key)
    if spec.provider == "nvidia":
        return bool(settings.nvidia_api_key)
    if spec.provider == "ollama":
        try:
            response = httpx.get(
                f"{settings.ollama_base_url.rstrip('/')}/api/tags",
                timeout=2.0,
            )
            if response.status_code != 200:
                return False
            installed = response.json().get("models", [])
        except (httpx.HTTPError, ValueError, AttributeError):
            return False
        return any(
            model.get("name") == spec.api_model
            or model.get("model") == spec.api_model
            for model in installed
            if isinstance(model, dict)
        )
    return False


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rank_models(
    task_type: str, *, include_unavailable: bool = False
) -> list[tuple[ModelSpec, float]]:
    models = [
        model
        for model in load_models().values()
        if model.enabled and model.provider != "mock"
    ]
    routing = load_routing()
    if not include_unavailable:
        models = [model for model in models if provider_available(model)]
    if not models:
        return []
    costs = [m.input_cost_per_1k + m.output_cost_per_1k for m in models]
    lats = [float(m.typical_latency_ms) for m in models]
    cost_n = _normalize(costs)
    lat_n = _normalize(lats)
    scored: list[tuple[ModelSpec, float]] = []
    for spec, c, l in zip(models, cost_n, lat_n):
        quality = spec.strengths.get(task_type, 0.5)
        score = (
            routing.quality * quality
            - routing.cost * c
            - routing.latency * l
        )
        scored.append((spec, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def baseline_spec() -> ModelSpec:
    routing: RoutingConfig = load_routing()
    models = load_models()
    spec = models.get(routing.baseline_model)
    if spec is None:
        raise RuntimeError("baseline model missing from registry")
    return spec
