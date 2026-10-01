from __future__ import annotations

from app.config import get_settings
from app.registry import ModelSpec, RoutingConfig, load_models, load_routing


def provider_available(provider: str) -> bool:
    settings = get_settings()
    if provider == "mock":
        return True
    if provider == "openai":
        return bool(settings.openai_api_key)
    if provider == "anthropic":
        return bool(settings.anthropic_api_key)
    if provider == "ollama":
        return True
    return False


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rank_models(task_type: str, *, include_unavailable: bool = False) -> list[tuple[ModelSpec, float]]:
    models = [m for m in load_models().values() if m.enabled]
    routing = load_routing()
    settings = get_settings()
    if routing.prefer_mock_when_no_keys and not settings.openai_api_key and not settings.anthropic_api_key:
        models = [m for m in models if m.provider == "mock"] or models
    if not include_unavailable:
        models = [m for m in models if provider_available(m.provider)]
    if not models:
        models = [m for m in load_models().values() if m.provider == "mock" and m.enabled]
    costs = [m.input_cost_per_1k + m.output_cost_per_1k for m in models]
    lats = [float(m.typical_latency_ms) for m in models]
    cost_n = _normalize(costs)
    lat_n = _normalize(lats)
    scored: list[tuple[ModelSpec, float]] = []
    for spec, c, l in zip(models, cost_n, lat_n):
        quality = spec.strengths.get(task_type, 0.5)
        score = routing.quality * quality - routing.cost * c - routing.latency * l
        scored.append((spec, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def baseline_spec() -> ModelSpec:
    routing: RoutingConfig = load_routing()
    models = load_models()
    spec = models.get(routing.baseline_model) or models.get("mock-strong")
    if spec is None:
        raise RuntimeError("baseline model missing from registry")
    return spec
