from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    display_name: str
    api_model: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    typical_latency_ms: int
    strengths: dict[str, float]
    enabled: bool = True


@dataclass(frozen=True)
class RoutingConfig:
    quality: float
    cost: float
    latency: float
    baseline_model: str
    prefer_mock_when_no_keys: bool
    timeout_seconds: float
    max_fallback_attempts: int


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def load_models() -> dict[str, ModelSpec]:
    settings = get_settings()
    raw = _load_yaml(settings.models_config_path)
    models: dict[str, ModelSpec] = {}
    for item in raw["models"]:
        spec = ModelSpec(
            id=item["id"],
            provider=item["provider"],
            display_name=item["display_name"],
            api_model=item.get("api_model", item["id"]),
            input_cost_per_1k=float(item["input_cost_per_1k"]),
            output_cost_per_1k=float(item["output_cost_per_1k"]),
            typical_latency_ms=int(item["typical_latency_ms"]),
            strengths={k: float(v) for k, v in item["strengths"].items()},
            enabled=bool(item.get("enabled", True)),
        )
        models[spec.id] = spec
    return models


@lru_cache
def load_routing() -> RoutingConfig:
    settings = get_settings()
    raw = _load_yaml(settings.routing_config_path)
    w = raw["weights"]
    baseline = raw.get("baseline_model") or settings.baseline_model
    return RoutingConfig(
        quality=float(w["quality"]),
        cost=float(w["cost"]),
        latency=float(w["latency"]),
        baseline_model=baseline,
        prefer_mock_when_no_keys=bool(raw.get("prefer_mock_when_no_keys", True)),
        timeout_seconds=float(raw.get("timeout_seconds", 20)),
        max_fallback_attempts=int(raw.get("max_fallback_attempts", 3)),
    )


def estimate_cost(spec: ModelSpec, tokens_in: int, tokens_out: int) -> float:
    return (tokens_in / 1000.0) * spec.input_cost_per_1k + (tokens_out / 1000.0) * spec.output_cost_per_1k
