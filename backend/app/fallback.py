from __future__ import annotations

from dataclasses import dataclass

from app.adapters import ADAPTERS
from app.adapters.base import AdapterError, AdapterResult
from app.config import get_settings
from app.quality import looks_low_confidence
from app.registry import ModelSpec, estimate_cost, load_routing
from app.router import rank_models


@dataclass
class CompletionOutcome:
    spec: ModelSpec
    result: AdapterResult
    est_cost: float
    fallback_used: bool
    fallback_from: str | None
    ranked_ids: list[str]
    error: str | None = None
    success: bool = True


def _call(spec: ModelSpec, prompt: str) -> AdapterResult:
    adapter = ADAPTERS.get(spec.provider)
    if adapter is None:
        raise AdapterError(f"no adapter for provider {spec.provider}")
    timeout = load_routing().timeout_seconds
    return adapter.complete(api_model=spec.api_model, prompt=prompt, timeout=timeout)


def complete_with_fallback(
    prompt: str,
    task_type: str,
    *,
    force_model: str | None = None,
    ranked: list[ModelSpec] | None = None,
) -> CompletionOutcome:
    routing = load_routing()
    settings = get_settings()
    if ranked is None:
        ranked = [spec for spec, _ in rank_models(task_type)]
    if force_model:
        from app.registry import load_models

        catalog = load_models()
        if force_model in catalog:
            forced = catalog[force_model]
            ranked = [forced] + [m for m in ranked if m.id != force_model]
    ranked_ids = [m.id for m in ranked]
    attempts = ranked[: routing.max_fallback_attempts]
    last_error = "no models available"
    first_id = attempts[0].id if attempts else None
    for i, spec in enumerate(attempts):
        try:
            result = _call(spec, prompt)
            if looks_low_confidence(result.output):
                last_error = "low-confidence output"
                continue
            return CompletionOutcome(
                spec=spec,
                result=result,
                est_cost=estimate_cost(spec, result.tokens_in, result.tokens_out),
                fallback_used=i > 0,
                fallback_from=first_id if i > 0 else None,
                ranked_ids=ranked_ids,
            )
        except AdapterError as exc:
            last_error = str(exc)
            continue
    if settings.allow_mock_fallback and (not attempts or attempts[0].provider != "mock"):
        from app.registry import load_models

        mock = next(m for m in load_models().values() if m.provider == "mock" and m.enabled)
        result = _call(mock, prompt)
        return CompletionOutcome(
            spec=mock,
            result=result,
            est_cost=estimate_cost(mock, result.tokens_in, result.tokens_out),
            fallback_used=True,
            fallback_from=first_id,
            ranked_ids=ranked_ids,
        )
    empty = AdapterResult(output="", tokens_in=0, tokens_out=0, latency_ms=0)
    from app.registry import load_models

    failed_spec = attempts[0] if attempts else next(iter(load_models().values()))
    return CompletionOutcome(
        spec=failed_spec,
        result=empty,
        est_cost=0.0,
        fallback_used=len(attempts) > 1,
        fallback_from=first_id,
        ranked_ids=ranked_ids,
        error=last_error,
        success=False,
    )
