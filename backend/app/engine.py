from __future__ import annotations

from sqlalchemy.orm import Session

from app.classifier import classify
from app.db import RequestLog
from app.fallback import complete_with_fallback
from app.quality import quality_score
from app.registry import load_models
from app.router import baseline_spec, rank_models
from app.schemas import CompleteRequest, CompleteResponse


def handle_complete(db: Session, body: CompleteRequest, *, eval_run_id: str | None = None) -> CompleteResponse:
    task_type = classify(body.prompt, body.task_type)
    ranked = [spec for spec, _ in rank_models(task_type)]
    force = body.force_model
    catalog = load_models()
    if force and force not in catalog:
        force = None
    if body.policy == "baseline" and not force:
        force = baseline_spec().id
        ranked = [catalog[force]] + [m for m in ranked if m.id != force]
    outcome = complete_with_fallback(
        body.prompt,
        task_type,
        force_model=force,
        ranked=ranked,
    )
    q = (
        quality_score(
            body.prompt,
            outcome.result.output,
            task_type,
            outcome.spec.strengths.get(task_type, 0.5),
        )
        if outcome.success
        else 0.0
    )
    preview = outcome.result.output[:500]
    row = RequestLog(
        prompt=body.prompt,
        task_type=task_type,
        policy=body.policy,
        model=outcome.spec.id,
        provider=outcome.spec.provider,
        tokens_in=outcome.result.tokens_in,
        tokens_out=outcome.result.tokens_out,
        est_cost=outcome.est_cost,
        latency_ms=outcome.result.latency_ms,
        success=outcome.success,
        fallback_used=outcome.fallback_used,
        fallback_from=outcome.fallback_from,
        quality_score=q,
        error=outcome.error,
        response_preview=preview,
        eval_run_id=eval_run_id,
    )
    db.add(row)
    db.commit()
    return CompleteResponse(
        output=outcome.result.output,
        task_type=task_type,
        model=outcome.spec.id,
        provider=outcome.spec.provider,
        policy=body.policy,
        tokens_in=outcome.result.tokens_in,
        tokens_out=outcome.result.tokens_out,
        est_cost=round(outcome.est_cost, 6),
        latency_ms=outcome.result.latency_ms,
        success=outcome.success,
        fallback_used=outcome.fallback_used,
        fallback_from=outcome.fallback_from,
        quality_score=round(q, 4),
        ranked=outcome.ranked_ids,
    )
