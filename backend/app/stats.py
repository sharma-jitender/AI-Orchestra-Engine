from __future__ import annotations

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.db import EvalRun, RequestLog


def _safe_agg(db: Session, policy: str | None = None, eval_run_id: str | None = None) -> dict:
    q = db.query(
        func.count(RequestLog.id).label("n"),
        func.coalesce(func.sum(RequestLog.est_cost), 0.0).label("cost"),
        func.coalesce(func.avg(RequestLog.latency_ms), 0.0).label("avg_latency"),
        func.coalesce(func.avg(RequestLog.quality_score), 0.0).label("quality"),
        func.avg(case((RequestLog.success.is_(True), 1.0), else_=0.0)).label("success_rate"),
        func.avg(case((RequestLog.fallback_used.is_(True), 1.0), else_=0.0)).label("fallback_rate"),
    )
    if policy:
        q = q.filter(RequestLog.policy == policy)
    if eval_run_id:
        q = q.filter(RequestLog.eval_run_id == eval_run_id)
    row = q.one()
    n = int(row.n or 0)
    avg_lat = float(row.avg_latency or 0)
    return {
        "requests": n,
        "total_cost": round(float(row.cost or 0), 6),
        "avg_latency_ms": round(avg_lat, 1),
        "p50_latency_ms": round(avg_lat, 1),
        "avg_quality": round(float(row.quality or 0), 4),
        "success_rate": round(float(row.success_rate or 0), 4),
        "fallback_rate": round(float(row.fallback_rate or 0), 4),
    }


def overview(db: Session, eval_run_id: str | None = None) -> dict:
    routed = _safe_agg(db, "routed", eval_run_id)
    baseline = _safe_agg(db, "baseline", eval_run_id)
    savings = None
    if baseline["total_cost"] > 0:
        savings = round((baseline["total_cost"] - routed["total_cost"]) / baseline["total_cost"], 4)
    return {"routed": routed, "baseline": baseline, "cost_savings_vs_baseline": savings}


def by_model(db: Session) -> list[dict]:
    rows = (
        db.query(
            RequestLog.model,
            RequestLog.provider,
            func.count(RequestLog.id),
            func.coalesce(func.sum(RequestLog.est_cost), 0.0),
            func.coalesce(func.avg(RequestLog.latency_ms), 0.0),
            func.coalesce(func.avg(RequestLog.quality_score), 0.0),
            func.avg(case((RequestLog.success.is_(True), 1.0), else_=0.0)),
        )
        .group_by(RequestLog.model, RequestLog.provider)
        .all()
    )
    return [
        {
            "model": r[0],
            "provider": r[1],
            "requests": int(r[2]),
            "total_cost": round(float(r[3]), 6),
            "avg_latency_ms": round(float(r[4]), 1),
            "avg_quality": round(float(r[5]), 4),
            "success_rate": round(float(r[6] or 0), 4),
        }
        for r in rows
    ]


def by_task(db: Session) -> list[dict]:
    rows = (
        db.query(
            RequestLog.task_type,
            RequestLog.policy,
            func.count(RequestLog.id),
            func.coalesce(func.sum(RequestLog.est_cost), 0.0),
            func.coalesce(func.avg(RequestLog.latency_ms), 0.0),
            func.coalesce(func.avg(RequestLog.quality_score), 0.0),
        )
        .group_by(RequestLog.task_type, RequestLog.policy)
        .all()
    )
    return [
        {
            "task_type": r[0],
            "policy": r[1],
            "requests": int(r[2]),
            "total_cost": round(float(r[3]), 6),
            "avg_latency_ms": round(float(r[4]), 1),
            "avg_quality": round(float(r[5]), 4),
        }
        for r in rows
    ]


def recent_requests(db: Session, limit: int = 50) -> list[dict]:
    rows = db.query(RequestLog).order_by(RequestLog.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "task_type": r.task_type,
            "policy": r.policy,
            "model": r.model,
            "provider": r.provider,
            "est_cost": r.est_cost,
            "latency_ms": r.latency_ms,
            "success": r.success,
            "fallback_used": r.fallback_used,
            "quality_score": r.quality_score,
            "prompt": r.prompt[:180],
            "response_preview": (r.response_preview or "")[:180],
            "eval_run_id": r.eval_run_id,
        }
        for r in rows
    ]


def latest_eval(db: Session) -> dict | None:
    run = db.query(EvalRun).order_by(EvalRun.created_at.desc()).first()
    if not run:
        return None
    return {
        "id": run.id,
        "created_at": run.created_at.isoformat(),
        "prompt_count": run.prompt_count,
        **overview(db, run.id),
    }
