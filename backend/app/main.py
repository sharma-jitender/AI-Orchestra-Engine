from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db, init_db
from app.engine import handle_complete
from app.eval_runner import run_eval
from app.registry import load_models, load_routing
from app.router import rank_models
from app.schemas import CompleteRequest, EvalRunRequest
from app.stats import by_model, by_task, latest_eval, overview, recent_requests

app = FastAPI(title="LLM Orchestration Engine", version="0.1.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/complete")
def complete(body: CompleteRequest, db: Session = Depends(get_db)):
    return handle_complete(db, body)


@app.get("/v1/models")
def models():
    routing = load_routing()
    items = []
    for spec in load_models().values():
        items.append(
            {
                "id": spec.id,
                "provider": spec.provider,
                "display_name": spec.display_name,
                "input_cost_per_1k": spec.input_cost_per_1k,
                "output_cost_per_1k": spec.output_cost_per_1k,
                "typical_latency_ms": spec.typical_latency_ms,
                "strengths": spec.strengths,
                "enabled": spec.enabled,
            }
        )
    return {"baseline_model": routing.baseline_model, "weights": {
        "quality": routing.quality,
        "cost": routing.cost,
        "latency": routing.latency,
    }, "models": items}


@app.get("/v1/rank")
def rank(task_type: str = Query(default="qa")):
    ranked = [{"id": spec.id, "score": round(score, 4), "provider": spec.provider} for spec, score in rank_models(task_type)]
    return {"task_type": task_type, "ranked": ranked}


@app.get("/v1/stats/overview")
def stats_overview(eval_run_id: str | None = None, db: Session = Depends(get_db)):
    return overview(db, eval_run_id)


@app.get("/v1/stats/by-model")
def stats_models(db: Session = Depends(get_db)):
    return by_model(db)


@app.get("/v1/stats/by-task")
def stats_tasks(db: Session = Depends(get_db)):
    return by_task(db)


@app.get("/v1/requests")
def list_requests(limit: int = 50, db: Session = Depends(get_db)):
    return recent_requests(db, limit)


@app.post("/v1/eval/run")
def eval_run(body: EvalRunRequest | None = None, db: Session = Depends(get_db)):
    body = body or EvalRunRequest()
    return run_eval(db, limit=body.limit, notes=body.notes)


@app.get("/v1/eval/latest")
def eval_latest(db: Session = Depends(get_db)):
    return latest_eval(db) or {"id": None}
