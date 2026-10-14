from __future__ import annotations

import json
import time
import uuid

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import EvalRun
from app.engine import handle_complete
from app.schemas import CompleteRequest
from app.stats import overview


def run_eval(
    db: Session, limit: int | None = None, notes: str | None = None
) -> dict:
    path = get_settings().eval_fixture_path
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        prompts = prompts[:limit]
    run_id = str(uuid.uuid4())
    db.add(EvalRun(id=run_id, prompt_count=len(prompts), notes=notes))
    db.commit()
    call_count = len(prompts) * 2
    call_index = 0
    for item in prompts:
        hint = item.get("task_type")
        prompt = item["prompt"]
        for policy in ("baseline", "routed"):
            handle_complete(
                db,
                CompleteRequest(prompt=prompt, task_type=hint, policy=policy),
                eval_run_id=run_id,
            )
            call_index += 1
            if call_index < call_count:
                time.sleep(1)
    return {"id": run_id, "prompt_count": len(prompts), **overview(db, run_id)}
