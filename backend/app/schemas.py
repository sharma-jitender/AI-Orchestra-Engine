from typing import Literal

from pydantic import BaseModel, Field

TaskType = Literal["qa", "code", "summarization", "reasoning", "creative"]
Policy = Literal["routed", "baseline"]


class CompleteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    task_type: TaskType | None = None
    policy: Policy = "routed"
    force_model: str | None = None


class CompleteResponse(BaseModel):
    output: str
    task_type: str
    model: str
    provider: str
    policy: str
    tokens_in: int
    tokens_out: int
    est_cost: float
    latency_ms: int
    success: bool
    fallback_used: bool
    fallback_from: str | None
    quality_score: float
    ranked: list[str]


class EvalRunRequest(BaseModel):
    limit: int | None = None
    notes: str | None = None
