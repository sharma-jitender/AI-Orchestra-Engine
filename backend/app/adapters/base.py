from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdapterResult:
    output: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


class AdapterError(Exception):
    pass


class ModelAdapter:
    provider: str

    def complete(self, *, api_model: str, prompt: str, timeout: float) -> AdapterResult:
        raise NotImplementedError
