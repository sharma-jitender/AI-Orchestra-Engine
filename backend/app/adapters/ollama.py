from __future__ import annotations

import time

import httpx

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter
from app.config import get_settings


class OllamaAdapter(ModelAdapter):
    provider = "ollama"

    def complete(
        self, *, api_model: str, prompt: str, timeout: float
    ) -> AdapterResult:
        base = get_settings().ollama_base_url.rstrip("/")
        started = time.perf_counter()
        try:
            resp = httpx.post(
                f"{base}/api/chat",
                json={
                    "model": api_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": 512},
                },
                timeout=120,
            )
        except httpx.TimeoutException as exc:
            raise AdapterError("ollama timeout") from exc
        except httpx.HTTPError as exc:
            raise AdapterError(f"ollama transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise AdapterError(f"ollama {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        text = ((data.get("message") or {}).get("content")) or ""
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AdapterResult(
            output=text,
            tokens_in=int(
                data.get("prompt_eval_count") or max(8, len(prompt) // 4)
            ),
            tokens_out=int(data.get("eval_count") or max(8, len(text) // 4)),
            latency_ms=latency_ms,
        )
