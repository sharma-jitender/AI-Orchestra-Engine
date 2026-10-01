from __future__ import annotations

import time

import httpx

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter
from app.config import get_settings


class OpenAIAdapter(ModelAdapter):
    provider = "openai"

    def complete(self, *, api_model: str, prompt: str, timeout: float) -> AdapterResult:
        key = get_settings().openai_api_key
        if not key:
            raise AdapterError("OPENAI_API_KEY is not set")
        started = time.perf_counter()
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": api_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise AdapterError("openai timeout") from exc
        except httpx.HTTPError as exc:
            raise AdapterError(f"openai transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise AdapterError(f"openai {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choice = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AdapterResult(
            output=choice,
            tokens_in=int(usage.get("prompt_tokens") or max(8, len(prompt) // 4)),
            tokens_out=int(usage.get("completion_tokens") or max(8, len(choice) // 4)),
            latency_ms=latency_ms,
        )
