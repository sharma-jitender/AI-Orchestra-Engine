from __future__ import annotations

import time

import httpx

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter
from app.config import get_settings


class AnthropicAdapter(ModelAdapter):
    provider = "anthropic"

    def complete(self, *, api_model: str, prompt: str, timeout: float) -> AdapterResult:
        key = get_settings().anthropic_api_key
        if not key:
            raise AdapterError("ANTHROPIC_API_KEY is not set")
        started = time.perf_counter()
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": api_model,
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise AdapterError("anthropic timeout") from exc
        except httpx.HTTPError as original:
            raise AdapterError(f"anthropic transport error: {original}") from original
        if resp.status_code >= 400:
            raise AdapterError(f"anthropic {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        parts = data.get("content") or []
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        usage = data.get("usage") or {}
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AdapterResult(
            output=text,
            tokens_in=int(usage.get("input_tokens") or max(8, len(prompt) // 4)),
            tokens_out=int(usage.get("output_tokens") or max(8, len(text) // 4)),
            latency_ms=latency_ms,
        )
