from __future__ import annotations

import hashlib
import time

from app.adapters.base import AdapterError, AdapterResult, ModelAdapter
from app.classifier import classify


class MockAdapter(ModelAdapter):
    provider = "mock"

    def complete(self, *, api_model: str, prompt: str, timeout: float) -> AdapterResult:
        if "FAIL_ME" in prompt:
            raise AdapterError("mock forced failure")
        start = time.perf_counter()
        task = classify(prompt)
        seed = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16)
        bodies = {
            "qa": f"[{api_model}] Direct answer: {prompt.split('?')[0].strip()[:80]}. Confidence is demo-grade.",
            "code": f"[{api_model}]\n```python\n# demo implementation for: {prompt[:60]}\ndef solution():\n    return {seed % 97}\n```",
            "summarization": f"[{api_model}] Summary: {prompt[:140]}… → cheaper model is enough for this compression task.",
            "reasoning": f"[{api_model}] Step 1: restate the problem. Step 2: apply a simple invariant. Step 3: answer derived from prompt hash {seed % 10}.",
            "creative": f"[{api_model}] In a dim server room, a router chose a cheaper muse.\nIt still wrote: {prompt[:50]}…",
        }
        output = bodies.get(task, bodies["qa"])
        if api_model == "mock-strong":
            output += "\n[strong model extra check: answer reviewed]"
        tokens_in = max(8, len(prompt) // 4)
        tokens_out = max(16, len(output) // 4)
        # Deterministic fake latency so eval charts are stable.
        latency = {"mock-fast": 70, "mock-code": 160, "mock-strong": 380}.get(api_model, 120)
        elapsed = (time.perf_counter() - start) * 1000
        return AdapterResult(
            output=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=int(max(latency, elapsed)),
        )
