from __future__ import annotations

import re

_LOW = re.compile(r"\b(i don'?t know|as an ai|cannot help with that)\b", re.I)


def quality_score(prompt: str, output: str, task_type: str, strength: float) -> float:
    if not output.strip():
        return 0.0
    score = 0.45 + 0.4 * strength
    if _LOW.search(output):
        score -= 0.25
    n = len(output.strip())
    if task_type == "summarization" and 40 <= n <= 600:
        score += 0.1
    if task_type == "code" and ("```" in output or "def " in output or "SELECT" in output.upper()):
        score += 0.12
    if task_type == "creative" and n >= 40:
        score += 0.08
    if task_type in {"qa", "reasoning"} and n >= 20:
        score += 0.05
    if n < 8:
        score -= 0.2
    return max(0.0, min(1.0, score))


def looks_low_confidence(output: str) -> bool:
    return (not output.strip()) or bool(_LOW.search(output))
