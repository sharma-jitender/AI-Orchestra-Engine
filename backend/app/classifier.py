from __future__ import annotations

import re
from typing import Optional

from app.schemas import TaskType

_CODE = re.compile(
    r"\b(python|typescript|javascript|sql|regex|function|def |class |bug|compile|"
    r"fastapi|endpoint|linked list|binary search|implement|fix this)\b",
    re.I,
)
_SUM = re.compile(r"\b(summarize|summary|tldr|tl;dr|short summary)\b", re.I)
_REASON = re.compile(
    r"\b(step by step|prove|explain why|logic|how do you|are all|think step)\b",
    re.I,
)
_CREATIVE = re.compile(r"\b(poem|haiku|story|fictional|fable|invent|sci-fi|write a)\b", re.I)


def classify(prompt: str, hint: Optional[TaskType] = None) -> TaskType:
    if hint:
        return hint
    if _CODE.search(prompt):
        return "code"
    if _SUM.search(prompt):
        return "summarization"
    if _REASON.search(prompt):
        return "reasoning"
    if _CREATIVE.search(prompt):
        return "creative"
    return "qa"
