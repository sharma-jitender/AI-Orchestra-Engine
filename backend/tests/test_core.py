from app.classifier import classify
from app.quality import quality_score
from app.router import rank_models


def test_classifier_hint_and_rules():
    assert classify("hello", "qa") == "qa"
    assert classify("Write a Python function that sorts a list") == "code"
    assert classify("TL;DR this paragraph about databases") == "summarization"
    assert classify("Think step by step: why is fallback cheaper on average?") == "reasoning"
    assert classify("Write a haiku about logs") == "creative"
    assert classify("What is HTTP?") == "qa"


def test_rank_prefers_specialist_for_code():
    ranked = [spec.id for spec, _ in rank_models("code")]
    assert ranked[0] in {"mock-code", "gpt-4o-mini", "gpt-4o"}
    assert "mock-code" in ranked


def test_quality_empty_is_zero():
    assert quality_score("q", "", "qa", 0.9) == 0.0
    assert quality_score("q", "```python\npass\n```", "code", 0.9) > 0.5
