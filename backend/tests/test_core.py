from app.classifier import classify
from app.adapters.base import AdapterResult
from app.adapters.groq import GroqAdapter
from app.adapters.nvidia import NvidiaAdapter
from app.adapters.ollama import OllamaAdapter
from app.fallback import complete_with_fallback
from app.quality import quality_score
from app.registry import load_models
from app.router import provider_available, rank_models


def test_classifier_hint_and_rules():
    assert classify("hello", "qa") == "qa"
    assert classify("Write a Python function that sorts a list") == "code"
    assert classify("TL;DR this paragraph about databases") == "summarization"
    assert (
        classify("Think step by step: why is fallback cheaper on average?")
        == "reasoning"
    )
    assert classify("Write a haiku about logs") == "creative"
    assert classify("What is HTTP?") == "qa"


def test_rank_uses_only_configured_real_models(monkeypatch):
    monkeypatch.setattr("app.router.provider_available", lambda spec: True)
    ranked = [spec.id for spec, _ in rank_models("code")]
    assert set(ranked) == {
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3.2:1b",
        "nvidia-llama-3.1-8b",
        "nvidia-llama-3.3-70b",
        "nvidia-nemotron-70b",
    }


def test_groq_requires_api_key(monkeypatch):
    settings = type("Settings", (), {"groq_api_key": ""})()
    monkeypatch.setattr("app.router.get_settings", lambda: settings)
    groq_model = load_models()["llama-3.1-8b-instant"]

    assert not provider_available(groq_model)
    settings.groq_api_key = "configured"
    assert provider_available(groq_model)


def test_nvidia_requires_api_key(monkeypatch):
    settings = type("Settings", (), {"nvidia_api_key": ""})()
    monkeypatch.setattr("app.router.get_settings", lambda: settings)
    nvidia_model = load_models()["nvidia-llama-3.1-8b"]

    assert not provider_available(nvidia_model)
    settings.nvidia_api_key = "configured"
    assert provider_available(nvidia_model)


def test_nvidia_adapter_uses_openai_compatible_endpoint(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "A concise answer"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }

    request = {}

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(
        "app.adapters.nvidia.get_settings",
        lambda: type("Settings", (), {"nvidia_api_key": "test-key"})(),
    )
    monkeypatch.setattr("app.adapters.nvidia.httpx.post", post)

    result = NvidiaAdapter().complete(
        api_model="meta/llama-3.1-8b-instruct", prompt="Question", timeout=5
    )

    assert result.output == "A concise answer"
    assert request["url"] == (
        "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["model"] == "meta/llama-3.1-8b-instruct"


def test_ollama_requires_installed_model(monkeypatch):
    settings = type(
        "Settings",
        (),
        {
            "ollama_base_url": "http://localhost:11434",
            "groq_api_key": "",
        },
    )()
    monkeypatch.setattr("app.router.get_settings", lambda: settings)
    response = type(
        "Response",
        (),
        {
            "status_code": 200,
            "json": lambda self: {"models": [{"name": "llama3.2:1b"}]},
        },
    )()
    monkeypatch.setattr(
        "app.router.httpx.get", lambda *args, **kwargs: response
    )

    local_model = load_models()["llama3.2:1b"]
    groq_model = load_models()["llama-3.1-8b-instant"]
    assert provider_available(local_model)
    assert not provider_available(groq_model)


def test_quality_empty_is_zero():
    assert quality_score("q", "", "qa", 0.9) == 0.0
    assert quality_score("q", "```python\npass\n```", "code", 0.9) > 0.5


def test_fallback_continues_after_unexpected_adapter_error(monkeypatch):
    calls = []

    def call(spec, prompt):
        calls.append(spec.id)
        if len(calls) == 1:
            raise RuntimeError("provider response could not be parsed")
        return AdapterResult("usable answer", 8, 4, 25)

    monkeypatch.setattr("app.fallback._call", call)
    models = load_models()
    outcome = complete_with_fallback(
        "What is HTTP?",
        "qa",
        ranked=[models["llama-3.3-70b-versatile"], models["llama3.2:1b"]],
    )

    assert calls == ["llama-3.3-70b-versatile", "llama3.2:1b"]
    assert outcome.spec.id == "llama3.2:1b"
    assert outcome.fallback_used is True


def test_groq_retries_rate_limit_once(monkeypatch):
    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "rate limited"

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "def reverse(value): return value[::-1]"
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 12},
            }

    responses = iter([Response(429), Response(200)])
    requests = []
    urls = []
    sleeps = []

    def post(*args, **kwargs):
        urls.append(args[0])
        requests.append(kwargs)
        return next(responses)

    monkeypatch.setattr(
        "app.adapters.groq.get_settings",
        lambda: type("Settings", (), {"groq_api_key": "test-key"})(),
    )
    monkeypatch.setattr("app.adapters.groq.httpx.post", post)
    monkeypatch.setattr("app.adapters.groq.time.sleep", sleeps.append)

    result = GroqAdapter().complete(
        api_model="llama-3.1-8b-instant", prompt="Write reverse", timeout=5
    )

    assert result.output.startswith("def reverse")
    assert len(requests) == 2
    assert sleeps == [2]
    assert urls == [
        "https://api.groq.com/openai/v1/chat/completions",
        "https://api.groq.com/openai/v1/chat/completions",
    ]
    assert requests[0]["headers"]["Authorization"] == "Bearer test-key"
    assert requests[0]["json"]["max_tokens"] == 512


def test_ollama_uses_long_timeout_and_generation_limit(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {
                "message": {
                    "content": "def reverse(value): return value[::-1]"
                }
            }

    request = {}

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    settings = type(
        "Settings", (), {"ollama_base_url": "http://localhost:11434"}
    )()
    monkeypatch.setattr("app.adapters.ollama.get_settings", lambda: settings)
    monkeypatch.setattr("app.adapters.ollama.httpx.post", post)

    result = OllamaAdapter().complete(
        api_model="llama3.2:1b", prompt="Write reverse", timeout=1
    )

    assert result.output.startswith("def reverse")
    assert request["timeout"] == 120
    assert request["json"]["options"]["num_predict"] == 512
