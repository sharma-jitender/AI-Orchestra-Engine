from app.adapters.anthropic import AnthropicAdapter
from app.adapters.groq import GroqAdapter
from app.adapters.mock import MockAdapter
from app.adapters.nvidia import NvidiaAdapter
from app.adapters.ollama import OllamaAdapter
from app.adapters.openai import OpenAIAdapter

ADAPTERS = {
    "mock": MockAdapter(),
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "groq": GroqAdapter(),
    "nvidia": NvidiaAdapter(),
    "ollama": OllamaAdapter(),
}
