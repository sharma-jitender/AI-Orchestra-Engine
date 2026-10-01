from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres in docker-compose; SQLite if Docker/Postgres is not installed.
    database_url: str = "sqlite:///./orchestrator.sqlite"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    allow_mock_fallback: bool = True
    baseline_model: str = "mock-strong"
    models_config_path: Path = ROOT / "config" / "models.yaml"
    routing_config_path: Path = ROOT / "config" / "routing.yaml"
    eval_fixture_path: Path = ROOT / "fixtures" / "eval_prompts.json"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
