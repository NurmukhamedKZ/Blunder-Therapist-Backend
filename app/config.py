"""Application configuration loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    model_fast: str = "gpt-4o-mini"  # for Tilt Detector + DNA
    model_smart: str = "gpt-4o"  # for Coach chat (when context matters)

    # Database (Supabase Postgres connection string)
    database_url: str = "sqlite+aiosqlite:///./dev.db"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
