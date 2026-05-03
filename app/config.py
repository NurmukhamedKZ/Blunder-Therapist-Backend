"""Application configuration loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    model_fast: str = "gpt-4o-mini"
    model_smart: str = "gpt-5.4-nano"

    # Database
    database_url: str = "sqlite+aiosqlite:///./dev.db"

    # Supabase Auth
    supabase_url: str
    supabase_jwt_secret: str

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Logging
    log_level: str = "INFO"
    log_format: str = "console"


settings = Settings()