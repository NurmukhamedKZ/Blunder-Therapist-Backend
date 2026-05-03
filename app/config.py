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

    @property
    def cors_origins_list(self) -> list[str]:
        """Return cors_origins as a list, handling potential string input from env."""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins

    # Logging
    log_level: str = "INFO"
    log_format: str = "console"


settings = Settings()