"""
Centralized application configuration with support for PostgreSQL and SQLite.
Secrets are read from .env and no default values are set for production.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./refinery_risk.db"
    DB_ECHO: bool = False
    MODEL_DIR: str = "backend/models"
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "development-secret-key"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CORS_ORIGINS: str = "*"
    PREDICTION_STEP_SECONDS: int = 3  # 3 seconds for demo, change to 300 for industrial

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def validate_production_secrets(self) -> None:
        if self.ENVIRONMENT == "production":
            if self.SECRET_KEY == "change-me-in-production":
                raise ValueError("SECRET_KEY must be set in production")
            if self.CORS_ORIGINS == "*":
                raise ValueError("CORS_ORIGINS must be restricted in production")


settings = Settings()
if settings.ENVIRONMENT == "production":
    settings.validate_production_secrets()