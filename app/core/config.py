"""
Application configuration, loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "AI Health Risk Prediction & Telemedicine API"

    # MySQL connection, e.g. mysql+pymysql://user:password@localhost:3306/health_db
    DATABASE_URL: str = "mysql+pymysql://root:password@localhost:3306/health_db"

    # JWT auth
    JWT_SECRET_KEY: str = "change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    PASSWORD_RESET_EXPIRE_MINUTES: int = 30

    # CORS - the Next.js frontend origin(s)
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "https://your-frontend.vercel.app",
    ]

    # Path to the directory holding the three serialized model files
    ML_MODELS_DIR: str = "app/ml_models"


settings = Settings()
