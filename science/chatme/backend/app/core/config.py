from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://chatme:chatme_secret@db:5432/chatme"
    REDIS_URL: str = "redis://redis:6379"
    SECRET_KEY: str = "change_me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    CORS_ORIGINS: str = "*"

    TURN_REALM: str = "chatme.local"
    TURN_USER: str = "chatme"
    TURN_PASSWORD: str = "turnpassword"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
