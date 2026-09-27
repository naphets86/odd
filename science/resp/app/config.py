from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}

    DATABASE_URL: str = "postgresql+asyncpg://resp_user:resp_pass@localhost:5432/resp_db"
    API_PREFIX: str = "/api"


settings = Settings()
