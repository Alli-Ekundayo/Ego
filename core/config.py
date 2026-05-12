from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    GOOGLE_API_KEY: SecretStr = SecretStr("")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gemma-4-26b-a4b-it"


settings = Settings()
