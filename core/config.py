from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GOOGLE_API_KEY: SecretStr = SecretStr("")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gemini-flash-latest"
    TURBOVEC_STORAGE_DIR: str = "scratch/cache/turbovec"


settings = Settings()
