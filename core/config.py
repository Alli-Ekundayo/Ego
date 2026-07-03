from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Alibaba Cloud / Qwen (DashScope) ──────────────────────────────────────
    # DashScope international endpoint: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    DASHSCOPE_API_KEY: SecretStr | None = None
    QWEN_MODEL: str = "qwen-plus"           # default; override with qwen-max / qwen-turbo
    LLM_MODEL: str = "qwen-plus"            # fallback for existing code referencing LLM_MODEL

    # ── Embeddings & Storage ───────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TURBOVEC_STORAGE_DIR: str = "scratch/cache/turbovec"


settings = Settings()
