from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Google Gemini ──────────────────────────────────────────────────────────
    GOOGLE_API_KEY: SecretStr = SecretStr("")
    LLM_MODEL: str = "gemini-flash-latest"

    # ── Alibaba Cloud / Qwen (DashScope) ──────────────────────────────────────
    # DashScope international endpoint: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    # Set DASHSCOPE_API_KEY to enable Qwen-powered memory consolidation.
    # If absent the system falls back to Gemini transparently.
    DASHSCOPE_API_KEY: SecretStr | None = None
    QWEN_MODEL: str = "qwen-plus"           # default; override with qwen-max / qwen-turbo

    # ── Embeddings & Storage ───────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TURBOVEC_STORAGE_DIR: str = "scratch/cache/turbovec"


settings = Settings()
