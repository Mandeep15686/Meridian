"""Central configuration — all settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    VERSION: str = "1.0.3"
    APP_NAME: str = "meridian"

    # ── Required credentials ──────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    ANTHROPIC_API_KEY: str = Field(..., description="Anthropic Claude API key")
    HF_API_TOKEN: str = Field(..., description="HuggingFace Inference API token")
    DATABASE_URL: str = Field(..., description="PostgreSQL connection string")
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    MERIDIAN_API_KEY: str = Field(..., min_length=32, description="Bearer token for API auth")

    # ── Vector store ─────────────────────────────────────────────────────────
    VECTOR_STORE: Literal["pgvector", "pinecone"] = "pgvector"
    PINECONE_API_KEY: str | None = None
    PINECONE_INDEX_NAME: str = "meridian"
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    IVFFLAT_PROBES: int = Field(default=10, ge=1, le=100)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K_DENSE: int = Field(default=20, ge=1, le=100)
    RETRIEVAL_TOP_K_BM25: int = Field(default=20, ge=1, le=100)
    RETRIEVAL_TOP_K_RERANK: int = Field(default=5, ge=1, le=20)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    RETRIEVAL_CACHE_TTL: int = Field(default=3600, ge=0)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    GROUNDEDNESS_THRESHOLD: float = Field(default=0.80, ge=0.0, le=1.0)
    MAX_SYNTHESIS_RETRIES: int = Field(default=2, ge=0, le=5)
    SYNTHESIS_MODEL: str = "claude-sonnet-4-6"
    SYNTHESIS_MAX_TOKENS: int = Field(default=4096, ge=256, le=8192)
    MAX_DOCUMENT_TOKENS: int = Field(default=100_000, ge=1000)
    CHUNK_SIZE_TARGET: int = Field(default=512, ge=64, le=2048)
    CHUNK_OVERLAP: int = Field(default=64, ge=0, le=256)

    # ── Audio ─────────────────────────────────────────────────────────────────
    WHISPER_INITIAL_PROMPT: str = (
        "GDPR, CFPB, SOC-2, data controller, data subject, "
        "lawful basis, DPO, retention period, HIPAA, regulation"
    )
    AUDIO_CHUNK_DURATION_S: int = 30
    AUDIO_PARALLEL_CHUNKS: int = Field(default=2, ge=1, le=8)

    # ── Job management ────────────────────────────────────────────────────────
    JOB_TTL_DAYS: int = Field(default=7, ge=1, le=90)
    MAX_FILE_SIZE_MB: int = Field(default=500, ge=1, le=2000)
    MAX_FILES_PER_JOB: int = Field(default=10, ge=1, le=50)
    CELERY_CONCURRENCY: int = Field(default=4, ge=1, le=32)
    CELERY_TASK_TIMEOUT: int = Field(default=900, ge=60)
    RATE_LIMIT_SUBMIT: int = Field(default=20, ge=1, le=1000)
    WEBHOOK_RETRY_DELAYS: str = "30,90,270"
    WEBHOOK_SECRET: str | None = None

    # ── Storage ───────────────────────────────────────────────────────────────
    STORAGE_BACKEND: Literal["local", "gcs"] = "local"
    LOCAL_STORAGE_PATH: Path = Path("/tmp/meridian")
    GCS_UPLOADS_BUCKET: str | None = None
    GCS_REPORTS_BUCKET: str | None = None

    # ── LLMOps ───────────────────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str = "meridian"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "meridian-eval"
    WANDB_API_KEY: str | None = None
    WANDB_PROJECT: str = "meridian"
    SENTRY_DSN: str | None = None

    # ── Corpus ────────────────────────────────────────────────────────────────
    CORPUS_SOURCES: str = "gdpr,soc2,iso27001,sec_sp,cfpb"
    CORPUS_REFRESH_SCHEDULE: str = "0 2 * * *"
    CORPUS_MAX_STALENESS_DAYS: int = 30
    EDGAR_USER_AGENT: str = "Meridian/1.0 contact@example.com"

    # ── Application ───────────────────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"
    WORKERS: int = Field(default=2, ge=1, le=16)
    PORT: int = Field(default=8000, ge=1024, le=65535)
    ALLOWED_ORIGINS: str = "*"
    DB_POOL_SIZE: int = Field(default=5, ge=1, le=50)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0, le=50)

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def active_corpora(self) -> list[str]:
        return [s.strip() for s in self.CORPUS_SOURCES.split(",") if s.strip()]

    @property
    def webhook_retry_delays_list(self) -> list[int]:
        return [int(d.strip()) for d in self.WEBHOOK_RETRY_DELAYS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql"):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection string")
        return v

    @model_validator(mode="after")
    def validate_pinecone_config(self) -> "Settings":
        if self.VECTOR_STORE == "pinecone" and not self.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required when VECTOR_STORE=pinecone")
        return self

    @model_validator(mode="after")
    def validate_gcs_config(self) -> "Settings":
        if self.STORAGE_BACKEND == "gcs":
            if not self.GCS_UPLOADS_BUCKET or not self.GCS_REPORTS_BUCKET:
                raise ValueError(
                    "GCS_UPLOADS_BUCKET and GCS_REPORTS_BUCKET are required "
                    "when STORAGE_BACKEND=gcs"
                )
        return self

    @model_validator(mode="after")
    def set_log_format_for_production(self) -> "Settings":
        if self.ENVIRONMENT == "production":
            object.__setattr__(self, "LOG_FORMAT", "json")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance. Call once at startup."""
    return Settings()  # type: ignore[call-arg]


# Module-level convenience alias
settings = get_settings()
