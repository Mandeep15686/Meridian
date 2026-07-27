"""Pydantic v2 request and response schemas for all API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ── Shared types ──────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str


# ── Submit ────────────────────────────────────────────────────────────────────

VALID_SCOPES = {"gdpr", "soc2", "iso27001", "sec_sp", "cfpb", "sec_sid", "eu_ai_act"}
VALID_FORMATS = {"pdf", "json", "markdown"}


class SubmitOptions(BaseModel):
    groundedness_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    max_synthesis_retries: int = Field(default=2, ge=0, le=5)
    include_regulatory_appendix: bool = True
    severity_filter: list[Literal["critical", "major", "minor"]] | None = None
    max_gaps_returned: int = Field(default=50, ge=1, le=200)


class SubmitRequest(BaseModel):
    """Parsed from multipart form fields (files handled separately)."""
    regulation_scope: list[str] = Field(..., min_length=1)
    webhook_url: str | None = None
    webhook_secret: str | None = None
    report_formats: list[str] = Field(default=["pdf", "json"])
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")
    options: SubmitOptions = Field(default_factory=SubmitOptions)

    @field_validator("regulation_scope")
    @classmethod
    def validate_scope(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_SCOPES
        if invalid:
            raise ValueError(f"Unknown regulation scope(s): {invalid}. Valid: {VALID_SCOPES}")
        return v

    @field_validator("report_formats")
    @classmethod
    def validate_formats(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_FORMATS
        if invalid:
            raise ValueError(f"Unknown report format(s): {invalid}")
        return v

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_https(cls, v: str | None) -> str | None:
        if v and not v.startswith("https://"):
            raise ValueError("webhook_url must use HTTPS")
        return v


class SubmitResponse(BaseModel):
    job_id: str
    status: str = "queued"
    submitted_at: datetime
    estimated_completion_seconds: int = 180
    poll_url: str
    report_url: str


# ── Status ────────────────────────────────────────────────────────────────────

class JobSummary(BaseModel):
    total_gaps: int
    by_severity: dict[str, int]
    by_framework: dict[str, int]
    groundedness_pass_rate: float


class JobError(BaseModel):
    code: str
    message: str
    stage: str
    retry_count: int = 0


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "complete", "failed", "cancelled"]
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: int | None = None
    current_stage: str | None = None
    stages_complete: list[str] = Field(default_factory=list)
    progress_pct: int | None = None
    summary: JobSummary | None = None
    report_url: str | None = None
    langsmith_trace_url: str | None = None
    error: JobError | None = None


# ── Report ────────────────────────────────────────────────────────────────────

class ComplianceScoreDetail(BaseModel):
    score: float
    gaps: int
    checks_performed: int


class GapResponse(BaseModel):
    gap_id: str
    severity: Literal["critical", "major", "minor"]
    framework: str
    regulatory_article: str
    regulatory_requirement: str
    regulatory_quote: str
    regulatory_chunk_id: str | None = None
    policy_reference: str | None = None
    policy_text: str | None = None
    gap_description: str
    severity_justification: str
    remediation: str
    confidence: float
    groundedness_score: float
    is_verified: bool = True
    is_uncertain: bool = False


class SubmittedFileInfo(BaseModel):
    filename: str
    modality: str
    size_bytes: int
    duration_seconds: int | None = None
    page_count: int | None = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    regulation: str
    article: str | None
    content: str
    similarity_score: float


class ModelMetadata(BaseModel):
    synthesis_model: str
    retrieval_model: str
    reranker_model: str
    ner_model: str
    asr_model: str
    pipeline_version: str


class ReportResponse(BaseModel):
    job_id: str
    generated_at: datetime
    regulation_scope: list[str]
    submitted_files: list[SubmittedFileInfo]
    executive_summary: str
    compliance_score: dict[str, ComplianceScoreDetail]
    gaps: list[GapResponse]
    model_metadata: ModelMetadata
    retrieved_chunks: list[RetrievedChunkResponse] | None = None


# ── Corpus status ─────────────────────────────────────────────────────────────

class CorpusInfo(BaseModel):
    id: str
    name: str
    jurisdiction: str
    version: str
    source_url: str | None
    document_count: int
    chunk_count: int
    last_ingested: datetime
    freshness_status: Literal["current", "stale", "unknown"]


class CorpusStatusResponse(BaseModel):
    corpora: list[CorpusInfo]
    total_chunks: int
    index_last_rebuilt: datetime | None


# ── Health ────────────────────────────────────────────────────────────────────

class DependencyStatus(BaseModel):
    database: Literal["healthy", "unhealthy"]
    redis: Literal["healthy", "unhealthy"]
    hf_api: Literal["healthy", "unhealthy"]
    anthropic_api: Literal["healthy", "unhealthy"]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime
    dependencies: DependencyStatus
    message: str | None = None
