"""SQLAlchemy ORM models for all Meridian database tables."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────────


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileModality(enum.StrEnum):
    DOCUMENT = "document"
    AUDIO = "audio"
    IMAGE = "image"
    TABULAR = "tabular"
    UNKNOWN = "unknown"


class AgentType(enum.StrEnum):
    DOC_AGENT = "doc_agent"
    AUDIO_AGENT = "audio_agent"
    VISION_AGENT = "vision_agent"
    DATA_AGENT = "data_agent"


class GapSeverity(enum.StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class ReportFormat(enum.StrEnum):
    JSON = "json"
    PDF = "pdf"
    MARKDOWN = "markdown"


class EvalType(enum.StrEnum):
    RAGAS = "ragas"
    GAP_DETECTION_F1 = "gap_detection_f1"
    AGENT_JUDGE = "agent_judge"
    LATENCY = "latency"


# ── Tables ────────────────────────────────────────────────────────────────────


class Corpus(Base):
    __tablename__ = "corpora"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    jurisdiction: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_refreshed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    documents: Mapped[list[Document]] = relationship("Document", back_populates="corpus")
    chunks: Mapped[list[Chunk]] = relationship("Chunk", back_populates="corpus")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    corpus_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("corpora.id", ondelete="CASCADE")
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(8), default="en")
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    storage_key: Mapped[str | None] = mapped_column(Text)

    corpus: Mapped[Corpus] = relationship("Corpus", back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship("Chunk", back_populates="document")

    __table_args__ = (UniqueConstraint("corpus_id", "content_hash"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE")
    )
    corpus_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("corpora.id"))
    regulation: Mapped[str] = mapped_column(String(64), nullable=False)
    article: Mapped[str | None] = mapped_column(String(128))
    article_title: Mapped[str | None] = mapped_column(Text)
    jurisdiction: Mapped[str] = mapped_column(String(32), nullable=False)
    effective_date: Mapped[Any | None] = mapped_column(Date)
    section_path: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Any | None] = mapped_column(Vector(1536))
    ts_vector: Mapped[Any | None] = mapped_column(TSVECTOR)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship("Document", back_populates="chunks")
    corpus: Mapped[Corpus] = relationship("Corpus", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        Index(
            "chunks_embedding_idx",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 200},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("chunks_ts_vector_idx", "ts_vector", postgresql_using="gin"),
        Index("chunks_corpus_regulation_idx", "corpus_id", "regulation"),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)  # ULID
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"), default=JobStatus.QUEUED
    )
    regulation_scope: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    report_formats: Mapped[list[str]] = mapped_column(ARRAY(Text), default=["pdf", "json"])
    language: Mapped[str] = mapped_column(String(8), default="en")
    options: Mapped[dict] = mapped_column(JSONB, default=dict)

    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    current_stage: Mapped[str | None] = mapped_column(String(64))
    stages_complete: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)

    webhook_url: Mapped[str | None] = mapped_column(Text)
    webhook_delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    webhook_attempts: Mapped[int] = mapped_column(Integer, default=0)

    total_gaps: Mapped[int | None] = mapped_column(Integer)
    gaps_critical: Mapped[int | None] = mapped_column(Integer)
    gaps_major: Mapped[int | None] = mapped_column(Integer)
    gaps_minor: Mapped[int | None] = mapped_column(Integer)
    groundedness_pass_rate: Mapped[float | None] = mapped_column(Numeric(4, 3))

    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    error_stage: Mapped[str | None] = mapped_column(String(64))
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    langsmith_run_id: Mapped[str | None] = mapped_column(Text)
    langsmith_trace_url: Mapped[str | None] = mapped_column(Text)

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    files: Mapped[list[JobFile]] = relationship("JobFile", back_populates="job")
    extractions: Mapped[list[AgentExtraction]] = relationship(
        "AgentExtraction", back_populates="job"
    )
    gaps: Mapped[list[ComplianceGap]] = relationship("ComplianceGap", back_populates="job")
    reports: Mapped[list[Report]] = relationship("Report", back_populates="job")

    __table_args__ = (
        Index("jobs_status_submitted_idx", "status", "submitted_at"),
        Index("jobs_expires_idx", "expires_at"),
    )


class JobFile(Base):
    __tablename__ = "job_files"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    job_id: Mapped[str] = mapped_column(String(26), ForeignKey("jobs.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    modality: Mapped[FileModality] = mapped_column(
        Enum(FileModality, name="file_modality"), default=FileModality.UNKNOWN
    )
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    page_count: Mapped[int | None] = mapped_column(Integer)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_error: Mapped[str | None] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    job: Mapped[Job] = relationship("Job", back_populates="files")

    __table_args__ = (Index("job_files_job_idx", "job_id"),)


class AgentExtraction(Base):
    __tablename__ = "agent_extractions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    job_id: Mapped[str] = mapped_column(String(26), ForeignKey("jobs.id", ondelete="CASCADE"))
    file_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("job_files.id"))
    agent: Mapped[AgentType] = mapped_column(Enum(AgentType, name="agent_type"))
    raw_text: Mapped[str | None] = mapped_column(Text)
    ner_entities: Mapped[list | None] = mapped_column(JSONB)
    qa_results: Mapped[list | None] = mapped_column(JSONB)
    summary: Mapped[str | None] = mapped_column(Text)
    transcript: Mapped[list | None] = mapped_column(JSONB)
    speakers: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    image_caption: Mapped[str | None] = mapped_column(Text)
    vqa_results: Mapped[list | None] = mapped_column(JSONB)
    colpali_matches: Mapped[list | None] = mapped_column(JSONB)
    table_summary: Mapped[str | None] = mapped_column(Text)
    tapas_answers: Mapped[list | None] = mapped_column(JSONB)
    anomaly_scores: Mapped[list | None] = mapped_column(JSONB)
    forecast_output: Mapped[list | None] = mapped_column(JSONB)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[Job] = relationship("Job", back_populates="extractions")

    __table_args__ = (Index("agent_extractions_job_agent_idx", "job_id", "agent"),)


class ComplianceGap(Base):
    __tablename__ = "compliance_gaps"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    job_id: Mapped[str] = mapped_column(String(26), ForeignKey("jobs.id", ondelete="CASCADE"))
    severity: Mapped[GapSeverity] = mapped_column(Enum(GapSeverity, name="gap_severity"))
    framework: Mapped[str] = mapped_column(String(64), nullable=False)
    regulatory_article: Mapped[str] = mapped_column(String(256), nullable=False)
    regulatory_chunk_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chunks.id")
    )
    regulatory_requirement: Mapped[str] = mapped_column(Text, nullable=False)
    regulatory_quote: Mapped[str] = mapped_column(Text, nullable=False)
    policy_file_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("job_files.id")
    )
    policy_reference: Mapped[str | None] = mapped_column(Text)
    policy_text: Mapped[str | None] = mapped_column(Text)
    gap_description: Mapped[str] = mapped_column(Text, nullable=False)
    severity_justification: Mapped[str] = mapped_column(Text, nullable=False)
    remediation: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    groundedness_score: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    is_uncertain: Mapped[bool] = mapped_column(Boolean, default=False)
    display_order: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[Job] = relationship("Job", back_populates="gaps")

    __table_args__ = (
        Index("compliance_gaps_job_idx", "job_id"),
        Index("compliance_gaps_job_severity_idx", "job_id", "severity"),
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    job_id: Mapped[str] = mapped_column(String(26), ForeignKey("jobs.id", ondelete="CASCADE"))
    format: Mapped[ReportFormat] = mapped_column(Enum(ReportFormat, name="report_format"))
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    job: Mapped[Job] = relationship("Job", back_populates="reports")

    __table_args__ = (UniqueConstraint("job_id", "format"),)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    eval_type: Mapped[EvalType] = mapped_column(Enum(EvalType, name="eval_type"))
    triggered_by: Mapped[str] = mapped_column(String(32), default="scheduled")
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    ragas_faithfulness: Mapped[float | None] = mapped_column(Numeric(5, 4))
    ragas_answer_relevancy: Mapped[float | None] = mapped_column(Numeric(5, 4))
    ragas_context_precision: Mapped[float | None] = mapped_column(Numeric(5, 4))
    ragas_context_recall: Mapped[float | None] = mapped_column(Numeric(5, 4))
    gap_f1: Mapped[float | None] = mapped_column(Numeric(5, 4))
    gap_precision: Mapped[float | None] = mapped_column(Numeric(5, 4))
    gap_recall: Mapped[float | None] = mapped_column(Numeric(5, 4))
    gap_threshold: Mapped[float | None] = mapped_column(Numeric(4, 3))
    routing_accuracy: Mapped[float | None] = mapped_column(Numeric(5, 4))
    tool_use_quality_avg: Mapped[float | None] = mapped_column(Numeric(5, 4))
    citation_accuracy_avg: Mapped[float | None] = mapped_column(Numeric(5, 4))
    traces_evaluated: Mapped[int | None] = mapped_column(Integer)
    p50_latency_seconds: Mapped[float | None] = mapped_column(Numeric(8, 2))
    p95_latency_seconds: Mapped[float | None] = mapped_column(Numeric(8, 2))
    p99_latency_seconds: Mapped[float | None] = mapped_column(Numeric(8, 2))
    all_thresholds_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("eval_runs_type_started_idx", "eval_type", "started_at"),)
