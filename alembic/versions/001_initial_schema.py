"""Initial Meridian schema — all tables, indexes, enums, and pgvector extension.

Revision ID: 001
Revises: —
Create Date: 2026-02-24 09:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Extensions ────────────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")  # for fast LIKE queries

    # ── Enums ─────────────────────────────────────────────────────────────────
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE job_status AS ENUM
                ('queued','processing','complete','failed','cancelled');
        EXCEPTION WHEN duplicate_object THEN NULL; END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE file_modality AS ENUM
                ('document','audio','image','tabular','unknown');
        EXCEPTION WHEN duplicate_object THEN NULL; END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE agent_type AS ENUM
                ('doc_agent','audio_agent','vision_agent','data_agent');
        EXCEPTION WHEN duplicate_object THEN NULL; END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE gap_severity AS ENUM ('critical','major','minor');
        EXCEPTION WHEN duplicate_object THEN NULL; END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE report_format AS ENUM ('json','pdf','markdown');
        EXCEPTION WHEN duplicate_object THEN NULL; END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE eval_type AS ENUM
                ('ragas','gap_detection_f1','agent_judge','latency');
        EXCEPTION WHEN duplicate_object THEN NULL; END $$;
    """)

    # ── corpora ───────────────────────────────────────────────────────────────
    op.create_table(
        "corpora",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("slug", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("jurisdiction", sa.String(32), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("source_url", sa.Text),
        sa.Column("document_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_refreshed", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
    )

    # ── documents ─────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "corpus_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("corpora.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("source_url", sa.Text),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("page_count", sa.Integer),
        sa.Column("token_count", sa.Integer),
        sa.Column("language", sa.String(8), nullable=False, server_default="en"),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("storage_key", sa.Text),
        sa.UniqueConstraint("corpus_id", "content_hash", name="uq_documents_corpus_hash"),
    )
    op.create_index("ix_documents_corpus_id", "documents", ["corpus_id"])

    # ── chunks ────────────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "corpus_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("corpora.id"), nullable=False
        ),
        sa.Column("regulation", sa.String(64), nullable=False),
        sa.Column("article", sa.String(128)),
        sa.Column("article_title", sa.Text),
        sa.Column("jurisdiction", sa.String(32), nullable=False),
        sa.Column("effective_date", sa.Date),
        sa.Column("section_path", postgresql.ARRAY(sa.Text)),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("embedding", sa.Text),  # will be altered to vector(1536)
        sa.Column("ts_vector", postgresql.TSVECTOR),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_idx"),
    )

    # Alter embedding column to vector type after table creation
    op.execute(
        "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector;"
    )

    # IVFFlat index for approximate nearest neighbor search
    op.execute("""
        CREATE INDEX chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 200);
    """)
    op.execute("CREATE INDEX chunks_ts_vector_idx ON chunks USING gin (ts_vector);")
    op.execute("CREATE INDEX chunks_corpus_regulation_idx ON chunks (corpus_id, regulation);")

    # ts_vector auto-update trigger
    op.execute("""
        CREATE OR REPLACE FUNCTION chunks_ts_vector_update() RETURNS TRIGGER AS $$
        BEGIN
            NEW.ts_vector := to_tsvector('english', NEW.content);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER chunks_ts_vector_trigger
            BEFORE INSERT OR UPDATE OF content ON chunks
            FOR EACH ROW EXECUTE FUNCTION chunks_ts_vector_update();
    """)

    # ── jobs ──────────────────────────────────────────────────────────────────
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column(
            "status",
            postgresql.ENUM(
                "queued",
                "processing",
                "complete",
                "failed",
                "cancelled",
                name="job_status",
                create_type=False,
            ),
            nullable=False,
            server_default="queued",
        ),
        sa.Column("regulation_scope", postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column("report_formats", postgresql.ARRAY(sa.Text), server_default="{pdf,json}"),
        sa.Column("language", sa.String(8), nullable=False, server_default="en"),
        sa.Column("options", postgresql.JSONB, server_default="{}"),
        sa.Column("submitted_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("current_stage", sa.String(64)),
        sa.Column("stages_complete", postgresql.ARRAY(sa.Text), server_default="{}"),
        sa.Column("webhook_url", sa.Text),
        sa.Column("webhook_delivered", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("webhook_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_gaps", sa.Integer),
        sa.Column("gaps_critical", sa.Integer),
        sa.Column("gaps_major", sa.Integer),
        sa.Column("gaps_minor", sa.Integer),
        sa.Column("groundedness_pass_rate", sa.Numeric(4, 3)),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.Text),
        sa.Column("error_stage", sa.String(64)),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("langsmith_run_id", sa.Text),
        sa.Column("langsmith_trace_url", sa.Text),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_status_submitted", "jobs", ["status", "submitted_at"])
    op.create_index("ix_jobs_expires", "jobs", ["expires_at"])

    # ── job_files ─────────────────────────────────────────────────────────────
    op.create_table(
        "job_files",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("job_id", sa.String(26), sa.ForeignKey("jobs.id", ondelete="CASCADE")),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column(
            "modality",
            postgresql.ENUM(
                "document",
                "audio",
                "image",
                "tabular",
                "unknown",
                name="file_modality",
                create_type=False,
            ),
            server_default="unknown",
        ),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("storage_key", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("duration_seconds", sa.Integer),
        sa.Column("page_count", sa.Integer),
        sa.Column("processed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("processed_at", sa.DateTime(timezone=True)),
        sa.Column("processing_error", sa.Text),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_job_files_job_id", "job_files", ["job_id"])

    # ── agent_extractions ─────────────────────────────────────────────────────
    op.create_table(
        "agent_extractions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("job_id", sa.String(26), sa.ForeignKey("jobs.id", ondelete="CASCADE")),
        sa.Column("file_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("job_files.id")),
        sa.Column(
            "agent",
            postgresql.ENUM(
                "doc_agent",
                "audio_agent",
                "vision_agent",
                "data_agent",
                name="agent_type",
                create_type=False,
            ),
        ),
        sa.Column("raw_text", sa.Text),
        sa.Column("ner_entities", postgresql.JSONB),
        sa.Column("qa_results", postgresql.JSONB),
        sa.Column("summary", sa.Text),
        sa.Column("transcript", postgresql.JSONB),
        sa.Column("speakers", postgresql.ARRAY(sa.Text)),
        sa.Column("image_caption", sa.Text),
        sa.Column("vqa_results", postgresql.JSONB),
        sa.Column("colpali_matches", postgresql.JSONB),
        sa.Column("table_summary", sa.Text),
        sa.Column("tapas_answers", postgresql.JSONB),
        sa.Column("anomaly_scores", postgresql.JSONB),
        sa.Column("forecast_output", postgresql.JSONB),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_agent_extractions_job_agent", "agent_extractions", ["job_id", "agent"])

    # ── compliance_gaps ───────────────────────────────────────────────────────
    op.create_table(
        "compliance_gaps",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("job_id", sa.String(26), sa.ForeignKey("jobs.id", ondelete="CASCADE")),
        sa.Column(
            "severity",
            postgresql.ENUM("critical", "major", "minor", name="gap_severity", create_type=False),
        ),
        sa.Column("framework", sa.String(64), nullable=False),
        sa.Column("regulatory_article", sa.String(256), nullable=False),
        sa.Column(
            "regulatory_chunk_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("chunks.id")
        ),
        sa.Column("regulatory_requirement", sa.Text, nullable=False),
        sa.Column("regulatory_quote", sa.Text, nullable=False),
        sa.Column("policy_file_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("job_files.id")),
        sa.Column("policy_reference", sa.Text),
        sa.Column("policy_text", sa.Text),
        sa.Column("gap_description", sa.Text, nullable=False),
        sa.Column("severity_justification", sa.Text, nullable=False),
        sa.Column("remediation", sa.Text, nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column("groundedness_score", sa.Numeric(4, 3), nullable=False),
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_uncertain", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("display_order", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_compliance_gaps_job", "compliance_gaps", ["job_id"])

    # ── reports ───────────────────────────────────────────────────────────────
    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("job_id", sa.String(26), sa.ForeignKey("jobs.id", ondelete="CASCADE")),
        sa.Column(
            "format",
            postgresql.ENUM("json", "pdf", "markdown", name="report_format", create_type=False),
        ),
        sa.Column("storage_key", sa.Text, nullable=False),
        sa.Column("size_bytes", sa.BigInteger),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("job_id", "format", name="uq_reports_job_format"),
    )

    # ── eval_runs ─────────────────────────────────────────────────────────────
    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "eval_type",
            postgresql.ENUM(
                "ragas",
                "gap_detection_f1",
                "agent_judge",
                "latency",
                name="eval_type",
                create_type=False,
            ),
        ),
        sa.Column("triggered_by", sa.String(32), server_default="scheduled"),
        sa.Column("pipeline_version", sa.String(32), nullable=False),
        sa.Column("ragas_faithfulness", sa.Numeric(5, 4)),
        sa.Column("ragas_answer_relevancy", sa.Numeric(5, 4)),
        sa.Column("ragas_context_precision", sa.Numeric(5, 4)),
        sa.Column("ragas_context_recall", sa.Numeric(5, 4)),
        sa.Column("gap_f1", sa.Numeric(5, 4)),
        sa.Column("gap_precision", sa.Numeric(5, 4)),
        sa.Column("gap_recall", sa.Numeric(5, 4)),
        sa.Column("gap_threshold", sa.Numeric(4, 3)),
        sa.Column("routing_accuracy", sa.Numeric(5, 4)),
        sa.Column("tool_use_quality_avg", sa.Numeric(5, 4)),
        sa.Column("citation_accuracy_avg", sa.Numeric(5, 4)),
        sa.Column("traces_evaluated", sa.Integer),
        sa.Column("p50_latency_seconds", sa.Numeric(8, 2)),
        sa.Column("p95_latency_seconds", sa.Numeric(8, 2)),
        sa.Column("p99_latency_seconds", sa.Numeric(8, 2)),
        sa.Column("all_thresholds_passed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("mlflow_run_id", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("duration_seconds", sa.Integer),
    )
    op.create_index("ix_eval_runs_type_started", "eval_runs", ["eval_type", "started_at"])


def downgrade() -> None:
    op.drop_table("eval_runs")
    op.drop_table("reports")
    op.drop_table("compliance_gaps")
    op.drop_table("agent_extractions")
    op.drop_table("job_files")
    op.drop_table("jobs")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("corpora")

    for enum in [
        "job_status",
        "file_modality",
        "agent_type",
        "gap_severity",
        "report_format",
        "eval_type",
    ]:
        op.execute(f"DROP TYPE IF EXISTS {enum};")

    op.execute("DROP FUNCTION IF EXISTS chunks_ts_vector_update CASCADE;")
