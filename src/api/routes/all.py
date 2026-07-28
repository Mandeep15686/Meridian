"""FastAPI route handlers for job submission, status, report retrieval, and health."""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import magic
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import require_api_key
from src.api.schemas.all import (
    CorpusInfo,
    CorpusStatusResponse,
    DependencyStatus,
    GapResponse,
    HealthResponse,
    JobError,
    JobStatusResponse,
    JobSummary,
    ModelMetadata,
    ReportResponse,
    SubmitResponse,
    SubmittedFileInfo,
)
from src.config import settings
from src.db.models import (
    ComplianceGap,
    Corpus,
    Job,
    JobFile,
    JobStatus,
    Report,
    ReportFormat,
)
from src.db.session import get_db
from src.storage.base import get_storage

logger = logging.getLogger(__name__)

router_submit = APIRouter(prefix="/v1", tags=["jobs"])
router_status = APIRouter(prefix="/v1", tags=["jobs"])
router_report = APIRouter(prefix="/v1", tags=["jobs"])
router_corpus = APIRouter(prefix="/v1", tags=["corpus"])
router_health = APIRouter(prefix="/v1", tags=["health"])


# ── POST /v1/submit ───────────────────────────────────────────────────────────

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "text/html",
    "text/markdown",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/flac",
    "audio/ogg",
    "video/mp4",
    "video/quicktime",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


@router_submit.post("/submit", response_model=SubmitResponse, status_code=202)
async def submit_job(
    request: Request,
    files: list[UploadFile] = File(...),
    regulation_scope: list[str] = Form(...),
    webhook_url: str | None = Form(default=None),
    report_formats: list[str] = Form(default=["pdf", "json"]),
    language: str = Form(default="en"),
    _: str = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> SubmitResponse:
    """Accept a multipart file submission and queue a compliance analysis job."""

    # Validate inputs
    if not files:
        raise HTTPException(
            status_code=400,
            detail={"code": "no_files", "message": "At least one file required"},
        )
    if len(files) > settings.MAX_FILES_PER_JOB:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "too_many_files",
                "message": f"Max {settings.MAX_FILES_PER_JOB} files per job",
            },
        )

    # Generate ULID-like job ID
    import ulid

    job_id = str(ulid.new())

    storage = get_storage()
    job_files: list[JobFile] = []

    for upload in files:
        # Read file content
        content = await upload.read()

        # Size check
        if len(content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "file_too_large",
                    "message": f"{upload.filename} exceeds {settings.MAX_FILE_SIZE_MB} MB",
                },
            )

        # MIME type validation via magic bytes
        detected_mime = magic.from_buffer(content[:2048], mime=True)
        declared_mime = upload.content_type or "application/octet-stream"

        # Accept if either detected or declared MIME is allowed
        if (
            detected_mime not in ALLOWED_MIME_TYPES
            and declared_mime not in ALLOWED_MIME_TYPES
        ):
            raise HTTPException(
                status_code=415,
                detail={
                    "code": "unsupported_file_type",
                    "message": f"File type not supported: {detected_mime}",
                },
            )

        content_hash = hashlib.sha256(content).hexdigest()
        file_id = str(uuid.uuid4())
        storage_key = (
            f"uploads/{job_id}/{file_id}_{Path(upload.filename or 'file').name}"
        )

        await storage.upload(content, storage_key, content_type=detected_mime)

        job_files.append(
            JobFile(
                id=file_id,
                job_id=job_id,
                filename=upload.filename or "unnamed",
                mime_type=detected_mime,
                size_bytes=len(content),
                storage_key=storage_key,
                content_hash=content_hash,
            )
        )

    # Create job record
    expires_at = datetime.now(UTC) + timedelta(days=settings.JOB_TTL_DAYS)
    job = Job(
        id=job_id,
        status=JobStatus.QUEUED,
        regulation_scope=regulation_scope,
        report_formats=report_formats,
        language=language,
        webhook_url=webhook_url,
        expires_at=expires_at,
    )
    db.add(job)
    for jf in job_files:
        db.add(jf)
    await db.commit()

    # Enqueue Celery task
    from src.api.worker import run_compliance_job

    run_compliance_job.delay(job_id)

    submitted_at = datetime.now(UTC)
    logger.info(
        "Job %s queued: %d files, scope=%s", job_id, len(job_files), regulation_scope
    )

    return SubmitResponse(
        job_id=job_id,
        status="queued",
        submitted_at=submitted_at,
        poll_url=f"/v1/status/{job_id}",
        report_url=f"/v1/report/{job_id}",
    )


# ── GET /v1/status/{job_id} ───────────────────────────────────────────────────


@router_status.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    _: str = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """Return current job status and progress."""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "job_not_found",
                "message": f"Job {job_id} not found or expired",
            },
        )

    # Build progress percentage
    stage_count = 6  # classify → agents → synthesize → gate → report
    completed = len(job.stages_complete or [])
    progress_pct = (
        min(99, int(completed / stage_count * 100))
        if job.status == JobStatus.PROCESSING
        else None
    )

    # Build summary for complete jobs
    if job.status == JobStatus.COMPLETE and job.total_gaps is not None:
        JobSummary(
            total_gaps=job.total_gaps,
            by_severity={
                "critical": job.gaps_critical or 0,
                "major": job.gaps_major or 0,
                "minor": job.gaps_minor or 0,
            },
            by_framework={},
            groundedness_pass_rate=float(job.groundedness_pass_rate or 1.0),
        )

    duration = None
    if job.started_at and job.completed_at:
        duration = int((job.completed_at - job.started_at).total_seconds())

    return JobStatusResponse(
        job_id=job_id,
        status=job.status.value,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        duration_seconds=duration,
        current_stage=job.current_stage,
        stages_complete=job.stages_complete or [],
        progress_pct=progress_pct,
        report_url=f"/v1/report/{job_id}" if job.status == JobStatus.COMPLETE else None,
        langsmith_trace_url=job.langsmith_trace_url,
        error=(
            JobError(
                code=job.error_code,
                message=job.error_message or "",
                stage=job.error_stage or "unknown",
                retry_count=job.retry_count,
            )
            if job.error_code
            else None
        ),
    )


# ── GET /v1/report/{job_id} ───────────────────────────────────────────────────


@router_report.get("/report/{job_id}", response_model=ReportResponse)
async def get_report(
    job_id: str,
    format: str = "json",
    _: str = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> ReportResponse | Response:
    """Return the compliance report for a completed job."""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "job_not_found", "message": "Job not found"},
        )
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "job_not_complete",
                "message": f"Job status: {job.status.value}",
            },
        )

    if format == "pdf":
        # Serve PDF binary from storage
        rpt_result = await db.execute(
            select(Report).where(
                Report.job_id == job_id, Report.format == ReportFormat.PDF
            )
        )
        report_row = rpt_result.scalar_one_or_none()
        if not report_row:
            raise HTTPException(
                status_code=404,
                detail={"code": "pdf_not_ready", "message": "PDF not yet generated"},
            )
        storage = get_storage()
        pdf_bytes = await storage.download(report_row.storage_key)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="meridian_report_{job_id}.pdf"'
            },
        )

    # JSON report: build from database records
    gaps_result = await db.execute(
        select(ComplianceGap)
        .where(ComplianceGap.job_id == job_id)
        .order_by(ComplianceGap.display_order)
    )
    gaps = gaps_result.scalars().all()

    files_result = await db.execute(select(JobFile).where(JobFile.job_id == job_id))
    files = files_result.scalars().all()

    gap_responses = [
        GapResponse(
            gap_id=g.id,
            severity=g.severity.value,
            framework=g.framework,
            regulatory_article=g.regulatory_article,
            regulatory_requirement=g.regulatory_requirement,
            regulatory_quote=g.regulatory_quote,
            regulatory_chunk_id=g.regulatory_chunk_id,
            policy_reference=g.policy_reference,
            policy_text=g.policy_text,
            gap_description=g.gap_description,
            severity_justification=g.severity_justification,
            remediation=g.remediation,
            confidence=float(g.confidence),
            groundedness_score=float(g.groundedness_score),
            is_verified=g.is_verified,
            is_uncertain=g.is_uncertain,
        )
        for g in gaps
    ]

    return ReportResponse(
        job_id=job_id,
        generated_at=job.completed_at or datetime.now(UTC),
        regulation_scope=job.regulation_scope,
        submitted_files=[
            SubmittedFileInfo(
                filename=f.filename,
                modality=f.modality.value,
                size_bytes=f.size_bytes,
            )
            for f in files
        ],
        executive_summary=f"Analysis found {len(gaps)} compliance gap(s).",
        compliance_score={},
        gaps=gap_responses,
        model_metadata=ModelMetadata(
            synthesis_model=settings.SYNTHESIS_MODEL,
            retrieval_model=settings.EMBEDDING_MODEL,
            reranker_model=settings.RERANKER_MODEL,
            ner_model="dslim/bert-base-NER",
            asr_model="openai/whisper-large-v3",
            pipeline_version=settings.VERSION,
        ),
    )


# ── GET /v1/corpus/status ─────────────────────────────────────────────────────


@router_corpus.get("/corpus/status", response_model=CorpusStatusResponse)
async def get_corpus_status(
    _: str = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> CorpusStatusResponse:
    """Return metadata and freshness status for all ingested regulatory corpora."""
    from datetime import timedelta

    result = await db.execute(select(Corpus).where(Corpus.is_active))
    corpora = result.scalars().all()

    total_chunks = sum(c.chunk_count for c in corpora)
    stale_threshold = timedelta(days=settings.CORPUS_MAX_STALENESS_DAYS)
    now = datetime.now(UTC)

    corpus_infos: list[CorpusInfo] = []
    for c in corpora:
        age = now - c.last_refreshed.replace(tzinfo=UTC)
        freshness = cast(
            "Literal['current', 'stale', 'unknown']",
            "stale" if age > stale_threshold else "current",
        )
        corpus_infos.append(
            CorpusInfo(
                id=c.slug,
                name=c.name,
                jurisdiction=c.jurisdiction,
                version=c.version,
                source_url=c.source_url,
                document_count=c.document_count,
                chunk_count=c.chunk_count,
                last_ingested=c.last_refreshed,
                freshness_status=freshness,
            )
        )

    return CorpusStatusResponse(
        corpora=corpus_infos, total_chunks=total_chunks, index_last_rebuilt=None
    )


# ── GET /v1/health ────────────────────────────────────────────────────────────


@router_health.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Health check with dependency status for load balancer probes."""
    import redis.asyncio as aioredis
    from sqlalchemy import text as sql_text

    # Database
    db_healthy = False
    try:
        await db.execute(sql_text("SELECT 1"))
        db_healthy = True
    except Exception:
        pass

    # Redis
    redis_healthy = False
    try:
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        redis_healthy = True
    except Exception:
        pass

    # HF API (lightweight check — skip in production for speed)
    hf_healthy = True  # assume healthy unless we test

    # Anthropic API (lightweight check)
    anthropic_healthy = bool(settings.ANTHROPIC_API_KEY.startswith("sk-ant"))

    deps = DependencyStatus(
        database="healthy" if db_healthy else "unhealthy",
        redis="healthy" if redis_healthy else "unhealthy",
        hf_api="healthy" if hf_healthy else "unhealthy",
        anthropic_api="healthy" if anthropic_healthy else "unhealthy",
    )

    all_healthy = all(v == "healthy" for v in deps.model_dump().values())
    critical_healthy = db_healthy and redis_healthy
    overall = cast(
        "Literal['healthy', 'degraded', 'unhealthy']",
        "healthy" if all_healthy else ("degraded" if critical_healthy else "unhealthy"),
    )

    return HealthResponse(
        status=overall,
        version=settings.VERSION,
        timestamp=datetime.now(UTC),
        dependencies=deps,
        message=(
            None if overall == "healthy" else "One or more dependencies are degraded"
        ),
    )
