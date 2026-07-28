"""Celery worker — runs the Meridian LangGraph pipeline asynchronously."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, cast

from celery import Celery
from sqlalchemy import select

from src.config import settings

logger = logging.getLogger(__name__)

# ── Celery application ────────────────────────────────────────────────────────

celery_app = Celery(
    "meridian",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,  # acknowledge after completion (safe for crash recovery)
    task_reject_on_worker_lost=True,  # re-queue on worker crash
    worker_prefetch_multiplier=1,  # one job at a time per worker thread
    task_soft_time_limit=settings.CELERY_TASK_TIMEOUT
    - 60,  # soft limit triggers warning
    task_time_limit=settings.CELERY_TASK_TIMEOUT,  # hard limit kills the task
    result_expires=settings.JOB_TTL_DAYS * 86400,
    beat_schedule={
        "expire-old-jobs": {
            "task": "src.api.worker.expire_old_jobs",
            "schedule": 3600.0,  # hourly
        },
        "deliver-pending-webhooks": {
            "task": "src.api.worker.deliver_pending_webhooks",
            "schedule": 60.0,
        },
    },
)


def _run_async(coro: Any) -> Any:
    """Run a coroutine in a new event loop (Celery workers are sync by default)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


# ── Main compliance job task ──────────────────────────────────────────────────


@celery_app.task(
    name="run_compliance_job",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def run_compliance_job(self: Any, job_id: str) -> dict[str, Any]:
    """
    Execute the full Meridian compliance analysis pipeline for a job.

    This task:
    1. Loads job and files from the database
    2. Invokes the LangGraph pipeline
    3. Persists compliance gaps and reports to the database
    4. Delivers the webhook notification
    """
    return cast(dict[str, Any], _run_async(_run_compliance_job_async(self, job_id)))


async def _run_compliance_job_async(task: Any, job_id: str) -> dict[str, Any]:
    """Async implementation of the compliance job task."""
    from src.db.models import Job, JobFile, JobStatus
    from src.db.session import get_db_session
    from src.graph.graph import run_pipeline
    from src.graph.state import UploadedFile

    logger.info("Starting compliance job: %s", job_id)

    async with get_db_session() as db:
        # Load job
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()

        if job is None:
            logger.error("Job %s not found in database", job_id)
            return {"status": "failed", "error": "Job not found"}

        # Mark as processing
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now(UTC)
        job.current_stage = "classify_input"
        await db.commit()

    try:
        async with get_db_session() as db:
            # Load files
            files_result = await db.execute(
                select(JobFile).where(JobFile.job_id == job_id)
            )
            db_files = files_result.scalars().all()

            input_files = [
                UploadedFile(
                    file_id=f.id,
                    filename=f.filename,
                    modality="unknown",
                    mime_type=f.mime_type,
                    size_bytes=f.size_bytes,
                    storage_key=f.storage_key,
                    content_hash=f.content_hash,
                    duration_seconds=f.duration_seconds,
                    page_count=f.page_count,
                )
                for f in db_files
            ]

        # Run the LangGraph pipeline
        final_state = await run_pipeline(
            job_id=job_id,
            input_files=input_files,
            regulation_scope=job.regulation_scope,
        )

        # Check for pipeline error
        if final_state.get("error"):
            await _mark_failed(
                job_id,
                str(final_state.get("error", "Unknown pipeline error")),
                str(final_state.get("error_stage", "unknown")),
                task.request.retries,
            )
            return {"status": "failed"}

        # Persist results
        report = final_state.get("final_report")
        if report:
            await _persist_results(
                job_id, cast(dict[str, Any], final_state), list(db_files)
            )

        logger.info("Compliance job %s completed successfully", job_id)
        return {"status": "complete", "job_id": job_id}

    except Exception as exc:
        logger.exception("Compliance job %s failed: %s", job_id, exc)
        await _mark_failed(job_id, str(exc), "unknown", task.request.retries)

        # Retry if we have retries left
        if task.request.retries < task.max_retries:
            raise task.retry(
                exc=exc,
                countdown=30 * (task.request.retries + 1),
            ) from exc

        return {"status": "failed", "error": str(exc)}


async def _persist_results(
    job_id: str, final_state: dict[str, Any], db_files: list[Any]
) -> None:
    """Save compliance gaps, update job stats, and generate reports."""
    from src.db.models import (
        ComplianceGap,
        GapSeverity,
        Job,
        JobStatus,
        Report,
        ReportFormat,
    )
    from src.db.session import get_db_session

    report = final_state.get("final_report")
    if not report:
        return

    async with get_db_session() as db:
        # Persist compliance gaps
        for i, gap in enumerate(report.gaps):
            db_gap = ComplianceGap(
                job_id=job_id,
                severity=GapSeverity(gap.severity),
                framework=gap.framework,
                regulatory_article=gap.regulatory_article,
                regulatory_chunk_id=gap.regulatory_chunk_id,
                regulatory_requirement=gap.regulatory_requirement,
                regulatory_quote=gap.regulatory_quote,
                policy_reference=gap.policy_reference,
                policy_text=gap.policy_text,
                gap_description=gap.gap_description,
                severity_justification=gap.severity_justification,
                remediation=gap.remediation,
                confidence=gap.confidence,
                groundedness_score=gap.groundedness_score,
                is_verified=gap.is_verified,
                is_uncertain=gap.is_uncertain,
                display_order=i,
            )
            db.add(db_gap)

        # Update job stats
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()
        job.status = JobStatus.COMPLETE
        job.completed_at = datetime.now(UTC)
        job.total_gaps = report.total_gaps
        job.gaps_critical = report.gaps_critical
        job.gaps_major = report.gaps_major
        job.gaps_minor = report.gaps_minor
        job.groundedness_pass_rate = report.groundedness_pass_rate
        job.current_stage = None

        # Create JSON report record
        import json
        from datetime import timedelta

        report_key = f"reports/{job_id}/report.json"
        from src.storage.base import get_storage

        storage = get_storage()
        report_json = json.dumps(
            report.model_dump(mode="json"), indent=2, default=str
        ).encode()
        await storage.upload(report_json, report_key, "application/json")

        db.add(
            Report(
                job_id=job_id,
                format=ReportFormat.JSON,
                storage_key=report_key,
                size_bytes=len(report_json),
                expires_at=datetime.now(UTC) + timedelta(days=7),
            )
        )

        await db.commit()

    # Deliver webhook if configured
    await _deliver_webhook(job_id)


async def _mark_failed(
    job_id: str, error_message: str, error_stage: str, retry_count: int
) -> None:
    """Mark a job as failed in the database."""
    from src.db.models import Job, JobStatus
    from src.db.session import get_db_session

    async with get_db_session() as db:
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(UTC)
            job.error_code = "pipeline_failed"
            job.error_message = error_message[:1000]
            job.error_stage = error_stage
            job.retry_count = retry_count
            await db.commit()


async def _deliver_webhook(job_id: str) -> None:
    """Deliver the completion webhook if configured."""
    import hashlib
    import hmac
    import json

    import httpx

    from src.db.models import Job
    from src.db.session import get_db_session

    async with get_db_session() as db:
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()

        if not job or not job.webhook_url:
            return

        payload = {
            "event": "job.complete",
            "job_id": job_id,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "summary": {
                "total_gaps": job.total_gaps,
                "critical": job.gaps_critical,
                "major": job.gaps_major,
                "minor": job.gaps_minor,
            },
            "report_url": f"/v1/report/{job_id}",
        }
        body = json.dumps(payload).encode()

        headers = {
            "Content-Type": "application/json",
            "X-Meridian-Job-Id": job_id,
            "X-Meridian-Timestamp": str(int(datetime.now(UTC).timestamp())),
        }

        if settings.WEBHOOK_SECRET:
            sig = hmac.new(
                settings.WEBHOOK_SECRET.encode(), body, hashlib.sha256
            ).hexdigest()
            headers["X-Meridian-Signature"] = f"sha256={sig}"

        retry_delays = settings.webhook_retry_delays_list
        for attempt, delay in enumerate(retry_delays):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.post(
                        job.webhook_url, content=body, headers=headers
                    )
                    if response.status_code < 400:
                        job.webhook_delivered = True
                        job.webhook_attempts = attempt + 1
                        await db.commit()
                        logger.info("Webhook delivered for job %s", job_id)
                        return
            except Exception as exc:
                logger.warning(
                    "Webhook attempt %d failed for job %s: %s", attempt + 1, job_id, exc
                )

            if attempt < len(retry_delays) - 1:
                await asyncio.sleep(delay)

        job.webhook_attempts = len(retry_delays)
        await db.commit()
        logger.error("All webhook attempts failed for job %s", job_id)


# ── Maintenance tasks ─────────────────────────────────────────────────────────


@celery_app.task(name="expire_old_jobs")
def expire_old_jobs() -> int:
    """Delete jobs and associated data that have exceeded their TTL."""
    return cast(int, _run_async(_expire_old_jobs_async()))


async def _expire_old_jobs_async() -> int:
    from sqlalchemy import delete

    from src.db.models import Job
    from src.db.session import get_db_session

    now = datetime.now(UTC)
    async with get_db_session() as db:
        result = await db.execute(select(Job.id).where(Job.expires_at < now))
        expired_ids = [row[0] for row in result.all()]

        if expired_ids:
            await db.execute(delete(Job).where(Job.id.in_(expired_ids)))
            await db.commit()
            logger.info("Expired %d old jobs", len(expired_ids))

    return len(expired_ids)


@celery_app.task(name="deliver_pending_webhooks")
def deliver_pending_webhooks() -> int:
    """Retry undelivered webhooks for completed jobs."""
    return cast(int, _run_async(_deliver_pending_webhooks_async()))


async def _deliver_pending_webhooks_async() -> int:
    from src.db.models import Job, JobStatus
    from src.db.session import get_db_session

    async with get_db_session() as db:
        result = await db.execute(
            select(Job.id)
            .where(
                Job.status == JobStatus.COMPLETE,
                Job.webhook_url.isnot(None),
                Job.webhook_delivered.is_(False),
                Job.webhook_attempts < 3,
            )
            .limit(10)
        )
        job_ids = [row[0] for row in result.all()]

    for job_id in job_ids:
        await _deliver_webhook(job_id)

    return len(job_ids)


def start() -> None:
    """Entry point for `meridian-worker` CLI command."""
    celery_app.worker_main(
        argv=[
            "worker",
            "--loglevel=info",
            f"--concurrency={settings.CELERY_CONCURRENCY}",
        ]
    )
