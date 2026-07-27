"""FastAPI application factory and startup configuration."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.all import (
    router_corpus,
    router_health,
    router_report,
    router_status,
    router_submit,
)
from src.config import settings

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic."""
    logger.info("Starting Meridian API v%s (%s)", settings.VERSION, settings.ENVIRONMENT)

    # Initialise Sentry if configured
    if settings.SENTRY_DSN:
        import sentry_sdk
        sentry_sdk.init(dsn=settings.SENTRY_DSN, environment=settings.ENVIRONMENT)
        logger.info("Sentry error tracking enabled")

    # Warm up the local similarity model (used by hallucination gate)
    try:
        from src.models.retrieval import SimilarityModel
        _ = SimilarityModel()._get_model()
        logger.info("SimilarityModel warm-up complete")
    except Exception as exc:
        logger.warning("SimilarityModel warm-up failed: %s", exc)

    yield

    logger.info("Meridian API shutting down")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Meridian API",
        version=settings.VERSION,
        description="Multimodal Regulatory Intelligence & Compliance Automation Platform",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    origins = (
        settings.ALLOWED_ORIGINS.split(",")
        if settings.ALLOWED_ORIGINS != "*"
        else ["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # ── Request ID middleware ─────────────────────────────────────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)

        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        logger.info(
            "HTTP %s %s %d %dms req_id=%s",
            request.method, request.url.path,
            response.status_code, duration_ms, request_id,
        )
        return response

    # ── Rate limiting (simple Redis sliding window) ───────────────────────────
    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        # Only rate-limit authenticated routes
        if not request.url.path.startswith("/v1/submit"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key_prefix = auth_header[7:15]
            try:
                import redis.asyncio as aioredis
                from datetime import datetime
                r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
                minute_key = f"ratelimit:{api_key_prefix}:{int(time.time() // 60)}"
                count = await r.incr(minute_key)
                await r.expire(minute_key, 60)
                if count > settings.RATE_LIMIT_SUBMIT:
                    return JSONResponse(
                        status_code=429,
                        content={"error": {"code": "rate_limited", "message": "Rate limit exceeded"}},
                        headers={
                            "X-RateLimit-Limit": str(settings.RATE_LIMIT_SUBMIT),
                            "X-RateLimit-Remaining": "0",
                            "Retry-After": "60",
                        },
                    )
            except Exception:
                pass  # Don't fail requests on Redis error

        return await call_next(request)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "An internal error occurred"}},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(router_submit)
    app.include_router(router_status)
    app.include_router(router_report)
    app.include_router(router_corpus)
    app.include_router(router_health)

    return app


app = create_app()
