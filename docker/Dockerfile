# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files only (layer-cache friendly)
COPY pyproject.toml requirements.txt ./

# Install all Python dependencies into user site-packages
RUN pip install --user --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="meridian@example.com"
LABEL version="1.0.3"
LABEL description="Meridian Compliance Intelligence API"

WORKDIR /app

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # File type detection (python-magic)
    libmagic1 \
    # PDF rendering for vision agent (pdf2image)
    poppler-utils \
    # Audio processing (ffmpeg for format conversion)
    ffmpeg \
    # PostgreSQL client library
    libpq5 \
    # curl for Docker health checks
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/
COPY data/golden/ ./data/golden/

# Create non-root user for security
RUN groupadd --gid 1001 meridian \
    && useradd --uid 1001 --gid meridian --shell /bin/bash --create-home meridian

# Create local storage directory with correct permissions
RUN mkdir -p /tmp/meridian/uploads /tmp/meridian/reports \
    && chown -R meridian:meridian /tmp/meridian /app

USER meridian

# Ensure .local/bin is in PATH
ENV PATH="/home/meridian/.local/bin:/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app"

EXPOSE 8000

# Health check: ping the health endpoint every 30 seconds
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Default: run the API server
# Override with "celery -A src.api.worker worker" for worker containers
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--no-access-log"]
