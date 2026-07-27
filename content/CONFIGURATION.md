# Configuration — Meridian

All configuration is driven by environment variables following the [12-factor app](https://12factor.net/config) convention. No configuration lives in code or committed files. This document is the canonical reference for every configurable value in the system.

---

## Loading order

Variables are resolved in this order (later values override earlier ones):

1. System environment
2. `.env` file in the project root (development only; never committed)
3. GCP Secret Manager (production; mounted at runtime by Cloud Run)

In production, `.env` files are not used. All values come from environment variables injected by Cloud Run or Secret Manager.

---

## Required variables

These must be set for the application to start. Missing required variables cause an immediate startup error with a descriptive message.

### `ANTHROPIC_API_KEY`

The API key for the Anthropic Claude API. Used by the synthesis agent for compliance gap reasoning.

- **Format:** `sk-ant-api03-...`
- **Where to get it:** https://console.anthropic.com/
- **Production storage:** GCP Secret Manager as `anthropic-api-key`
- **Rotation:** Rotate quarterly; old keys continue working for 24 hours after rotation

```env
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

---

### `HF_API_TOKEN`

HuggingFace Inference API token. Used by all specialist model calls (NER, ASR, QA, captioning, tabular QA, forecasting, reranking, classification).

- **Format:** `hf_...`
- **Where to get it:** https://huggingface.co/settings/tokens
- **Required permissions:** Read access to public models; must have accepted usage terms for `pyannote/speaker-diarization-3.1`
- **Free tier limits:** 30,000 requests/month on shared inference; upgrade to Inference Endpoints for dedicated throughput
- **Production storage:** GCP Secret Manager as `hf-api-token`

```env
HF_API_TOKEN=hf_your_token_here
```

---

### `DATABASE_URL`

PostgreSQL connection string including credentials. Must point to a PostgreSQL 15+ instance with the `pgvector` extension installed.

- **Format:** `postgresql://user:password@host:port/dbname`
- **Cloud SQL format:** `postgresql://user:password@/dbname?host=/cloudsql/project:region:instance`
- **Production storage:** GCP Secret Manager as `database-url`
- **Pool settings:** SQLAlchemy uses `pool_size=5, max_overflow=10` by default; override with `DB_POOL_SIZE` and `DB_MAX_OVERFLOW`

```env
# Local development
DATABASE_URL=postgresql://meridian:password@localhost:5432/meridian

# GCP Cloud SQL (via Unix socket)
DATABASE_URL=postgresql://meridian:password@/meridian?host=/cloudsql/my-project:us-central1:meridian-db
```

---

### `REDIS_URL`

Redis connection string. Used as the Celery task broker and result backend, and for rate limiting and retrieval caching.

- **Format:** `redis://[:password@]host:port/db`
- **Default:** `redis://localhost:6379/0`
- **Production:** A Redis instance accessible from Cloud Run via VPC; Memorystore (GCP) or a dedicated Redis Cloud instance

```env
REDIS_URL=redis://localhost:6379/0
```

---

### `MERIDIAN_API_KEY`

The Bearer token clients use to authenticate with the Meridian API. Generate with `openssl rand -hex 32`.

- **Minimum length:** 32 characters
- **Format:** Any string; conventionally prefixed `mer_live_` for production and `mer_test_` for development
- **Production storage:** GCP Secret Manager as `meridian-api-key`
- **Note:** v1.0 supports a single global API key. Multi-key management is planned for v2.0.

```env
MERIDIAN_API_KEY=mer_live_your_64_char_hex_string_here
```

---

## Optional variables

These have sensible defaults and only need to be set to override default behavior.

### Vector store

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE` | `pgvector` | Active vector store backend. Options: `pgvector`, `pinecone`. When set to `pinecone`, `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` are also required. |
| `PINECONE_API_KEY` | — | Pinecone API key. Required when `VECTOR_STORE=pinecone`. |
| `PINECONE_INDEX_NAME` | `meridian` | Name of the Pinecone index to use. |
| `PINECONE_ENVIRONMENT` | `us-east-1-aws` | Pinecone environment (region). |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model. Changing this invalidates all existing embeddings; a full re-ingestion is required. |
| `EMBEDDING_DIMENSIONS` | `1536` | Dimensions of the embedding model output. Must match the `EMBEDDING_MODEL` output dimensions and the `chunks.embedding` column type. |
| `IVFFLAT_PROBES` | `10` | Number of IVFFlat index lists to probe at query time. Higher = better recall, higher latency. |

---

### Retrieval pipeline

| Variable | Default | Description |
|---|---|---|
| `RETRIEVAL_TOP_K_DENSE` | `20` | Number of candidates from dense retrieval before RRF. |
| `RETRIEVAL_TOP_K_BM25` | `20` | Number of candidates from BM25 retrieval before RRF. |
| `RETRIEVAL_TOP_K_RERANK` | `5` | Number of chunks passed to the LLM after reranking. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | HuggingFace cross-encoder model ID for reranking. |
| `RETRIEVAL_CACHE_TTL` | `3600` | Seconds to cache identical retrieval queries in Redis. Set to `0` to disable. |

---

### Pipeline behavior

| Variable | Default | Description |
|---|---|---|
| `GROUNDEDNESS_THRESHOLD` | `0.80` | Minimum cosine similarity between a synthesized claim and its cited source chunk. Claims below this threshold trigger a re-synthesis or are marked uncertain. |
| `MAX_SYNTHESIS_RETRIES` | `2` | Maximum times the synthesis agent retries after the hallucination gate rejects one or more claims. |
| `SYNTHESIS_MODEL` | `claude-sonnet-4-6` | Anthropic model string for the synthesis agent. |
| `SYNTHESIS_MAX_TOKENS` | `4096` | Maximum tokens for the synthesis LLM response. |
| `MAX_DOCUMENT_TOKENS` | `100000` | Documents larger than this (in tokens) are summarized before synthesis to fit the context window. |
| `CHUNK_SIZE_TARGET` | `512` | Target token size for semantic chunks. The semantic chunker uses this as a guideline, not a hard limit. |
| `CHUNK_OVERLAP` | `64` | Token overlap between consecutive chunks to preserve cross-boundary context. |

---

### Job management

| Variable | Default | Description |
|---|---|---|
| `JOB_TTL_DAYS` | `7` | Number of days to retain completed job data (uploads, reports, database records) before deletion. |
| `MAX_FILE_SIZE_MB` | `500` | Maximum file size per upload in megabytes. |
| `MAX_FILES_PER_JOB` | `10` | Maximum number of files in a single submission. |
| `CELERY_CONCURRENCY` | `4` | Number of parallel Celery worker threads per worker process. |
| `CELERY_TASK_TIMEOUT` | `900` | Maximum seconds a single Celery task may run before being killed and retried. |
| `WEBHOOK_RETRY_DELAYS` | `30,90,270` | Comma-separated retry delay seconds for failed webhook deliveries (exponential backoff). |
| `WEBHOOK_SECRET` | — | HMAC-SHA256 signing key for webhook payloads. If unset, webhooks are sent unsigned. |

---

### Rate limiting

| Variable | Default | Description |
|---|---|---|
| `RATE_LIMIT_SUBMIT` | `10` | Max job submissions per minute per API key. |
| `RATE_LIMIT_STATUS` | `60` | Max status poll requests per minute per API key. |
| `RATE_LIMIT_REPORT` | `30` | Max report retrieval requests per minute per API key. |

---

### Storage

| Variable | Default | Description |
|---|---|---|
| `STORAGE_BACKEND` | `gcs` | Storage backend for uploads and reports. Options: `gcs` (GCP Cloud Storage), `local` (local filesystem, development only). |
| `GCS_UPLOADS_BUCKET` | — | GCP Cloud Storage bucket name for uploaded files. Required when `STORAGE_BACKEND=gcs`. |
| `GCS_REPORTS_BUCKET` | — | GCP Cloud Storage bucket name for generated reports. Required when `STORAGE_BACKEND=gcs`. |
| `LOCAL_STORAGE_PATH` | `/tmp/meridian` | Root path for local file storage. Used when `STORAGE_BACKEND=local`. |

---

### LLMOps and observability

| Variable | Default | Description |
|---|---|---|
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith distributed tracing. Set to `true` to activate. Requires `LANGCHAIN_API_KEY`. |
| `LANGCHAIN_API_KEY` | — | LangSmith API key. Required when `LANGCHAIN_TRACING_V2=true`. |
| `LANGCHAIN_PROJECT` | `meridian` | LangSmith project name for grouping traces. |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URI. Set to a remote server in staging/production. |
| `MLFLOW_EXPERIMENT_NAME` | `meridian-eval` | MLflow experiment name for eval runs. |
| `WANDB_API_KEY` | — | Weights & Biases API key. If unset, W&B logging is disabled. |
| `WANDB_PROJECT` | `meridian` | W&B project name. |
| `SENTRY_DSN` | — | Sentry error tracking DSN. If unset, Sentry is disabled. |

---

### Corpus management

| Variable | Default | Description |
|---|---|---|
| `CORPUS_SOURCES` | `gdpr,soc2,iso27001,sec_sp,cfpb` | Comma-separated list of regulatory corpora to activate. Available: `gdpr`, `soc2`, `iso27001`, `sec_sp`, `cfpb`, `eu_ai_act` (v1.1+). |
| `CORPUS_REFRESH_SCHEDULE` | `0 2 * * *` | Cron expression for the automated corpus freshness check (Celery Beat). Default: 2am UTC daily. |
| `CORPUS_MAX_STALENESS_DAYS` | `30` | Alert threshold: raise an error in the health check if any corpus has not been refreshed within this many days. |
| `EDGAR_USER_AGENT` | `Meridian/1.0 contact@example.com` | User-Agent header for SEC EDGAR API requests. EDGAR requires a contact email in the user agent. |

---

### Application

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Deployment environment. Options: `development`, `staging`, `production`. Controls log format (JSON in production), debug mode, and error detail in API responses. |
| `LOG_LEVEL` | `INFO` | Python logging level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. In production, `INFO` is recommended; `DEBUG` logs model inputs and outputs and should never be used in production. |
| `LOG_FORMAT` | `text` | Log output format. Options: `text` (human-readable), `json` (structured, for log aggregation). Automatically set to `json` when `ENVIRONMENT=production`. |
| `WORKERS` | `2` | Number of uvicorn worker processes for the API. Set to `1` in development for easier debugging. |
| `PORT` | `8000` | Port the API server listens on. |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins. In production, restrict to your frontend domain. |
| `DB_POOL_SIZE` | `5` | SQLAlchemy connection pool size per worker process. |
| `DB_MAX_OVERFLOW` | `10` | SQLAlchemy max overflow connections (temporary pool expansion under load). |

---

## Example `.env.example`

This file is committed to the repo. Copy it to `.env` and fill in real values.

```env
# ── Required ──────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-api03-replace-me
HF_API_TOKEN=hf_replace_me
DATABASE_URL=postgresql://meridian:password@localhost:5432/meridian
REDIS_URL=redis://localhost:6379/0
MERIDIAN_API_KEY=mer_test_replace_with_32_char_random_string

# ── Vector store ───────────────────────────────────────────────
VECTOR_STORE=pgvector
# PINECONE_API_KEY=
# PINECONE_INDEX_NAME=meridian

# ── Pipeline behavior ─────────────────────────────────────────
GROUNDEDNESS_THRESHOLD=0.80
MAX_SYNTHESIS_RETRIES=2
SYNTHESIS_MODEL=claude-sonnet-4-6

# ── Storage (use 'local' for development) ─────────────────────
STORAGE_BACKEND=local
LOCAL_STORAGE_PATH=/tmp/meridian

# ── LLMOps (all optional in development) ──────────────────────
LANGCHAIN_TRACING_V2=false
# LANGCHAIN_API_KEY=
# MLFLOW_TRACKING_URI=http://localhost:5000
# WANDB_API_KEY=

# ── Application ───────────────────────────────────────────────
ENVIRONMENT=development
LOG_LEVEL=DEBUG
WORKERS=1
```

---

## Validating configuration

At startup, `src/config.py` validates all configuration using Pydantic Settings v2. Any missing required variable or invalid value causes a descriptive startup error:

```
pydantic_settings.SettingsError: 1 validation error for Settings
ANTHROPIC_API_KEY
  Field required [type=missing, input_url=...]
  Hint: Set ANTHROPIC_API_KEY in your .env file or environment.
```

To validate configuration without starting the server:

```bash
python -c "from src.config import settings; print('Configuration valid')"
```
