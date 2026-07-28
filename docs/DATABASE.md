# Database — Meridian

Meridian uses three persistent storage systems:

| System | Role |
|---|---|
| **PostgreSQL 15 + pgvector** | Primary datastore — jobs, documents, chunks, embeddings, audit log |
| **Redis 7** | Celery task broker and result backend; short-lived job status cache |
| **GCP Cloud Storage (S3-compatible)** | Raw uploaded files and generated PDF reports |

---

## 1. PostgreSQL schema

### Overview

```
corpora
  └── documents
        └── chunks (with pgvector embeddings + tsvector for BM25)

jobs
  ├── job_files (junction: job ↔ document)
  ├── agent_extractions
  ├── retrieved_chunks (junction: job ↔ chunk, with rank metadata)
  ├── compliance_gaps
  └── reports
```

---

### Table: `corpora`

Tracks each ingested regulatory framework as a versioned corpus.

```sql
CREATE TABLE corpora (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug            TEXT NOT NULL UNIQUE,          -- 'gdpr', 'soc2', 'iso27001', 'sec_sp', 'cfpb'
    name            TEXT NOT NULL,                 -- 'General Data Protection Regulation (GDPR)'
    jurisdiction    TEXT NOT NULL,                 -- 'EU', 'US', 'global'
    version         TEXT NOT NULL,                 -- '2018-05-25', '2022', etc.
    source_url      TEXT,
    document_count  INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_refreshed  TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT corpora_slug_check CHECK (slug ~ '^[a-z0-9_]+$')
);

CREATE INDEX ON corpora (slug) WHERE is_active = TRUE;
```

---

### Table: `documents`

Stores each ingested source document (one row per regulatory PDF or file).

```sql
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id       UUID NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    source_url      TEXT,
    content_hash    TEXT NOT NULL,          -- SHA-256 of raw content (dedup key)
    page_count      INTEGER,
    token_count     INTEGER,
    language        TEXT NOT NULL DEFAULT 'en',
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    storage_key     TEXT,                   -- Cloud Storage path for raw file

    UNIQUE (corpus_id, content_hash)
);

CREATE INDEX ON documents (corpus_id);
CREATE INDEX ON documents (content_hash);
```

---

### Table: `chunks`

The core retrieval unit. Each row is a semantically chunked excerpt from a regulatory document, with both a vector embedding (for dense retrieval) and a tsvector column (for BM25 keyword retrieval).

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    corpus_id       UUID NOT NULL REFERENCES corpora(id),   -- denormalized for fast filter

    -- Regulatory metadata
    regulation      TEXT NOT NULL,          -- 'gdpr', 'soc2', etc.
    article         TEXT,                   -- 'Article 13', 'CC6.1', 'Annex A.8.1', etc.
    article_title   TEXT,                   -- 'Information to be provided where data is collected'
    jurisdiction    TEXT NOT NULL,
    effective_date  DATE,
    section_path    TEXT[],                 -- ['Chapter III', 'Section 2', 'Article 13', '2', 'a']

    -- Content
    chunk_index     INTEGER NOT NULL,       -- position within document
    content         TEXT NOT NULL,
    token_count     INTEGER NOT NULL,

    -- Retrieval indexes
    embedding       VECTOR(1536),           -- text-embedding-3-small dimensions
    ts_vector       TSVECTOR,               -- PostgreSQL full-text search

    -- Bookkeeping
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (document_id, chunk_index)
);

-- IVFFlat index for approximate nearest neighbor (ANN) search
-- lists = sqrt(chunk_count); rebuild when chunk count grows significantly
CREATE INDEX chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 200);

-- GIN index for full-text search (BM25 via ts_rank_cd)
CREATE INDEX chunks_ts_vector_idx
    ON chunks USING GIN (ts_vector);

-- Partial index for fast corpus-scoped filtering
CREATE INDEX chunks_corpus_regulation_idx
    ON chunks (corpus_id, regulation)
    WHERE embedding IS NOT NULL;

-- Trigger: keep ts_vector in sync with content
CREATE FUNCTION chunks_ts_vector_update() RETURNS TRIGGER AS $$
BEGIN
    NEW.ts_vector := to_tsvector('english', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chunks_ts_vector_trigger
    BEFORE INSERT OR UPDATE OF content ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_ts_vector_update();
```

**Index strategy notes:**

The IVFFlat index with `lists = 200` is calibrated for approximately 500,000 chunks. At query time, `probes = 10` (the default) scans 5% of lists, giving a good recall/latency tradeoff. If the corpus grows beyond 2M chunks, migrate to HNSW (`CREATE INDEX USING hnsw`) or move to Pinecone — the LlamaIndex abstraction makes this a config-level change.

BM25 scoring is approximated using PostgreSQL's `ts_rank_cd` function over the `ts_vector` column. True BM25 requires document frequency statistics not natively available in PostgreSQL; the `ts_rank_cd` approximation is close enough for the hybrid retrieval use case.

---

### Table: `jobs`

One row per compliance analysis submission.

```sql
CREATE TYPE job_status AS ENUM (
    'queued', 'processing', 'complete', 'failed', 'cancelled'
);

CREATE TABLE jobs (
    id                  TEXT PRIMARY KEY,           -- ULID (sortable, url-safe)
    status              job_status NOT NULL DEFAULT 'queued',
    regulation_scope    TEXT[] NOT NULL,            -- ['gdpr', 'soc2']
    report_formats      TEXT[] NOT NULL DEFAULT ARRAY['pdf', 'json'],
    language            TEXT NOT NULL DEFAULT 'en',
    options             JSONB NOT NULL DEFAULT '{}',

    -- Timing
    submitted_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_seconds    INTEGER GENERATED ALWAYS AS (
                            EXTRACT(EPOCH FROM (completed_at - started_at))::INTEGER
                        ) STORED,

    -- Progress
    current_stage       TEXT,
    stages_complete     TEXT[] NOT NULL DEFAULT '{}',

    -- Webhook
    webhook_url         TEXT,
    webhook_delivered   BOOLEAN NOT NULL DEFAULT FALSE,
    webhook_attempts    INTEGER NOT NULL DEFAULT 0,

    -- Results (summary; full results in compliance_gaps and reports tables)
    total_gaps          INTEGER,
    gaps_critical       INTEGER,
    gaps_major          INTEGER,
    gaps_minor          INTEGER,
    groundedness_pass_rate NUMERIC(4,3),

    -- Error state
    error_code          TEXT,
    error_message       TEXT,
    error_stage         TEXT,
    retry_count         INTEGER NOT NULL DEFAULT 0,

    -- LangSmith
    langsmith_run_id    TEXT,
    langsmith_trace_url TEXT,

    -- Expiry
    expires_at          TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '7 days')
);

CREATE INDEX ON jobs (status, submitted_at DESC);
CREATE INDEX ON jobs (expires_at) WHERE status != 'cancelled';

-- Partial index for jobs awaiting webhook delivery
CREATE INDEX ON jobs (webhook_delivered, status)
    WHERE webhook_url IS NOT NULL AND webhook_delivered = FALSE AND status = 'complete';
```

---

### Table: `job_files`

Maps submitted files to jobs. Each uploaded file gets its own row.

```sql
CREATE TYPE file_modality AS ENUM ('document', 'audio', 'image', 'tabular', 'unknown');

CREATE TABLE job_files (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    modality        file_modality NOT NULL DEFAULT 'unknown',
    mime_type       TEXT NOT NULL,
    size_bytes      BIGINT NOT NULL,
    storage_key     TEXT NOT NULL,          -- Cloud Storage path
    content_hash    TEXT NOT NULL,          -- SHA-256

    -- For audio files
    duration_seconds INTEGER,

    -- For document files
    page_count      INTEGER,

    -- Processing state
    processed       BOOLEAN NOT NULL DEFAULT FALSE,
    processed_at    TIMESTAMPTZ,
    processing_error TEXT,

    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON job_files (job_id);
```

---

### Table: `agent_extractions`

Stores the structured output from each specialist agent node for a given job.

```sql
CREATE TYPE agent_type AS ENUM ('doc_agent', 'audio_agent', 'vision_agent', 'data_agent');

CREATE TABLE agent_extractions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    file_id         UUID REFERENCES job_files(id),
    agent           agent_type NOT NULL,

    -- Core extraction payload (agent-specific structure)
    raw_text        TEXT,
    ner_entities    JSONB,      -- [{type, text, start, end, confidence}]
    qa_results      JSONB,      -- [{question, answer, confidence, source_chunk}]
    summary         TEXT,

    -- Audio-specific
    transcript      JSONB,      -- [{speaker, start, end, text}]
    speakers        TEXT[],

    -- Vision-specific
    image_caption   TEXT,
    vqa_results     JSONB,      -- [{question, answer, confidence}]
    colpali_matches JSONB,      -- [{chunk_id, similarity}]

    -- Tabular-specific
    table_summary   TEXT,
    tapas_answers   JSONB,      -- [{question, answer}]
    anomaly_scores  JSONB,      -- [{row_index, score, flags}]
    forecast_output JSONB,      -- [{timestamp, forecast, lower, upper}]

    -- Timing
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    duration_ms     INTEGER,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON agent_extractions (job_id);
CREATE INDEX ON agent_extractions (job_id, agent);
```

---

### Table: `retrieved_chunks`

Records which chunks were retrieved for each job and at what rank — essential for RAGAS evaluation and citation traceability.

```sql
CREATE TABLE retrieved_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    chunk_id        UUID NOT NULL REFERENCES chunks(id),

    -- Retrieval scores
    dense_rank      INTEGER,        -- rank in dense retrieval (1 = best)
    dense_score     NUMERIC(6,4),   -- cosine similarity
    bm25_rank       INTEGER,        -- rank in BM25 retrieval
    bm25_score      NUMERIC(6,4),   -- ts_rank_cd score
    rrf_score       NUMERIC(8,6),   -- reciprocal rank fusion score
    rerank_score    NUMERIC(6,4),   -- cross-encoder score
    final_rank      INTEGER,        -- rank after reranking (1 = top, passed to LLM)

    -- Usage
    passed_to_llm   BOOLEAN NOT NULL DEFAULT FALSE,   -- only top-5 passed
    used_in_gap     UUID[],         -- compliance_gap IDs that cite this chunk

    retrieved_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON retrieved_chunks (job_id);
CREATE INDEX ON retrieved_chunks (job_id, passed_to_llm) WHERE passed_to_llm = TRUE;
```

---

### Table: `compliance_gaps`

One row per identified compliance gap. This is the primary deliverable of the pipeline.

```sql
CREATE TYPE gap_severity AS ENUM ('critical', 'major', 'minor');

CREATE TABLE compliance_gaps (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id                  TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    -- Classification
    severity                gap_severity NOT NULL,
    framework               TEXT NOT NULL,          -- 'gdpr', 'soc2', etc.
    regulatory_article      TEXT NOT NULL,          -- 'GDPR Article 13(2)(a)'

    -- Regulatory side
    regulatory_chunk_id     UUID REFERENCES chunks(id),
    regulatory_requirement  TEXT NOT NULL,
    regulatory_quote        TEXT NOT NULL,

    -- Policy side
    policy_file_id          UUID REFERENCES job_files(id),
    policy_reference        TEXT,                   -- 'Section 4 — Data Retention'
    policy_text             TEXT,                   -- relevant policy excerpt, or NULL if absent

    -- Analysis
    gap_description         TEXT NOT NULL,
    severity_justification  TEXT NOT NULL,
    remediation             TEXT NOT NULL,

    -- Quality signals
    confidence              NUMERIC(4,3) NOT NULL,
    groundedness_score      NUMERIC(4,3) NOT NULL,
    is_verified             BOOLEAN NOT NULL DEFAULT FALSE,     -- passed hallucination gate
    is_uncertain            BOOLEAN NOT NULL DEFAULT FALSE,     -- marked uncertain by gate

    -- Ordering
    display_order           INTEGER,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON compliance_gaps (job_id);
CREATE INDEX ON compliance_gaps (job_id, severity);
CREATE INDEX ON compliance_gaps (job_id, framework);
```

---

### Table: `reports`

Stores generated report metadata and storage references.

```sql
CREATE TYPE report_format AS ENUM ('json', 'pdf', 'markdown');

CREATE TABLE reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    format          report_format NOT NULL,
    storage_key     TEXT NOT NULL,          -- Cloud Storage path
    size_bytes      BIGINT,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '7 days'),

    UNIQUE (job_id, format)
);

CREATE INDEX ON reports (job_id);
CREATE INDEX ON reports (expires_at);
```

---

### Table: `eval_runs`

Tracks nightly evaluation runs for historical analysis in MLflow and W&B.

```sql
CREATE TYPE eval_type AS ENUM ('ragas', 'gap_detection_f1', 'agent_judge', 'latency');

CREATE TABLE eval_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_type       eval_type NOT NULL,
    triggered_by    TEXT NOT NULL DEFAULT 'scheduled',   -- 'scheduled', 'manual', 'ci'
    pipeline_version TEXT NOT NULL,

    -- RAGAS metrics
    ragas_faithfulness          NUMERIC(5,4),
    ragas_answer_relevancy      NUMERIC(5,4),
    ragas_context_precision     NUMERIC(5,4),
    ragas_context_recall        NUMERIC(5,4),

    -- Gap detection metrics
    gap_f1                      NUMERIC(5,4),
    gap_precision               NUMERIC(5,4),
    gap_recall                  NUMERIC(5,4),
    gap_threshold               NUMERIC(4,3),

    -- Agent judge metrics
    routing_accuracy            NUMERIC(5,4),
    tool_use_quality_avg        NUMERIC(5,4),
    citation_accuracy_avg       NUMERIC(5,4),
    traces_evaluated            INTEGER,

    -- Latency
    p50_latency_seconds         NUMERIC(8,2),
    p95_latency_seconds         NUMERIC(8,2),
    p99_latency_seconds         NUMERIC(8,2),

    -- Result
    all_thresholds_passed       BOOLEAN NOT NULL DEFAULT FALSE,
    mlflow_run_id               TEXT,

    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    duration_seconds INTEGER
);

CREATE INDEX ON eval_runs (eval_type, started_at DESC);
```

---

### Table: `audit_log`

Immutable append-only log of all API requests for security auditing.

```sql
CREATE TABLE audit_log (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    api_key_prefix  TEXT,           -- first 8 chars of the API key (not the full key)
    method          TEXT NOT NULL,
    path            TEXT NOT NULL,
    status_code     INTEGER NOT NULL,
    job_id          TEXT,
    request_id      TEXT NOT NULL,
    ip_address      INET,
    user_agent      TEXT,
    duration_ms     INTEGER
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions (automate with pg_partman in production)
CREATE TABLE audit_log_2026_04
    PARTITION OF audit_log
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
```

---

## 2. Alembic migration strategy

Migrations are managed with Alembic. All schema changes must go through a migration file — no manual `ALTER TABLE` in production.

```bash
# Create a new migration
alembic revision --autogenerate -m "add_corpus_freshness_alert_field"

# Apply all pending migrations
alembic upgrade head

# Roll back one migration
alembic downgrade -1

# Show current revision
alembic current
```

**Conventions:**
- Migration files are named `YYYYMMDD_HHMMSS_description.py`
- Every `upgrade()` must have a corresponding `downgrade()`
- Index creation uses `IF NOT EXISTS` to be idempotent
- Migrations that rebuild large indexes (e.g., IVFFlat) are run during low-traffic windows and noted in `CHANGELOG.md`

---

## 3. Redis data structures

Redis is used exclusively for Celery and short-lived caching. Nothing in Redis is the source of truth — all durable state lives in PostgreSQL.

| Key pattern | Type | TTL | Content |
|---|---|---|---|
| `celery:task:{task_id}` | Hash | 24h | Celery task metadata (managed by Celery) |
| `job:status:{job_id}` | String | 1h | Cached job status (avoid repeated DB queries during polling) |
| `ratelimit:{api_key}:{minute}` | Counter | 60s | Sliding window rate limit counter |
| `chunk:query:{query_hash}` | String | 1h | Cached retrieval results for repeated queries |

---

## 4. GCP Cloud Storage layout

```
meridian-uploads/
  {job_id}/
    {file_id}_{original_filename}       -- raw uploaded file

meridian-reports/
  {job_id}/
    report.pdf
    report.json
    report.md
```

All objects use GCP default server-side AES-256 encryption. Object lifecycle rules delete files after 7 days, matching the job TTL in PostgreSQL.

---

## 5. Data retention and deletion

| Data type | Retention | Deletion mechanism |
|---|---|---|
| Raw uploaded files | 7 days | GCP lifecycle rule |
| Generated reports | 7 days | GCP lifecycle rule |
| Job records (PostgreSQL) | 7 days | Celery Beat scheduled task: `scripts/expire_jobs.py` |
| Chunk embeddings | Indefinite | Manual corpus refresh only |
| Audit log | 90 days | PostgreSQL partition drop |
| Eval run records | Indefinite | Manual cleanup only |

---

## 6. Performance notes

### pgvector tuning

The IVFFlat index performs approximate nearest neighbor search. Key parameters:

```sql
-- At query time, increase probes for better recall at the cost of latency
SET ivfflat.probes = 10;    -- default; search 10/200 lists = 5% of index
SET ivfflat.probes = 20;    -- higher recall, ~2× latency
```

For the current corpus size (~50K–500K chunks), `lists = 200` and `probes = 10` gives approximately 95% recall with sub-200ms query time. Rebuild the index when chunk count grows by more than 50% since the last build:

```sql
REINDEX INDEX CONCURRENTLY chunks_embedding_idx;
```

### Connection pooling

In production, use PgBouncer in transaction mode between the application and PostgreSQL:

```
App → PgBouncer (transaction mode, pool_size=20) → PostgreSQL
```

SQLAlchemy is configured with `pool_size=5, max_overflow=10` per worker to avoid exhausting the PgBouncer pool.

### Slow query log

PostgreSQL `log_min_duration_statement = 500ms` is enabled in production to catch slow queries. The most common slow queries are:
- IVFFlat scans with high `probes` values — tune down if latency is unacceptable
- `ts_rank_cd` on large chunk sets — ensure `ts_vector` GIN index is up to date
- Full `compliance_gaps` scan without `job_id` filter — always filter by job first
