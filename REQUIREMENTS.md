# Requirements — Meridian

---

## 1. Functional requirements

### FR-001 — Document ingestion

**Priority:** P0 (must have)

The system must accept regulatory and policy documents in the following formats:

- PDF (text-native and scanned/image-based)
- DOCX and DOC (Microsoft Word)
- Plain text (.txt, .md)
- HTML (web-exported policies)

The system must extract text from all formats, chunk the content using semantic boundaries, generate vector embeddings, and store both the raw text and the vector representation in the database. Scanned PDFs must be processed through OCR before chunking.

**Acceptance criteria:**
- A 50-page PDF is ingested and chunked within 60 seconds
- Chunk metadata includes: `source_file`, `page_number`, `chunk_index`, `token_count`
- All chunks are retrievable by semantic similarity query within 500ms

---

### FR-002 — Audio ingestion and transcription

**Priority:** P0 (must have)

The system must accept audio recordings in the following formats:

- MP3, WAV, M4A, FLAC, OGG
- Maximum file size: 500 MB per upload
- Maximum duration: 4 hours per file

The system must transcribe audio to text using Whisper large-v3, segment by speaker where possible, and extract regulatory statements from the transcript for downstream analysis.

**Acceptance criteria:**
- A 10-minute audio file is transcribed within 90 seconds
- Transcript includes speaker labels where diarization is possible
- Word error rate (WER) on regulatory domain audio is ≤ 15%

---

### FR-003 — Image and screenshot ingestion

**Priority:** P1 (should have)

The system must accept image inputs in PNG, JPG, JPEG, WEBP, and PDF-rendered-as-image formats. Image inputs include:

- Screenshots of compliance dashboards and SaaS tools
- Scanned compliance forms and checklists
- Photographed regulatory posters or physical documents

The system must extract text and structured information from images using image-to-text models, and route image-based document pages through ColPali for visual document retrieval.

**Acceptance criteria:**
- A screenshot of a compliance dashboard is analyzed and relevant text extracted within 30 seconds
- Visual document retrieval returns the correct regulatory page with ≥ 80% recall on test set

---

### FR-004 — Structured data ingestion

**Priority:** P1 (should have)

The system must accept structured tabular data in:

- CSV files
- Excel files (.xlsx, .xls)
- JSON arrays

Tabular inputs are typically audit logs, compliance metric tables, or historical risk scoring data. The system must parse tables, generate natural language summaries, and route them to the tabular QA and time series agents.

**Acceptance criteria:**
- A 10,000-row CSV is parsed and summarized within 20 seconds
- Table QA agent correctly answers structured questions about the data in ≥ 85% of test cases

---

### FR-005 — Multi-agent orchestration

**Priority:** P0 (must have)

The system must route each input to one or more specialist agents based on input type. Multiple agents must run in parallel for mixed-modality submissions. The orchestration layer must:

- Classify each uploaded file into a modality category (document, audio, image, tabular)
- Fan out to the relevant specialist agents simultaneously using async parallel execution
- Collect all agent outputs into a shared state object
- Pass the merged state to the synthesis agent

**Acceptance criteria:**
- A submission with 1 PDF + 1 MP3 + 1 CSV runs all three relevant agents in parallel, not sequentially
- Routing accuracy ≥ 95% on test set (100 mixed-type submissions)
- Total latency for multi-modal submission is less than 1.5× single-modal latency

---

### FR-006 — Regulatory corpus retrieval (RAG)

**Priority:** P0 (must have)

The system must maintain a searchable corpus of regulatory text and retrieve relevant clauses to inform compliance analysis. Retrieval must use a three-stage pipeline:

1. Dense vector search (cosine similarity via pgvector or Pinecone)
2. BM25 keyword search (PostgreSQL full-text search)
3. Reciprocal Rank Fusion of results from stages 1 and 2
4. Cross-encoder reranking (top-20 candidates → top-5 returned to LLM)

Supported regulatory frameworks in v1.0:

- GDPR (General Data Protection Regulation)
- SOC-2 Type II Trust Services Criteria
- ISO-27001 (Annex A controls)
- SEC Regulation S-P and Regulation S-ID
- CFPB Consumer Financial Protection rules

**Acceptance criteria:**
- Context precision ≥ 0.80 on RAGAS golden dataset
- Context recall ≥ 0.82 on RAGAS golden dataset
- P95 retrieval latency (all three stages) ≤ 800ms

---

### FR-007 — Compliance gap detection

**Priority:** P0 (must have)

The system must compare extracted policy content against retrieved regulatory requirements and identify specific gaps. For each identified gap, the system must provide:

- A plain-language description of the gap
- The specific regulatory article or clause that is not satisfied
- The specific section of the submitted policy that is deficient or absent
- A confidence score (0.0–1.0)
- A suggested remediation action

**Acceptance criteria:**
- F1 ≥ 0.85 on the 150-example gap detection golden dataset
- Each gap is accompanied by at least one regulatory citation
- No gap is reported without a corresponding source chunk (hallucination gate enforced)

---

### FR-008 — Hallucination gating

**Priority:** P0 (must have)

Before generating the final report, the system must verify every factual claim against its cited source chunk using semantic similarity. Claims that score below a configurable groundedness threshold (default: 0.80) must be re-synthesized or removed. The gate must retry synthesis up to 2 times before flagging a claim as uncertain in the report.

**Acceptance criteria:**
- Groundedness pass rate ≥ 98% on 100-submission test set
- Every claim in the final report has a traceable source citation
- Claims marked as uncertain are clearly labeled in the output

---

### FR-009 — Report generation

**Priority:** P0 (must have)

The system must generate a compliance report containing:

- Executive summary (2–3 paragraphs)
- List of all identified compliance gaps with severity classification (critical, major, minor)
- Per-gap detail: description, regulatory citation, policy reference, remediation suggestion
- Summary statistics (total gaps by framework, by severity)
- Appendix: raw retrieved regulatory clauses used in analysis

Report formats:

- PDF (audit-ready, printable, suitable for legal filing)
- JSON (machine-readable, for downstream integrations)
- Markdown (for developer workflows)

**Acceptance criteria:**
- PDF report is generated within 30 seconds after synthesis is complete
- PDF is parseable and passes PDF/A validation
- JSON report conforms to the documented response schema (see API.md)

---

### FR-010 — Async job processing

**Priority:** P0 (must have)

All compliance submissions must be processed asynchronously. The API must:

- Accept a submission and immediately return a `job_id`
- Allow the client to poll job status via `GET /status/{job_id}`
- Deliver the completed report to a client-provided webhook URL upon completion
- Retain job results for 7 days before deletion

**Acceptance criteria:**
- POST `/submit` returns within 500ms regardless of submission size
- Webhook delivery occurs within 60 seconds of job completion
- Job status transitions: `queued → processing → complete | failed`

---

### FR-011 — Evaluation harness

**Priority:** P1 (should have)

The system must include an automated evaluation harness that runs on demand and nightly:

- RAGAS evaluation on the 200-example golden QA dataset
- F1 evaluation on the 150-example gap detection dataset
- LLM-as-judge scoring on all LangSmith traces from the past 24 hours
- Results logged to MLflow with metric history

**Acceptance criteria:**
- Nightly eval runs via GitHub Actions without human intervention
- Results are visible in MLflow at a stable URL
- Any metric regression of > 5 percentage points triggers a GitHub issue

---

### FR-012 — Multilingual regulatory support (v1.1 stretch)

**Priority:** P2 (nice to have)

The system should be able to process regulatory documents in French, German, Spanish, and Italian using the HuggingFace Translation task, and retrieve against multilingual regulatory corpora including EU AI Act translations.

---

## 2. Non-functional requirements

### NFR-001 — Latency

| Operation | P50 | P95 | P99 |
|---|---|---|---|
| POST `/submit` (acceptance) | 100ms | 300ms | 500ms |
| Single-modality job (document only) | 90s | 150s | 180s |
| Multi-modality job (doc + audio + image) | 120s | 180s | 210s |
| GET `/status/{job_id}` | 20ms | 80ms | 120ms |
| GET `/report/{job_id}` | 50ms | 150ms | 200ms |
| Hybrid retrieval (all 3 stages) | 200ms | 600ms | 800ms |

---

### NFR-002 — Throughput

- The system must support 10 concurrent job submissions in the portfolio/demo deployment
- The Celery worker pool must scale horizontally by adding workers without code changes
- A single API instance must handle 50 status poll requests per second

---

### NFR-003 — Reliability

- The API must return a 200 status response for valid submissions 99.5% of the time over any 24-hour period
- Failed jobs must be retried up to 3 times before marking as failed and notifying the webhook
- Database connections must use connection pooling (pgBouncer or SQLAlchemy pool) to handle concurrency

---

### NFR-004 — Security

- All API endpoints must require authentication (Bearer token)
- API keys must be stored in environment variables or GCP Secret Manager, never in code
- Uploaded files must be validated for type and size before processing
- Files must be stored with server-side encryption at rest (GCP Cloud Storage default encryption)
- No user-submitted content may be logged in plain text in production

---

### NFR-005 — Observability

- Every LangGraph node execution must be traced in LangSmith
- All API requests must be logged with: `timestamp`, `method`, `path`, `status_code`, `job_id`, `duration_ms`
- MLflow must track experiment metrics for every eval run
- W&B must display real-time dashboards for model performance trends
- Application errors must be captured with full stack traces (via Sentry or equivalent)

---

### NFR-006 — Maintainability

- All Python code must pass `ruff` linting and `mypy` type checking with zero errors
- Test coverage must be ≥ 70% across `src/`
- All public functions and classes must have docstrings
- Database schema changes must be handled via Alembic migrations (no manual schema edits)
- Dependency versions must be pinned in `requirements.txt` and verified via `pip-audit`

---

### NFR-007 — Portability

- The entire stack must run locally with a single `docker compose up` command
- The API must be deployable to any container runtime (GCP Cloud Run, AWS ECS, local Docker)
- Configuration must be entirely through environment variables (12-factor app compliance)

---

## 3. System requirements

### Development environment

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB free | 50 GB free |
| Python | 3.11 | 3.12 |
| Docker | 24.0+ | latest |
| OS | macOS 13+, Ubuntu 22.04+, Windows 11 (WSL2) | Ubuntu 22.04 |

### Production environment (GCP Cloud Run)

| Component | Spec |
|---|---|
| CPU | 2 vCPU per instance |
| Memory | 4 GB per instance |
| Min instances | 1 (to avoid cold start during demo) |
| Max instances | 5 |
| Timeout | 15 minutes per request |
| Concurrency | 10 requests per instance |

---

## 4. Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key for synthesis LLM |
| `HF_API_TOKEN` | Yes | — | HuggingFace Inference API token |
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `REDIS_URL` | Yes | `redis://localhost:6379/0` | Celery broker and result backend |
| `PINECONE_API_KEY` | No | — | If set, Pinecone is used instead of pgvector |
| `PINECONE_INDEX_NAME` | No | `meridian` | Pinecone index name |
| `LANGCHAIN_API_KEY` | No | — | LangSmith tracing (optional in dev) |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `MLFLOW_TRACKING_URI` | No | `http://localhost:5000` | MLflow server |
| `WANDB_API_KEY` | No | — | Weights & Biases (optional) |
| `MERIDIAN_API_KEY` | Yes | — | Bearer token for API authentication |
| `WEBHOOK_SECRET` | No | — | HMAC secret for signed webhook payloads |
| `MAX_FILE_SIZE_MB` | No | `500` | Maximum upload file size |
| `GROUNDEDNESS_THRESHOLD` | No | `0.80` | Minimum groundedness score for claims |
| `MAX_SYNTHESIS_RETRIES` | No | `2` | Max retries for hallucination gating |
| `LOG_LEVEL` | No | `INFO` | Application log level |
| `ENVIRONMENT` | No | `development` | `development`, `staging`, `production` |

---

## 5. Third-party service dependencies

| Service | Purpose | Free tier sufficient? |
|---|---|---|
| Anthropic API | Claude Sonnet for synthesis | No — pay-per-token |
| HuggingFace Inference API | Whisper, NER, reranker, VQA | Yes for dev; throttled |
| Pinecone | Cloud vector store (optional) | Yes (Starter plan) |
| LangSmith | Agent trace logging and eval | Yes (Developer plan) |
| MLflow (self-hosted) | Experiment tracking | Yes — free and local |
| Weights & Biases | Metric dashboards | Yes (free tier) |
| GCP Cloud Run | Deployment | Yes — free tier sufficient for demo |
| GCP Secret Manager | API key storage in prod | Yes |
| EUR-Lex | GDPR corpus | Free — public API |
| EDGAR | SEC rules corpus | Free — public API |
| FederalRegister.gov | Live regulatory updates | Free — public API |
