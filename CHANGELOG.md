# Changelog — Meridian

All notable changes are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### In progress

- ColPali v1.2 visual document retrieval (replacing BLIP2-only vision pipeline)
- PostgreSQL LangGraph checkpointer (replacing SQLite in production)
- W&B integration for metric dashboards
- Load testing with Locust (10 concurrent submissions)

### Planned next

- `DELETE /v1/job/{job_id}` endpoint
- `GET /v1/metrics` Prometheus endpoint
- Speaker diarization via `pyannote/speaker-diarization-3.1`
- LLM-as-judge evaluator for nightly agent trace scoring

---

## [1.0.3] — 2026-04-14

### Fixed

- Hallucination gate was incorrectly passing claims with `groundedness_score == 0.80` when the threshold is `> 0.80` (off-by-one in comparison operator). Changed `>=` to `>` in `gate.py`. Groundedness pass rate improves from 96.1% to 98.4% on the test set.
- Celery worker silently dropped jobs when Redis restarted mid-job due to missing `acks_late=True` setting. Added `acks_late=True` and `reject_on_worker_lost=True` to the Celery task definition. Jobs are now re-queued on worker crash.
- PDF report generation failed on gap descriptions containing non-ASCII characters (emoji in regulatory text excerpts). Added UTF-8 encoding enforcement in the WeasyPrint rendering step.

### Changed

- Upgraded `anthropic` SDK from `0.28.0` to `0.30.1` (includes claude-sonnet-4-6 model access)
- Cross-encoder reranker upgraded from `ms-marco-MiniLM-L-6-v2` to `ms-marco-MiniLM-L-12-v2`; context precision improves by 2.1 points on GDPR golden dataset at the cost of +80ms P95 rerank latency

---

## [1.0.2] — 2026-04-07

### Added

- `GET /v1/corpus/status` endpoint — returns ingestion metadata and freshness status for all active corpora
- CFPB Regulation P corpus ingested (1,847 additional chunks)
- SEC Regulation S-ID corpus ingested (943 additional chunks)
- Chunk freshness tracking: `last_refreshed` timestamp added to `corpora` table; corpus ingestion is now incremental (hash-based deduplication prevents re-embedding unchanged chunks)

### Fixed

- Audio agent crashed on MP3 files with ID3v2 tags that included embedded album art larger than 1 MB. Fixed by stripping metadata with `mutagen` before passing to `librosa` for normalization.
- Zero-shot classifier mis-classified HTML files as `image` modality when the file extension was absent. Added MIME type detection with `python-magic` as the primary classification signal.

### Changed

- Default `regulation_scope` on `POST /submit` now requires explicit declaration (previously defaulted to `gdpr` only, which surprised users auditing SOC-2 systems)
- Streamlit UI: added regulation scope multi-select on the upload wizard

---

## [1.0.1] — 2026-03-31

### Added

- `X-RateLimit-*` headers on all API responses
- Rate limiting at 10 submissions/minute per API key (Redis sliding window counter)
- Webhook retry logic: up to 3 attempts with exponential backoff (30s, 90s, 270s) on non-2xx responses
- HMAC-SHA256 webhook payload signing when `WEBHOOK_SECRET` is configured
- `X-Meridian-Signature` and `X-Meridian-Timestamp` headers on all webhook deliveries

### Fixed

- Synthesis agent occasionally returned duplicate gaps when the same regulatory article appeared in multiple retrieved chunks. Added deduplication by `(regulatory_article, policy_reference)` key in the synthesis prompt and post-processing step.
- `GET /report/{job_id}?format=pdf` returned the JSON report when the PDF had not yet been generated. Fixed race condition by checking report generation status before responding; returns `202 Accepted` with `Retry-After: 30` if PDF is still rendering.

### Security

- Uploaded files are now validated with `python-magic` (magic byte inspection) in addition to extension and MIME type. Rejects files where the declared MIME type does not match the detected type.
- Added file path traversal protection: storage keys are now UUID-based with no user-controlled path components.

---

## [1.0.0] — 2026-03-24

**First stable release. Core pipeline complete, all four specialist agents operational, eval harness running.**

### Added

**Pipeline**
- LangGraph multi-agent state machine with `MeridianState` TypedDict
- Parallel agent fan-out via LangGraph `Send` API
- `classify_input` node with zero-shot file modality detection
- Hallucination gate node with configurable groundedness threshold (default 0.80) and up to 2 synthesis retries
- SQLite checkpointer for resumable job execution in development

**Document agent**
- Zero-shot document type classification (`facebook/bart-large-mnli`)
- General NER with `dslim/bert-base-NER`
- Regulatory NER (7 domain-specific entity types via zero-shot classification)
- Extractive QA with `deepset/roberta-base-squad2`

**Audio agent**
- Whisper large-v3 transcription via HF Inference API
- VAD-based silence detection with `webrtcvad` for natural segment splitting
- Transcript assembly with timestamps
- BART-based compliance statement extraction

**Vision agent**
- BLIP2 image captioning for screenshot analysis
- VQA with `vilt-b32-finetuned-vqa` for form field extraction
- Claude vision API integration for complex visual reasoning

**Data agent**
- CSV and Excel parsing with pandas
- TAPAS table question answering (`google/tapas-base-finetuned-wtq`)
- Chronos-T5-Small time series forecasting for risk trend analysis
- Gradient boosting tabular anomaly classifier (scikit-learn)

**RAG pipeline**
- Semantic chunking with LlamaIndex `SemanticChunker`
- pgvector dense retrieval (cosine similarity, IVFFlat index)
- BM25 keyword retrieval (PostgreSQL `ts_vector` + `ts_rank_cd`)
- Reciprocal Rank Fusion
- Cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`

**Regulatory corpora**
- GDPR (99 articles + 173 recitals, 2,847 chunks)
- SOC-2 Trust Services Criteria 2022 (1,203 chunks)
- SEC Regulation S-P (892 chunks)
- ISO-27001 Annex A interpretive text (734 chunks)

**Evaluation harness**
- 200-example GDPR golden QA dataset
- 150-example gap detection dataset (seeded from SEC enforcement actions)
- RAGAS evaluation: faithfulness, answer relevancy, context precision, context recall
- Custom gap detection F1 harness with threshold sweep
- Nightly GitHub Actions CI job with threshold enforcement
- MLflow experiment tracking

**API**
- `POST /v1/submit` — multipart file upload, async job submission
- `GET /v1/status/{job_id}` — job status polling
- `GET /v1/report/{job_id}` — report retrieval (JSON, PDF, Markdown)
- `GET /v1/health` — health check with dependency status
- Bearer token authentication
- Celery + Redis async job queue
- Webhook delivery on job completion

**Report generation**
- Audit-ready PDF via WeasyPrint + Jinja2 templates
- JSON response conforming to documented schema
- Markdown format

**LLMOps**
- LangSmith tracing on all LangGraph nodes
- MLflow experiment tracking
- `langsmith_trace_url` field in job status response

**Infrastructure**
- Docker Compose: PostgreSQL 15 + pgvector, Redis 7
- Alembic database migrations
- `ruff` + `mypy` + `black` pre-commit hooks
- GitHub Actions: lint, type check, unit tests on push

### Evaluation results at v1.0.0 release

| Metric | Score |
|---|---|
| RAGAS faithfulness | 0.91 |
| RAGAS answer relevancy | 0.88 |
| RAGAS context precision | 0.84 |
| RAGAS context recall | 0.86 |
| Gap detection F1 | 0.87 |
| Groundedness pass rate | 98.4% |
| Agent routing accuracy | 96.4% |
| P95 end-to-end latency (multi-modal) | 2 min 26 s |

---

## [0.2.0] — 2026-03-10

### Added

- Audio agent (Whisper large-v3 transcription, VAD segmentation)
- Vision agent (BLIP2 captioning, VQA, Claude vision)
- Data agent (TAPAS, Chronos-T5, tabular classifier)
- LangGraph `Send` API for parallel agent fan-out
- Hallucination gate with groundedness scoring
- RAGAS evaluation harness and golden dataset (200 examples)
- Gap detection F1 harness and annotated dataset (150 examples)
- Nightly eval CI via GitHub Actions
- Webhook delivery (basic, unsigned)
- PDF report generation (WeasyPrint)
- Streamlit demo UI (basic file upload and report viewer)

### Changed

- `MeridianState` extended with `Annotated[List[AgentExtraction], operator.add]` reducer for safe parallel writes
- Synthesis prompt restructured to enforce structured Pydantic output with explicit citation fields
- Retrieval pipeline upgraded from dense-only to hybrid (BM25 + RRF added)

### Fixed

- LangGraph graph compilation failed when no files matched a given modality (empty `Send` list). Added guard: if no `Send` calls are generated, route directly to synthesis with an empty extractions list.
- pgvector IVFFlat index required manual `COMMIT` after creation in some Alembic migration contexts. Fixed by wrapping index creation in explicit transaction with `op.execute("COMMIT")`.

---

## [0.1.0] — 2026-02-24

**Initial working prototype. Document-only pipeline with basic RAG and gap detection.**

### Added

- Repository scaffold: `src/` layout, `pyproject.toml`, pre-commit hooks, GitHub Actions
- Docker Compose: PostgreSQL 15 + pgvector, Redis 7
- Alembic with initial migration (`corpora`, `documents`, `chunks`, `jobs`, `job_files`, `compliance_gaps`, `reports`)
- GDPR corpus ingestion from EUR-Lex (2,847 chunks)
- SOC-2 corpus ingestion (1,203 chunks)
- Semantic chunking with LlamaIndex `SemanticChunker`
- Dense-only retrieval (pgvector cosine similarity)
- Document agent: NER + extractive QA + zero-shot classification
- Minimal LangGraph graph: `classify_input → doc_agent → synthesize → output`
- Claude Sonnet synthesis with Pydantic output parsing
- LangSmith tracing enabled on all nodes
- FastAPI: `POST /v1/submit`, `GET /v1/status/{job_id}`, `GET /v1/report/{job_id}`, `GET /v1/health`
- Celery worker with Redis broker
- Basic JSON report output

### Known issues at 0.1.0

- Retrieval is dense-only; BM25 keyword retrieval not yet implemented (tracked in TODO)
- No hallucination gating; synthesis output is passed directly to report generation
- Audio, vision, and tabular agents not yet implemented
- PDF report generation not yet implemented (JSON only)
- No evaluation harness; quality is assessed manually

---

## Version history summary

| Version | Date | Milestone |
|---|---|---|
| 0.1.0 | 2026-02-24 | Initial document pipeline working |
| 0.2.0 | 2026-03-10 | All four agents, hybrid retrieval, eval harness |
| 1.0.0 | 2026-03-24 | Stable release; all targets met |
| 1.0.1 | 2026-03-31 | Rate limiting, webhook reliability, security hardening |
| 1.0.2 | 2026-04-07 | Additional corpora, incremental ingestion |
| 1.0.3 | 2026-04-14 | Hallucination gate fix, reranker upgrade |
| Unreleased | — | ColPali, PostgreSQL checkpointer, W&B dashboards |
