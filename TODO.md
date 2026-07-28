# TODO — Meridian

Tasks are organized by area and priority. Items marked `[blocked]` are waiting on a dependency. Items marked `[stretch]` are post-v1.0 scope.

Last updated: 2026-04-15

---

## Infrastructure and setup

- [x] Scaffold repo structure and pyproject.toml
- [x] Configure Docker Compose (PostgreSQL + pgvector + Redis)
- [x] Set up Alembic with initial migration
- [x] Configure pre-commit hooks (ruff, mypy, black)
- [x] Set up GitHub Actions skeleton
- [ ] Add `pip-audit` to CI for dependency vulnerability scanning
- [ ] Add `dependabot.yml` for automatic dependency update PRs
- [ ] Configure Sentry error capture in production build
- [ ] Add Docker multi-stage build (builder + runtime stages, minimize final image size)
- [ ] Write `docker-compose.prod.yml` with resource limits for Cloud Run parity

---

## Corpus ingestion

- [x] GDPR corpus from EUR-Lex (English, all 99 articles + recitals)
- [x] SOC-2 Trust Services Criteria (CC controls)
- [x] SEC Regulation S-P (safeguarding customer records)
- [ ] ISO-27001 Annex A controls (public interpretive text)
- [ ] CFPB Regulation P (privacy of consumer financial information)
- [ ] FederalRegister.gov polling script for live updates
- [ ] EU AI Act (English and French translations) [stretch]
- [ ] HIPAA Privacy Rule summary text [stretch]
- [ ] Add corpus version tracking: each chunk records the corpus version it was ingested from
- [ ] Write corpus freshness check: alert if any corpus hasn't been refreshed in 30 days

---

## RAG pipeline

- [x] Semantic chunking with LlamaIndex SemanticChunker
- [x] pgvector dense retrieval (cosine similarity)
- [x] BM25 keyword retrieval (PostgreSQL full-text search)
- [x] Reciprocal Rank Fusion
- [x] Cross-encoder reranking with ms-marco-MiniLM-L-6-v2
- [ ] Add metadata filtering UI in Streamlit (jurisdiction, effective date range)
- [ ] Benchmark retrieval latency at 100K, 500K, and 1M chunks
- [ ] Evaluate `bge-reranker-v2-m3` as a reranker upgrade (multilingual support)
- [ ] Add retrieval caching with Redis for repeated queries (TTL: 1 hour)
- [ ] Instrument each retrieval stage with separate LangSmith spans

---

## Document agent

- [x] Zero-shot document type classification (`facebook/bart-large-mnli`)
- [x] General NER (`dslim/bert-base-NER`)
- [x] Regulatory entity NER (zero-shot classification pipeline)
- [x] Extractive QA (`deepset/roberta-base-squad2`)
- [ ] Add `facebook/bart-large-cnn` summarization for documents over 10,000 tokens
- [ ] Handle password-protected PDFs (prompt user to remove protection)
- [ ] Add DOCX support via `python-docx` parser
- [ ] Evaluate `microsoft/Phi-3-mini-4k-instruct` as a local lightweight QA model for cost reduction

---

## Audio agent

- [x] Whisper large-v3 transcription via HF Inference API
- [x] VAD silence detection with `webrtcvad`
- [x] Transcript assembly with timestamps
- [ ] `pyannote/speaker-diarization-3.1` integration — currently using basic speaker detection only
- [ ] Handle video files (MP4, MOV) by extracting audio track with `ffmpeg`
- [ ] Add transcript editing UI in Streamlit (allow user to correct ASR errors)
- [ ] Cache transcripts by file hash to avoid re-transcription of same file

---

## Vision agent

- [x] BLIP2 image captioning for screenshots
- [x] VQA with `vilt-b32-finetuned-vqa`
- [x] Claude vision API for complex reasoning
- [ ] ColPali v1.2 integration for visual document retrieval — model loading slow on CPU, needs optimization
- [ ] Build ColPali regulatory image index (render regulatory PDFs to images, embed, store)
- [ ] Add PDF page rendering with `pdf2image` + `poppler`
- [ ] Evaluate `Qwen2-VL` as a single-model replacement for BLIP2 + VQA (more capable, fewer API calls)

---

## Data agent

- [x] CSV/Excel parsing with pandas
- [x] TAPAS table QA
- [x] Chronos-T5 time series forecasting
- [ ] Train tabular anomaly classifier on synthetic audit log dataset (currently using random forest placeholder)
- [ ] Add Excel pivot table support
- [ ] Handle multi-sheet Excel files (currently processes only first sheet)
- [ ] Add `xls` (legacy Excel) support via `xlrd`

---

## LangGraph and orchestration

- [x] `MeridianState` TypedDict with list reducers for parallel writes
- [x] Conditional routing via `classify_input` node
- [x] Parallel fan-out via `Send` API
- [x] `gate` node with groundedness check and retry logic
- [x] SQLite checkpointer for resumable execution (dev)
- [ ] Upgrade checkpointer to PostgreSQL for production (LangGraph `AsyncPostgresSaver`)
- [ ] Add human-in-the-loop pause point: pause before synthesis if confidence is low, await user approval
- [ ] Add interrupt handler: allow cancellation of in-progress jobs via `POST /cancel/{job_id}`
- [ ] Profile graph execution: identify which node is the p95 bottleneck

---

## Evaluation harness

- [x] RAGAS evaluation with faithfulness, answer relevancy, context precision/recall
- [x] Gap detection F1 harness (precision, recall, F1 with threshold sweep)
- [x] Golden GDPR QA dataset (200 examples)
- [x] Gap detection dataset (150 annotated examples from SEC enforcement actions)
- [x] Nightly GitHub Actions CI job
- [ ] Add LLM-as-judge evaluator for agent traces (`agent_judge.py`)
- [ ] Add SOC-2 specific golden dataset (currently GDPR-only)
- [ ] Raise F1 threshold from 0.82 to 0.85 (currently failing by 3 points — retrieval tuning needed)
- [ ] Add latency regression test: fail if P95 exceeds 3 minutes
- [ ] Build MLflow dashboard with metric trend charts (currently raw numbers only)
- [ ] Add eval comparison: baseline (dense-only) vs current (hybrid + rerank) to quantify improvement

---

## LLMOps

- [x] LangSmith tracing enabled for all nodes
- [x] MLflow experiment tracking for eval metrics
- [ ] W&B integration for real-time metric dashboards
- [ ] MLflow model registry: register each specialist model version
- [ ] Add model version pinning: each job records exact model versions used (in report appendix)
- [ ] Set up automated alert: Slack notification if nightly eval F1 drops > 5 points
- [ ] Add cost tracking: log token counts and API call costs per job to MLflow

---

## API

- [x] `POST /submit` — accept multipart upload
- [x] `GET /status/{job_id}` — poll job status
- [x] `GET /report/{job_id}` — retrieve completed report
- [x] `GET /health` — health check
- [ ] `DELETE /job/{job_id}` — cancel and delete a job
- [ ] `GET /metrics` — Prometheus-compatible endpoint for infra monitoring
- [ ] `GET /corpus/status` — show ingested corpora and their version/freshness dates
- [ ] Add rate limiting: 10 submissions/minute per API key (use Redis sliding window)
- [ ] Add API key management: `POST /keys` and `DELETE /keys/{key_id}` (admin only)
- [ ] Write OpenAPI spec validation tests with `schemathesis`

---

## Streamlit UI

- [x] File upload wizard
- [x] Job status polling
- [x] Report viewer
- [ ] LangSmith trace link per job
- [ ] Eval dashboard tab with RAGAS and F1 trend charts
- [ ] "Run demo case" button (auto-loads Meta privacy policy vs GDPR scenario)
- [ ] Download report button (PDF and JSON)
- [ ] Side-by-side diff view: submitted policy text vs regulatory requirement (per gap)

---

## Deployment

- [x] Local Docker Compose stack
- [ ] Production Dockerfile (multi-stage, non-root user, minimal image)
- [ ] GCP Cloud Run deployment scripts (`scripts/deploy_gcp.sh`)
- [ ] GCP Secret Manager integration for API keys
- [ ] Terraform configuration for GCP resources (Cloud Run, Cloud SQL, Secret Manager)
- [ ] Load test with Locust: 10 concurrent submissions; verify p95 ≤ 3 minutes
- [ ] CI: build and push Docker image to GCP Artifact Registry on main branch push

---

## Documentation and portfolio

- [x] README.md
- [x] ARCHITECTURE.md
- [x] FEATURES.md
- [x] PROJECT_PLAN.md
- [x] REQUIREMENTS.md
- [x] API.md
- [x] DATABASE.md
- [x] ROADMAP.md
- [x] CHANGELOG.md
- [x] TODO.md
- [ ] Record 5-minute Loom demo walkthrough
- [ ] Write case study 1: Meta privacy policy vs GDPR (public document)
- [ ] Write case study 2: Coinbase public disclosures vs SEC Regulation S-P
- [ ] Add shields.io badges for Python version, test coverage, nightly eval status
- [ ] Add `CONTRIBUTING.md`
- [ ] Add `CODE_OF_CONDUCT.md`

---

## Testing

- [ ] Unit tests for each RAG stage (chunker, dense retrieval, BM25, RRF, reranker)
- [ ] Unit tests for each agent node (mock HF API responses)
- [ ] Integration tests for full LangGraph pipeline (mock external APIs)
- [ ] API integration tests with `httpx` and `pytest-asyncio`
- [ ] Performance test: benchmark retrieval latency at scale
- [ ] Achieve 70%+ test coverage across `src/`
