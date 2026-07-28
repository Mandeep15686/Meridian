# Project plan — Meridian

## Overview

**Project:** Meridian — Multimodal Regulatory Intelligence & Compliance Automation  
**Type:** Solo AI engineering portfolio project  
**Duration:** 12 weeks (3 months)  
**Goal:** A production-grade, deployable system demonstrating multi-agent orchestration, multimodal RAG, LLMOps, and a full evaluation harness — targeted at mid-level AI engineering roles in 2026.

---

## Objectives

### Primary objectives

- Build a working multi-agent pipeline that processes PDF, audio, image, and tabular inputs through a LangGraph state machine
- Achieve F1 > 0.85 on compliance gap detection against a human-labeled golden dataset
- Deploy the system as a containerized API on GCP Cloud Run with p95 latency under 3 minutes
- Build a full evaluation harness (RAGAS + custom F1 + LLM-as-judge) with nightly CI runs

### Secondary objectives

- Demonstrate LLMOps maturity: MLflow experiment tracking, LangSmith tracing, W&B dashboards
- Produce a portfolio-quality repository with clean architecture, documentation, and a demo video
- Create real-world eval data from publicly available regulatory enforcement actions
- Cover 12+ distinct HuggingFace task types across four modality categories

### Non-goals (explicitly out of scope for v1.0)

- Multi-tenant SaaS — single-user or team deployment only
- Fine-tuned models — use off-the-shelf HF models and Claude via API
- Real-time streaming responses — async job model with webhook delivery
- Mobile app — Streamlit demo UI is sufficient for portfolio purposes

---

## Timeline and milestones

### Phase 1 — Infrastructure and ingestion (weeks 1–4)

**Goal:** A working end-to-end pipeline for a single modality (documents) with RAG retrieval and basic agent output.

#### Week 1 — Setup and corpus ingestion

- [ ] Scaffold repo structure, pyproject.toml, pre-commit hooks, GitHub Actions skeleton
- [ ] Configure Docker Compose with PostgreSQL + pgvector + Redis
- [ ] Write Alembic migration for initial schema (documents, chunks, embeddings tables)
- [ ] Build corpus ingestion script: GDPR from EUR-Lex, SOC-2, initial SEC rules
- [ ] Verify pgvector similarity queries on ingested chunks
- [ ] Set up MLflow tracking server (local, Dockerized)

**Milestone M1:** Corpus of ≥ 50,000 regulatory chunks indexed and queryable. ✓

#### Week 2 — RAG pipeline

- [ ] Implement semantic chunking with `SemanticChunker` (LlamaIndex)
- [ ] Add chunk metadata: `regulation_name`, `article_number`, `jurisdiction`, `effective_date`
- [ ] Build dense retrieval with pgvector (cosine similarity)
- [ ] Add BM25 keyword retrieval (PostgreSQL full-text search)
- [ ] Implement Reciprocal Rank Fusion over dense + BM25 results
- [ ] Add cross-encoder reranking with `ms-marco-MiniLM-L-6-v2` (top-20 → top-5)
- [ ] Write unit tests for each retrieval stage

**Milestone M2:** Hybrid retrieval pipeline returning cited chunks with metadata. ✓

#### Week 3 — Document agent and basic LangGraph graph

- [ ] Define `MeridianState` TypedDict
- [ ] Implement `doc_agent` node: NER (`dslim/bert-base-NER`) + document QA (`deepset/roberta-base-squad2`) + RAG retrieval
- [ ] Implement `classify_input` router node (zero-shot classification via HF)
- [ ] Build minimal LangGraph graph: `classify_input → doc_agent → synthesize → output`
- [ ] Wire Claude Sonnet as the synthesis LLM
- [ ] Integrate LangSmith tracing — every node logged

**Milestone M3:** Document-only pipeline runs end-to-end with cited output. ✓

#### Week 4 — Audio agent and Whisper integration

- [ ] Implement `audio_agent` node: Whisper large-v3 transcription via HF Inference API
- [ ] Add speaker diarization via `wav2vec2` audio classification
- [ ] Segment transcript by speaker, extract regulatory statements
- [ ] Route audio outputs into shared RAG retrieval (transcribed text as query)
- [ ] Write integration tests for audio pipeline

**Milestone M4:** Audio inputs processed and merged with document pipeline. ✓

---

### Phase 2 — Multi-agent completion and evaluation harness (weeks 5–8)

**Goal:** All four specialist agents wired, parallel execution via LangGraph `Send`, and full evaluation harness operational.

#### Week 5 — Vision agent

- [ ] Implement `vision_agent` node: ColPali v1.2 for visual document retrieval
- [ ] Add screenshot analysis via Claude vision (image-text-to-text)
- [ ] Implement form VQA with `vilt-b32-finetuned-vqa` for structured form extraction
- [ ] Handle multi-page PDF rendering to images for ColPali ingestion
- [ ] Write unit tests for vision pipeline

**Milestone M5:** Image and screenshot inputs processed with extracted text and visual QA. ✓

#### Week 6 — Data agent and parallel execution

- [ ] Implement `data_agent` node: TAPAS for table QA, Chronos-T5 for time series risk scoring
- [ ] Build tabular classification for audit log anomaly detection (scikit-learn baseline + HF model)
- [ ] Upgrade LangGraph graph to use `Send` API for parallel agent execution
- [ ] Implement conditional routing: fan-out to N relevant agents based on input classifier
- [ ] Add `gate` node: groundedness check on all synthesis claims before output
- [ ] Measure end-to-end latency improvement from parallelism

**Milestone M6:** All four agents running in parallel with conditional routing. ✓

#### Week 7 — Evaluation harness (RAGAS + F1)

- [ ] Build golden dataset: 200 Q&A pairs from GDPR/SOC-2 documents
- [ ] Build gap detection dataset: 150 annotated gap examples from SEC enforcement actions
- [ ] Implement `ragas_eval.py`: faithfulness, answer relevancy, context precision/recall
- [ ] Implement `gap_detection_eval.py`: precision, recall, F1 with threshold sweep
- [ ] Log all eval results to MLflow with parameter tracking
- [ ] Set up nightly GitHub Actions job to run eval suite
- [ ] Write failing tests that enforce minimum thresholds (F1 > 0.80 initially)

**Milestone M7:** Eval harness running nightly with results logged to MLflow. ✓

#### Week 8 — LLM-as-judge and LLMOps

- [ ] Implement `agent_judge.py`: LLM-as-judge evaluator that scores LangSmith traces
- [ ] Define scoring rubric: routing correctness (0–5), tool use quality (0–5), citation accuracy (0–5)
- [ ] Wire W&B for metric dashboards and run comparison tables
- [ ] Implement hallucination gating in `gate` node with retry-up-to-2 logic
- [ ] Add MLflow model registry entries for each specialist model version
- [ ] Raise F1 threshold to 0.85 and tune retrieval to hit it

**Milestone M8:** Full LLMOps loop operational; F1 > 0.85 on gap detection. ✓

---

### Phase 3 — API, deployment, and portfolio polish (weeks 9–12)

**Goal:** Production-ready API, deployed service, polished portfolio artifacts.

#### Week 9 — FastAPI backend

- [ ] Build FastAPI routes: `/submit`, `/status/{job_id}`, `/report/{job_id}`, `/health`
- [ ] Implement Celery task queue for async job processing
- [ ] Add Pydantic v2 request/response schemas with full validation
- [ ] Implement webhook delivery for job completion
- [ ] Add API key authentication (simple Bearer token for portfolio)
- [ ] Write API integration tests with `httpx` and `pytest`

**Milestone M9:** FastAPI service passes all integration tests; async jobs work end-to-end. ✓

#### Week 10 — Streamlit demo UI

- [ ] Build multi-step upload wizard (files, regulation scope, options)
- [ ] Add real-time job status polling with progress indicators
- [ ] Render compliance report inline with highlighted gaps
- [ ] Add LangSmith trace viewer link per job
- [ ] Add eval dashboard tab showing RAGAS and F1 trend charts
- [ ] Add "run demo case" button that auto-loads a sample privacy policy

**Milestone M10:** Streamlit UI complete and demo scenario runs without manual steps. ✓

#### Week 11 — Containerization and deployment

- [ ] Write production Dockerfile (multi-stage, minimal image)
- [ ] Write docker-compose.yml for full local stack
- [ ] Configure GCP Cloud Run deployment (Terraform or gcloud CLI scripts)
- [ ] Set up environment variable management via GCP Secret Manager
- [ ] Implement health checks and graceful shutdown
- [ ] Load test with Locust: 10 concurrent submissions, measure p95 latency
- [ ] Set up Cloud Run autoscaling (min 0, max 5 instances)

**Milestone M11:** Service deployed on GCP Cloud Run with a public URL. ✓

#### Week 12 — Portfolio polish

- [ ] Write `ARCHITECTURE.md` with detailed design decisions and alternatives considered
- [ ] Write all remaining documentation files
- [ ] Record 5-minute Loom demo walkthrough
- [ ] Add demo badge and video link to README
- [ ] Write at least 2 real-data case studies (e.g., Meta privacy policy vs. GDPR, Coinbase vs. SEC)
- [ ] Clean up code: remove debug prints, add docstrings, run `ruff` and `mypy`
- [ ] Ensure 70%+ test coverage
- [ ] Final review of all portfolio artifacts

**Milestone M12:** Repository is portfolio-ready and publicly visible on GitHub. ✓

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| HF Inference API rate limits on free tier | High | Medium | Cache model outputs aggressively; use local models as fallback |
| Whisper transcription quality on regulatory audio | Medium | Medium | Post-process with diarization; allow manual correction |
| LangGraph version breaking changes | Low | High | Pin to exact versions; read changelog before upgrades |
| Corpus ingestion blocked by source website | Medium | Low | Mirror corpus locally; use EDGAR API for SEC content |
| p95 latency exceeds 3-minute target | Medium | Medium | Parallelize agents earlier; add caching on repeated doc types |
| Golden dataset quality too low for meaningful eval | Medium | High | Seed dataset from real SEC enforcement actions, not synthetic data |
| GCP Cloud Run cold start adds latency | High | Low | Keep min instances at 1 for demo period; document in README |

---

## Success metrics

At project completion, Meridian should demonstrate:

| Metric | Target |
|---|---|
| RAGAS faithfulness | ≥ 0.88 |
| RAGAS answer relevancy | ≥ 0.85 |
| Gap detection F1 | ≥ 0.85 |
| Groundedness pass rate | ≥ 98% |
| P95 end-to-end latency | ≤ 3 min |
| Agent routing accuracy | ≥ 95% |
| Test coverage | ≥ 70% |
| Nightly eval CI passing | ✓ |
| Deployed public URL | ✓ |
| Demo video recorded | ✓ |

---

## External dependencies

| Dependency | Type | Risk if unavailable |
|---|---|---|
| Anthropic API | Required | Pipeline cannot synthesize; no fallback |
| HuggingFace Inference API | Required for specialist models | Fallback to local `transformers` |
| EUR-Lex GDPR source | Required for corpus | Mirror locally after first download |
| EDGAR full-text search API | Required for SEC corpus | Cache all queries |
| Pinecone | Optional (pgvector is default) | pgvector handles everything in dev |
| GCP Cloud Run | Required for deployment | Local Docker alternative for demo |
