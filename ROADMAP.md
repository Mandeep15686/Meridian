# Roadmap — Meridian

This roadmap covers planned releases from the initial MVP through a hypothetical production SaaS offering. Versions follow semantic versioning. All dates are targets, not commitments.

---

## v0.1 — Core pipeline (weeks 1–4)

**Theme:** Document-only pipeline with working RAG retrieval and a deployed demo.

**Target:** End of month 1

### Shipped

- [x] PostgreSQL + pgvector infrastructure (Docker Compose)
- [x] GDPR and SOC-2 corpus ingested and indexed
- [x] Semantic chunking with LlamaIndex
- [x] Hybrid retrieval (dense + BM25 + RRF + cross-encoder rerank)
- [x] Document agent: NER + extractive QA + zero-shot classification
- [x] Minimal LangGraph graph: classify → doc_agent → synthesize → output
- [x] Claude Sonnet synthesis with Pydantic-parsed gap output
- [x] LangSmith tracing on all nodes
- [x] FastAPI: POST /submit, GET /status, GET /report
- [x] Celery + Redis async job queue

### Metrics target

- RAGAS faithfulness ≥ 0.85
- Gap detection F1 ≥ 0.75 (v0.1 baseline)
- P95 latency (document only) ≤ 3 min

---

## v0.2 — Multimodal + evaluation harness (weeks 5–8)

**Theme:** All four specialist agents operational, parallel execution, and a proper evaluation harness with a golden dataset.

**Target:** End of month 2

### Planned

- [ ] Audio agent: Whisper large-v3 + speaker diarization
- [ ] Vision agent: BLIP2 + VQA + Claude vision
- [ ] Data agent: TAPAS + Chronos-T5 + tabular classifier
- [ ] LangGraph `Send` API for parallel fan-out
- [ ] Hallucination gate with groundedness scoring and retry
- [ ] Golden dataset: 200 GDPR Q&A pairs + 150 gap detection examples
- [ ] RAGAS eval harness with nightly GitHub Actions CI
- [ ] Gap detection F1 harness
- [ ] LLM-as-judge agent trace evaluator
- [ ] MLflow experiment tracking
- [ ] Webhook delivery with HMAC signing
- [ ] Streamlit demo UI (basic version)
- [ ] PDF report generation with WeasyPrint

### Metrics target

- RAGAS faithfulness ≥ 0.88
- RAGAS context precision ≥ 0.80
- Gap detection F1 ≥ 0.85
- Groundedness pass rate ≥ 98%
- P95 latency (multi-modal) ≤ 3 min

---

## v1.0 — Portfolio-ready release (weeks 9–12)

**Theme:** Production-hardened, deployed on GCP, polished portfolio artifacts, documentation complete.

**Target:** End of month 3

### Planned

- [ ] ColPali v1.2 visual document retrieval
- [ ] Additional regulatory corpora: ISO-27001, CFPB, SEC Regulation S-ID
- [ ] SEC corpus from EDGAR with real enforcement action examples in eval set
- [ ] PostgreSQL LangGraph checkpointer (replacing SQLite)
- [ ] Production Dockerfile (multi-stage, non-root, minimal image)
- [ ] GCP Cloud Run deployment with Secret Manager integration
- [ ] Load testing with Locust (10 concurrent submissions)
- [ ] Weights & Biases dashboards
- [ ] Complete Streamlit UI: eval dashboard, trace links, demo button
- [ ] Full documentation suite
- [ ] 5-minute Loom demo video
- [ ] 2 real-data case studies (Meta vs GDPR, Coinbase vs SEC)
- [ ] 70%+ test coverage
- [ ] Public GitHub repository with README badges

### Metrics target

- RAGAS faithfulness ≥ 0.91
- RAGAS answer relevancy ≥ 0.88
- Gap detection F1 ≥ 0.87
- Agent routing accuracy ≥ 96%
- Groundedness pass rate ≥ 98%
- P95 end-to-end latency ≤ 3 min
- Test coverage ≥ 70%

---

## v1.1 — Multilingual support (post-portfolio stretch)

**Theme:** Process regulatory documents in multiple languages using the HuggingFace Translation task.

**Target:** TBD (stretch goal after job search)

### Planned

- [ ] HuggingFace Translation pipeline (`Helsinki-NLP/opus-mt-*` family)
- [ ] EU AI Act corpus in English, French, German, Spanish, Italian
- [ ] Language auto-detection with `langdetect`
- [ ] Multilingual embeddings: evaluate `multilingual-e5-large` vs `paraphrase-multilingual-mpnet-base-v2`
- [ ] GDPR corpus in all 24 EU official languages (EUR-Lex provides these)
- [ ] Multilingual NER: `Babelscape/wikineural-multilingual-ner`
- [ ] Cross-lingual retrieval: French policy queried against English GDPR corpus via shared embedding space
- [ ] UI language selector in Streamlit

### Why this matters for the portfolio

Adding the Translation HuggingFace task (clearly visible in the screenshots) with a multilingual corpus demonstrates breadth beyond English-only NLP and directly addresses the EU AI Act — the most talked-about regulatory development in AI in 2025–2026. Saying "Meridian processes French and German regulatory submissions against the multilingual EU AI Act corpus" in an interview is a strong differentiator.

---

## v1.2 — Live regulatory monitoring (post-portfolio stretch)

**Theme:** Continuously monitor regulatory feeds and alert when a new rule affects a stored policy.

**Target:** TBD

### Planned

- [ ] FederalRegister.gov API polling (daily, via Celery Beat scheduler)
- [ ] EUR-Lex RSS feed monitoring for GDPR amendments
- [ ] Tavily web search tool as a LangGraph node for ad-hoc regulatory lookups
- [ ] Delta detection: compare newly ingested regulation chunks against stored policy analyses
- [ ] Alert system: send email or Slack notification when a new rule potentially creates a new gap
- [ ] Corpus change log: track regulatory updates with effective dates
- [ ] Re-analysis scheduling: automatically queue a re-analysis job when relevant corpus changes

---

## v2.0 — Multi-tenant SaaS (hypothetical production roadmap)

**Theme:** Turn Meridian into a multi-tenant product with proper access control, billing, and enterprise integrations.

**Target:** TBD (would require significant additional engineering beyond portfolio scope)

### Planned

- [ ] Multi-tenant data isolation (row-level security in PostgreSQL)
- [ ] OAuth2 authentication (replace simple Bearer token with proper auth)
- [ ] Stripe billing integration (per-job or subscription pricing)
- [ ] Enterprise SSO: SAML 2.0 / OKTA integration
- [ ] Custom policy library: upload and version-control internal policies per tenant
- [ ] Role-based access control: admin, analyst, viewer
- [ ] Slack and Teams integration: deliver reports directly to channels
- [ ] Jira integration: create tickets for each identified gap automatically
- [ ] Custom regulatory framework upload: compliance teams can add their own regulatory text
- [ ] API SLA: 99.9% uptime commitment with status page
- [ ] SOC-2 Type II certification for the platform itself (dogfooding)

---

## Discarded ideas

### Why not a Slack bot interface

Building the primary interface as a Slack bot was considered for accessibility but rejected. Slack bots have significant UX friction for file uploads (multiple steps, size limits), and portfolio evaluators expect to see a proper API and demo UI. A Slack integration is planned for v2.0 as a delivery channel, not the primary interface.

### Why not fine-tuning models

Fine-tuning a domain-specific NER or QA model would be technically impressive, but requires a labeled training dataset that would take months to build at meaningful scale. The portfolio value of demonstrating production patterns (RAG, multi-agent, eval, LLMOps) is higher than the value of a fine-tuned model with limited generalization. Fine-tuning is flagged as a potential v2.0 enhancement if commercial deployment generates enough domain-specific training data.

### Why not GraphRAG

Microsoft GraphRAG (knowledge graph + RAG) was evaluated for the regulatory corpus. Regulatory text has strong entity relationships (GDPR Article 6 references Article 9, which references Article 89) that a graph would capture well. However, GraphRAG has significantly higher ingestion cost and latency, and the portfolio timeline doesn't allow for the experimentation needed to tune it. It's flagged as a potential v1.2 enhancement and a good talking point in interviews.
