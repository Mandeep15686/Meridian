# Architecture — Meridian

This document covers the full technical design of Meridian: component layout, data flow, agent internals, retrieval pipeline, LLMOps setup, and every significant design decision with the alternatives considered and rejected. It is written for an engineer joining the project, or an interviewer evaluating technical depth.

---

## 1. System overview

Meridian is structured as a set of loosely coupled services, each with a single responsibility:

```
                        ┌─────────────────────────────────┐
                        │         Client (browser/API)     │
                        └────────────────┬────────────────┘
                                         │ HTTPS
                        ┌────────────────▼────────────────┐
                        │       FastAPI  (port 8000)       │
                        │  Auth · validation · job mgmt    │
                        └────────────────┬────────────────┘
                                         │ Enqueue
                        ┌────────────────▼────────────────┐
                        │     Redis (Celery broker)        │
                        └────────────────┬────────────────┘
                                         │ Dequeue
                        ┌────────────────▼────────────────┐
                        │      Celery worker               │
                        │   Runs MeridianGraph.invoke()    │
                        └────────────────┬────────────────┘
                                         │ State machine
                        ┌────────────────▼────────────────┐
                        │       LangGraph pipeline         │
                        │  Orchestrates all agents + RAG   │
                        └──┬──────┬──────┬──────┬─────────┘
                           │      │      │      │
                     ┌─────▼┐  ┌──▼──┐ ┌▼───┐ ┌▼────┐
                     │ Doc  │  │Audio│ │Vis │ │Data │
                     │agent │  │agent│ │agt │ │agent│
                     └──────┘  └─────┘ └────┘ └─────┘
                           │      │      │      │
                        ┌──▼──────▼──────▼──────▼────┐
                        │   Shared external services  │
                        │ pgvector · HF API · Claude  │
                        └─────────────────────────────┘
```

All persistent state lives in PostgreSQL. Redis is used only for the Celery task queue and short-lived job metadata cache. S3 (GCP Cloud Storage) holds raw uploaded files.

---

## 2. LangGraph state machine

### Why LangGraph over alternatives

**Considered:** Raw LangChain `AgentExecutor`, OpenAI function calling, raw Python async, CrewAI, AutoGen.

**Chosen:** LangGraph.

The core requirement is a stateful graph where nodes share a typed state object, edges are conditional, and some branches execute in parallel. LangGraph provides exactly this as a first-class primitive. `AgentExecutor` is a single-agent abstraction and doesn't model multi-agent fan-out well. CrewAI and AutoGen are higher-level frameworks that hide the graph structure — useful for rapid prototyping but opaque for an engineering portfolio. Raw Python async would work but gives up built-in LangSmith tracing, checkpointing, and the human-in-the-loop primitives that LangGraph provides for free.

The `Send` API (introduced in LangGraph 0.1.x) is the specific feature that makes parallel agent execution clean: the router node emits a `Send` for each modality detected in the submission, and LangGraph runs those nodes concurrently with no extra threading code.

### State definition

```python
from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator

class MeridianState(TypedDict):
    # Input
    job_id: str
    input_files: List[UploadedFile]
    regulation_scope: List[str]         # e.g. ["gdpr", "soc2"]
    
    # Per-agent outputs (accumulated with list reducer)
    raw_extractions: Annotated[List[AgentExtraction], operator.add]
    
    # RAG outputs
    retrieved_chunks: List[RetrievedChunk]
    ner_entities: List[Entity]
    
    # Synthesis
    candidate_gaps: List[CandidateGap]
    groundedness_scores: Dict[str, float]
    verified_gaps: List[VerifiedGap]
    synthesis_retries: int
    
    # Output
    final_report: Optional[ComplianceReport]
    error: Optional[str]
    metadata: Dict[str, Any]
```

`Annotated[List[AgentExtraction], operator.add]` is the key pattern: because multiple agents write to `raw_extractions` concurrently, LangGraph needs a reducer function to merge their outputs. `operator.add` concatenates the lists. Without this, concurrent writes would clobber each other.

### Graph structure

```python
from langgraph.graph import StateGraph, END
from langgraph.constants import Send

builder = StateGraph(MeridianState)

# Nodes
builder.add_node("classify",   classify_input_node)
builder.add_node("doc_agent",  doc_agent_node)
builder.add_node("audio_agent",audio_agent_node)
builder.add_node("vision_agent",vision_agent_node)
builder.add_node("data_agent", data_agent_node)
builder.add_node("synthesize", synthesis_node)
builder.add_node("gate",       hallucination_gate_node)
builder.add_node("report",     report_generation_node)

# Entry
builder.set_entry_point("classify")

# Fan-out: classify emits Send() for each detected modality
def route_to_agents(state: MeridianState):
    sends = []
    for file in state["input_files"]:
        match file.modality:
            case "document": sends.append(Send("doc_agent",   {"file": file}))
            case "audio":    sends.append(Send("audio_agent", {"file": file}))
            case "image":    sends.append(Send("vision_agent",{"file": file}))
            case "tabular":  sends.append(Send("data_agent",  {"file": file}))
    return sends

builder.add_conditional_edges("classify", route_to_agents)

# Fan-in: all agents converge to synthesize
for agent in ["doc_agent", "audio_agent", "vision_agent", "data_agent"]:
    builder.add_edge(agent, "synthesize")

# Gate logic: retry or proceed
def gate_routing(state: MeridianState):
    failed = [k for k, v in state["groundedness_scores"].items() if v < 0.80]
    if failed and state["synthesis_retries"] < 2:
        return "synthesize"
    return "report"

builder.add_edge("synthesize", "gate")
builder.add_conditional_edges("gate", gate_routing, {"synthesize": "synthesize", "report": "report"})
builder.add_edge("report", END)

graph = builder.compile(checkpointer=SqliteSaver.from_conn_string("checkpoints.db"))
```

The `checkpointer` enables resumable execution — if the worker process dies mid-job, the graph resumes from the last completed node rather than starting over.

---

## 3. RAG pipeline

### Why three-stage retrieval

**Stage 1 — Dense vector search** is fast and captures semantic similarity, but misses exact terminology matches. Regulatory text is full of precise defined terms ("data subject," "supervisory authority," "lawful basis") where BM25 significantly outperforms dense retrieval.

**Stage 2 — BM25 keyword search** catches exact term matches but misses paraphrasing. A policy that says "individuals whose data we process" won't match a BM25 query for "data subjects."

**Stage 3 — RRF + cross-encoder reranking** combines both candidate lists and then uses a dedicated reranker model (`ms-marco-MiniLM-L-6-v2`) to score the merged top-20 by relevance to the actual query. This consistently outperforms either retrieval strategy alone.

In internal testing on the GDPR golden dataset, hybrid + rerank improves context precision by 11 percentage points over dense-only retrieval.

### Chunking strategy

**Considered:** Fixed-size (512 tokens), recursive character split, semantic chunking, page-level.

**Chosen:** Semantic chunking with LlamaIndex `SemanticChunker`.

Regulatory documents have natural semantic units: GDPR Article 6 is a single coherent clause about lawful basis for processing. Fixed-size chunking slices these units arbitrarily, producing chunks that start mid-clause and end mid-sentence. The reranker can partially compensate, but context precision suffers. Semantic chunking respects these boundaries.

Page-level chunking was rejected because a single GDPR page can contain parts of 3–4 different articles, creating noisy retrieval targets.

### Chunk metadata schema

```sql
CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id   UUID REFERENCES corpora(id),
    source_doc  TEXT NOT NULL,
    regulation  TEXT NOT NULL,    -- 'gdpr', 'soc2', etc.
    article     TEXT,             -- 'Article 6', 'CC6.1', etc.
    jurisdiction TEXT,            -- 'EU', 'US', 'global'
    effective_date DATE,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    content     TEXT NOT NULL,
    embedding   VECTOR(1536),     -- text-embedding-3-small dimensions
    ts_vector   TSVECTOR          -- PostgreSQL full-text search
);

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON chunks USING GIN (ts_vector);
```

### Embedding model choice

**Considered:** `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`, `nomic-embed-text-v1.5` (local), `bge-large-en-v1.5`.

**Chosen:** `text-embedding-3-small` (OpenAI, via LlamaIndex).

`text-embedding-3-small` offers the best cost/performance ratio for the corpus size. `3-large` is 5× more expensive with marginal gains on regulatory text. Local models (`nomic`, `bge`) were evaluated but require GPU for production throughput; CPU inference at this corpus size (50K+ chunks) is too slow for online retrieval. The LlamaIndex abstraction means the embedding model can be swapped with one configuration change if a better model emerges.

---

## 4. Specialist agents

### Document agent

**Models used:**
- `deepset/roberta-base-squad2` — extractive QA over retrieved chunks
- `dslim/bert-base-NER` — named entity recognition (persons, organizations, dates, regulation names)
- `facebook/bart-large-mnli` — zero-shot classification of document type

**Flow:**
1. Zero-shot classify the document type (policy, contract, audit report, terms of service)
2. Run NER over the full text; extract: dates, obligation verbs, data subject categories, DPO mentions, retention periods
3. For each NER-extracted entity, construct a targeted RAG query and retrieve relevant regulatory clauses
4. Run extractive QA to answer specific compliance questions (e.g., "What is the stated data retention period?")
5. Return structured `AgentExtraction` object to shared state

### Audio agent

**Models used:**
- `openai/whisper-large-v3` (via HF Inference API) — transcription
- `pyannote/speaker-diarization-3.1` — speaker segmentation
- `facebook/wav2vec2-base-960h` — audio classification (meeting type detection)

**Flow:**
1. Classify audio type: board meeting, regulatory hearing, earnings call, training session
2. Split audio into 30-second chunks for Whisper (API limit)
3. Transcribe each chunk; concatenate with timestamps
4. Apply speaker diarization; annotate transcript with speaker labels
5. Use BART summarization to produce a concise summary of compliance-relevant statements
6. Return transcript, summary, and speaker-attributed quotes to shared state

**Design note on chunking:** Whisper works best on complete sentences. Splitting at 30-second boundaries mid-sentence degrades accuracy. The implementation detects silence gaps and splits at natural pauses using `librosa` before sending to Whisper.

### Vision agent

**Models used:**
- `vidore/colpali-v1.2` — visual document retrieval (page-level image embeddings)
- `Salesforce/blip2-opt-2.7b` — image-to-text for screenshot captioning
- `dandelin/vilt-b32-finetuned-vqa` — visual question answering on forms
- Claude vision API — screenshot analysis with reasoning

**Flow:**
1. For scanned PDFs: render each page as a 768×1024px image; embed with ColPali; retrieve most relevant pages
2. For screenshots: caption with BLIP2; route to Claude vision with targeted compliance questions
3. For structured forms: run VQA to extract specific fields (retention period, DPO name, consent mechanism)
4. Merge visual extractions with text-based findings in shared state

**Why ColPali over text-extraction-then-embed:** Scanned regulatory forms often have layout-dependent meaning — a checkbox next to "I consent to processing" is only meaningful with the visual context. ColPali embeds the whole page visually, preserving layout. Text-first approaches OCR the text and lose the spatial relationships.

### Data agent

**Models used:**
- `google/tapas-base-finetuned-wtq` — table question answering
- `amazon/chronos-t5-small` — time series forecasting for risk scoring
- scikit-learn gradient boosting — tabular anomaly classification

**Flow:**
1. Parse CSV/Excel into a pandas DataFrame
2. Run TAPAS to answer structured queries (e.g., "What is the maximum data access duration in this audit log?")
3. If time series data detected (date column + metric column), run Chronos-T5 to forecast and flag anomalous trends
4. Run tabular classifier to score rows for risk indicators (failed access attempts, unusual data export volumes)
5. Return structured findings to shared state

### Synthesis agent

The synthesis agent is the only node that calls Claude directly (all specialist agents use HF models for extraction and Claude only for reasoning-heavy tasks). It receives the merged `AgentExtraction` list, the retrieved regulatory chunks, and the NER entities, and reasons over all of them to produce candidate gaps.

**System prompt design:**
The synthesis prompt explicitly instructs Claude to: (a) only cite gaps for which retrieved regulatory text exists in context, (b) quote the specific regulatory article, (c) specify the specific policy section that is deficient or absent, and (d) rate severity as critical/major/minor with a one-sentence justification. This structured output format is enforced via a Pydantic output parser.

---

## 5. Evaluation architecture

### Why RAGAS + custom F1 rather than a single framework

RAGAS measures retrieval quality — it tells you whether the RAG pipeline is surfacing relevant, faithful context. It does not measure whether the downstream gap detection task is accurate. A pipeline with perfect RAGAS scores could still miss compliance gaps if the synthesis agent reasons poorly. The custom F1 harness measures the end-to-end task accuracy that actually matters for the compliance use case.

The LLM-as-judge layer adds a third lens: it evaluates agent behavior (routing, tool selection, reasoning quality) on dimensions that neither RAGAS nor F1 capture.

### Golden dataset construction

The gap detection golden dataset is built from real SEC enforcement actions available on EDGAR. Each enforcement action cites the specific rule the company violated and includes the company's public filings. This gives us genuine `{policy_text, regulation_text, gap: True}` pairs where the ground truth was determined by actual regulators, not synthetic annotation.

Negative examples (no gap) are constructed from policy documents that pass expert review, paired with regulatory clauses they satisfy.

### Nightly CI pipeline

```yaml
# .github/workflows/nightly_eval.yml
on:
  schedule:
    - cron: '0 3 * * *'   # 3am UTC

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run RAGAS eval
        run: python src/eval/run_ragas.py --dataset data/golden/gdpr_qa.jsonl
      - name: Run gap detection F1
        run: python src/eval/run_gap_detection.py --threshold 0.85
      - name: Run agent trace judge
        run: python src/eval/run_agent_judge.py --lookback-hours 24
      - name: Check thresholds
        run: python src/eval/check_thresholds.py
      - name: Open GitHub issue on regression
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({...})
```

---

## 6. API and async processing

### Why Celery over asyncio background tasks

FastAPI's native `BackgroundTasks` runs in the same process as the web server. A long-running LangGraph invocation (2–3 minutes) would block worker threads and degrade API responsiveness under load. Celery with Redis separates the web server from the worker, allows multiple workers to process jobs in parallel, provides built-in retry logic, and gives visibility into the job queue (via Flower UI).

**Considered:** FastAPI BackgroundTasks, asyncio task groups, RQ (Redis Queue), Celery.

**Chosen:** Celery with Redis as broker.

RQ is simpler than Celery and would also work, but Celery has better support for task routing, priority queues, and distributed workers — all relevant for the roadmap's multi-tenant extension.

---

## 7. Data storage

See `DATABASE.md` for full schema. Summary of storage allocation:

| Store | What lives there | Why |
|---|---|---|
| PostgreSQL | Jobs, documents, chunks, embeddings (pgvector), audit log | Single source of truth; transactional |
| Redis | Celery job queue, job status cache, rate limit counters | Ephemeral; speed |
| GCP Cloud Storage | Raw uploaded files, generated PDFs | Blob storage; durable, cheap |
| MLflow (SQLite local / Postgres prod) | Eval metrics, run parameters, artifacts | Experiment tracking |

---

## 8. Security considerations

### API key management

API keys (Anthropic, HuggingFace, Pinecone) are loaded from environment variables. In production on GCP, they are stored in Secret Manager and injected at runtime via Cloud Run's secret mounting. They never appear in code, logs, or Docker images.

### File upload validation

All uploads are validated before processing:
1. File size check (≤ 500 MB) before reading content
2. MIME type validation using `python-magic` (magic bytes, not filename extension)
3. PDF validation with `pypdf` before passing to the ingestion pipeline
4. Audio/video validation with `ffprobe` before passing to Whisper

### Data retention

Uploaded files are deleted from Cloud Storage after 7 days. Extracted text and chunk embeddings are retained for the job's 7-day result window, then deleted. No user-submitted content is retained beyond this window in the default configuration.

---

## 9. Scalability path

Meridian's current architecture supports horizontal scaling with no code changes:

- **API layer:** Cloud Run scales instances automatically up to the configured maximum. Each instance is stateless; all state lives in PostgreSQL/Redis.
- **Workers:** Additional Celery workers can be started on any machine with network access to Redis and PostgreSQL. Worker count scales the job processing throughput linearly.
- **Vector store:** Switching from pgvector to Pinecone (one config change) removes the single-machine vector index bottleneck for corpora beyond ~10M chunks.
- **Corpus updates:** The ingestion pipeline is idempotent and can be run incrementally. New regulatory documents are chunked and upserted without touching existing embeddings.

For a multi-tenant production deployment, the main additions would be: tenant isolation in the database (row-level security), per-tenant rate limiting in the API, and a proper authentication service (OAuth2 / API key management) replacing the simple Bearer token.
