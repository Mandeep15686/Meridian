# Performance — Meridian

---

## Baseline benchmarks (v1.0.3)

All benchmarks run on GCP Cloud Run with 2 vCPU / 4 GB RAM per instance, using the full production corpus (8,466 chunks), with 1 Celery worker. Test files are pre-uploaded to avoid measuring network upload time.

### End-to-end job latency

| Scenario | P50 | P95 | P99 |
|---|---|---|---|
| Document only (5-page PDF, ~3,000 tokens) | 48s | 78s | 112s |
| Document only (30-page PDF, ~18,000 tokens) | 91s | 134s | 165s |
| Audio only (5-min MP3) | 62s | 98s | 122s |
| Audio only (30-min MP3) | 195s | 248s | 290s |
| Image only (PNG screenshot) | 22s | 41s | 58s |
| Tabular only (CSV, 5,000 rows) | 31s | 52s | 71s |
| Multi-modal (30-page PDF + 5-min MP3 + CSV) | 108s | 158s | 194s |

### Stage-level latency breakdown (multi-modal job, P95)

| Stage | Duration | % of total |
|---|---|---|
| File validation and routing | 0.8s | 0.5% |
| Doc agent (NER + QA + retrieval) | 28s | 17.7% |
| Audio agent (Whisper + segmentation) | 42s | 26.6% |
| Vision agent (BLIP2 + VQA) | 19s | 12.0% |
| Data agent (TAPAS + Chronos) | 14s | 8.9% |
| RAG retrieval (dense + BM25 + RRF + rerank) | 1.8s | 1.1% |
| Synthesis (Claude Sonnet) | 34s | 21.5% |
| Hallucination gate | 2.1s | 1.3% |
| Report generation (PDF) | 16s | 10.1% |
| Overhead (DB writes, state updates) | 0.3s | 0.2% |

**Observations:**
- Audio agent (Whisper via HF API) is the dominant bottleneck for audio-heavy submissions. Local Whisper inference would be ~3× faster but requires a GPU.
- Synthesis (Claude API) is the dominant bottleneck for document-heavy submissions. Claude's TTFT (time to first token) varies from 2–8s; total generation for a typical compliance report is 20–40s.
- RAG retrieval is fast (<2s P95) because the IVFFlat index and BM25 both run in PostgreSQL — no network round-trips between the pipeline and a separate vector store.

---

## API endpoint latency

| Endpoint | P50 | P95 | P99 |
|---|---|---|---|
| POST /v1/submit | 42ms | 118ms | 230ms |
| GET /v1/status/{job_id} | 8ms | 24ms | 48ms |
| GET /v1/report/{job_id} (JSON) | 18ms | 52ms | 91ms |
| GET /v1/report/{job_id} (PDF, cached) | 28ms | 65ms | 104ms |
| GET /v1/health | 4ms | 12ms | 22ms |

---

## Retrieval latency

Individual retrieval stage timings (on the production corpus, 8,466 chunks):

| Stage | P50 | P95 | Notes |
|---|---|---|---|
| Text embedding (query) | 18ms | 38ms | OpenAI API call |
| Dense retrieval (pgvector ANN) | 12ms | 28ms | IVFFlat, probes=10 |
| BM25 retrieval (PostgreSQL FTS) | 8ms | 19ms | GIN index |
| RRF fusion | <1ms | 2ms | In-memory sort |
| Cross-encoder reranking (20 → 5) | 95ms | 180ms | HF API call, latency-dominant |
| **Total retrieval pipeline** | **134ms** | **267ms** | |

The cross-encoder reranking step is the retrieval bottleneck. Mitigations:
- Cache reranking results by `(query_hash, chunk_ids)` in Redis (TTL: 1 hour). Repeated queries against the same corpus return cached rankings in <1ms.
- Reduce candidate set from 20 to 10 before reranking: cuts reranking time by ~40% with a ~3% precision loss.
- Switch to a local cross-encoder (loaded in-process) to eliminate HF API round-trip latency.

---

## Throughput

With 1 Celery worker (4 concurrent threads), the system processes approximately:

| Scenario | Jobs/hour |
|---|---|
| Document only (5-page) | ~45 |
| Audio only (5-min) | ~35 |
| Multi-modal | ~22 |

Adding a second worker instance doubles throughput linearly. Cloud Run autoscaling triggers a new instance when the existing worker's job queue exceeds 5 pending jobs.

---

## Memory usage

| Component | Idle | Peak (multi-modal job) |
|---|---|---|
| FastAPI process | 280 MB | 340 MB |
| Celery worker (1 thread active) | 420 MB | 1.6 GB |
| PostgreSQL (shared buffers) | 1.0 GB | 1.2 GB |
| Redis | 80 MB | 120 MB |

The Celery worker's peak memory is driven by:
- Loading the full text of a 30-page PDF into memory for chunking: ~50 MB
- `sentence-transformers/all-MiniLM-L6-v2` loaded in-process for the hallucination gate: ~90 MB
- Pandas DataFrame for tabular processing: variable, up to ~200 MB for a 100,000-row CSV

Limit worker memory by setting `CELERY_CONCURRENCY=1` for memory-constrained environments.

---

## Profiling a job locally

To identify bottlenecks in a specific submission:

```bash
# Run the pipeline with cProfile
python -m cProfile -o profile.out \
  scripts/profile_job.py \
  --file data/sample_docs/sample_policy_full.pdf \
  --scope gdpr

# View the top 20 slowest functions
python -c "
import pstats
p = pstats.Stats('profile.out')
p.sort_stats('cumulative')
p.print_stats(20)
"
```

For LLM-specific latency, the LangSmith trace provides the most actionable view: each node shows its start time, end time, and token counts. Navigate to the LangSmith UI for any job via the `langsmith_trace_url` in the job status response.

---

## Tuning guide

### Reduce synthesis latency

Synthesis (Claude Sonnet) is the dominant latency bottleneck for document-heavy jobs. Strategies to reduce it:

**1. Pre-summarize long documents**

Documents over `MAX_DOCUMENT_TOKENS` (default 100,000 tokens) are summarized before synthesis. Lowering this threshold reduces the amount of context sent to Claude:

```env
MAX_DOCUMENT_TOKENS=50000   # summarize aggressively
```

Trade-off: lower threshold → faster synthesis, but may miss details in long policies.

**2. Reduce `RETRIEVAL_TOP_K_RERANK`**

The synthesis context window includes up to 5 retrieved chunks. Reducing this to 3 cuts the context size by ~40%:

```env
RETRIEVAL_TOP_K_RERANK=3
```

Trade-off: fewer context chunks → higher risk of missing relevant regulatory clauses (lower context recall).

**3. Reduce `SYNTHESIS_MAX_TOKENS`**

The default maximum response length is 4,096 tokens. For policies with fewer expected gaps, reduce this:

```env
SYNTHESIS_MAX_TOKENS=2048
```

Trade-off: shorter responses → truncated reports if many gaps are found.

---

### Reduce retrieval latency

**1. Enable retrieval caching**

Identical queries (same query text, same regulation scope) are cached in Redis:

```env
RETRIEVAL_CACHE_TTL=3600   # cache for 1 hour
```

Benefit: repeated submissions of similar documents (e.g., multiple versions of the same policy) hit the cache and skip reranking.

**2. Reduce candidate set before reranking**

```env
RETRIEVAL_TOP_K_DENSE=10
RETRIEVAL_TOP_K_BM25=10
```

This cuts reranking input from 20 to 10 candidates. P95 reranking latency drops from ~180ms to ~110ms. Context precision loss is approximately 2–3 percentage points.

**3. Use a lighter reranker**

Switch from `ms-marco-MiniLM-L-12-v2` (heavier, more accurate) to `ms-marco-MiniLM-L-6-v2` (lighter, ~40ms faster):

```env
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

### Reduce audio agent latency

Audio transcription via the HF Inference API is the largest single bottleneck for audio-heavy submissions. Strategies:

**1. Reduce audio quality before upload**

Lower bitrate audio transcribes faster (fewer bytes to process):

```bash
ffmpeg -i meeting.mp3 -b:a 64k meeting_compressed.mp3
# Reduces file size ~4× with minimal WER impact for speech content
```

**2. Run Whisper locally (requires GPU)**

For high-volume deployments, a local Whisper instance eliminates HF API latency entirely:

```env
ASR_BACKEND=local          # uses local transformers, requires GPU
ASR_MODEL_LOCAL=openai/whisper-large-v3
```

On an NVIDIA T4 GPU, a 30-minute audio file transcribes in ~45 seconds vs ~3.5 minutes via HF API.

**3. Parallelize audio chunk processing**

The default audio segmentation processes Whisper chunks sequentially. Enable concurrent chunk processing:

```env
AUDIO_PARALLEL_CHUNKS=4
```

This reduces transcription time for long audio by ~3× at the cost of higher peak memory (~200 MB additional per parallel stream).

---

### Scale horizontally

The most effective scaling strategy for production workloads is adding Celery workers:

```bash
# Local: start a second worker process
celery -A src.api.worker worker --loglevel=info --concurrency=4 &

# GCP Cloud Run: scale max instances
gcloud run services update meridian-worker --max-instances=10

# Kubernetes: scale the worker deployment
kubectl scale deployment meridian-worker --replicas=4
```

Each worker handles jobs independently; the only shared state is the Redis job queue and PostgreSQL database. Worker scaling is fully linear up to the database connection pool limit.

---

## Performance regression detection

The nightly eval job includes a latency benchmark (`src/eval/run_latency_benchmark.py`) that fails if P95 latency exceeds the enforced thresholds. Latency metrics are logged to MLflow alongside accuracy metrics, enabling correlation analysis — for example, detecting that a retrieval accuracy improvement came at a 15% latency cost.

A latency regression of > 20% on any P95 metric relative to the previous 7-day average triggers a GitHub issue with the `performance` label.
