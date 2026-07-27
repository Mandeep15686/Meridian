# Troubleshooting — Meridian

---

## Quick diagnostics

Before digging into specific issues, run the built-in diagnostics script:

```bash
python scripts/diagnose.py
```

This checks: database connectivity, pgvector extension, Redis connectivity, HF API reachability, Anthropic API key validity, corpus chunk counts, and Celery worker availability. It prints a summary with ✓/✗ for each check.

---

## Installation and startup

### `ERROR: No module named 'magic'`

**Symptom:** `ImportError: No module named 'magic'` at startup.

**Cause:** `python-magic` requires the `libmagic` system library, which is not installed.

**Fix:**

```bash
# macOS
brew install libmagic

# Ubuntu / Debian
sudo apt-get install libmagic1

# Windows (WSL2)
sudo apt-get install libmagic1
```

---

### `ERROR: Could not load the 'poppler' library`

**Symptom:** Vision agent or ColPali fails with a poppler error when processing PDF images.

**Cause:** `pdf2image` requires the `poppler-utils` system package.

**Fix:**

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt-get install poppler-utils

# Docker (add to Dockerfile)
RUN apt-get install -y poppler-utils
```

---

### `pgvector extension not found`

**Symptom:** `ProgrammingError: type "vector" does not exist` on first run.

**Cause:** The `vector` PostgreSQL extension is not installed in the database.

**Fix:**

```bash
# Connect to the database and install the extension
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"

# For Docker Compose, this runs automatically if you use the pgvector image:
# image: pgvector/pgvector:pg15
# If you used the standard postgres image, switch to the pgvector variant.
```

---

### `alembic.util.exc.CommandError: Can't locate revision`

**Symptom:** `alembic upgrade head` fails with a revision error after pulling new code.

**Cause:** Local database is behind the migration history in the repo.

**Fix:**

```bash
# See current database revision
alembic current

# See all available revisions
alembic history

# Upgrade to latest
alembic upgrade head

# If the database is in an inconsistent state (rare), stamp it to the base:
alembic stamp head   # use with caution — marks DB as up to date without running migrations
```

---

### `redis.exceptions.ConnectionError: Error connecting to Redis`

**Symptom:** Celery worker or API fails to start with a Redis connection error.

**Cause:** Redis is not running or `REDIS_URL` points to the wrong host/port.

**Fix:**

```bash
# Check Redis is running
docker compose ps redis

# Start it if not running
docker compose up -d redis

# Test connectivity manually
redis-cli -u $REDIS_URL ping   # should return PONG
```

---

## API errors

### `HTTP 401 Unauthorized`

**Symptom:** All API requests return `{"error": {"code": "unauthorized"}}`.

**Cause:** Missing or incorrect `Authorization` header.

**Fix:** Ensure the header is formatted exactly as:

```
Authorization: Bearer <your-MERIDIAN_API_KEY-value>
```

Note: the `Bearer ` prefix (with trailing space) is required. Verify `MERIDIAN_API_KEY` matches the key you are sending.

---

### `HTTP 415 Unsupported Media Type`

**Symptom:** File upload rejected with `unsupported_file_type`.

**Cause:** Either the file extension is not on the allowlist, or the actual file magic bytes don't match the declared MIME type.

**Diagnosis:**

```bash
# Check what MIME type python-magic detects
python -c "import magic; print(magic.from_file('your_file.pdf', mime=True))"
```

**Fix:** Ensure the file is a genuine PDF/audio/image/CSV. Renaming a file does not change its MIME type. Re-export the file from its source application if the magic bytes are wrong.

---

### `HTTP 413 Request Entity Too Large`

**Symptom:** Upload rejected before processing starts.

**Cause:** File exceeds `MAX_FILE_SIZE_MB` (default 500 MB).

**Fix:** Compress the audio file before uploading:

```bash
# Compress MP3 with ffmpeg (reduce bitrate)
ffmpeg -i large_meeting.mp3 -b:a 64k compressed_meeting.mp3

# Compress PDF (remove embedded images)
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen \
   -dNOPAUSE -dBATCH -sOutputFile=compressed.pdf original.pdf
```

---

### `HTTP 409 Job Not Complete`

**Symptom:** `GET /report/{job_id}` returns 409 immediately after submission.

**Cause:** The job is still processing. Reports are only available after the job reaches `complete` status.

**Fix:** Poll `GET /status/{job_id}` first and only fetch the report when `status == "complete"`.

---

## Pipeline failures

### Job status stuck at `queued`

**Symptom:** A submitted job stays in `queued` status indefinitely and never transitions to `processing`.

**Cause:** The Celery worker is not running, has crashed, or is not connected to the same Redis instance as the API.

**Diagnosis:**

```bash
# Check if worker is running
docker compose ps worker

# Check worker logs
docker compose logs worker --tail=50

# Check the Celery queue length in Redis
redis-cli -u $REDIS_URL llen celery

# Check if Flower UI shows workers as online
open http://localhost:5555
```

**Fix:** Restart the worker.

```bash
docker compose restart worker
```

---

### Job fails at `audio_agent` with `RuntimeError: ffprobe not found`

**Cause:** `ffmpeg`/`ffprobe` is not installed in the environment.

**Fix:**

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Verify
ffprobe -version
```

---

### Job fails at `synthesis` with `anthropic.APIStatusError: 529`

**Symptom:** Synthesis agent fails with a 529 Overloaded error from Anthropic.

**Cause:** Anthropic API is temporarily overloaded.

**Fix:** This is handled automatically by the Celery retry logic (up to 3 retries with exponential backoff). If the job still fails after retries, it will appear in the failed state with `error_code: "synthesis_failed"`. Re-submit the job.

For frequent 529 errors, consider:
1. Upgrading to an Anthropic plan with higher rate limits
2. Adding a longer backoff: set `CELERY_TASK_MAX_RETRIES=5` and increase `CELERY_TASK_TIMEOUT`

---

### Job fails at `gate` with `synthesis_retries exhausted`

**Symptom:** Job fails with `error_code: "synthesis_failed"` and `error_stage: "gate"`.

**Cause:** The synthesis agent produced claims that repeatedly failed the groundedness check (cosine similarity below `GROUNDEDNESS_THRESHOLD`).

**Diagnosis:** Check the LangSmith trace for the job — the gate node log shows which claims failed and their scores.

**Fix options:**

1. **Lower the threshold temporarily** (not recommended for production):
   ```env
   GROUNDEDNESS_THRESHOLD=0.70
   ```

2. **Check the retrieved chunks** — if the RAG pipeline is returning low-quality chunks, the synthesis agent cannot ground its claims. Run the retrieval diagnostic:
   ```bash
   python scripts/diagnose_retrieval.py --job-id <job_id>
   ```

3. **Check for out-of-scope regulation** — if the submitted document has no overlap with the requested regulatory framework, synthesis may hallucinate. Try a different `regulation_scope`.

---

### RAG returns irrelevant chunks

**Symptom:** Gap detection quality is poor; retrieved chunks are not related to the submitted policy content.

**Diagnosis:**

```bash
# Run a manual retrieval query against the corpus
python scripts/test_retrieval.py --query "data retention period" --scope gdpr --top-k 5
```

If the returned chunks are clearly wrong, the likely causes are:

**Cause 1: IVFFlat index is stale.**
The IVFFlat index must be rebuilt when the chunk count grows significantly (more than 50% since last build).

```sql
REINDEX INDEX CONCURRENTLY chunks_embedding_idx;
```

**Cause 2: Embedding model mismatch.**
If `EMBEDDING_MODEL` was changed after ingestion, existing embeddings were generated by a different model and are incompatible with the new query embeddings.

**Fix:** Re-ingest the entire corpus.

```bash
python scripts/ingest_corpus.py --source all --force-reingest
```

**Cause 3: Low corpus coverage.**
The requested regulatory framework may not be in the corpus or may be sparsely indexed.

```bash
# Check chunk counts per corpus
curl http://localhost:8000/v1/corpus/status
```

---

### Whisper transcription quality is poor

**Symptom:** Audio transcription contains many errors, especially on regulatory terminology ("CFPB" transcribed as "see FBI", "SOC-2" as "sock two").

**Cause:** Background noise, low audio quality, or strong accents reduce Whisper accuracy.

**Fix options:**

1. **Pre-process the audio** before uploading:
   ```bash
   # Denoise with ffmpeg (simple high-pass filter)
   ffmpeg -i noisy.mp3 -af "highpass=f=200, lowpass=f=3000" cleaned.mp3
   ```

2. **Use a custom vocabulary prompt** — Whisper accepts an initial prompt that biases recognition toward expected vocabulary. This is configurable via `WHISPER_INITIAL_PROMPT`:
   ```env
   WHISPER_INITIAL_PROMPT="GDPR, CFPB, SOC-2, data controller, data subject, lawful basis, DPO"
   ```

3. **Use the larger Whisper model** — if on a paid HF plan, `openai/whisper-large-v3` is the default and already the most accurate. Confirm the model is not being downgraded by a cached response.

---

## Evaluation issues

### RAGAS eval fails with `KeyError: 'answer'`

**Cause:** The golden dataset JSONL file has a schema mismatch. RAGAS expects `question`, `answer`, `contexts`, and `ground_truth` fields.

**Fix:** Validate the golden dataset format:

```bash
python scripts/validate_golden_dataset.py --path data/golden/gdpr_qa.jsonl
```

---

### Gap detection F1 below threshold in CI

**Symptom:** Nightly eval fails: `F1 0.81 < threshold 0.85`.

**Cause:** A recent change to the retrieval pipeline, synthesis prompt, or hallucination gate has degraded gap detection accuracy.

**Diagnosis:**

```bash
# Run F1 eval with verbose output (shows which examples failed)
python src/eval/run_gap_detection.py --threshold 0.85 --verbose

# Compare against the last passing eval run in MLflow
python src/eval/compare_runs.py --run-a latest --run-b last-passing
```

**Common causes and fixes:**

| Cause | Fix |
|---|---|
| Synthesis prompt changed | Revert or tune the prompt; run eval before merging |
| Reranker model upgraded | Compare new vs old reranker on golden set |
| Retrieval `top-k` reduced | Increase `RETRIEVAL_TOP_K_RERANK` back to 5 |
| New corpus chunks polluting retrieval | Check recent ingestion for quality issues |
| Groundedness threshold raised | Lower threshold or improve synthesis grounding |

---

## Performance issues

### P95 latency exceeds 3 minutes

**Symptom:** Jobs consistently take longer than expected. `GET /status/{job_id}` shows the job running for over 3 minutes.

**Diagnosis:**

```bash
# Check which stage is slowest (from LangSmith traces)
python scripts/analyze_latency.py --lookback-hours 24

# Profile a single job locally
python scripts/profile_job.py --file data/sample_docs/sample_policy_full.pdf
```

**Common bottlenecks:**

| Stage | Symptom | Fix |
|---|---|---|
| Corpus ingestion (first run) | First job is slow, subsequent fast | This is expected; corpus is cached after first ingestion |
| HF API rate limiting | Constant ~500ms added to each model call | Upgrade HF plan; add model call caching (`RETRIEVAL_CACHE_TTL`) |
| Cross-encoder reranking | Reranker alone takes >1s | Reduce `RETRIEVAL_TOP_K_DENSE` and `RETRIEVAL_TOP_K_BM25` from 20 to 10 |
| PDF rendering for ColPali | Vision agent slow on large PDFs | Limit to first 20 pages for ColPali; process rest as text |
| Synthesis (Claude API) | Synthesis node takes >60s | Reduce `MAX_DOCUMENT_TOKENS`; pre-summarize large docs |

---

### High memory usage in Celery worker

**Symptom:** Worker process consumes >3 GB RAM and is killed by OOM.

**Cause:** Large audio files or PDFs are loaded entirely into memory.

**Fix:** Enable streaming file processing:

```env
USE_STREAMING_UPLOADS=true
AUDIO_CHUNK_SIZE_MB=50
```

Also limit worker concurrency to 1 for heavy workloads:

```env
CELERY_CONCURRENCY=1
```

---

## Docker issues

### `docker compose up` fails: `port 5432 is already in use`

**Cause:** A PostgreSQL instance is already running on port 5432.

**Fix:**

```bash
# Find what is using port 5432
lsof -i :5432

# Stop the conflicting process, or remap the Docker port
# In docker-compose.yml, change:
#   ports: ["5432:5432"]
# to:
#   ports: ["5433:5432"]
# Then update DATABASE_URL to use port 5433
```

---

### `WARN: The "HF_API_TOKEN" variable is not set`

**Symptom:** Docker Compose starts but shows variable warnings.

**Cause:** The `.env` file is missing or not in the project root.

**Fix:**

```bash
# Verify .env exists in the project root
ls -la .env

# If missing, copy from the example
cp .env.example .env
# Then fill in required values
```

---

## Getting more help

If the issue is not covered here:

1. **Check the LangSmith trace** for the failed job — the node-level spans show exactly where and why the job failed
2. **Enable DEBUG logging** locally: set `LOG_LEVEL=DEBUG` and re-run the job; this logs all model inputs and outputs
3. **Open a GitHub issue** with: the error message, relevant log lines (redact API keys), the LangSmith trace URL if available, and the output of `python scripts/diagnose.py`
