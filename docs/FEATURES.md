# Features — Meridian

Each feature is described with its user story, the HuggingFace tasks it relies on, the implementation approach, and acceptance criteria.

---

## F-01 — Multimodal document ingestion

**Status:** Implemented (v0.1)  
**Priority:** P0

### User story

As a compliance analyst, I want to upload a company privacy policy PDF alongside a board meeting recording and a compliance dashboard screenshot, so that all relevant data is analyzed together in a single compliance review.

### HuggingFace tasks used

- Feature extraction (embedding generation)
- Document question answering (extractive QA over chunks)
- Image-text-to-text (screenshot analysis)
- Automatic speech recognition (audio transcription)
- Audio classification (audio type detection)

### Implementation

Uploads are accepted via `POST /submit` as a multipart form. Each file is validated (type, size), written to Cloud Storage, and a Celery task is enqueued. The `classify_input` LangGraph node inspects each file's MIME type and extension to assign a modality. LangGraph's `Send` API then routes each file to its specialist agent in parallel.

Supported input combinations in v1.0:

| Combination | Agents invoked |
|---|---|
| PDF only | doc_agent |
| MP3 only | audio_agent |
| PNG only | vision_agent |
| CSV only | data_agent |
| PDF + MP3 | doc_agent + audio_agent |
| PDF + MP3 + PNG | doc_agent + audio_agent + vision_agent |
| All four | All four agents in parallel |

### Acceptance criteria

- [ ] A ZIP file containing 1 PDF, 1 MP3, and 1 CSV is accepted and all three agents invoked
- [ ] Invalid file types return HTTP 422 with a descriptive error
- [ ] Files over 500 MB are rejected before upload completes (client-side and server-side check)
- [ ] Mixed-modality submission runs in ≤ 1.5× the time of single-modality

---

## F-02 — Hybrid regulatory RAG retrieval

**Status:** Implemented (v0.1)  
**Priority:** P0

### User story

As a compliance analyst, I want the system to retrieve the most relevant regulatory clauses for my submitted policy, so that gap detection is grounded in authoritative regulatory text rather than LLM hallucination.

### HuggingFace tasks used

- Sentence similarity (cross-encoder reranking)
- Feature extraction (dense retrieval embeddings)
- Text ranking (BM25 scoring)

### Implementation

Three-stage pipeline executed on every compliance query:

1. **Dense retrieval** — pgvector cosine similarity search on `text-embedding-3-small` embeddings; top-20 candidates
2. **BM25 keyword retrieval** — PostgreSQL `ts_vector` full-text search on chunk content; top-20 candidates
3. **RRF fusion** — Reciprocal Rank Fusion merges the two ranked lists into a unified top-20
4. **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores all 20 candidates; top-5 passed to LLM

Metadata filters are applied before retrieval to scope results to the requested regulatory frameworks (GDPR only, SOC-2 only, all frameworks, etc.).

### Acceptance criteria

- [ ] RAGAS context precision ≥ 0.80 on GDPR golden dataset (200 Q&A pairs)
- [ ] RAGAS context recall ≥ 0.82 on GDPR golden dataset
- [ ] Retrieval latency (all 3 stages combined) ≤ 800ms at P95
- [ ] Metadata filtering correctly restricts results to requested frameworks

---

## F-03 — Named entity recognition for regulatory obligations

**Status:** Implemented (v0.1)  
**Priority:** P0

### User story

As a compliance engineer, I want the system to automatically identify regulatory entities in submitted documents — data subjects, retention periods, DPO mentions, consent mechanisms — so that specific obligations can be checked against requirements.

### HuggingFace tasks used

- Token classification (NER with `dslim/bert-base-NER`)
- Zero-shot classification (entity type classification for domain-specific categories)

### Implementation

After text extraction, the document is passed through two NER stages:

**Stage 1 — General NER** (`dslim/bert-base-NER`): extracts PERSON, ORGANIZATION, DATE, LOCATION entities from the text.

**Stage 2 — Regulatory NER** (custom zero-shot classification pipeline): classifies spans as one of these regulatory-specific categories:
- `RETENTION_PERIOD` — "for 5 years," "until account closure"
- `DATA_SUBJECT_CATEGORY` — "customers," "employees," "EU residents"
- `CONSENT_MECHANISM` — "opt-in checkbox," "explicit written consent"
- `DPO_MENTION` — "Data Protection Officer," "privacy@company.com"
- `LAWFUL_BASIS` — "legitimate interest," "contractual necessity"
- `THIRD_PARTY_TRANSFER` — "we share with," "our partners include"

These entities become structured query terms for the RAG retrieval step — the system checks each entity type against the corresponding regulatory requirement.

### Acceptance criteria

- [ ] `RETENTION_PERIOD` entity extracted from 90%+ of documents that contain one
- [ ] `DPO_MENTION` correctly identified in 95%+ of test documents
- [ ] No false `LAWFUL_BASIS` entities extracted from unrelated text
- [ ] NER pipeline adds ≤ 5 seconds to total job duration

---

## F-04 — ASR transcription with speaker diarization

**Status:** Implemented (v0.1)  
**Priority:** P0

### User story

As a compliance analyst, I want to upload a board meeting recording and have it transcribed with speaker attribution, so that specific compliance statements can be traced to the individual who made them.

### HuggingFace tasks used

- Automatic speech recognition (`openai/whisper-large-v3`)
- Audio classification (`pyannote/speaker-diarization-3.1`)

### Implementation

1. Audio is preprocessed: normalized to 16kHz mono WAV using `librosa`
2. Silence detection splits audio at natural pause boundaries (VAD via `webrtcvad`)
3. Each segment is sent to Whisper large-v3 via HF Inference API
4. Speaker diarization runs in parallel on the full audio
5. Transcript segments are annotated with speaker labels by alignment
6. BART summarization extracts compliance-relevant statements from the full transcript

Output format:

```json
{
  "transcript": [
    {"speaker": "SPEAKER_01", "start": 0.0, "end": 8.4, "text": "We need to review our data retention policy before Q4."},
    {"speaker": "SPEAKER_02", "start": 8.7, "end": 14.1, "text": "The GDPR audit flagged the 7-year retention on marketing data."}
  ],
  "compliance_statements": ["7-year retention on marketing data flagged in GDPR audit"],
  "summary": "Board discussed data retention policy review ahead of Q4 GDPR audit deadline."
}
```

### Acceptance criteria

- [ ] WER ≤ 15% on regulatory domain audio (tested on 10 sample board meeting recordings)
- [ ] Speaker labels present for audio with 2+ speakers using diarization
- [ ] 60-minute audio transcribed in ≤ 8 minutes
- [ ] Compliance-relevant statements extracted from transcript

---

## F-05 — Visual document analysis

**Status:** Implemented (v1.0)  
**Priority:** P1

### User story

As a compliance analyst, I want to upload a screenshot of our cookie consent banner and have the system evaluate whether it meets GDPR requirements, so that visual compliance signals are captured alongside text-based analysis.

### HuggingFace tasks used

- Image-text-to-text (`Salesforce/blip2-opt-2.7b`)
- Visual question answering (`dandelin/vilt-b32-finetuned-vqa`)
- Visual document retrieval (ColPali `vidore/colpali-v1.2`)

### Implementation

Screenshots are routed through BLIP2 for general image captioning, then through VQA for targeted compliance questions:

- "Is there a clear 'reject all' option visible?"
- "Does the banner include a link to the privacy policy?"
- "Are consent options pre-checked?"

For scanned PDFs and document images, ColPali embeds the page visually and retrieves matching regulatory pages from a pre-indexed regulatory image corpus. This handles scanned forms where layout is meaning.

Claude vision API is used for complex visual reasoning that BLIP2 and VQA cannot handle (e.g., "Does this data flow diagram show personal data leaving the EU?").

### Acceptance criteria

- [ ] Cookie consent banner screenshot correctly identified as compliant or non-compliant in ≥ 85% of test cases
- [ ] ColPali retrieves correct regulatory page for scanned form inputs with ≥ 80% recall
- [ ] Claude vision analysis adds ≤ 20 seconds to job duration

---

## F-06 — Tabular risk scoring

**Status:** Implemented (v1.0)  
**Priority:** P1

### User story

As an IT auditor, I want to upload an access log CSV and have the system flag unusual data access patterns that may indicate a compliance risk, so that anomalies are surfaced before a manual review.

### HuggingFace tasks used

- Table question answering (TAPAS)
- Time series forecasting (Chronos-T5)
- Tabular classification (scikit-learn + HF)

### Implementation

The data agent processes CSV/Excel inputs through three lenses:

1. **Structured QA (TAPAS):** Answers specific questions about the table — max access durations, top data exporters, permission levels present
2. **Anomaly classification:** A gradient boosting classifier (trained on synthetic audit log data) scores each row for risk indicators: access outside business hours, mass export events, new user with admin privileges
3. **Trend analysis (Chronos-T5):** If a time column is detected, Chronos-T5 forecasts the next 7 days of the primary metric and flags significant deviations from expected trend

### Acceptance criteria

- [ ] TAPAS correctly answers structured queries in ≥ 85% of test cases on tabular golden dataset
- [ ] Anomaly classifier flags test anomalies with precision ≥ 0.80 and recall ≥ 0.75
- [ ] Time series forecasting runs in ≤ 10 seconds on a 12-month dataset

---

## F-07 — Compliance gap detection with citations

**Status:** Implemented (v0.1)  
**Priority:** P0

### User story

As a compliance officer, I want to receive a list of specific compliance gaps with citations to both the regulatory requirement and the relevant section of my submitted policy, so that I can remediate each gap with confidence that the finding is accurate.

### HuggingFace tasks used

- Sentence similarity (groundedness verification)
- Summarization (gap description generation)
- Text generation (remediation suggestions)

### Implementation

Claude Sonnet receives the merged agent outputs and the top-5 retrieved regulatory chunks and reasons over the complete context to produce candidate gaps. Each gap includes:

```python
@dataclass
class CandidateGap:
    description: str              # Plain English description
    regulatory_article: str       # e.g., "GDPR Article 13(2)(a)"
    regulatory_quote: str         # Exact text from retrieved chunk
    policy_reference: str         # Section of submitted policy
    policy_quote: str             # Relevant policy text (or "Not found")
    severity: Literal["critical", "major", "minor"]
    severity_justification: str
    remediation: str              # Suggested action
    confidence: float             # 0.0–1.0
```

The hallucination gate then verifies each `regulatory_quote` against the retrieved chunk via `sentence-transformers/all-MiniLM-L6-v2` cosine similarity. If the score is below 0.80, the gap is either re-synthesized or marked as uncertain.

### Acceptance criteria

- [ ] F1 ≥ 0.85 on 150-example gap detection golden dataset
- [ ] Every gap in the output has a non-empty `regulatory_article` and `regulatory_quote`
- [ ] Groundedness score ≥ 0.80 for all verified gaps (enforced by gate)
- [ ] Severity classification agrees with expert annotation in ≥ 80% of test cases

---

## F-08 — Audit-ready PDF report generation

**Status:** Implemented (v0.2)  
**Priority:** P0

### User story

As a compliance officer, I want to receive a professionally formatted PDF report that I can submit to a regulator or board, with all gaps, citations, and remediations presented clearly.

### HuggingFace tasks used

None directly — report generation uses Claude for final formatting and `weasyprint` for PDF rendering.

### Implementation

Report sections:

1. **Cover page** — Company name (extracted from submission), date, regulatory scope, report ID
2. **Executive summary** — 2–3 paragraph synthesis of overall compliance posture
3. **Gap inventory** — Table of all gaps: severity, framework, article, brief description
4. **Gap detail** — Per-gap section with full description, regulatory citation, policy reference, remediation
5. **Compliance score** — Overall score by framework (gaps found / total checks)
6. **Appendix A** — Raw regulatory clauses retrieved during analysis
7. **Appendix B** — Job metadata and model versions used (for audit trail)

PDF is rendered from a Jinja2 HTML template using WeasyPrint. The report also conforms to PDF/A-1b (ISO 19005-1) for archival purposes.

### Acceptance criteria

- [ ] PDF generated in ≤ 30 seconds after synthesis completes
- [ ] PDF passes PDF/A-1b validation via `verapdf`
- [ ] All section links and bookmarks function correctly in PDF readers
- [ ] JSON report matches the schema defined in API.md

---

## F-09 — LangSmith trace monitoring

**Status:** Implemented (v0.1)  
**Priority:** P1

### User story

As an AI engineer maintaining this system, I want every agent node execution to be traced in LangSmith so that I can debug failures, track latency, and run evaluations against production traces.

### Implementation

LangSmith tracing is enabled via environment variable (`LANGCHAIN_TRACING_V2=true`). The LangGraph `compile()` call automatically instruments all nodes. Additional metadata tagged on each run: `job_id`, `modalities`, `regulation_scope`, `submission_timestamp`.

The nightly eval job pulls traces from LangSmith's API using a 24-hour lookback window and passes them to the LLM-as-judge evaluator.

### Acceptance criteria

- [ ] Every job appears in LangSmith with correct node-level spans
- [ ] Node latency is visible per-run in LangSmith UI
- [ ] LLM-as-judge eval can retrieve and score traces from the past 24 hours

---

## F-10 — Webhook delivery

**Status:** Implemented (v0.2)  
**Priority:** P1

### User story

As a developer integrating Meridian into a larger compliance workflow, I want to receive a webhook notification when a job completes, so that my system can react to results without polling.

### Implementation

The `POST /submit` request accepts an optional `webhook_url` parameter. On job completion (success or failure), Celery sends an HTTP POST to the webhook URL with:

```json
{
  "job_id": "uuid",
  "status": "complete",
  "completed_at": "2026-04-15T14:23:11Z",
  "report_url": "https://api.meridian.dev/report/uuid",
  "summary": {
    "total_gaps": 5,
    "critical": 2,
    "major": 2,
    "minor": 1
  }
}
```

Payloads are signed with HMAC-SHA256 using `WEBHOOK_SECRET` if configured. Delivery is retried up to 3 times with exponential backoff on non-2xx responses.

### Acceptance criteria

- [ ] Webhook delivered within 60 seconds of job completion
- [ ] HMAC signature present and verifiable when `WEBHOOK_SECRET` is set
- [ ] Retry logic fires on 4xx/5xx responses from webhook endpoint
- [ ] Webhook URL is validated (must be HTTPS in production) before job submission is accepted
