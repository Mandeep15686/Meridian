# API reference — Meridian

Base URL: `https://api.meridian.dev/v1` (production) or `http://localhost:8000/v1` (local)

All endpoints require a `Bearer` token in the `Authorization` header:

```
Authorization: Bearer mer_live_yourtoken
```

All request and response bodies are `application/json` unless noted. File uploads use `multipart/form-data`.

---

## Authentication

### Bearer tokens

API keys are issued in the format `mer_live_...` for production and `mer_test_...` for test/development environments. Test-mode keys return real processing results but against the test corpus only and are excluded from production analytics.

Generate a key by setting `MERIDIAN_API_KEY` in your environment (single-key auth for v1.0 portfolio deployment).

---

## Endpoints

### POST /v1/submit

Submit a compliance analysis job. Returns a job ID immediately; processing happens asynchronously.

**Content-type:** `multipart/form-data`

#### Request fields

| Field | Type | Required | Description |
|---|---|---|---|
| `files` | File[] | Yes | One or more files to analyze. PDF, DOCX, MP3, WAV, M4A, PNG, JPG, WEBP, CSV, XLSX supported. Max 500 MB per file. |
| `regulation_scope` | string[] | Yes | Regulatory frameworks to check against. Options: `gdpr`, `soc2`, `iso27001`, `sec_sp`, `cfpb`. Send as repeated form fields or comma-separated. |
| `webhook_url` | string | No | HTTPS URL to receive a POST notification on job completion. |
| `webhook_secret` | string | No | If provided, webhook payloads will be HMAC-SHA256 signed using this secret. |
| `report_formats` | string[] | No | Default: `["pdf", "json"]`. Options: `pdf`, `json`, `markdown`. |
| `language` | string | No | Default: `en`. Submitted document language for multilingual support (v1.1+). ISO 639-1 code. |
| `options` | JSON string | No | Advanced options — see options schema below. |

#### Options schema

```json
{
  "groundedness_threshold": 0.80,
  "max_synthesis_retries": 2,
  "include_regulatory_appendix": true,
  "severity_filter": ["critical", "major"],
  "max_gaps_returned": 50
}
```

#### Response `202 Accepted`

```json
{
  "job_id": "01J3KM9VXKQ4BFHRZS8G7WNPD",
  "status": "queued",
  "submitted_at": "2026-04-15T09:23:44Z",
  "estimated_completion_seconds": 180,
  "poll_url": "/v1/status/01J3KM9VXKQ4BFHRZS8G7WNPD",
  "report_url": "/v1/report/01J3KM9VXKQ4BFHRZS8G7WNPD"
}
```

#### Errors

| Status | Code | Description |
|---|---|---|
| 400 | `no_files` | Request contains no files |
| 400 | `invalid_scope` | One or more `regulation_scope` values not recognized |
| 413 | `file_too_large` | A file exceeds the 500 MB limit |
| 415 | `unsupported_file_type` | A file's MIME type is not supported |
| 422 | `invalid_webhook_url` | `webhook_url` is not a valid HTTPS URL |
| 429 | `rate_limited` | Submission rate limit exceeded (10/minute) |

#### Example

```bash
curl -X POST https://api.meridian.dev/v1/submit \
  -H "Authorization: Bearer mer_live_yourtoken" \
  -F "files=@privacy_policy.pdf" \
  -F "files=@board_meeting.mp3" \
  -F "regulation_scope=gdpr" \
  -F "regulation_scope=soc2" \
  -F "webhook_url=https://yourapp.com/webhooks/meridian"
```

---

### GET /v1/status/{job_id}

Poll the status of a submitted job.

#### Path parameters

| Parameter | Type | Description |
|---|---|---|
| `job_id` | string | The job ID returned by POST /submit |

#### Response `200 OK`

```json
{
  "job_id": "01J3KM9VXKQ4BFHRZS8G7WNPD",
  "status": "processing",
  "submitted_at": "2026-04-15T09:23:44Z",
  "started_at": "2026-04-15T09:23:46Z",
  "completed_at": null,
  "current_stage": "audio_agent",
  "stages_complete": ["classify_input", "doc_agent"],
  "stages_total": 6,
  "progress_pct": 33
}
```

**Status values:**

| Value | Description |
|---|---|
| `queued` | Job is waiting in the Celery queue |
| `processing` | Job is actively running through the agent pipeline |
| `complete` | Job finished successfully; report is available |
| `failed` | Job failed after all retries; see `error` field |
| `cancelled` | Job was cancelled via DELETE /v1/job/{job_id} |

**On completion** (`status: "complete"`):

```json
{
  "job_id": "01J3KM9VXKQ4BFHRZS8G7WNPD",
  "status": "complete",
  "submitted_at": "2026-04-15T09:23:44Z",
  "started_at": "2026-04-15T09:23:46Z",
  "completed_at": "2026-04-15T09:26:12Z",
  "duration_seconds": 146,
  "summary": {
    "total_gaps": 5,
    "by_severity": {"critical": 2, "major": 2, "minor": 1},
    "by_framework": {"gdpr": 4, "soc2": 1},
    "groundedness_pass_rate": 1.0
  },
  "report_url": "/v1/report/01J3KM9VXKQ4BFHRZS8G7WNPD",
  "langsmith_trace_url": "https://smith.langchain.com/runs/..."
}
```

**On failure** (`status: "failed"`):

```json
{
  "job_id": "01J3KM9VXKQ4BFHRZS8G7WNPD",
  "status": "failed",
  "error": {
    "code": "synthesis_failed",
    "message": "Synthesis agent exhausted retries without reaching groundedness threshold",
    "stage": "gate",
    "retry_count": 2
  }
}
```

#### Errors

| Status | Code | Description |
|---|---|---|
| 404 | `job_not_found` | No job with this ID exists or it has expired (7-day TTL) |

---

### GET /v1/report/{job_id}

Retrieve the completed compliance report. Defaults to JSON format; use the `Accept` header or `format` query parameter for other formats.

#### Path parameters

| Parameter | Type | Description |
|---|---|---|
| `job_id` | string | The job ID of a completed job |

#### Query parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `format` | string | `json` | `json`, `pdf`, or `markdown` |

#### JSON response `200 OK`

```json
{
  "job_id": "01J3KM9VXKQ4BFHRZS8G7WNPD",
  "generated_at": "2026-04-15T09:26:12Z",
  "regulation_scope": ["gdpr", "soc2"],
  "submitted_files": [
    {"filename": "privacy_policy.pdf", "modality": "document", "size_bytes": 124800},
    {"filename": "board_meeting.mp3", "modality": "audio", "duration_seconds": 612}
  ],
  "executive_summary": "The submitted privacy policy is materially non-compliant with GDPR in two critical areas...",
  "compliance_score": {
    "gdpr": {"score": 0.62, "gaps": 4, "checks_performed": 12},
    "soc2": {"score": 0.87, "gaps": 1, "checks_performed": 8}
  },
  "gaps": [
    {
      "gap_id": "gap_001",
      "severity": "critical",
      "framework": "gdpr",
      "regulatory_article": "GDPR Article 13(2)(a)",
      "regulatory_requirement": "The controller shall provide information on the period for which personal data will be stored.",
      "regulatory_chunk_id": "chunk_gdpr_art13_2a",
      "policy_reference": "Section 4 — Data Retention",
      "policy_text": "We retain your data as long as necessary.",
      "gap_description": "The policy states data is retained 'as long as necessary' without specifying a concrete retention period or the criteria used to determine that period, as required by Article 13(2)(a).",
      "severity_justification": "Critical: absence of a specific retention period is a frequently cited violation in GDPR enforcement actions (see ICO enforcement tracker).",
      "remediation": "Add a specific retention table to Section 4 listing each data category, its retention period (e.g., '3 years from account closure'), and the legal basis for that period.",
      "confidence": 0.94,
      "groundedness_score": 0.91
    }
  ],
  "model_metadata": {
    "synthesis_model": "claude-sonnet-4-6",
    "retrieval_model": "text-embedding-3-small",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ner_model": "dslim/bert-base-NER",
    "asr_model": "openai/whisper-large-v3",
    "pipeline_version": "1.0.3"
  },
  "retrieved_chunks": [
    {
      "chunk_id": "chunk_gdpr_art13_2a",
      "regulation": "gdpr",
      "article": "Article 13(2)(a)",
      "content": "The period for which the personal data will be stored, or if that is not possible, the criteria used to determine that period.",
      "similarity_score": 0.94
    }
  ]
}
```

#### PDF response

When `format=pdf` or `Accept: application/pdf`, returns the binary PDF file with:

```
Content-Type: application/pdf
Content-Disposition: attachment; filename="meridian_report_01J3KM9VXKQ4BFHRZS8G7WNPD.pdf"
```

#### Errors

| Status | Code | Description |
|---|---|---|
| 404 | `job_not_found` | Job does not exist or has expired |
| 409 | `job_not_complete` | Job is still processing; poll `/status` first |

---

### DELETE /v1/job/{job_id}

Cancel a queued or in-progress job, and delete all associated data.

#### Response `200 OK`

```json
{
  "job_id": "01J3KM9VXKQ4BFHRZS8G7WNPD",
  "status": "cancelled",
  "deleted_at": "2026-04-15T09:24:12Z"
}
```

#### Errors

| Status | Code | Description |
|---|---|---|
| 404 | `job_not_found` | Job does not exist |
| 409 | `job_already_complete` | Complete jobs cannot be cancelled; use DELETE to remove data |

---

### GET /v1/corpus/status

Returns the status of all ingested regulatory corpora: version, document count, chunk count, and last refreshed date.

#### Response `200 OK`

```json
{
  "corpora": [
    {
      "id": "gdpr",
      "name": "General Data Protection Regulation (GDPR)",
      "jurisdiction": "EU",
      "version": "2018-05-25",
      "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679/oj",
      "document_count": 1,
      "chunk_count": 2847,
      "last_ingested": "2026-04-01T02:00:00Z",
      "freshness_status": "current"
    },
    {
      "id": "soc2",
      "name": "SOC-2 Trust Services Criteria (2022)",
      "jurisdiction": "US",
      "version": "2022",
      "chunk_count": 1203,
      "last_ingested": "2026-03-15T02:00:00Z",
      "freshness_status": "current"
    }
  ],
  "total_chunks": 7842,
  "index_last_rebuilt": "2026-04-01T03:15:00Z"
}
```

---

### GET /v1/health

Health check endpoint for load balancer and monitoring probes.

#### Response `200 OK` (healthy)

```json
{
  "status": "healthy",
  "version": "1.0.3",
  "timestamp": "2026-04-15T09:30:00Z",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy",
    "hf_api": "healthy",
    "anthropic_api": "healthy"
  }
}
```

#### Response `503 Service Unavailable` (degraded)

```json
{
  "status": "degraded",
  "version": "1.0.3",
  "timestamp": "2026-04-15T09:30:00Z",
  "dependencies": {
    "database": "healthy",
    "redis": "unhealthy",
    "hf_api": "healthy",
    "anthropic_api": "healthy"
  },
  "message": "Redis is unavailable; new job submissions are paused."
}
```

---

## Webhook payloads

When a `webhook_url` is provided and `WEBHOOK_SECRET` is configured, each webhook POST includes:

- `X-Meridian-Signature`: `sha256=<hmac_hex_digest>` — computed over the raw request body
- `X-Meridian-Timestamp`: Unix timestamp — verify this is within 5 minutes to prevent replay attacks
- `X-Meridian-Job-Id`: The job ID for easy routing

### Completion webhook

```json
{
  "event": "job.complete",
  "job_id": "01J3KM9VXKQ4BFHRZS8G7WNPD",
  "completed_at": "2026-04-15T09:26:12Z",
  "duration_seconds": 146,
  "summary": {
    "total_gaps": 5,
    "critical": 2,
    "major": 2,
    "minor": 1
  },
  "report_url": "https://api.meridian.dev/v1/report/01J3KM9VXKQ4BFHRZS8G7WNPD"
}
```

### Failure webhook

```json
{
  "event": "job.failed",
  "job_id": "01J3KM9VXKQ4BFHRZS8G7WNPD",
  "failed_at": "2026-04-15T09:27:00Z",
  "error": {
    "code": "synthesis_failed",
    "message": "Synthesis agent exhausted retries",
    "stage": "gate"
  }
}
```

---

## Error schema

All error responses follow this schema:

```json
{
  "error": {
    "code": "snake_case_error_code",
    "message": "Human-readable description of the error.",
    "details": {}
  },
  "request_id": "req_01J3KM9VXKQ4BFHRZS8G7WNPD"
}
```

---

## Rate limits

| Endpoint | Limit |
|---|---|
| POST /v1/submit | 10 requests / minute per API key |
| GET /v1/status | 60 requests / minute per API key |
| GET /v1/report | 30 requests / minute per API key |
| GET /v1/corpus/status | 10 requests / minute |

Rate limit headers are returned on every response:

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1713175260
```

---

## Versioning

The API is versioned by URL path (`/v1/`). Breaking changes will be introduced in `/v2/` with a minimum 90-day deprecation window for `/v1/`. Non-breaking additions (new fields, new optional parameters) may be made to the current version without a version bump.
