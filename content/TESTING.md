# Testing — Meridian

---

## Philosophy

Meridian is an AI pipeline: its outputs are probabilistic and its dependencies (HF Inference API, Anthropic API, pgvector) are external. This creates three distinct testing problems:

1. **Unit testing agent logic** — requires mocking all external APIs so tests are fast, deterministic, and offline-capable
2. **Integration testing the pipeline** — requires real infrastructure (PostgreSQL, Redis) but can still mock LLM/model APIs
3. **Eval-as-test** — the RAGAS and F1 harnesses are tests with numerical thresholds, run nightly by CI

The test suite is designed to be runnable in three modes:

| Mode | Command | Requires | Use when |
|---|---|---|---|
| Unit only | `pytest tests/unit/` | Python only | Local development, pre-commit |
| Integration | `pytest tests/integration/` | Docker Compose | Pre-PR, CI |
| Full eval | `python src/eval/run_all.py` | Docker + API keys | Nightly CI, before release |

---

## Test structure

```
tests/
├── conftest.py                    # Shared fixtures (state factories, mock builders)
├── unit/
│   ├── rag/
│   │   ├── test_chunker.py        # Semantic chunker behavior
│   │   ├── test_dense_retrieval.py
│   │   ├── test_bm25_retrieval.py
│   │   ├── test_rrf.py
│   │   └── test_reranker.py
│   ├── agents/
│   │   ├── test_doc_agent.py
│   │   ├── test_audio_agent.py
│   │   ├── test_vision_agent.py
│   │   └── test_data_agent.py
│   ├── graph/
│   │   ├── test_router.py         # Input classifier / modality detection
│   │   ├── test_gate.py           # Hallucination gate logic
│   │   └── test_state.py          # State reducer behavior
│   ├── models/
│   │   ├── test_ner_wrapper.py
│   │   └── test_similarity_wrapper.py
│   └── api/
│       ├── test_schemas.py        # Pydantic schema validation
│       └── test_auth.py           # Bearer token validation
├── integration/
│   ├── test_pipeline_document.py  # Full doc-only pipeline end-to-end
│   ├── test_pipeline_multimodal.py
│   ├── test_rag_retrieval.py      # Real pgvector queries
│   ├── test_api_submit.py         # API endpoint with real Celery
│   └── test_webhook_delivery.py
├── eval/                          # Eval harness (separate from pytest)
│   ├── run_ragas.py
│   ├── run_gap_detection.py
│   └── run_agent_judge.py
└── fixtures/
    ├── documents/
    │   ├── sample_policy_short.txt     # 500-token policy excerpt
    │   ├── sample_policy_full.pdf      # 12-page policy PDF
    │   └── sample_gdpr_chunk.txt       # Pre-chunked GDPR article for RAG mocking
    ├── audio/
    │   ├── sample_meeting_30s.mp3
    │   └── sample_meeting_10min.mp3
    ├── images/
    │   ├── sample_cookie_banner.png
    │   └── sample_compliance_dashboard.png
    └── tables/
        ├── sample_audit_log.csv
        └── sample_access_metrics.xlsx
```

---

## Running tests

### Prerequisites

```bash
# Unit tests — no Docker needed
pip install -e ".[dev]"

# Integration tests — Docker Compose must be running
docker compose up -d postgres redis
alembic upgrade head
python scripts/ingest_corpus.py --source gdpr --dev-mode  # ~5 min, 500 chunks
```

### Commands

```bash
# All unit tests (fast, ~30s)
pytest tests/unit/ -v

# All integration tests (~5 min)
pytest tests/integration/ -v

# A single test file
pytest tests/unit/rag/test_rrf.py -v

# A single test by name
pytest tests/unit/agents/test_doc_agent.py::test_ner_extracts_retention_period -v

# With coverage report
pytest tests/unit/ tests/integration/ --cov=src --cov-report=html --cov-report=term-missing

# Fail fast on first failure
pytest tests/ -x

# Show slow tests (over 1 second)
pytest tests/ --durations=10

# Run only tests marked as "fast"
pytest tests/ -m fast
```

### pytest markers

```python
# Available markers (defined in pyproject.toml)
@pytest.mark.unit          # no external dependencies
@pytest.mark.integration   # requires Docker Compose stack
@pytest.mark.slow          # takes more than 5 seconds
@pytest.mark.eval          # part of the eval harness
@pytest.mark.gpu           # requires GPU (skipped in CI by default)
```

---

## Writing unit tests

### The core pattern: mock all external calls

Every test that touches an agent node must mock HF API calls and Anthropic API calls. The project provides a `MockHFClient` and `MockAnthropicClient` in `tests/conftest.py`.

```python
# tests/unit/agents/test_doc_agent.py

import pytest
from unittest.mock import patch, MagicMock
from src.agents.doc_agent import doc_agent_node
from src.graph.state import MeridianState
from tests.conftest import make_state, sample_policy_text


def test_doc_agent_extracts_retention_period(sample_policy_text):
    """Doc agent should identify a data retention period from policy text."""

    # Arrange
    state = make_state(
        text_content=sample_policy_text,   # "We retain data for 5 years after account closure."
        regulation_scope=["gdpr"],
    )

    # Mock the NER model to return a known entity
    with patch("src.models.ner.NERModel.predict") as mock_ner, \
         patch("src.models.qa.QAModel.predict") as mock_qa, \
         patch("src.rag.retrieve.hybrid_retrieve") as mock_retrieve:

        mock_ner.return_value = [
            {"type": "RETENTION_PERIOD", "text": "5 years after account closure",
             "start": 20, "end": 49, "confidence": 0.97}
        ]
        mock_qa.return_value = {"answer": "5 years", "score": 0.94, "start": 20, "end": 26}
        mock_retrieve.return_value = [
            MagicMock(content="Controllers must specify a storage period per Article 13(2)(a).",
                      chunk_id="chunk_gdpr_art13_2a", regulation="gdpr")
        ]

        # Act
        result = doc_agent_node(state, config={})

    # Assert
    assert "raw_extractions" in result
    extractions = result["raw_extractions"]
    assert len(extractions) == 1
    ner_entities = extractions[0].ner_entities
    retention_entities = [e for e in ner_entities if e["type"] == "RETENTION_PERIOD"]
    assert len(retention_entities) == 1
    assert "5 years" in retention_entities[0]["text"]
```

### Testing the hallucination gate

```python
# tests/unit/graph/test_gate.py

import pytest
from src.graph.nodes.gate import hallucination_gate_node
from tests.conftest import make_state_with_gaps


def test_gate_passes_high_groundedness_claims():
    state = make_state_with_gaps(
        groundedness_scores={"gap_001": 0.92, "gap_002": 0.88},
        synthesis_retries=0,
    )
    result = hallucination_gate_node(state, config={})
    assert result["verified_gaps"] == state["candidate_gaps"]
    assert result["synthesis_retries"] == 0  # no retry triggered


def test_gate_rejects_low_groundedness_and_increments_retries():
    state = make_state_with_gaps(
        groundedness_scores={"gap_001": 0.92, "gap_002": 0.61},  # gap_002 fails
        synthesis_retries=0,
    )
    result = hallucination_gate_node(state, config={})
    # Gate should signal a retry, not pass to report
    assert result["synthesis_retries"] == 1
    assert "gap_002" in result.get("failed_gap_ids", [])


def test_gate_marks_uncertain_after_max_retries():
    state = make_state_with_gaps(
        groundedness_scores={"gap_001": 0.61},
        synthesis_retries=2,   # already at max
    )
    result = hallucination_gate_node(state, config={})
    uncertain = [g for g in result["verified_gaps"] if g.is_uncertain]
    assert len(uncertain) == 1
    assert uncertain[0].gap_id == "gap_001"
```

### Testing RRF fusion

```python
# tests/unit/rag/test_rrf.py

from src.rag.retrieve import reciprocal_rank_fusion


def test_rrf_promotes_document_in_both_lists():
    dense_results = [
        {"chunk_id": "A", "score": 0.95},
        {"chunk_id": "B", "score": 0.90},
        {"chunk_id": "C", "score": 0.85},
    ]
    bm25_results = [
        {"chunk_id": "C", "score": 12.4},   # C ranks higher in BM25
        {"chunk_id": "A", "score": 10.1},
        {"chunk_id": "D", "score": 8.8},    # D only in BM25
    ]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

    # A appears in both lists at high ranks → should be #1
    assert fused[0]["chunk_id"] == "A"
    # C appears in both lists → should outrank B (dense-only) and D (BM25-only)
    c_rank = next(i for i, r in enumerate(fused) if r["chunk_id"] == "C")
    b_rank = next(i for i, r in enumerate(fused) if r["chunk_id"] == "B")
    assert c_rank < b_rank
```

---

## Writing integration tests

Integration tests run the real pipeline against a real PostgreSQL + Redis stack but mock LLM API calls to avoid cost and nondeterminism.

```python
# tests/integration/test_pipeline_document.py

import pytest
import httpx
from unittest.mock import patch

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def api_client():
    return httpx.Client(base_url="http://localhost:8000", timeout=60.0)


def test_document_submission_creates_job(api_client, sample_policy_pdf_path):
    """A valid PDF submission should return a job_id and queue a Celery task."""

    with open(sample_policy_pdf_path, "rb") as f:
        response = api_client.post(
            "/v1/submit",
            files={"files": ("policy.pdf", f, "application/pdf")},
            data={"regulation_scope": "gdpr"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )

    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert body["poll_url"].startswith("/v1/status/")


def test_document_job_completes_with_gaps(api_client, sample_policy_pdf_path):
    """An end-to-end job with mocked LLM should complete and return compliance gaps."""

    mock_gaps = [
        {
            "gap_id": "gap_001",
            "severity": "critical",
            "framework": "gdpr",
            "regulatory_article": "GDPR Article 13(2)(a)",
            "gap_description": "No data retention period specified.",
            "confidence": 0.91,
            "groundedness_score": 0.89,
        }
    ]

    with patch("src.agents.synthesis_agent.claude_client.messages.create") as mock_llm:
        mock_llm.return_value = MagicMock(
            content=[MagicMock(text=json.dumps({"gaps": mock_gaps}))]
        )

        # Submit job
        with open(sample_policy_pdf_path, "rb") as f:
            response = api_client.post("/v1/submit", ...)
        job_id = response.json()["job_id"]

        # Poll until complete (max 30s)
        for _ in range(30):
            status = api_client.get(f"/v1/status/{job_id}", ...).json()
            if status["status"] in ("complete", "failed"):
                break
            time.sleep(1)

    assert status["status"] == "complete"
    assert status["summary"]["total_gaps"] == 1
    assert status["summary"]["critical"] == 1

    report = api_client.get(f"/v1/report/{job_id}", ...).json()
    assert len(report["gaps"]) == 1
    assert report["gaps"][0]["regulatory_article"] == "GDPR Article 13(2)(a)"
```

---

## CI pipeline

GitHub Actions runs tests on every push and PR:

```yaml
# .github/workflows/test.yml

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install ruff mypy
      - run: ruff check src/ tests/
      - run: mypy src/

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: meridian_test
        ports: ["5432:5432"]
      redis:
        image: redis:7
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -e ".[dev]"
      - run: alembic upgrade head
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/meridian_test
      - run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/meridian_test
          REDIS_URL: redis://localhost:6379/0
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          HF_API_TOKEN: ${{ secrets.HF_API_TOKEN }}
          MERIDIAN_API_KEY: test_key_for_ci
```

---

## Coverage targets

| Module | Current | Target |
|---|---|---|
| `src/rag/` | 78% | 80% |
| `src/agents/` | 65% | 75% |
| `src/graph/` | 82% | 85% |
| `src/api/` | 71% | 75% |
| `src/models/` | 58% | 70% |
| **Overall** | **70%** | **75%** |

Coverage is measured with `pytest-cov` and reported to Codecov. The CI job fails if overall coverage drops below 70%.
