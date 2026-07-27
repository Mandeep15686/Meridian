# Contributing — Meridian

Thank you for your interest in contributing. This document covers everything you need to get a development environment running, understand the code conventions, and submit changes.

---

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Getting started](#getting-started)
- [Development workflow](#development-workflow)
- [Code style and conventions](#code-style-and-conventions)
- [Testing](#testing)
- [Submitting a pull request](#submitting-a-pull-request)
- [Issue reporting](#issue-reporting)
- [Areas where help is welcome](#areas-where-help-is-welcome)

---

## Code of conduct

This project follows the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you agree to uphold these standards. Report unacceptable behavior to the maintainer via the contact in `SECURITY.md`.

---

## Getting started

### Prerequisites

- Python 3.11 or 3.12
- Docker Desktop 24.0+ (or Docker Engine + Compose v2)
- Git 2.40+
- A HuggingFace account with a free API token
- An Anthropic API key (required for synthesis; contact maintainer for a shared dev key)

### Fork and clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/your-username/meridian.git
cd meridian
git remote add upstream https://github.com/original-owner/meridian.git
```

### Environment setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Install all dependencies including dev tools
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

### Configure environment

```bash
cp .env.example .env.dev
# Fill in the required values — see CONFIGURATION.md for the full reference
# Minimum required for development:
#   ANTHROPIC_API_KEY
#   HF_API_TOKEN
#   DATABASE_URL (postgres or sqlite for unit tests)
#   REDIS_URL
```

### Start infrastructure

```bash
docker compose up -d postgres redis
```

### Run migrations and seed corpus

```bash
# Apply database schema
alembic upgrade head

# Ingest a minimal corpus for development (GDPR only, ~5 min)
python scripts/ingest_corpus.py --source gdpr --dev-mode
```

`--dev-mode` ingests only the first 500 chunks of each corpus, sufficient for local development and integration tests.

### Verify everything works

```bash
# Run the full test suite
pytest

# Run a smoke test against the live API
uvicorn src.api.main:app --reload &
curl -X GET http://localhost:8000/v1/health
```

---

## Development workflow

### Branch naming

```
feature/short-description        # new features
fix/short-description            # bug fixes
eval/short-description           # evaluation or metrics changes
docs/short-description           # documentation only
refactor/short-description       # no behavior change
chore/short-description          # dependency updates, tooling
```

### Staying up to date

```bash
git fetch upstream
git rebase upstream/main
```

Prefer rebase over merge to keep the commit history linear.

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description

Longer explanation if needed. Wrap at 72 characters.

Refs: #issue-number
```

Types: `feat`, `fix`, `eval`, `docs`, `refactor`, `test`, `chore`, `perf`

Scopes: `rag`, `agents`, `graph`, `api`, `eval`, `db`, `ui`, `deploy`, `docs`

Examples:

```
feat(agents): add speaker diarization to audio agent

Uses pyannote/speaker-diarization-3.1 via HF API. Annotates transcript
segments with SPEAKER_01, SPEAKER_02, etc. Falls back gracefully to
unlabeled transcript if diarization fails.

Refs: #42
```

```
fix(rag): correct off-by-one in groundedness threshold comparison

The gate was passing claims with score == 0.80 when the threshold
is exclusive (> 0.80). Changed >= to > in gate.py line 87.

Refs: #58
```

---

## Code style and conventions

### Formatting and linting

All code is formatted with `black` and linted with `ruff`. Type annotations are checked with `mypy`. Pre-commit runs these automatically on every commit.

```bash
# Run manually
black src/ tests/
ruff check src/ tests/ --fix
mypy src/
```

Configuration lives in `pyproject.toml`. Do not disable rules without discussion.

### Python conventions

- All public functions and classes must have docstrings (Google style)
- Type annotations are required on all function signatures
- Avoid `Any` unless genuinely necessary; use `TypeVar` or `Protocol` instead
- Prefer `dataclass` or `TypedDict` over `dict` for structured data
- Use `pathlib.Path` instead of string paths
- Log with `logging`, not `print`
- Secrets and config come from environment variables only — never hardcode

### LangGraph conventions

- Every graph node is defined as a pure function that takes `MeridianState` and returns a `dict` of state updates
- Nodes must not have side effects beyond: writing to the database, calling external APIs, and logging
- Every node must handle its own exceptions and return an `error` key in state rather than raising to the graph runner
- Add LangSmith metadata to every node via `config["metadata"]` (see `src/graph/graph.py` for the pattern)

### HuggingFace model conventions

- All HF model calls go through the wrappers in `src/models/`; never call `transformers` or the HF Inference API directly from agent code
- Every model wrapper must implement a `health_check()` method that verifies the model is reachable
- Model identifiers are defined as constants in `src/models/registry.py`, not scattered through agent code
- Cache model outputs by input hash in Redis with a configurable TTL (default 1 hour)

---

## Testing

### Test structure

```
tests/
├── unit/
│   ├── test_rag_chunker.py
│   ├── test_rag_retrieval.py
│   ├── test_agents/
│   │   ├── test_doc_agent.py
│   │   ├── test_audio_agent.py
│   │   ├── test_vision_agent.py
│   │   └── test_data_agent.py
│   ├── test_gate.py
│   └── test_report_generator.py
├── integration/
│   ├── test_pipeline_document_only.py
│   ├── test_pipeline_multimodal.py
│   └── test_api_endpoints.py
└── fixtures/
    ├── sample_policy.pdf
    ├── sample_audio.mp3
    └── sample_table.csv
```

### Running tests

```bash
# All tests
pytest

# Unit tests only (no Docker required)
pytest tests/unit/ -v

# Integration tests (requires Docker Compose stack)
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_rag_retrieval.py -v

# With coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Writing tests for agent nodes

Agent nodes call external APIs (HF Inference API, Anthropic). Unit tests must mock these calls:

```python
from unittest.mock import patch, MagicMock

def test_doc_agent_extracts_ner_entities(sample_policy_text):
    with patch("src.models.ner.NERModel.predict") as mock_ner:
        mock_ner.return_value = [
            {"type": "RETENTION_PERIOD", "text": "for 3 years", "confidence": 0.94}
        ]
        state = MeridianState(input_files=[...], ...)
        result = doc_agent_node(state, config={})
        assert "ner_entities" in result
        assert result["ner_entities"][0]["type"] == "RETENTION_PERIOD"
```

Integration tests may call real APIs but must be gated with a `@pytest.mark.integration` marker and skipped in CI unless `ENABLE_INTEGRATION_TESTS=true` is set.

### Coverage requirements

Minimum 70% coverage is enforced in CI. New code should not lower the coverage percentage. Coverage reports are uploaded to Codecov on every PR.

---

## Submitting a pull request

1. **Open an issue first** for any non-trivial change, to discuss approach before writing code
2. Create a branch from `main` with the naming convention above
3. Make your changes with tests
4. Ensure `pre-commit`, `pytest`, and `mypy` all pass locally
5. Push and open a PR against `main`
6. Fill in the PR template (auto-generated) — include: what changed, why, how to test, screenshots if UI changed
7. A maintainer will review within 3 business days
8. Address review comments; request re-review when ready

### PR checklist

- [ ] Tests added or updated for the changed behavior
- [ ] `pytest` passes locally
- [ ] `mypy` reports no new errors
- [ ] Documentation updated if behavior changes (README, FEATURES, API, ARCHITECTURE as appropriate)
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] No secrets, credentials, or personal data in the diff

---

## Issue reporting

### Bug reports

Use the `bug` issue template. Include:
- Python version, OS, Docker version
- Steps to reproduce (minimal reproduction is best)
- Expected behavior vs actual behavior
- Relevant log output (redact any API keys)
- LangSmith trace URL if the issue is in the agent pipeline

### Feature requests

Use the `feature_request` template. Include:
- The problem you're trying to solve (not just the proposed solution)
- Which HuggingFace task or model you think could help
- Whether you're willing to implement it

---

## Areas where help is welcome

These are explicitly open for external contributions:

- **Additional regulatory corpora** — HIPAA, EU AI Act, PCI-DSS, NIST CSF. See `scripts/ingest_corpus.py` and the existing corpus loaders in `src/rag/corpus/` for the pattern.
- **ColPali integration** — The visual document retrieval agent is partially implemented; full integration with the regulatory image index is needed (tracked in TODO.md).
- **Speaker diarization** — `pyannote/speaker-diarization-3.1` integration in the audio agent (see `src/agents/audio_agent.py`).
- **Multilingual support** — Translation pipeline for non-English policy documents (planned for v1.1).
- **Test coverage** — Any test that raises overall coverage toward 80%.
- **Documentation** — Typo fixes, clearer explanations, additional examples are always welcome.

If you're unsure whether a contribution fits, open a discussion issue before writing code.
