"""
Shared pytest fixtures for Meridian test suite.

Provides:
- In-memory test state factories
- Mock HF API response builders
- Async test database session
- Sample fixture data helpers
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.state import (
    AgentExtraction,
    CandidateGap,
    MeridianState,
    RetrievedChunk,
    UploadedFile,
    VerifiedGap,
)

# ── Fixtures directory ────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DOCUMENTS_DIR = FIXTURES_DIR / "documents"
AUDIO_DIR = FIXTURES_DIR / "audio"
IMAGES_DIR = FIXTURES_DIR / "images"
TABLES_DIR = FIXTURES_DIR / "tables"


# ── Event loop ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def event_loop():
    """Provide a shared event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Sample data factories ─────────────────────────────────────────────────────


def make_uploaded_file(
    filename: str = "test_policy.pdf",
    modality: str = "document",
    file_id: str = "file-001",
    storage_key: str = "uploads/job-001/file-001_test_policy.pdf",
) -> UploadedFile:
    return UploadedFile(
        file_id=file_id,
        filename=filename,
        modality=modality,
        mime_type="application/pdf",
        size_bytes=50_000,
        storage_key=storage_key,
        content_hash="abc123def456",
    )


def make_retrieved_chunk(
    chunk_id: str = "chunk-gdpr-001",
    regulation: str = "gdpr",
    article: str = "Article 13(2)(a)",
    content: str | None = None,
    final_rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        regulation=regulation,
        article=article,
        content=content
        or (
            f"{regulation.upper()} {article}: "
            "The period for which the personal data will be stored, or if that is not "
            "possible, the criteria used to determine that period."
        ),
        jurisdiction="EU",
        dense_score=0.92,
        bm25_score=0.85,
        rrf_score=0.031,
        rerank_score=0.94,
        final_rank=final_rank,
    )


def make_candidate_gap(
    gap_id: str = "gap_001",
    severity: str = "critical",
    framework: str = "gdpr",
    article: str = "GDPR Article 13(2)(a)",
    chunk_id: str = "chunk-gdpr-001",
    confidence: float = 0.91,
) -> CandidateGap:
    return CandidateGap(
        gap_id=gap_id,
        severity=severity,
        framework=framework,
        regulatory_article=article,
        regulatory_requirement=(
            "Controllers must specify the period for which personal data will be stored."
        ),
        regulatory_quote=(
            "The period for which the personal data will be stored, or if that is not "
            "possible, the criteria used to determine that period."
        ),
        regulatory_chunk_id=chunk_id,
        policy_reference="Section 4 — Data Retention",
        policy_text="We retain your data as long as necessary.",
        gap_description=(
            "The policy does not specify a concrete retention period or the criteria used "
            "to determine that period, as required by GDPR Article 13(2)(a)."
        ),
        severity_justification=(
            "Critical: absent retention period is a frequently cited GDPR violation."
        ),
        remediation=(
            "Add a retention table to Section 4 specifying each data category, its "
            "retention period, and the legal basis for that period."
        ),
        confidence=confidence,
    )


def make_verified_gap(
    gap_id: str = "gap_001",
    severity: str = "critical",
    groundedness_score: float = 0.91,
    is_uncertain: bool = False,
) -> VerifiedGap:
    cg = make_candidate_gap(gap_id=gap_id, severity=severity)
    return VerifiedGap(
        gap_id=cg.gap_id,
        severity=cg.severity,
        framework=cg.framework,
        regulatory_article=cg.regulatory_article,
        regulatory_requirement=cg.regulatory_requirement,
        regulatory_quote=cg.regulatory_quote,
        regulatory_chunk_id=cg.regulatory_chunk_id,
        policy_reference=cg.policy_reference,
        policy_text=cg.policy_text,
        gap_description=cg.gap_description,
        severity_justification=cg.severity_justification,
        remediation=cg.remediation,
        confidence=cg.confidence,
        groundedness_score=groundedness_score,
        is_verified=not is_uncertain,
        is_uncertain=is_uncertain,
    )


def make_state(
    job_id: str = "01JTEST000000000000000000",
    regulation_scope: list[str] | None = None,
    input_files: list[UploadedFile] | None = None,
    raw_extractions: list[AgentExtraction] | None = None,
    retrieved_chunks: list[RetrievedChunk] | None = None,
    candidate_gaps: list[CandidateGap] | None = None,
    verified_gaps: list[VerifiedGap] | None = None,
    groundedness_scores: dict | None = None,
    synthesis_retries: int = 0,
    **kwargs,
) -> MeridianState:
    """Build a MeridianState dict for testing."""
    return {
        "job_id": job_id,
        "input_files": (
            input_files if input_files is not None else [make_uploaded_file()]
        ),
        "regulation_scope": regulation_scope or ["gdpr"],
        "options": {},
        "raw_extractions": raw_extractions or [],
        "retrieved_chunks": (
            retrieved_chunks
            if retrieved_chunks is not None
            else [make_retrieved_chunk()]
        ),
        "ner_entities": [],
        "candidate_gaps": candidate_gaps or [],
        "verified_gaps": verified_gaps or [],
        "groundedness_scores": groundedness_scores or {},
        "failed_gap_ids": [],
        "synthesis_retries": synthesis_retries,
        "final_report": None,
        "error": None,
        "error_stage": None,
        "metadata": {},
        **kwargs,
    }


def make_state_with_gaps(
    groundedness_scores: dict[str, float],
    synthesis_retries: int = 0,
    gap_ids: list[str] | None = None,
) -> MeridianState:
    """Build state with candidate gaps for gate testing."""
    ids = gap_ids or list(groundedness_scores.keys())
    gaps = [make_candidate_gap(gap_id=gid) for gid in ids]
    return make_state(
        candidate_gaps=gaps,
        groundedness_scores=groundedness_scores,
        synthesis_retries=synthesis_retries,
        retrieved_chunks=[make_retrieved_chunk(chunk_id="chunk-gdpr-001")],
    )


# ── Text fixtures ─────────────────────────────────────────────────────────────

SAMPLE_POLICY_TEXT = """
Privacy Policy — Acme Corp

1. Introduction
Acme Corp ("we", "us", "our") is committed to protecting your personal data.
This Privacy Policy explains how we collect, use, and protect your information.

2. Data Controller
Acme Corp, 123 Business Street, London, UK.

3. Data We Collect
We collect the following categories of personal data:
- Name and email address (account creation)
- Payment information (purchases)
- Usage data (product analytics)

4. How We Use Your Data
We process your personal data for the following purposes:
- Providing our services under our Terms of Service
- Sending you marketing communications (with your consent)
- Improving our products through analytics

5. Data Retention
We retain your data as long as your account is active or as needed to provide
services. We may retain certain information for longer periods as required by law.

6. Your Rights
You may request access to, correction of, or deletion of your personal data.
Please contact privacy@acme.com.

7. Contact Us
For privacy inquiries: privacy@acme.com
"""

SAMPLE_AUDIO_TRANSCRIPT = """
Meeting transcript — Q4 Compliance Review
Participants: Alice (DPO), Bob (Legal), Carol (Engineering)

Alice: We need to review our GDPR compliance before the Q4 deadline.
The ICO audit flagged our data retention policy — we don't specify
concrete retention periods for each data category.

Bob: That's correct. Article 13(2)(a) requires us to tell users specifically
how long we keep their data, not just "as long as necessary."

Carol: The current policy says "as long as necessary" which is too vague.
We need to add a retention table with specific periods.

Alice: Agreed. Also, we should check our lawful basis for marketing emails.
Are we relying on consent or legitimate interests?

Bob: Currently consent, but the consent mechanism on the website
isn't compliant — the checkbox is pre-ticked.
"""


@pytest.fixture
def sample_policy_text() -> str:
    return SAMPLE_POLICY_TEXT


@pytest.fixture
def sample_audio_transcript() -> str:
    return SAMPLE_AUDIO_TRANSCRIPT


@pytest.fixture
def sample_policy_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal PDF file for testing."""
    pdf_path = tmp_path / "test_policy.pdf"
    # Minimal valid PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R
/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Privacy Policy) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000369 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
452
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Create a sample audit log CSV for testing."""
    csv_path = tmp_path / "audit_log.csv"
    csv_path.write_text(
        "timestamp,user_id,action,data_category,retention_days\n"
        "2026-01-15T09:00:00,user_001,READ,email,365\n"
        "2026-01-15T09:05:00,user_002,EXPORT,payment_data,2555\n"
        "2026-01-15T02:30:00,user_003,DELETE,marketing,90\n"  # outside business hours
        "2026-01-15T09:10:00,admin_001,BULK_EXPORT,all_data,9999\n"  # anomaly
    )
    return csv_path


# ── Mock HF model factories ───────────────────────────────────────────────────


def mock_ner_model(entities: list[dict] | None = None):
    """Build a mock NERModel that returns preset entities."""
    from src.models.nlp import NEREntity

    default_entities = entities or [
        NEREntity(entity_group="MISC", word="5 years", start=20, end=27, score=0.92),
        NEREntity(entity_group="ORG", word="Acme Corp", start=0, end=9, score=0.99),
    ]
    mock = AsyncMock()
    mock.extract.return_value = default_entities
    return mock


def mock_qa_model(answer: str = "5 years", score: float = 0.88):
    """Build a mock QAModel that returns a preset answer."""
    from src.models.nlp import QAAnswer

    mock = AsyncMock()
    mock.answer.return_value = QAAnswer(answer=answer, score=score, start=20, end=27)
    return mock


def mock_claude_client(gaps: list[dict] | None = None):
    """Build a mock ClaudeClient that returns preset gap dicts."""
    default_gaps = gaps or [
        {
            "gap_id": "gap_001",
            "severity": "critical",
            "framework": "gdpr",
            "regulatory_article": "GDPR Article 13(2)(a)",
            "regulatory_requirement": "Controller must specify storage period.",
            "regulatory_quote": (
                "The period for which the personal data will be stored, or if that is not "
                "possible, the criteria used to determine that period."
            ),
            "regulatory_chunk_id": "chunk-gdpr-001",
            "policy_reference": "Section 5 — Data Retention",
            "policy_text": "We retain your data as long as necessary.",
            "gap_description": "No concrete retention period specified.",
            "severity_justification": "Critical: frequently cited GDPR violation.",
            "remediation": "Add a retention table with specific periods per data category.",
            "confidence": 0.91,
        }
    ]
    mock = AsyncMock()
    mock.synthesize.return_value = default_gaps
    mock.generate_executive_summary.return_value = "Test executive summary."
    return mock


def mock_storage(file_bytes: bytes = b"test content"):
    """Build a mock storage backend."""
    mock = AsyncMock()
    mock.download.return_value = file_bytes
    mock.upload.return_value = "uploads/test/file.pdf"
    mock.exists.return_value = True
    return mock


# ── Database fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_db_session():
    """Return a MagicMock async session for unit tests that don't need real DB."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session
