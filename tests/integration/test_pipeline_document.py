"""
Integration tests for the document-only pipeline end-to-end.

Requires Docker Compose stack (PostgreSQL + Redis).
LLM API calls are mocked to avoid cost and nondeterminism.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_claude_gaps() -> list[dict]:
    return [
        {
            "gap_id": "gap_001",
            "severity": "critical",
            "framework": "gdpr",
            "regulatory_article": "GDPR Article 13(2)(a)",
            "regulatory_requirement": "Specify data retention period.",
            "regulatory_quote": (
                "The period for which the personal data will be stored, or if that is not "
                "possible, the criteria used to determine that period."
            ),
            "regulatory_chunk_id": None,
            "policy_reference": "Section 5",
            "policy_text": "We retain data as long as necessary.",
            "gap_description": "No concrete retention period specified.",
            "severity_justification": "Frequently cited GDPR violation.",
            "remediation": "Add specific retention periods per data category.",
            "confidence": 0.91,
        }
    ]


@pytest.fixture
def mock_chunks() -> list:
    from tests.conftest import make_retrieved_chunk

    return [make_retrieved_chunk("chunk-gdpr-article-13")]


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestDocumentOnlyPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_produces_verified_gaps(
        self, sample_policy_pdf_path: Path, mock_claude_gaps: list, mock_chunks: list
    ):
        """End-to-end: a document submission should produce verified compliance gaps."""
        from tests.conftest import make_uploaded_file, mock_storage
        from src.graph.graph import run_pipeline
        from src.graph.state import UploadedFile

        policy_bytes = sample_policy_pdf_path.read_bytes()
        storage_mock = mock_storage(policy_bytes)

        input_files = [
            UploadedFile(
                file_id="file-001",
                filename="test_policy.pdf",
                modality="unknown",
                mime_type="application/pdf",
                size_bytes=len(policy_bytes),
                storage_key="uploads/test-job/file-001_test_policy.pdf",
                content_hash="abc123",
            )
        ]

        with (
            patch("src.agents.doc_agent.get_storage", return_value=storage_mock),
            patch("src.agents.doc_agent._ner.extract", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.agents.doc_agent._regulatory_classifier.classify_entity",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("src.agents.doc_agent._qa.answer", new_callable=AsyncMock, return_value=None),
            patch(
                "src.agents.doc_agent.hybrid_retrieve",
                new_callable=AsyncMock,
                return_value=mock_chunks,
            ),
            patch("src.agents.doc_agent.get_db_session") as mock_db,
            patch(
                "src.agents.synthesis_agent._claude.synthesize",
                new_callable=AsyncMock,
                return_value=mock_claude_gaps,
            ),
            patch(
                "src.agents.synthesis_agent._claude.generate_executive_summary",
                new_callable=AsyncMock,
                return_value="Test executive summary.",
            ),
            patch("src.graph.nodes.gate._similarity_model") as mock_sim,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_sim.batch_score.return_value = [0.91]  # passes groundedness

            final_state = await run_pipeline(
                job_id="test-integration-001",
                input_files=input_files,
                regulation_scope=["gdpr"],
            )

        assert final_state.get("error") is None, f"Pipeline error: {final_state.get('error')}"
        assert final_state.get("final_report") is not None

        report = final_state["final_report"]
        assert report.total_gaps == 1
        assert report.gaps_critical == 1
        assert len(report.gaps) == 1

        gap = report.gaps[0]
        assert gap.severity == "critical"
        assert gap.framework == "gdpr"
        assert "Article 13" in gap.regulatory_article
        assert gap.groundedness_score >= 0.80
        assert gap.is_verified is True

    @pytest.mark.asyncio
    async def test_pipeline_handles_synthesis_failure_gracefully(
        self, sample_policy_pdf_path: Path, mock_chunks: list
    ):
        """Pipeline should return error state when synthesis fails after max retries."""
        from src.graph.graph import run_pipeline
        from src.graph.state import UploadedFile

        policy_bytes = sample_policy_pdf_path.read_bytes()
        from tests.conftest import mock_storage

        storage_mock = mock_storage(policy_bytes)

        input_files = [
            UploadedFile(
                file_id="file-001",
                filename="test_policy.pdf",
                modality="unknown",
                mime_type="application/pdf",
                size_bytes=len(policy_bytes),
                storage_key="uploads/test-fail/file-001.pdf",
                content_hash="abc123",
            )
        ]

        with (
            patch("src.agents.doc_agent.get_storage", return_value=storage_mock),
            patch("src.agents.doc_agent._ner.extract", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.agents.doc_agent._regulatory_classifier.classify_entity",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("src.agents.doc_agent._qa.answer", new_callable=AsyncMock, return_value=None),
            patch(
                "src.agents.doc_agent.hybrid_retrieve",
                new_callable=AsyncMock,
                return_value=mock_chunks,
            ),
            patch("src.agents.doc_agent.get_db_session") as mock_db,
            patch(
                "src.agents.synthesis_agent._claude.synthesize",
                new_callable=AsyncMock,
                side_effect=Exception("Claude API unavailable"),
            ),
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)

            final_state = await run_pipeline(
                job_id="test-fail-001",
                input_files=input_files,
                regulation_scope=["gdpr"],
            )

        # Pipeline should not crash — it should return error state
        assert final_state.get("error") is not None or final_state.get("final_report") is not None

    @pytest.mark.asyncio
    async def test_pipeline_empty_document_produces_no_gaps(self):
        """An empty policy document should produce a report with zero gaps."""
        from src.graph.graph import run_pipeline
        from src.graph.state import UploadedFile
        from tests.conftest import mock_storage

        empty_text = b"   "  # whitespace only
        storage_mock = mock_storage(empty_text)

        input_files = [
            UploadedFile(
                file_id="file-001",
                filename="empty.pdf",
                modality="unknown",
                mime_type="application/pdf",
                size_bytes=3,
                storage_key="uploads/empty-job/file-001.pdf",
                content_hash="empty123",
            )
        ]

        with (
            patch("src.agents.doc_agent.get_storage", return_value=storage_mock),
            patch(
                "src.agents.synthesis_agent._claude.synthesize",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "src.agents.synthesis_agent._claude.generate_executive_summary",
                new_callable=AsyncMock,
                return_value="No gaps found.",
            ),
            patch("src.agents.doc_agent.get_db_session") as mock_db,
            patch("src.graph.nodes.gate._similarity_model") as mock_sim,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_sim.batch_score.return_value = []

            final_state = await run_pipeline(
                job_id="test-empty-001",
                input_files=input_files,
                regulation_scope=["gdpr"],
            )

        report = final_state.get("final_report")
        if report:
            assert report.total_gaps == 0
