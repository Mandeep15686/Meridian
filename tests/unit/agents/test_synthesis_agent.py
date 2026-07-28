"""Unit tests for the synthesis agent and report generation nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.graph.state import (
    AgentExtraction,
    Entity,
    TranscriptSegment,
)
from tests.conftest import (
    make_retrieved_chunk,
    make_state,
    make_verified_gap,
    mock_claude_client,
)

# ── Context formatter tests ──────────────────────────────────────────────────


class TestFormatRegulatoryContext:
    """Tests for _format_regulatory_context helper."""

    def test_empty_chunks_returns_fallback(self):
        from src.agents.synthesis_agent import _format_regulatory_context

        result = _format_regulatory_context([])
        assert result == "No regulatory context retrieved."

    def test_single_chunk_formatted(self):
        from src.agents.synthesis_agent import _format_regulatory_context

        chunk = make_retrieved_chunk(
            chunk_id="c1",
            regulation="gdpr",
            article="Article 5",
            content="Personal data shall be processed lawfully.",
        )
        result = _format_regulatory_context([chunk])
        assert "[1]" in result
        assert "GDPR" in result
        assert "Article 5" in result
        assert "Personal data shall be processed lawfully." in result
        assert "EU" in result

    def test_multiple_chunks_numbered(self):
        from src.agents.synthesis_agent import _format_regulatory_context

        chunks = [
            make_retrieved_chunk(chunk_id="c1", article="Article 5"),
            make_retrieved_chunk(chunk_id="c2", article="Article 6"),
        ]
        result = _format_regulatory_context(chunks)
        assert "[1]" in result
        assert "[2]" in result

    def test_chunk_without_article(self):
        from src.agents.synthesis_agent import _format_regulatory_context

        chunk = make_retrieved_chunk(
            chunk_id="c1",
            article=None,
            content="Personal data shall be processed lawfully.",
        )
        result = _format_regulatory_context([chunk])
        assert "GDPR" in result
        # Should not contain " — None"
        assert " — None" not in result


class TestFormatPolicyContent:
    """Tests for _format_policy_content helper."""

    def test_empty_extractions_returns_fallback(self):
        from src.agents.synthesis_agent import _format_policy_content

        result = _format_policy_content([])
        assert result == "No policy content extracted."

    def test_extraction_with_raw_text(self):
        from src.agents.synthesis_agent import _format_policy_content

        ext = AgentExtraction(
            agent="doc_agent",
            file_id="f1",
            raw_text="Privacy policy content here.",
        )
        result = _format_policy_content([ext])
        assert "Doc Agent" in result
        assert "f1" in result
        assert "Privacy policy content here." in result

    def test_extraction_with_summary(self):
        from src.agents.synthesis_agent import _format_policy_content

        ext = AgentExtraction(
            agent="doc_agent",
            file_id="f1",
            summary="A brief summary of findings.",
        )
        result = _format_policy_content([ext])
        assert "Summary: A brief summary of findings." in result

    def test_extraction_with_transcript(self):
        from src.agents.synthesis_agent import _format_policy_content

        ext = AgentExtraction(
            agent="audio_agent",
            file_id="f2",
            transcript=[
                TranscriptSegment(
                    speaker="Alice", start=0.0, end=5.0, text="We need GDPR compliance."
                ),
                TranscriptSegment(
                    speaker="Bob", start=5.0, end=10.0, text="Agreed, let us review."
                ),
            ],
        )
        result = _format_policy_content([ext])
        assert "Transcript (excerpt):" in result
        assert "GDPR compliance" in result

    def test_extraction_with_image_caption(self):
        from src.agents.synthesis_agent import _format_policy_content

        ext = AgentExtraction(
            agent="vision_agent",
            file_id="f3",
            image_caption="A diagram showing data flows.",
        )
        result = _format_policy_content([ext])
        assert "Image caption: A diagram showing data flows." in result

    def test_extraction_with_vqa_results(self):
        from src.agents.synthesis_agent import _format_policy_content

        ext = AgentExtraction(
            agent="vision_agent",
            file_id="f3",
            vqa_results=[{"question": "What is shown?", "answer": "Data flow", "score": 0.95}],
        )
        result = _format_policy_content([ext])
        assert "VQA findings:" in result
        assert "What is shown?" in result
        assert "Data flow" in result

    def test_extraction_with_tapas_answers(self):
        from src.agents.synthesis_agent import _format_policy_content

        ext = AgentExtraction(
            agent="data_agent",
            file_id="f4",
            tapas_answers=[{"question": "Max retention?", "answer": "365 days"}],
        )
        result = _format_policy_content([ext])
        assert "Table QA findings:" in result
        assert "Max retention?" in result

    def test_extraction_with_table_summary(self):
        from src.agents.synthesis_agent import _format_policy_content

        ext = AgentExtraction(
            agent="data_agent",
            file_id="f4",
            table_summary="Audit log with 4 records.",
        )
        result = _format_policy_content([ext])
        assert "Table summary: Audit log with 4 records." in result


class TestFormatNERSummary:
    """Tests for _format_ner_summary helper."""

    def test_empty_entities_returns_fallback(self):
        from src.agents.synthesis_agent import _format_ner_summary

        result = _format_ner_summary([])
        assert result == "No regulatory entities extracted."

    def test_single_entity_formatted(self):
        from src.agents.synthesis_agent import _format_ner_summary

        entities = [
            Entity(
                type="RETENTION_PERIOD",
                text="5 years",
                start=10,
                end=17,
                confidence=0.92,
            )
        ]
        result = _format_ner_summary(entities)
        assert "RETENTION_PERIOD" in result
        assert "5 years" in result

    def test_multiple_types_grouped(self):
        from src.agents.synthesis_agent import _format_ner_summary

        entities = [
            Entity(type="RETENTION_PERIOD", text="5 years", start=0, end=7, confidence=0.9),
            Entity(type="DPO_MENTION", text="privacy@acme.com", start=10, end=26, confidence=0.95),
            Entity(type="RETENTION_PERIOD", text="90 days", start=30, end=37, confidence=0.85),
        ]
        result = _format_ner_summary(entities)
        assert "RETENTION_PERIOD" in result
        assert "DPO_MENTION" in result
        assert "5 years" in result
        assert "90 days" in result

    def test_deduplicates_entities(self):
        from src.agents.synthesis_agent import _format_ner_summary

        entities = [
            Entity(type="ORG", text="Acme Corp", start=0, end=9, confidence=0.99),
            Entity(type="ORG", text="Acme Corp", start=50, end=59, confidence=0.98),
        ]
        result = _format_ner_summary(entities)
        # Should only appear once due to deduplication
        assert result.count("Acme Corp") == 1


# ── Synthesis node tests ─────────────────────────────────────────────────────


class TestSynthesisNode:
    """Tests for synthesis_node."""

    @pytest.mark.asyncio
    async def test_empty_state_returns_no_gaps(self):
        """Synthesis node should return empty gaps when there is nothing to synthesise."""
        state = make_state(raw_extractions=[], retrieved_chunks=[])

        with patch("src.agents.synthesis_agent._claude", mock_claude_client()):
            from src.agents.synthesis_agent import synthesis_node

            result = await synthesis_node(state)

        assert result["candidate_gaps"] == []
        assert result["failed_gap_ids"] == []

    @pytest.mark.asyncio
    async def test_produces_candidate_gaps(self):
        """Synthesis node should return CandidateGap objects from Claude output."""
        ext = AgentExtraction(
            agent="doc_agent",
            file_id="f1",
            raw_text="We retain data as long as necessary.",
        )
        state = make_state(
            raw_extractions=[ext],
            retrieved_chunks=[make_retrieved_chunk()],
        )

        claude_mock = mock_claude_client()
        with patch("src.agents.synthesis_agent._claude", claude_mock):
            from src.agents.synthesis_agent import synthesis_node

            result = await synthesis_node(state)

        assert len(result["candidate_gaps"]) == 1
        gap = result["candidate_gaps"][0]
        assert gap.gap_id == "gap_001"
        assert gap.severity == "critical"
        assert result["failed_gap_ids"] == []

    @pytest.mark.asyncio
    async def test_claude_failure_returns_error(self):
        """Synthesis node should return error state when Claude API fails."""
        ext = AgentExtraction(agent="doc_agent", file_id="f1", raw_text="Policy text.")
        state = make_state(
            raw_extractions=[ext],
            retrieved_chunks=[make_retrieved_chunk()],
        )

        claude_mock = AsyncMock()
        claude_mock.synthesize.side_effect = RuntimeError("API timeout")

        with patch("src.agents.synthesis_agent._claude", claude_mock):
            from src.agents.synthesis_agent import synthesis_node

            result = await synthesis_node(state)

        assert result["error"] == "API timeout"
        assert result["error_stage"] == "synthesis"
        assert result["candidate_gaps"] == []

    @pytest.mark.asyncio
    async def test_metadata_includes_duration(self):
        """Synthesis node should record duration in metadata."""
        ext = AgentExtraction(agent="doc_agent", file_id="f1", raw_text="Content.")
        state = make_state(
            raw_extractions=[ext],
            retrieved_chunks=[make_retrieved_chunk()],
        )

        claude_mock = mock_claude_client()
        with patch("src.agents.synthesis_agent._claude", claude_mock):
            from src.agents.synthesis_agent import synthesis_node

            result = await synthesis_node(state)

        assert "metadata" in result
        assert "synthesis_duration_ms" in result["metadata"]
        assert result["metadata"]["synthesis_duration_ms"] >= 0


# ── Report node tests ────────────────────────────────────────────────────────


class TestReportNode:
    """Tests for report_node."""

    @pytest.mark.asyncio
    async def test_report_with_verified_gaps(self):
        """Report node should assemble a ComplianceReport from verified gaps."""
        gaps = [
            make_verified_gap(gap_id="g1", severity="critical"),
            make_verified_gap(gap_id="g2", severity="major"),
            make_verified_gap(gap_id="g3", severity="minor"),
        ]
        state = make_state(verified_gaps=gaps, regulation_scope=["gdpr"])

        claude_mock = mock_claude_client()
        with patch("src.agents.synthesis_agent._claude", claude_mock):
            from src.agents.synthesis_agent import report_node

            result = await report_node(state)

        report = result["final_report"]
        assert report.job_id == state["job_id"]
        assert report.total_gaps == 3
        assert report.gaps_critical == 1
        assert report.gaps_major == 1
        assert report.gaps_minor == 1
        assert report.executive_summary == "Test executive summary."

    @pytest.mark.asyncio
    async def test_report_empty_gaps(self):
        """Report node should handle zero verified gaps gracefully."""
        state = make_state(verified_gaps=[], regulation_scope=["gdpr"])

        claude_mock = mock_claude_client()
        with patch("src.agents.synthesis_agent._claude", claude_mock):
            from src.agents.synthesis_agent import report_node

            result = await report_node(state)

        report = result["final_report"]
        assert report.total_gaps == 0
        assert report.groundedness_pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_report_compliance_scores_per_framework(self):
        """Report should compute compliance scores for each framework in scope."""
        gaps = [make_verified_gap(gap_id="g1", severity="critical")]
        state = make_state(verified_gaps=gaps, regulation_scope=["gdpr", "hipaa"])

        claude_mock = mock_claude_client()
        with patch("src.agents.synthesis_agent._claude", claude_mock):
            from src.agents.synthesis_agent import report_node

            result = await report_node(state)

        report = result["final_report"]
        assert "gdpr" in report.compliance_scores
        assert "hipaa" in report.compliance_scores
        assert report.compliance_scores["hipaa"]["gaps"] == 0
        assert report.compliance_scores["hipaa"]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_report_claude_failure_uses_fallback_summary(self):
        """Report node should use fallback summary when Claude fails."""
        gaps = [make_verified_gap(gap_id="g1", severity="critical")]
        state = make_state(verified_gaps=gaps, regulation_scope=["gdpr"])

        claude_mock = AsyncMock()
        claude_mock.generate_executive_summary.side_effect = RuntimeError("API error")

        with patch("src.agents.synthesis_agent._claude", claude_mock):
            from src.agents.synthesis_agent import report_node

            result = await report_node(state)

        report = result["final_report"]
        assert "1 compliance gap" in report.executive_summary
        assert "1 critical" in report.executive_summary

    @pytest.mark.asyncio
    async def test_report_metadata_includes_duration(self):
        """Report node should record duration and stats in metadata."""
        state = make_state(verified_gaps=[], regulation_scope=["gdpr"])

        claude_mock = mock_claude_client()
        with patch("src.agents.synthesis_agent._claude", claude_mock):
            from src.agents.synthesis_agent import report_node

            result = await report_node(state)

        assert "metadata" in result
        assert "report_duration_ms" in result["metadata"]
        assert result["metadata"]["total_gaps"] == 0
        assert result["metadata"]["groundedness_pass_rate"] == 1.0
