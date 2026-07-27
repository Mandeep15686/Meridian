"""Unit tests for the document specialist agent node."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.state import AgentExtraction, Entity, MeridianState, QAResult, UploadedFile
from tests.conftest import (
    SAMPLE_POLICY_TEXT,
    make_retrieved_chunk,
    make_state,
    make_uploaded_file,
    mock_storage,
)


class TestDocAgent:
    """Tests for doc_agent_node in src/agents/doc_agent.py."""

    @pytest.mark.asyncio
    async def test_extracts_retention_period_entity(self):
        """Doc agent should identify a RETENTION_PERIOD entity from policy text."""
        from src.models.nlp import NEREntity

        file = make_uploaded_file(filename="policy.pdf", modality="document")
        state = make_state(input_files=[file])
        state["_current_file"] = file

        storage_mock = mock_storage(SAMPLE_POLICY_TEXT.encode())
        ner_entities = [
            NEREntity(entity_group="MISC", word="as long as necessary",
                      start=100, end=120, score=0.82),
        ]
        retrieval_chunks = [make_retrieved_chunk()]

        with (
            patch("src.agents.doc_agent.get_storage", return_value=storage_mock),
            patch("src.agents.doc_agent._ner.extract", new_callable=AsyncMock,
                  return_value=ner_entities),
            patch("src.agents.doc_agent._regulatory_classifier.classify_entity",
                  new_callable=AsyncMock, return_value="RETENTION_PERIOD"),
            patch("src.agents.doc_agent._qa.answer",
                  new_callable=AsyncMock, return_value=None),
            patch("src.agents.doc_agent.hybrid_retrieve",
                  new_callable=AsyncMock, return_value=retrieval_chunks),
            patch("src.agents.doc_agent.get_db_session") as mock_db,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)

            from src.agents.doc_agent import doc_agent_node
            result = await doc_agent_node(state)

        assert "raw_extractions" in result
        extractions: list[AgentExtraction] = result["raw_extractions"]
        assert len(extractions) == 1
        assert extractions[0].agent == "doc_agent"

        entities = extractions[0].ner_entities
        retention_entities = [e for e in entities if e.type == "RETENTION_PERIOD"]
        assert len(retention_entities) >= 1, "Should find at least one RETENTION_PERIOD entity"

    @pytest.mark.asyncio
    async def test_handles_empty_text_gracefully(self):
        """Doc agent should return empty extractions for a blank document."""
        file = make_uploaded_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        storage_mock = mock_storage(b"   ")  # whitespace only

        with (
            patch("src.agents.doc_agent.get_storage", return_value=storage_mock),
        ):
            from src.agents.doc_agent import doc_agent_node
            result = await doc_agent_node(state)

        assert result["raw_extractions"] == []

    @pytest.mark.asyncio
    async def test_returns_retrieved_chunks_in_state(self):
        """Doc agent should surface retrieved chunks to the shared state."""
        file = make_uploaded_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        expected_chunks = [make_retrieved_chunk("chunk-001"), make_retrieved_chunk("chunk-002")]
        storage_mock = mock_storage(SAMPLE_POLICY_TEXT.encode())

        with (
            patch("src.agents.doc_agent.get_storage", return_value=storage_mock),
            patch("src.agents.doc_agent._ner.extract",
                  new_callable=AsyncMock, return_value=[]),
            patch("src.agents.doc_agent._regulatory_classifier.classify_entity",
                  new_callable=AsyncMock, return_value=None),
            patch("src.agents.doc_agent._qa.answer",
                  new_callable=AsyncMock, return_value=None),
            patch("src.agents.doc_agent.hybrid_retrieve",
                  new_callable=AsyncMock, return_value=expected_chunks),
            patch("src.agents.doc_agent.get_db_session") as mock_db,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)

            from src.agents.doc_agent import doc_agent_node
            result = await doc_agent_node(state)

        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) == len(expected_chunks)

    @pytest.mark.asyncio
    async def test_storage_failure_returns_error(self):
        """Doc agent should return error state when file download fails."""
        file = make_uploaded_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        storage_mock = AsyncMock()
        storage_mock.download.side_effect = FileNotFoundError("File not found in storage")

        with patch("src.agents.doc_agent.get_storage", return_value=storage_mock):
            from src.agents.doc_agent import doc_agent_node
            result = await doc_agent_node(state)

        assert result.get("error") is not None
        assert result.get("error_stage") == "doc_agent"
        assert result["raw_extractions"] == []

    @pytest.mark.asyncio
    async def test_qa_answers_included_in_extraction(self):
        """QA results should be included in the AgentExtraction output."""
        from src.models.nlp import QAAnswer

        file = make_uploaded_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        storage_mock = mock_storage(SAMPLE_POLICY_TEXT.encode())
        qa_answer = QAAnswer(answer="5 years", score=0.92, start=50, end=57)

        with (
            patch("src.agents.doc_agent.get_storage", return_value=storage_mock),
            patch("src.agents.doc_agent._ner.extract",
                  new_callable=AsyncMock, return_value=[]),
            patch("src.agents.doc_agent._regulatory_classifier.classify_entity",
                  new_callable=AsyncMock, return_value=None),
            patch("src.agents.doc_agent._qa.answer",
                  new_callable=AsyncMock, return_value=qa_answer),
            patch("src.agents.doc_agent.hybrid_retrieve",
                  new_callable=AsyncMock, return_value=[]),
            patch("src.agents.doc_agent.get_db_session") as mock_db,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)

            from src.agents.doc_agent import doc_agent_node
            result = await doc_agent_node(state)

        assert result["raw_extractions"]
        extraction = result["raw_extractions"][0]
        assert len(extraction.qa_results) > 0
        assert any(qa.answer == "5 years" for qa in extraction.qa_results)

    @pytest.mark.asyncio
    async def test_duration_ms_recorded(self):
        """Doc agent should record processing duration in milliseconds."""
        file = make_uploaded_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file
        storage_mock = mock_storage(SAMPLE_POLICY_TEXT.encode())

        with (
            patch("src.agents.doc_agent.get_storage", return_value=storage_mock),
            patch("src.agents.doc_agent._ner.extract",
                  new_callable=AsyncMock, return_value=[]),
            patch("src.agents.doc_agent._regulatory_classifier.classify_entity",
                  new_callable=AsyncMock, return_value=None),
            patch("src.agents.doc_agent._qa.answer",
                  new_callable=AsyncMock, return_value=None),
            patch("src.agents.doc_agent.hybrid_retrieve",
                  new_callable=AsyncMock, return_value=[]),
            patch("src.agents.doc_agent.get_db_session") as mock_db,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)

            from src.agents.doc_agent import doc_agent_node
            result = await doc_agent_node(state)

        if result["raw_extractions"]:
            extraction = result["raw_extractions"][0]
            assert extraction.duration_ms is not None
            assert extraction.duration_ms >= 0

    def test_compliance_questions_defined(self):
        """Compliance question list should be non-empty and contain expected keys."""
        from src.agents.doc_agent import COMPLIANCE_QUESTIONS

        assert len(COMPLIANCE_QUESTIONS) > 0
        questions_text = " ".join(COMPLIANCE_QUESTIONS).lower()
        assert "retention" in questions_text, "Should ask about data retention"
        assert "lawful basis" in questions_text or "lawful" in questions_text
        assert "consent" in questions_text
