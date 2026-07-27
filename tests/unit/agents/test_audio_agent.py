"""Unit tests for the audio specialist agent node."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.state import AgentExtraction, TranscriptSegment
from tests.conftest import SAMPLE_AUDIO_TRANSCRIPT, make_state, make_uploaded_file, mock_storage


def _make_audio_file(filename: str = "meeting.mp3") -> object:
    return make_uploaded_file(
        filename=filename,
        modality="audio",
        file_id="audio-001",
        storage_key="uploads/job-001/audio-001_meeting.mp3",
    )


class TestAudioAgent:

    @pytest.mark.asyncio
    async def test_transcription_included_in_extraction(self):
        """Audio agent should include full transcript text in the extraction."""
        from src.models.asr import Transcription, TranscriptSegment as ASRSegment

        file = _make_audio_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        mock_audio_bytes = b"fake_mp3_content"
        storage_mock = mock_storage(mock_audio_bytes)

        mock_transcription = Transcription(
            full_text=SAMPLE_AUDIO_TRANSCRIPT,
            segments=[
                ASRSegment(speaker="SPEAKER_00", start=0.0, end=8.4,
                           text="We need to review our GDPR compliance."),
                ASRSegment(speaker="SPEAKER_01", start=8.7, end=14.0,
                           text="Article 13(2)(a) requires specific retention periods."),
            ],
            language="en",
            duration_seconds=62.0,
            word_count=len(SAMPLE_AUDIO_TRANSCRIPT.split()),
        )

        with (
            patch("src.agents.audio_agent.get_storage", return_value=storage_mock),
            patch("src.agents.audio_agent._asr.transcribe",
                  new_callable=AsyncMock, return_value=mock_transcription),
            patch("src.agents.audio_agent._summarizer.summarize",
                  new_callable=AsyncMock, return_value="GDPR compliance review discussed."),
        ):
            from src.agents.audio_agent import audio_agent_node
            result = await audio_agent_node(state)

        assert "raw_extractions" in result
        extractions: list[AgentExtraction] = result["raw_extractions"]
        assert len(extractions) == 1

        extraction = extractions[0]
        assert extraction.agent == "audio_agent"
        assert extraction.raw_text is not None
        assert len(extraction.raw_text) > 0
        assert "GDPR" in extraction.raw_text or "compliance" in extraction.raw_text.lower()

    @pytest.mark.asyncio
    async def test_transcript_segments_in_extraction(self):
        """Audio agent should include structured transcript segments."""
        from src.models.asr import Transcription, TranscriptSegment as ASRSegment

        file = _make_audio_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        segments = [
            ASRSegment(speaker="SPEAKER_00", start=0.0, end=5.0, text="First segment."),
            ASRSegment(speaker="SPEAKER_01", start=5.5, end=10.0, text="Second segment."),
        ]
        mock_transcription = Transcription(
            full_text="First segment. Second segment.",
            segments=segments,
            language="en",
            duration_seconds=10.0,
            word_count=4,
        )

        with (
            patch("src.agents.audio_agent.get_storage", return_value=mock_storage(b"fake")),
            patch("src.agents.audio_agent._asr.transcribe",
                  new_callable=AsyncMock, return_value=mock_transcription),
            patch("src.agents.audio_agent._summarizer.summarize",
                  new_callable=AsyncMock, return_value="Summary."),
        ):
            from src.agents.audio_agent import audio_agent_node
            result = await audio_agent_node(state)

        extraction = result["raw_extractions"][0]
        assert extraction.transcript is not None
        assert len(extraction.transcript) == 2
        assert extraction.transcript[0].speaker == "SPEAKER_00"
        assert extraction.transcript[1].speaker == "SPEAKER_01"

    @pytest.mark.asyncio
    async def test_speakers_extracted(self):
        """Audio agent should surface unique speaker labels."""
        from src.models.asr import Transcription, TranscriptSegment as ASRSegment

        file = _make_audio_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        segments = [
            ASRSegment(speaker="SPEAKER_00", start=0.0, end=5.0, text="Alice speaking."),
            ASRSegment(speaker="SPEAKER_01", start=5.5, end=10.0, text="Bob speaking."),
            ASRSegment(speaker="SPEAKER_00", start=11.0, end=15.0, text="Alice again."),
        ]
        mock_transcription = Transcription(
            full_text="Alice speaking. Bob speaking. Alice again.",
            segments=segments,
            language="en",
            duration_seconds=15.0,
            word_count=6,
        )

        with (
            patch("src.agents.audio_agent.get_storage", return_value=mock_storage(b"fake")),
            patch("src.agents.audio_agent._asr.transcribe",
                  new_callable=AsyncMock, return_value=mock_transcription),
            patch("src.agents.audio_agent._summarizer.summarize",
                  new_callable=AsyncMock, return_value="Summary."),
        ):
            from src.agents.audio_agent import audio_agent_node
            result = await audio_agent_node(state)

        extraction = result["raw_extractions"][0]
        assert extraction.speakers is not None
        assert set(extraction.speakers) == {"SPEAKER_00", "SPEAKER_01"}

    @pytest.mark.asyncio
    async def test_empty_transcription_returns_empty_extractions(self):
        """An audio file that produces empty transcription should return no extractions."""
        from src.models.asr import Transcription

        file = _make_audio_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        empty_transcription = Transcription(
            full_text="   ",
            segments=[],
            language="en",
            duration_seconds=5.0,
            word_count=0,
        )

        with (
            patch("src.agents.audio_agent.get_storage", return_value=mock_storage(b"fake")),
            patch("src.agents.audio_agent._asr.transcribe",
                  new_callable=AsyncMock, return_value=empty_transcription),
        ):
            from src.agents.audio_agent import audio_agent_node
            result = await audio_agent_node(state)

        assert result["raw_extractions"] == []

    @pytest.mark.asyncio
    async def test_storage_failure_returns_error_state(self):
        """Audio agent should return error state when file download fails."""
        file = _make_audio_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        storage_mock = AsyncMock()
        storage_mock.download.side_effect = FileNotFoundError("Audio not found")

        with patch("src.agents.audio_agent.get_storage", return_value=storage_mock):
            from src.agents.audio_agent import audio_agent_node
            result = await audio_agent_node(state)

        assert result.get("error") is not None
        assert result.get("error_stage") == "audio_agent"
        assert result["raw_extractions"] == []

    @pytest.mark.asyncio
    async def test_compliance_summary_generated(self):
        """Audio agent should generate a summary of compliance-relevant statements."""
        from src.models.asr import Transcription, TranscriptSegment as ASRSegment

        file = _make_audio_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        compliance_text = (
            "We need to review our data retention policy. "
            "The GDPR audit flagged missing retention periods."
        )
        mock_transcription = Transcription(
            full_text=compliance_text,
            segments=[ASRSegment(speaker="S0", start=0.0, end=10.0, text=compliance_text)],
            language="en",
            duration_seconds=10.0,
            word_count=len(compliance_text.split()),
        )
        expected_summary = "GDPR data retention review discussed."

        with (
            patch("src.agents.audio_agent.get_storage", return_value=mock_storage(b"fake")),
            patch("src.agents.audio_agent._asr.transcribe",
                  new_callable=AsyncMock, return_value=mock_transcription),
            patch("src.agents.audio_agent._summarizer.summarize",
                  new_callable=AsyncMock, return_value=expected_summary),
        ):
            from src.agents.audio_agent import audio_agent_node
            result = await audio_agent_node(state)

        extraction = result["raw_extractions"][0]
        assert extraction.summary == expected_summary

    @pytest.mark.asyncio
    async def test_duration_ms_recorded(self):
        """Audio agent should record processing duration."""
        from src.models.asr import Transcription

        file = _make_audio_file()
        state = make_state(input_files=[file])
        state["_current_file"] = file

        mock_transcription = Transcription(
            full_text="Test transcription.",
            segments=[],
            language="en",
            duration_seconds=5.0,
            word_count=2,
        )

        with (
            patch("src.agents.audio_agent.get_storage", return_value=mock_storage(b"fake")),
            patch("src.agents.audio_agent._asr.transcribe",
                  new_callable=AsyncMock, return_value=mock_transcription),
            patch("src.agents.audio_agent._summarizer.summarize",
                  new_callable=AsyncMock, return_value="Summary."),
        ):
            from src.agents.audio_agent import audio_agent_node
            result = await audio_agent_node(state)

        if result.get("raw_extractions"):
            extraction = result["raw_extractions"][0]
            assert extraction.duration_ms is not None
            assert extraction.duration_ms >= 0
