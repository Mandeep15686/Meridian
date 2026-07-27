"""Unit tests for the input classifier and LangGraph routing logic."""

from __future__ import annotations

import pytest

from src.graph.router import _detect_modality, classify_input_node, route_to_agents
from src.graph.state import UploadedFile
from tests.conftest import make_state, make_uploaded_file


def _make_file(filename: str, mime_type: str, modality: str = "unknown") -> UploadedFile:
    return UploadedFile(
        file_id=f"file-{filename}",
        filename=filename,
        modality=modality,
        mime_type=mime_type,
        size_bytes=1000,
        storage_key=f"uploads/test/{filename}",
        content_hash="abc123",
    )


class TestModalityDetection:

    def test_pdf_detected_as_document(self):
        f = _make_file("policy.pdf", "application/pdf")
        assert _detect_modality(f) == "document"

    def test_docx_detected_as_document(self):
        f = _make_file(
            "policy.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert _detect_modality(f) == "document"

    def test_txt_detected_as_document(self):
        f = _make_file("policy.txt", "text/plain")
        assert _detect_modality(f) == "document"

    def test_mp3_detected_as_audio(self):
        f = _make_file("meeting.mp3", "audio/mpeg")
        assert _detect_modality(f) == "audio"

    def test_wav_detected_as_audio(self):
        f = _make_file("call.wav", "audio/wav")
        assert _detect_modality(f) == "audio"

    def test_m4a_detected_as_audio(self):
        f = _make_file("recording.m4a", "audio/m4a")
        assert _detect_modality(f) == "audio"

    def test_mp4_detected_as_audio(self):
        """MP4 video files should be routed to audio agent for audio track extraction."""
        f = _make_file("meeting.mp4", "video/mp4")
        assert _detect_modality(f) == "audio"

    def test_png_detected_as_image(self):
        f = _make_file("screenshot.png", "image/png")
        assert _detect_modality(f) == "image"

    def test_jpg_detected_as_image(self):
        f = _make_file("dashboard.jpg", "image/jpeg")
        assert _detect_modality(f) == "image"

    def test_csv_detected_as_tabular(self):
        f = _make_file("audit_log.csv", "text/csv")
        assert _detect_modality(f) == "tabular"

    def test_xlsx_detected_as_tabular(self):
        f = _make_file(
            "data.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        assert _detect_modality(f) == "tabular"

    def test_extension_fallback_when_mime_unknown(self):
        """When MIME type is generic, extension should be used as fallback."""
        f = _make_file("report.pdf", "application/octet-stream")
        # Extension .pdf should trigger document modality
        assert _detect_modality(f) == "document"

    def test_csv_extension_fallback(self):
        f = _make_file("data.csv", "application/octet-stream")
        assert _detect_modality(f) == "tabular"

    def test_unknown_type_returns_unknown(self):
        f = _make_file("mystery.xyz", "application/x-unknown-format")
        result = _detect_modality(f)
        assert result == "unknown"


class TestClassifyInputNode:

    def test_single_file_classified(self):
        file = _make_file("policy.pdf", "application/pdf")
        state = make_state(input_files=[file])

        result = classify_input_node(state)

        assert "input_files" in result
        classified_files = result["input_files"]
        assert len(classified_files) == 1
        assert classified_files[0].modality == "document"

    def test_multiple_mixed_files_classified(self):
        files = [
            _make_file("policy.pdf", "application/pdf"),
            _make_file("meeting.mp3", "audio/mpeg"),
            _make_file("screenshot.png", "image/png"),
            _make_file("data.csv", "text/csv"),
        ]
        state = make_state(input_files=files)
        result = classify_input_node(state)

        classified = result["input_files"]
        modalities = {f.modality for f in classified}
        assert "document" in modalities
        assert "audio" in modalities
        assert "image" in modalities
        assert "tabular" in modalities

    def test_empty_files_returns_error(self):
        state = make_state(input_files=[])
        result = classify_input_node(state)

        assert result.get("error") is not None

    def test_metadata_updated_with_detected_modalities(self):
        files = [
            _make_file("policy.pdf", "application/pdf"),
            _make_file("audio.mp3", "audio/mpeg"),
        ]
        state = make_state(input_files=files)
        result = classify_input_node(state)

        metadata = result.get("metadata", {})
        detected = metadata.get("modalities_detected", [])
        assert "document" in detected
        assert "audio" in detected

    def test_synthesis_retries_initialised_to_zero(self):
        state = make_state(input_files=[_make_file("test.pdf", "application/pdf")])
        result = classify_input_node(state)
        assert result.get("synthesis_retries") == 0


class TestRouteToAgents:

    def test_document_file_routes_to_doc_agent(self):
        from langgraph.constants import Send

        file = _make_file("policy.pdf", "application/pdf", modality="document")
        state = make_state(input_files=[file])

        sends = route_to_agents(state)

        assert len(sends) == 1
        assert isinstance(sends[0], Send)
        assert sends[0].node == "doc_agent"

    def test_audio_file_routes_to_audio_agent(self):
        from langgraph.constants import Send

        file = _make_file("meeting.mp3", "audio/mpeg", modality="audio")
        state = make_state(input_files=[file])

        sends = route_to_agents(state)

        assert len(sends) == 1
        assert sends[0].node == "audio_agent"

    def test_image_file_routes_to_vision_agent(self):
        from langgraph.constants import Send

        file = _make_file("screenshot.png", "image/png", modality="image")
        state = make_state(input_files=[file])

        sends = route_to_agents(state)

        assert len(sends) == 1
        assert sends[0].node == "vision_agent"

    def test_tabular_file_routes_to_data_agent(self):
        from langgraph.constants import Send

        file = _make_file("data.csv", "text/csv", modality="tabular")
        state = make_state(input_files=[file])

        sends = route_to_agents(state)

        assert len(sends) == 1
        assert sends[0].node == "data_agent"

    def test_mixed_files_produce_multiple_sends(self):
        from langgraph.constants import Send

        files = [
            _make_file("policy.pdf", "application/pdf", modality="document"),
            _make_file("meeting.mp3", "audio/mpeg", modality="audio"),
            _make_file("screenshot.png", "image/png", modality="image"),
        ]
        state = make_state(input_files=files)

        sends = route_to_agents(state)

        assert len(sends) == 3
        agent_nodes = {s.node for s in sends}
        assert "doc_agent" in agent_nodes
        assert "audio_agent" in agent_nodes
        assert "vision_agent" in agent_nodes

    def test_unknown_modality_skipped(self):
        from langgraph.constants import Send

        files = [
            _make_file("policy.pdf", "application/pdf", modality="document"),
            _make_file("mystery.xyz", "application/unknown", modality="unknown"),
        ]
        state = make_state(input_files=files)

        sends = route_to_agents(state)

        # Only the document should be routed — unknown is skipped
        assert len(sends) == 1
        assert sends[0].node == "doc_agent"

    def test_no_valid_files_routes_to_synthesize(self):
        from langgraph.constants import Send

        files = [_make_file("mystery.xyz", "application/unknown", modality="unknown")]
        state = make_state(input_files=files)

        sends = route_to_agents(state)

        # Falls back to synthesize directly
        assert len(sends) == 1
        assert sends[0].node == "synthesize"

    def test_each_file_gets_current_file_in_state(self):
        """Each Send should include _current_file in the state payload."""
        from langgraph.constants import Send

        files = [
            _make_file("policy1.pdf", "application/pdf", modality="document"),
            _make_file("policy2.pdf", "application/pdf", modality="document"),
        ]
        state = make_state(input_files=files)

        sends = route_to_agents(state)

        assert len(sends) == 2
        # Each Send contains _current_file pointing to its specific file
        file_ids_in_sends = [s.arg.get("_current_file").file_id for s in sends]
        assert "file-policy1.pdf" in file_ids_in_sends
        assert "file-policy2.pdf" in file_ids_in_sends
