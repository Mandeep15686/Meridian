"""Input classifier node — detects file modalities and emits LangGraph Send objects."""

from __future__ import annotations

import logging
import mimetypes
from typing import Literal

from langgraph.constants import Send

from src.graph.state import MeridianState, UploadedFile

logger = logging.getLogger(__name__)

# ── MIME type → modality mapping ──────────────────────────────────────────────

_MIME_TO_MODALITY: dict[str, str] = {
    # Documents
    "application/pdf": "document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
    "application/msword": "document",
    "text/plain": "document",
    "text/html": "document",
    "text/markdown": "document",
    # Audio
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/mp4": "audio",
    "audio/m4a": "audio",
    "audio/x-m4a": "audio",
    "audio/flac": "audio",
    "audio/ogg": "audio",
    "video/mp4": "audio",  # extract audio track
    "video/quicktime": "audio",
    # Images
    "image/png": "image",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/webp": "image",
    "image/gif": "image",
    # Tabular
    "text/csv": "tabular",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "tabular",
    "application/vnd.ms-excel": "tabular",
    "application/json": "tabular",
}

_EXTENSION_TO_MODALITY: dict[str, str] = {
    ".pdf": "document",
    ".docx": "document",
    ".doc": "document",
    ".txt": "document",
    ".md": "document",
    ".html": "document",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".mp4": "audio",
    ".mov": "audio",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".csv": "tabular",
    ".xlsx": "tabular",
    ".xls": "tabular",
}

# Map modality → agent node name
_MODALITY_TO_AGENT: dict[str, str] = {
    "document": "doc_agent",
    "audio": "audio_agent",
    "image": "vision_agent",
    "tabular": "data_agent",
}

Modality = Literal["document", "audio", "image", "tabular", "unknown"]


def _detect_modality(file: UploadedFile) -> Modality:
    """Determine the modality of a file from MIME type and extension."""
    # 1. Direct MIME type lookup
    if file.mime_type in _MIME_TO_MODALITY:
        return _MIME_TO_MODALITY[file.mime_type]  # type: ignore[return-value]

    # 2. Extension fallback
    import pathlib

    ext = pathlib.Path(file.filename).suffix.lower()
    if ext in _EXTENSION_TO_MODALITY:
        return _EXTENSION_TO_MODALITY[ext]  # type: ignore[return-value]

    # 3. Guess from filename via stdlib mimetypes
    guessed_mime, _ = mimetypes.guess_type(file.filename)
    if guessed_mime and guessed_mime in _MIME_TO_MODALITY:
        return _MIME_TO_MODALITY[guessed_mime]  # type: ignore[return-value]

    logger.warning(
        "Could not determine modality for file %s (%s)", file.filename, file.mime_type
    )
    return "unknown"


def classify_input_node(state: MeridianState) -> dict:
    """
    LangGraph node: classify each uploaded file and annotate with its modality.

    This node does NOT return Send objects — routing is handled by the
    conditional edge function ``route_to_agents`` below.
    """
    files = state.get("input_files", [])
    if not files:
        logger.error("No input files in state for job %s", state.get("job_id"))
        return {"error": "No input files provided", "error_stage": "classify_input"}

    updated_files: list[UploadedFile] = []
    for f in files:
        modality = _detect_modality(f)
        updated_file = UploadedFile(
            file_id=f.file_id,
            filename=f.filename,
            modality=modality,
            mime_type=f.mime_type,
            size_bytes=f.size_bytes,
            storage_key=f.storage_key,
            content_hash=f.content_hash,
            duration_seconds=f.duration_seconds,
            page_count=f.page_count,
        )
        updated_files.append(updated_file)
        logger.info("File %s classified as %s", f.filename, modality)

    return {
        "input_files": updated_files,
        "raw_extractions": [],
        "synthesis_retries": 0,
        "metadata": {
            **(state.get("metadata") or {}),
            "modalities_detected": list({f.modality for f in updated_files}),
        },
    }


def route_to_agents(state: MeridianState) -> list[Send]:
    """
    LangGraph conditional edge: fan out to specialist agents in parallel.

    Emits one Send per file (not per modality) so each file gets its own
    agent invocation and all results accumulate in raw_extractions via
    the list-concatenation reducer.
    """
    files = state.get("input_files", [])
    sends: list[Send] = []

    for f in files:
        agent_name = _MODALITY_TO_AGENT.get(f.modality)
        if agent_name is None:
            logger.warning(
                "No agent for modality %s, skipping file %s", f.modality, f.filename
            )
            continue
        sends.append(Send(agent_name, {**state, "_current_file": f}))

    if not sends:
        logger.warning("No agent routes produced — routing directly to synthesize")
        return [Send("synthesize", state)]

    return sends
