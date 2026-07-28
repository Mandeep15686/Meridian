"""Audio agent LangGraph node — ASR transcription + compliance statement extraction."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from src.graph.state import (
    AgentExtraction,
    MeridianState,
    TranscriptSegment,
    UploadedFile,
)
from src.models.asr import ASRModel, Transcription
from src.models.nlp import Summarizer
from src.storage.base import get_storage

logger = logging.getLogger(__name__)

_asr = ASRModel()
_summarizer = Summarizer()


async def audio_agent_node(state: MeridianState) -> dict:
    """
    LangGraph node: transcribe an audio file and extract compliance signals.

    Pipeline:
    1. Download audio file from storage
    2. Normalise and chunk audio with VAD silence detection
    3. Transcribe each chunk with Whisper large-v3
    4. Assemble full transcript with timestamps
    5. Summarize compliance-relevant statements
    6. Return AgentExtraction with transcript and summary
    """
    t_start = time.monotonic()
    file: UploadedFile = state["_current_file"]
    job_id = state.get("job_id", "unknown")

    logger.info("[audio_agent] Processing file: %s (job=%s)", file.filename, job_id)

    try:
        # ── 1. Download file ───────────────────────────────────────────────────
        storage = get_storage()
        file_bytes = await storage.download(file.storage_key)

        suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            # ── 2 & 3. Transcribe via Whisper ──────────────────────────────────
            transcription: Transcription = await _asr.transcribe(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not transcription.full_text.strip():
            logger.warning("[audio_agent] Empty transcription from %s", file.filename)
            return {"raw_extractions": []}

        logger.info(
            "[audio_agent] Transcribed %s: %.0f words, %.0fs duration",
            file.filename,
            transcription.word_count,
            transcription.duration_seconds,
        )

        # ── 4. Build structured segments ──────────────────────────────────────
        state_segments: list[TranscriptSegment] = [
            TranscriptSegment(
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            for seg in transcription.segments
        ]

        # ── 5. Summarize compliance-relevant statements ────────────────────────
        compliance_text = "\n".join(transcription.compliance_statements)
        summary: str | None = None

        if compliance_text.strip():
            try:
                summary = await _summarizer.summarize(
                    compliance_text,
                    max_length=256,
                    min_length=48,
                )
            except Exception as exc:
                logger.warning("[audio_agent] Summarization failed: %s", exc)
                # Fall back to joining top statements
                summary = " ".join(transcription.compliance_statements[:5])
        else:
            # Use general summary if no compliance keywords found
            try:
                summary = await _summarizer.summarize(
                    transcription.full_text[:3000],
                    max_length=192,
                    min_length=32,
                )
            except Exception:
                summary = transcription.full_text[:500]

        # ── 6. Extract unique speakers ─────────────────────────────────────────
        speakers = list({seg.speaker for seg in state_segments if seg.speaker})

        duration_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "[audio_agent] %s complete: %d segments, %d speakers, %dms",
            file.filename,
            len(state_segments),
            len(speakers),
            duration_ms,
        )

        extraction = AgentExtraction(
            agent="audio_agent",
            file_id=file.file_id,
            raw_text=transcription.full_text[:10_000],
            summary=summary,
            transcript=state_segments,
            speakers=speakers,
            duration_ms=duration_ms,
        )

        return {"raw_extractions": [extraction]}

    except Exception as exc:
        logger.exception("[audio_agent] Failed for file %s: %s", file.filename, exc)
        return {
            "raw_extractions": [],
            "error": str(exc),
            "error_stage": "audio_agent",
        }
