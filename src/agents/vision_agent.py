"""Vision agent LangGraph node — image captioning + VQA + compliance analysis."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.graph.state import AgentExtraction, MeridianState, UploadedFile
from src.models.vision import ImageCaptioner, VQAModel
from src.storage.base import get_storage

logger = logging.getLogger(__name__)

_captioner = ImageCaptioner()
_vqa = VQAModel()


async def vision_agent_node(state: MeridianState) -> dict:
    """
    LangGraph node: analyze an image or screenshot for compliance signals.

    Pipeline:
    1. Download image from storage
    2. Generate compliance-focused caption with BLIP-2
    3. Run compliance VQA question battery with ViLT
    4. (Claude vision for complex reasoning — optional, controlled by options)
    5. Return AgentExtraction with caption and VQA results
    """
    t_start = time.monotonic()
    file: UploadedFile = state["_current_file"]
    job_id = state.get("job_id", "unknown")

    logger.info("[vision_agent] Processing file: %s (job=%s)", file.filename, job_id)

    try:
        storage = get_storage()
        image_bytes = await storage.download(file.storage_key)

        # ── Caption ────────────────────────────────────────────────────────────
        caption_result = await _captioner.caption_compliance(image_bytes)
        logger.debug("[vision_agent] Caption: %s", caption_result.caption[:100])

        # ── VQA ────────────────────────────────────────────────────────────────
        vqa_answers = await _vqa.ask_compliance_questions(image_bytes)
        vqa_dicts = [
            {"question": a.question, "answer": a.answer, "score": a.score} for a in vqa_answers
        ]

        # Build a summary from caption + high-confidence VQA answers
        high_conf = [f"{a.question}: {a.answer}" for a in vqa_answers if a.score > 0.3]
        summary_parts = [caption_result.caption]
        if high_conf:
            summary_parts.append("Key findings: " + "; ".join(high_conf))
        summary = " ".join(summary_parts)

        duration_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "[vision_agent] %s complete: %d VQA answers, %dms",
            file.filename,
            len(vqa_answers),
            duration_ms,
        )

        extraction = AgentExtraction(
            agent="vision_agent",
            file_id=file.file_id,
            raw_text=caption_result.caption,
            summary=summary,
            image_caption=caption_result.caption,
            vqa_results=vqa_dicts,
            duration_ms=duration_ms,
        )

        return {"raw_extractions": [extraction]}

    except Exception as exc:
        logger.exception("[vision_agent] Failed for file %s: %s", file.filename, exc)
        return {
            "raw_extractions": [],
            "error": str(exc),
            "error_stage": "vision_agent",
        }
