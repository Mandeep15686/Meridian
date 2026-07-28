"""Vision model wrappers: BLIP-2 image captioning and ViLT visual QA."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.models.base import BaseHFModel
from src.models.registry import MODELS

logger = logging.getLogger(__name__)

# Compliance-specific VQA questions asked of every screenshot
COMPLIANCE_VQA_QUESTIONS = [
    "Is there a clearly labeled reject all option visible?",
    "Are any consent checkboxes pre-checked?",
    "Is a privacy policy link visible?",
    "Is a cookie banner or consent dialog present?",
    "What data retention period is shown?",
    "Are any warning or alert indicators visible?",
    "Is a data protection officer contact shown?",
]


def _image_to_base64(image_path: Path) -> str:
    """Read an image file and return base64-encoded bytes."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


# ── BLIP-2 Image Captioning ───────────────────────────────────────────────────


@dataclass
class CaptionResult:
    caption: str
    model: str = MODELS.captioner


class ImageCaptioner(BaseHFModel):
    """
    Image-to-text captioning using Salesforce/blip2-opt-2.7b.

    Accepts image bytes or file paths. Returns a free-form natural
    language description suitable for downstream RAG retrieval.
    """

    def __init__(self) -> None:
        super().__init__(MODELS.captioner)

    def _parse_response(self, raw: Any) -> str:
        if isinstance(raw, list) and raw:
            first = cast(dict[str, Any], raw[0])
            return cast(str, first.get("generated_text", "")).strip()
        if isinstance(raw, dict):
            return cast(str, raw.get("generated_text", "")).strip()
        return ""

    async def caption(self, image_source: Path | bytes, prompt: str | None = None) -> CaptionResult:
        """Generate a caption for an image."""
        if isinstance(image_source, Path):
            b64 = _image_to_base64(image_source)
        else:
            b64 = _image_bytes_to_base64(image_source)

        payload: dict[str, Any] = {
            "inputs": {"image": b64},
        }
        if prompt:
            payload["inputs"]["prompt"] = prompt

        caption_text = cast(str, await self.predict(payload))
        return CaptionResult(caption=caption_text)

    async def caption_compliance(self, image_source: Path | bytes) -> CaptionResult:
        """Caption an image with a compliance-focused prompt."""
        return await self.caption(
            image_source,
            prompt="Describe this compliance or privacy interface in detail, "
            "noting any consent options, data retention information, "
            "privacy notices, or regulatory elements.",
        )


# ── ViLT Visual Question Answering ────────────────────────────────────────────


@dataclass
class VQAAnswer:
    question: str
    answer: str
    score: float


class VQAModel(BaseHFModel):
    """
    Visual question answering using dandelin/vilt-b32-finetuned-vqa.

    Answers specific natural-language questions about an image.
    Used to extract structured compliance signals from screenshots.
    """

    def __init__(self) -> None:
        super().__init__(MODELS.vqa)

    def _parse_response(self, raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, list):
            return raw
        return []

    async def ask(self, image_source: Path | bytes, question: str) -> VQAAnswer:
        """Ask a single question about an image."""
        if isinstance(image_source, Path):
            b64 = _image_to_base64(image_source)
        else:
            b64 = _image_bytes_to_base64(image_source)

        payload = {
            "inputs": {
                "image": b64,
                "question": question,
            }
        }

        raw_results = await self.predict(payload)

        if not raw_results:
            return VQAAnswer(question=question, answer="", score=0.0)

        top = raw_results[0]
        return VQAAnswer(
            question=question,
            answer=top.get("answer", ""),
            score=float(top.get("score", 0.0)),
        )

    async def ask_compliance_questions(
        self,
        image_source: Path | bytes,
        questions: list[str] | None = None,
        min_score: float = 0.1,
    ) -> list[VQAAnswer]:
        """
        Ask a battery of compliance-specific questions about an image.

        Args:
            image_source: Image path or bytes.
            questions: Optional override; defaults to COMPLIANCE_VQA_QUESTIONS.
            min_score: Minimum confidence to include an answer.

        Returns:
            List of VQAAnswer instances for questions with meaningful answers.
        """
        questions = questions or COMPLIANCE_VQA_QUESTIONS
        answers: list[VQAAnswer] = []

        for question in questions:
            try:
                answer = await self.ask(image_source, question)
                if answer.answer and answer.score >= min_score:
                    answers.append(answer)
            except Exception as exc:
                logger.warning("VQA failed for question '%s': %s", question[:60], exc)

        return answers
