"""NER, QA, zero-shot classification, and summarisation HF model wrappers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from src.models.base import BaseHFModel
from src.models.registry import MODELS

logger = logging.getLogger(__name__)


# ── Named Entity Recognition ──────────────────────────────────────────────────


@dataclass
class NEREntity:
    entity_group: str
    word: str
    start: int
    end: int
    score: float


class NERModel(BaseHFModel):
    """
    Token classification using dslim/bert-base-NER.

    Returns grouped entities (not token-level).
    """

    def __init__(self) -> None:
        super().__init__(MODELS.ner)

    def _parse_response(self, raw: Any) -> list[NEREntity]:
        if not isinstance(raw, list):
            return []
        entities: list[NEREntity] = []
        for item in raw:
            entities.append(
                NEREntity(
                    entity_group=item.get("entity_group", item.get("entity", "O")),
                    word=item.get("word", ""),
                    start=item.get("start", 0),
                    end=item.get("end", 0),
                    score=float(item.get("score", 0.0)),
                )
            )
        return entities

    async def extract(self, text: str, min_score: float = 0.7) -> list[NEREntity]:
        """Extract entities from text, filtering by confidence."""
        # HF NER API uses aggregation_strategy to return grouped entities
        payload = {
            "inputs": text[:512],  # BERT-base context limit
            "parameters": {"aggregation_strategy": "simple"},
        }
        entities = cast(list[NEREntity], await self.predict(payload))
        return [e for e in entities if e.score >= min_score]


# ── Extractive QA ─────────────────────────────────────────────────────────────


@dataclass
class QAAnswer:
    answer: str
    score: float
    start: int
    end: int


class QAModel(BaseHFModel):
    """Extractive QA using deepset/roberta-base-squad2."""

    # SQuAD2 returns {"answer": "", "score": 0.0} for unanswerable questions
    NO_ANSWER_THRESHOLD = 0.1

    def __init__(self) -> None:
        super().__init__(MODELS.qa)

    def _parse_response(self, raw: Any) -> QAAnswer | None:
        if not isinstance(raw, dict):
            return None
        score = float(raw.get("score", 0.0))
        answer = raw.get("answer", "").strip()
        if score < self.NO_ANSWER_THRESHOLD or not answer:
            return None
        return QAAnswer(
            answer=answer,
            score=score,
            start=raw.get("start", 0),
            end=raw.get("end", 0),
        )

    async def answer(self, question: str, context: str) -> QAAnswer | None:
        """Answer a question given a context passage."""
        payload = {
            "inputs": {
                "question": question[:256],
                "context": context[:512],
            }
        }
        return cast(QAAnswer | None, await self.predict(payload))


# ── Zero-shot Classification ──────────────────────────────────────────────────


@dataclass
class ClassificationResult:
    labels: list[str]
    scores: list[float]

    @property
    def top_label(self) -> str:
        return self.labels[0] if self.labels else "unknown"

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    def label_score(self, label: str) -> float:
        try:
            return self.scores[self.labels.index(label)]
        except ValueError:
            return 0.0


class ZeroShotClassifier(BaseHFModel):
    """Zero-shot classification using facebook/bart-large-mnli."""

    def __init__(self) -> None:
        super().__init__(MODELS.classifier)

    def _parse_response(self, raw: Any) -> ClassificationResult:
        if not isinstance(raw, dict):
            return ClassificationResult(labels=[], scores=[])
        labels = cast(list[str], raw.get("labels", []))
        scores = cast(list[float], raw.get("scores", []))
        # Sort by score descending (API may not always guarantee this)
        sorted_pairs = sorted(zip(labels, scores, strict=True), key=lambda x: -x[1])
        return ClassificationResult(
            labels=[p[0] for p in sorted_pairs],
            scores=[p[1] for p in sorted_pairs],
        )

    async def classify(
        self,
        text: str,
        candidate_labels: list[str],
        multi_label: bool = False,
    ) -> ClassificationResult:
        """Classify text against candidate labels."""
        payload = {
            "inputs": text[:1024],
            "parameters": {
                "candidate_labels": candidate_labels,
                "multi_label": multi_label,
            },
        }
        return cast(ClassificationResult, await self.predict(payload))


# ── Regulatory-specific entity classifier ────────────────────────────────────

REGULATORY_ENTITY_LABELS = [
    "data retention period",
    "data subject category",
    "consent mechanism",
    "data protection officer mention",
    "lawful basis for processing",
    "third party data transfer",
    "security measure description",
    "unrelated general text",
]

LABEL_TO_TYPE = {
    "data retention period": "RETENTION_PERIOD",
    "data subject category": "DATA_SUBJECT_CATEGORY",
    "consent mechanism": "CONSENT_MECHANISM",
    "data protection officer mention": "DPO_MENTION",
    "lawful basis for processing": "LAWFUL_BASIS",
    "third party data transfer": "THIRD_PARTY_TRANSFER",
    "security measure description": "SECURITY_MEASURE",
}


class RegulatoryEntityClassifier(ZeroShotClassifier):
    """
    Classifies text spans as domain-specific regulatory entity types.
    Used as second-pass NER after the general bert-base-NER model.
    """

    async def classify_entity(
        self,
        span_text: str,
        min_score: float = 0.55,
    ) -> str | None:
        """
        Return the regulatory entity type for a text span, or None
        if the span doesn't match any regulatory category.
        """
        result = await self.classify(span_text, REGULATORY_ENTITY_LABELS)
        if result.top_label == "unrelated general text" or result.top_score < min_score:
            return None
        return LABEL_TO_TYPE.get(result.top_label)


# ── Summarisation ─────────────────────────────────────────────────────────────


class Summarizer(BaseHFModel):
    """
    Abstractive summarisation using facebook/bart-large-cnn.
    Chunk-and-summarise strategy for documents over 1024 tokens.
    """

    MAX_INPUT_TOKENS = 1024
    CHARS_PER_TOKEN = 4  # approximate

    def __init__(self) -> None:
        super().__init__(MODELS.summarizer)

    def _parse_response(self, raw: Any) -> str:
        if isinstance(raw, list) and raw:
            first = cast(dict[str, Any], raw[0])
            return cast(str, first.get("summary_text", ""))
        return ""

    async def summarize(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 64,
    ) -> str:
        """Summarise text, chunking if it exceeds the model's context limit."""
        max_chars = self.MAX_INPUT_TOKENS * self.CHARS_PER_TOKEN

        if len(text) <= max_chars:
            return await self._summarize_single(text, max_length, min_length)

        # Chunk-and-summarise
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
        chunk_summaries = []
        for chunk in chunks[:8]:  # cap at 8 chunks to bound latency
            summary = await self._summarize_single(
                chunk, max_length // 2, min_length // 2
            )
            if summary:
                chunk_summaries.append(summary)

        combined = " ".join(chunk_summaries)
        if len(combined) > max_chars:
            return await self._summarize_single(
                combined[:max_chars], max_length, min_length
            )
        return combined

    async def _summarize_single(
        self, text: str, max_length: int, min_length: int
    ) -> str:
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False,
            },
        }
        return cast(str, await self.predict(payload))
