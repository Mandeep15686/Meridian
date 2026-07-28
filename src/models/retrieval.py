"""Sentence similarity (local) and cross-encoder reranker (HF API) wrappers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.base import BaseHFModel
from src.models.registry import MODELS

logger = logging.getLogger(__name__)


# ── Sentence similarity (loaded locally — used by hallucination gate) ─────────


class SimilarityModel:
    """
    Wraps sentence-transformers/all-MiniLM-L6-v2 loaded in-process.

    Loaded once per worker process for sub-10ms inference on CPU.
    Used exclusively by the hallucination gate for claim ↔ source scoring.
    """

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading SentenceTransformer: %s", MODELS.similarity)
            self._model = SentenceTransformer(MODELS.similarity)
        return self._model

    def score(self, claim: str, source: str | None) -> float:
        """
        Return cosine similarity between a synthesized claim and its cited source.
        Returns 0.0 when source is None (ungounded claim).
        """
        if not source or not claim:
            return 0.0
        model = self._get_model()
        embeddings = model.encode([claim, source], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))

    def batch_score(self, pairs: list[tuple[str, str | None]]) -> list[float]:
        """
        Score multiple (claim, source) pairs efficiently with a single encode call.
        """
        if not pairs:
            return []

        valid_indices: list[int] = []
        texts: list[str] = []
        for i, (claim, source) in enumerate(pairs):
            if claim and source:
                valid_indices.append(i)
                texts.extend([claim, source])

        scores = [0.0] * len(pairs)
        if not texts:
            return scores

        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64)

        for j, idx in enumerate(valid_indices):
            e1 = embeddings[j * 2]
            e2 = embeddings[j * 2 + 1]
            scores[idx] = float(max(0.0, min(1.0, np.dot(e1, e2))))

        return scores


# ── Cross-encoder reranker (HF Inference API) ────────────────────────────────


@dataclass
class RankedChunk:
    chunk_id: str
    content: str
    original_rank: int
    rerank_score: float


class CrossEncoderReranker(BaseHFModel):
    """
    Re-ranks candidate chunks using a cross-encoder model.
    Calls the HF text-classification endpoint which accepts a list of
    (query, passage) sentence pairs.
    """

    def __init__(self) -> None:
        super().__init__(MODELS.reranker)

    def _parse_response(self, raw: Any) -> list[dict[str, Any]]:
        """Raw response is a list of [{label, score}] per input pair."""
        if isinstance(raw, list):
            return raw
        return []

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """
        Re-rank a list of candidate chunks against a query.

        Args:
            query: The retrieval query string.
            candidates: List of dicts with keys: chunk_id, content, rank (from RRF).
            top_k: Number of top chunks to return.

        Returns:
            Re-ranked list of RankedChunk, sorted by rerank_score descending.
        """
        if not candidates:
            return []

        # Build sentence pairs: [[query, passage], ...]
        sentence_pairs = [[query, c["content"][:512]] for c in candidates]

        payload = {"inputs": sentence_pairs}

        try:
            raw_scores = await self.predict(payload, use_cache=True)
        except Exception as exc:
            logger.warning("Reranker failed, falling back to RRF order: %s", exc)
            return [
                RankedChunk(
                    chunk_id=c["chunk_id"],
                    content=c["content"],
                    original_rank=i,
                    rerank_score=c.get("rrf_score", 0.0),
                )
                for i, c in enumerate(candidates[:top_k])
            ]

        # Extract scores from response — cross-encoder returns [{label, score}]
        def _extract_score(item: Any) -> float:
            if isinstance(item, list):
                # Find the score for the positive/relevant label
                for entry in item:
                    if isinstance(entry, dict) and entry.get("label") in (
                        "entailment",
                        "LABEL_1",
                        "1",
                    ):
                        return float(entry.get("score", 0.0))
                return max(float(e.get("score", 0.0)) for e in item if isinstance(e, dict))
            if isinstance(item, dict):
                return float(item.get("score", 0.0))
            return 0.0

        scored: list[RankedChunk] = []
        for i, (candidate, score_item) in enumerate(zip(candidates, raw_scores, strict=False)):
            scored.append(
                RankedChunk(
                    chunk_id=candidate["chunk_id"],
                    content=candidate["content"],
                    original_rank=i,
                    rerank_score=_extract_score(score_item),
                )
            )

        scored.sort(key=lambda c: -c.rerank_score)
        return scored[:top_k]
