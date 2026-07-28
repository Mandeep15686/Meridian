"""Hallucination gate node — verifies every claim against its cited source chunk."""

from __future__ import annotations

import logging

from src.config import settings
from src.graph.state import CandidateGap, MeridianState, RetrievedChunk, VerifiedGap
from src.models.similarity import SimilarityModel

logger = logging.getLogger(__name__)

_similarity_model = SimilarityModel()


def _find_chunk_content(chunk_id: str | None, retrieved_chunks: list[RetrievedChunk]) -> str | None:
    """Return the content of a chunk by ID, or None if not found."""
    if not chunk_id:
        return None
    for chunk in retrieved_chunks:
        if chunk.chunk_id == chunk_id:
            return chunk.content
    return None


def hallucination_gate_node(state: MeridianState) -> dict:
    """
    LangGraph node: check every candidate gap's cited quote against the
    retrieved chunk it references.

    Decision logic:
    - If groundedness_score >= GROUNDEDNESS_THRESHOLD → gap is verified.
    - If score < threshold and retries < MAX_SYNTHESIS_RETRIES → signal retry.
    - If score < threshold and retries == MAX_SYNTHESIS_RETRIES → mark uncertain.
    """
    candidate_gaps: list[CandidateGap] = state.get("candidate_gaps", [])
    retrieved_chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])
    synthesis_retries: int = state.get("synthesis_retries", 0)
    threshold: float = settings.GROUNDEDNESS_THRESHOLD

    if not candidate_gaps:
        logger.info("No candidate gaps to verify — skipping gate")
        return {"verified_gaps": [], "groundedness_scores": {}, "failed_gap_ids": []}

    groundedness_scores: dict[str, float] = {}
    failed_gap_ids: list[str] = []
    verified_gaps: list[VerifiedGap] = []

    # Batch encode all (claim, source) pairs for efficiency
    pairs: list[tuple[str, str | None]] = []
    for gap in candidate_gaps:
        source_content = _find_chunk_content(gap.regulatory_chunk_id, retrieved_chunks)
        pairs.append((gap.regulatory_quote, source_content))

    # Compute similarity scores (handles None source gracefully → score 0.0)
    scores = _similarity_model.batch_score(pairs)

    for gap, score in zip(candidate_gaps, scores, strict=True):
        groundedness_scores[gap.gap_id] = score
        logger.debug(
            "Gap %s groundedness score: %.3f (threshold %.2f)",
            gap.gap_id,
            score,
            threshold,
        )

        if score < threshold:
            failed_gap_ids.append(gap.gap_id)

    if failed_gap_ids and synthesis_retries < settings.MAX_SYNTHESIS_RETRIES:
        logger.info(
            "Gate: %d gap(s) failed groundedness (%.2f). Triggering retry %d/%d.",
            len(failed_gap_ids),
            threshold,
            synthesis_retries + 1,
            settings.MAX_SYNTHESIS_RETRIES,
        )
        return {
            "groundedness_scores": groundedness_scores,
            "failed_gap_ids": failed_gap_ids,
            "synthesis_retries": synthesis_retries + 1,
            "verified_gaps": [],
        }

    # Either all passed, or retries are exhausted — produce verified gaps
    for gap in candidate_gaps:
        score = groundedness_scores.get(gap.gap_id, 0.0)
        is_uncertain = gap.gap_id in failed_gap_ids  # still failed after max retries

        if is_uncertain:
            logger.warning(
                "Gap %s marked uncertain after %d retries (score %.3f < %.2f)",
                gap.gap_id,
                synthesis_retries,
                score,
                threshold,
            )

        verified_gaps.append(
            VerifiedGap(
                gap_id=gap.gap_id,
                severity=gap.severity,
                framework=gap.framework,
                regulatory_article=gap.regulatory_article,
                regulatory_requirement=gap.regulatory_requirement,
                regulatory_quote=gap.regulatory_quote,
                regulatory_chunk_id=gap.regulatory_chunk_id,
                policy_reference=gap.policy_reference,
                policy_text=gap.policy_text,
                gap_description=gap.gap_description,
                severity_justification=gap.severity_justification,
                remediation=gap.remediation,
                confidence=gap.confidence,
                groundedness_score=score,
                is_verified=not is_uncertain,
                is_uncertain=is_uncertain,
            )
        )

    # Sort: critical first, then by confidence descending
    _severity_order = {"critical": 0, "major": 1, "minor": 2}
    verified_gaps.sort(key=lambda g: (_severity_order.get(g.severity, 3), -g.confidence))

    pass_rate = (
        sum(1 for g in verified_gaps if not g.is_uncertain) / len(verified_gaps)
        if verified_gaps
        else 1.0
    )

    logger.info(
        "Gate complete: %d gaps verified, %d uncertain. Groundedness pass rate: %.1f%%",
        sum(1 for g in verified_gaps if not g.is_uncertain),
        sum(1 for g in verified_gaps if g.is_uncertain),
        pass_rate * 100,
    )

    return {
        "verified_gaps": verified_gaps,
        "groundedness_scores": groundedness_scores,
        "failed_gap_ids": [],
        "synthesis_retries": synthesis_retries,
    }


def gate_routing(state: MeridianState) -> str:
    """
    Conditional edge function: route to 'synthesize' for retry or 'report' to proceed.
    """
    failed = state.get("failed_gap_ids", [])
    retries = state.get("synthesis_retries", 0)

    if failed and retries <= settings.MAX_SYNTHESIS_RETRIES:
        return "synthesize"
    return "report"
