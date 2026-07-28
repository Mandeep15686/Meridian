"""Synthesis agent and report generation LangGraph nodes."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from src.config import settings
from src.graph.state import (
    AgentExtraction,
    CandidateGap,
    ComplianceReport,
    Entity,
    MeridianState,
    RetrievedChunk,
    VerifiedGap,
)
from src.models.llm import ClaudeClient
from src.models.registry import MODELS

logger = logging.getLogger(__name__)

_claude = ClaudeClient()


# ── Context formatters ────────────────────────────────────────────────────────


def _format_regulatory_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered context blocks for the LLM."""
    if not chunks:
        return "No regulatory context retrieved."

    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        article = f" — {chunk.article}" if chunk.article else ""
        lines.append(
            f"[{i}] {chunk.regulation.upper()}{article} "
            f"(jurisdiction: {chunk.jurisdiction}, "
            f"chunk_id: {chunk.chunk_id})\n"
            f"{chunk.content}\n"
        )
    return "\n".join(lines)


def _format_policy_content(extractions: list[AgentExtraction]) -> str:
    """Merge all agent extractions into a unified policy content block."""
    sections: list[str] = []

    for ext in extractions:
        agent_label = ext.agent.replace("_", " ").title()
        parts: list[str] = [f"=== {agent_label} (file_id: {ext.file_id}) ==="]

        if ext.raw_text:
            parts.append(f"Extracted text:\n{ext.raw_text[:5000]}")
        if ext.summary:
            parts.append(f"Summary: {ext.summary}")
        if ext.transcript:
            transcript_text = " ".join(s.text for s in ext.transcript[:20])
            parts.append(f"Transcript (excerpt): {transcript_text[:2000]}")
        if ext.image_caption:
            parts.append(f"Image caption: {ext.image_caption}")
        if ext.vqa_results:
            vqa_lines = [
                f"  Q: {r['question']} → A: {r['answer']} ({r.get('score', 0):.2f})"
                for r in ext.vqa_results[:8]
            ]
            parts.append("VQA findings:\n" + "\n".join(vqa_lines))
        if ext.tapas_answers:
            tapas_lines = [
                f"  Q: {r['question']} → A: {r['answer']}" for r in ext.tapas_answers[:6]
            ]
            parts.append("Table QA findings:\n" + "\n".join(tapas_lines))
        if ext.table_summary:
            parts.append(f"Table summary: {ext.table_summary}")

        sections.append("\n".join(parts))

    return "\n\n".join(sections) if sections else "No policy content extracted."


def _format_ner_summary(entities: list[Entity]) -> str:
    """Summarise NER entities as a structured list for the synthesis prompt."""
    if not entities:
        return "No regulatory entities extracted."

    by_type: dict[str, list[str]] = {}
    for ent in entities:
        by_type.setdefault(ent.type, []).append(ent.text)

    lines = []
    for ent_type, texts in by_type.items():
        unique_texts = list(dict.fromkeys(texts))[:5]
        lines.append(f"  {ent_type}: {', '.join(unique_texts)}")

    return "\n".join(lines)


# ── Synthesis node ────────────────────────────────────────────────────────────


async def synthesis_node(state: MeridianState) -> dict:
    """
    LangGraph node: generate candidate compliance gaps via Claude Sonnet.

    Receives all agent extractions and retrieved regulatory chunks,
    calls Claude with a structured synthesis prompt, and returns
    CandidateGap objects ready for hallucination gating.
    """
    t_start = time.monotonic()
    job_id = state.get("job_id", "unknown")
    regulation_scope = state.get("regulation_scope", [])
    extractions: list[AgentExtraction] = state.get("raw_extractions") or []
    retrieved_chunks: list[RetrievedChunk] = state.get("retrieved_chunks") or []
    ner_entities: list[Entity] = state.get("ner_entities") or []
    failed_gap_ids: list[str] = state.get("failed_gap_ids") or []
    synthesis_retries: int = state.get("synthesis_retries", 0)

    logger.info(
        "[synthesis] job=%s retries=%d extractions=%d chunks=%d",
        job_id,
        synthesis_retries,
        len(extractions),
        len(retrieved_chunks),
    )

    if not extractions and not retrieved_chunks:
        logger.warning("[synthesis] No content to synthesize for job %s", job_id)
        return {"candidate_gaps": [], "failed_gap_ids": []}

    # Format context blocks
    regulatory_context = _format_regulatory_context(retrieved_chunks[:10])
    policy_content = _format_policy_content(extractions)
    ner_summary = _format_ner_summary(ner_entities)

    try:
        gap_dicts = await _claude.synthesize(
            regulatory_context=regulatory_context,
            policy_content=policy_content,
            regulation_scope=regulation_scope,
            ner_summary=ner_summary,
            failed_gap_ids=failed_gap_ids if synthesis_retries > 0 else None,
        )
    except Exception as exc:
        logger.exception("[synthesis] Claude API call failed: %s", exc)
        return {
            "error": str(exc),
            "error_stage": "synthesis",
            "candidate_gaps": [],
        }

    # Parse raw gap dicts into typed CandidateGap objects
    candidate_gaps: list[CandidateGap] = []
    for raw_gap in gap_dicts:
        try:
            candidate_gaps.append(
                CandidateGap(
                    gap_id=raw_gap.get("gap_id") or f"gap_{uuid4().hex[:6]}",
                    severity=raw_gap.get("severity", "minor"),
                    framework=raw_gap.get(
                        "framework", regulation_scope[0] if regulation_scope else "unknown"
                    ),
                    regulatory_article=raw_gap.get("regulatory_article", "Unknown Article"),
                    regulatory_requirement=raw_gap.get("regulatory_requirement", ""),
                    regulatory_quote=raw_gap.get("regulatory_quote", ""),
                    regulatory_chunk_id=raw_gap.get("regulatory_chunk_id"),
                    policy_reference=raw_gap.get("policy_reference"),
                    policy_text=raw_gap.get("policy_text"),
                    gap_description=raw_gap.get("gap_description", ""),
                    severity_justification=raw_gap.get("severity_justification", ""),
                    remediation=raw_gap.get("remediation", ""),
                    confidence=float(raw_gap.get("confidence", 0.5)),
                )
            )
        except Exception as exc:
            logger.warning("[synthesis] Failed to parse gap dict: %s — %s", raw_gap, exc)

    duration_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "[synthesis] job=%s produced %d candidate gaps in %dms",
        job_id,
        len(candidate_gaps),
        duration_ms,
    )

    return {
        "candidate_gaps": candidate_gaps,
        "failed_gap_ids": [],
        "metadata": {
            **(state.get("metadata") or {}),
            "synthesis_duration_ms": duration_ms,
            "synthesis_model": MODELS.llm,
        },
    }


# ── Report node ───────────────────────────────────────────────────────────────


async def report_node(state: MeridianState) -> dict:
    """
    LangGraph node: assemble the final ComplianceReport from verified gaps.
    """
    t_start = time.monotonic()
    job_id = state.get("job_id", "unknown")
    verified_gaps: list[VerifiedGap] = state.get("verified_gaps", [])
    regulation_scope = state.get("regulation_scope", [])
    groundedness_scores: dict[str, float] = state.get("groundedness_scores", {})

    logger.info(
        "[report] job=%s assembling report for %d verified gaps",
        job_id,
        len(verified_gaps),
    )

    # Compute summary statistics
    total_gaps = len(verified_gaps)
    gaps_critical = sum(1 for g in verified_gaps if g.severity == "critical")
    gaps_major = sum(1 for g in verified_gaps if g.severity == "major")
    gaps_minor = sum(1 for g in verified_gaps if g.severity == "minor")

    pass_count = sum(1 for g in verified_gaps if not g.is_uncertain)
    groundedness_pass_rate = pass_count / total_gaps if total_gaps > 0 else 1.0

    # Per-framework compliance scores
    compliance_scores: dict = {}
    for scope in regulation_scope:
        scope_gaps = [g for g in verified_gaps if g.framework == scope]
        # Conservative estimate: assume 10 checks per framework
        checks_performed = max(10, len(scope_gaps) * 2)
        gap_count = len(scope_gaps)
        compliance_scores[scope] = {
            "score": round(1.0 - gap_count / checks_performed, 3),
            "gaps": gap_count,
            "checks_performed": checks_performed,
        }

    # Generate executive summary via Claude
    gap_dicts = [g.model_dump() for g in verified_gaps]
    try:
        executive_summary = await _claude.generate_executive_summary(gap_dicts, regulation_scope)
    except Exception as exc:
        logger.warning("[report] Executive summary generation failed: %s", exc)
        executive_summary = (
            f"Analysis complete. Found {total_gaps} compliance gap(s) "
            f"({gaps_critical} critical, {gaps_major} major, {gaps_minor} minor)."
        )

    report = ComplianceReport(
        job_id=job_id,
        executive_summary=executive_summary,
        gaps=verified_gaps,
        compliance_scores=compliance_scores,
        total_gaps=total_gaps,
        gaps_critical=gaps_critical,
        gaps_major=gaps_major,
        gaps_minor=gaps_minor,
        groundedness_pass_rate=groundedness_pass_rate,
        model_metadata={
            "synthesis_model": MODELS.llm,
            "reranker_model": MODELS.reranker,
            "embedding_model": MODELS.embedder,
            "ner_model": MODELS.ner,
            "asr_model": MODELS.asr,
            "pipeline_version": settings.VERSION,
        },
    )

    duration_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "[report] job=%s report assembled: %d gaps, pass_rate=%.1f%%, %dms",
        job_id,
        total_gaps,
        groundedness_pass_rate * 100,
        duration_ms,
    )

    return {
        "final_report": report,
        "metadata": {
            **(state.get("metadata") or {}),
            "report_duration_ms": duration_ms,
            "total_gaps": total_gaps,
            "groundedness_pass_rate": groundedness_pass_rate,
        },
    }
