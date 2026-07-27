"""LangGraph shared state for the Meridian multi-agent pipeline."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel


# ── Sub-types stored in state ─────────────────────────────────────────────────

@dataclass
class UploadedFile:
    """A file submitted as part of a job."""
    file_id: str
    filename: str
    modality: Literal["document", "audio", "image", "tabular", "unknown"]
    mime_type: str
    size_bytes: int
    storage_key: str
    content_hash: str
    duration_seconds: int | None = None
    page_count: int | None = None


@dataclass
class Entity:
    """A named entity extracted from text by the NER model."""
    type: str          # RETENTION_PERIOD, DPO_MENTION, LAWFUL_BASIS, etc.
    text: str
    start: int
    end: int
    confidence: float
    source_file_id: str | None = None


@dataclass
class QAResult:
    """An extractive QA answer from a policy document."""
    question: str
    answer: str
    score: float
    start: int
    end: int
    source_file_id: str | None = None


@dataclass
class TranscriptSegment:
    """A timestamped segment from audio transcription."""
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class AgentExtraction:
    """Output from a specialist agent node for one file."""
    agent: str                          # "doc_agent", "audio_agent", etc.
    file_id: str
    raw_text: str | None = None
    ner_entities: list[Entity] = field(default_factory=list)
    qa_results: list[QAResult] = field(default_factory=list)
    summary: str | None = None
    transcript: list[TranscriptSegment] | None = None
    speakers: list[str] | None = None
    image_caption: str | None = None
    vqa_results: list[dict[str, Any]] | None = None
    table_summary: str | None = None
    tapas_answers: list[dict[str, Any]] | None = None
    anomaly_scores: list[dict[str, Any]] | None = None
    forecast_output: list[dict[str, Any]] | None = None
    duration_ms: int | None = None


@dataclass
class RetrievedChunk:
    """A regulatory chunk returned by the RAG pipeline."""
    chunk_id: str
    regulation: str
    article: str | None
    content: str
    jurisdiction: str
    dense_score: float
    bm25_score: float
    rrf_score: float
    rerank_score: float
    final_rank: int


class CandidateGap(BaseModel):
    """A compliance gap identified by the synthesis agent (pre-verification)."""
    gap_id: str
    severity: Literal["critical", "major", "minor"]
    framework: str
    regulatory_article: str
    regulatory_requirement: str
    regulatory_quote: str
    regulatory_chunk_id: str | None = None
    policy_reference: str | None = None
    policy_text: str | None = None
    gap_description: str
    severity_justification: str
    remediation: str
    confidence: float

    class Config:
        frozen = True


class VerifiedGap(BaseModel):
    """A compliance gap that has passed the hallucination gate."""
    gap_id: str
    severity: Literal["critical", "major", "minor"]
    framework: str
    regulatory_article: str
    regulatory_requirement: str
    regulatory_quote: str
    regulatory_chunk_id: str | None = None
    policy_reference: str | None = None
    policy_text: str | None = None
    gap_description: str
    severity_justification: str
    remediation: str
    confidence: float
    groundedness_score: float
    is_verified: bool = True
    is_uncertain: bool = False

    class Config:
        frozen = True


class ComplianceReport(BaseModel):
    """Final synthesized compliance report."""
    job_id: str
    executive_summary: str
    gaps: list[VerifiedGap]
    compliance_scores: dict[str, dict[str, Any]]  # {framework: {score, gaps, checks}}
    total_gaps: int
    gaps_critical: int
    gaps_major: int
    gaps_minor: int
    groundedness_pass_rate: float
    model_metadata: dict[str, str]


# ── Main state TypedDict ──────────────────────────────────────────────────────

from typing import TypedDict  # noqa: E402 — after dataclasses to avoid circular issues


class MeridianState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.

    Fields annotated with ``operator.add`` use list concatenation as the
    reducer — safe for parallel writes from multiple agent nodes running
    via ``Send``.
    """

    # ── Job context ───────────────────────────────────────────────────────────
    job_id: str
    input_files: list[UploadedFile]
    regulation_scope: list[str]
    options: dict[str, Any]

    # ── Agent outputs (list-reducer — safe for parallel Send fan-out) ─────────
    raw_extractions: Annotated[list[AgentExtraction], operator.add]

    # ── RAG outputs ───────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    ner_entities: list[Entity]

    # ── Synthesis ─────────────────────────────────────────────────────────────
    candidate_gaps: list[CandidateGap]
    groundedness_scores: dict[str, float]   # {gap_id: score}
    failed_gap_ids: list[str]               # gaps that failed the gate
    verified_gaps: list[VerifiedGap]
    synthesis_retries: int

    # ── Output ────────────────────────────────────────────────────────────────
    final_report: ComplianceReport | None
    error: str | None
    error_stage: str | None
    metadata: dict[str, Any]
