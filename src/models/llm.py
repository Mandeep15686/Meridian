"""Claude Sonnet wrapper for synthesis reasoning with Pydantic structured output."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models.registry import MODELS

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _extract_message_text(content: Any) -> str:
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


SYNTHESIS_SYSTEM_PROMPT = """You are a regulatory compliance expert. Your task is to analyze
a company's policy documents against specific regulatory requirements and identify compliance gaps.

STRICT RULES:
1. Only cite gaps supported by the retrieved regulatory text provided in context.
2. Every gap MUST include a specific regulatory article (e.g. "GDPR Article 13(2)(a)").
3. The regulatory_quote field MUST be text that appears in the provided context chunks.
4. Never invent regulatory citations from your training knowledge.
5. If the policy satisfies a requirement, do NOT create a gap for it.
6. Rate severity: critical (enforcement risk), major (clear violation), minor (best practice).
7. policy_text should be the exact policy excerpt that is deficient, or null if entirely absent.
8. remediation must be a specific, actionable instruction (not generic advice).

OUTPUT FORMAT: Return only valid JSON matching the ComplianceGapOutput schema. No markdown, no preamble."""

SYNTHESIS_USER_TEMPLATE = """REGULATORY CONTEXT (retrieved chunks):
{regulatory_context}

POLICY CONTENT (extracted from submitted documents):
{policy_content}

REGULATION SCOPE: {regulation_scope}

NER ENTITIES FOUND IN POLICY:
{ner_summary}

INSTRUCTIONS:
Analyze the policy against the regulatory context. Identify every compliance gap.
Return a JSON object with a single key "gaps" containing a list of gap objects.

Each gap object must have:
- gap_id: unique string (e.g. "gap_001")
- severity: "critical" | "major" | "minor"
- framework: one of {regulation_scope}
- regulatory_article: specific article/section (e.g. "GDPR Article 13(2)(a)")
- regulatory_requirement: one sentence describing what the regulation requires
- regulatory_quote: exact text from the regulatory context above (verbatim)
- policy_reference: section of the policy being evaluated (or null)
- policy_text: relevant policy excerpt (or null if the requirement is entirely absent)
- gap_description: clear plain-English description of the gap
- severity_justification: one sentence explaining the severity rating
- remediation: specific action the company must take
- confidence: float 0.0–1.0

Return empty "gaps": [] if no gaps are found.
"""


class ClaudeClient:
    """
    Anthropic Claude client for synthesis reasoning.

    Wraps the Anthropic Python SDK with:
    - Structured Pydantic output parsing
    - Retry logic for rate limits and overload
    - Token usage logging to MLflow
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._async_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    async def synthesize(
        self,
        regulatory_context: str,
        policy_content: str,
        regulation_scope: list[str],
        ner_summary: str,
        failed_gap_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run the synthesis agent to identify compliance gaps.

        Args:
            regulatory_context: Formatted retrieved regulatory chunks.
            policy_content: Merged text and extractions from all agents.
            regulation_scope: Active regulatory frameworks.
            ner_summary: Formatted summary of NER entities found.
            failed_gap_ids: Gap IDs that failed the groundedness gate (for retry context).

        Returns:
            List of gap dictionaries matching CandidateGap schema.
        """
        user_content = SYNTHESIS_USER_TEMPLATE.format(
            regulatory_context=regulatory_context[:60_000],
            policy_content=policy_content[:40_000],
            regulation_scope=", ".join(regulation_scope),
            ner_summary=ner_summary[:2000],
        )

        # Add retry context if this is a re-synthesis after gate failure
        if failed_gap_ids:
            user_content += (
                f"\n\nNOTE: The following gap IDs failed groundedness verification in a "
                f"previous attempt: {failed_gap_ids}. Re-examine these gaps and ensure "
                f"their regulatory_quote values are exact verbatim excerpts from the "
                f"regulatory context provided above."
            )

        message = await self._async_client.messages.create(
            model=MODELS.llm,
            max_tokens=settings.SYNTHESIS_MAX_TOKENS,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        raw_text = _extract_message_text(message.content)
        logger.info(
            "Synthesis complete. Input tokens: %d, Output tokens: %d",
            message.usage.input_tokens,
            message.usage.output_tokens,
        )

        self._log_token_usage(message.usage)
        return self._parse_gaps(raw_text)

    def _parse_gaps(self, raw_text: str) -> list[dict[str, Any]]:
        """Parse Claude's JSON output into a list of gap dicts."""
        # Strip markdown code fences if present
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3].rstrip()

        try:
            data = json.loads(text)
            gaps = data.get("gaps", [])
            if not isinstance(gaps, list):
                logger.error("Synthesis output 'gaps' is not a list: %s", type(gaps))
                return []
            return gaps
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse synthesis JSON: %s\nRaw: %s", exc, raw_text[:500])
            return []

    def _log_token_usage(self, usage: Any) -> None:
        """Log token counts to MLflow if tracking is enabled."""
        if not settings.LANGCHAIN_TRACING_V2:
            return
        try:
            import mlflow

            mlflow.log_metrics(
                {
                    "synthesis_input_tokens": usage.input_tokens,
                    "synthesis_output_tokens": usage.output_tokens,
                }
            )
        except Exception:
            pass  # Don't fail the pipeline for metrics logging

    async def generate_executive_summary(
        self,
        gaps: list[dict[str, Any]],
        regulation_scope: list[str],
    ) -> str:
        """Generate a 2–3 paragraph executive summary of the compliance findings."""
        if not gaps:
            return (
                "The analysis found no material compliance gaps against the "
                f"requested regulatory frameworks: {', '.join(regulation_scope).upper()}. "
                "The submitted policy documents appear to satisfy the reviewed requirements."
            )

        critical = sum(1 for g in gaps if g.get("severity") == "critical")
        major = sum(1 for g in gaps if g.get("severity") == "major")
        minor = sum(1 for g in gaps if g.get("severity") == "minor")

        frameworks = {g.get("framework", "unknown") for g in gaps}
        top_articles = [
            g.get("regulatory_article", "") for g in gaps if g.get("severity") == "critical"
        ][:3]

        prompt = (
            f"Write a concise 2–3 paragraph executive summary of these compliance findings. "
            f"Total gaps: {len(gaps)} ({critical} critical, {major} major, {minor} minor). "
            f"Affected frameworks: {', '.join(frameworks).upper()}. "
            f"Most critical issues: {', '.join(top_articles)}. "
            f"Write for a senior compliance officer. Be direct and specific. "
            f"Do not use bullet points. No markdown."
        )

        message = await self._async_client.messages.create(
            model=MODELS.llm,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        summary_text = _extract_message_text(message.content).strip()
        return summary_text if summary_text else "Summary unavailable."
