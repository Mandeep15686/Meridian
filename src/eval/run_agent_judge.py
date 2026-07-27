"""
LLM-as-judge evaluation of LangSmith agent traces.

Evaluates three dimensions per trace:
  1. Routing correctness (0–5): Did classify_input correctly route each file?
  2. Tool use quality (0–5): Did agents call the right models/tools?
  3. Citation accuracy (0–5): Do gap citations trace to retrieved chunks?
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from src.config import settings
from src.eval.thresholds import THRESHOLDS

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class TraceScore:
    run_id: str
    job_id: str
    routing_score: float         # 0–5
    tool_use_score: float        # 0–5
    citation_accuracy_score: float   # 0–5
    overall_score: float         # mean of three
    reasoning: str
    latency_ms: int = 0


@dataclass
class JudgeEvalSummary:
    traces_evaluated: int
    routing_accuracy: float       # mean routing / 5
    tool_use_quality: float       # mean tool_use / 5
    citation_accuracy: float      # mean citation / 5
    overall_quality: float        # mean overall / 5
    low_quality_runs: list[str]   # run IDs with overall < 3.0
    duration_seconds: float = 0.0


# ── LangSmith trace fetching ──────────────────────────────────────────────────

def _fetch_recent_traces(lookback_hours: int = 24) -> list[dict[str, Any]]:
    """
    Fetch recent LangSmith traces for the Meridian project.

    Returns a list of run dictionaries from the LangSmith API.
    """
    if not settings.LANGCHAIN_TRACING_V2 or not settings.LANGCHAIN_API_KEY:
        console.print("[yellow]LangSmith tracing not configured. Returning empty trace list.[/yellow]")
        return []

    try:
        from langsmith import Client
        client = Client(api_key=settings.LANGCHAIN_API_KEY)

        start_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        runs = list(client.list_runs(
            project_name=settings.LANGCHAIN_PROJECT,
            run_type="chain",
            start_time=start_time,
            filter=f'eq(name, "meridian_pipeline")',
            limit=100,
        ))

        return [
            {
                "id": str(r.id),
                "name": r.name,
                "inputs": r.inputs or {},
                "outputs": r.outputs or {},
                "child_runs": [
                    {
                        "name": cr.name,
                        "inputs": cr.inputs or {},
                        "outputs": cr.outputs or {},
                        "error": cr.error,
                        "latency_ms": int((cr.end_time - cr.start_time).total_seconds() * 1000)
                        if cr.end_time and cr.start_time else 0,
                    }
                    for cr in (r.child_runs or [])
                ],
                "metadata": r.extra or {},
            }
            for r in runs
        ]
    except Exception as exc:
        logger.warning("Failed to fetch LangSmith traces: %s", exc)
        return []


def _fetch_trace_by_job(job_id: str) -> dict[str, Any] | None:
    """Fetch a single LangSmith trace by Meridian job ID."""
    if not settings.LANGCHAIN_API_KEY:
        return None
    try:
        from langsmith import Client
        client = Client(api_key=settings.LANGCHAIN_API_KEY)
        runs = list(client.list_runs(
            project_name=settings.LANGCHAIN_PROJECT,
            filter=f'has(metadata, {{"job_id": "{job_id}"}})',
            limit=1,
        ))
        if not runs:
            return None
        r = runs[0]
        return {"id": str(r.id), "name": r.name, "inputs": r.inputs, "outputs": r.outputs,
                "child_runs": []}
    except Exception:
        return None


# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert AI systems evaluator reviewing execution traces
of a compliance analysis pipeline. Evaluate the trace on three dimensions and return
a JSON object. Be objective and precise.

DIMENSIONS:
1. routing_score (0–5): Did the input classifier correctly identify file modalities?
   5 = all files correctly routed; 3 = minor misrouting; 1 = major misrouting; 0 = complete failure.

2. tool_use_score (0–5): Did each agent call the appropriate models and tools in the right order?
   5 = optimal tool use; 3 = adequate but suboptimal; 1 = poor tool use; 0 = no tools used.

3. citation_accuracy_score (0–5): Do the compliance gap citations reference chunks that were
   actually retrieved? Do the regulatory_quote values appear in the retrieved context?
   5 = all citations grounded; 3 = some ungrounded; 1 = mostly ungrounded; 0 = no citations.

Return ONLY valid JSON:
{
  "routing_score": <0-5>,
  "tool_use_score": <0-5>,
  "citation_accuracy_score": <0-5>,
  "reasoning": "<one sentence per dimension, separated by | >"
}"""


JUDGE_USER_TEMPLATE = """TRACE SUMMARY:
Job ID: {job_id}
Files processed: {files}
Agents invoked: {agents}
Gaps produced: {gap_count}

ROUTING LOG:
{routing_log}

TOOL CALLS (per agent):
{tool_calls}

SAMPLE GAPS WITH CITATIONS:
{gap_sample}

Evaluate this trace and return JSON scores."""


# ── Judge evaluation ──────────────────────────────────────────────────────────

async def _judge_trace(trace: dict[str, Any]) -> TraceScore | None:
    """Ask Claude to evaluate a single LangSmith trace."""
    from src.models.llm import ClaudeClient
    from src.models.registry import MODELS

    claude = ClaudeClient()
    t_start = time.monotonic()

    try:
        # Extract meaningful info from trace
        child_runs = trace.get("child_runs", [])
        agents_invoked = [
            cr["name"] for cr in child_runs
            if any(a in cr["name"] for a in ("doc_agent", "audio_agent", "vision_agent", "data_agent"))
        ]

        routing_log = ""
        tool_calls_text = ""
        gap_sample_text = ""

        for cr in child_runs:
            if cr["name"] == "classify_input":
                routing_log = json.dumps(cr.get("outputs", {}), indent=2)[:500]
            if "agent" in cr["name"]:
                tool_calls_text += f"\n{cr['name']}: {json.dumps(cr.get('inputs', {}))[:300]}"

        outputs = trace.get("outputs", {})
        final_report = outputs.get("final_report", {})
        gaps = (final_report.get("gaps", []) if isinstance(final_report, dict) else [])[:3]
        for g in gaps:
            if isinstance(g, dict):
                gap_sample_text += (
                    f"\n  Article: {g.get('regulatory_article', 'N/A')}"
                    f"\n  Quote: {g.get('regulatory_quote', 'N/A')[:100]}"
                    f"\n  Confidence: {g.get('confidence', 0):.2f}\n"
                )

        user_content = JUDGE_USER_TEMPLATE.format(
            job_id=trace.get("metadata", {}).get("job_id", trace["id"]),
            files=trace.get("metadata", {}).get("modalities_detected", "unknown"),
            agents=", ".join(agents_invoked) or "none detected",
            gap_count=len(gaps),
            routing_log=routing_log or "No routing log available",
            tool_calls=tool_calls_text or "No tool calls logged",
            gap_sample=gap_sample_text or "No gaps produced",
        )

        message = await claude._async_client.messages.create(
            model=MODELS.llm,
            max_tokens=256,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = message.content[0].text.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rstrip("`").rstrip()

        scores_data = json.loads(raw)
        routing_score = float(scores_data.get("routing_score", 3.0))
        tool_use_score = float(scores_data.get("tool_use_score", 3.0))
        citation_score = float(scores_data.get("citation_accuracy_score", 3.0))
        overall = (routing_score + tool_use_score + citation_score) / 3.0

        latency_ms = int((time.monotonic() - t_start) * 1000)

        return TraceScore(
            run_id=trace["id"],
            job_id=trace.get("metadata", {}).get("job_id", trace["id"]),
            routing_score=routing_score,
            tool_use_score=tool_use_score,
            citation_accuracy_score=citation_score,
            overall_score=overall,
            reasoning=scores_data.get("reasoning", ""),
            latency_ms=latency_ms,
        )

    except Exception as exc:
        logger.warning("Judge evaluation failed for trace %s: %s", trace["id"], exc)
        return None


async def run_agent_judge(
    lookback_hours: int = 24,
    job_id: str | None = None,
    output_mlflow: bool = False,
    verbose: bool = False,
) -> JudgeEvalSummary:
    """
    Run LLM-as-judge evaluation on recent LangSmith traces.

    Args:
        lookback_hours: How far back to look for traces.
        job_id: Optional specific job ID to evaluate.
        output_mlflow: Log results to MLflow.
        verbose: Print per-trace scores.

    Returns:
        JudgeEvalSummary with aggregated quality metrics.
    """
    t_start = time.monotonic()

    if job_id:
        trace = _fetch_trace_by_job(job_id)
        traces = [trace] if trace else []
    else:
        traces = _fetch_recent_traces(lookback_hours)

    if not traces:
        console.print("[yellow]No traces found to evaluate.[/yellow]")
        return JudgeEvalSummary(
            traces_evaluated=0,
            routing_accuracy=0.0,
            tool_use_quality=0.0,
            citation_accuracy=0.0,
            overall_quality=0.0,
            low_quality_runs=[],
        )

    console.print(f"[cyan]Evaluating {len(traces)} agent traces...[/cyan]")

    # Evaluate with limited concurrency (LLM calls)
    semaphore = asyncio.Semaphore(2)

    async def judge_with_semaphore(t: dict) -> TraceScore | None:
        async with semaphore:
            return await _judge_trace(t)

    raw_scores = await asyncio.gather(*[judge_with_semaphore(t) for t in traces])
    valid_scores = [s for s in raw_scores if s is not None]

    if not valid_scores:
        return JudgeEvalSummary(
            traces_evaluated=0,
            routing_accuracy=0.0,
            tool_use_quality=0.0,
            citation_accuracy=0.0,
            overall_quality=0.0,
            low_quality_runs=[],
        )

    routing_acc = sum(s.routing_score for s in valid_scores) / (5.0 * len(valid_scores))
    tool_quality = sum(s.tool_use_score for s in valid_scores) / (5.0 * len(valid_scores))
    citation_acc = sum(s.citation_accuracy_score for s in valid_scores) / (5.0 * len(valid_scores))
    overall = sum(s.overall_score for s in valid_scores) / (5.0 * len(valid_scores))

    low_quality = [s.run_id for s in valid_scores if s.overall_score < 3.0]

    if verbose:
        for score in valid_scores:
            status = "[green]✓[/green]" if score.overall_score >= 3.0 else "[red]✗[/red]"
            console.print(
                f"  {status} Run {score.run_id[:8]}... "
                f"routing={score.routing_score:.1f} "
                f"tools={score.tool_use_score:.1f} "
                f"citations={score.citation_accuracy_score:.1f}"
            )

    summary = JudgeEvalSummary(
        traces_evaluated=len(valid_scores),
        routing_accuracy=routing_acc,
        tool_use_quality=tool_quality,
        citation_accuracy=citation_acc,
        overall_quality=overall,
        low_quality_runs=low_quality,
        duration_seconds=time.monotonic() - t_start,
    )

    _print_judge_results(summary)

    if output_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="agent_judge_eval"):
                mlflow.log_metrics({
                    "agent_routing_accuracy": routing_acc,
                    "tool_use_quality_avg": tool_quality,
                    "citation_accuracy_avg": citation_acc,
                    "agent_overall_quality": overall,
                    "traces_evaluated": len(valid_scores),
                    "low_quality_count": len(low_quality),
                })
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)

    return summary


def _print_judge_results(summary: JudgeEvalSummary) -> None:
    table = Table(title="Agent Judge Evaluation", show_header=True)
    table.add_column("Dimension", style="cyan", width=30)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Threshold", justify="right", width=10)
    table.add_column("Status", width=8)

    rows = [
        ("Traces evaluated", str(summary.traces_evaluated), "—"),
        ("Routing accuracy", f"{summary.routing_accuracy:.4f}", f"{THRESHOLDS.agent_routing_accuracy:.4f}"),
        ("Tool use quality", f"{summary.tool_use_quality:.4f}", "—"),
        ("Citation accuracy", f"{summary.citation_accuracy:.4f}", "—"),
        ("Overall quality", f"{summary.overall_quality:.4f}", "—"),
        ("Low-quality runs", str(len(summary.low_quality_runs)), "—"),
    ]

    for label, value, threshold in rows:
        if threshold == "—":
            table.add_row(label, value, threshold, "")
        else:
            t_val = float(threshold)
            v_val = float(value)
            passed = v_val >= t_val
            table.add_row(
                label, value, threshold,
                "[green]✓[/green]" if passed else "[red]✗[/red]"
            )

    console.print(table)


@app.command()
def main(
    lookback: int = typer.Option(24, "--lookback-hours", "-l"),
    job: str | None = typer.Option(None, "--job-id", "-j"),
    output_mlflow: bool = typer.Option(False, "--output-mlflow"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run LLM-as-judge evaluation on recent LangSmith traces."""
    asyncio.run(run_agent_judge(
        lookback_hours=lookback,
        job_id=job,
        output_mlflow=output_mlflow,
        verbose=verbose,
    ))


if __name__ == "__main__":
    app()
