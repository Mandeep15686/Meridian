"""
Combined nightly evaluation runner.

Runs all three evaluation layers in sequence:
  1. RAGAS (RAG quality)
  2. Gap detection F1
  3. Agent judge (if LangSmith configured)

Exits with code 0 if all thresholds pass, 1 otherwise.
Used by .github/workflows/nightly_eval.yml.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from src.config import settings
from src.eval.thresholds import THRESHOLDS, check_thresholds

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()


async def run_all_evals(
    ragas_dataset: Path,
    gap_dataset: Path,
    lookback_hours: int,
    output_mlflow: bool,
    verbose: bool,
) -> dict[str, float]:
    """
    Run all evaluation layers and return aggregated metrics.

    Args:
        ragas_dataset: Path to the JSONL RAGAS golden QA dataset.
        gap_dataset: Path to the JSONL gap detection dataset.
        lookback_hours: Hours to look back for LangSmith traces.
        output_mlflow: Log all results to MLflow.
        verbose: Verbose output per evaluation.

    Returns:
        Dict of all metrics from all eval layers.
    """
    all_metrics: dict[str, float] = {}
    eval_errors: list[str] = []

    # ── 1. RAGAS evaluation ───────────────────────────────────────────────────
    console.print(Rule("[bold cyan]RAGAS evaluation[/bold cyan]"))

    if ragas_dataset.exists():
        try:
            from src.eval.run_ragas import run_ragas_eval

            ragas_metrics = await run_ragas_eval(
                dataset_path=ragas_dataset,
                regulation_scope=["gdpr"],
                output_mlflow=output_mlflow,
                verbose=verbose,
            )
            all_metrics.update(ragas_metrics)
            console.print("[green]✓ RAGAS evaluation complete[/green]")
        except Exception as exc:
            logger.exception("RAGAS evaluation failed: %s", exc)
            eval_errors.append(f"RAGAS: {exc}")
            console.print(f"[red]✗ RAGAS evaluation failed: {exc}[/red]")
    else:
        console.print(f"[yellow]RAGAS dataset not found at {ragas_dataset} — skipping[/yellow]")

    # ── 2. Gap detection F1 ───────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Gap detection F1[/bold cyan]"))

    if gap_dataset.exists():
        try:
            from src.eval.run_gap_detection import run_gap_detection_eval

            gap_summary = await run_gap_detection_eval(
                dataset_path=gap_dataset,
                confidence_threshold=0.5,
                output_mlflow=output_mlflow,
                verbose=verbose,
            )
            all_metrics.update(
                {
                    "gap_f1": gap_summary.f1,
                    "gap_precision": gap_summary.precision,
                    "gap_recall": gap_summary.recall,
                }
            )
            console.print("[green]✓ Gap detection evaluation complete[/green]")
        except Exception as exc:
            logger.exception("Gap detection evaluation failed: %s", exc)
            eval_errors.append(f"Gap detection: {exc}")
            console.print(f"[red]✗ Gap detection evaluation failed: {exc}[/red]")
    else:
        console.print(f"[yellow]Gap dataset not found at {gap_dataset} — skipping[/yellow]")

    # ── 3. Agent judge ────────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Agent judge[/bold cyan]"))

    if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
        try:
            from src.eval.run_agent_judge import run_agent_judge

            judge_summary = await run_agent_judge(
                lookback_hours=lookback_hours,
                output_mlflow=output_mlflow,
                verbose=verbose,
            )
            all_metrics.update(
                {
                    "agent_routing_accuracy": judge_summary.routing_accuracy,
                    "tool_use_quality_avg": judge_summary.tool_use_quality,
                    "citation_accuracy_avg": judge_summary.citation_accuracy,
                }
            )
            console.print("[green]✓ Agent judge evaluation complete[/green]")
        except Exception as exc:
            logger.exception("Agent judge evaluation failed: %s", exc)
            eval_errors.append(f"Agent judge: {exc}")
            console.print(f"[red]✗ Agent judge evaluation failed: {exc}[/red]")
    else:
        console.print(
            "[yellow]LangSmith not configured (LANGCHAIN_TRACING_V2=false or "
            "LANGCHAIN_API_KEY missing) — skipping agent judge[/yellow]"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(Rule("[bold]Evaluation summary[/bold]"))

    _print_final_summary(all_metrics, eval_errors)

    return all_metrics


def _print_final_summary(metrics: dict[str, float], errors: list[str]) -> None:
    """Print the final summary table with pass/fail status for each metric."""
    threshold_map = {
        "ragas_faithfulness": THRESHOLDS.ragas_faithfulness,
        "ragas_answer_relevancy": THRESHOLDS.ragas_answer_relevancy,
        "ragas_context_precision": THRESHOLDS.ragas_context_precision,
        "ragas_context_recall": THRESHOLDS.ragas_context_recall,
        "gap_f1": THRESHOLDS.gap_detection_f1,
        "agent_routing_accuracy": THRESHOLDS.agent_routing_accuracy,
    }

    table = Table(title="Nightly eval results", show_header=True)
    table.add_column("Metric", style="cyan", width=32)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Threshold", justify="right", width=10)
    table.add_column("Status", width=8)

    all_passed = True
    for metric_name, threshold in threshold_map.items():
        value = metrics.get(metric_name)
        if value is None:
            table.add_row(metric_name, "—", f"{threshold:.4f}", "[yellow]skipped[/yellow]")
            continue
        passed = value >= threshold
        if not passed:
            all_passed = False
        table.add_row(
            metric_name,
            f"{value:.4f}",
            f"{threshold:.4f}",
            "[green]✓[/green]" if passed else "[red]✗[/red]",
        )

    console.print(table)

    if errors:
        console.print(f"\n[red]Evaluation errors ({len(errors)}):[/red]")
        for err in errors:
            console.print(f"  • {err}")

    if all_passed and not errors:
        console.print("\n[green bold]All thresholds passed. ✓[/green bold]")
    else:
        console.print("\n[red bold]One or more thresholds failed. ✗[/red bold]")


@app.command()
def main(
    ragas_dataset: Path = typer.Option(
        Path("data/golden/gdpr_qa.jsonl"),
        "--ragas-dataset",
        help="Path to RAGAS golden QA dataset",
    ),
    gap_dataset: Path = typer.Option(
        Path("data/golden/gap_detection.jsonl"),
        "--gap-dataset",
        help="Path to gap detection dataset",
    ),
    lookback_hours: int = typer.Option(
        26,
        "--lookback-hours",
        help="Hours to look back for LangSmith traces (agent judge)",
    ),
    output_mlflow: bool = typer.Option(False, "--output-mlflow", help="Log to MLflow"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run all Meridian evaluations and check thresholds."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    t_start = time.monotonic()
    metrics = asyncio.run(
        run_all_evals(
            ragas_dataset=ragas_dataset,
            gap_dataset=gap_dataset,
            lookback_hours=lookback_hours,
            output_mlflow=output_mlflow,
            verbose=verbose,
        )
    )

    duration = time.monotonic() - t_start
    console.print(f"\nTotal eval time: {duration:.1f}s")

    check = check_thresholds(metrics)
    raise SystemExit(0 if check.passed else 1)


if __name__ == "__main__":
    app()
