"""
RAGAS evaluation harness.

Evaluates RAG pipeline quality against the golden QA dataset using
four metrics: faithfulness, answer_relevancy, context_precision, context_recall.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from src.config import settings
from src.eval.thresholds import THRESHOLDS, check_thresholds

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_golden_dataset(path: Path) -> list[dict[str, Any]]:
    """
    Load a JSONL golden QA dataset.

    Expected schema per line:
    {
        "question": str,
        "ground_truth": str,
        "contexts": [str, ...]   # optional — fetched via retrieval if absent
    }
    """
    examples: list[dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON on line %d: %s", i + 1, exc)
    return examples


# ── Retrieval for context augmentation ────────────────────────────────────────

async def _fetch_contexts_for_example(
    example: dict[str, Any],
    regulation_scope: list[str],
) -> list[str]:
    """If the example has no contexts, retrieve them live from the RAG pipeline."""
    if example.get("contexts"):
        return example["contexts"]

    from src.db.session import get_db_session
    from src.rag.retrieve import hybrid_retrieve

    async with get_db_session() as db:
        chunks = await hybrid_retrieve(db, example["question"], regulation_scope)

    return [c.content for c in chunks]


# ── RAGAS evaluation ─────────────────────────────────────────────────────────

async def run_ragas_eval(
    dataset_path: Path,
    regulation_scope: list[str],
    output_mlflow: bool = False,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Run the full RAGAS evaluation pipeline.

    Args:
        dataset_path: Path to the JSONL golden dataset.
        regulation_scope: Regulatory frameworks to retrieve from.
        output_mlflow: Whether to log results to MLflow.
        verbose: Print per-example results.

    Returns:
        Dict mapping metric name to float score.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "RAGAS and datasets are required for evaluation. "
            "Install with: pip install ragas datasets"
        )

    from src.models.llm import ClaudeClient
    claude = ClaudeClient()

    examples = load_golden_dataset(dataset_path)
    if not examples:
        raise ValueError(f"No examples loaded from {dataset_path}")

    console.print(f"[cyan]Running RAGAS evaluation on {len(examples)} examples...[/cyan]")
    t_start = time.monotonic()

    # Build RAGAS dataset
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for i, ex in enumerate(examples):
        if verbose or i % 20 == 0:
            console.print(f"  Processing example {i + 1}/{len(examples)}...")

        question = ex["question"]
        ground_truth = ex["ground_truth"]

        # Fetch contexts via RAG
        ctx = await _fetch_contexts_for_example(ex, regulation_scope)

        # Generate answer via Claude using retrieved context
        context_text = "\n\n".join(ctx[:3])
        try:
            answer_msg = await claude._async_client.messages.create(
                model=settings.SYNTHESIS_MODEL,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Context:\n{context_text}\n\n"
                        f"Question: {question}\n\n"
                        f"Answer concisely based only on the context provided."
                    ),
                }],
            )
            generated_answer = answer_msg.content[0].text.strip()
        except Exception as exc:
            logger.warning("Answer generation failed for example %d: %s", i + 1, exc)
            generated_answer = "Unable to generate answer."

        questions.append(question)
        answers.append(generated_answer)
        contexts.append(ctx)
        ground_truths.append(ground_truth)

    # Build HuggingFace Dataset for RAGAS
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run RAGAS
    console.print("[cyan]Computing RAGAS metrics...[/cyan]")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    df = result.to_pandas()
    metrics: dict[str, float] = {
        "ragas_faithfulness": float(df["faithfulness"].mean()),
        "ragas_answer_relevancy": float(df["answer_relevancy"].mean()),
        "ragas_context_precision": float(df["context_precision"].mean()),
        "ragas_context_recall": float(df["context_recall"].mean()),
    }

    duration = time.monotonic() - t_start
    metrics["ragas_eval_duration_s"] = round(duration, 1)

    # Print results table
    _print_results_table(metrics)

    # Check thresholds
    check_result = check_thresholds(metrics)
    if check_result.passed:
        console.print("[green]✓ All RAGAS thresholds passed[/green]")
    else:
        console.print(f"[red]✗ RAGAS threshold failures:[/red]\n{check_result.summary()}")

    # Log to MLflow
    if output_mlflow:
        _log_to_mlflow(metrics, dataset_path, regulation_scope)

    return metrics


def _print_results_table(metrics: dict[str, float]) -> None:
    table = Table(title="RAGAS Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Threshold", justify="right", width=10)
    table.add_column("Status", width=8)

    threshold_map = {
        "ragas_faithfulness": THRESHOLDS.ragas_faithfulness,
        "ragas_answer_relevancy": THRESHOLDS.ragas_answer_relevancy,
        "ragas_context_precision": THRESHOLDS.ragas_context_precision,
        "ragas_context_recall": THRESHOLDS.ragas_context_recall,
    }

    for metric, score in metrics.items():
        if metric not in threshold_map:
            continue
        threshold = threshold_map[metric]
        passed = score >= threshold
        table.add_row(
            metric,
            f"{score:.4f}",
            f"{threshold:.4f}",
            "[green]✓[/green]" if passed else "[red]✗[/red]",
        )

    console.print(table)


def _log_to_mlflow(
    metrics: dict[str, float],
    dataset_path: Path,
    regulation_scope: list[str],
) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="ragas_eval"):
            mlflow.log_params({
                "pipeline_version": settings.VERSION,
                "dataset": str(dataset_path),
                "regulation_scope": ",".join(regulation_scope),
                "retrieval_top_k": settings.RETRIEVAL_TOP_K_RERANK,
                "reranker_model": settings.RERANKER_MODEL,
                "synthesis_model": settings.SYNTHESIS_MODEL,
            })
            mlflow.log_metrics(metrics)
        console.print("[green]Results logged to MLflow[/green]")
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)


# ── CLI entry point ───────────────────────────────────────────────────────────

@app.command()
def main(
    dataset: Path = typer.Option(
        Path("data/golden/gdpr_qa.jsonl"),
        "--dataset", "-d",
        help="Path to the JSONL golden QA dataset",
    ),
    scope: list[str] = typer.Option(
        ["gdpr"],
        "--scope", "-s",
        help="Regulatory scope(s) for retrieval",
    ),
    output_mlflow: bool = typer.Option(False, "--output-mlflow", help="Log to MLflow"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the RAGAS evaluation against the golden dataset."""
    metrics = asyncio.run(
        run_ragas_eval(
            dataset_path=dataset,
            regulation_scope=scope,
            output_mlflow=output_mlflow,
            verbose=verbose,
        )
    )

    # Exit with non-zero code if thresholds fail (for CI)
    check = check_thresholds(metrics)
    raise SystemExit(0 if check.passed else 1)


if __name__ == "__main__":
    app()
