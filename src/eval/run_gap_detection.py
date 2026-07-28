"""
Gap detection F1 evaluation harness.

Measures precision, recall, and F1 for compliance gap detection
against a hand-labeled golden dataset of 150 examples seeded from
real SEC enforcement actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
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
class GapExample:
    """A single labeled gap detection example."""

    example_id: str
    policy_text: str
    regulation_scope: list[str]
    ground_truth_gaps: list[dict[str, Any]]  # [{has_gap, regulatory_article, gap_type}]
    source: str = "manual"  # "sec_enforcement", "manual", "synthetic"


@dataclass
class PredictionResult:
    example_id: str
    predicted_gaps: list[dict[str, Any]]
    ground_truth_gaps: list[dict[str, Any]]
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    latency_ms: int = 0


@dataclass
class EvalSummary:
    precision: float
    recall: float
    f1: float
    threshold: float
    total_examples: int
    total_true_positives: int
    total_false_positives: int
    total_false_negatives: int
    per_framework: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: list[PredictionResult] = field(default_factory=list)
    duration_seconds: float = 0.0


# ── Dataset loading ───────────────────────────────────────────────────────────


def load_gap_dataset(path: Path) -> list[GapExample]:
    """Load a JSONL gap detection dataset."""
    examples: list[GapExample] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                examples.append(
                    GapExample(
                        example_id=data["example_id"],
                        policy_text=data["policy_text"],
                        regulation_scope=data["regulation_scope"],
                        ground_truth_gaps=data["ground_truth_gaps"],
                        source=data.get("source", "manual"),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping invalid example on line %d: %s", i + 1, exc)
    return examples


# ── Gap matching ──────────────────────────────────────────────────────────────


def _normalize_article(article: str) -> str:
    """Normalise article strings for comparison (case-insensitive, strip whitespace)."""
    return " ".join(article.lower().strip().split())


def _articles_match(predicted: str, ground_truth: str) -> bool:
    """
    Return True if the predicted regulatory article matches the ground truth.

    Handles partial matches — e.g. "GDPR Article 13" matches "GDPR Article 13(2)(a)".
    """
    p = _normalize_article(predicted)
    g = _normalize_article(ground_truth)
    return p == g or p.startswith(g) or g.startswith(p)


def _match_gaps(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    confidence_threshold: float,
) -> tuple[int, int, int]:
    """
    Match predicted gaps against ground truth.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    # Filter predictions by confidence threshold
    confident_preds = [
        g for g in predicted if float(g.get("confidence", 1.0)) >= confidence_threshold
    ]

    # Match each ground truth gap to at most one prediction
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for pi, pred in enumerate(confident_preds):
        pred_article = pred.get("regulatory_article", "")
        # pred_has_gap = True  # predicted gaps always assert a gap exists

        for gi, gt in enumerate(ground_truth):
            if gi in matched_gt:
                continue
            if not gt.get("has_gap", True):
                continue  # skip ground truth negatives

            gt_article = gt.get("regulatory_article", "")
            if _articles_match(pred_article, gt_article):
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_pred)
    fp = len(confident_preds) - tp

    # Ground truth positives that weren't matched
    gt_positives = [g for g in ground_truth if g.get("has_gap", True)]
    fn = len(gt_positives) - len(matched_gt)

    return max(0, tp), max(0, fp), max(0, fn)


# ── Pipeline runner (mocked for eval harness) ─────────────────────────────────


async def _run_detection_on_example(
    example: GapExample,
    confidence_threshold: float,
    verbose: bool,
) -> PredictionResult:
    """Run the full synthesis pipeline on a single example and return predictions."""
    from src.db.session import get_db_session
    from src.models.llm import ClaudeClient
    from src.rag.retrieve import hybrid_retrieve

    claude = ClaudeClient()
    t_start = time.monotonic()

    try:
        # Retrieve regulatory context
        async with get_db_session() as db:
            chunks = await hybrid_retrieve(db, example.policy_text[:512], example.regulation_scope)

        # Format context and run synthesis
        ctx_text = "\n\n".join(
            f"[{c.regulation.upper()} {c.article or ''}]\n{c.content}" for c in chunks[:5]
        )

        gap_dicts = await claude.synthesize(
            regulatory_context=ctx_text,
            policy_content=example.policy_text,
            regulation_scope=example.regulation_scope,
            ner_summary="",
        )

    except Exception as exc:
        logger.warning("Pipeline failed for example %s: %s", example.example_id, exc)
        gap_dicts = []

    latency_ms = int((time.monotonic() - t_start) * 1000)

    tp, fp, fn = _match_gaps(gap_dicts, example.ground_truth_gaps, confidence_threshold)

    result = PredictionResult(
        example_id=example.example_id,
        predicted_gaps=gap_dicts,
        ground_truth_gaps=example.ground_truth_gaps,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        latency_ms=latency_ms,
    )

    if verbose and (fp > 0 or fn > 0):
        console.print(
            f"  [yellow]Example {example.example_id}:[/yellow] "
            f"TP={tp} FP={fp} FN={fn} ({latency_ms}ms)"
        )

    return result


# ── Main evaluation ───────────────────────────────────────────────────────────


async def run_gap_detection_eval(
    dataset_path: Path,
    confidence_threshold: float = 0.5,
    output_mlflow: bool = False,
    verbose: bool = False,
    sweep_thresholds: list[float] | None = None,
) -> EvalSummary:
    """
    Run the gap detection F1 evaluation.

    Args:
        dataset_path: Path to the JSONL gap detection dataset.
        confidence_threshold: Minimum gap confidence to count as a prediction.
        output_mlflow: Log results to MLflow.
        verbose: Print per-example failures.
        sweep_thresholds: If provided, run eval at each threshold value.

    Returns:
        EvalSummary with precision, recall, F1, and per-framework breakdown.
    """
    examples = load_gap_dataset(dataset_path)
    if not examples:
        raise ValueError(f"No examples loaded from {dataset_path}")

    console.print(
        f"[cyan]Running gap detection eval on {len(examples)} examples "
        f"(threshold={confidence_threshold})...[/cyan]"
    )
    t_start = time.monotonic()

    # Run examples (with concurrency limit to avoid hammering the API)
    semaphore = asyncio.Semaphore(3)

    async def run_with_semaphore(ex: GapExample) -> PredictionResult:
        async with semaphore:
            return await _run_detection_on_example(ex, confidence_threshold, verbose)

    results = await asyncio.gather(
        *[run_with_semaphore(ex) for ex in examples],
        return_exceptions=False,
    )

    # Aggregate statistics
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Per-framework breakdown
    per_framework: dict[str, dict[str, Any]] = {}
    for ex, result in zip(examples, results, strict=True):
        for scope in ex.regulation_scope:
            if scope not in per_framework:
                per_framework[scope] = {"tp": 0, "fp": 0, "fn": 0}
            per_framework[scope]["tp"] += result.true_positives
            per_framework[scope]["fp"] += result.false_positives
            per_framework[scope]["fn"] += result.false_negatives

    framework_scores: dict[str, dict[str, float]] = {}
    for fw, counts in per_framework.items():
        fw_tp, fw_fp, fw_fn = counts["tp"], counts["fp"], counts["fn"]
        fw_p = fw_tp / (fw_tp + fw_fp) if (fw_tp + fw_fp) > 0 else 0.0
        fw_r = fw_tp / (fw_tp + fw_fn) if (fw_tp + fw_fn) > 0 else 0.0
        fw_f1 = 2 * fw_p * fw_r / (fw_p + fw_r) if (fw_p + fw_r) > 0 else 0.0
        framework_scores[fw] = {"precision": fw_p, "recall": fw_r, "f1": fw_f1}

    failures = [r for r in results if r.false_positives > 0 or r.false_negatives > 0]
    duration = time.monotonic() - t_start

    summary = EvalSummary(
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=confidence_threshold,
        total_examples=len(examples),
        total_true_positives=total_tp,
        total_false_positives=total_fp,
        total_false_negatives=total_fn,
        per_framework=framework_scores,
        failures=failures,
        duration_seconds=duration,
    )

    # Print results
    _print_gap_results(summary)

    # Log to MLflow
    if output_mlflow:
        _log_gap_to_mlflow(summary, dataset_path)

    # Threshold sweep
    if sweep_thresholds:
        console.print("\n[cyan]Threshold sweep:[/cyan]")
        sweep_table = Table(show_header=True)
        sweep_table.add_column("Threshold", width=12)
        sweep_table.add_column("Precision", width=12)
        sweep_table.add_column("Recall", width=12)
        sweep_table.add_column("F1", width=12)
        for t in sorted(sweep_thresholds):
            tp_s = sum(_match_gaps(r.predicted_gaps, r.ground_truth_gaps, t)[0] for r in results)
            fp_s = sum(_match_gaps(r.predicted_gaps, r.ground_truth_gaps, t)[1] for r in results)
            fn_s = sum(_match_gaps(r.predicted_gaps, r.ground_truth_gaps, t)[2] for r in results)
            p_s = tp_s / (tp_s + fp_s) if tp_s + fp_s > 0 else 0.0
            r_s = tp_s / (tp_s + fn_s) if tp_s + fn_s > 0 else 0.0
            f1_s = 2 * p_s * r_s / (p_s + r_s) if p_s + r_s > 0 else 0.0
            sweep_table.add_row(f"{t:.2f}", f"{p_s:.4f}", f"{r_s:.4f}", f"{f1_s:.4f}")
        console.print(sweep_table)

    return summary


def _print_gap_results(summary: EvalSummary) -> None:
    passed = summary.f1 >= THRESHOLDS.gap_detection_f1

    table = Table(title="Gap Detection Evaluation", show_header=True)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", justify="right", width=12)

    table.add_row("Examples", str(summary.total_examples))
    table.add_row("True positives", str(summary.total_true_positives))
    table.add_row("False positives", str(summary.total_false_positives))
    table.add_row("False negatives", str(summary.total_false_negatives))
    table.add_row("Precision", f"{summary.precision:.4f}")
    table.add_row("Recall", f"{summary.recall:.4f}")
    table.add_row(
        "F1",
        f"[green]{summary.f1:.4f}[/green]" if passed else f"[red]{summary.f1:.4f}[/red]",
    )
    table.add_row("Threshold", f"{THRESHOLDS.gap_detection_f1:.4f}")
    table.add_row("Duration", f"{summary.duration_seconds:.1f}s")

    console.print(table)

    if summary.per_framework:
        fw_table = Table(title="Per-framework breakdown", show_header=True)
        fw_table.add_column("Framework", width=15)
        fw_table.add_column("Precision", width=12)
        fw_table.add_column("Recall", width=12)
        fw_table.add_column("F1", width=12)
        for fw, scores in summary.per_framework.items():
            fw_table.add_row(
                fw.upper(),
                f"{scores['precision']:.4f}",
                f"{scores['recall']:.4f}",
                f"{scores['f1']:.4f}",
            )
        console.print(fw_table)

    if passed:
        console.print(
            f"[green]✓ F1 {summary.f1:.4f} >= threshold {THRESHOLDS.gap_detection_f1}[/green]"
        )
    else:
        console.print(f"[red]✗ F1 {summary.f1:.4f} < threshold {THRESHOLDS.gap_detection_f1}[/red]")


def _log_gap_to_mlflow(summary: EvalSummary, dataset_path: Path) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="gap_detection_eval"):
            mlflow.log_params(
                {
                    "pipeline_version": settings.VERSION,
                    "dataset": str(dataset_path),
                    "confidence_threshold": summary.threshold,
                    "total_examples": summary.total_examples,
                }
            )
            mlflow.log_metrics(
                {
                    "gap_precision": summary.precision,
                    "gap_recall": summary.recall,
                    "gap_f1": summary.f1,
                    "gap_tp": summary.total_true_positives,
                    "gap_fp": summary.total_false_positives,
                    "gap_fn": summary.total_false_negatives,
                }
            )
        console.print("[green]Results logged to MLflow[/green]")
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    dataset: Path = typer.Option(
        Path("data/golden/gap_detection.jsonl"),
        "--dataset",
        "-d",
    ),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Confidence threshold"),
    output_mlflow: bool = typer.Option(False, "--output-mlflow"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    sweep: str = typer.Option("", "--sweep-thresholds", help="Comma-separated thresholds to sweep"),
) -> None:
    """Run the compliance gap detection F1 evaluation."""
    sweep_list = [float(t) for t in sweep.split(",") if t.strip()] if sweep else None

    summary = asyncio.run(
        run_gap_detection_eval(
            dataset_path=dataset,
            confidence_threshold=threshold,
            output_mlflow=output_mlflow,
            verbose=verbose,
            sweep_thresholds=sweep_list,
        )
    )
    raise SystemExit(0 if summary.f1 >= THRESHOLDS.gap_detection_f1 else 1)


if __name__ == "__main__":
    app()
