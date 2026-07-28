"""
Threshold check script for CI.

Reads the latest eval metrics from MLflow and exits non-zero
if any metric is below its threshold — blocking the CI pipeline.

Usage:
    python scripts/check_thresholds.py
    python scripts/check_thresholds.py --run-id <mlflow_run_id>
    python scripts/check_thresholds.py --metrics-file /tmp/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

from src.eval.thresholds import check_thresholds

console = Console()
app = typer.Typer()


def _load_from_mlflow(run_id: str | None = None) -> dict[str, float]:
    """Load the latest eval metrics from MLflow."""
    try:
        import mlflow

        from src.config import settings

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        if run_id:
            run = client.get_run(run_id)
            return dict(run.data.metrics)

        # Get the latest run from the eval experiment
        experiment = client.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            console.print(
                f"[yellow]MLflow experiment '{settings.MLFLOW_EXPERIMENT_NAME}' not found[/yellow]"
            )
            return {}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            console.print("[yellow]No MLflow runs found[/yellow]")
            return {}

        latest_run = runs[0]
        console.print(f"Using MLflow run: {latest_run.info.run_id}")
        return dict(latest_run.data.metrics)

    except Exception as exc:
        console.print(f"[red]MLflow load failed: {exc}[/red]")
        return {}


def _load_from_file(metrics_file: Path) -> dict[str, float]:
    """Load metrics from a JSON file."""
    try:
        data = json.loads(metrics_file.read_text())
        return {k: float(v) for k, v in data.items()}
    except Exception as exc:
        console.print(f"[red]Failed to load metrics file {metrics_file}: {exc}[/red]")
        return {}


@app.command()
def main(
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Specific MLflow run ID to check. If not set, uses the latest run.",
    ),
    metrics_file: Path | None = typer.Option(
        None,
        "--metrics-file",
        "-f",
        help="JSON file with metrics dict (alternative to MLflow).",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Exit 1 if any threshold fails (default). Use --no-strict to warn only.",
    ),
) -> None:
    """
    Check the latest eval metrics against enforced CI thresholds.

    Exits 0 if all thresholds pass, 1 if any fail.
    """
    console.print("[bold]Meridian threshold checker[/bold]\n")

    # Load metrics
    metrics = (
        _load_from_file(metrics_file) if metrics_file else _load_from_mlflow(run_id)
    )

    if not metrics:
        console.print("[yellow]No metrics available — cannot check thresholds[/yellow]")
        raise SystemExit(0 if not strict else 1)

    # Display loaded metrics
    console.print("Loaded metrics:")
    for k, v in sorted(metrics.items()):
        console.print(f"  {k}: {v:.4f}")
    console.print()

    # Check thresholds
    result = check_thresholds(metrics)
    console.print(result.summary())

    if not result.passed:
        if strict:
            console.print("\n[red]Threshold check FAILED — blocking CI pipeline[/red]")
            raise SystemExit(1)
        else:
            console.print(
                "\n[yellow]Threshold check FAILED (non-strict mode — not blocking)[/yellow]"
            )
    else:
        console.print("\n[green]Threshold check PASSED[/green]")

    raise SystemExit(0)


if __name__ == "__main__":
    app()
