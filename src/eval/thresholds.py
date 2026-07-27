"""Evaluation thresholds enforced in CI and groundedness scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass


# ── Thresholds ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalThresholds:
    """Minimum acceptable metric values enforced by the nightly CI job."""
    ragas_faithfulness: float = 0.85
    ragas_answer_relevancy: float = 0.82
    ragas_context_precision: float = 0.80
    ragas_context_recall: float = 0.80
    gap_detection_f1: float = 0.85
    agent_routing_accuracy: float = 0.95
    groundedness_pass_rate: float = 0.98
    p95_latency_seconds: float = 180.0  # 3 minutes


THRESHOLDS = EvalThresholds()


@dataclass
class ThresholdCheckResult:
    passed: bool
    failures: list[str]

    def summary(self) -> str:
        if self.passed:
            return "All thresholds passed ✓"
        return "FAILED:\n" + "\n".join(f"  ✗ {f}" for f in self.failures)


def check_thresholds(metrics: dict[str, float]) -> ThresholdCheckResult:
    """
    Check a metrics dict against all enforced thresholds.

    Args:
        metrics: Dict mapping metric name to float value.

    Returns:
        ThresholdCheckResult with pass/fail status and failure list.
    """
    failures: list[str] = []

    checks = {
        "ragas_faithfulness": THRESHOLDS.ragas_faithfulness,
        "ragas_answer_relevancy": THRESHOLDS.ragas_answer_relevancy,
        "ragas_context_precision": THRESHOLDS.ragas_context_precision,
        "ragas_context_recall": THRESHOLDS.ragas_context_recall,
        "gap_f1": THRESHOLDS.gap_detection_f1,
        "agent_routing_accuracy": THRESHOLDS.agent_routing_accuracy,
        "groundedness_pass_rate": THRESHOLDS.groundedness_pass_rate,
    }

    for metric_name, threshold in checks.items():
        value = metrics.get(metric_name)
        if value is None:
            continue  # skip missing metrics (not all evals run every time)
        if value < threshold:
            failures.append(
                f"{metric_name}: {value:.4f} < threshold {threshold:.4f} "
                f"(delta: {value - threshold:+.4f})"
            )

    return ThresholdCheckResult(passed=len(failures) == 0, failures=failures)
