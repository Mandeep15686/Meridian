"""Evaluation harness — RAGAS, gap detection F1, LLM-as-judge, and threshold enforcement."""

from src.eval.thresholds import THRESHOLDS, EvalThresholds, check_thresholds, ThresholdCheckResult

__all__ = ["THRESHOLDS", "EvalThresholds", "check_thresholds", "ThresholdCheckResult"]
