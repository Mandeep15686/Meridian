"""Evaluation harness — RAGAS, gap detection F1, LLM-as-judge, and threshold enforcement."""

from src.eval.thresholds import THRESHOLDS, EvalThresholds, ThresholdCheckResult, check_thresholds

__all__ = ["THRESHOLDS", "EvalThresholds", "check_thresholds", "ThresholdCheckResult"]
