"""Tabular model wrappers: TAPAS table QA and Chronos-T5 time-series forecasting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from src.models.base import BaseHFModel
from src.models.registry import MODELS

logger = logging.getLogger(__name__)


# ── TAPAS Table Question Answering ────────────────────────────────────────────

@dataclass
class TAPASAnswer:
    question: str
    answer: str
    cells: list[str] = field(default_factory=list)
    aggregator: str = "NONE"
    coordinates: list[tuple[int, int]] = field(default_factory=list)


class TAPASModel(BaseHFModel):
    """
    Table question answering using google/tapas-base-finetuned-wtq.

    Converts pandas DataFrames to the HF table format and returns
    cell-referenced answers with optional aggregation.
    """

    MAX_ROWS = 200      # TAPAS performance degrades with very large tables
    MAX_COLS = 20

    def __init__(self) -> None:
        super().__init__(MODELS.tapas)

    def _parse_response(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list) and raw:
            return cast(dict[str, Any], raw[0])
        return {}

    @staticmethod
    def _df_to_hf_table(df: pd.DataFrame) -> dict[str, Any]:
        """Convert a DataFrame to the HuggingFace table-QA input format."""
        # Ensure all values are strings (TAPAS requirement)
        table = {col: df[col].astype(str).tolist() for col in df.columns}
        return cast(dict[str, Any], table)

    async def answer(self, df: pd.DataFrame, question: str) -> TAPASAnswer:
        """
        Answer a natural language question about a DataFrame.

        Args:
            df: The table to query (rows × columns).
            question: Natural language question.

        Returns:
            TAPASAnswer with the extracted answer and cell references.
        """
        # Truncate large tables to stay within TAPAS limits
        if len(df) > self.MAX_ROWS:
            logger.warning("Table has %d rows; truncating to %d for TAPAS", len(df), self.MAX_ROWS)
            df = df.head(self.MAX_ROWS)
        if len(df.columns) > self.MAX_COLS:
            df = df.iloc[:, : self.MAX_COLS]

        # Sanitise column names
        df = df.copy()
        df.columns = [str(c).strip()[:64] for c in df.columns]

        payload = {
            "inputs": {
                "query": question,
                "table": self._df_to_hf_table(df),
            }
        }

        raw = cast(dict[str, Any], await self.predict(payload))

        cells = raw.get("cells", [])
        coordinates = raw.get("coordinates", [])
        aggregator = raw.get("aggregator", "NONE")

        # Build answer string from cells + aggregation
        if aggregator not in ("", "NONE") and cells:
            try:
                numeric_cells = [float(c.replace(",", "")) for c in cells]
                if aggregator == "SUM":
                    answer_str = str(sum(numeric_cells))
                elif aggregator == "AVERAGE":
                    answer_str = str(round(sum(numeric_cells) / len(numeric_cells), 2))
                elif aggregator == "COUNT":
                    answer_str = str(int(raw.get("answer", len(cells))))
                else:
                    answer_str = ", ".join(cells)
            except ValueError:
                answer_str = ", ".join(cells)
        else:
            answer_str = ", ".join(cells) if cells else raw.get("answer", "")

        return TAPASAnswer(
            question=question,
            answer=answer_str,
            cells=cells,
            aggregator=aggregator,
            coordinates=[tuple(c) for c in coordinates],
        )

    async def answer_batch(
        self, df: pd.DataFrame, questions: list[str]
    ) -> list[TAPASAnswer]:
        """Answer multiple questions about the same table."""
        answers: list[TAPASAnswer] = []
        for q in questions:
            try:
                ans = await self.answer(df, q)
                answers.append(ans)
            except Exception as exc:
                logger.warning("TAPAS failed for '%s': %s", q[:60], exc)
                answers.append(TAPASAnswer(question=q, answer="", cells=[], aggregator="NONE"))
        return answers


# ── Chronos-T5 Time Series Forecasting ────────────────────────────────────────

@dataclass
class ForecastResult:
    timestamps: list[str]
    forecast_median: list[float]
    forecast_lower: list[float]   # 10th percentile
    forecast_upper: list[float]   # 90th percentile
    anomaly_periods: list[str]    # timestamps where actual deviates significantly
    horizon: int


class TimeSeriesForecaster:
    """
    Zero-shot time series forecasting using amazon/chronos-t5-small.

    Loaded locally via the chronos-forecasting library to avoid
    HF API latency for this compute-bound step.
    """

    HISTORY_MIN = 10        # minimum historical points required
    HORIZON = 7             # forecast horizon in periods
    ANOMALY_Z_THRESHOLD = 2.0   # z-score threshold for anomaly flagging

    def __init__(self) -> None:
        self._pipeline = None

    def _get_pipeline(self) -> Any:
        """Lazy-load Chronos pipeline on first use."""
        if self._pipeline is None:
            try:
                import torch
                from chronos import ChronosPipeline

                self._pipeline = ChronosPipeline.from_pretrained(
                    MODELS.forecaster,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                )
                logger.info("Chronos pipeline loaded: %s", MODELS.forecaster)
            except ImportError:
                logger.warning(
                    "chronos-forecasting not installed. "
                    "Install with: pip install chronos-forecasting"
                )
                raise
        return self._pipeline

    def detect_time_series(self, df: pd.DataFrame) -> tuple[str | None, str | None]:
        """
        Detect time and value columns in a DataFrame.

        Returns:
            Tuple of (time_column_name, value_column_name) or (None, None).
        """
        date_cols = [
            col for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col])
            or "date" in str(col).lower()
            or "time" in str(col).lower()
            or "timestamp" in str(col).lower()
        ]
        numeric_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
            and col not in date_cols
        ]

        if not date_cols or not numeric_cols:
            return None, None

        return date_cols[0], numeric_cols[0]

    def forecast(
        self,
        time_series: list[float],
        timestamps: list[str] | None = None,
    ) -> ForecastResult:
        """
        Forecast the next HORIZON periods from a historical series.

        Args:
            time_series: Ordered list of historical numeric values.
            timestamps: Optional human-readable labels for historical points.

        Returns:
            ForecastResult with median, lower/upper bounds, and anomaly flags.
        """
        import torch

        if len(time_series) < self.HISTORY_MIN:
            logger.warning(
                "Time series has only %d points (min %d); using statistical fallback",
                len(time_series),
                self.HISTORY_MIN,
            )
            return self._statistical_fallback(time_series, timestamps)

        try:
            pipeline = self._get_pipeline()
            context = torch.tensor(time_series[-512:], dtype=torch.float32).unsqueeze(0)

            quantile_levels = [0.1, 0.5, 0.9]
            forecast_tensor = pipeline.predict(
                context=context,
                prediction_length=self.HORIZON,
                num_samples=100,
            )

            # forecast_tensor shape: (1, num_samples, horizon)
            samples = forecast_tensor[0].numpy()  # (num_samples, horizon)
            lower = np.quantile(samples, 0.1, axis=0).tolist()
            median = np.quantile(samples, 0.5, axis=0).tolist()
            upper = np.quantile(samples, 0.9, axis=0).tolist()

        except Exception as exc:
            logger.warning("Chronos inference failed, using statistical fallback: %s", exc)
            return self._statistical_fallback(time_series, timestamps)

        # Detect anomalies in historical data (where actual deviates from model expectation)
        anomaly_periods = self._detect_historical_anomalies(time_series, timestamps)

        # Generate forecast timestamp labels
        forecast_labels = [f"T+{i + 1}" for i in range(self.HORIZON)]

        return ForecastResult(
            timestamps=forecast_labels,
            forecast_median=median,
            forecast_lower=lower,
            forecast_upper=upper,
            anomaly_periods=anomaly_periods,
            horizon=self.HORIZON,
        )

    def _statistical_fallback(
        self, series: list[float], timestamps: list[str] | None
    ) -> ForecastResult:
        """Simple mean + std forecast for short series."""
        arr = np.array(series[-30:] if len(series) > 30 else series)
        mean, std = float(np.mean(arr)), float(np.std(arr))

        return ForecastResult(
            timestamps=[f"T+{i + 1}" for i in range(self.HORIZON)],
            forecast_median=[mean] * self.HORIZON,
            forecast_lower=[mean - 1.5 * std] * self.HORIZON,
            forecast_upper=[mean + 1.5 * std] * self.HORIZON,
            anomaly_periods=self._detect_historical_anomalies(series, timestamps),
            horizon=self.HORIZON,
        )

    def _detect_historical_anomalies(
        self, series: list[float], timestamps: list[str] | None
    ) -> list[str]:
        """Flag historical periods where value deviates by > Z_THRESHOLD std."""
        if len(series) < 4:
            return []
        arr = np.array(series)
        mean, std = np.mean(arr), np.std(arr)
        if std == 0:
            return []
        z_scores = np.abs((arr - mean) / std)
        anomaly_indices = np.where(z_scores > self.ANOMALY_Z_THRESHOLD)[0].tolist()
        if timestamps and len(timestamps) == len(series):
            return [timestamps[i] for i in anomaly_indices]
        return [f"Period {i}" for i in anomaly_indices]
