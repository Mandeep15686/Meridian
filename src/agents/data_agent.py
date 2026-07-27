"""Data agent LangGraph node — TAPAS table QA + Chronos time-series + anomaly scoring."""

from __future__ import annotations

import io
import logging
import time

import pandas as pd

from src.graph.state import AgentExtraction, MeridianState, UploadedFile
from src.models.tabular import TAPASModel, TimeSeriesForecaster
from src.storage.base import get_storage

logger = logging.getLogger(__name__)

_tapas = TAPASModel()
_forecaster = TimeSeriesForecaster()

COMPLIANCE_TABLE_QUESTIONS = [
    "What is the maximum data access duration?",
    "How many failed access attempts are recorded?",
    "What is the most recent data export date?",
    "Which users have administrative privileges?",
    "What is the total number of records processed?",
    "Are there any entries outside business hours?",
]


async def data_agent_node(state: MeridianState) -> dict:
    """
    LangGraph node: analyze structured data (CSV / Excel) for compliance signals.

    Pipeline:
    1. Download file and parse into DataFrame
    2. Run TAPAS table QA with compliance questions
    3. Detect time series column and forecast with Chronos
    4. Run anomaly detection on numeric columns
    5. Return AgentExtraction with structured findings
    """
    t_start = time.monotonic()
    file: UploadedFile = state["_current_file"]
    job_id = state.get("job_id", "unknown")

    logger.info("[data_agent] Processing file: %s (job=%s)", file.filename, job_id)

    try:
        storage = get_storage()
        file_bytes = await storage.download(file.storage_key)

        # ── 1. Parse into DataFrame ────────────────────────────────────────────
        df = _parse_tabular(file_bytes, file.filename)
        if df.empty:
            logger.warning("[data_agent] Empty DataFrame from %s", file.filename)
            return {"raw_extractions": []}

        logger.info("[data_agent] Parsed %s: %d rows × %d cols", file.filename, len(df), len(df.columns))

        # ── 2. TAPAS table QA ─────────────────────────────────────────────────
        tapas_answers = await _tapas.answer_batch(df, COMPLIANCE_TABLE_QUESTIONS)
        tapas_dicts = [
            {"question": a.question, "answer": a.answer, "aggregator": a.aggregator}
            for a in tapas_answers
            if a.answer.strip()
        ]

        # ── 3. Time series forecasting ────────────────────────────────────────
        forecast_output = None
        time_col, value_col = _forecaster.detect_time_series(df)

        if time_col and value_col:
            try:
                # Parse timestamps as strings for labels
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                series_df = df.dropna(subset=[time_col, value_col]).sort_values(time_col)
                timestamps = series_df[time_col].dt.strftime("%Y-%m-%d").tolist()
                values = series_df[value_col].astype(float).tolist()

                forecast = _forecaster.forecast(values, timestamps)
                forecast_output = {
                    "metric": value_col,
                    "timestamps": forecast.timestamps,
                    "median": forecast.forecast_median,
                    "lower": forecast.forecast_lower,
                    "upper": forecast.forecast_upper,
                    "anomaly_periods": forecast.anomaly_periods,
                }
                logger.info(
                    "[data_agent] Forecast complete: %d anomalies detected",
                    len(forecast.anomaly_periods),
                )
            except Exception as exc:
                logger.warning("[data_agent] Forecasting failed: %s", exc)

        # ── 4. Anomaly scoring ────────────────────────────────────────────────
        anomaly_scores = _score_anomalies(df)

        # ── 5. Build summary ──────────────────────────────────────────────────
        summary_lines = [f"Table: {len(df)} rows, {len(df.columns)} columns"]
        for ans in tapas_dicts[:4]:
            if ans["answer"]:
                summary_lines.append(f"{ans['question']}: {ans['answer']}")
        if forecast_output and forecast_output.get("anomaly_periods"):
            summary_lines.append(
                f"Time series anomalies detected at: {', '.join(forecast_output['anomaly_periods'][:3])}"
            )
        high_risk = [a for a in anomaly_scores if a.get("risk_score", 0) > 0.7]
        if high_risk:
            summary_lines.append(f"{len(high_risk)} high-risk rows detected")

        duration_ms = int((time.monotonic() - t_start) * 1000)
        logger.info("[data_agent] %s complete: %d TAPAS answers, %dms",
                    file.filename, len(tapas_dicts), duration_ms)

        extraction = AgentExtraction(
            agent="data_agent",
            file_id=file.file_id,
            raw_text=df.to_string(max_rows=20, max_cols=10),
            summary="\n".join(summary_lines),
            table_summary="\n".join(summary_lines),
            tapas_answers=tapas_dicts,
            anomaly_scores=anomaly_scores[:50],  # cap stored anomalies
            forecast_output=[forecast_output] if forecast_output else None,
            duration_ms=duration_ms,
        )

        return {"raw_extractions": [extraction]}

    except Exception as exc:
        logger.exception("[data_agent] Failed for file %s: %s", file.filename, exc)
        return {
            "raw_extractions": [],
            "error": str(exc),
            "error_stage": "data_agent",
        }


def _parse_tabular(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV or Excel bytes into a DataFrame."""
    ext = filename.lower()
    buf = io.BytesIO(file_bytes)

    if ext.endswith(".csv"):
        return pd.read_csv(buf, nrows=10_000)
    if ext.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf, nrows=10_000)
    if ext.endswith(".json"):
        return pd.read_json(buf)

    # Try CSV as fallback
    try:
        buf.seek(0)
        return pd.read_csv(buf, nrows=10_000)
    except Exception:
        return pd.DataFrame()


def _score_anomalies(df: pd.DataFrame) -> list[dict]:
    """
    Simple anomaly scoring using IQR method on numeric columns.

    Returns list of {row_index, column, value, risk_score} for outliers.
    """
    import numpy as np

    anomalies: list[dict] = []
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()

    for col in numeric_cols[:5]:  # check up to 5 numeric columns
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        upper_fence = q3 + 3 * iqr
        lower_fence = q1 - 3 * iqr
        outlier_mask = (series < lower_fence) | (series > upper_fence)

        for idx in series[outlier_mask].index[:20]:
            val = series.loc[idx]
            z = abs((val - series.mean()) / series.std())
            anomalies.append({
                "row_index": int(idx),
                "column": col,
                "value": float(val),
                "risk_score": min(1.0, float(z) / 10),
            })

    return sorted(anomalies, key=lambda a: -a["risk_score"])
