"""Export fitted WeightLayer diagnostics to long-format CSV for BI tools.

Rows are one per (phase × weight-layer ticker × model/stream id). Model and stream
identifiers come from runtime diagnostics, not from a fixed column list, so new
ensembles or base models do not require code changes.

**Column semantics (avoid confusing weights with FDM)**

- ``stream_weight``: nonnegative share of the *synthetic* global portfolio for that
  stream. For ``weight_layer_ticker == __GLOBAL__``, stream weights **sum to 1** across
  all rows in that phase (Carver-style combination of many streams).
- ``fdm_multiplier``: **not** a portfolio weight. It is the forecast diversification
  multiplier applied to the *combined* weighted forecast (often capped at ``fdm_max``,
  e.g. 2.0). It may be greater than 1 by design.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, cast

import pandas as pd

from ensemble.portfolio_impl.global_weight_layer_adapter import decode_global_stream_id

# Must stay aligned with ``_GLOBAL_WEIGHT_LAYER_TICKER`` in global_weight_layer_adapter.
_GLOBAL_SYNTHETIC_TICKER = "__GLOBAL__"

_EXPORT_COLUMNS: tuple[str, ...] = (
    "phase",
    "fit_window_start",
    "fit_window_end",
    "predict_window_start",
    "predict_window_end",
    "weight_method",
    "fdm_max",
    "weight_layer_ticker",
    "stream_or_model_id",
    "stream_weight",
    "fdm_multiplier",
    "mean_signal_correlation",
    "mean_cluster_correlation",
    "cluster_id",
    "stream_instrument_ticker",
    "stream_signal_timeframe",
    "stream_source_model_name",
)


def _decoded_stream_columns(stream_id: str) -> dict[str, str | None]:
    try:
        decoded = decode_global_stream_id(stream_id)
    except ValueError:
        return {
            "stream_instrument_ticker": None,
            "stream_signal_timeframe": None,
            "stream_source_model_name": None,
        }
    return {
        "stream_instrument_ticker": decoded["ticker"],
        "stream_signal_timeframe": decoded["timeframe"],
        "stream_source_model_name": decoded["original_model_name"],
    }


def weight_layer_diagnostics_to_dataframe(
    diagnostics: Mapping[str, object],
    *,
    phase: str,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    predict_start: pd.Timestamp,
    predict_end: pd.Timestamp,
) -> pd.DataFrame:
    """Flatten ``BaseWeightLayer.get_diagnostics()`` into a long DataFrame."""
    base_meta = {
        "phase": phase,
        "fit_window_start": fit_start.isoformat(),
        "fit_window_end": fit_end.isoformat(),
        "predict_window_start": predict_start.isoformat(),
        "predict_window_end": predict_end.isoformat(),
        "weight_method": diagnostics.get("weight_method"),
        "fdm_max": diagnostics.get("fdm_max"),
    }

    if diagnostics.get("is_fitted") is not True:
        return pd.DataFrame(columns=list(_EXPORT_COLUMNS))

    tickers_raw = diagnostics.get("tickers")
    if not isinstance(tickers_raw, dict):
        return pd.DataFrame(columns=list(_EXPORT_COLUMNS))

    tickers_map = cast(dict[str, Mapping[str, object]], tickers_raw)
    rows: list[dict[str, object]] = []

    for wl_ticker, ticker_info in tickers_map.items():
        weights_raw = ticker_info.get("weights")
        weights = (
            cast(dict[str, float], weights_raw)
            if isinstance(weights_raw, dict)
            else {}
        )
        assignments_raw = ticker_info.get("cluster_assignments")
        assignments = (
            cast(dict[str, str], assignments_raw)
            if isinstance(assignments_raw, dict)
            else {}
        )
        fdm = ticker_info.get("fdm")
        mean_sig = ticker_info.get("mean_signal_correlation")
        mean_cl = ticker_info.get("mean_cluster_correlation")

        for stream_or_model_id, weight in weights.items():
            row: dict[str, object] = {
                **base_meta,
                "weight_layer_ticker": wl_ticker,
                "stream_or_model_id": stream_or_model_id,
                "stream_weight": float(weight),
                "fdm_multiplier": float(fdm) if isinstance(fdm, (int, float)) else fdm,
                "mean_signal_correlation": mean_sig,
                "mean_cluster_correlation": mean_cl,
                "cluster_id": assignments.get(str(stream_or_model_id)),
            }
            if wl_ticker == _GLOBAL_SYNTHETIC_TICKER:
                row.update(_decoded_stream_columns(str(stream_or_model_id)))
            else:
                row["stream_instrument_ticker"] = None
                row["stream_signal_timeframe"] = None
                row["stream_source_model_name"] = None
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(_EXPORT_COLUMNS))

    frame = pd.DataFrame(rows)
    return frame.reindex(columns=list(_EXPORT_COLUMNS))


def write_weight_layer_csv(df: pd.DataFrame, path: Path) -> None:
    """Write diagnostics DataFrame to UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
