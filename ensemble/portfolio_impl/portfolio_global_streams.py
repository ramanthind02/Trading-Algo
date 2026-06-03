from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def build_global_signals_df(
    forecast_vectors: List[pd.DataFrame],
) -> pd.DataFrame:
    """Build the global date x model signal matrix used by WeightLayer."""
    if not forecast_vectors:
        return pd.DataFrame()

    all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
    if all_forecasts.empty:
        return pd.DataFrame()
    if "datetime" not in all_forecasts.columns or "signal" not in all_forecasts.columns:
        return pd.DataFrame()

    df = all_forecasts.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.normalize()
    return (
        df.pivot_table(
            index="date",
            columns="model_name",
            values="signal",
            aggfunc="mean",
        )
        .fillna(0.0)
        .sort_index()
    )


def build_daily_grid(
    forecast_vectors: List[pd.DataFrame],
    reference_index: Optional[pd.Index] = None,
) -> pd.DatetimeIndex:
    """Build a shared normalized daily grid for strategy-stream alignment."""
    if reference_index is not None and len(reference_index) > 0:
        idx = pd.to_datetime(reference_index)
        if isinstance(idx, pd.Series):
            idx = idx.dt.normalize()
        else:
            idx = pd.DatetimeIndex(idx).normalize()
        idx = idx[~idx.isna()]
        if len(idx) > 0:
            return pd.DatetimeIndex(sorted(set(idx)))

    all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
    if all_forecasts.empty or "datetime" not in all_forecasts.columns:
        return pd.DatetimeIndex([])

    dates = pd.to_datetime(all_forecasts["datetime"]).dt.normalize().dropna()
    if dates.empty:
        return pd.DatetimeIndex([])
    return pd.date_range(dates.min(), dates.max(), freq="D")


def align_forecast_vectors_to_daily_grid(
    forecast_vectors: List[pd.DataFrame],
    daily_grid: pd.DatetimeIndex,
) -> List[pd.DataFrame]:
    """Project each ticker/model stream onto a common daily grid via forward-fill."""
    if not forecast_vectors or len(daily_grid) == 0:
        return forecast_vectors

    combined = pd.concat(forecast_vectors, ignore_index=True)
    if combined.empty:
        return forecast_vectors

    required_cols = {"ticker", "datetime", "model_name", "forecast", "signal", "timeframe"}
    if not required_cols.issubset(set(combined.columns)):
        return forecast_vectors

    aligned_parts: List[pd.DataFrame] = []
    for (ticker, model_name, timeframe), grp in combined.groupby(
        ["ticker", "model_name", "timeframe"], sort=False
    ):
        base = grp.copy()
        base["datetime"] = pd.to_datetime(base["datetime"]).dt.normalize()
        base = (
            base.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .set_index("datetime")
        )
        stream = base[["forecast", "signal"]].astype(float)
        # Reindex through the union of bar dates and the daily grid so that
        # non-trading-day bar closes (e.g. weekly bars timestamped on Sunday)
        # are forward-filled into the next valid trading day before extraction.
        superset = pd.DatetimeIndex(sorted(set(daily_grid) | set(stream.index)))
        aligned = (
            stream.reindex(superset).ffill().fillna(0.0).reindex(daily_grid).reset_index()
        )
        aligned = aligned.rename(columns={"index": "datetime"})
        aligned["ticker"] = ticker
        aligned["model_name"] = model_name
        aligned["timeframe"] = timeframe
        aligned_parts.append(
            aligned[
                [
                    "ticker",
                    "datetime",
                    "model_name",
                    "forecast",
                    "signal",
                    "timeframe",
                ]
            ]
        )

    if not aligned_parts:
        return forecast_vectors
    return [pd.concat(aligned_parts, ignore_index=True)]


def normalize_global_signals_by_downside_vol(
    forecast_vectors: List[pd.DataFrame],
    global_returns: Optional[pd.Series],
) -> List[pd.DataFrame]:
    """Scale each stream by downside volatility of signal times aggregate returns."""
    if global_returns is None or global_returns.empty:
        return forecast_vectors

    clean_returns = global_returns.astype(float).copy()
    clean_returns.index = pd.to_datetime(clean_returns.index).normalize()

    combined = pd.concat(forecast_vectors, ignore_index=True).copy()
    combined["date"] = pd.to_datetime(combined["datetime"]).dt.normalize()

    scales: dict[tuple[str, str], float] = {}
    for (ticker, model_name), grp in combined.groupby(["ticker", "model_name"], sort=False):
        merged = (
            grp[["date", "signal"]]
            .merge(
                clean_returns.rename("ret").to_frame(),
                left_on="date",
                right_index=True,
                how="left",
            )
            .fillna({"ret": 0.0})
        )
        signal_ret = merged["signal"].astype(float) * merged["ret"].astype(float)
        downside = np.minimum(signal_ret.to_numpy(dtype=float), 0.0)
        downside_vol = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
        scales[(str(ticker), str(model_name))] = max(downside_vol, 1e-8)

    normalized: List[pd.DataFrame] = []
    for vec in forecast_vectors:
        out = vec.copy()
        out["signal"] = out.apply(
            lambda row: float(row["signal"])
            / scales.get((str(row["ticker"]), str(row["model_name"])), 1.0),
            axis=1,
        )
        normalized.append(out)
    return normalized
