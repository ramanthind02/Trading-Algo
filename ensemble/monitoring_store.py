"""
Decay monitoring data store.

Stores raw (signal, target) vectors per base model in Parquet files under
vault/{timeframe}/{ensemble}/monitoring/. All decay metrics — rolling Sharpe,
CUSUM, Page-Hinkley — are computed on demand from these raw vectors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ensemble.vault_manager import _resolve_ensemble_path


def _monitoring_path(ensemble_dir: str, feature_name: str, model_id: str) -> Path:
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    return ensemble_path / "monitoring" / f"{feature_name}__{model_id}.parquet"


def initialize_monitoring(
    ensemble_dir: str,
    feature_name: str,
    model_id: str,
    signals: pd.Series,
    targets: pd.Series,
    period: str = "IS",
) -> Path:
    """Write the initial monitoring file (IS seed data).

    Parameters
    ----------
    ensemble_dir:
        Ensemble directory path (e.g. 'vault/D/commodity_breakout_long').
    feature_name:
        Canonical feature name (e.g. 'rsi_signal_D').
    model_id:
        Auto-generated model ID (e.g. 'signal_3').
    signals:
        Position-multiplier vector with DatetimeIndex.
    targets:
        Bar-return vector with DatetimeIndex.
    period:
        Period tag, defaults to 'IS'.

    Returns
    -------
    Path
        Path to the created parquet file.

    Raises
    ------
    FileExistsError
        If a monitoring file already exists for this feature/model_id pair.
    """
    path = _monitoring_path(ensemble_dir, feature_name, model_id)
    if path.exists():
        raise FileExistsError(
            f"Monitoring file already exists: {path}. "
            "Use append_monitoring_data() to add new observations."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _build_df(signals, targets, period)
    df.to_parquet(path, engine="fastparquet")
    return path


def append_monitoring_data(
    ensemble_dir: str,
    feature_name: str,
    model_id: str,
    signals: pd.Series,
    targets: pd.Series,
    period: str = "LIVE",
) -> None:
    """Append new observations to an existing monitoring file.

    Deduplicates on date index — last value wins for any repeated date.
    Creates the file if it does not yet exist.

    Parameters
    ----------
    ensemble_dir:
        Ensemble directory path.
    feature_name:
        Canonical feature name.
    model_id:
        Model ID.
    signals:
        New signal observations with DatetimeIndex.
    targets:
        New target observations with DatetimeIndex.
    period:
        Period tag for the new rows, defaults to 'LIVE'.
    """
    path = _monitoring_path(ensemble_dir, feature_name, model_id)
    new_df = _build_df(signals, targets, period)
    if path.exists():
        existing = pd.read_parquet(path, engine="fastparquet")
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df
    combined.to_parquet(path, engine="fastparquet")


def load_monitoring_data(
    ensemble_dir: str,
    feature_name: str,
    model_id: str,
    period: Optional[str] = None,
) -> pd.DataFrame:
    """Load signal/target vectors for a base model.

    Parameters
    ----------
    ensemble_dir:
        Ensemble directory path.
    feature_name:
        Canonical feature name.
    model_id:
        Model ID.
    period:
        If provided, filter to rows matching this period tag ('IS', 'OOS', 'LIVE').
        If None, return all periods.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['signal', 'target', 'period'] and DatetimeIndex.

    Raises
    ------
    FileNotFoundError
        If no monitoring file exists for this feature/model_id pair.
    """
    path = _monitoring_path(ensemble_dir, feature_name, model_id)
    if not path.exists():
        raise FileNotFoundError(f"No monitoring data found: {path}")
    df = pd.read_parquet(path, engine="fastparquet")
    if period is not None:
        df = df[df["period"] == period]
    return df


def _build_df(signals: pd.Series, targets: pd.Series, period: str) -> pd.DataFrame:
    idx = signals.index.union(targets.index)
    df = pd.DataFrame(
        {"signal": signals.reindex(idx), "target": targets.reindex(idx), "period": period},
        index=idx,
    ).dropna(subset=["signal", "target"])
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df
