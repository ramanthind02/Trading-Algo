"""Persist walk-forward ensemble return series for aligned QuantStats tearsheets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

WF_CANDIDATE_TRAIN_RETURNS_CSV = "wf_candidate_train_returns.csv"
WF_CANDIDATE_VALIDATION_RETURNS_CSV = "wf_candidate_validation_returns.csv"


def write_wf_tearsheet_returns_csv(
    *,
    strategy_returns: pd.Series,
    baseline_returns: pd.Series,
    output_path: Path,
) -> Path:
    """Write strategy and baseline daily returns used for a walk-forward ensemble tearsheet."""
    strategy = strategy_returns.astype(float).copy()
    strategy.index = pd.to_datetime(strategy.index).normalize()
    baseline = baseline_returns.astype(float).reindex(strategy.index).fillna(0.0)
    baseline.index = pd.to_datetime(baseline.index).normalize()
    frame = pd.DataFrame(
        {
            "strategy_return": strategy,
            "baseline_return": baseline,
        }
    )
    frame.index.name = "datetime"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, encoding="utf-8")
    return output_path


def load_wf_tearsheet_returns_csv(path: Path) -> tuple[pd.Series, pd.Series]:
    """Load strategy and baseline series written by ``write_wf_tearsheet_returns_csv``."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    strategy = frame["strategy_return"].astype(float)
    strategy.index = pd.to_datetime(strategy.index).normalize()
    baseline = frame["baseline_return"].astype(float)
    baseline.index = pd.to_datetime(baseline.index).normalize()
    return strategy, baseline
