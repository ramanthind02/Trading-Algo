"""Long-format CSV export for feature–vault correlation (Power BI / BI tools)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

_VAULT_CORRELATION_LONG_COLS: list[str] = [
    "fold_id",
    "window_kind",
    "research_param_combo_label",
    "research_feature_name",
    "vault_member_id",
    "vault_ensemble_path",
    "vault_feature_name",
    "ticker",
    "metric_name",
    "metric_value",
    "n_obs",
]


def _dataframe_with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reindex to exact column list; missing columns become NaN, extras are dropped."""
    return df.reindex(columns=columns)


def _write_frame_csv(df: pd.DataFrame, stem: Path) -> Path:
    csv_path = stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def write_vault_correlation_powerbi_long(
    rows: Sequence[Mapping[str, object]],
    output_csv_stem: Path,
) -> Path:
    """Write long-format vault vs research correlation metrics for Power BI (stable columns)."""
    df = _dataframe_with_columns(pd.DataFrame(list(rows)), _VAULT_CORRELATION_LONG_COLS)
    output_csv_stem.parent.mkdir(parents=True, exist_ok=True)
    return _write_frame_csv(df, output_csv_stem)
