from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pandas as pd

from feature_research.core_helpers import (
    combo_key,
    normalize_datetime_index,
    normalize_series_datetime_index,
)

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig


def build_signed_signal_walkforward_evaluator(
    combo_signal_target: dict[tuple[tuple[str, object], ...], pd.DataFrame],
) -> Callable[..., pd.Series]:
    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
        *,
        train_end: pd.Timestamp | None = None,
    ) -> pd.Series:
        _ = train_end
        if "selected_bin" in params:
            raise ValueError("selected_bin expansion is unsupported in frozen-signal research.")
        combo = combo_key(params)
        combo_data = combo_signal_target.get(combo)
        if combo_data is None:
            return pd.Series(dtype=float)

        fold_index_norm = normalize_datetime_index(fold_candles.index)
        combo_index_norm = normalize_datetime_index(combo_data.index)
        fold_data = combo_data.loc[combo_index_norm.isin(fold_index_norm)].dropna()
        if fold_data.empty:
            return pd.Series(dtype=float)

        signal = normalize_series_datetime_index(fold_data["signal"])
        target = normalize_series_datetime_index(fold_data["target"])
        return signal.mul(target)

    return evaluate_param_combo
