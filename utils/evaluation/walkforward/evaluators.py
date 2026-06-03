from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pandas as pd

from feature_research._internal.core_helpers import (
    combo_key,
    normalize_datetime_index,
    normalize_series_datetime_index,
)

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig


def build_signed_signal_walkforward_evaluator(
    combo_signal_target: dict[tuple[tuple[str, object], ...], pd.DataFrame],
) -> Callable[..., pd.Series]:
    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series] = {}
    combo_indices: dict[tuple[tuple[str, object], ...], pd.DatetimeIndex] = {}
    for combo, combo_data in combo_signal_target.items():
        if combo_data.empty:
            continue
        if "returns" in combo_data.columns:
            returns = combo_data["returns"]
        else:
            signal = normalize_series_datetime_index(combo_data["signal"])
            target = normalize_series_datetime_index(combo_data["target"])
            returns = signal.mul(target)
        normalized_returns = normalize_series_datetime_index(returns)
        combo_returns[combo] = normalized_returns
        combo_indices[combo] = normalize_datetime_index(normalized_returns.index)

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
        returns = combo_returns.get(combo)
        combo_index_norm = combo_indices.get(combo)
        if returns is None or combo_index_norm is None:
            return pd.Series(dtype=float)

        fold_index_norm = normalize_datetime_index(fold_candles.index)
        fold_mask = combo_index_norm.isin(fold_index_norm)
        if not fold_mask.any():
            return pd.Series(dtype=float)
        return returns.loc[fold_mask]

    return evaluate_param_combo
