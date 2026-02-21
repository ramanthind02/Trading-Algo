from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:
    from feature_research.walkforward.config import WalkforwardResearchConfig


def compute_trade_frequency(signal: pd.Series) -> float:
    if signal.empty:
        return 0.0
    return float((signal != 0).sum() / len(signal))


def _canonical_param_label(params: dict[str, object]) -> str:
    return "|".join(f"{key}={params[key]}" for key in sorted(params))


@dataclass(frozen=True)
class EnhancedSelectionResult:
    selected_labels: list[str]
    trade_frequencies: dict[str, float]


def run_enhanced_selection(
    training_data: pd.DataFrame,
    training_target: pd.Series,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series],
    smoothed_objectives: dict[str, float],
    config: WalkforwardResearchConfig,
) -> EnhancedSelectionResult:
    param_label_map = {_canonical_param_label(params): params for params in param_grid}
    signal_series = {
        label: evaluate_param_combo(training_data, training_target, params)
        for label, params in param_label_map.items()
    }

    trade_frequencies = {
        label: compute_trade_frequency(signal)
        for label, signal in signal_series.items()
    }
    surviving_labels = [
        label for label in param_label_map if trade_frequencies[label] >= config.trade_freq_min
    ]

    if not surviving_labels:
        return EnhancedSelectionResult(
            selected_labels=[],
            trade_frequencies=trade_frequencies,
        )

    ranked_survivors = sorted(
        surviving_labels,
        key=lambda label: (-float(smoothed_objectives.get(label, float("-inf"))), label),
    )
    selected_labels = ranked_survivors[: config.top_k]

    return EnhancedSelectionResult(
        selected_labels=selected_labels,
        trade_frequencies=trade_frequencies,
    )
