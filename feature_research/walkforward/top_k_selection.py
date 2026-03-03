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


def _extract_series_from_evaluator_result(
    result: pd.Series | tuple[pd.Series, object],
) -> pd.Series:
    """Unpack evaluator result: continuous returns (series, meta), rule-based returns series."""
    if isinstance(result, tuple):
        return result[0]
    return result


def compute_all_trade_frequencies(
    training_data: pd.DataFrame,
    training_target: pd.Series,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[
        [pd.DataFrame, pd.Series, dict[str, object]], pd.Series | tuple[pd.Series, object]
    ],
) -> dict[str, float]:
    """Evaluate every param combo on training data and return trade frequencies.

    This shared helper is used by both enhanced selection and stable region
    selection so the signal evaluation is performed only once.
    Handles both tuple (series, meta) from continuous evaluator and plain series from rule-based.

    Returns
    -------
    dict mapping canonical_param_label → trade_frequency
    """
    param_label_map = {_canonical_param_label(params): params for params in param_grid}
    return {
        label: compute_trade_frequency(
            _extract_series_from_evaluator_result(
                evaluate_param_combo(training_data, training_target, params)
            )
        )
        for label, params in param_label_map.items()
    }


@dataclass(frozen=True)
class EnhancedSelectionResult:
    selected_labels: list[str]
    trade_frequencies: dict[str, float]


def run_enhanced_selection(
    training_data: pd.DataFrame,
    training_target: pd.Series,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[
        [pd.DataFrame, pd.Series, dict[str, object]], pd.Series | tuple[pd.Series, object]
    ],
    smoothed_objectives: dict[str, float],
    config: "WalkforwardResearchConfig",
    precomputed_trade_frequencies: dict[str, float] | None = None,
    strategy: str | None = None,
) -> EnhancedSelectionResult:
    """Select top-k params filtered by trade frequency.

    Parameters
    ----------
    precomputed_trade_frequencies : dict, optional
        If provided, skip signal evaluation and use these frequencies directly.
        Allows sharing the evaluation cost with MPS or other selection.
    strategy : str, optional
        When "long" and objective is t_stat, only params with positive smoothed_objective are considered.
    """
    if precomputed_trade_frequencies is not None:
        trade_frequencies = precomputed_trade_frequencies
    else:
        trade_frequencies = compute_all_trade_frequencies(
            training_data, training_target, param_grid, evaluate_param_combo
        )

    param_labels = [_canonical_param_label(p) for p in param_grid]
    surviving_labels = [
        label for label in param_labels if trade_frequencies.get(label, 0.0) >= config.trade_freq_min
    ]
    # Long-only + t_stat: exclude params with non-positive objective so we never select a short-biased combo.
    if strategy == "long" and getattr(config, "objective_metric_name", "") == "t_stat":
        surviving_labels = [
            label for label in surviving_labels
            if smoothed_objectives.get(label, float("-inf")) > 0
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
