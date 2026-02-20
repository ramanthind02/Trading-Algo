from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from feature_research.walkforward.config import WalkforwardResearchConfig


def compute_trade_frequency(signal: pd.Series) -> float:
    if signal.empty:
        return 0.0
    return float((signal != 0).sum() / len(signal))


def compute_robustness_score(block_metrics: list[float]) -> float:
    if not block_metrics:
        return 0.0

    metrics = np.array(block_metrics, dtype=float)
    mean_value = float(np.mean(metrics))
    if mean_value <= 0.0:
        return 0.0
    if len(metrics) == 1:
        return mean_value

    std_value = float(np.std(metrics, ddof=1))
    coefficient_of_variation = std_value / abs(mean_value)
    return mean_value / (1.0 + coefficient_of_variation)


def compute_signal_correlation_matrix(signal_series: dict[str, pd.Series]) -> pd.DataFrame:
    labels = list(signal_series)
    if not labels:
        return pd.DataFrame()

    signal_frame = pd.DataFrame({label: signal_series[label] for label in labels})
    return signal_frame.corr(method="pearson")


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}

    numeric_values = np.array(list(values.values()), dtype=float)
    min_value = float(np.min(numeric_values))
    max_value = float(np.max(numeric_values))
    if min_value == max_value:
        return {label: 0.0 for label in values}

    value_range = max_value - min_value
    return {
        label: float((value - min_value) / value_range)
        for label, value in values.items()
    }


def compute_quality_scores(
    smoothed_objectives: dict[str, float],
    robustness_scores: dict[str, float],
    weight_stability: float = 0.5,
    weight_robustness: float = 0.5,
) -> dict[str, float]:
    labels = list(smoothed_objectives)
    normalized_stability = _normalize(smoothed_objectives)
    normalized_robustness = _normalize(robustness_scores)

    return {
        label: (
            weight_stability * normalized_stability.get(label, 0.0)
            + weight_robustness * normalized_robustness.get(label, 0.0)
        )
        for label in labels
    }


def greedy_diversity_select(
    quality_scores: dict[str, float],
    corr_matrix: pd.DataFrame,
    k: int,
    diversity_weight: float = 0.4,
) -> list[str]:
    if not quality_scores:
        return []

    remaining = sorted(quality_scores, key=quality_scores.__getitem__, reverse=True)
    selected: list[str] = [remaining.pop(0)]

    while len(selected) < k and remaining:
        best_candidate: str | None = None
        best_adjusted_score = float("-inf")

        for candidate in remaining:
            if candidate in corr_matrix.index:
                max_abs_corr = max(
                    (
                        abs(float(corr_matrix.loc[candidate, chosen]))
                        for chosen in selected
                        if chosen in corr_matrix.columns
                    ),
                    default=0.0,
                )
            else:
                max_abs_corr = 0.0

            adjusted_score = quality_scores[candidate] * (
                1.0 - diversity_weight * max_abs_corr
            )
            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def _canonical_param_label(params: dict[str, object]) -> str:
    return "|".join(f"{key}={params[key]}" for key in sorted(params))


@dataclass(frozen=True)
class EnhancedSelectionResult:
    selected_labels: list[str]
    trade_frequencies: dict[str, float]
    robustness_scores: dict[str, float]
    quality_scores: dict[str, float]
    corr_matrix: pd.DataFrame


def run_enhanced_selection(
    training_data: pd.DataFrame,
    training_target: pd.Series,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series],
    smoothed_objectives: dict[str, float],
    objective_metric: Callable[[pd.Series], float],
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
            robustness_scores={},
            quality_scores={},
            corr_matrix=pd.DataFrame(),
        )

    block_count = max(1, int(config.n_robustness_blocks))
    index_blocks = [
        block_index
        for block_index in np.array_split(training_data.index, block_count)
        if len(block_index) > 0
    ]
    robustness_scores = {
        label: compute_robustness_score(
            [
                float(
                    objective_metric(
                        evaluate_param_combo(
                            training_data.loc[block_index],
                            training_target.loc[block_index],
                            param_label_map[label],
                        )
                    )
                )
                for block_index in index_blocks
            ]
        )
        for label in surviving_labels
    }

    robust_labels = [label for label in surviving_labels if robustness_scores[label] > 0.0]
    if not robust_labels:
        return EnhancedSelectionResult(
            selected_labels=[],
            trade_frequencies=trade_frequencies,
            robustness_scores=robustness_scores,
            quality_scores={},
            corr_matrix=pd.DataFrame(),
        )

    quality_scores = compute_quality_scores(
        smoothed_objectives={label: smoothed_objectives.get(label, 0.0) for label in robust_labels},
        robustness_scores={label: robustness_scores[label] for label in robust_labels},
        weight_stability=config.weight_stability,
        weight_robustness=config.weight_robustness,
    )
    corr_matrix = compute_signal_correlation_matrix(
        {label: signal_series[label] for label in robust_labels}
    )
    selected_labels = greedy_diversity_select(
        quality_scores=quality_scores,
        corr_matrix=corr_matrix,
        k=config.top_k,
        diversity_weight=config.diversity_weight,
    )

    return EnhancedSelectionResult(
        selected_labels=selected_labels,
        trade_frequencies=trade_frequencies,
        robustness_scores=robustness_scores,
        quality_scores=quality_scores,
        corr_matrix=corr_matrix,
    )
