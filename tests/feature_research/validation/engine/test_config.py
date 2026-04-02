from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

from utils.evaluation.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
    WeightLayerAlgorithm,
)


def test_defaults_are_deterministic() -> None:
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2021, 1, 1),
    )

    assert config.enabled is False
    assert config.test_step == 730
    assert config.num_steps == 4
    assert config.top_k == 5
    assert config.objective_metric_name == "t_stat"
    assert config.selection_method == WalkforwardSelectionMethod.TOP_K
    assert config.min_fold_samples == 10
    assert config.output_root == Path("feature_research/shared_results")


def test_config_is_frozen() -> None:
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2021, 1, 1),
    )

    with pytest.raises(FrozenInstanceError):
        config.top_k = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"train_end": datetime(2020, 1, 1)}, "train_end"),
        ({"test_step": 0}, "test_step"),
        ({"num_steps": 0}, "num_steps"),
        ({"top_k": 0}, "top_k"),
        ({"objective_metric_name": "SHARPE"}, "objective_metric_name"),
        ({"objective_metric_name": "omega"}, "objective_metric_name"),
        ({"min_fold_samples": 9}, "min_fold_samples"),
        ({"output_root": Path("")}, "output_root"),
        ({"output_root": "feature_research/shared_results"}, "output_root"),
    ],
)
def test_validation_bounds(kwargs: dict[str, object], expected_message: str) -> None:
    all_kwargs = {
        "train_start": datetime(2020, 1, 1),
        "train_end": datetime(2021, 1, 1),
        **kwargs,
    }
    with pytest.raises(ValueError, match=expected_message):
        WalkforwardResearchConfig(**all_kwargs)


def test_config_selection_method_defaults() -> None:
    config = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2015, 1, 1),
    )

    assert config.selection_method == WalkforwardSelectionMethod.TOP_K
    assert config.trade_freq_min == pytest.approx(0.01)


def test_config_rejects_invalid_trade_freq_min() -> None:
    with pytest.raises(ValueError, match="trade_freq_min"):
        WalkforwardResearchConfig(
            train_start=datetime(2000, 1, 1),
            train_end=datetime(2015, 1, 1),
            trade_freq_min=1.5,
        )


@pytest.mark.parametrize(
    ("selection_method", "expected"),
    [
        (WalkforwardSelectionMethod.ENHANCED, WalkforwardSelectionMethod.ENHANCED),
        ("top_k", WalkforwardSelectionMethod.TOP_K),
    ],
)
def test_selection_method_accepts_enum_or_enum_coercible_string(
    selection_method: WalkforwardSelectionMethod | str,
    expected: WalkforwardSelectionMethod,
) -> None:
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2021, 1, 1),
        selection_method=selection_method,
    )

    assert config.selection_method == expected


@pytest.mark.parametrize(
    ("weight_layer_algorithm", "expected"),
    [
        (
            WeightLayerAlgorithm.HRP_CLASSIC,
            WeightLayerAlgorithm.HRP_CLASSIC,
        ),
        ("equal_signal", WeightLayerAlgorithm.EQUAL_SIGNAL),
    ],
)
def test_weight_layer_algorithm_accepts_enum_or_enum_coercible_string(
    weight_layer_algorithm: WeightLayerAlgorithm | str,
    expected: WeightLayerAlgorithm,
) -> None:
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2021, 1, 1),
        weight_layer_algorithm=weight_layer_algorithm,
    )

    assert config.weight_layer_algorithm == expected


def test_selection_method_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="selection_method must be one of"):
        WalkforwardResearchConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2021, 1, 1),
            selection_method="not_a_real_method",
        )


def test_weight_layer_algorithm_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="weight_layer_algorithm must be one of"):
        WalkforwardResearchConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2021, 1, 1),
            weight_layer_algorithm="not_a_real_algorithm",
        )
