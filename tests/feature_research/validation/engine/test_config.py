from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

from utils.evaluation.walkforward.config import (
    WalkforwardResearchConfig,
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
    assert config.objective_metric_name == "t_stat"
    assert config.min_fold_samples == 10
    assert config.output_root == Path("feature_research/shared_results")


def test_config_is_frozen() -> None:
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2021, 1, 1),
    )

    with pytest.raises(FrozenInstanceError):
        config.num_steps = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"train_end": datetime(2020, 1, 1)}, "train_end"),
        ({"test_step": 0}, "test_step"),
        ({"num_steps": 0}, "num_steps"),
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


@pytest.mark.parametrize(
    ("weight_layer_algorithm", "expected"),
    [
        (
            WeightLayerAlgorithm.HIERARCHY_EQUAL,
            WeightLayerAlgorithm.HIERARCHY_EQUAL,
        ),
        ("equal_signal", WeightLayerAlgorithm.EQUAL_SIGNAL),
        (
            "inverse_avg_pairwise_corr",
            WeightLayerAlgorithm.INVERSE_AVG_PAIRWISE_CORR,
        ),
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


def test_weight_layer_algorithm_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="weight_layer_algorithm must be one of"):
        WalkforwardResearchConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2021, 1, 1),
            weight_layer_algorithm="not_a_real_algorithm",
        )
