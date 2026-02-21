from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_research.walkforward.config import WalkforwardResearchConfig


def test_defaults_are_deterministic() -> None:
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2021, 1, 1),
    )

    assert config.enabled is False
    assert config.test_step == 365
    assert config.num_steps == 8
    assert config.top_k == 3
    assert config.objective_metric_name == "sortino"
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


def test_config_enhanced_selection_defaults() -> None:
    config = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2015, 1, 1),
    )

    assert config.use_enhanced_selection is False
    assert config.trade_freq_min == pytest.approx(0.05)


def test_config_rejects_invalid_trade_freq_min() -> None:
    with pytest.raises(ValueError, match="trade_freq_min"):
        WalkforwardResearchConfig(
            train_start=datetime(2000, 1, 1),
            train_end=datetime(2015, 1, 1),
            trade_freq_min=1.5,
        )
