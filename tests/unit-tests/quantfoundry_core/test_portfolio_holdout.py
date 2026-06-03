from __future__ import annotations

import math

import numpy as np
import pandas as pd
from quantfoundry_core.robustness.portfolio_holdout import (
    correlation_realisation_test,
    drawdown_correlation_realisation,
    effective_rank,
)


def test_effective_rank_positive_for_identity() -> None:
    identity = np.eye(4)
    assert effective_rank(identity) == 4.0


def test_effective_rank_short_holdout_singular_matrix() -> None:
    """Short holdout windows can yield ill-conditioned sample correlation matrices."""

    rng = np.random.default_rng(1)
    holdout = pd.DataFrame(
        {f"s{i}": rng.normal(0, 0.01, 40) for i in range(12)}
    )
    holdout["flat"] = 0.0
    corr = holdout.corr(min_periods=2).to_numpy(dtype=np.float64)
    rank = effective_rank(corr)
    assert math.isfinite(rank)
    assert rank >= 1.0


def test_correlation_realisation_short_holdout_window() -> None:
    rng = np.random.default_rng(2)
    is_frame = pd.DataFrame(
        {f"s{i}": rng.normal(0, 0.01, 500) for i in range(8)}
    )
    holdout_frame = pd.DataFrame(
        {f"s{i}": rng.normal(0, 0.01, 45) for i in range(8)}
    )
    result = correlation_realisation_test(is_frame, holdout_frame)
    assert math.isfinite(result.eff_rank_holdout)
    assert result.C_holdout.shape == result.C_is.shape
    assert len(result.strategy_names) == result.C_is.shape[0]


def test_correlation_realisation_detects_shift() -> None:
    rng = np.random.default_rng(0)
    is_frame = pd.DataFrame(
        {
            "a": rng.normal(0, 0.01, 100),
            "b": rng.normal(0, 0.01, 100),
        }
    )
    holdout_frame = is_frame.copy()
    holdout_frame["a"] = holdout_frame["b"]
    result = correlation_realisation_test(is_frame, holdout_frame)
    assert result.avg_corr_holdout >= result.avg_corr_is
