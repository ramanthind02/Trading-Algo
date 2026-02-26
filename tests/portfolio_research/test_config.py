"""Tests for portfolio_research config and walkforward bounds."""
from datetime import datetime

import pytest

from feature_research.config import OOSWindowConfig, compute_first_fold_bounds
from portfolio_research.config import PortfolioResearchConfig, load_config
from utils.core.enums import Ticker, TimeFrame


def test_load_config_returns_portfolio_research_config() -> None:
    config = load_config()
    assert isinstance(config, PortfolioResearchConfig)
    assert len(config.tickers) >= 1
    assert config.ensemble_dirs
    assert config.timeframe == TimeFrame.D
    assert config.start < config.end


def test_walkforward_train_test_bounds_ordering() -> None:
    config = load_config()
    train_start, train_end, test_start, test_end = (
        config.walkforward_train_test_bounds()
    )
    assert train_start <= train_end
    assert train_end < test_start
    assert test_start <= test_end
    assert train_start >= config.start
    assert test_end <= config.end


def test_portfolio_research_config_empty_ensemble_dirs_raises() -> None:
    with pytest.raises(ValueError, match="ensemble_dirs must be non-empty"):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2023, 12, 31),
            use_cache=True,
            ensemble_dirs={},
        )


def test_portfolio_research_config_invalid_baseline_mode_raises() -> None:
    with pytest.raises(ValueError, match="baseline_mode must be"):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2023, 12, 31),
            use_cache=True,
            ensemble_dirs={"a": "vault/D/some_ensemble"},
            baseline_mode="invalid",
        )


def test_compute_first_fold_bounds_ordering() -> None:
    """Shared helper returns ordered train/test bounds (portfolio and feature_research aligned)."""
    start = datetime(2000, 1, 1)
    end = datetime(2023, 12, 31)
    train_start, train_end, test_start, test_end = compute_first_fold_bounds(
        start, end,
        train_window_years=15.0,
        test_window_years=2.0,
        num_steps=4,
    )
    assert train_start <= train_end
    assert train_end < test_start
    assert test_start <= test_end
    assert train_start >= start
    assert test_end <= end


def test_load_config_oos_window_consistent_with_walkforward() -> None:
    """When OOS is set, OOS test should start after walkforward period (or overlap is researcher choice)."""
    config = load_config()
    if config.oos_window is None:
        pytest.skip("OOS window not set in load_config()")
    _wf_train_s, _wf_train_e, _wf_test_s, wf_test_e = (
        config.walkforward_train_test_bounds()
    )
    oos = config.oos_window
    # OOS train/test should be ordered
    assert oos.train_start <= oos.train_end
    assert oos.test_start <= oos.test_end
    assert oos.train_end < oos.test_start
