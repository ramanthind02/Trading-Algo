"""Tests for timeframe-aware holdout monitoring thresholds."""

from __future__ import annotations

from portfolio_research.config import load_config
from portfolio_research.holdout.strategy_monitoring import (
    _ensemble_trading_timeframe,
    _min_evaluation_bars_for_strategy,
)
from utils.core.enums import TimeFrame


def test_buy_hold_ensemble_maps_to_monthly_timeframe() -> None:
    config = load_config()
    assert _ensemble_trading_timeframe(config, "buy_hold_long") is TimeFrame.M


def test_monthly_strategy_uses_lower_min_evaluation_bars() -> None:
    config = load_config()
    assert _min_evaluation_bars_for_strategy(config, "buy_hold_long") == 6
    assert _min_evaluation_bars_for_strategy(config, "calendar_ensemble_long") == 60
