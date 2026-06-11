"""Unit tests for the StrategySpec object and its construction-time validation.

Dependency-light: the spec module imports only the domain enums + vault constants, so these
tests never touch the heavy research/pipeline configs.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from research.spec.strategy_spec import (
    MAX_GRID_COMBOS,
    AccountSpec,
    DataFeed,
    Direction,
    ExecutionSpec,
    FillFeed,
    Holding,
    OrderPolicy,
    ResearchWindows,
    RiskSpec,
    SignalSpec,
    StrategyMode,
    StrategySpec,
    Ticker,
    TimeFrame,
    UnfilledLimitPolicy,
    VaultTarget,
    combo_count,
)


def _make_spec(**overrides) -> StrategySpec:
    """A minimal valid DAILY spec; ``overrides`` replaces top-level fields."""

    base = dict(
        name="daily_rsi_mr",
        hypothesis="RSI(2) mean reversion on equity indices.",
        author="tester",
        created=datetime(2026, 1, 1),
        tickers=(Ticker.ES,),
        data_feed=DataFeed.DARWINEX_CFD,
        mode=StrategyMode.DAILY,
        timeframe=TimeFrame.D,
        signal=SignalSpec("rsi_signal", {"rsi_period": [2, 3], "oversold": [20, 25]}),
        direction=Direction.LONG_SHORT,
        vault=VaultTarget("mean_reversion_indices", "rsi_mr"),
    )
    base.update(overrides)
    return StrategySpec(**base)


# ── combo_count / grid cap ──────────────────────────────────────────────────


def test_combo_count_is_product_of_list_lengths():
    assert combo_count({"a": [1, 2, 3], "b": [4, 5]}) == 6


def test_signal_spec_at_cap_is_allowed():
    grid = {"a": list(range(15)), "b": list(range(20))}  # 300 exactly
    spec = SignalSpec("x", grid)
    assert spec.num_combos == MAX_GRID_COMBOS


def test_signal_spec_over_cap_raises():
    grid = {"a": list(range(20)), "b": list(range(20))}  # 400
    with pytest.raises(ValueError, match="exceeding the cap"):
        SignalSpec("x", grid)


def test_signal_spec_rejects_non_list_param():
    with pytest.raises(ValueError, match="list/tuple"):
        SignalSpec("x", {"period": 14})  # scalar, not a 1-element list


def test_signal_spec_rejects_empty_param_list():
    with pytest.raises(ValueError, match="non-empty list"):
        SignalSpec("x", {"period": []})


def test_signal_spec_rejects_empty_grid():
    with pytest.raises(ValueError, match="at least one parameter"):
        SignalSpec("x", {})


# ── windows ─────────────────────────────────────────────────────────────────


def test_windows_valid():
    w = ResearchWindows(
        train=(datetime(2000, 1, 1), datetime(2018, 12, 31)),
        validation=(datetime(2019, 1, 1), datetime(2022, 12, 31)),
        test=(datetime(2023, 1, 1), datetime(2026, 1, 1)),
    )
    assert w.train[1] < w.validation[0]


def test_windows_train_overlaps_validation_raises():
    with pytest.raises(ValueError, match="train must end before validation"):
        ResearchWindows(
            train=(datetime(2000, 1, 1), datetime(2019, 6, 1)),
            validation=(datetime(2019, 1, 1), datetime(2022, 12, 31)),
            test=(datetime(2023, 1, 1), datetime(2026, 1, 1)),
        )


def test_windows_start_after_end_raises():
    with pytest.raises(ValueError, match="start must be before end"):
        ResearchWindows(
            train=(datetime(2018, 12, 31), datetime(2000, 1, 1)),
            validation=(datetime(2019, 1, 1), datetime(2022, 12, 31)),
            test=(datetime(2023, 1, 1), datetime(2026, 1, 1)),
        )


# ── vault target ────────────────────────────────────────────────────────────


def test_vault_target_rejects_unknown_sleeve():
    with pytest.raises(ValueError, match="not a configured sleeve"):
        VaultTarget("not_a_real_sleeve", "x")


def test_vault_target_rejects_blank_ensemble_name():
    with pytest.raises(ValueError, match="ensemble_name"):
        VaultTarget("momentum", "  ")


# ── risk ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("field,value", [("target_vol", 0.0), ("forecast_cap", -1.0), ("max_position_pct", 0.0)])
def test_risk_spec_bounds(field, value):
    with pytest.raises(ValueError):
        RiskSpec(**{field: value})


def test_account_spec_rejects_nonpositive_capital():
    with pytest.raises(ValueError, match="capital"):
        AccountSpec(capital=0.0)


# ── execution / fill feed ───────────────────────────────────────────────────


def test_fill_feed_derives_signal_bar_for_market_legs():
    ex = ExecutionSpec()  # both legs MARKET_ON_OPEN
    assert not ex.uses_limit()
    assert ex.resolved_fill_feed() is FillFeed.SIGNAL_BAR


def test_fill_feed_derives_fine_for_limit_leg():
    ex = ExecutionSpec(exit_policy=OrderPolicy.LIMIT_AT_TOUCH)
    assert ex.uses_limit()
    assert ex.resolved_fill_feed() is FillFeed.BARS_M1


def test_explicit_fill_feed_is_respected():
    ex = ExecutionSpec(entry_policy=OrderPolicy.LIMIT_IMPROVE, fill_feed=FillFeed.TICKS)
    assert ex.resolved_fill_feed() is FillFeed.TICKS


def test_limit_leg_with_signal_bar_feed_raises():
    with pytest.raises(ValueError, match="fine fill_feed"):
        ExecutionSpec(entry_policy=OrderPolicy.LIMIT_AT_TOUCH, fill_feed=FillFeed.SIGNAL_BAR)


def test_carry_unfilled_policy_constructs():
    ex = ExecutionSpec(unfilled_limit=UnfilledLimitPolicy.CARRY)
    assert ex.unfilled_limit is UnfilledLimitPolicy.CARRY


# ── StrategySpec ────────────────────────────────────────────────────────────


def test_valid_daily_spec_builds():
    spec = _make_spec()
    assert spec.windows is None  # adapter fills the default
    assert spec.vol_scaling.value == "blended"  # default
    assert spec.signal.num_combos == 4


def test_intraday_spec_with_h1_builds():
    spec = _make_spec(mode=StrategyMode.INTRADAY, timeframe=TimeFrame.H1)
    assert spec.timeframe is TimeFrame.H1


def test_daily_mode_with_intraday_timeframe_raises():
    with pytest.raises(ValueError, match="DAILY mode requires"):
        _make_spec(mode=StrategyMode.DAILY, timeframe=TimeFrame.H1)


def test_intraday_mode_with_daily_timeframe_raises():
    with pytest.raises(ValueError, match="INTRADAY mode requires"):
        _make_spec(mode=StrategyMode.INTRADAY, timeframe=TimeFrame.D)


def test_empty_tickers_raises():
    with pytest.raises(ValueError, match="at least one ticker"):
        _make_spec(tickers=())


def test_blank_name_raises():
    with pytest.raises(ValueError, match="name"):
        _make_spec(name="   ")


def test_spec_is_frozen():
    spec = _make_spec()
    with pytest.raises(Exception):
        spec.name = "mutated"  # type: ignore[misc]
