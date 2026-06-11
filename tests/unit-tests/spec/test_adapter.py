"""Unit tests for the StrategySpec → pipeline-config adapter.

These assert the **field mapping** of the fresh config objects the adapter builds; they do
NOT run the pipeline. The Nautilus engine mapping is guarded behind ``importorskip`` because
constructing the engine pulls in ``nautilus_trader``.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from research.feature.config import FeatureType
from research.portfolio.config import EnsembleDirsPolicy
from research.spec.adapter import (
    apply_vol_scaling,
    build_bias_spec,
    default_windows,
    ewsd_blend_for,
    exploration_feed_for,
    feed_literal,
    to_feature_config,
    to_portfolio_config,
)
from research.spec.strategy_spec import (
    DataFeed,
    Direction,
    Holding,
    OrderPolicy,
    SignalSpec,
    StrategyMode,
    StrategySpec,
    Ticker,
    TimeFrame,
    UnfilledLimitPolicy,
    VaultTarget,
    VolScaling,
)
from research.spec.strategy_spec import ExecutionSpec, RiskSpec, AccountSpec


def _make_spec(**overrides) -> StrategySpec:
    base = dict(
        name="daily_rsi_mr",
        hypothesis="RSI(2) mean reversion.",
        author="tester",
        created=datetime(2026, 1, 1),
        tickers=(Ticker.ES, Ticker.NQ),
        data_feed=DataFeed.DARWINEX_CFD,
        mode=StrategyMode.DAILY,
        timeframe=TimeFrame.D,
        signal=SignalSpec("rsi_signal", {"rsi_period": [2, 3], "oversold": [20, 25]}),
        direction=Direction.LONG_SHORT,
        risk=RiskSpec(target_vol=0.20, max_position_pct=4.0),
        vault=VaultTarget("mean_reversion_indices", "rsi_mr"),
    )
    base.update(overrides)
    return StrategySpec(**base)


# ── windows / small mappers ─────────────────────────────────────────────────


def test_default_windows_daily_ordering():
    w = default_windows(StrategyMode.DAILY)
    assert w.train[0] == datetime(2000, 1, 1)
    assert w.train[1] < w.validation[0] < w.validation[1] < w.test[0]


def test_default_windows_intraday_start():
    w = default_windows(StrategyMode.INTRADAY)
    assert w.train[0] == datetime(2018, 1, 1)
    assert w.test[0] == datetime(2025, 1, 1)


def test_feed_literal_mapping():
    assert feed_literal(DataFeed.DARWINEX_CFD) == "cfd"
    assert feed_literal(DataFeed.NORGATE_FUTURES) == "futures"


def test_exploration_feed_for_routes_by_asset_class():
    # DAILY futures-backed tickers (a ratio parquet exists) → faithful ratio futures.
    assert exploration_feed_for(_make_spec(tickers=(Ticker.ES, Ticker.NQ))) == "futures_ratio"
    # Forex / pure-CFD (no ratio parquet — no contract roll) → CFD.
    assert exploration_feed_for(_make_spec(tickers=(Ticker.AUDNZD,))) == "cfd"
    # Mixed: any ticker lacking a ratio series forces the whole run onto CFD.
    assert exploration_feed_for(_make_spec(tickers=(Ticker.ES, Ticker.AUDNZD))) == "cfd"
    # Intraday → CFD regardless (no intraday ratio series exists).
    intraday = _make_spec(
        tickers=(Ticker.ES,), mode=StrategyMode.INTRADAY, timeframe=TimeFrame.H1
    )
    assert exploration_feed_for(intraday) == "cfd"


def test_build_bias_spec_shape():
    spec = _make_spec()
    bias = build_bias_spec(spec)
    assert bias["module_name"] == "rsi_signal"
    assert bias["timeframes"] == [TimeFrame.D]
    assert bias["params"]["rsi_period"] == [2, 3]
    assert bias["params"]["oversold"] == [20, 25]


# ── feature config ──────────────────────────────────────────────────────────


def test_to_feature_config_maps_core_fields():
    spec = _make_spec()
    cfg = to_feature_config(spec)
    assert cfg.tickers == [Ticker.ES, Ticker.NQ]
    assert cfg.timeframe is TimeFrame.D
    assert cfg.feature_type is FeatureType.SIGNED_SIGNAL
    # config.data_feed is the realistic/validation feed = post-2018 CFD (the spec's data_feed is
    # an ignored label). Vectorized exploration overrides to the ratio futures feed at run time.
    assert cfg.data_feed == "cfd"
    # windows → ResearchWindowConfig (train + validation only; no project test holdout)
    assert cfg.research_window is not None
    assert cfg.research_window.train_start == datetime(2000, 1, 1)
    assert cfg.research_window.val_end == datetime(2022, 12, 31)


def test_to_feature_config_threads_bias_spec_and_direction():
    spec = _make_spec()
    cfg = to_feature_config(spec)
    phase = cfg.in_sample_defaults.signed_signal
    assert phase.bias_spec["module_name"] == "rsi_signal"
    assert phase.bias_spec["params"]["rsi_period"] == [2, 3]
    assert phase.strategy is Direction.LONG_SHORT


def test_to_feature_config_eval_spec_is_single_center_combo():
    # 5 periods x 5 std_devs x 1 mode = 25 combos; eval must collapse to the center scalar.
    spec = _make_spec(
        signal=SignalSpec(
            "percent_b_signal",
            {"period": [10, 15, 20, 25, 30], "std_dev": [2.0, 2.25, 2.5, 2.75, 3.0]},
        )
    )
    cfg = to_feature_config(spec)
    eval_params = cfg.eval_bias_spec["params"]
    assert eval_params["period"] == 20  # center of the 5-element grid
    assert eval_params["std_dev"] == 2.5
    # scalar, not a list (Pass-1 decile analysis requires exactly one combo)
    assert not isinstance(eval_params["period"], list)


def test_to_feature_config_sets_vault_save_target():
    spec = _make_spec()
    cfg = to_feature_config(spec)
    assert cfg.vault_save is not None
    assert cfg.vault_save.weight_hierarchy_group == "mean_reversion_indices"
    assert cfg.vault_save.ensemble_name == "rsi_mr"
    assert cfg.vault_save.direction is Direction.LONG_SHORT


def test_to_feature_config_data_feed_is_validation_cfd():
    # config.data_feed is the realistic/validation feed (post-2018 CFD), independent of the spec's
    # data_feed label — there is no per-spec feed selection. Exploration uses the ratio futures
    # feed (set by the executor at run time), not config.data_feed.
    assert to_feature_config(_make_spec(data_feed=DataFeed.NORGATE_FUTURES)).data_feed == "cfd"
    assert to_feature_config(_make_spec(data_feed=DataFeed.DARWINEX_CFD)).data_feed == "cfd"


def test_to_feature_config_default_execution_matches_today_validation_defaults():
    # Default ExecutionSpec → the proven validation lane: overnight (rollover overlay) +
    # market-on-open + cross_after. These config defaults reproduce today's hard-coded
    # behavior, so the parity gate / canonical configs are unaffected.
    cfg = to_feature_config(_make_spec())
    assert cfg.execution_entry_policy == "market_on_open"
    assert cfg.execution_unfilled_limit == "cross_after"
    assert cfg.execution_holding == "overnight"


def test_to_feature_config_threads_spec_execution_policy():
    spec = _make_spec(
        mode=StrategyMode.INTRADAY,
        timeframe=TimeFrame.H1,
        execution=ExecutionSpec(
            entry_policy=OrderPolicy.LIMIT_AT_TOUCH,
            holding=Holding.INTRADAY,
            unfilled_limit=UnfilledLimitPolicy.CARRY,
        ),
    )
    cfg = to_feature_config(spec)
    assert cfg.execution_entry_policy == "limit_at_touch"
    assert cfg.execution_unfilled_limit == "carry"
    assert cfg.execution_holding == "intraday"


# ── portfolio config ────────────────────────────────────────────────────────


def test_to_portfolio_config_maps_core_fields():
    spec = _make_spec()
    cfg = to_portfolio_config(spec)
    assert cfg.tickers == [Ticker.ES, Ticker.NQ]
    assert cfg.timeframe is TimeFrame.D
    # Portfolio-addition (realistic) scoring uses post-2018 CFD; data_feed is "cfd" regardless of
    # the spec label (DARWINEX_CFD here).
    assert cfg.data_feed == "cfd"
    assert cfg.target_volatility == pytest.approx(0.20)
    assert cfg.max_position_pct == pytest.approx(4.0)
    # empty ensemble_dirs must be allowed (candidate wired at run time)
    assert cfg.ensemble_dirs_policy is EnsembleDirsPolicy.ALLOW_EMPTY
    assert dict(cfg.ensemble_dirs) == {}


def test_to_portfolio_config_window_bounds_locked_test():
    spec = _make_spec()
    cfg = to_portfolio_config(spec)
    assert cfg.train_window.start == datetime(2000, 1, 1)
    # DAILY validation now starts at the 2018-01-01 ratio→CFD cutover (was 2019).
    assert cfg.validation_window.start == datetime(2018, 1, 1)
    assert cfg.test_window.start == datetime(2023, 1, 1)


def test_to_portfolio_config_respects_spec_windows_override():
    windows = default_windows(StrategyMode.DAILY)
    spec = _make_spec(windows=windows)
    cfg = to_portfolio_config(spec)
    assert cfg.test_window.end == windows.test[1]


# ── pnl engine (heavy — guarded) ────────────────────────────────────────────


def test_ewsd_blend_for_modes():
    assert ewsd_blend_for(VolScaling.BLENDED) == (0.7, 0.3)
    assert ewsd_blend_for(VolScaling.LONG_ONLY) == (0.0, 1.0)
    assert ewsd_blend_for(VolScaling.OFF) is None


def test_apply_vol_scaling_sets_and_resets_global_blend():
    from lib.compute.daily_ewsd_volatility import (
        DailyEWSDVolatilityService,
        reset_ewsd_blend_weights,
    )

    try:
        apply_vol_scaling(_make_spec(vol_scaling=VolScaling.LONG_ONLY))
        cfg = DailyEWSDVolatilityService().config
        assert (cfg.blend_short_weight, cfg.blend_long_weight) == (0.0, 1.0)
    finally:
        reset_ewsd_blend_weights()
    cfg = DailyEWSDVolatilityService().config
    assert (cfg.blend_short_weight, cfg.blend_long_weight) == (0.7, 0.3)


def test_apply_vol_scaling_off_sets_bypass_flag():
    from lib.compute.daily_ewsd_volatility import is_vol_scaling_off, set_vol_scaling_off

    try:
        apply_vol_scaling(_make_spec(vol_scaling=VolScaling.OFF))
        assert is_vol_scaling_off()
    finally:
        set_vol_scaling_off(False)
    assert not is_vol_scaling_off()


def test_to_pnl_engine_overnight_market_mapping():
    pytest.importorskip("nautilus_trader")
    from research.portfolio.pnl.nautilus_engine import (
        ExecutionPolicy,
        ExecutionWindowPolicy,
    )
    from research.spec.adapter import to_pnl_engine

    spec = _make_spec(account=AccountSpec(capital=75_000.0))
    engine = to_pnl_engine(spec)
    assert engine.window_policy is ExecutionWindowPolicy.CLOSE_TO_CLOSE
    assert engine.execution_policy is ExecutionPolicy.MARKET_ON_OPEN
    assert engine.capital == pytest.approx(75_000.0)


def test_to_pnl_engine_intraday_limit_and_carry_mapping():
    pytest.importorskip("nautilus_trader")
    from research.portfolio.pnl.nautilus_engine import (
        ExecutionPolicy,
        ExecutionWindowPolicy,
    )
    from research.spec.adapter import to_pnl_engine

    spec = _make_spec(
        mode=StrategyMode.INTRADAY,
        timeframe=TimeFrame.H1,
        execution=ExecutionSpec(
            entry_policy=OrderPolicy.LIMIT_AT_TOUCH,
            holding=Holding.INTRADAY,
            unfilled_limit=UnfilledLimitPolicy.CARRY,
        ),
    )
    engine = to_pnl_engine(spec)
    assert engine.window_policy is ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE
    assert engine.execution_policy is ExecutionPolicy.LIMIT_AT_TOUCH
    # CARRY → pure passive (never cross)
    assert engine.cross_after.session_fraction == pytest.approx(1.0)
    assert not engine.cross_after.is_enabled()
