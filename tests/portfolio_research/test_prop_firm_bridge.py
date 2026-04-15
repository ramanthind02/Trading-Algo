"""Tests for portfolio -> prop_firm bridge."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from portfolio_research.pipelines.portfolio_test import PhaseResult
from portfolio_research.prop_firm_bridge import (
    align_portfolio_returns_with_report_engine,
    build_prop_firm_returns,
    run_prop_firm_simulation,
)


def test_build_prop_firm_returns_empty_phase_raises() -> None:
    phase = PhaseResult(
        name="Test",
        output_dir=Path("."),
        combined_strategy_returns=pd.Series(dtype=float, index=pd.DatetimeIndex([])),
        combined_baseline_returns=pd.Series(dtype=float, index=pd.DatetimeIndex([])),
    )
    with pytest.raises(ValueError, match="non-empty"):
        build_prop_firm_returns(phase)


def test_build_prop_firm_returns_produces_datetime_index_series() -> None:
    candles = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["ES", "ES"],
            "close": [100.0, 110.0],
        }
    )
    positions = pd.DataFrame(
        {
            "ticker": ["ES"],
            "datetime": pd.to_datetime(["2024-01-01"]),
            "position_fraction": [1.0],
        }
    )
    phase = PhaseResult(
        name="Test",
        output_dir=Path("."),
        combined_strategy_returns=pd.Series(dtype=float),
        combined_baseline_returns=pd.Series(dtype=float),
        daily_test_candles=candles,
        combined_positions=positions,
    )
    out = build_prop_firm_returns(phase)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == 1
    assert abs(float(out.iloc[0]) - 0.1) < 1e-12


def test_run_prop_firm_simulation_dispatches_lucid(monkeypatch: pytest.MonkeyPatch) -> None:
    import portfolio_research.prop_firm_bridge as mod

    captured: list[str] = []

    class FakeEngine:
        def simulate(self, request: object) -> object:
            captured.append(getattr(request, "account_code", ""))
            return Mock()

    monkeypatch.setattr(mod, "create_lucid_simulator", lambda: FakeEngine())
    returns = pd.Series([0.01], index=pd.DatetimeIndex(["2024-01-02"]))
    run_prop_firm_simulation(provider="lucid", account_code="LUCID_25K", returns=returns)
    assert captured == ["LUCID_25K"]


def test_align_portfolio_returns_with_report_engine_date_filter() -> None:
    from prop_firms.base.portfolio_models import ReturnEngineConfig

    raw = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        name="p",
    )
    cfg = ReturnEngineConfig(
        target_annual_volatility=None,
        target_sharpe=1.0,
        annualization_factor=252.0,
        start_date="2020-01-03",
        end_date="2020-01-03",
        random_seed=0,
    )
    out = align_portfolio_returns_with_report_engine(raw, cfg)
    assert len(out) == 1
    assert float(out.iloc[0]) == pytest.approx(0.02)


def test_run_prop_firm_simulation_dispatches_apex(monkeypatch: pytest.MonkeyPatch) -> None:
    import portfolio_research.prop_firm_bridge as mod

    captured: list[str] = []

    class FakeEngine:
        def simulate(self, request: object) -> object:
            captured.append(getattr(request, "account_code", ""))
            return Mock()

    monkeypatch.setattr(mod, "create_apex_simulator", lambda: FakeEngine())
    returns = pd.Series([0.01], index=pd.DatetimeIndex(["2024-01-02"]))
    run_prop_firm_simulation(provider="apex", account_code="APEX_50K", returns=returns)
    assert captured == ["APEX_50K"]
