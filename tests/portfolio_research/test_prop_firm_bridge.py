"""Tests for portfolio -> prop_firm bridge."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from quantfoundry_core.prop_firm import PortfolioSimulationConfig, ReturnEngineConfig

from dataclasses import replace

from research.portfolio.config import (
    UNLIMITED_FUNDED_ACCOUNT_CAP,
    PropFirmReportConfig,
    load_config,
)
from research.portfolio.pipelines.portfolio_test import PhaseResult
from research.portfolio.prop_firm_bridge import (
    align_portfolio_returns_with_report_engine,
    build_prop_firm_returns,
    create_prop_firm_portfolio_simulator,
    portfolio_simulation_config,
    return_engine_for_research,
)


def _minimal_portfolio_config():
    return load_config()


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


def test_return_engine_for_research_uses_train_through_test() -> None:
    cfg = _minimal_portfolio_config()
    report_cfg = PropFirmReportConfig()
    engine = return_engine_for_research(report_cfg, cfg)
    assert engine.start_date == cfg.train_window.start.strftime("%Y-%m-%d")
    assert engine.end_date == cfg.test_window.end.strftime("%Y-%m-%d")


def test_align_portfolio_returns_with_report_engine_date_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.portfolio.prop_firm_bridge as mod

    raw = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        name="p",
    )
    portfolio_cfg = _minimal_portfolio_config()
    report_cfg = PropFirmReportConfig(return_target_annual_volatility=None)

    def _narrow_engine(
        phase_name: str,
        report: PropFirmReportConfig,
        portfolio: object,
    ) -> ReturnEngineConfig:
        _ = phase_name
        base = return_engine_for_research(report, portfolio)
        return ReturnEngineConfig(
            target_annual_volatility=base.target_annual_volatility,
            target_sharpe=base.target_sharpe,
            annualization_factor=base.annualization_factor,
            start_date="2020-01-03",
            end_date="2020-01-03",
            random_seed=base.random_seed,
        )

    monkeypatch.setattr(mod, "return_engine_for_phase", _narrow_engine)
    out = align_portfolio_returns_with_report_engine(
        raw, "validation", report_cfg, portfolio_cfg
    )
    assert len(out) == 1
    assert float(out.iloc[0]) == pytest.approx(0.02)


def test_create_prop_firm_portfolio_simulator_fundednext() -> None:
    sim = create_prop_firm_portfolio_simulator("fundednext")
    assert hasattr(sim, "simulate")


def test_portfolio_simulation_config_builds_qf_config() -> None:
    portfolio_cfg = _minimal_portfolio_config()
    report_cfg = PropFirmReportConfig()
    sim_cfg = portfolio_simulation_config("validation", report_cfg, portfolio_cfg)
    assert isinstance(sim_cfg, PortfolioSimulationConfig)
    assert sim_cfg.account_code == "50000"
    assert sim_cfg.purchase_policy.funded_account_cap == UNLIMITED_FUNDED_ACCOUNT_CAP

    preset_cap = portfolio_simulation_config(
        "validation",
        replace(report_cfg, funded_account_cap=None),
        portfolio_cfg,
    )
    assert preset_cap.purchase_policy.funded_account_cap is None
