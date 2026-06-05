"""Tests for integrated prop-firm report orchestration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from dataclasses import replace

from research.portfolio.config import PropFirmReportConfig, load_config
from research.portfolio.pipelines.portfolio_test import PhaseResult
from research.portfolio.prop_firm_reports import run_prop_firm_reports_for_phases


def _phase(name: str, out: Path) -> PhaseResult:
    return PhaseResult(
        name=name,
        output_dir=out,
        combined_strategy_returns=pd.Series(dtype=float),
        combined_baseline_returns=pd.Series(dtype=float),
        daily_test_candles=pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "ticker": ["ES", "ES"],
                "close": [100.0, 110.0],
            }
        ),
        combined_positions=pd.DataFrame(
            {
                "ticker": ["ES"],
                "datetime": pd.to_datetime(["2024-01-01"]),
                "position_fraction": [1.0],
            }
        ),
    )


def test_run_prop_firm_reports_disabled_returns_empty(tmp_path: Path) -> None:
    cfg = replace(
        load_config(),
        output_root=tmp_path,
        prop_firm_report=PropFirmReportConfig(enabled=False),
    )
    assert run_prop_firm_reports_for_phases(cfg, {"test": _phase("test", tmp_path / "test")}) == {}


def test_run_prop_firm_reports_for_phases_writes_artifacts(tmp_path: Path) -> None:
    phase_dir = tmp_path / "validation"
    phase_dir.mkdir(parents=True)
    cfg = replace(
        load_config(),
        output_root=tmp_path,
        prop_firm_report=PropFirmReportConfig(
            enabled=True,
            phases=("validation",),
            firm_id="fundednext",
        ),
    )
    fake_result = Mock()
    fake_artifacts = Mock(
        report_markdown_path=phase_dir / "prop_firm" / "fundednext" / "report.md",
        report_html_path=phase_dir / "prop_firm" / "fundednext" / "report.html",
    )

    with (
        patch(
            "research.portfolio.prop_firm_reports.build_prop_firm_returns",
            return_value=pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
        ),
        patch(
            "research.portfolio.prop_firm_reports.align_portfolio_returns_with_report_engine",
            return_value=pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
        ),
        patch(
            "research.portfolio.prop_firm_reports.create_prop_firm_portfolio_simulator",
        ) as mock_create,
        patch(
            "research.portfolio.prop_firm_reports.generate_portfolio_report",
            return_value=fake_artifacts,
        ) as mock_report,
    ):
        mock_sim = Mock()
        mock_sim.simulate.return_value = fake_result
        rolling_mock = Mock()
        rolling_mock.windows = [Mock(), Mock()]
        mock_sim.simulate_rolling.return_value = rolling_mock
        mock_create.return_value = mock_sim

        out = run_prop_firm_reports_for_phases(
            cfg,
            {"validation": _phase("validation", phase_dir)},
        )

    assert "validation" in out
    mock_sim.simulate.assert_called_once()
    mock_sim.simulate_rolling.assert_called_once()
    mock_report.assert_called_once()
    report_kwargs = mock_report.call_args.kwargs
    assert report_kwargs["output_dir"] == phase_dir / "prop_firm" / "fundednext"
    assert report_kwargs["rolling"] is rolling_mock


def test_run_prop_firm_reports_skips_rolling_on_value_error(tmp_path: Path) -> None:
    phase_dir = tmp_path / "test"
    phase_dir.mkdir(parents=True)
    cfg = replace(
        load_config(),
        output_root=tmp_path,
        prop_firm_report=PropFirmReportConfig(
            enabled=True,
            phases=("test",),
            rolling_enabled=True,
        ),
    )
    fake_artifacts = Mock(
        report_markdown_path=phase_dir / "report.md",
        report_html_path=phase_dir / "report.html",
    )

    with (
        patch(
            "research.portfolio.prop_firm_reports.build_prop_firm_returns",
            return_value=pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
        ),
        patch(
            "research.portfolio.prop_firm_reports.align_portfolio_returns_with_report_engine",
            return_value=pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
        ),
        patch(
            "research.portfolio.prop_firm_reports.create_prop_firm_portfolio_simulator",
        ) as mock_create,
        patch(
            "research.portfolio.prop_firm_reports.generate_portfolio_report",
            return_value=fake_artifacts,
        ) as mock_report,
    ):
        mock_sim = Mock()
        mock_sim.simulate.return_value = Mock()
        mock_sim.simulate_rolling.side_effect = ValueError("no complete windows")
        mock_create.return_value = mock_sim

        run_prop_firm_reports_for_phases(cfg, {"test": _phase("test", phase_dir)})

    mock_report.assert_called_once()
    assert mock_report.call_args.kwargs["rolling"] is None


def test_run_prop_firm_reports_rolling_disabled(tmp_path: Path) -> None:
    phase_dir = tmp_path / "test"
    phase_dir.mkdir(parents=True)
    cfg = replace(
        load_config(),
        output_root=tmp_path,
        prop_firm_report=PropFirmReportConfig(
            enabled=True,
            phases=("test",),
            rolling_enabled=False,
        ),
    )

    with (
        patch(
            "research.portfolio.prop_firm_reports.build_prop_firm_returns",
            return_value=pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
        ),
        patch(
            "research.portfolio.prop_firm_reports.align_portfolio_returns_with_report_engine",
            return_value=pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
        ),
        patch(
            "research.portfolio.prop_firm_reports.create_prop_firm_portfolio_simulator",
        ) as mock_create,
        patch(
            "research.portfolio.prop_firm_reports.generate_portfolio_report",
            return_value=Mock(),
        ) as mock_report,
    ):
        mock_sim = Mock()
        mock_sim.simulate.return_value = Mock()
        mock_create.return_value = mock_sim

        run_prop_firm_reports_for_phases(cfg, {"test": _phase("test", phase_dir)})

    mock_sim.simulate_rolling.assert_not_called()
    assert mock_report.call_args.kwargs["rolling"] is None
