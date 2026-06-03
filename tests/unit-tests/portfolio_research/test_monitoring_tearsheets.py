from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from portfolio_research.config import (
    PortfolioFitMode,
    PortfolioHoldoutRobustnessConfig,
    PortfolioResearchConfig,
    ResearchWindow,
)
from portfolio_research.holdout.monitoring_policy import (
    MonitoringWindow,
    ReferenceCalibration,
    ReferenceSigmaMethod,
    StrategyReferenceSlices,
)
from portfolio_research.holdout.monitoring_tearsheets import (
    full_period_strategy_returns,
    write_strategy_monitoring_tearsheet,
)
from utils.core.enums import Ticker, TimeFrame


def test_full_period_strategy_returns_stitches_without_duplicates() -> None:
    train = pd.Series(0.001, index=pd.date_range("2018-01-01", periods=5, freq="B"))
    val = pd.Series(0.002, index=pd.date_range("2020-01-01", periods=5, freq="B"))
    holdout = pd.Series(0.003, index=pd.date_range("2023-01-01", periods=5, freq="B"))
    combined = full_period_strategy_returns(train, val, holdout)
    assert len(combined) == 15
    assert combined.iloc[-1] == pytest.approx(0.003)


def test_write_strategy_monitoring_tearsheet_skips_when_disabled(tmp_path: Path) -> None:
    reference = StrategyReferenceSlices(
        validation_returns=pd.Series(0.001, index=pd.date_range("2020-01-01", periods=30, freq="B")),
        train_returns=pd.Series(0.001, index=pd.date_range("2018-01-01", periods=30, freq="B")),
        full_holdout_returns=pd.Series(0.001, index=pd.date_range("2023-01-01", periods=30, freq="B")),
        validation_window=MonitoringWindow(
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2022-12-31"),
        ),
        train_window=MonitoringWindow(
            start=pd.Timestamp("2018-01-01"),
            end=pd.Timestamp("2019-12-31"),
        ),
        reference_volatility_returns=pd.Series(
            0.001, index=pd.date_range("2018-01-01", periods=60, freq="B")
        ),
        calibration=ReferenceCalibration(
            mu=0.001,
            sigma=0.01,
            sigma_method=ReferenceSigmaMethod.POOLED_TRAIN_VALIDATION,
            sigma_train=0.01,
            sigma_validation=0.01,
            sigma_relative_shift=0.0,
        ),
    )
    config = PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2018, 1, 1),
        end=datetime(2026, 5, 13),
        use_cache=False,
        train_window=ResearchWindow(datetime(2018, 1, 1), datetime(2019, 12, 31)),
        validation_window=ResearchWindow(datetime(2020, 1, 1), datetime(2022, 12, 31)),
        test_window=ResearchWindow(datetime(2023, 1, 1), datetime(2026, 5, 13)),
        ensemble_dirs={"demo": "vault/D/demo"},
        portfolio_fit_mode=PortfolioFitMode.SINGLE_FIT,
        output_root=tmp_path,
        holdout_robustness=PortfolioHoldoutRobustnessConfig(
            export_strategy_monitoring_tearsheets=False,
        ),
    )
    out_dir = tmp_path / "holdout" / "strategies" / "demo"
    out_dir.mkdir(parents=True)
    assert (
        write_strategy_monitoring_tearsheet(
            config,
            strategy_name="demo",
            reference=reference,
            output_dir=out_dir,
        )
        is None
    )


def test_write_strategy_monitoring_tearsheet_writes_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_generate_tearsheet(**kwargs: object) -> None:
        captured.update(kwargs)
        output = Path(str(kwargs["output_file"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<html>tearsheet</html>", encoding="utf-8")

    monkeypatch.setattr(
        "metrics.plotting.graphing.quantstats_reports.generate_tearsheet",
        _fake_generate_tearsheet,
    )
    monkeypatch.setattr(
        "portfolio_research.holdout.monitoring_tearsheets._load_benchmark_returns_for_index",
        lambda _config, index: pd.Series(0.0005, index=index, name="ES_buy_hold"),
    )
    reference = StrategyReferenceSlices(
        validation_returns=pd.Series(0.001, index=pd.date_range("2020-01-01", periods=30, freq="B")),
        train_returns=pd.Series(0.001, index=pd.date_range("2018-01-01", periods=30, freq="B")),
        full_holdout_returns=pd.Series(0.001, index=pd.date_range("2023-01-01", periods=30, freq="B")),
        validation_window=MonitoringWindow(
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2022-12-31"),
        ),
        train_window=MonitoringWindow(
            start=pd.Timestamp("2018-01-01"),
            end=pd.Timestamp("2019-12-31"),
        ),
        reference_volatility_returns=pd.Series(
            0.001, index=pd.date_range("2018-01-01", periods=60, freq="B")
        ),
        calibration=ReferenceCalibration(
            mu=0.001,
            sigma=0.01,
            sigma_method=ReferenceSigmaMethod.POOLED_TRAIN_VALIDATION,
            sigma_train=0.01,
            sigma_validation=0.01,
            sigma_relative_shift=0.0,
        ),
    )
    config = PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2018, 1, 1),
        end=datetime(2026, 5, 13),
        use_cache=False,
        train_window=ResearchWindow(datetime(2018, 1, 1), datetime(2019, 12, 31)),
        validation_window=ResearchWindow(datetime(2020, 1, 1), datetime(2022, 12, 31)),
        test_window=ResearchWindow(datetime(2023, 1, 1), datetime(2026, 5, 13)),
        ensemble_dirs={"demo": "vault/D/demo"},
        portfolio_fit_mode=PortfolioFitMode.SINGLE_FIT,
        output_root=tmp_path,
    )
    out_dir = tmp_path / "holdout" / "strategies" / "demo"
    out_dir.mkdir(parents=True)
    path = write_strategy_monitoring_tearsheet(
        config,
        strategy_name="demo",
        reference=reference,
        output_dir=out_dir,
    )
    assert path is not None
    assert path.is_file()
    assert captured["mode"] == "html"
    returns = captured["strategy_returns"]
    assert isinstance(returns, pd.Series)
    assert len(returns) >= 60
