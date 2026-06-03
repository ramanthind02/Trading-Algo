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
    MonitoringTrafficLight,
    MonitoringWindow,
    ReferenceSigmaMethod,
    advisory_weight_fraction,
    build_monitoring_status,
    build_reference_calibration,
    count_failed_tests,
    effective_weight,
    month_end_dates,
    resolve_reference_sigma,
    slice_trailing_window,
    traffic_light_from_fail_count,
    trend_traffic_lights,
)
from portfolio_research.holdout.strategy_monitoring import run_strategy_holdout_monitoring
from utils.core.enums import Ticker, TimeFrame
from utils.evaluation.holdout_robustness import HoldoutRobustnessConfig, run_holdout_robustness_pipeline


@pytest.mark.parametrize(
    ("failed", "expected"),
    [
        (0, MonitoringTrafficLight.GREEN),
        (1, MonitoringTrafficLight.GREEN),
        (2, MonitoringTrafficLight.YELLOW),
        (3, MonitoringTrafficLight.RED),
        (4, MonitoringTrafficLight.RED),
    ],
)
def test_traffic_light_from_fail_count(failed: int, expected: MonitoringTrafficLight) -> None:
    assert traffic_light_from_fail_count(failed) == expected


def test_advisory_and_effective_weights() -> None:
    assert advisory_weight_fraction(MonitoringTrafficLight.GREEN) == 1.0
    assert advisory_weight_fraction(MonitoringTrafficLight.YELLOW) == 0.5
    assert advisory_weight_fraction(MonitoringTrafficLight.RED) == 0.0
    assert effective_weight(0.5, None) == 0.5
    assert effective_weight(0.5, 0.25) == 0.25


def test_slice_trailing_window_twelve_months() -> None:
    index = pd.date_range("2024-01-01", "2026-05-13", freq="B")
    series = pd.Series(1.0, index=index)
    end = pd.Timestamp("2026-05-13")
    trailing = slice_trailing_window(series, end=end, months=12)
    assert trailing.index.min() >= pd.Timestamp("2025-05-13")
    assert trailing.index.max() == end


def test_month_end_dates_returns_last_trading_day_per_month() -> None:
    index = pd.date_range("2023-01-03", "2024-06-28", freq="B")
    ends = month_end_dates(
        index,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2024-06-30"),
    )
    assert len(ends) >= 17
    assert ends[0].month == 1
    assert ends[-1] <= pd.Timestamp("2024-06-30")


def test_build_monitoring_status_from_pipeline_report() -> None:
    index = pd.date_range("2018-01-01", periods=300, freq="B")
    is_returns = pd.Series(0.0008, index=index)
    holdout_returns = pd.Series(-0.001, index=index[-120:])
    report = run_holdout_robustness_pipeline(
        is_returns,
        holdout_returns,
        config=HoldoutRobustnessConfig(n_bootstrap=30, random_seed=1),
        include_rank_correlation=False,
    )
    status = build_monitoring_status(
        report,
        evaluation_window=MonitoringWindow(
            start=pd.Timestamp(holdout_returns.index.min()),
            end=pd.Timestamp(holdout_returns.index.max()),
        ),
        reference_window=MonitoringWindow(
            start=pd.Timestamp("2018-01-01"),
            end=pd.Timestamp("2022-12-31"),
        ),
        override_weight_fraction=0.25,
    )
    assert status.tests_failed == count_failed_tests(report)
    assert status.effective_weight_fraction == 0.25
    assert status.traffic_light == traffic_light_from_fail_count(status.tests_failed)


def test_resolve_reference_sigma_pools_when_regimes_align() -> None:
    index = pd.date_range("2018-01-01", periods=200, freq="B")
    train = pd.Series(0.001, index=index)
    validation = pd.Series(0.001, index=index[-100:])
    sigma, method, _, _, _ = resolve_reference_sigma(
        train,
        validation,
        regime_shift_threshold=0.30,
        recent_weight_on_shift=0.70,
    )
    pooled = pd.concat([train, validation])
    assert method == ReferenceSigmaMethod.POOLED_TRAIN_VALIDATION
    assert sigma == pytest.approx(float(pooled.std(ddof=1)), rel=1e-6)


def test_resolve_reference_sigma_weights_toward_validation_on_shift() -> None:
    train = pd.Series([0.01, -0.01, 0.01, -0.01] * 30)
    validation = pd.Series([0.001] * 120)
    sigma, method, sigma_train, sigma_val, shift = resolve_reference_sigma(
        train,
        validation,
        regime_shift_threshold=0.30,
        recent_weight_on_shift=0.70,
    )
    assert method == ReferenceSigmaMethod.WEIGHTED_TOWARD_VALIDATION
    assert shift > 0.30
    expected = 0.70 * sigma_val + 0.30 * sigma_train
    assert sigma == pytest.approx(expected, rel=1e-6)


def test_build_reference_calibration_split_mu_from_validation() -> None:
    train = pd.Series([0.02, -0.02] * 50)
    validation = pd.Series([0.001] * 100)
    calibration = build_reference_calibration(
        train,
        validation,
        regime_shift_threshold=0.30,
        recent_weight_on_shift=0.70,
        split_calibration=True,
    )
    assert calibration.mu == pytest.approx(float(validation.mean()))
    assert calibration.sigma_method == ReferenceSigmaMethod.WEIGHTED_TOWARD_VALIDATION


def test_trend_traffic_lights_from_history() -> None:
    history = pd.DataFrame(
        [
            {"as_of_date": "2025-03-31", "traffic_light": "GREEN"},
            {"as_of_date": "2025-04-30", "traffic_light": "YELLOW"},
            {"as_of_date": "2025-05-31", "traffic_light": "RED"},
        ]
    )
    prev, two_ago = trend_traffic_lights(history, current_as_of="2025-06-15")
    assert prev == "RED"
    assert two_ago == "YELLOW"


def test_run_strategy_holdout_monitoring_writes_rollup(tmp_path: Path) -> None:
    index = pd.date_range("2000-01-03", "2026-05-13", freq="B")
    rng = __import__("numpy").random.default_rng(9)
    research = pd.DataFrame(
        {"demo": rng.normal(0.0008, 0.01, size=len(index))},
        index=index,
    )
    holdout = pd.DataFrame(
        {"demo": rng.normal(0.0004, 0.01, size=len(index[index >= "2023-01-01"]))},
        index=index[index >= "2023-01-01"],
    )
    config = PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2018, 1, 1),
        end=datetime(2026, 5, 13),
        use_cache=False,
        train_window=ResearchWindow(datetime(2000, 1, 1), datetime(2017, 12, 31)),
        validation_window=ResearchWindow(datetime(2018, 1, 1), datetime(2022, 12, 31)),
        test_window=ResearchWindow(datetime(2023, 1, 1), datetime(2026, 5, 13)),
        ensemble_dirs={"demo": "vault/D/demo"},
        portfolio_fit_mode=PortfolioFitMode.SINGLE_FIT,
        output_root=tmp_path,
        holdout_robustness=PortfolioHoldoutRobustnessConfig(
            n_bootstrap=30,
            random_seed=1,
            min_evaluation_bars=30,
            generate_monthly_history=True,
        ),
    )
    artifacts = run_strategy_holdout_monitoring(
        config,
        research_strategy_returns=research,
        holdout_strategy_returns=holdout,
    )
    assert "demo" in artifacts
    rollup = tmp_path / "holdout" / "monitoring_rollup.csv"
    assert rollup.is_file()
    history = tmp_path / "holdout" / "strategies" / "demo" / "monitoring_history.csv"
    assert history.is_file()
    payload = __import__("json").loads(
        (tmp_path / "holdout" / "strategies" / "demo" / "holdout_robustness_report.json").read_text()
    )
    assert "monitoring" in payload
    assert payload["monitoring"]["traffic_light"] in {"GREEN", "YELLOW", "RED"}
