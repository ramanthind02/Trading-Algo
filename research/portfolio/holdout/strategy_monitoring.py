"""Per-strategy holdout monitoring (validation-parity charts on trailing evaluation window)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from quantfoundry_core.robustness.validation import ValidationRobustnessReport

from research.feature.validation.robustness_runner import (
    _build_equity_bands_frame,
    _build_z_cusum_sr_frame,
    _markdown_summary,
)
from research.feature.visualization.validation_reports import (
    PORTFOLIO_HOLDOUT_PLOT_LABELS,
    write_validation_robustness_plots,
)
from pathlib import Path as _Path

from research.portfolio.config import PortfolioResearchConfig
from lib.core.enums import TimeFrame
from research.portfolio.holdout.monitoring_policy import (
    MonitoringStatus,
    MonitoringWindow,
    ReferenceCalibration,
    StrategyReferenceSlices,
    build_monitoring_status,
    build_reference_calibration,
    history_row_from_status,
    month_end_dates,
    slice_trailing_window,
    trend_traffic_lights,
)
from research.portfolio.holdout.monitoring_tearsheets import write_strategy_monitoring_tearsheet
from research.portfolio.shared.visualization_paths import holdout_root, holdout_strategy_dir
from research.evaluation.holdout_robustness import HoldoutRobustnessConfig, run_holdout_robustness_pipeline


def _holdout_config_from_portfolio(
    config: PortfolioResearchConfig,
) -> HoldoutRobustnessConfig:
    holdout = config.holdout_robustness
    return HoldoutRobustnessConfig(
        sharpe_confidence=holdout.sharpe_confidence,
        n_bootstrap=holdout.n_bootstrap,
        random_seed=holdout.random_seed,
        cusum_alpha=holdout.cusum_alpha,
        equity_band_fraction_limit=holdout.equity_band_fraction_limit,
        rolling_window=holdout.rolling_window,
        rolling_z_threshold=holdout.rolling_z_threshold,
        rolling_fraction_limit=holdout.rolling_fraction_limit,
        periods_per_year=holdout.periods_per_year,
    )


def _slice_series(frame: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Strategy column {column!r} missing from returns frame.")
    series = frame[column].dropna()
    return series.loc[(series.index >= start) & (series.index <= end)].sort_index()


def _ensemble_trading_timeframe(
    config: PortfolioResearchConfig,
    strategy_name: str,
) -> TimeFrame | None:
    ensemble_path = config.ensemble_dirs.get(strategy_name)
    if ensemble_path is None:
        return None
    for part in _Path(str(ensemble_path)).parts:
        if part in {tf.name for tf in TimeFrame}:
            return TimeFrame[part]
    return None


def _min_evaluation_bars_for_strategy(
    config: PortfolioResearchConfig,
    strategy_name: str,
) -> int:
    """Use lower bar thresholds for sparse monthly/weekly return series."""
    default_min = config.holdout_robustness.min_evaluation_bars
    timeframe = _ensemble_trading_timeframe(config, strategy_name)
    if timeframe is TimeFrame.M:
        return min(default_min, 6)
    if timeframe is TimeFrame.W:
        return min(default_min, 12)
    return default_min


def _override_for_strategy(
    config: PortfolioResearchConfig,
    strategy_name: str,
) -> float | None:
    overrides = config.holdout_robustness.monitoring_weight_overrides
    if overrides is None:
        return None
    return overrides.get(strategy_name)


def _build_strategy_reference_slices(
    config: PortfolioResearchConfig,
    *,
    research_strategy_returns: pd.DataFrame,
    holdout_strategy_returns: pd.DataFrame,
    column: str,
) -> StrategyReferenceSlices | None:
    holdout_cfg = config.holdout_robustness
    train_start = pd.Timestamp(config.train_window.start)
    train_end = pd.Timestamp(config.train_window.end)
    val_start = pd.Timestamp(config.validation_window.start)
    val_end = pd.Timestamp(config.validation_window.end)
    holdout_start = pd.Timestamp(config.test_window.start)
    holdout_end = pd.Timestamp(config.test_window.end)

    train_returns = _slice_series(research_strategy_returns, column, train_start, train_end)
    validation_returns = _slice_series(research_strategy_returns, column, val_start, val_end)
    full_holdout = _slice_series(holdout_strategy_returns, column, holdout_start, holdout_end)
    if validation_returns.shape[0] < 2 or full_holdout.shape[0] < 2:
        return None

    split_calibration = (
        holdout_cfg.split_reference_calibration and train_returns.shape[0] >= 2
    )
    calibration = build_reference_calibration(
        train_returns,
        validation_returns,
        regime_shift_threshold=holdout_cfg.reference_vol_regime_shift_threshold,
        recent_weight_on_shift=holdout_cfg.reference_vol_recent_weight_on_shift,
        split_calibration=split_calibration,
    )
    reference_volatility_returns = (
        pd.concat([train_returns, validation_returns]).sort_index()
        if split_calibration
        else validation_returns
    )
    return StrategyReferenceSlices(
        validation_returns=validation_returns,
        train_returns=train_returns,
        full_holdout_returns=full_holdout,
        validation_window=MonitoringWindow(start=val_start, end=val_end),
        train_window=MonitoringWindow(start=train_start, end=train_end),
        reference_volatility_returns=reference_volatility_returns,
        calibration=calibration,
    )


def _evaluation_slice(
    full_holdout: pd.Series,
    *,
    as_of: pd.Timestamp,
    trailing_months: int,
) -> pd.Series:
    return slice_trailing_window(
        full_holdout,
        end=as_of,
        months=trailing_months,
    )


def _run_monitoring_pipeline(
    reference: StrategyReferenceSlices,
    eval_returns: pd.Series,
    holdout_cfg: HoldoutRobustnessConfig,
) -> ValidationRobustnessReport:
    report = run_holdout_robustness_pipeline(
        reference.validation_returns,
        eval_returns,
        config=holdout_cfg,
        include_rank_correlation=False,
        reference_sigma=reference.calibration.sigma,
        reference_volatility_returns=reference.reference_volatility_returns,
    )
    assert isinstance(report, ValidationRobustnessReport)
    return report


def _reference_volatility_window(reference: StrategyReferenceSlices) -> MonitoringWindow:
    vol_index = reference.reference_volatility_returns.index
    return MonitoringWindow(
        start=pd.Timestamp(vol_index.min()),
        end=pd.Timestamp(vol_index.max()),
    )


def _build_monthly_history(
    reference: StrategyReferenceSlices,
    *,
    config: PortfolioResearchConfig,
    holdout_cfg: HoldoutRobustnessConfig,
    override: float | None,
) -> pd.DataFrame:
    holdout = config.holdout_robustness
    if not holdout.generate_monthly_history:
        return pd.DataFrame()

    holdout_start = pd.Timestamp(config.test_window.start)
    holdout_end = pd.Timestamp(config.test_window.end)
    full_holdout = reference.full_holdout_returns
    as_of_dates = month_end_dates(
        full_holdout.index,
        start=holdout_start,
        end=holdout_end,
    )
    rows = [
        history_row_from_status(
            build_monitoring_status(
                _run_monitoring_pipeline(reference, eval_slice, holdout_cfg),
                evaluation_window=MonitoringWindow(
                    start=pd.Timestamp(eval_slice.index.min()),
                    end=pd.Timestamp(eval_slice.index.max()),
                ),
                reference_window=reference.validation_window,
                override_weight_fraction=override,
            ),
            as_of_date=as_of,
        )
        for as_of in as_of_dates
        if (
            (eval_slice := _evaluation_slice(
                full_holdout,
                as_of=as_of,
                trailing_months=holdout.evaluation_trailing_months,
            )).shape[0]
            >= holdout.min_evaluation_bars
        )
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _write_monitoring_rollup(
    rollup_rows: list[dict[str, object]],
    output_root: Path,
) -> Path:
    rollup_dir = holdout_root(output_root)
    rollup_dir.mkdir(parents=True, exist_ok=True)
    rollup_path = rollup_dir / "monitoring_rollup.csv"
    pd.DataFrame(rollup_rows).to_csv(rollup_path, index=False)
    json_path = rollup_dir / "monitoring_rollup.json"
    json_path.write_text(json.dumps(rollup_rows, indent=2), encoding="utf-8")
    return rollup_path


def run_strategy_holdout_monitoring(
    config: PortfolioResearchConfig,
    *,
    research_strategy_returns: pd.DataFrame,
    holdout_strategy_returns: pd.DataFrame,
) -> dict[str, dict[str, Path]]:
    """Run holdout robustness for every strategy column and write artifacts."""

    if not config.holdout_robustness.enabled:
        return {}

    holdout_cfg = _holdout_config_from_portfolio(config)
    holdout = config.holdout_robustness
    holdout_end = pd.Timestamp(config.test_window.end)
    artifacts_by_strategy: dict[str, dict[str, Path]] = {}
    rollup_rows: list[dict[str, object]] = []

    for column in holdout_strategy_returns.columns:
        reference = _build_strategy_reference_slices(
            config,
            research_strategy_returns=research_strategy_returns,
            holdout_strategy_returns=holdout_strategy_returns,
            column=str(column),
        )
        if reference is None:
            continue
        strategy_name = str(column)
        override = _override_for_strategy(config, strategy_name)
        full_holdout = reference.full_holdout_returns
        out_dir = holdout_strategy_dir(config.output_root, strategy_name)
        paths: dict[str, Path] = {}

        tearsheet_path = write_strategy_monitoring_tearsheet(
            config,
            strategy_name=strategy_name,
            reference=reference,
            output_dir=out_dir,
        )
        if tearsheet_path is not None:
            paths["full_period_tearsheet"] = tearsheet_path

        eval_returns = _evaluation_slice(
            full_holdout,
            as_of=holdout_end,
            trailing_months=holdout.evaluation_trailing_months,
        )
        min_bars = _min_evaluation_bars_for_strategy(config, strategy_name)
        if eval_returns.shape[0] < min_bars:
            if paths:
                artifacts_by_strategy[strategy_name] = paths
            continue

        report = _run_monitoring_pipeline(reference, eval_returns, holdout_cfg)
        eval_window = MonitoringWindow(
            start=pd.Timestamp(eval_returns.index.min()),
            end=pd.Timestamp(eval_returns.index.max()),
        )
        status = build_monitoring_status(
            report,
            evaluation_window=eval_window,
            reference_window=reference.validation_window,
            override_weight_fraction=override,
        )
        vol_window = _reference_volatility_window(reference)

        paths.update(_write_strategy_holdout_summary(
            report,
            out_dir,
            strategy_name=strategy_name,
            holdout_datetimes=[str(index) for index in eval_returns.index],
            monitoring_status=status,
            reference_calibration=reference.calibration,
            reference_volatility_window=vol_window,
        ))

        history = _build_monthly_history(
            reference,
            config=config,
            holdout_cfg=holdout_cfg,
            override=override,
        )
        history_path = out_dir / "monitoring_history.csv"
        history.to_csv(history_path, index=False)
        paths["monitoring_history"] = history_path

        current_as_of = status.evaluation_window.end.date().isoformat()
        prev_light, two_ago_light = trend_traffic_lights(history, current_as_of=current_as_of)
        rollup_rows.append(
            {
                "strategy": strategy_name,
                "traffic_light": status.traffic_light.name,
                "tests_failed": status.tests_failed,
                "advisory_weight_fraction": status.advisory_weight_fraction,
                "override_weight_fraction": status.override_weight_fraction,
                "effective_weight_fraction": status.effective_weight_fraction,
                "eval_start": status.evaluation_window.start.date().isoformat(),
                "eval_end": status.evaluation_window.end.date().isoformat(),
                "traffic_light_prev_month": prev_light,
                "traffic_light_2mo_ago": two_ago_light,
                "all_passed": report.all_passed,
                "reference_sigma_method": reference.calibration.sigma_method.value,
                "sigma_relative_shift": reference.calibration.sigma_relative_shift,
            }
        )
        artifacts_by_strategy[strategy_name] = paths

    if rollup_rows:
        rollup_path = _write_monitoring_rollup(rollup_rows, config.output_root)
        for paths in artifacts_by_strategy.values():
            paths["monitoring_rollup"] = rollup_path

    return artifacts_by_strategy


def _markdown_holdout_monitoring_summary(
    report: ValidationRobustnessReport,
    *,
    strategy_name: str,
    status: MonitoringStatus,
    reference_calibration: ReferenceCalibration,
    reference_volatility_window: MonitoringWindow,
) -> str:
    detail = _markdown_summary(report, chosen_combo_label=strategy_name)
    detail = detail.replace("# Validation Robustness Summary", "## Test detail")
    detail = detail.replace("Val ", "Holdout ")
    override_display = (
        f"{status.override_weight_fraction:.2f}"
        if status.override_weight_fraction is not None
        else "—"
    )
    lines = [
        "# Strategy Holdout Monitoring",
        "",
        f"**Strategy:** {strategy_name}",
        "",
        "## Monitoring status",
        "",
        f"- **Traffic light:** {status.traffic_light.name}",
        f"- **Tests failed:** {status.tests_failed}/4",
        f"- **Advisory weight:** {status.advisory_weight_fraction:.2f}",
        f"- **Override weight:** {override_display}",
        f"- **Effective weight:** {status.effective_weight_fraction:.2f}",
        f"- **Evaluation window:** {status.evaluation_window.start.date()} → {status.evaluation_window.end.date()}",
        f"- **Reference μ window (validation):** {status.reference_window.start.date()} → {status.reference_window.end.date()}",
        f"- **Reference σ window:** {reference_volatility_window.start.date()} → {reference_volatility_window.end.date()}",
        f"- **Reference μ:** {reference_calibration.mu:.6f}",
        f"- **Reference σ:** {reference_calibration.sigma:.6f} ({reference_calibration.sigma_method.value})",
        f"- **σ train / val:** {reference_calibration.sigma_train:.6f} / {reference_calibration.sigma_validation:.6f} (shift {reference_calibration.sigma_relative_shift:.1%})",
        "",
        f"**Legacy overall (all tests):** {'PASS' if report.all_passed else 'FAIL'} — {report.interpretation}",
        "",
        detail,
    ]
    return "\n".join(lines)


def _enriched_report_payload(
    report: ValidationRobustnessReport,
    status: MonitoringStatus,
    *,
    reference_calibration: ReferenceCalibration,
    reference_volatility_window: MonitoringWindow,
) -> dict[str, object]:
    payload = report.to_json_dict()
    payload["monitoring"] = status.to_dict_with_calibration(
        reference_calibration,
        reference_volatility_window=reference_volatility_window,
    )
    return payload


def _write_strategy_holdout_summary(
    report: ValidationRobustnessReport,
    output_dir: Path,
    *,
    strategy_name: str,
    holdout_datetimes: list[str],
    monitoring_status: MonitoringStatus,
    reference_calibration: ReferenceCalibration,
    reference_volatility_window: MonitoringWindow,
    artifact_prefix: str = "holdout",
    plot_output_dir: Path | None = None,
    write_plots: bool = True,
) -> dict[str, Path]:
    """Persist holdout robustness artifacts for one strategy."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "holdout_robustness_report.json"
    status_path = output_dir / "monitoring_status.json"
    markdown_path = output_dir / "holdout_robustness_summary.md"
    z_cusum_csv = output_dir / f"{artifact_prefix}_z_cusum_sr.csv"
    bands_csv = output_dir / f"{artifact_prefix}_equity_bands.csv"

    monitoring_payload = monitoring_status.to_dict_with_calibration(
        reference_calibration,
        reference_volatility_window=reference_volatility_window,
    )
    json_path.write_text(
        json.dumps(
            _enriched_report_payload(
                report,
                monitoring_status,
                reference_calibration=reference_calibration,
                reference_volatility_window=reference_volatility_window,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    status_path.write_text(json.dumps(monitoring_payload, indent=2), encoding="utf-8")
    markdown_path.write_text(
        _markdown_holdout_monitoring_summary(
            report,
            strategy_name=strategy_name,
            status=monitoring_status,
            reference_calibration=reference_calibration,
            reference_volatility_window=reference_volatility_window,
        ),
        encoding="utf-8",
    )
    _build_z_cusum_sr_frame(report, holdout_datetimes).to_csv(z_cusum_csv, index=False)
    _build_equity_bands_frame(report).to_csv(bands_csv, index=False)

    plot_dir = plot_output_dir if plot_output_dir is not None else output_dir / "matplotlib"
    plot_paths: list[Path] = []
    if write_plots:
        plot_paths = write_validation_robustness_plots(
            report,
            plot_csv_dir=output_dir,
            output_dir=plot_dir,
            plot_labels=PORTFOLIO_HOLDOUT_PLOT_LABELS,
            include_rank_scatter=False,
            artifact_prefix=artifact_prefix,
        )
    return {
        "json": json_path,
        "monitoring_status": status_path,
        "markdown": markdown_path,
        "z_cusum_sr_csv": z_cusum_csv,
        "equity_bands_csv": bands_csv,
        "plot_dir": plot_dir,
        **{f"plot_{index}": path for index, path in enumerate(plot_paths)},
    }
