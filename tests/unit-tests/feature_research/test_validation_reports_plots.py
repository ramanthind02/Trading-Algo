from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from quantfoundry_core.robustness.validation import (
    CUSUMResult,
    EquityCurveBandsResult,
    RankCorrelationResult,
    RollingSharpezScoreResult,
    SharpeComparisonResult,
    ValidationRobustnessReport,
    assemble_validation_report,
    cusum_vs_is_params,
    equity_curve_confidence_bands,
    parameter_rank_correlation,
    rolling_sharpe_zscore,
    sharpe_comparison,
)
from quantfoundry_core.robustness import BootstrapCI, SharpeCI, sharpe_confidence_interval

from research.feature.validation.robustness_runner import (
    load_validation_robustness_report_from_json,
    write_validation_robustness_summary,
)
from research.feature.visualization.validation_reports import write_validation_robustness_plots


def _synthetic_report() -> ValidationRobustnessReport:
    is_returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01, 0.008, 0.012, 0.004] * 20, dtype=float)
    val_returns = np.array([0.008, 0.006, -0.002, 0.01, 0.005, 0.004, 0.007, 0.003] * 10, dtype=float)
    mu_is = float(is_returns.mean())
    sigma_is = float(is_returns.std(ddof=1))
    sharpe_ci_is = sharpe_confidence_interval(
        sr_observed=float(is_returns.mean() / is_returns.std(ddof=1)),
        n_obs=int(is_returns.shape[0]),
        skewness=float(pd.Series(is_returns).skew()),
        excess_kurtosis=float(pd.Series(is_returns).kurt()),
    )
    bootstrap_ci_val = BootstrapCI(
        point_estimate=0.25,
        lower=0.05,
        upper=0.45,
        ci_level=0.95,
        n_bootstrap=100,
        block_length=5,
        seed=1,
    )
    sr_is = float(is_returns.mean() / is_returns.std(ddof=1))
    sr_val = float(val_returns.mean() / val_returns.std(ddof=1))
    sharpe_result = sharpe_comparison(
        sr_is=sr_is,
        sharpe_ci_is=sharpe_ci_is,
        sr_val=sr_val,
        n_val=int(val_returns.shape[0]),
        skewness_val=float(pd.Series(val_returns).skew()),
        excess_kurtosis_val=float(pd.Series(val_returns).kurt()),
        bootstrap_ci_val=bootstrap_ci_val,
    )
    cusum_result = cusum_vs_is_params(val_returns, mu_is, sigma_is)
    bands_result = equity_curve_confidence_bands(val_returns, mu_is, sigma_is)
    rolling_result = rolling_sharpe_zscore(is_returns, val_returns, window=20)
    rank_result = parameter_rank_correlation(
        [1.2, 0.8, 0.4],
        [1.0, 0.6, 0.3],
    )
    return assemble_validation_report(
        sharpe_result,
        cusum_result,
        bands_result,
        rolling_result,
        rank_result,
    )


def test_write_validation_robustness_plots_from_csv_inputs(tmp_path: Path) -> None:
    report = _synthetic_report()
    rank_rows = [
        type("Row", (), {"param_combo_label": "a", "is_metric": 1.2, "val_metric": 1.0, "is_chosen": True})(),
        type("Row", (), {"param_combo_label": "b", "is_metric": 0.8, "val_metric": 0.6, "is_chosen": False})(),
        type("Row", (), {"param_combo_label": "c", "is_metric": 0.4, "val_metric": 0.3, "is_chosen": False})(),
    ]
    artifacts = write_validation_robustness_summary(
        report,
        tmp_path,
        rank_scatter_rows=rank_rows,  # type: ignore[arg-type]
        chosen_combo_label="a",
    )
    plot_paths = write_validation_robustness_plots(
        report,
        plot_csv_dir=tmp_path,
        output_dir=tmp_path / "matplotlib",
        chosen_combo_label="a",
    )

    assert artifacts["json"].exists()
    assert len(plot_paths) == 4
    assert all(path.exists() for path in plot_paths)
    reloaded = load_validation_robustness_report_from_json(artifacts["json"])
    assert isinstance(reloaded, ValidationRobustnessReport)
    assert isinstance(reloaded.sharpe_comparison, SharpeComparisonResult)
    assert isinstance(reloaded.cusum, CUSUMResult)
    assert isinstance(reloaded.equity_curve_bands, EquityCurveBandsResult)
    assert isinstance(reloaded.rolling_sharpe_zscore, RollingSharpezScoreResult)
    assert isinstance(reloaded.rank_correlation, RankCorrelationResult)
