"""Legacy FeatureExplorer contract tests.

These tests pin currently consumed output schemas during the validator API migration.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eda.feature_explorer import FeatureExplorer
from feature_selection.base_models.quantile_binning import QuantileBinningModel
from metrics.performance import SortinoRatio


def _build_minimal_explorer(seed: int = 123, n_rows: int = 250) -> FeatureExplorer:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_rows)

    feature_col = "rsi_signal_D_lookback_14"
    features_df = pd.DataFrame({feature_col: rng.randn(n_rows)}, index=dates)
    targets_df = pd.DataFrame({"log_return": rng.randn(n_rows) * 0.01}, index=dates)

    metadata = {
        "feature_metadata": {
            feature_col: {
                "module": "rsi",
                "parameters": {"lookback": 14},
                "base_name": "rsi",
                "full_name": feature_col,
            }
        }
    }
    return FeatureExplorer(features_df=features_df, targets_df=targets_df, metadata=metadata)


def test_run_permutation_test_schema_contract() -> None:
    explorer = _build_minimal_explorer()

    results = explorer.run_permutation_test(
        target_col="log_return",
        base_model=QuantileBinningModel(n_bins=3, selection_metric="sortino"),
        metric=SortinoRatio(annualization_factor=252),
        nreps=8,
        n_jobs=1,
        alpha=0.1,
        verbose=False,
    )

    required_columns = {"feature", "original_criterion", "pval", "significant"}
    assert required_columns.issubset(results.columns)
    assert len(results) == 1
    assert 0.0 <= float(results.iloc[0]["pval"]) <= 1.0


def test_generate_parameter_sensitivity_report_keys_contract() -> None:
    explorer = _build_minimal_explorer()

    report = explorer.generate_parameter_sensitivity_report(
        module_name="rsi",
        param_names=["lookback"],
        verbose=False,
    )

    assert tuple(report.keys()) == (
        "results_df",
        "summary_stats",
        "robustness_scores",
        "figures",
        "parameter_sensitivity",
        "report_text",
        "report_path",
    )
