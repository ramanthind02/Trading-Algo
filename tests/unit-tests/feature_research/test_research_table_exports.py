"""Unit tests for Power BI / tabular research exports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_research.binning.transforms import build_param_combo_long_table
from feature_research.core_helpers import combo_key
from feature_research.research_table_exports import (
    objective_metric_display_label,
    permutation_vector_shuffle_records,
    walkforward_power_bi_dir,
    write_in_sample_equity_curve_powerbi_csv,
    write_param_sensitivity_powerbi_tables,
    write_permutation_vector_shuffle_exports,
    write_walkforward_equity_powerbi_csvs,
)
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from feature_selection.validation.stability_analysis import _param_combo_name
from feature_selection.validation.reports import (
    ComboDecisionRecord,
    FunnelStatistics,
    PermutationTestSuite,
    VectorShuffleReport,
    WalkforwardStabilityReport,
)
from utils.core.enums import TimeFrame
from utils.evaluation.walkforward.selected_params_codec import serialize_selected_params


def test_walkforward_power_bi_dir_stable_under_output_root(tmp_path: Path) -> None:
    root = tmp_path / "shared_results"
    assert walkforward_power_bi_dir(root, "validation") == root / "powerbi" / "validation"
    assert walkforward_power_bi_dir(root, "oos") == root / "powerbi" / "oos"


def test_objective_metric_display_label_builtin() -> None:
    spec = ObjectiveMetricSpec(builtin="sortino")
    assert objective_metric_display_label(spec) == "sortino"


def test_objective_metric_display_label_none() -> None:
    assert objective_metric_display_label(None) == "unknown"


def test_write_param_sensitivity_powerbi_tables(tmp_path: Path) -> None:
    rows = [
        {
            "param_combo_label": "a=1",
            "feature_name": "feat_x",
            "n_observations": 10,
            "n_nonzero_signal": 7,
            "sharpe": 0.5,
            "t_stat": 1.2,
            "sortino": 0.9,
        }
    ]
    pairs = [("a=1", {"a": 1})]
    by_ticker = [
        {
            "param_sensitivity_by_ticker_key": "a=1__ES",
            "param_combo_label": "a=1",
            "feature_name": "feat_x",
            "ticker": "ES",
            "n_observations": 5,
            "n_nonzero_signal": 4,
            "sharpe": 0.4,
            "t_stat": 1.0,
            "sortino": 0.8,
        }
    ]
    paths = write_param_sensitivity_powerbi_tables(
        rows, pairs, sensitivity_by_ticker_rows=by_ticker, powerbi_parent_dir=tmp_path
    )
    assert paths["param_sensitivity_csv"].exists()
    assert paths["param_sensitivity_by_ticker_csv"].exists()
    assert paths["param_combo_long_csv"].exists()
    assert "param_combo_wide_csv" not in paths
    assert not (tmp_path / "powerbi" / "param_combo_wide.csv").exists()
    df = pd.read_csv(paths["param_sensitivity_csv"])
    assert "param_a" not in df.columns
    assert float(df["sharpe"].iloc[0]) == pytest.approx(0.5)
    dfb = pd.read_csv(paths["param_sensitivity_by_ticker_csv"])
    assert dfb["param_sensitivity_by_ticker_key"].iloc[0] == "a=1__ES"
    assert dfb["param_sensitivity_by_ticker_key"].is_unique
    assert dfb["ticker"].iloc[0] == "ES"
    assert float(dfb["sharpe"].iloc[0]) == pytest.approx(0.4)
    assert "param_a" not in dfb.columns
    long_df = pd.read_csv(paths["param_combo_long_csv"])
    assert set(long_df.columns) == {"param_combo_label", "param_key", "param_value", "param_sort_order"}
    assert float(long_df.loc[long_df["param_key"].eq("a"), "param_value"].iloc[0]) == pytest.approx(1.0)


def test_param_combo_long_flattens_filter_gate_like_flat_bias(tmp_path: Path) -> None:
    """Composite filter_gate params become scalar rows (f_* / s_*), not JSON blobs."""
    rows = [
        {
            "param_combo_label": "combo_x",
            "feature_name": "feat_gate",
            "n_observations": 10,
            "n_nonzero_signal": 5,
            "sharpe": 0.1,
            "t_stat": 0.2,
            "sortino": 0.15,
        }
    ]
    gate_params = {
        "filter_module": "atr_percentile_filter",
        "filter_params": {
            "atr_period": 126,
            "lookback": 200,
            "max_rank_fraction": 0.3,
            "rank_metric": "atr_pct",
        },
        "signal_module": "rsi_signal",
        "signal_params": {
            "rsi_period": 2,
            "oversold": 25.0,
            "overbought": 75.0,
            "strategy_mode": "long",
            "exit_policy": "threshold_or_bars",
            "exit_bars": 10,
        },
    }
    paths = write_param_sensitivity_powerbi_tables(
        rows, [("combo_x", gate_params)], sensitivity_by_ticker_rows=[], powerbi_parent_dir=tmp_path
    )
    long_df = pd.read_csv(paths["param_combo_long_csv"])
    keys = set(long_df["param_key"].tolist())
    assert "filter_params" not in keys
    assert "signal_params" not in keys
    assert "f_atr_period" in keys
    assert "s_rsi_period" in keys
    assert long_df.loc[long_df["param_key"].eq("f_atr_period"), "param_value"].iloc[0] == "126"
    assert (
        long_df.loc[long_df["param_key"].eq("filter_module"), "param_value"].iloc[0]
        == "atr_percentile_filter"
    )


def test_param_combo_long_flattens_dual_signal() -> None:
    long_df = build_param_combo_long_table(
        [
            (
                "ds1",
                {
                    "moduleA": "rsi_signal",
                    "moduleB": "ewmac",
                    "paramsA": {"rsi_period": 2},
                    "paramsB": {"span_fast": 16, "span_slow": 64},
                },
            )
        ]
    )
    keys = set(long_df["param_key"].tolist())
    assert "paramsA" not in keys and "paramsB" not in keys
    assert "a_rsi_period" in keys
    assert "b_span_fast" in keys


def test_write_param_sensitivity_omits_by_ticker_when_empty(tmp_path: Path) -> None:
    rows = [
        {
            "param_combo_label": "a=1",
            "feature_name": "feat_x",
            "n_observations": 10,
            "n_nonzero_signal": 7,
            "sharpe": 0.5,
            "t_stat": 1.2,
            "sortino": 0.9,
        }
    ]
    paths = write_param_sensitivity_powerbi_tables(
        rows, [("a=1", {"a": 1})], sensitivity_by_ticker_rows=[], powerbi_parent_dir=tmp_path
    )
    assert "param_sensitivity_by_ticker_csv" not in paths
    assert "param_combo_long_csv" in paths
    assert "param_combo_wide_csv" not in paths


def test_write_in_sample_equity_curve_powerbi_csv(tmp_path: Path) -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    sig = pd.Series([1.0, -1.0, 0.0], index=idx, name="signal")
    tgt = pd.Series([0.01, 0.02, 0.03], index=idx, name="target")
    tkr = pd.Series(["ES"] * 3, index=idx, dtype=str)
    store = {"combo_a": (sig, tgt, "feat_x", tkr, {"k": 1})}
    path = write_in_sample_equity_curve_powerbi_csv(store, powerbi_parent_dir=tmp_path)
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == [
        "datetime",
        "param_combo_label",
        "feature_name",
        "ticker",
        "strategy_return",
        "cumulative_strategy_return",
    ]
    assert df["param_combo_label"].eq("combo_a").all()
    assert float(df["strategy_return"].iloc[0]) == pytest.approx(0.01)
    assert float(df["strategy_return"].iloc[1]) == pytest.approx(-0.02)
    assert float(df["strategy_return"].iloc[2]) == pytest.approx(0.0)
    assert float(df["cumulative_strategy_return"].iloc[2]) == pytest.approx(-0.01)


def _make_ohlcv_candles(
    dates: pd.DatetimeIndex,
    ticker: str,
    base_price: float = 100.0,
    daily_return: float = 0.001,
) -> pd.DataFrame:
    """Synthetic OHLCV candles for portfolio simulation tests."""
    closes = [base_price * (1 + daily_return) ** i for i in range(len(dates))]
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1000] * len(dates),
            "ticker": ticker,
            "timeframe": "D",
        }
    )


def test_write_walkforward_equity_powerbi_csvs_holdout_and_extended(tmp_path: Path) -> None:
    """Legacy path (no portfolio_candles): signal × target cumsum."""
    params = {"long_period": 100, "rsi_period": 2, "short_period": 4}
    key = combo_key(params)
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    # Slight target variation so rolling std > 0 (rolling Sharpe is defined).
    tgt_vals = [0.010, 0.011, 0.009, 0.012, 0.010, 0.011]
    paired = pd.DataFrame(
        {
            "signal": [1.0] * 6,
            "target": tgt_vals,
            "ticker": ["ES"] * 6,
            "returns": [1.0 * t for t in tgt_vals],
        },
        index=idx,
    )
    combo_map = {key: paired}
    label = "long_period=100|rsi_period=2|short_period=4"
    summary = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": label,
                "selected_raw_objective": 1.0,
                "selected_smoothed_objective": 1.0,
                "selected_params_json": serialize_selected_params(params),
            }
        ]
    )
    out = tmp_path / "pb_legacy"
    paths = write_walkforward_equity_powerbi_csvs(
        combo_signal_target=combo_map,
        selection_summary_df=summary,
        module_name="cyclical_rsi",
        timeframe=TimeFrame.D,
        holdout_start=pd.Timestamp("2020-01-04"),
        holdout_end=pd.Timestamp("2020-01-06"),
        extended_start=pd.Timestamp("2020-01-01"),
        extended_end=pd.Timestamp("2020-01-06"),
        output_powerbi_dir=out,
        holdout_csv_stem="equity_curve_validation_only",
        extended_csv_stem="equity_curve_train_and_validation",
        rolling_sharpe_window_bars=2,
    )
    assert paths["holdout"].exists()
    assert paths["extended"].exists()
    dh = pd.read_csv(paths["holdout"])
    de = pd.read_csv(paths["extended"])
    assert dh["fold_id"].eq(0).all()
    assert len(dh) == 3
    assert len(de) == 6
    assert "cumulative_strategy_return" in dh.columns
    assert "rolling_sharpe_annualized" in dh.columns
    assert dh["rolling_sharpe_window_bars"].eq(2).all()
    assert pd.isna(dh["rolling_sharpe_annualized"].iloc[0])
    assert np.isfinite(dh["rolling_sharpe_annualized"].iloc[-1])
    assert float(dh["strategy_return"].sum()) == pytest.approx(0.012 + 0.010 + 0.011)


def test_write_walkforward_equity_powerbi_csvs_portfolio_path(tmp_path: Path) -> None:
    """Portfolio path: TFPortfolio IDM + calculate_strategy_returns_from_positions."""
    params = {"long_period": 100, "rsi_period": 2, "short_period": 4}
    key = combo_key(params)
    full_dates = pd.date_range("2020-01-02", periods=20, freq="B")
    # Signal fires for every bar
    signal = pd.Series([1.0] * len(full_dates), index=full_dates, name="signal")
    target = pd.Series([0.001] * len(full_dates), index=full_dates, name="target")
    ticker_s = pd.Series(["ES"] * len(full_dates), index=full_dates, name="ticker")
    paired = pd.DataFrame(
        {
            "signal": signal,
            "target": target,
            "returns": signal * target,
            "ticker": ticker_s,
        }
    )
    combo_map = {key: paired}
    label = "long_period=100|rsi_period=2|short_period=4"
    summary = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": label,
                "selected_raw_objective": 1.0,
                "selected_smoothed_objective": 1.0,
                "selected_params_json": serialize_selected_params(params),
            }
        ]
    )
    candles = _make_ohlcv_candles(full_dates, ticker="ES")
    extended_start = full_dates[0]
    holdout_start = full_dates[10]
    holdout_end = full_dates[-1]
    out = tmp_path / "pb_portfolio"
    paths = write_walkforward_equity_powerbi_csvs(
        combo_signal_target=combo_map,
        selection_summary_df=summary,
        module_name="cyclical_rsi",
        timeframe=TimeFrame.D,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        extended_start=extended_start,
        extended_end=holdout_end,
        output_powerbi_dir=out,
        holdout_csv_stem="equity_curve_validation_only",
        extended_csv_stem="equity_curve_train_and_validation",
        portfolio_candles=candles,
        rolling_sharpe_window_bars=5,
    )
    assert paths["holdout"].exists()
    assert paths["extended"].exists()
    dh = pd.read_csv(paths["holdout"])
    de = pd.read_csv(paths["extended"])
    assert not dh.empty
    assert not de.empty
    assert "cumulative_strategy_return" in dh.columns
    assert "rolling_sharpe_annualized" in de.columns
    assert de["rolling_sharpe_window_bars"].eq(5).all()
    assert de["rolling_sharpe_annualized"].notna().any()
    assert "fold_id" in dh.columns
    assert dh["fold_id"].eq(0).all()
    # Holdout rows should be fewer than extended rows (subset of dates)
    assert len(dh) < len(de)
    # With a constantly-rising price and signal=1 throughout, all strategy_returns > 0
    assert (pd.to_numeric(dh["strategy_return"], errors="coerce") >= 0).all()


def test_equity_curve_separate_cumsum_per_ticker_on_duplicate_dates(tmp_path: Path) -> None:
    """Same calendar date for ES and NQ: each instrument gets its own cumsum (no cross-mixing)."""
    d = pd.Timestamp("2020-01-03", tz="UTC")
    idx = pd.Index([d, d, d, d])
    sig = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
    tgt = pd.Series([0.01, -0.02, 0.03, -0.04], index=idx)
    tkr = pd.Series(["ES", "NQ", "ES", "NQ"], index=idx, dtype=str)
    store = {"c1": (sig, tgt, "f", tkr, {})}
    path = write_in_sample_equity_curve_powerbi_csv(store, powerbi_parent_dir=tmp_path)
    df = pd.read_csv(path)
    es = df.loc[df["ticker"].eq("ES")].reset_index(drop=True)
    nq = df.loc[df["ticker"].eq("NQ")].reset_index(drop=True)
    assert float(es["cumulative_strategy_return"].iloc[0]) == pytest.approx(0.01)
    assert float(es["cumulative_strategy_return"].iloc[1]) == pytest.approx(0.04)
    assert float(nq["cumulative_strategy_return"].iloc[0]) == pytest.approx(-0.02)
    assert float(nq["cumulative_strategy_return"].iloc[1]) == pytest.approx(-0.06)


def _minimal_permutation_suite() -> PermutationTestSuite:
    null = np.array([0.01, -0.02, 0.0], dtype=float)
    r1 = VectorShuffleReport(
        param_combo="p1",
        original_metric=0.4,
        null_distribution=null,
        critical_value=0.1,
        p_value=0.03,
        passed=True,
        alpha=0.1,
        nreps=3,
    )
    wf = WalkforwardStabilityReport(
        feature_name="f",
        feature_type="signed_signal",
        fold_results=[],
        consistency_metrics={},
        is_stable=True,
        stability_verdict="Stage 3 (walkforward permutation) removed.",
        top_k=1,
    )
    fs = FunnelStatistics(
        total_params=1,
        stage1_pass=1,
        stage2_pass=0,
        stable_params=0,
        ensemble_candidates=0,
        computational_savings_pct=0.0,
    )
    return PermutationTestSuite(
        feature_name="feat",
        feature_type="signed_signal",
        stage1_reports={"p1": r1},
        stage2_reports={},
        stage3_report=wf,
        funnel_stats=fs,
        ensemble_candidates=["p1"],
        summary="test",
        combo_decisions={
            "p1": ComboDecisionRecord(
                param_combo="p1",
                stage1_passed=True,
                stage2_passed=False,
                walkforward_stable=False,
                oos_passed=False,
                final_status="candidate",
            )
        },
    )


def test_permutation_vector_shuffle_records_and_export(tmp_path: Path) -> None:
    suite = _minimal_permutation_suite()
    recs = permutation_vector_shuffle_records(suite, "sharpe")
    assert len(recs) == 1
    assert recs[0]["param_combo"] == "p1"
    assert recs[0]["param_combo_label"] == "p1"
    assert recs[0]["objective_metric"] == "sharpe"
    out = write_permutation_vector_shuffle_exports(
        suite, objective_metric_label="sharpe", powerbi_parent_dir=tmp_path
    )
    assert out["permutation_vector_shuffle_csv"].exists()
    df = pd.read_csv(out["permutation_vector_shuffle_csv"])
    assert list(df.columns) == [
        "feature_name",
        "feature_type",
        "objective_metric",
        "param_combo",
        "param_combo_label",
        "observed_metric",
        "p_value",
        "passed",
        "alpha",
        "n_reps",
        "critical_value",
    ]


def test_permutation_vector_shuffle_records_resolves_readable_label() -> None:
    grid_params = {"lookback": 14}
    combo_key = _param_combo_name(grid_params)
    null = np.array([0.01], dtype=float)
    r1 = VectorShuffleReport(
        param_combo=combo_key,
        original_metric=0.1,
        null_distribution=null,
        critical_value=0.05,
        p_value=0.2,
        passed=False,
        alpha=0.1,
        nreps=1,
    )
    wf = WalkforwardStabilityReport(
        feature_name="f",
        feature_type="signed_signal",
        fold_results=[],
        consistency_metrics={},
        is_stable=False,
        stability_verdict="Stage 3 (walkforward permutation) removed.",
        top_k=0,
    )
    fs = FunnelStatistics(
        total_params=1,
        stage1_pass=0,
        stage2_pass=0,
        stable_params=0,
        ensemble_candidates=0,
        computational_savings_pct=0.0,
    )
    suite = PermutationTestSuite(
        feature_name="feat",
        feature_type="signed_signal",
        stage1_reports={combo_key: r1},
        stage2_reports={},
        stage3_report=wf,
        funnel_stats=fs,
        ensemble_candidates=[],
        summary="test",
        combo_decisions={},
    )
    recs = permutation_vector_shuffle_records(suite, "mean_return", param_grid=[grid_params])
    assert recs[0]["param_combo"] == combo_key
    assert recs[0]["param_combo_label"] == "lookback_14"
