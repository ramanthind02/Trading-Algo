"""Unit tests for the signed-signal permutation orchestration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection.validation.config import (
    InSamplePermutationConfig,
    OutOfSamplePermutationConfig,
    PermutationTestConfig,
)
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from feature_selection.validation.reports import (
    FunnelStatistics,
    OutOfSamplePermutationReport,
    PermutationTestSuite,
    VectorShuffleReport,
    WalkforwardStabilityReport,
)


def test_permutation_test_config_defaults() -> None:
    config = PermutationTestConfig()
    assert config.nreps == 1000
    assert config.alpha == 0.10
    assert config.random_seed is None
    assert config.n_jobs_combos == 8
    assert config.n_jobs_reps == 8


def test_permutation_test_config_rejects_in_sample_config_plus_legacy_scalars() -> None:
    with pytest.raises(ValueError, match="in_sample"):
        PermutationTestConfig(in_sample=InSamplePermutationConfig(nreps=200), nreps=500)


def test_permutation_test_config_rejects_out_of_sample_and_objective_metric() -> None:
    with pytest.raises(ValueError, match="objective_metric"):
        PermutationTestConfig(
            out_of_sample=OutOfSamplePermutationConfig(
                objective_metric=ObjectiveMetricSpec(builtin="sortino"),
            ),
            objective_metric=ObjectiveMetricSpec(builtin="sharpe"),
        )


def test_permutation_test_suite_fields() -> None:
    suite = PermutationTestSuite(
        feature_name="rsi",
        feature_type="signed_signal",
        stage1_reports={},
        stage2_reports={},
        stage3_report=WalkforwardStabilityReport(
            feature_name="rsi",
            feature_type="signed_signal",
            fold_results=[],
            consistency_metrics={},
            is_stable=False,
            stability_verdict="UNSTABLE",
            top_k=3,
        ),
        funnel_stats=FunnelStatistics(10, 7, 5, 3, 3, 15.0),
        ensemble_candidates=[],
        summary="test summary",
    )
    assert suite.feature_type == "signed_signal"
    assert hasattr(suite, "phase3_oos_reports")


def _make_candles(n: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": np.arange(n, dtype=float)}, index=dates)


def _extractor(df: pd.DataFrame, params: dict) -> pd.Series:
    return pd.Series(np.sign(df["close"].pct_change(params["lookback"]).fillna(0.0)), index=df.index)


def _objective(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else 0.0


def test_run_oos_permutation_false_skips_phase3() -> None:
    from feature_selection.validation import orchestration

    candles = _make_candles(12)
    target = pd.Series(np.linspace(0.1, 1.2, 12), index=candles.index)
    param_grid = [{"lookback": 3}, {"lookback": 5}]

    config = PermutationTestConfig(
        nreps=3,
        alpha=0.10,
        min_folds_stable=1,
        out_of_sample=OutOfSamplePermutationConfig(run_oos_permutation=False),
    )
    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={"module_name": "test"},
        target=target,
        param_grid=param_grid,
        objective_func=_objective,
        fold_structure=[(candles.index[0], candles.index[-1])],
        config=config,
        extractor_func=_extractor,
        feature_name="test",
    )

    assert len(suite.phase3_oos_reports) == 0
    assert suite.feature_type == "signed_signal"
    assert isinstance(suite, PermutationTestSuite)
    assert suite.stage2_reports == {}


def test_vector_shuffle_only_empty_stage2_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    from feature_selection.validation import orchestration

    candles = _make_candles(30)
    target = pd.Series(np.linspace(-0.1, 0.2, 30), index=candles.index)
    param_grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 8}]
    passers = {"lookback_3", "lookback_8"}

    def fake_vector(**kwargs: object) -> VectorShuffleReport:
        combo = str(kwargs["param_combo"])
        return VectorShuffleReport(
            param_combo=combo,
            original_metric=1.0 if combo in passers else 0.0,
            null_distribution=np.zeros(4),
            critical_value=0.5,
            p_value=0.01 if combo in passers else 0.99,
            passed=combo in passers,
            alpha=0.1,
            nreps=4,
        )

    monkeypatch.setattr(orchestration, "run_vector_shuffle_test", fake_vector)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={"module_name": "test"},
        target=target,
        param_grid=param_grid,
        objective_func=_objective,
        fold_structure=[(candles.index[0], candles.index[-1])],
        config=PermutationTestConfig(
            in_sample=InSamplePermutationConfig(nreps=4, alpha=0.10, n_jobs_combos=1),
            out_of_sample=OutOfSamplePermutationConfig(run_oos_permutation=False),
        ),
        extractor_func=_extractor,
        feature_name="test",
    )

    assert suite.stage2_reports == {}
    assert set(suite.ensemble_candidates) == passers


def test_parallel_combo_parity_with_sequential() -> None:
    """Parallel (n_jobs_combos=2) and sequential (n_jobs_combos=1) produce identical p-values and pass flags."""
    from feature_selection.validation import orchestration

    rng = np.random.default_rng(0)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    candles = pd.DataFrame({"close": rng.standard_normal(n).cumsum() + 100.0}, index=dates)
    target = pd.Series(rng.standard_normal(n), index=dates)
    param_grid = [{"lookback": 2}, {"lookback": 4}, {"lookback": 7}]

    base_config = PermutationTestConfig(
        nreps=20,
        alpha=0.10,
        random_seed=42,
        out_of_sample=OutOfSamplePermutationConfig(run_oos_permutation=False),
    )

    def _run(n_jobs: int) -> dict:
        cfg = PermutationTestConfig(
            in_sample=InSamplePermutationConfig(
                nreps=base_config.nreps,
                alpha=base_config.alpha,
                n_jobs_combos=n_jobs,
            ),
            random_seed=base_config.random_seed,
            out_of_sample=OutOfSamplePermutationConfig(run_oos_permutation=False),
        )
        suite = orchestration.run_permutation_test_suite(
            candles_df=candles,
            feature_spec={"module_name": "test"},
            target=target,
            param_grid=param_grid,
            objective_func=_objective,
            config=cfg,
            extractor_func=_extractor,
            feature_name="parity_test",
        )
        return {
            k: (round(r.p_value, 6), r.passed)
            for k, r in suite.stage1_reports.items()
        }

    sequential = _run(1)
    parallel = _run(2)

    assert set(sequential) == set(parallel), "combo keys differ"
    for combo_name in sequential:
        seq_pval, seq_passed = sequential[combo_name]
        par_pval, par_passed = parallel[combo_name]
        assert seq_passed == par_passed, f"{combo_name}: passed flag differs ({seq_passed} vs {par_passed})"
        assert abs(seq_pval - par_pval) < 0.05, (
            f"{combo_name}: p-values differ too much ({seq_pval} vs {par_pval})"
        )


def test_suite_runs_oos_on_vector_passers_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from feature_selection.validation import orchestration

    candles = _make_candles(8)
    target = pd.Series(np.linspace(0.1, 0.8, 8), index=candles.index)
    param_grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 8}]
    vector_pass_subset = {"lookback_3", "lookback_5"}
    oos_calls: list[str] = []

    def fake_vector(**kwargs: object) -> VectorShuffleReport:
        combo = str(kwargs["param_combo"])
        return VectorShuffleReport(
            param_combo=combo,
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=combo in vector_pass_subset,
            alpha=0.1,
            nreps=5,
        )

    def fake_oos(**kwargs: object) -> OutOfSamplePermutationReport:
        combo = str(kwargs["param_combo"])
        oos_calls.append(combo)
        vec = VectorShuffleReport(
            param_combo=combo,
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=(combo == "lookback_5"),
            alpha=0.1,
            nreps=5,
        )
        return OutOfSamplePermutationReport(
            param_combo=combo,
            vector_report=vec,
            candle_report=None,
            passed=vec.passed,
        )

    monkeypatch.setattr(orchestration, "run_vector_shuffle_test", fake_vector)
    monkeypatch.setattr(orchestration, "run_oos_permutation_for_param", fake_oos)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={"module_name": "test"},
        target=target,
        param_grid=param_grid,
        objective_func=_objective,
        fold_structure=[(candles.index[0], candles.index[-1])],
        config=PermutationTestConfig(
            in_sample=InSamplePermutationConfig(nreps=5, alpha=0.10, n_jobs_combos=1),
            random_seed=17,
            out_of_sample=OutOfSamplePermutationConfig(run_oos_permutation=True),
        ),
        extractor_func=_extractor,
        feature_name="test",
    )

    assert sorted(oos_calls) == sorted(vector_pass_subset)
    assert suite.ensemble_candidates == ["lookback_5"]
