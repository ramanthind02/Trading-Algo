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
    PipelinePermutationReport,
    VectorShuffleReport,
    WalkforwardStabilityReport,
)


def test_permutation_test_config_defaults() -> None:
    config = PermutationTestConfig()
    assert config.nreps == 1000
    assert config.alpha == 0.10
    assert config.permutation_mode_stage2 == "candle_shuffle"
    assert config.random_seed is None


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


def test_stage2_receives_only_stage1_passers(monkeypatch: pytest.MonkeyPatch) -> None:
    from feature_selection.validation import orchestration

    candles = _make_candles(30)
    target = pd.Series(np.linspace(-0.1, 0.2, 30), index=candles.index)
    param_grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 8}]
    passers = {"lookback_3", "lookback_8"}
    batch_calls: list[list[str]] = []

    def fake_stage1(**kwargs: object) -> VectorShuffleReport:
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

    def fake_stage2_batch(**kwargs: object) -> dict[str, PipelinePermutationReport]:
        combo_names = [item.param_combo for item in kwargs["items"]]  # type: ignore[index]
        batch_calls.append(combo_names)
        return {
            combo_name: PipelinePermutationReport(
                param_combo=combo_name,
                feature_type="signed_signal",
                permutation_mode="candle_shuffle",
                original_metric=0.0,
                null_distribution=np.zeros(4),
                critical_value=0.0,
                p_value=1.0,
                passed=False,
                alpha=0.1,
                nreps=4,
                no_trade_permutations=4,
            )
            for combo_name in combo_names
        }

    monkeypatch.setattr(orchestration, "run_vector_shuffle_test", fake_stage1)
    monkeypatch.setattr(orchestration, "_run_pipeline_permutation_batch", fake_stage2_batch)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={"module_name": "test"},
        target=target,
        param_grid=param_grid,
        objective_func=_objective,
        fold_structure=[(candles.index[0], candles.index[-1])],
        config=PermutationTestConfig(nreps=4, alpha=0.10, min_folds_stable=1),
        extractor_func=_extractor,
        feature_name="test",
    )

    assert len(batch_calls) == 1
    assert sorted(batch_calls[0]) == sorted(passers)
    assert len(suite.stage2_reports) == len(passers)


def test_suite_runs_oos_on_stage2_passers_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from feature_selection.validation import orchestration

    candles = _make_candles(8)
    target = pd.Series(np.linspace(0.1, 0.8, 8), index=candles.index)
    param_grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 8}]
    stage2_passers = {"lookback_3", "lookback_5"}
    oos_calls: list[tuple[str, int | None]] = []

    def fake_stage1(**kwargs: object) -> VectorShuffleReport:
        combo = str(kwargs["param_combo"])
        return VectorShuffleReport(
            param_combo=combo,
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=True,
            alpha=0.1,
            nreps=5,
        )

    def fake_stage2_batch(**kwargs: object) -> dict[str, PipelinePermutationReport]:
        return {
            item.param_combo: PipelinePermutationReport(
                param_combo=item.param_combo,
                feature_type="signed_signal",
                permutation_mode="candle_shuffle",
                original_metric=1.0,
                null_distribution=np.zeros(5),
                critical_value=0.0,
                p_value=0.01,
                passed=item.param_combo in stage2_passers,
                alpha=0.1,
                nreps=5,
                no_trade_permutations=0,
            )
            for item in kwargs["items"]  # type: ignore[index]
        }

    def fake_oos(**kwargs: object) -> OutOfSamplePermutationReport:
        combo = str(kwargs["param_combo"])
        oos_calls.append((combo, kwargs.get("random_seed")))  # type: ignore[arg-type]
        vec = VectorShuffleReport(
            param_combo=combo,
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=True,
            alpha=0.1,
            nreps=5,
        )
        candle = PipelinePermutationReport(
            param_combo=combo,
            feature_type="signed_signal",
            permutation_mode="candle_shuffle",
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=(combo == "lookback_5"),
            alpha=0.1,
            nreps=5,
            no_trade_permutations=0,
        )
        return OutOfSamplePermutationReport(param_combo=combo, vector_report=vec, candle_report=candle, passed=candle.passed)

    monkeypatch.setattr(orchestration, "run_vector_shuffle_test", fake_stage1)
    monkeypatch.setattr(orchestration, "_run_pipeline_permutation_batch", fake_stage2_batch)
    monkeypatch.setattr(orchestration, "run_oos_permutation_for_param", fake_oos)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={"module_name": "test"},
        target=target,
        param_grid=param_grid,
        objective_func=_objective,
        fold_structure=[(candles.index[0], candles.index[-1])],
        config=PermutationTestConfig(
            nreps=5,
            alpha=0.10,
            random_seed=17,
            min_folds_stable=1,
            out_of_sample=OutOfSamplePermutationConfig(run_oos_permutation=True),
        ),
        extractor_func=_extractor,
        feature_name="test",
    )

    assert sorted(suite.phase3_oos_reports.keys()) == sorted(stage2_passers)
    assert sorted(combo for combo, _ in oos_calls) == sorted(stage2_passers)
    assert {seed for _, seed in oos_calls} == {17}
    assert suite.ensemble_candidates == ["lookback_5"]
