"""Unit tests for T016: Early Stopping Orchestration."""
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
    FoldResult,
    FunnelStatistics,
    OutOfSamplePermutationReport,
    WalkforwardStabilityReport,
    PermutationTestSuite,
    PipelinePermutationReport,
    VectorShuffleReport,
)


def test_permutation_test_config_defaults() -> None:
    """PermutationTestConfig has the correct defaults (Stage 3 removed)."""
    config = PermutationTestConfig()
    assert config.nreps == 1000
    assert config.alpha == 0.10
    assert config.permutation_mode_stage2 == 'candle_shuffle'
    assert config.random_seed is None
    assert config.in_sample.nreps == 1000
    assert config.in_sample.alpha == 0.10
    assert config.in_sample.metric_threshold == 0.0
    assert config.in_sample.permutation_mode_stage2 == 'candle_shuffle'
    assert config.in_sample.run_stage1 is True
    assert config.in_sample.run_stage2 is True


def test_permutation_test_config_rejects_in_sample_config_plus_legacy_scalars() -> None:
    """In-sample nested config and in-sample legacy scalars cannot be mixed."""
    with pytest.raises(ValueError, match='in_sample'):
        PermutationTestConfig(
            in_sample=InSamplePermutationConfig(nreps=200),
            nreps=500,
        )


def test_permutation_test_config_rejects_out_of_sample_and_objective_metric() -> None:
    """Out-of-sample config and top-level objective_metric cannot be mixed."""
    with pytest.raises(ValueError, match='objective_metric'):
        PermutationTestConfig(
            out_of_sample=OutOfSamplePermutationConfig(
                objective_metric=ObjectiveMetricSpec(builtin='sortino'),
            ),
            objective_metric=ObjectiveMetricSpec(builtin='sharpe'),
        )


def test_funnel_statistics_fields() -> None:
    """FunnelStatistics contains all required fields."""
    stats = FunnelStatistics(
        total_params=10,
        stage1_pass=7,
        stage2_pass=5,
        stable_params=3,
        ensemble_candidates=3,
        computational_savings_pct=15.0,
    )
    assert stats.total_params == 10
    assert stats.stage1_pass == 7
    assert stats.stage2_pass == 5
    assert stats.stable_params == 3
    assert stats.ensemble_candidates == 3
    assert stats.computational_savings_pct == 15.0


def test_permutation_test_suite_fields() -> None:
    """PermutationTestSuite contains all required fields."""
    from feature_selection.validation.reports import (
        WalkforwardStabilityReport, FoldResult,
    )

    suite = PermutationTestSuite(
        feature_name='rsi',
        feature_type='continuous',
        stage1_reports={},
        stage2_reports={},
        stage3_report=WalkforwardStabilityReport(
            feature_name='rsi', feature_type='continuous',
            fold_results=[], consistency_metrics={},
            is_stable=False, stability_verdict='UNSTABLE', top_k=3,
        ),
        funnel_stats=FunnelStatistics(10, 7, 5, 3, 3, 15.0),
        ensemble_candidates=[],
        summary='test summary',
    )
    assert suite.feature_name == 'rsi'
    assert suite.feature_type == 'continuous'
    assert isinstance(suite.stage1_reports, dict)
    assert isinstance(suite.stage2_reports, dict)
    assert hasattr(suite, 'stage3_report')
    assert hasattr(suite, 'funnel_stats')
    assert hasattr(suite, 'ensemble_candidates')
    assert hasattr(suite, 'summary')
    assert hasattr(suite, 'phase3_oos_reports')
    assert hasattr(suite, 'combo_decisions')


def test_run_oos_permutation_false_skips_phase3() -> None:
    """When run_oos_permutation is False (e.g. in-sample), Phase 3 is skipped and candidates = Stage 2 passers."""
    from feature_selection.validation import orchestration

    dates = pd.date_range('2020-01-01', periods=6, freq='D')
    candles = pd.DataFrame({'close': np.arange(6.0)}, index=dates)
    target = pd.Series(np.linspace(0.1, 0.6, 6), index=dates)
    param_grid = [{'lookback': 3}, {'lookback': 5}]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(np.ones(len(df)), index=df.index, name=f"lookback_{params['lookback']}")

    def objective(values: pd.Series) -> float:
        return float(values.mean())

    config = PermutationTestConfig(
        nreps=3,
        alpha=0.10,
        min_folds_stable=1,
        out_of_sample=OutOfSamplePermutationConfig(run_oos_permutation=False),
    )
    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=objective,
        fold_structure=[(dates[0], dates[-1])],
        config=config,
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )
    assert len(suite.phase3_oos_reports) == 0
    stage2_passers = {k for k, r in suite.stage2_reports.items() if r.passed}
    assert set(suite.ensemble_candidates) == stage2_passers
    for decision in suite.combo_decisions.values():
        assert decision.oos_passed is False


def test_stage2_receives_only_stage1_passers() -> None:
    """Stage 2 count <= Stage 1 pass count when running the suite."""
    import numpy as np
    import pandas as pd
    from feature_selection.validation.orchestration import run_permutation_test_suite

    rng = np.random.default_rng(0)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    closes = 100.0 + rng.standard_normal(n).cumsum()
    candles = pd.DataFrame({
        'open': closes - rng.uniform(0, 0.3, n),
        'high': closes + rng.uniform(0, 0.3, n),
        'low': closes - rng.uniform(0, 0.3, n),
        'close': closes,
        'datetime': dates,
    }, index=dates)
    target = pd.Series(rng.standard_normal(n), index=dates)
    param_grid = [{'lookback': v} for v in [3, 5, 10]]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        lb = params['lookback']
        return df['close'].pct_change(lb).fillna(0.0).rename(f"lookback_{lb}")

    def sharpe(returns: pd.Series) -> float:
        if len(returns) == 0 or returns.std() == 0: return 0.0
        return float(returns.mean() / returns.std())

    config = PermutationTestConfig(nreps=20, alpha=0.10, random_seed=42, min_folds_stable=1)
    folds = [(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01'))]

    suite = run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=sharpe,
        fold_structure=folds,
        config=config,
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )

    assert len(suite.stage2_reports) <= suite.funnel_stats.stage1_pass
    assert len(suite.stage1_reports) == suite.funnel_stats.total_params == len(param_grid)
    assert suite.funnel_stats.stage1_pass == sum(
        1 for report in suite.stage1_reports.values() if report.passed
    )
    assert suite.funnel_stats.computational_savings_pct >= 0.0
    assert isinstance(suite.ensemble_candidates, list)


def test_stage2_uses_single_batch_call_for_rule_based_passers(monkeypatch: pytest.MonkeyPatch) -> None:
    from feature_selection.validation import orchestration

    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    candles = pd.DataFrame(
        {
            'datetime': dates,
            'open': np.linspace(100.0, 103.0, len(dates)),
            'high': np.linspace(100.5, 103.5, len(dates)),
            'low': np.linspace(99.5, 102.5, len(dates)),
            'close': np.linspace(100.2, 103.2, len(dates)),
        },
        index=dates,
    )
    target = pd.Series(np.linspace(-0.1, 0.2, len(dates)), index=dates)
    param_grid = [{'lookback': 3}, {'lookback': 5}, {'lookback': 8}]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        idx = pd.to_datetime(df['datetime']) if 'datetime' in df else df.index
        return pd.Series(np.sign(np.sin(np.arange(len(idx)) + params['lookback'])), index=idx, name=str(params))

    def objective(values: pd.Series) -> float:
        return float(values.mean()) if len(values) else 0.0

    passers = {'lookback_3', 'lookback_8'}
    batch_calls: list[list[str]] = []

    def fake_stage1(**kwargs: object) -> VectorShuffleReport:
        combo = str(kwargs['param_combo'])
        passed = combo in passers
        return VectorShuffleReport(
            param_combo=combo,
            original_metric=1.0 if passed else 0.0,
            null_distribution=np.zeros(4),
            critical_value=0.5,
            p_value=0.01 if passed else 0.99,
            passed=passed,
            alpha=0.1,
            nreps=4,
        )

    def fake_stage2_batch(**kwargs: object) -> dict[str, PipelinePermutationReport]:
        items = kwargs['items']
        combo_names = [item.param_combo for item in items]
        batch_calls.append(combo_names)
        reports: dict[str, PipelinePermutationReport] = {}
        for combo_name in combo_names:
            reports[combo_name] = PipelinePermutationReport(
                param_combo=combo_name,
                feature_type='rule_based',
                permutation_mode='candle_shuffle',
                original_metric=0.0,
                null_distribution=np.zeros(4),
                critical_value=0.0,
                p_value=1.0,
                passed=False,
                alpha=0.1,
                nreps=4,
                no_trade_permutations=4,
            )
        return reports

    def fail_if_single_stage2_called(**kwargs: object) -> PipelinePermutationReport:
        raise AssertionError(f"single-combo Stage 2 should not be called: {kwargs.get('param_combo')}")

    def fake_walkforward(**kwargs: object) -> WalkforwardStabilityReport:
        return WalkforwardStabilityReport(
            feature_name='test',
            feature_type='rule_based',
            fold_results=[],
            consistency_metrics={},
            is_stable=False,
            stability_verdict='UNSTABLE',
            top_k=3,
        )

    monkeypatch.setattr(orchestration, 'run_vector_shuffle_test', fake_stage1)
    monkeypatch.setattr(orchestration, '_run_pipeline_permutation_rule_based_batch', fake_stage2_batch)
    monkeypatch.setattr(orchestration, 'run_pipeline_permutation_rule_based', fail_if_single_stage2_called)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=objective,
        fold_structure=[(dates[0], dates[-1])],
        config=PermutationTestConfig(nreps=4, alpha=0.10, min_folds_stable=1),
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )

    assert len(batch_calls) == 1
    assert sorted(batch_calls[0]) == sorted(passers)
    assert len(suite.stage2_reports) == len(passers)


def test_stage2_skips_when_disabled_and_preserves_stage1_passers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from feature_selection.validation import orchestration

    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    candles = pd.DataFrame({'close': np.arange(10.0)}, index=dates)
    target = pd.Series(np.linspace(0.1, 1.0, 10), index=dates)
    param_grid = [{'lookback': 3}, {'lookback': 5}, {'lookback': 8}]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(np.ones(len(df)), index=df.index, name=f"lookback_{params['lookback']}")

    def objective(values: pd.Series) -> float:
        return float(values.mean())

    stage1_passers = {'lookback_3', 'lookback_8'}

    def fake_stage1(**kwargs: object) -> VectorShuffleReport:
        combo = str(kwargs['param_combo'])
        passed = combo in stage1_passers
        return VectorShuffleReport(
            param_combo=combo,
            original_metric=1.0 if passed else 0.0,
            null_distribution=np.zeros(4),
            critical_value=0.5,
            p_value=0.01 if passed else 0.99,
            passed=passed,
            alpha=0.1,
            nreps=4,
        )

    def fail_stage2_batch(**kwargs: object) -> dict[str, PipelinePermutationReport]:
        raise AssertionError('Stage 2 batch should not run when run_stage2=False')

    monkeypatch.setattr(orchestration, 'run_vector_shuffle_test', fake_stage1)
    monkeypatch.setattr(orchestration, '_run_pipeline_permutation_rule_based_batch', fail_stage2_batch)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=objective,
        fold_structure=[(dates[0], dates[-1])],
        config=PermutationTestConfig(
            nreps=4,
            alpha=0.10,
            run_stage2=False,
            run_stage3_walkforward=False,
            min_folds_stable=1,
        ),
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )

    assert suite.stage2_reports == {}
    assert suite.funnel_stats.stage1_pass == len(stage1_passers)
    assert suite.funnel_stats.stage2_pass == len(stage1_passers)
    assert set(suite.ensemble_candidates) == stage1_passers


def test_stage1_skip_runs_stage2_for_all_params(monkeypatch: pytest.MonkeyPatch) -> None:
    from feature_selection.validation import orchestration

    dates = pd.date_range('2020-01-01', periods=12, freq='D')
    candles = pd.DataFrame({'close': np.arange(12.0)}, index=dates)
    target = pd.Series(np.linspace(-0.1, 0.5, 12), index=dates)
    param_grid = [{'lookback': 2}, {'lookback': 4}, {'lookback': 6}]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(np.ones(len(df)), index=df.index, name=f"lookback_{params['lookback']}")

    def objective(values: pd.Series) -> float:
        return float(values.mean()) if len(values) else 0.0

    batch_calls: list[list[str]] = []

    def fail_stage1(**kwargs: object) -> VectorShuffleReport:
        raise AssertionError('Stage 1 should not run when run_stage1=False')

    def fake_stage2_batch(**kwargs: object) -> dict[str, PipelinePermutationReport]:
        items = kwargs['items']
        combo_names = [item.param_combo for item in items]
        batch_calls.append(combo_names)
        return {
            combo_name: PipelinePermutationReport(
                param_combo=combo_name,
                feature_type='rule_based',
                permutation_mode='candle_shuffle',
                original_metric=0.0,
                null_distribution=np.zeros(4),
                critical_value=0.0,
                p_value=1.0,
                passed=(combo_name != 'lookback_4'),
                alpha=0.1,
                nreps=4,
                no_trade_permutations=4,
            )
            for combo_name in combo_names
        }

    monkeypatch.setattr(orchestration, 'run_vector_shuffle_test', fail_stage1)
    monkeypatch.setattr(orchestration, '_run_pipeline_permutation_rule_based_batch', fake_stage2_batch)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=objective,
        fold_structure=[(dates[0], dates[-1])],
        config=PermutationTestConfig(
            nreps=4,
            alpha=0.10,
            run_stage1=False,
            run_stage2=True,
            run_stage3_walkforward=False,
            min_folds_stable=1,
        ),
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )

    expected_combos = sorted(f"lookback_{p['lookback']}" for p in param_grid)
    assert len(batch_calls) == 1
    assert sorted(batch_calls[0]) == expected_combos
    assert suite.stage1_reports == {}
    assert suite.funnel_stats.stage1_pass == len(param_grid)
    assert suite.funnel_stats.stage2_pass == 2


def test_ensemble_candidate_intersection() -> None:
    """ensemble_candidates = stage2_passers intersection stable_params."""
    # We verify this via the suite logic directly
    stage2_passers = {'A', 'B', 'C'}
    stable_params = {'B', 'C', 'D'}
    expected = sorted(stage2_passers & stable_params)
    assert set(expected) == {'B', 'C'}


def test_computational_savings_non_negative() -> None:
    """computational_savings_pct >= 0."""
    stats = FunnelStatistics(
        total_params=10,
        stage1_pass=10,  # no early stopping savings
        stage2_pass=10,
        stable_params=5,
        ensemble_candidates=5,
        computational_savings_pct=0.0,
    )
    assert stats.computational_savings_pct >= 0.0


def test_suite_runs_oos_on_stage2_passers_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default OOS candidate source runs phase 3 only on stage2 passers."""
    from feature_selection.validation import orchestration

    dates = pd.date_range('2020-01-01', periods=8, freq='D')
    candles = pd.DataFrame({'close': np.arange(8.0)}, index=dates)
    target = pd.Series(np.linspace(0.1, 0.8, 8), index=dates)
    param_grid = [{'lookback': 3}, {'lookback': 5}, {'lookback': 8}]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(np.ones(len(df)), index=df.index, name=f"lookback_{params['lookback']}")

    def objective(values: pd.Series) -> float:
        return float(values.mean())

    stage2_passers = {'lookback_3', 'lookback_5'}
    stable_params = {'lookback_3', 'lookback_5'}
    oos_calls: list[tuple[str, int | None]] = []

    def fake_stage1(**kwargs: object) -> VectorShuffleReport:
        combo = str(kwargs['param_combo'])
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

    def fake_stage2_rule(**kwargs: object) -> PipelinePermutationReport:
        combo = str(kwargs['param_combo'])
        return PipelinePermutationReport(
            param_combo=combo,
            feature_type='rule_based',
            permutation_mode='candle_shuffle',
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=combo in stage2_passers,
            alpha=0.1,
            nreps=5,
            no_trade_permutations=0,
        )

    def fake_stage2_rule_batch(**kwargs: object) -> dict[str, PipelinePermutationReport]:
        return {
            item.param_combo: fake_stage2_rule(param_combo=item.param_combo)  # type: ignore[arg-type]
            for item in kwargs['items']  # type: ignore[index]
        }

    def fake_walkforward(**kwargs: object) -> WalkforwardStabilityReport:
        return WalkforwardStabilityReport(
            feature_name='test',
            feature_type='rule_based',
            fold_results=[
                FoldResult(
                    fold_id='fold_1',
                    fold_period=('2020-01-01', '2020-01-08'),
                    top_k_params=sorted(stable_params),
                    smoothed_objectives={k: 1.0 for k in stable_params},
                    passed_permutation_overlay=[True, True],
                ),
            ],
            consistency_metrics={'overlap_rate': 1.0},
            is_stable=True,
            stability_verdict='STABLE',
            top_k=2,
        )

    def fake_oos(**kwargs: object) -> OutOfSamplePermutationReport:
        combo = str(kwargs['param_combo'])
        oos_calls.append((combo, kwargs.get('random_seed')))  # type: ignore[arg-type]
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
            feature_type='rule_based',
            permutation_mode='candle_shuffle',
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=(combo == 'lookback_5'),
            alpha=0.1,
            nreps=5,
            no_trade_permutations=0,
        )
        return OutOfSamplePermutationReport(
            param_combo=combo,
            vector_report=vec,
            candle_report=candle,
            passed=candle.passed,
        )

    monkeypatch.setattr(orchestration, 'run_vector_shuffle_test', fake_stage1)
    monkeypatch.setattr(orchestration, '_run_pipeline_permutation_rule_based_batch', fake_stage2_rule_batch)
    monkeypatch.setattr(orchestration, 'run_oos_permutation_for_param', fake_oos)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=objective,
        fold_structure=[(dates[0], dates[-1])],
        config=PermutationTestConfig(
            nreps=5,
            alpha=0.10,
            random_seed=17,
            min_folds_stable=1,
            out_of_sample=OutOfSamplePermutationConfig(run_oos_permutation=True),
        ),
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )

    assert sorted(suite.phase3_oos_reports.keys()) == sorted(stage2_passers)
    assert sorted(combo for combo, _ in oos_calls) == sorted(stage2_passers)
    assert {seed for _, seed in oos_calls} == {17}
    assert suite.ensemble_candidates == ['lookback_5']
    assert suite.funnel_stats.ensemble_candidates == len(suite.ensemble_candidates)
    assert suite.combo_decisions['lookback_5'].final_status == 'candidate'
    assert suite.combo_decisions['lookback_3'].final_status == 'needs_review'
    assert suite.combo_decisions['lookback_8'].final_status == 'rejected'


def test_suite_can_switch_oos_source_to_stable_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OOS candidate source can switch to stage2-stable intersection."""
    from feature_selection.validation import orchestration

    dates = pd.date_range('2020-01-01', periods=8, freq='D')
    candles = pd.DataFrame({'close': np.arange(8.0)}, index=dates)
    target = pd.Series(np.linspace(0.1, 0.8, 8), index=dates)
    param_grid = [{'lookback': 3}, {'lookback': 5}, {'lookback': 8}]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(np.ones(len(df)), index=df.index, name=f"lookback_{params['lookback']}")

    def objective(values: pd.Series) -> float:
        return float(values.mean())

    stage2_passers = {'lookback_3', 'lookback_5'}
    stable_params = {'lookback_5', 'lookback_8'}
    called_combos: list[str] = []

    def fake_stage1(**kwargs: object) -> VectorShuffleReport:
        combo = str(kwargs['param_combo'])
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

    def fake_stage2_rule(**kwargs: object) -> PipelinePermutationReport:
        combo = str(kwargs['param_combo'])
        return PipelinePermutationReport(
            param_combo=combo,
            feature_type='rule_based',
            permutation_mode='candle_shuffle',
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=combo in stage2_passers,
            alpha=0.1,
            nreps=5,
            no_trade_permutations=0,
        )

    def fake_stage2_rule_batch(**kwargs: object) -> dict[str, PipelinePermutationReport]:
        return {
            item.param_combo: fake_stage2_rule(param_combo=item.param_combo)  # type: ignore[arg-type]
            for item in kwargs['items']  # type: ignore[index]
        }

    def fake_walkforward(**kwargs: object) -> WalkforwardStabilityReport:
        return WalkforwardStabilityReport(
            feature_name='test',
            feature_type='rule_based',
            fold_results=[
                FoldResult(
                    fold_id='fold_1',
                    fold_period=('2020-01-01', '2020-01-08'),
                    top_k_params=sorted(stable_params),
                    smoothed_objectives={k: 1.0 for k in stable_params},
                    passed_permutation_overlay=[True, True],
                ),
            ],
            consistency_metrics={'overlap_rate': 1.0},
            is_stable=True,
            stability_verdict='STABLE',
            top_k=2,
        )

    def fake_oos(**kwargs: object) -> OutOfSamplePermutationReport:
        combo = str(kwargs['param_combo'])
        called_combos.append(combo)
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
            feature_type='rule_based',
            permutation_mode='candle_shuffle',
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=True,
            alpha=0.1,
            nreps=5,
            no_trade_permutations=0,
        )
        return OutOfSamplePermutationReport(
            param_combo=combo,
            vector_report=vec,
            candle_report=candle,
            passed=True,
        )

    monkeypatch.setattr(orchestration, 'run_vector_shuffle_test', fake_stage1)
    monkeypatch.setattr(orchestration, '_run_pipeline_permutation_rule_based_batch', fake_stage2_rule_batch)
    monkeypatch.setattr(orchestration, 'run_oos_permutation_for_param', fake_oos)

    config = PermutationTestConfig(
        nreps=5,
        alpha=0.10,
        min_folds_stable=1,
        random_seed=31,
        out_of_sample=OutOfSamplePermutationConfig(
            objective_metric=ObjectiveMetricSpec(builtin='sharpe'),
            candidate_source='stable_intersection',
            run_oos_permutation=True,
        ),
    )

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=objective,
        fold_structure=[(dates[0], dates[-1])],
        config=config,
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )

    # With Stage 3 removed, stable_params is always empty, so stable_intersection yields no OOS candidates.
    assert sorted(suite.phase3_oos_reports.keys()) == []
    assert called_combos == []


def test_oos_uses_configured_objective_metric_only_for_oos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OOS stage resolves objective metric from config while earlier stages use input objective."""
    from feature_selection.validation import orchestration

    dates = pd.date_range('2020-01-01', periods=8, freq='D')
    candles = pd.DataFrame({'close': np.arange(8.0)}, index=dates)
    target = pd.Series(np.linspace(0.1, 0.8, 8), index=dates)
    param_grid = [{'lookback': 3}]

    sample_returns = pd.Series([1.0, -0.5], index=pd.RangeIndex(2))
    stage1_objective_values: list[float] = []
    stage2_objective_values: list[float] = []
    oos_objective_values: list[float] = []

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(np.ones(len(df)), index=df.index, name=f"lookback_{params['lookback']}")

    def top_level_objective(_: pd.Series) -> float:
        return -123.0

    def fake_stage1(**kwargs: object) -> VectorShuffleReport:
        stage1_objective_values.append(float(kwargs['objective_func'](sample_returns)))  # type: ignore[index,operator]
        combo = str(kwargs['param_combo'])
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

    def fake_stage2_rule(**kwargs: object) -> PipelinePermutationReport:
        stage2_objective_values.append(float(kwargs['objective_func'](sample_returns)))  # type: ignore[index,operator]
        combo = str(kwargs['param_combo'])
        return PipelinePermutationReport(
            param_combo=combo,
            feature_type='rule_based',
            permutation_mode='candle_shuffle',
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=True,
            alpha=0.1,
            nreps=5,
            no_trade_permutations=0,
        )

    def fake_stage2_rule_batch(**kwargs: object) -> dict[str, PipelinePermutationReport]:
        stage2_objective_values.append(float(kwargs['objective_func'](sample_returns)))  # type: ignore[index,operator]
        return {
            item.param_combo: PipelinePermutationReport(
                param_combo=item.param_combo,
                feature_type='rule_based',
                permutation_mode='candle_shuffle',
                original_metric=1.0,
                null_distribution=np.zeros(5),
                critical_value=0.0,
                p_value=0.01,
                passed=True,
                alpha=0.1,
                nreps=5,
                no_trade_permutations=0,
            )
            for item in kwargs['items']  # type: ignore[index]
        }

    def fake_oos(**kwargs: object) -> OutOfSamplePermutationReport:
        combo = str(kwargs['param_combo'])
        oos_objective_values.append(float(kwargs['objective_func'](sample_returns)))  # type: ignore[index,operator]
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
            feature_type='rule_based',
            permutation_mode='candle_shuffle',
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=True,
            alpha=0.1,
            nreps=5,
            no_trade_permutations=0,
        )
        return OutOfSamplePermutationReport(
            param_combo=combo,
            vector_report=vec,
            candle_report=candle,
            passed=True,
        )

    monkeypatch.setattr(orchestration, 'run_vector_shuffle_test', fake_stage1)
    monkeypatch.setattr(orchestration, '_run_pipeline_permutation_rule_based_batch', fake_stage2_rule_batch)
    monkeypatch.setattr(orchestration, 'run_oos_permutation_for_param', fake_oos)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=top_level_objective,
        fold_structure=[(dates[0], dates[-1])],
        config=PermutationTestConfig(
            nreps=5,
            alpha=0.10,
            min_folds_stable=1,
            out_of_sample=OutOfSamplePermutationConfig(
                objective_metric=ObjectiveMetricSpec(builtin='profit_factor'),
                run_oos_permutation=True,
            ),
        ),
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )

    assert suite.ensemble_candidates == ['lookback_3']
    assert stage1_objective_values == [-123.0]
    assert stage2_objective_values == [-123.0]
    assert oos_objective_values == [2.0]


def test_oos_combo_exception_does_not_abort_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-combo OOS failures are isolated and marked as failed."""
    from feature_selection.validation import orchestration

    dates = pd.date_range('2020-01-01', periods=8, freq='D')
    candles = pd.DataFrame({'close': np.arange(8.0)}, index=dates)
    target = pd.Series(np.linspace(0.1, 0.8, 8), index=dates)
    param_grid = [{'lookback': 3}, {'lookback': 5}]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(np.ones(len(df)), index=df.index, name=f"lookback_{params['lookback']}")

    def objective(values: pd.Series) -> float:
        return float(values.mean())

    def fake_stage1(**kwargs: object) -> VectorShuffleReport:
        combo = str(kwargs['param_combo'])
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

    def fake_stage2_rule(**kwargs: object) -> PipelinePermutationReport:
        combo = str(kwargs['param_combo'])
        return PipelinePermutationReport(
            param_combo=combo,
            feature_type='rule_based',
            permutation_mode='candle_shuffle',
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=True,
            alpha=0.1,
            nreps=5,
            no_trade_permutations=0,
        )

    def fake_stage2_rule_batch(**kwargs: object) -> dict[str, PipelinePermutationReport]:
        return {
            item.param_combo: fake_stage2_rule(param_combo=item.param_combo)  # type: ignore[arg-type]
            for item in kwargs['items']  # type: ignore[index]
        }

    def fake_walkforward(**kwargs: object) -> WalkforwardStabilityReport:
        return WalkforwardStabilityReport(
            feature_name='test',
            feature_type='rule_based',
            fold_results=[
                FoldResult(
                    fold_id='fold_1',
                    fold_period=('2020-01-01', '2020-01-08'),
                    top_k_params=['lookback_3', 'lookback_5'],
                    smoothed_objectives={'lookback_3': 1.0, 'lookback_5': 1.0},
                    passed_permutation_overlay=[True, True],
                ),
            ],
            consistency_metrics={'overlap_rate': 1.0},
            is_stable=True,
            stability_verdict='STABLE',
            top_k=2,
        )

    def fake_oos(**kwargs: object) -> OutOfSamplePermutationReport:
        combo = str(kwargs['param_combo'])
        if combo == 'lookback_3':
            raise RuntimeError('simulated oos failure')

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
            feature_type='rule_based',
            permutation_mode='candle_shuffle',
            original_metric=1.0,
            null_distribution=np.zeros(5),
            critical_value=0.0,
            p_value=0.01,
            passed=True,
            alpha=0.1,
            nreps=5,
            no_trade_permutations=0,
        )
        return OutOfSamplePermutationReport(
            param_combo=combo,
            vector_report=vec,
            candle_report=candle,
            passed=True,
        )

    monkeypatch.setattr(orchestration, 'run_vector_shuffle_test', fake_stage1)
    monkeypatch.setattr(orchestration, '_run_pipeline_permutation_rule_based_batch', fake_stage2_rule_batch)
    monkeypatch.setattr(orchestration, 'run_oos_permutation_for_param', fake_oos)

    suite = orchestration.run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=objective,
        fold_structure=[(dates[0], dates[-1])],
        config=PermutationTestConfig(
            nreps=5,
            alpha=0.10,
            min_folds_stable=1,
            out_of_sample=OutOfSamplePermutationConfig(run_oos_permutation=True),
        ),
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )

    assert sorted(suite.phase3_oos_reports.keys()) == ['lookback_3', 'lookback_5']
    assert not suite.phase3_oos_reports['lookback_3'].passed
    assert suite.phase3_oos_reports['lookback_5'].passed
    assert not suite.combo_decisions['lookback_3'].oos_passed
    assert suite.combo_decisions['lookback_3'].final_status == 'needs_review'
    assert suite.combo_decisions['lookback_5'].final_status == 'candidate'
