"""
T013 + T014: Permutation tests for the Feature Validator pipeline.

T013 — Vector Shuffle: Quick in-memory test that shuffles fitted position
        multipliers to build a null distribution.
T014 — Pipeline Permutation: Full-pipeline test that either shuffles raw
        feature values (feature_shuffle) or re-extracts features from
        shuffled candles (candle_shuffle).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validation.reports import (
    OutOfSamplePermutationReport,
    PipelinePermutationReport,
    VectorShuffleReport,
)


# ---------------------------------------------------------------------------
# T013 — Vector Shuffle Permutation Test
# ---------------------------------------------------------------------------

def run_vector_shuffle_test(
    fitted_feature: pd.Series,
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
    param_combo: str = 'default',
) -> VectorShuffleReport:
    """Stage 1: Vector Shuffle permutation test.

    Shuffles the fitted feature vector (position multipliers or discrete
    signals) to build a null distribution, preserving the marginal
    distribution of the signal.  The original metric must exceed the
    (1-alpha) quantile of the null distribution to pass.

    Args:
        fitted_feature: Already-fitted signal vector (e.g. position
            multipliers from a binning model, or ±1 / 0 from a rule-based
            model).  Active positions are those where the signal != 0.
        target: Forward-return series aligned to fitted_feature's index.
        objective_func: Callable(returns: pd.Series) -> float.  Evaluated
            on the returns of active positions only.
        nreps: Number of shuffle replications.
        alpha: Significance level; critical_value = (1-alpha) quantile of
            the null distribution.
        random_seed: Seed for ``numpy.random.default_rng`` for full
            reproducibility.
        param_combo: Human-readable label for this parameter combination.

    Returns:
        VectorShuffleReport with null_distribution, p_value, critical_value,
        and pass/fail verdict.

    References:
        docs/library/Feature_selection/Phase_1_IS/permutation_testing.md (Stage 1)
    """
    rng = np.random.default_rng(random_seed)

    aligned_target = target.reindex(fitted_feature.index)

    # Original metric on active (non-zero) positions
    active_mask = fitted_feature != 0.0
    original_returns = aligned_target * fitted_feature
    if active_mask.any():
        original_metric = objective_func(original_returns[active_mask])
    else:
        original_metric = 0.0

    feature_values = fitted_feature.values.copy()
    null_metrics = np.empty(nreps, dtype=float)

    for i in range(nreps):
        shuffled_values = rng.permutation(feature_values)
        shuffled_series = pd.Series(shuffled_values, index=fitted_feature.index)
        shuffled_active = shuffled_series != 0.0
        shuffled_returns = aligned_target * shuffled_series
        if shuffled_active.any():
            null_metrics[i] = objective_func(shuffled_returns[shuffled_active])
        else:
            null_metrics[i] = 0.0

    # Inclusive p-value: count original as one of the null sample (min p = 1/(nreps+1))
    n_null_ge_original = int((null_metrics >= original_metric).sum())
    p_value = float(1 + n_null_ge_original) / float(nreps + 1)
    critical_value = float(np.percentile(null_metrics, (1.0 - alpha) * 100.0))
    passed = bool(original_metric > critical_value)

    return VectorShuffleReport(
        param_combo=param_combo,
        original_metric=original_metric,
        null_distribution=null_metrics,
        critical_value=critical_value,
        p_value=p_value,
        passed=passed,
        alpha=alpha,
        nreps=nreps,
    )


# ---------------------------------------------------------------------------
# Shared helpers for T014
# ---------------------------------------------------------------------------

def _compute_metric_from_signals(
    signals: pd.Series,
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
) -> tuple[float, bool]:
    """Compute objective metric from position-multiplier signals.

    Returns:
        (metric_value, is_no_trade) — is_no_trade is True when all signals
        are zero and metric defaults to 0.0.
    """
    active_mask = signals != 0.0
    if not active_mask.any():
        return 0.0, True
    aligned_target = target.reindex(signals.index)
    signal_returns = (aligned_target * signals)[active_mask]
    return float(objective_func(signal_returns)), False


def _fit_and_predict(
    binning_model: BinningModelBase,
    feature: pd.Series,
    target: pd.Series,
    strategy: str = 'long',
) -> tuple[pd.Series, bool]:
    """Fit a deep-copy of binning_model then predict signals.

    Returns:
        (signals, fit_failed) — fit_failed=True means an exception was raised;
        signals is all-zeros in that case.
    """
    model_copy = copy.deepcopy(binning_model)
    try:
        aligned_target = target.reindex(feature.index).dropna()
        aligned_feature = feature.reindex(aligned_target.index).dropna()
        aligned_target = aligned_target.reindex(aligned_feature.index)
        model_copy.fit(aligned_feature, aligned_target)
        signals = model_copy.predict(aligned_feature, strategy=strategy)
        return signals, False
    except Exception:
        return pd.Series(0.0, index=feature.index), True


def _prepare_candles_for_shuffler(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure candles_df has a ``datetime`` column required by CandleShuffler.

    If the DataFrame has a DatetimeIndex but no 'datetime' column, a copy is
    returned with the index materialised as that column.
    """
    if 'datetime' not in candles_df.columns:
        if hasattr(candles_df.index, 'dtype') and str(candles_df.index.dtype).startswith(
            'datetime'
        ):
            df = candles_df.copy()
            df['datetime'] = candles_df.index
            return df
    return candles_df


@dataclass(frozen=True)
class _ContinuousPermutationBatchItem:
    """Internal Stage-2 continuous permutation batch input."""

    param_combo: str
    bias_node_extractor: Callable[[pd.DataFrame], pd.Series]
    binning_model: BinningModelBase


@dataclass(frozen=True)
class _RuleBasedPermutationBatchItem:
    """Internal Stage-2 rule-based permutation batch input."""

    param_combo: str
    rule_extractor: Callable[[pd.DataFrame], pd.Series]


def _derive_permutation_seeds(nreps: int, random_seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    return rng.integers(0, 2 ** 31, size=nreps)


def _build_pipeline_report(
    *,
    param_combo: str,
    feature_type: Literal['continuous', 'rule_based'],
    permutation_mode: Literal['feature_shuffle', 'candle_shuffle'],
    original_metric: float,
    null_metrics: np.ndarray,
    alpha: float,
    nreps: int,
    no_trade_permutations: int,
) -> PipelinePermutationReport:
    n_null_ge_original = int((null_metrics >= original_metric).sum())
    p_value = float(1 + n_null_ge_original) / float(nreps + 1)
    critical_value = float(np.percentile(null_metrics, (1.0 - alpha) * 100.0))
    passed = bool(original_metric > critical_value)
    return PipelinePermutationReport(
        param_combo=param_combo,
        feature_type=feature_type,
        permutation_mode=permutation_mode,
        original_metric=original_metric,
        null_distribution=null_metrics,
        critical_value=critical_value,
        p_value=p_value,
        passed=passed,
        alpha=alpha,
        nreps=nreps,
        no_trade_permutations=no_trade_permutations,
    )


def _run_pipeline_permutation_continuous_batch(
    *,
    candles_df: pd.DataFrame,
    items: list[_ContinuousPermutationBatchItem],
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    permutation_mode: Literal['feature_shuffle', 'candle_shuffle'] = 'candle_shuffle',
    metric_threshold: float = 0.0,
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
) -> dict[str, PipelinePermutationReport]:
    """Internal batch Stage-2 runner sharing shuffled candles across passers.

    Docs source of truth:
    - `docs/library/Feature_selection/Phase_1_IS/permutation_testing.md` (Stage 2 semantics)
    - `docs/library/Feature_selection/Phase_1_IS/candle_permutation.md` (candle shuffle invariants)
    """
    _ = metric_threshold  # Reserved for parity with public API.

    if permutation_mode not in ('feature_shuffle', 'candle_shuffle'):
        raise ValueError(
            f"Unknown permutation_mode: {permutation_mode!r}. "
            "Use 'feature_shuffle' or 'candle_shuffle'."
        )

    if not items:
        return {}

    seeds = _derive_permutation_seeds(nreps, random_seed)
    candles_prepared = _prepare_candles_for_shuffler(candles_df)

    original_features: dict[str, pd.Series] = {}
    original_metrics: dict[str, float] = {}
    null_metrics_by_combo: dict[str, np.ndarray] = {}
    no_trade_counts: dict[str, int] = {}
    items_by_combo = {item.param_combo: item for item in items}

    for item in items:
        original_feature = item.bias_node_extractor(candles_df)
        original_features[item.param_combo] = original_feature
        original_signals, original_fit_failed = _fit_and_predict(item.binning_model, original_feature, target)
        if original_fit_failed:
            original_metrics[item.param_combo] = 0.0
        else:
            original_metrics[item.param_combo], _ = _compute_metric_from_signals(
                original_signals, target, objective_func
            )
        null_metrics_by_combo[item.param_combo] = np.empty(nreps, dtype=float)
        no_trade_counts[item.param_combo] = 0

    prepared_shuffler = None
    if permutation_mode == 'candle_shuffle':
        from utils.evaluation.permutation_test.candle_shuffle import _prepare_candle_shuffle

        prepared_shuffler = _prepare_candle_shuffle(candles_prepared)

    for i in range(nreps):
        shuffled_candles: pd.DataFrame | None = None
        if permutation_mode == 'candle_shuffle':
            assert prepared_shuffler is not None
            shuffled_candles = prepared_shuffler.permute_with_seed(int(seeds[i]))

        for combo_name, item in items_by_combo.items():
            try:
                if permutation_mode == 'feature_shuffle':
                    original_feature = original_features[combo_name]
                    shuffled_values = np.random.default_rng(int(seeds[i])).permutation(
                        original_feature.values.copy()
                    )
                    shuffled_feature = pd.Series(
                        shuffled_values,
                        index=original_feature.index,
                        name=original_feature.name,
                    )
                else:
                    assert shuffled_candles is not None
                    shuffled_feature = item.bias_node_extractor(shuffled_candles).reindex(target.index)

                signals, fit_failed = _fit_and_predict(item.binning_model, shuffled_feature, target)
                if fit_failed:
                    null_metrics_by_combo[combo_name][i] = 0.0
                    no_trade_counts[combo_name] += 1
                    continue

                metric, is_no_trade = _compute_metric_from_signals(signals, target, objective_func)
                null_metrics_by_combo[combo_name][i] = metric
                if is_no_trade:
                    no_trade_counts[combo_name] += 1
            except Exception:
                null_metrics_by_combo[combo_name][i] = 0.0
                no_trade_counts[combo_name] += 1

    return {
        combo_name: _build_pipeline_report(
            param_combo=combo_name,
            feature_type='continuous',
            permutation_mode=permutation_mode,
            original_metric=original_metrics[combo_name],
            null_metrics=null_metrics_by_combo[combo_name],
            alpha=alpha,
            nreps=nreps,
            no_trade_permutations=no_trade_counts[combo_name],
        )
        for combo_name in items_by_combo
    }


def _run_pipeline_permutation_rule_based_batch(
    *,
    candles_df: pd.DataFrame,
    items: list[_RuleBasedPermutationBatchItem],
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    metric_threshold: float = 0.0,
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
) -> dict[str, PipelinePermutationReport]:
    """Internal batch Stage-2 runner for rule-based features (candle shuffle only)."""
    _ = metric_threshold  # Reserved for parity with public API.

    if not items:
        return {}

    from utils.evaluation.permutation_test.candle_shuffle import _prepare_candle_shuffle

    seeds = _derive_permutation_seeds(nreps, random_seed)
    candles_prepared = _prepare_candles_for_shuffler(candles_df)
    prepared_shuffler = _prepare_candle_shuffle(candles_prepared)

    original_metrics: dict[str, float] = {}
    null_metrics_by_combo: dict[str, np.ndarray] = {}
    no_trade_counts: dict[str, int] = {}
    items_by_combo = {item.param_combo: item for item in items}

    for item in items:
        original_rule = item.rule_extractor(candles_df)
        original_metric, _ = _compute_metric_from_signals(original_rule, target, objective_func)
        original_metrics[item.param_combo] = original_metric
        null_metrics_by_combo[item.param_combo] = np.empty(nreps, dtype=float)
        no_trade_counts[item.param_combo] = 0

    for i in range(nreps):
        shuffled_candles = prepared_shuffler.permute_with_seed(int(seeds[i]))
        for combo_name, item in items_by_combo.items():
            try:
                shuffled_rule = item.rule_extractor(shuffled_candles).reindex(target.index)
                metric, is_no_trade = _compute_metric_from_signals(
                    shuffled_rule, target, objective_func
                )
                null_metrics_by_combo[combo_name][i] = metric
                if is_no_trade:
                    no_trade_counts[combo_name] += 1
            except Exception:
                null_metrics_by_combo[combo_name][i] = 0.0
                no_trade_counts[combo_name] += 1

    return {
        combo_name: _build_pipeline_report(
            param_combo=combo_name,
            feature_type='rule_based',
            permutation_mode='candle_shuffle',
            original_metric=original_metrics[combo_name],
            null_metrics=null_metrics_by_combo[combo_name],
            alpha=alpha,
            nreps=nreps,
            no_trade_permutations=no_trade_counts[combo_name],
        )
        for combo_name in items_by_combo
    }


# ---------------------------------------------------------------------------
# T014 — Pipeline Permutation Test (continuous features)
# ---------------------------------------------------------------------------

def run_pipeline_permutation_continuous(
    candles_df: pd.DataFrame,
    bias_node_extractor: Callable[[pd.DataFrame], pd.Series],
    binning_model: BinningModelBase,
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    permutation_mode: Literal['feature_shuffle', 'candle_shuffle'] = 'candle_shuffle',
    metric_threshold: float = 0.0,
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
    param_combo: str = 'default',
) -> PipelinePermutationReport:
    """Stage 2a/2b: Full-pipeline permutation test for continuous features.

    Two permutation modes:

    * ``feature_shuffle`` — Shuffles raw continuous feature values (fast).
      Null: "No relationship between feature values and target."
    * ``candle_shuffle`` — Re-extracts feature from shuffled candles (slow).
      Null: "No temporal structure in price that the feature exploits."

    A fresh deep-copy of ``binning_model`` is fitted for every permutation
    to prevent state leakage.

    Args:
        candles_df: OHLCV DataFrame.  Must have open/high/low/close columns
            and either a DatetimeIndex or a ``datetime`` column.
        bias_node_extractor: ``(candles_df) -> named pd.Series`` of raw
            continuous feature values.
        binning_model: Template BinningModelBase.  Never mutated.
        target: Forward-return series for the full period.
        objective_func: ``(returns: pd.Series) -> float``.
        permutation_mode: ``'feature_shuffle'`` or ``'candle_shuffle'``.
        metric_threshold: Reserved (not enforced internally).
        nreps: Number of permutation replications.
        alpha: Significance level.
        random_seed: Master seed; per-permutation seeds are derived from it.
        param_combo: Label for this parameter combination.

    Returns:
        PipelinePermutationReport with feature_type='continuous'.

    References:
        docs/library/Feature_selection/Phase_1_IS/permutation_testing.md (Stage 2)
        docs/library/Feature_selection/Phase_1_IS/candle_permutation.md (candle mode invariants)
    """
    reports = _run_pipeline_permutation_continuous_batch(
        candles_df=candles_df,
        items=[
            _ContinuousPermutationBatchItem(
                param_combo=param_combo,
                bias_node_extractor=bias_node_extractor,
                binning_model=binning_model,
            )
        ],
        target=target,
        objective_func=objective_func,
        permutation_mode=permutation_mode,
        metric_threshold=metric_threshold,
        nreps=nreps,
        alpha=alpha,
        random_seed=random_seed,
    )
    return reports[param_combo]


# ---------------------------------------------------------------------------
# T014 — Pipeline Permutation Test (rule-based features)
# ---------------------------------------------------------------------------

def run_pipeline_permutation_rule_based(
    candles_df: pd.DataFrame,
    rule_extractor: Callable[[pd.DataFrame], pd.Series],
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    metric_threshold: float = 0.0,
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
    param_combo: str = 'default',
) -> PipelinePermutationReport:
    """Stage 2 (rule-based): Full-pipeline permutation test via candle shuffle.

    Rule-based outputs (−1, 0, +1) are serially correlated, so directly
    shuffling them creates unrealistic rapid level-changes.  Instead candles
    are shuffled and the rule is recomputed from the shuffled price series.

    Args:
        candles_df: OHLCV DataFrame with DatetimeIndex or ``datetime`` column.
        rule_extractor: ``(candles_df) -> pd.Series`` of discrete signals
            (−1, 0, +1).
        target: Forward-return series for the full period.
        objective_func: ``(returns: pd.Series) -> float``.
        metric_threshold: Reserved (not enforced internally).
        nreps: Number of permutation replications.
        alpha: Significance level.
        random_seed: Master seed.
        param_combo: Label for this parameter combination.

    Returns:
        PipelinePermutationReport with feature_type='rule_based' and
        permutation_mode='candle_shuffle'.

    References:
        docs/library/Feature_selection/Phase_1_IS/permutation_testing.md (Stage 2)
        docs/library/Feature_selection/Phase_1_IS/candle_permutation.md
    """
    reports = _run_pipeline_permutation_rule_based_batch(
        candles_df=candles_df,
        items=[_RuleBasedPermutationBatchItem(param_combo=param_combo, rule_extractor=rule_extractor)],
        target=target,
        objective_func=objective_func,
        metric_threshold=metric_threshold,
        nreps=nreps,
        alpha=alpha,
        random_seed=random_seed,
    )
    return reports[param_combo]


# ---------------------------------------------------------------------------
# T014/T016 — Out-of-sample permutation runner with vector-first gate
# ---------------------------------------------------------------------------

def run_oos_permutation_for_param(
    param_combo: str,
    feature_type: Literal['continuous', 'rule_based'],
    fitted_feature: pd.Series,
    candles_df: pd.DataFrame,
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    *,
    bias_node_extractor: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    binning_model: Optional[BinningModelBase] = None,
    rule_extractor: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    permutation_mode: Literal['feature_shuffle', 'candle_shuffle'] = 'candle_shuffle',
    metric_threshold: float = 0.0,
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
) -> OutOfSamplePermutationReport:
    """Run out-of-sample permutation for one parameter combo.

    The vector shuffle gate is always run first. If it fails, stage-2 candle
    permutation is skipped and ``candle_report`` is returned as ``None``.
    """
    if feature_type not in ('continuous', 'rule_based'):
        raise ValueError(
            f"Unknown feature_type: {feature_type!r}. "
            "Use 'continuous' or 'rule_based'."
        )

    if feature_type == 'rule_based' and permutation_mode != 'candle_shuffle':
        raise ValueError(
            "rule_based feature_type only supports permutation_mode='candle_shuffle'."
        )

    vector_report = run_vector_shuffle_test(
        fitted_feature=fitted_feature,
        target=target,
        objective_func=objective_func,
        nreps=nreps,
        alpha=alpha,
        random_seed=random_seed,
        param_combo=param_combo,
    )

    if not vector_report.passed:
        return OutOfSamplePermutationReport(
            param_combo=param_combo,
            vector_report=vector_report,
            candle_report=None,
            passed=False,
        )

    candle_report: PipelinePermutationReport
    if feature_type == 'continuous':
        if bias_node_extractor is None or binning_model is None:
            raise ValueError(
                'continuous feature_type requires bias_node_extractor and binning_model.',
            )
        candle_report = run_pipeline_permutation_continuous(
            candles_df=candles_df,
            bias_node_extractor=bias_node_extractor,
            binning_model=binning_model,
            target=target,
            objective_func=objective_func,
            permutation_mode=permutation_mode,
            metric_threshold=metric_threshold,
            nreps=nreps,
            alpha=alpha,
            random_seed=random_seed,
            param_combo=param_combo,
        )
    elif feature_type == 'rule_based':
        if rule_extractor is None:
            raise ValueError('rule_based feature_type requires rule_extractor.')
        candle_report = run_pipeline_permutation_rule_based(
            candles_df=candles_df,
            rule_extractor=rule_extractor,
            target=target,
            objective_func=objective_func,
            metric_threshold=metric_threshold,
            nreps=nreps,
            alpha=alpha,
            random_seed=random_seed,
            param_combo=param_combo,
        )
    else:
        raise ValueError(f'Unknown feature_type: {feature_type!r}')

    return OutOfSamplePermutationReport(
        param_combo=param_combo,
        vector_report=vector_report,
        candle_report=candle_report,
        passed=bool(vector_report.passed and candle_report.passed),
    )
