"""
T013 + T014: Signed-signal permutation tests for the Feature Validator pipeline.

The validation stack now assumes one frozen contract: extracted features are
already signed signals. No fitted-binning cloning or feature-type dispatch
remains in this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal, Optional, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from features.validation.objective_metrics import (
    ObjectiveMetricSpec,
    apply_objective_metric,
)
from features.validation.reports import (
    OutOfSamplePermutationReport,
    PipelinePermutationReport,
    VectorShuffleReport,
)
from lib.core.signal_alignment import align_signal_to_target

logger = logging.getLogger(__name__)

PermutationMode = Literal["feature_shuffle", "candle_shuffle"]


def _permutation_tail_stats(
    original_metric: float,
    null_metrics: np.ndarray,
    nreps: int,
    alpha: float,
) -> tuple[float, float, bool]:
    """One-sided Monte Carlo p-value with +1 pseudo-count (Phipson & Smyth).

    ``p_value = (1 + #{null_i >= observed}) / (nreps + 1)``. The extra ``+1`` in
    numerator and denominator avoids ``p=0`` and treats the observed statistic as
    one additional null draw. With ``nreps=100``, only values ``k/101`` for
    ``k in {1, …, 101}`` are possible — not multiples of ``1/100``.

    ``passed`` uses the empirical ``(1 - alpha)`` quantile of the null distribution
  (``critical_value``), not ``p_value <= alpha``.
    """
    null_ge_count = int((null_metrics >= original_metric).sum())
    p_value = float(1 + null_ge_count) / float(nreps + 1)
    critical_value = float(np.percentile(null_metrics, (1.0 - alpha) * 100.0))
    return p_value, critical_value, bool(original_metric > critical_value)


def _shuffle_feature_values(original_feature: pd.Series, seed: int) -> pd.Series:
    return pd.Series(
        np.random.default_rng(seed).permutation(original_feature.values.copy()),
        index=original_feature.index,
        name=original_feature.name,
    )


@dataclass(frozen=True)
class _SignedSignalPermutationBatchItem:
    """Internal Stage-2 batch item for one parameter combination."""

    param_combo: str
    signal_extractor: Callable[[pd.DataFrame], pd.Series]

def _compute_builtin_metric_matrix(
    spec: ObjectiveMetricSpec,
    returns_matrix: np.ndarray,
) -> np.ndarray:
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2D")
    n_rows, n_obs = returns_matrix.shape
    if n_rows == 0:
        return np.zeros(0, dtype=np.float64)
    if n_obs == 0:
        return np.zeros(n_rows, dtype=np.float64)
    return np.array(
        [
            apply_objective_metric(spec, pd.Series(row, dtype=np.float64))
            for row in returns_matrix
        ],
        dtype=np.float64,
    )


def _compute_metric_vectorized_for_feature(
    spec: ObjectiveMetricSpec,
    feature_values: np.ndarray,
    target_values: np.ndarray,
    permuted_targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    valid_mask = (feature_values != 0.0) & np.isfinite(feature_values) & np.isfinite(target_values)
    if not valid_mask.any():
        return (0.0, np.zeros(permuted_targets.shape[0], dtype=np.float64))

    active_feature = feature_values[valid_mask]
    active_target = target_values[valid_mask]
    active_permuted_targets = permuted_targets[:, valid_mask]

    observed = _compute_builtin_metric_matrix(
        spec,
        (active_target * active_feature)[np.newaxis, :],
    )[0]
    nulls = _compute_builtin_metric_matrix(
        spec,
        active_permuted_targets * active_feature[np.newaxis, :],
    )
    return (float(observed), nulls)


def run_vector_shuffle_test(
    fitted_feature: pd.Series,
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
    param_combo: str = "default",
) -> VectorShuffleReport:
    """Stage 1: shuffle a frozen signed-signal vector (sequential NumPy loop)."""
    aligned_target = cast(pd.Series, target.reindex(fitted_feature.index))
    active_mask = fitted_feature != 0.0
    original_returns = aligned_target * fitted_feature
    original_metric = (
        float(objective_func(original_returns[active_mask])) if active_mask.any() else 0.0
    )

    fv = np.asarray(fitted_feature.to_numpy(dtype=float, copy=True), dtype=np.float64)
    tv = np.asarray(aligned_target.to_numpy(dtype=float, copy=False), dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    null_metrics = np.empty(nreps, dtype=float)
    for i in range(nreps):
        permuted = fv[rng.permutation(len(fv))]
        act = permuted != 0.0
        if not act.any():
            null_metrics[i] = 0.0
        else:
            ret = tv * permuted
            null_metrics[i] = float(objective_func(pd.Series(ret[act], dtype=np.float64)))

    p_value, critical_value, passed = _permutation_tail_stats(
        original_metric, null_metrics, nreps, alpha
    )
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


def run_vector_shuffle_target_perm_batch(
    *,
    ordered_combo_names: list[str],
    features_by_combo: dict[str, pd.Series],
    target: pd.Series,
    metric_spec: ObjectiveMetricSpec,
    nreps: int,
    alpha: float,
    random_seed: Optional[int],
) -> dict[str, VectorShuffleReport]:
    """Many param combos: one shared target ``t``, feature matrix ``F``; each null rep permutes ``t`` once.

    For rep ``r``: ``t_perm = t[perm_r]``; combo ``c`` uses returns ``t_perm * F[:, c]`` on active bars.
    All sequential NumPy / small pandas slices — no threading.
    """
    if not ordered_combo_names:
        return {}
    feature_columns: list[np.ndarray] = []
    index_ref: pd.Index | None = None
    for name in ordered_combo_names:
        if name not in features_by_combo:
            raise KeyError(f"Missing feature series for combo {name!r}")
        feat = features_by_combo[name]
        if index_ref is None:
            index_ref = feat.index
        elif not feat.index.equals(index_ref):
            raise ValueError(f"Feature index mismatch for combo {name!r}")
        feature_columns.append(np.asarray(feat.to_numpy(dtype=float, copy=False), dtype=np.float64))

    assert index_ref is not None
    aligned_target = target.reindex(index_ref)
    t = np.asarray(aligned_target.to_numpy(dtype=float, copy=False), dtype=np.float64)
    F = np.column_stack(feature_columns)
    n, k = F.shape
    rng = np.random.default_rng(random_seed)
    perm = (
        np.vstack([rng.permutation(n) for _ in range(nreps)])
        if n > 1
        else np.zeros((nreps, n), dtype=np.int64)
    )
    permuted_targets = t[perm] if n > 0 else np.zeros((nreps, 0), dtype=np.float64)
    observed = np.zeros(k, dtype=np.float64)
    null_m = np.zeros((k, nreps), dtype=np.float64)
    for c in range(k):
        observed[c], null_m[c] = _compute_metric_vectorized_for_feature(
            metric_spec,
            F[:, c],
            t,
            permuted_targets,
        )

    reports: dict[str, VectorShuffleReport] = {}
    for c, name in enumerate(ordered_combo_names):
        orig = float(observed[c])
        nulls = null_m[c]
        p_value, critical_value, passed = _permutation_tail_stats(orig, nulls, nreps, alpha)
        reports[name] = VectorShuffleReport(
            param_combo=name,
            original_metric=orig,
            null_distribution=nulls,
            critical_value=critical_value,
            p_value=p_value,
            passed=passed,
            alpha=alpha,
            nreps=nreps,
        )
    return reports


def _compute_metric_from_signals(
    signals: pd.Series,
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
) -> tuple[float, bool]:
    active_mask = signals != 0.0
    if not active_mask.any():
        return 0.0, True
    aligned_target = target.reindex(signals.index)
    signal_returns = (aligned_target * signals)[active_mask]
    return float(objective_func(signal_returns)), False


def _prepare_candles_for_shuffler(candles_df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in candles_df.columns and hasattr(candles_df.index, "dtype") and str(
        candles_df.index.dtype
    ).startswith("datetime"):
        df = candles_df.copy()
        df["datetime"] = candles_df.index
        return df
    return candles_df


def _permute_candles_with_seed(
    candles_prepared: pd.DataFrame,
    seed: int,
    prepared_shufflers: object,
) -> pd.DataFrame:
    if isinstance(prepared_shufflers, dict):
        parts = [
            prepared_shufflers[ticker].permute_with_seed(seed)
            for ticker in sorted(prepared_shufflers)
        ]
        out = pd.concat(parts, axis=0, ignore_index=True)
        if "datetime" in out.columns and "ticker" in out.columns:
            out = out.sort_values(["datetime", "ticker"]).reset_index(drop=True)
        return out
    return prepared_shufflers.permute_with_seed(seed)


def _update_cross_ticker_store_from_shuffled_candles(
    shuffled_candles: pd.DataFrame,
    timeframes: list | None = None,
) -> None:
    try:
        if "ticker" not in shuffled_candles.columns:
            return

        from lib.core.enums import Ticker, TimeFrame
        from utils.data.cross_ticker_store import CrossTickerDataStore

        store = CrossTickerDataStore.get_instance()
        if timeframes is not None:
            tfs = timeframes
        elif "timeframe" in shuffled_candles.columns:
            tfs = [
                tf if isinstance(tf, TimeFrame) else TimeFrame[str(tf)]
                for tf in shuffled_candles["timeframe"].unique()
            ]
        else:
            tfs = [TimeFrame.D]

        for raw_ticker, ticker_df in shuffled_candles.groupby("ticker"):
            if isinstance(raw_ticker, Ticker):
                ticker_enum = raw_ticker
            else:
                raw_str = str(raw_ticker).strip().upper()
                ticker_enum = next(
                    (member for member in Ticker if member.name == raw_str or member.value == raw_str),
                    None,
                )
                if ticker_enum is None:
                    continue

            for tf in tfs:
                store.set_data(ticker_enum, tf, ticker_df.copy())
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("_update_cross_ticker_store_from_shuffled_candles failed: %s", exc)


def _derive_permutation_seeds(nreps: int, random_seed: Optional[int]) -> np.ndarray:
    return np.random.default_rng(random_seed).integers(0, 2**31, size=nreps)


def _run_one_rep_signed_signal(
    rep_index: int,
    seed: int,
    candles_prepared: pd.DataFrame,
    prepared_shufflers: object,
    items_by_combo: dict[str, _SignedSignalPermutationBatchItem],
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    permutation_mode: PermutationMode,
    original_features: dict[str, pd.Series],
) -> tuple[int, dict[str, tuple[float, bool]]]:
    shuffled_candles: pd.DataFrame | None = None
    if permutation_mode == "candle_shuffle":
        shuffled_candles = _permute_candles_with_seed(candles_prepared, seed, prepared_shufflers)
        _update_cross_ticker_store_from_shuffled_candles(shuffled_candles)

    results: dict[str, tuple[float, bool]] = {}
    for combo_name, item in items_by_combo.items():
        try:
            if permutation_mode == "feature_shuffle":
                shuffled_feature = _shuffle_feature_values(original_features[combo_name], seed)
            else:
                if shuffled_candles is None:
                    raise ValueError("shuffled_candles must exist for candle_shuffle.")
                shuffled_feature = item.signal_extractor(shuffled_candles).reindex(target.index)

            metric, no_trade = _compute_metric_from_signals(shuffled_feature, target, objective_func)
            results[combo_name] = (metric, no_trade)
        except Exception:
            results[combo_name] = (0.0, True)
    return rep_index, results


def _build_pipeline_report(
    *,
    param_combo: str,
    permutation_mode: PermutationMode,
    original_metric: float,
    null_metrics: np.ndarray,
    alpha: float,
    nreps: int,
    no_trade_permutations: int,
) -> PipelinePermutationReport:
    p_value, critical_value, passed = _permutation_tail_stats(
        original_metric, null_metrics, nreps, alpha
    )
    return PipelinePermutationReport(
        param_combo=param_combo,
        feature_type="signed_signal",  # collapsed contract
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


def _run_pipeline_permutation_batch(
    *,
    candles_df: pd.DataFrame,
    items: list[_SignedSignalPermutationBatchItem],
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    permutation_mode: PermutationMode = "candle_shuffle",
    metric_threshold: float = 0.0,
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
    n_jobs_reps: int = 1,
) -> dict[str, PipelinePermutationReport]:
    _ = metric_threshold
    if permutation_mode not in ("feature_shuffle", "candle_shuffle"):
        raise ValueError(
            f"Unknown permutation_mode: {permutation_mode!r}. Use 'feature_shuffle' or 'candle_shuffle'."
        )
    if not items:
        return {}

    seeds = _derive_permutation_seeds(nreps, random_seed)
    candles_prepared = _prepare_candles_for_shuffler(candles_df)
    items_by_combo = {item.param_combo: item for item in items}

    original_features: dict[str, pd.Series] = {}
    original_metrics: dict[str, float] = {}
    null_metrics_by_combo: dict[str, np.ndarray] = {}
    no_trade_counts: dict[str, int] = {}

    for item in items:
        original_feature = item.signal_extractor(candles_df)
        aligned_feature, aligned_target = align_signal_to_target(original_feature, target)
        original_features[item.param_combo] = aligned_feature
        original_metrics[item.param_combo], _ = _compute_metric_from_signals(
            aligned_feature, aligned_target, objective_func
        )
        null_metrics_by_combo[item.param_combo] = np.empty(nreps, dtype=float)
        no_trade_counts[item.param_combo] = 0

    prepared_shufflers: object = None
    if permutation_mode == "candle_shuffle":
        from research.evaluation.permutation_test.candle_shuffle import _prepare_candle_shuffle

        if "ticker" in candles_prepared.columns and candles_prepared["ticker"].nunique() > 1:
            prepared_shufflers = {
                ticker: _prepare_candle_shuffle(
                    candles_prepared[candles_prepared["ticker"] == ticker].copy()
                )
                for ticker in candles_prepared["ticker"].unique()
            }
        else:
            prepared_shufflers = _prepare_candle_shuffle(candles_prepared)

    if n_jobs_reps > 1:
        rep_results = Parallel(n_jobs=n_jobs_reps, backend="loky")(
            delayed(_run_one_rep_signed_signal)(
                i,
                int(seeds[i]),
                candles_prepared,
                prepared_shufflers,
                items_by_combo,
                target,
                objective_func,
                permutation_mode,
                original_features,
            )
            for i in range(nreps)
        )
        for rep_index, results in sorted(rep_results, key=lambda pair: pair[0]):
            for combo_name, (metric, no_trade) in results.items():
                null_metrics_by_combo[combo_name][rep_index] = metric
                no_trade_counts[combo_name] += int(no_trade)
    else:
        for i in tqdm(range(nreps), desc=f"Stage 2 ({permutation_mode})", unit="rep"):
            shuffled_candles: pd.DataFrame | None = None
            if permutation_mode == "candle_shuffle":
                if prepared_shufflers is None:
                    raise ValueError("prepared_shufflers must exist for candle_shuffle.")
                shuffled_candles = _permute_candles_with_seed(
                    candles_prepared,
                    int(seeds[i]),
                    prepared_shufflers,
                )
                _update_cross_ticker_store_from_shuffled_candles(shuffled_candles)

            for combo_name, item in items_by_combo.items():
                try:
                    if permutation_mode == "feature_shuffle":
                        shuffled_feature = _shuffle_feature_values(
                            original_features[combo_name], int(seeds[i])
                        )
                    else:
                        if shuffled_candles is None:
                            raise ValueError("shuffled_candles must exist for candle_shuffle.")
                        shuffled_feature = item.signal_extractor(shuffled_candles).reindex(target.index)

                    metric, no_trade = _compute_metric_from_signals(
                        shuffled_feature,
                        target,
                        objective_func,
                    )
                    null_metrics_by_combo[combo_name][i] = metric
                    no_trade_counts[combo_name] += int(no_trade)
                except Exception:
                    null_metrics_by_combo[combo_name][i] = 0.0
                    no_trade_counts[combo_name] += 1

    return {
        combo_name: _build_pipeline_report(
            param_combo=combo_name,
            permutation_mode=permutation_mode,
            original_metric=original_metrics[combo_name],
            null_metrics=null_metrics_by_combo[combo_name],
            alpha=alpha,
            nreps=nreps,
            no_trade_permutations=no_trade_counts[combo_name],
        )
        for combo_name in items_by_combo
    }


def run_pipeline_permutation(
    candles_df: pd.DataFrame,
    signal_extractor: Callable[[pd.DataFrame], pd.Series],
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    permutation_mode: PermutationMode = "candle_shuffle",
    metric_threshold: float = 0.0,
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
    param_combo: str = "default",
) -> PipelinePermutationReport:
    """Stage 2: run a single signed-signal permutation test."""
    reports = _run_pipeline_permutation_batch(
        candles_df=candles_df,
        items=[_SignedSignalPermutationBatchItem(param_combo=param_combo, signal_extractor=signal_extractor)],
        target=target,
        objective_func=objective_func,
        permutation_mode=permutation_mode,
        metric_threshold=metric_threshold,
        nreps=nreps,
        alpha=alpha,
        random_seed=random_seed,
    )
    return reports[param_combo]


def run_oos_permutation_for_param(
    param_combo: str,
    fitted_feature: pd.Series,
    candles_df: pd.DataFrame,
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    *,
    signal_extractor: Callable[[pd.DataFrame], pd.Series],
    permutation_mode: PermutationMode = "candle_shuffle",
    metric_threshold: float = 0.0,
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
) -> OutOfSamplePermutationReport:
    """Vector-shuffle OOS gate for one parameter combo (pipeline second stage removed).

    ``candles_df``, ``signal_extractor``, ``permutation_mode``, and ``metric_threshold`` are
    accepted for call-site compatibility but are not used.
    """
    _ = candles_df, signal_extractor, permutation_mode, metric_threshold
    vector_report = run_vector_shuffle_test(
        fitted_feature=fitted_feature,
        target=target,
        objective_func=objective_func,
        nreps=nreps,
        alpha=alpha,
        random_seed=random_seed,
        param_combo=param_combo,
    )
    return OutOfSamplePermutationReport(
        param_combo=param_combo,
        vector_report=vector_report,
        candle_report=None,
        passed=bool(vector_report.passed),
    )
