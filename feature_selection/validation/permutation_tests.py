"""
T013 + T014: Signed-signal permutation tests for the Feature Validator pipeline.

The validation stack now assumes one frozen contract: extracted features are
already signed signals. No fitted-binning cloning or feature-type dispatch
remains in this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from feature_selection.validation.reports import (
    OutOfSamplePermutationReport,
    PipelinePermutationReport,
    VectorShuffleReport,
)

logger = logging.getLogger(__name__)

PermutationMode = Literal["feature_shuffle", "candle_shuffle"]


@dataclass(frozen=True)
class _SignedSignalPermutationBatchItem:
    """Internal Stage-2 batch item for one parameter combination."""

    param_combo: str
    signal_extractor: Callable[[pd.DataFrame], pd.Series]


def _align_signal_to_target(signal: pd.Series, target: pd.Series) -> tuple[pd.Series, pd.Series]:
    aligned_signal = signal.reindex(target.index).dropna()
    aligned_target = target.reindex(aligned_signal.index).dropna()
    aligned_signal = aligned_signal.reindex(aligned_target.index)
    return aligned_signal, aligned_target


def run_vector_shuffle_test(
    fitted_feature: pd.Series,
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    nreps: int = 1000,
    alpha: float = 0.10,
    random_seed: Optional[int] = None,
    param_combo: str = "default",
) -> VectorShuffleReport:
    """Stage 1: shuffle a frozen signed-signal vector."""
    rng = np.random.default_rng(random_seed)
    aligned_target = target.reindex(fitted_feature.index)

    active_mask = fitted_feature != 0.0
    original_returns = aligned_target * fitted_feature
    original_metric = objective_func(original_returns[active_mask]) if active_mask.any() else 0.0

    feature_values = fitted_feature.values.copy()
    null_metrics = np.empty(nreps, dtype=float)
    for i in range(nreps):
        shuffled_series = pd.Series(
            rng.permutation(feature_values),
            index=fitted_feature.index,
        )
        shuffled_active = shuffled_series != 0.0
        shuffled_returns = aligned_target * shuffled_series
        null_metrics[i] = (
            objective_func(shuffled_returns[shuffled_active]) if shuffled_active.any() else 0.0
        )

    p_value = float(1 + int((null_metrics >= original_metric).sum())) / float(nreps + 1)
    critical_value = float(np.percentile(null_metrics, (1.0 - alpha) * 100.0))
    return VectorShuffleReport(
        param_combo=param_combo,
        original_metric=original_metric,
        null_distribution=null_metrics,
        critical_value=critical_value,
        p_value=p_value,
        passed=bool(original_metric > critical_value),
        alpha=alpha,
        nreps=nreps,
    )


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

        from utils.core.enums import Ticker, TimeFrame
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
                original_feature = original_features[combo_name]
                shuffled_feature = pd.Series(
                    np.random.default_rng(seed).permutation(original_feature.values.copy()),
                    index=original_feature.index,
                    name=original_feature.name,
                )
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
    p_value = float(1 + int((null_metrics >= original_metric).sum())) / float(nreps + 1)
    critical_value = float(np.percentile(null_metrics, (1.0 - alpha) * 100.0))
    return PipelinePermutationReport(
        param_combo=param_combo,
        feature_type="signed_signal",  # collapsed contract
        permutation_mode=permutation_mode,
        original_metric=original_metric,
        null_distribution=null_metrics,
        critical_value=critical_value,
        p_value=p_value,
        passed=bool(original_metric > critical_value),
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
        aligned_feature, aligned_target = _align_signal_to_target(original_feature, target)
        original_features[item.param_combo] = aligned_feature
        original_metrics[item.param_combo], _ = _compute_metric_from_signals(
            aligned_feature, aligned_target, objective_func
        )
        null_metrics_by_combo[item.param_combo] = np.empty(nreps, dtype=float)
        no_trade_counts[item.param_combo] = 0

    prepared_shufflers: object = None
    if permutation_mode == "candle_shuffle":
        from utils.evaluation.permutation_test.candle_shuffle import _prepare_candle_shuffle

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
                        original_feature = original_features[combo_name]
                        shuffled_feature = pd.Series(
                            np.random.default_rng(int(seeds[i])).permutation(original_feature.values.copy()),
                            index=original_feature.index,
                            name=original_feature.name,
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
    """Run the vector-first OOS gate for one parameter combo."""
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

    candle_report = run_pipeline_permutation(
        candles_df=candles_df,
        signal_extractor=signal_extractor,
        target=target,
        objective_func=objective_func,
        permutation_mode=permutation_mode,
        metric_threshold=metric_threshold,
        nreps=nreps,
        alpha=alpha,
        random_seed=random_seed,
        param_combo=param_combo,
    )
    return OutOfSamplePermutationReport(
        param_combo=param_combo,
        vector_report=vector_report,
        candle_report=candle_report,
        passed=bool(vector_report.passed and candle_report.passed),
    )
