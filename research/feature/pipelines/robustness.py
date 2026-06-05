from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from quantfoundry_core.robustness import (
    CorrelationMethod,
    DSRResult,
    GridCallableMetricAdapter,
    NEffectiveResult,
    PermutationCombination,
    PermutationGrid,
    PermutationMetricSpec,
    PermutationTestConfig,
    PermutationTestResult,
    RollingISResult,
    SharpeCI,
    SweepEntry,
    SweepReturns,
    compute_n_effective,
    deflated_sharpe_ratio,
    rolling_is_performance,
    run_grid_permutation_test,
    run_individual_combination_permutation_test,
    sharpe_confidence_interval,
)

from research.feature.config import FeatureType, ResearchConfig
from research.feature.exploration.filter_gate_catalog import resolved_exploration_bias_spec
from research.feature._internal.core_helpers import combo_key, normalize_datetime_index
from research.feature.binning.transforms import flatten_params_for_combo_long_table
from research.feature.filter_research_labels import research_display_label
from research.feature.in_sample.data_loader import BIAS_MODULE_COMBO_KEY
from research.feature.in_sample.data_loader import (
    expand_bias_specs,
    first_bias_spec,
    permutation_combo_display_name,
    populate_cache_if_needed,
)
from research.feature.research_table_exports import objective_metric_display_label
from research.feature.visualization.cusum_stability import write_rolling_cusum_artifacts
from features.validation.objective_metrics import (
    ObjectiveMetricSpec,
    apply_objective_metric,
)
from features.validation.stability_analysis import _param_combo_name
from research.evaluation.walkforward.research_data import (
    build_reference_target,
    load_signed_signal_research_data,
)


@dataclass(frozen=True)
class NeweyWestResult:
    t_naive: float
    t_adjusted: float
    inflation_factor: float
    long_run_variance: float
    max_lag: int

    def to_json_dict(self) -> dict[str, float | int]:
        return {
            "t_naive": self.t_naive,
            "t_adjusted": self.t_adjusted,
            "inflation_factor": self.inflation_factor,
            "long_run_variance": self.long_run_variance,
            "max_lag": self.max_lag,
        }


@dataclass(frozen=True)
class RobustnessComboScore:
    param_combo: str
    param_combo_label: str
    selection_score: float
    raw_sharpe_annualized: float
    nw_adjusted_sharpe_annualized: float
    n_obs: int
    is_best: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "param_combo": self.param_combo,
            "param_combo_label": self.param_combo_label,
            "selection_score": self.selection_score,
            "raw_sharpe_annualized": self.raw_sharpe_annualized,
            "nw_adjusted_sharpe_annualized": self.nw_adjusted_sharpe_annualized,
            "n_obs": self.n_obs,
            "is_best": self.is_best,
        }


@dataclass(frozen=True)
class InSampleRobustnessReport:
    feature_name: str
    feature_type: str
    selection_metric: str
    periods_per_year: int
    n_combinations: int
    n_effective: NEffectiveResult
    nw: NeweyWestResult
    sharpe_ci: SharpeCI
    dsr: DSRResult
    rolling_is: RollingISResult
    stability_chart: RollingISResult
    full_grid_permutation: PermutationTestResult | None
    individual_permutation: PermutationTestResult | None
    best_combination: PermutationCombination
    best_param_combo: str
    best_param_combo_label: str
    best_selection_score: float
    raw_sharpe_annualized: float
    nw_adjusted_sharpe_annualized: float
    skewness: float
    excess_kurtosis: float
    n_obs: int
    combo_scores: tuple[RobustnessComboScore, ...]
    interpretation: str
    observation_datetimes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "selection_metric": self.selection_metric,
            "periods_per_year": self.periods_per_year,
            "n_combinations": self.n_combinations,
            "n_effective": self.n_effective.to_json_dict(),
            "newey_west": self.nw.to_json_dict(),
            "sharpe_ci": self.sharpe_ci.to_json_dict(),
            "dsr": self.dsr.to_json_dict(),
            "rolling_is": self.rolling_is.to_json_dict(),
            "stability_chart": self.stability_chart.to_json_dict(),
            "full_grid_permutation": (
                None
                if self.full_grid_permutation is None
                else self.full_grid_permutation.to_json_dict()
            ),
            "individual_permutation": (
                None
                if self.individual_permutation is None
                else self.individual_permutation.to_json_dict()
            ),
            "best_combination": self.best_combination.to_json_dict(),
            "best_param_combo": self.best_param_combo,
            "best_param_combo_label": self.best_param_combo_label,
            "best_selection_score": self.best_selection_score,
            "raw_sharpe_annualized": self.raw_sharpe_annualized,
            "nw_adjusted_sharpe_annualized": self.nw_adjusted_sharpe_annualized,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "n_obs": self.n_obs,
            "combo_scores": [combo.to_json_dict() for combo in self.combo_scores],
            "interpretation": self.interpretation,
            "observation_datetimes": list(self.observation_datetimes),
        }


@dataclass(frozen=True)
class _PreparedRobustnessCombo:
    param_combo: str
    param_combo_label: str
    params: Mapping[str, object]
    signal: pd.Series
    returns: pd.Series


def newey_west_tstat(returns: np.ndarray, max_lag: int | None = None) -> NeweyWestResult:
    array = np.asarray(returns, dtype=np.float64)
    clean = array[np.isfinite(array)]
    n_obs = int(clean.shape[0])
    if n_obs < 2:
        return NeweyWestResult(
            t_naive=0.0,
            t_adjusted=0.0,
            inflation_factor=1.0,
            long_run_variance=0.0,
            max_lag=0,
        )

    resolved_lag = (
        int(4 * (n_obs / 100.0) ** (2.0 / 9.0))
        if max_lag is None
        else max(0, int(max_lag))
    )
    resolved_lag = min(resolved_lag, n_obs - 1)
    mean_return = float(clean.mean())
    demeaned = clean - mean_return
    gamma_zero = float(np.mean(np.square(demeaned, dtype=np.float64)))
    if gamma_zero <= 0.0:
        return NeweyWestResult(
            t_naive=0.0,
            t_adjusted=0.0,
            inflation_factor=1.0,
            long_run_variance=0.0,
            max_lag=resolved_lag,
        )

    gamma_values = np.asarray(
        [
            float(np.mean(demeaned[lag:] * demeaned[:-lag]))
            for lag in range(1, resolved_lag + 1)
        ],
        dtype=np.float64,
    )
    weights = (
        np.asarray(
            [1.0 - lag / float(resolved_lag + 1) for lag in range(1, resolved_lag + 1)],
            dtype=np.float64,
        )
        if resolved_lag > 0
        else np.zeros(0, dtype=np.float64)
    )
    long_run_variance = float(
        gamma_zero + 2.0 * np.sum(weights * gamma_values, dtype=np.float64)
    )
    bounded_lrv = max(long_run_variance, 1e-12)
    t_naive = mean_return * math.sqrt(float(n_obs)) / math.sqrt(gamma_zero)
    t_adjusted = mean_return * math.sqrt(float(n_obs)) / math.sqrt(bounded_lrv)
    return NeweyWestResult(
        t_naive=float(t_naive),
        t_adjusted=float(t_adjusted),
        inflation_factor=float(bounded_lrv / gamma_zero),
        long_run_variance=float(bounded_lrv),
        max_lag=resolved_lag,
    )


def _require_robustness_enabled(config: ResearchConfig) -> None:
    if not config.robustness.enabled:
        raise ValueError("Robustness is disabled.")
    if config.feature_type == FeatureType.CONTINUOUS:
        raise ValueError(
            "In-sample robustness is currently supported for signed-signal bias nodes only."
        )


def _core_json_scalar(value: object) -> object:
    """Thin adapter for repo enums before Core validates JSON-safe mappings."""

    if isinstance(value, Enum):
        return value.value
    return value


def _core_json_mapping(params: Mapping[str, object]) -> dict[str, object]:
    """JSON-safe flat mapping for Quant Foundry Core (no nested dict values)."""
    flat = flatten_params_for_combo_long_table(params)
    if BIAS_MODULE_COMBO_KEY in params:
        flat[BIAS_MODULE_COMBO_KEY] = _core_json_scalar(params[BIAS_MODULE_COMBO_KEY])
    return {key: _core_json_scalar(value) for key, value in flat.items()}


def _permutation_lookup_key(params: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    """Stable key for matching Core ``PermutationGrid`` entries to prepared combo signals."""
    return combo_key(_core_json_mapping(params))


def _collapse_series_to_datetime_mean(series: pd.Series, *, name: str) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        raw_index = series.index.get_level_values(0)
    else:
        raw_index = series.index
    normalized_index = normalize_datetime_index(pd.Index(raw_index))
    numeric = pd.to_numeric(series, errors="coerce")
    collapsed = (
        pd.Series(numeric.to_numpy(dtype=float), index=normalized_index, name=name)
        .dropna()
        .groupby(level=0)
        .mean()
        .sort_index(kind="mergesort")
    )
    collapsed.name = name
    return collapsed


def _collapse_frame_column_to_datetime_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Expected column {column!r} in combo frame.")
    return _collapse_series_to_datetime_mean(frame[column], name=column)


def _selection_metric_name(metric_spec: ObjectiveMetricSpec) -> str:
    builtin = metric_spec.builtin
    if builtin == "t_stat":
        return "newey_west_t_stat"
    if builtin == "sharpe":
        annualization = metric_spec.kwargs.get("annualization_factor")
        suffix = "_annualized" if annualization not in (None, 1, 1.0) else ""
        return f"newey_west_sharpe{suffix}"
    return objective_metric_display_label(metric_spec)


def _safe_float(value: float) -> float:
    return 0.0 if not math.isfinite(value) else float(value)


def _plain_grid_score(returns: pd.Series, metric_spec: ObjectiveMetricSpec) -> float:
    """Score combo returns without HAC/NW adjustment (for return-shuffle permutation nulls)."""
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return _safe_float(apply_objective_metric(metric_spec, clean))


def _selection_score(returns: pd.Series, metric_spec: ObjectiveMetricSpec) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    builtin = metric_spec.builtin
    if builtin == "t_stat":
        return newey_west_tstat(clean.to_numpy(dtype=float)).t_adjusted
    raw_score = _safe_float(apply_objective_metric(metric_spec, clean))
    if builtin != "sharpe":
        return raw_score
    nw = newey_west_tstat(clean.to_numpy(dtype=float))
    return raw_score / math.sqrt(max(nw.inflation_factor, 1e-12))


def _resolve_full_grid_permutation_metric(
    config: ResearchConfig,
    *,
    periods_per_year: int,
) -> ObjectiveMetricSpec:
    configured = config.robustness.full_grid_permutation_metric
    if configured is not None:
        return configured
    return ObjectiveMetricSpec(
        builtin="sharpe",
        kwargs={"annualization_factor": float(periods_per_year)},
    )


def _grid_permutation_metric_label(metric_spec: ObjectiveMetricSpec) -> str:
    annualization = metric_spec.kwargs.get("annualization_factor")
    base = objective_metric_display_label(metric_spec)
    if metric_spec.builtin == "sharpe" and annualization not in (None, 1, 1.0):
        return f"{base}_annualized"
    return base


def _annualized_sharpe(returns: pd.Series, periods_per_year: int) -> float:
    return _safe_float(
        apply_objective_metric(
            ObjectiveMetricSpec(
                builtin="sharpe",
                kwargs={"annualization_factor": float(periods_per_year)},
            ),
            returns,
        )
    )


def _per_period_sharpe(returns: pd.Series) -> float:
    return _safe_float(
        apply_objective_metric(
            ObjectiveMetricSpec(builtin="sharpe"),
            returns,
        )
    )


def _finite_stat_or_zero(value: float) -> float:
    return 0.0 if not math.isfinite(value) else float(value)


def _resolve_best_combo_frame(
    combo_signal_target: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    params: dict[str, Any],
) -> pd.DataFrame:
    key = combo_key(params)
    if key not in combo_signal_target:
        raise KeyError(f"Missing preloaded robustness data for params={params}.")
    return combo_signal_target[key]


def _prepare_combo(
    params: dict[str, Any],
    combo_signal_target: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    target: pd.Series,
) -> _PreparedRobustnessCombo:
    combo_name = _param_combo_name(params)
    frame = _resolve_best_combo_frame(combo_signal_target, params)
    collapsed_signal = (
        _collapse_frame_column_to_datetime_mean(frame, "signal")
        .reindex(target.index)
        .fillna(0.0)
        .rename("signal")
    )
    collapsed_returns = (
        _collapse_frame_column_to_datetime_mean(frame, "returns")
        if "returns" in frame.columns
        else _collapse_frame_column_to_datetime_mean(
            frame.assign(returns=frame["signal"] * frame["target"]),
            "returns",
        )
    )
    return _PreparedRobustnessCombo(
        param_combo=combo_name,
        param_combo_label=research_display_label(params),
        params=dict(params),
        signal=collapsed_signal,
        returns=collapsed_returns,
    )


def _build_prepared_combos(
    param_grid: Sequence[dict[str, Any]],
    combo_signal_target: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    target: pd.Series,
) -> tuple[_PreparedRobustnessCombo, ...]:
    return tuple(
        _prepare_combo(params, combo_signal_target, target)
        for params in param_grid
    )


def _build_core_permutation_grid(
    combos: Sequence[_PreparedRobustnessCombo],
) -> PermutationGrid:
    """Delegate permutation-grid construction/validation to Quant Foundry Core."""

    return PermutationGrid.from_params_list(
        [_core_json_mapping(combo.params) for combo in combos]
    )


def _build_metric_adapter(
    combos: Sequence[_PreparedRobustnessCombo],
    metric_spec: ObjectiveMetricSpec,
    *,
    nw_adjusted: bool,
) -> GridCallableMetricAdapter:
    signals_by_params = {
        _permutation_lookup_key(combo.params): combo.signal for combo in combos
    }
    metric_name = (
        _selection_metric_name(metric_spec)
        if nw_adjusted
        else _grid_permutation_metric_label(metric_spec)
    )
    grid_metric_spec = PermutationMetricSpec(
        metric_name=metric_name,
        metric_kwargs=_core_json_mapping(metric_spec.kwargs),
    )
    score_fn = _selection_score if nw_adjusted else _plain_grid_score

    def _score_grid(target: pd.Series, grid: PermutationGrid) -> tuple[float, ...]:
        return tuple(
            score_fn(
                signals_by_params[_permutation_lookup_key(combination.params)].mul(
                    target, fill_value=0.0
                ),
                metric_spec,
            )
            for combination in grid
        )

    return GridCallableMetricAdapter(spec=grid_metric_spec, scorer=_score_grid)


def _single_combo_n_effective(combo: _PreparedRobustnessCombo) -> NEffectiveResult:
    return NEffectiveResult(
        n_combinations=1,
        n_effective=1.0,
        mean_pairwise_corr=1.0,
        n_aligned_obs=int(combo.returns.dropna().shape[0]),
        n_pairs_excluded=0,
        method=CorrelationMethod.PEARSON,
        surface_label="single_combination",
        surface_interpretation=(
            "Single combination only - no search space to deflate, so N_eff = N = 1."
        ),
    )


def _compute_n_eff(
    combos: Sequence[_PreparedRobustnessCombo],
    *,
    min_overlap: int,
    correlation_method: str,
) -> NEffectiveResult:
    if len(combos) == 1:
        return _single_combo_n_effective(combos[0])
    try:
        method = CorrelationMethod(correlation_method)
    except ValueError as exc:
        raise ValueError(
            "robustness.correlation_method must be 'pearson' or 'spearman'."
        ) from exc
    sweep = SweepReturns(
        tuple(
            SweepEntry(
                params=_core_json_mapping(combo.params),
                returns=combo.returns,
            )
            for combo in combos
        )
    )
    return compute_n_effective(sweep, min_overlap=min_overlap, method=method)


def _resolve_rolling_window(requested_window: int, n_obs: int) -> int:
    bounded = max(2, int(requested_window))
    return min(bounded, max(2, n_obs - 1))


def _annualize_ci_interval(ci: SharpeCI, periods_per_year: int) -> tuple[float, float]:
    scale = math.sqrt(float(periods_per_year))
    return (ci.lower * scale, ci.upper * scale)


def _build_interpretation(
    report: InSampleRobustnessReport,
) -> str:
    dsr_probability = report.dsr.probability
    if dsr_probability >= 0.95:
        dsr_text = "strong evidence of real edge"
    elif dsr_probability >= 0.75:
        dsr_text = "moderate evidence of real edge"
    elif dsr_probability >= 0.50:
        dsr_text = "marginal evidence; search may still explain the result"
    else:
        dsr_text = "search is more likely than edge"
    rolling_text = "passed" if report.rolling_is.passes else "flagged instability"
    permutation_text = (
        ""
        if report.full_grid_permutation is None
        else f" Full-grid permutation p={report.full_grid_permutation.p_value:.3f}."
    )
    return (
        f"Best combo {report.best_param_combo_label} selected on {report.selection_metric}. "
        f"DSR={report.dsr.probability:.2f} ({dsr_text}), "
        f"N_eff={report.n_effective.n_effective:.1f}/{report.n_effective.n_combinations}, "
        f"and rolling IS {rolling_text}.{permutation_text}"
    )


def run_robustness_pipeline(
    config: ResearchConfig,
    output_dir: Path,
) -> tuple[InSampleRobustnessReport, list[dict[str, Any]]]:
    _require_robustness_enabled(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    exploration_spec = resolved_exploration_bias_spec(config)
    populate_cache_if_needed(config, bias_spec=exploration_spec)

    train_start, train_end = config.training_window_bounds
    load_config = replace(config, start=train_start, end=train_end)
    expanded = expand_bias_specs(exploration_spec)
    if not expanded:
        raise ValueError("No parameter combinations available for robustness suite.")

    data = load_signed_signal_research_data(load_config, expanded, print_loaded=False)
    if not data.successful_param_grid or data.reference_index is None:
        raise ValueError("Unable to load robustness research data.")

    target = _collapse_series_to_datetime_mean(
        build_reference_target(data.reference_index, data.reference_target_series),
        name="robustness_target",
    )
    if target.empty:
        raise ValueError("Reference target for robustness suite is empty.")

    param_grid = data.successful_param_grid
    first_combo_frame = next(iter(data.combo_signal_target.values()))
    signal_column = "signal" if "signal" in first_combo_frame.columns else "feature"
    feature_name = str(first_combo_frame[signal_column].name or signal_column)
    periods_per_year = int(load_config.timeframe.bars_per_year)
    combos = _build_prepared_combos(param_grid, data.combo_signal_target, target)
    permutation_grid = _build_core_permutation_grid(combos)
    selection_metric = load_config.robustness.selection_metric
    grid_perm_metric = _resolve_full_grid_permutation_metric(
        load_config,
        periods_per_year=periods_per_year,
    )
    metric_adapter = _build_metric_adapter(
        combos,
        selection_metric,
        nw_adjusted=True,
    )
    grid_perm_adapter = _build_metric_adapter(
        combos,
        grid_perm_metric,
        nw_adjusted=False,
    )
    observed_scores = tuple(
        float(score)
        for score in metric_adapter.score_grid(target, permutation_grid)
    )
    best_index = int(np.argmax(np.asarray(observed_scores, dtype=np.float64)))
    best_combo = combos[best_index]
    best_returns = best_combo.returns.dropna().sort_index(kind="mergesort")
    if best_returns.shape[0] < 2:
        raise ValueError("Best robustness combination has fewer than 2 observations.")

    n_eff = _compute_n_eff(
        combos,
        min_overlap=load_config.robustness.n_eff_min_overlap,
        correlation_method=load_config.robustness.correlation_method,
    )
    raw_per_period_sharpe = _per_period_sharpe(best_returns)
    raw_sharpe_annualized = _annualized_sharpe(best_returns, periods_per_year)
    nw = newey_west_tstat(best_returns.to_numpy(dtype=float))
    nw_adjusted_sharpe_annualized = raw_sharpe_annualized / math.sqrt(
        max(nw.inflation_factor, 1e-12)
    )
    skewness = _finite_stat_or_zero(float(best_returns.skew()))
    excess_kurtosis = _finite_stat_or_zero(float(best_returns.kurt()))
    sharpe_ci = sharpe_confidence_interval(
        sr_observed=raw_per_period_sharpe,
        n_obs=int(best_returns.shape[0]),
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        confidence=load_config.robustness.sharpe_confidence,
    )
    dsr = deflated_sharpe_ratio(
        sr_observed=raw_per_period_sharpe,
        n_obs=int(best_returns.shape[0]),
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        n_eff=n_eff.n_effective,
        sr_benchmark=load_config.robustness.sr_benchmark,
    )
    rolling_window = _resolve_rolling_window(
        load_config.robustness.rolling_window,
        int(best_returns.shape[0]),
    )
    rolling_is = rolling_is_performance(
        best_returns.to_numpy(dtype=float),
        window=rolling_window,
        min_positive_fraction=load_config.robustness.rolling_min_positive_fraction,
        alpha=load_config.robustness.rolling_alpha,
        periods_per_year=periods_per_year,
    )
    stability_window = _resolve_rolling_window(
        load_config.robustness.stability_rolling_window_days,
        int(best_returns.shape[0]),
    )
    stability_chart = rolling_is_performance(
        best_returns.to_numpy(dtype=float),
        window=stability_window,
        min_positive_fraction=load_config.robustness.rolling_min_positive_fraction,
        alpha=load_config.robustness.rolling_alpha,
        periods_per_year=periods_per_year,
    )

    permutation_seed = load_config.robustness.random_seed
    full_grid_permutation = (
        run_grid_permutation_test(
            target=target,
            grid=permutation_grid,
            metric=grid_perm_adapter,
            config=PermutationTestConfig.full_grid(
                n_permutations=load_config.robustness.n_permutations,
                seed=permutation_seed,
            ),
        )
        if load_config.robustness.run_full_grid_permutation
        else None
    )
    individual_permutation = (
        run_individual_combination_permutation_test(
            target=target,
            combination=permutation_grid[best_index],
            metric=grid_perm_adapter,
            config=PermutationTestConfig.individual_combination(
                n_permutations=load_config.robustness.n_permutations,
                seed=permutation_seed,
            ),
        )
        if load_config.robustness.run_individual_permutation
        else None
    )

    combo_scores = tuple(
        RobustnessComboScore(
            param_combo=combo.param_combo,
            param_combo_label=combo.param_combo_label,
            selection_score=observed_scores[idx],
            raw_sharpe_annualized=_annualized_sharpe(combo.returns, periods_per_year),
            nw_adjusted_sharpe_annualized=_annualized_sharpe(
                combo.returns,
                periods_per_year,
            )
            / math.sqrt(
                max(
                    newey_west_tstat(combo.returns.to_numpy(dtype=float)).inflation_factor,
                    1e-12,
                )
            ),
            n_obs=int(combo.returns.dropna().shape[0]),
            is_best=idx == best_index,
        )
        for idx, combo in enumerate(combos)
    )

    report = InSampleRobustnessReport(
        feature_name=feature_name,
        feature_type=load_config.feature_type.value,
        selection_metric=_selection_metric_name(load_config.robustness.selection_metric),
        periods_per_year=periods_per_year,
        n_combinations=len(combos),
        n_effective=n_eff,
        nw=nw,
        sharpe_ci=sharpe_ci,
        dsr=dsr,
        rolling_is=rolling_is,
        stability_chart=stability_chart,
        full_grid_permutation=full_grid_permutation,
        individual_permutation=individual_permutation,
        best_combination=PermutationCombination(
            params=_core_json_mapping(best_combo.params),
            label=best_combo.param_combo,
        ),
        best_param_combo=best_combo.param_combo,
        best_param_combo_label=best_combo.param_combo_label,
        best_selection_score=observed_scores[best_index],
        raw_sharpe_annualized=raw_sharpe_annualized,
        nw_adjusted_sharpe_annualized=nw_adjusted_sharpe_annualized,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        n_obs=int(best_returns.shape[0]),
        combo_scores=combo_scores,
        interpretation="",
        observation_datetimes=tuple(
            pd.Timestamp(ts).isoformat() for ts in best_returns.index
        ),
    )
    finalized_report = replace(report, interpretation=_build_interpretation(report))
    return finalized_report, list(param_grid)


def _summary_row(report: InSampleRobustnessReport) -> dict[str, object]:
    ci_lower, ci_upper = _annualize_ci_interval(report.sharpe_ci, report.periods_per_year)
    row: dict[str, object] = {
        "feature_name": report.feature_name,
        "feature_type": report.feature_type,
        "selection_metric": report.selection_metric,
        "n_combinations": report.n_combinations,
        "best_param_combo": report.best_param_combo,
        "best_param_combo_label": report.best_param_combo_label,
        "best_selection_score": report.best_selection_score,
        "raw_sharpe_annualized": report.raw_sharpe_annualized,
        "nw_adjusted_sharpe_annualized": report.nw_adjusted_sharpe_annualized,
        "nw_t_naive": report.nw.t_naive,
        "nw_t_adjusted": report.nw.t_adjusted,
        "nw_inflation_factor": report.nw.inflation_factor,
        "nw_max_lag": report.nw.max_lag,
        "sharpe_ci_lower_annualized": ci_lower,
        "sharpe_ci_upper_annualized": ci_upper,
        "dsr_probability": report.dsr.probability,
        "dsr_e_max_sr_under_null": report.dsr.e_max_sr_under_null,
        "n_effective": report.n_effective.n_effective,
        "mean_pairwise_corr": report.n_effective.mean_pairwise_corr,
        "surface_label": report.n_effective.surface_label,
        "rolling_positive_fraction": report.rolling_is.positive_fraction,
        "rolling_passes": report.rolling_is.passes,
        "cusum_statistic": report.rolling_is.cusum_statistic,
        "cusum_critical_value": report.rolling_is.cusum_critical_value,
        "cusum_break_detected": report.rolling_is.cusum_break_detected,
        "n_obs": report.n_obs,
        "interpretation": report.interpretation,
    }
    if report.full_grid_permutation is not None:
        row["full_grid_p_value"] = report.full_grid_permutation.p_value
    if report.individual_permutation is not None:
        row["individual_p_value"] = report.individual_permutation.p_value
    return row


def _combo_score_rows(
    report: InSampleRobustnessReport,
) -> list[dict[str, object]]:
    return [combo.to_json_dict() for combo in sorted(report.combo_scores, key=lambda item: item.selection_score, reverse=True)]


def _markdown_summary(report: InSampleRobustnessReport) -> str:
    ci_lower, ci_upper = _annualize_ci_interval(report.sharpe_ci, report.periods_per_year)
    permutation_lines = [
        (
            f"- Full-grid permutation p-value: {report.full_grid_permutation.p_value:.4f} "
            f"(metric=`{report.full_grid_permutation.metric.metric_name}`)"
        )
        if report.full_grid_permutation is not None
        else "- Full-grid permutation: not run",
        f"- Individual permutation p-value: {report.individual_permutation.p_value:.4f}"
        if report.individual_permutation is not None
        else "- Individual permutation: not run",
    ]
    combo_lines = [
        "| Param combo | Selection score | Raw SR (ann.) | NW SR (ann.) | Best |",
        "|------------|----------------:|--------------:|-------------:|:----:|",
        *[
            (
                f"| {combo.param_combo_label} | {combo.selection_score:.4f} | "
                f"{combo.raw_sharpe_annualized:.4f} | {combo.nw_adjusted_sharpe_annualized:.4f} | "
                f"{'Y' if combo.is_best else ''} |"
            )
            for combo in sorted(report.combo_scores, key=lambda item: item.selection_score, reverse=True)
        ],
    ]
    lines = [
        "# In-Sample Robustness Summary",
        "",
        f"**Feature:** {report.feature_name}",
        f"**Selection metric:** `{report.selection_metric}`",
        f"**Best combination:** {report.best_param_combo_label}",
        "",
        "## Summary",
        "",
        f"- Raw Sharpe (annualized): {report.raw_sharpe_annualized:.4f}",
        f"- NW-adjusted Sharpe (annualized): {report.nw_adjusted_sharpe_annualized:.4f}",
        f"- Sharpe CI (annualized): [{ci_lower:.4f}, {ci_upper:.4f}]",
        f"- Newey-West t-stat: {report.nw.t_adjusted:.4f} "
        f"(inflation_factor={report.nw.inflation_factor:.4f}, max_lag={report.nw.max_lag})",
        f"- DSR probability: {report.dsr.probability:.4f}",
        f"- N_eff: {report.n_effective.n_effective:.4f} / {report.n_effective.n_combinations}",
        f"- Mean pairwise correlation: {report.n_effective.mean_pairwise_corr:.4f}",
        f"- Rolling positive fraction: {report.rolling_is.positive_fraction:.4f}",
        f"- CUSUM: {report.rolling_is.cusum_statistic:.4f} / {report.rolling_is.cusum_critical_value:.4f}",
        *permutation_lines,
        "",
        "## Interpretation",
        "",
        report.interpretation,
        "",
        "## Combination Scores",
        "",
        *combo_lines,
        "",
    ]
    return "\n".join(lines)


def write_robustness_summary(
    report: InSampleRobustnessReport,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "robustness_summary.csv"
    combos_csv = output_dir / "robustness_combinations.csv"
    json_path = output_dir / "robustness_report.json"
    markdown_path = output_dir / "robustness_summary.md"

    pd.DataFrame([_summary_row(report)]).to_csv(summary_csv, index=False)
    pd.DataFrame(_combo_score_rows(report)).to_csv(combos_csv, index=False)
    json_path.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")

    cusum_artifacts = write_rolling_cusum_artifacts(
        report.stability_chart,
        output_dir,
        report.observation_datetimes,
        periods_per_year=report.periods_per_year,
    )

    artifacts = {
        "summary_csv": summary_csv,
        "combos_csv": combos_csv,
        "json": json_path,
        "markdown": markdown_path,
        **cusum_artifacts,
    }
    return artifacts
