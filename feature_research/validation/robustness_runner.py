"""Validation-zone robustness suite orchestration (Quant Foundry Core)."""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from quantfoundry_core.robustness import BootstrapCI, SharpeCI
from quantfoundry_core.robustness.validation import (
    CUSUMResult,
    EquityCurveBandsResult,
    RankCorrelationResult,
    RollingSharpezScoreResult,
    SharpeComparisonResult,
    ValidationRobustnessReport,
)

from feature_research._internal.core_helpers import combo_key
from feature_research.config import (
    RankCorrelationScope,
    ResearchConfig,
    resolve_validation_rank_scatter_n_jobs,
)
from feature_research.exploration.filter_gate_catalog import resolved_exploration_bias_spec
from feature_research.in_sample.data_loader import (
    BIAS_MODULE_COMBO_KEY,
    enrich_param_combo_with_module,
    expand_bias_specs,
    populate_cache_if_needed,
)
from feature_research.pipelines.robustness import (
    _build_prepared_combos,
    _collapse_series_to_datetime_mean,
    _selection_score,
)
from feature_research.shared.holdout_robustness_config import holdout_config_from_research
from utils.evaluation.holdout_robustness import run_holdout_robustness_pipeline
from feature_research.shared.visualization_paths import canonical_in_sample_visualization_dir
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from feature_selection.validation.stability_analysis import _param_combo_name
from utils.evaluation.walkforward.research_data import (
    FrozenSignalResearchData,
    build_reference_target,
    load_signed_signal_research_data,
)
from utils.evaluation.walkforward.selected_params_codec import decode_selected_params_list

ROLLING_Z_MEAN_WINDOW = 60
_DataclassT = TypeVar("_DataclassT")


def _dataclass_from_dict(cls: type[_DataclassT], payload: Mapping[str, object]) -> _DataclassT:
    field_names = {field.name for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
    filtered = {key: value for key, value in payload.items() if key in field_names}
    return cls(**filtered)  # type: ignore[call-arg, return-value]


def load_validation_robustness_report_from_json(path: Path) -> ValidationRobustnessReport:
    """Rehydrate a persisted validation robustness report."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return ValidationRobustnessReport(
        sharpe_comparison=_dataclass_from_dict(
            SharpeComparisonResult,
            {
                **payload["sharpe_comparison"],
                "ci_is": _dataclass_from_dict(SharpeCI, payload["sharpe_comparison"]["ci_is"]),
                "ci_val": _dataclass_from_dict(
                    BootstrapCI,
                    payload["sharpe_comparison"]["ci_val"],
                ),
            },
        ),
        cusum=_dataclass_from_dict(CUSUMResult, payload["cusum"]),
        equity_curve_bands=_dataclass_from_dict(
            EquityCurveBandsResult,
            payload["equity_curve_bands"],
        ),
        rolling_sharpe_zscore=_dataclass_from_dict(
            RollingSharpezScoreResult,
            payload["rolling_sharpe_zscore"],
        ),
        rank_correlation=_dataclass_from_dict(
            RankCorrelationResult,
            payload["rank_correlation"],
        ),
        all_passed=bool(payload["all_passed"]),
        interpretation=str(payload["interpretation"]),
        schema_version=str(payload.get("schema_version", "robustness-v1")),
    )


@dataclass(frozen=True)
class RankScatterRow:
    param_combo_label: str
    is_metric: float
    val_metric: float
    is_chosen: bool


def _require_validation_robustness_enabled(config: ResearchConfig) -> None:
    if not config.validation_robustness.enabled:
        raise ValueError("Validation robustness is disabled in config.")


def _require_research_window(config: ResearchConfig) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    window = config.research_window
    if window is None:
        raise ValueError("config.research_window is required for validation robustness.")
    return (
        pd.Timestamp(window.train_start),
        pd.Timestamp(window.train_end),
        pd.Timestamp(window.val_start),
        pd.Timestamp(window.val_end),
    )


def _slice_returns(
    returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    sliced = returns.loc[(returns.index >= start) & (returns.index <= end)].dropna()
    return sliced.sort_index(kind="mergesort")


def _metric_column_name(metric_spec: ObjectiveMetricSpec) -> str:
    builtin = metric_spec.builtin
    if builtin == "t_stat":
        return "t_stat"
    if builtin == "sharpe":
        return "sharpe"
    if builtin == "sortino":
        return "sortino"
    if builtin == "mean":
        return "mean"
    raise ValueError(f"Unsupported selection metric for rank correlation: {builtin!r}")


def _is_filter_gate_module(module_name: object) -> bool:
    return str(module_name or "") in {"filter_gate", "filter_gate_entry_only"}


def _signal_leg_params(params: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return nested ``signal_params`` for composite gates; else a plain signal dict."""
    module = params.get(BIAS_MODULE_COMBO_KEY, params.get("_bias_module"))
    if _is_filter_gate_module(module):
        nested = params.get("signal_params")
        return dict(nested) if isinstance(nested, dict) else None
    return {
        key: value
        for key, value in params.items()
        if key not in {BIAS_MODULE_COMBO_KEY, "_bias_module"}
    }


def _params_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if combo_key(left) == combo_key(right):
        return True
    left_signal = _signal_leg_params(left)
    right_signal = _signal_leg_params(right)
    if left_signal is not None and right_signal is not None:
        return combo_key(left_signal) == combo_key(right_signal)
    return False


def _rank_scatter_chosen_params(chosen_params: Mapping[str, Any]) -> dict[str, Any]:
    """Map eval composite params to the IS-grid signal leg for rank-scatter highlighting."""
    signal_params = _signal_leg_params(chosen_params)
    if signal_params is None:
        return dict(chosen_params)
    signal_module = chosen_params.get("signal_module")
    if isinstance(signal_module, str) and signal_module.strip():
        return enrich_param_combo_with_module(signal_params, signal_module)
    return enrich_param_combo_with_module(signal_params, None)


def _resolve_chosen_params(
    config: ResearchConfig,
    selection_summary_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Scalarize eval params and attach ``_bias_module`` for grid lookup."""

    eval_spec = config.eval_bias_spec
    base_params = dict(eval_spec["params"])
    module_name = eval_spec.get("module_name")

    if any(isinstance(value, list) for value in base_params.values()):
        if selection_summary_df is not None and not selection_summary_df.empty:
            payload = str(selection_summary_df.iloc[0].get("selected_params_json", ""))
            decoded = decode_selected_params_list(payload)
            if decoded:
                resolved = dict(decoded[0])
                if BIAS_MODULE_COMBO_KEY not in resolved:
                    return enrich_param_combo_with_module(resolved, module_name)
                return resolved
    return enrich_param_combo_with_module(base_params, module_name)


def _load_is_metrics_from_csv(
    config: ResearchConfig,
    metric_column: str,
) -> dict[str, float] | None:
    csv_path = canonical_in_sample_visualization_dir() / "param_sensitivity.csv"
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path)
    if "param_combo_label" not in frame.columns or metric_column not in frame.columns:
        return None
    return {
        str(row["param_combo_label"]): float(row[metric_column])
        for _, row in frame.iterrows()
        if pd.notna(row[metric_column])
    }


def _score_rank_scatter_combo(
    combo: Any,
    *,
    metric_spec: ObjectiveMetricSpec,
    is_metrics_by_label: dict[str, float] | None,
    chosen_params: Mapping[str, Any],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> RankScatterRow:
    train_returns = _slice_returns(combo.returns, train_start, train_end)
    val_returns = _slice_returns(combo.returns, val_start, val_end)
    val_metric = _selection_score(val_returns, metric_spec)
    if is_metrics_by_label is not None and combo.param_combo_label in is_metrics_by_label:
        is_metric = float(is_metrics_by_label[combo.param_combo_label])
    else:
        is_metric = _selection_score(train_returns, metric_spec)
    return RankScatterRow(
        param_combo_label=combo.param_combo_label,
        is_metric=is_metric,
        val_metric=val_metric,
        is_chosen=_params_match(combo.params, chosen_params),
    )


def _resolve_is_metrics_by_label(
    config: ResearchConfig,
    metric_column: str,
) -> dict[str, float] | None:
    scope = config.validation_robustness.rank_correlation_scope
    csv_metrics = _load_is_metrics_from_csv(config, metric_column)
    if scope is RankCorrelationScope.FROM_EXPLORATION_ARTIFACTS:
        if csv_metrics is None:
            csv_path = canonical_in_sample_visualization_dir() / "param_sensitivity.csv"
            raise ValueError(
                "rank_correlation_scope=from_exploration_artifacts requires "
                f"exploration param_sensitivity.csv at {csv_path}."
            )
        return csv_metrics
    return csv_metrics


def _build_rank_scatter_rows(
    config: ResearchConfig,
    combos: Sequence[Any],
    *,
    chosen_params: Mapping[str, Any],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> tuple[list[RankScatterRow], np.ndarray, np.ndarray]:
    metric_spec = config.robustness.selection_metric
    metric_column = _metric_column_name(metric_spec)
    is_metrics_by_label = _resolve_is_metrics_by_label(config, metric_column)
    n_jobs = resolve_validation_rank_scatter_n_jobs(config)

    if n_jobs <= 1 or len(combos) <= 1:
        rows = [
            _score_rank_scatter_combo(
                combo,
                metric_spec=metric_spec,
                is_metrics_by_label=is_metrics_by_label,
                chosen_params=chosen_params,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
            for combo in combos
        ]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=min(n_jobs, len(combos)), backend="threading")(
            delayed(_score_rank_scatter_combo)(
                combo,
                metric_spec=metric_spec,
                is_metrics_by_label=is_metrics_by_label,
                chosen_params=chosen_params,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
            for combo in combos
        )

    is_values = np.asarray([row.is_metric for row in rows], dtype=np.float64)
    val_values = np.asarray([row.val_metric for row in rows], dtype=np.float64)
    return rows, is_values, val_values


def _chosen_combo_from_eval_data(
    eval_research_data: FrozenSignalResearchData,
    chosen_params: Mapping[str, Any],
) -> Any | None:
    if not eval_research_data.successful_param_grid or eval_research_data.reference_index is None:
        return None
    target = _collapse_series_to_datetime_mean(
        build_reference_target(
            eval_research_data.reference_index,
            eval_research_data.reference_target_series,
        ),
        name="validation_robustness_eval_target",
    )
    eval_combos = _build_prepared_combos(
        eval_research_data.successful_param_grid,
        eval_research_data.combo_signal_target,
        target,
    )
    return next(
        (combo for combo in eval_combos if _params_match(combo.params, chosen_params)),
        eval_combos[0] if eval_combos else None,
    )


def _load_rank_correlation_grid(
    load_config: ResearchConfig,
) -> tuple[Any, ...]:
    exploration_spec = resolved_exploration_bias_spec(load_config)
    expanded = expand_bias_specs(exploration_spec)
    if not expanded:
        raise ValueError("No parameter combinations available for validation robustness.")
    populate_cache_if_needed(load_config, bias_spec=exploration_spec)
    data = load_signed_signal_research_data(load_config, expanded, print_loaded=False)
    if not data.successful_param_grid or data.reference_index is None:
        raise ValueError("Unable to load validation robustness research data.")
    target = _collapse_series_to_datetime_mean(
        build_reference_target(data.reference_index, data.reference_target_series),
        name="validation_robustness_target",
    )
    combos = _build_prepared_combos(data.successful_param_grid, data.combo_signal_target, target)
    return combos, data


def _resolve_chosen_combo_label(chosen_params: Mapping[str, Any]) -> str:
    return _param_combo_name(dict(chosen_params))


def run_validation_robustness_pipeline(
    config: ResearchConfig,
    *,
    selection_summary_df: pd.DataFrame | None = None,
    eval_research_data: FrozenSignalResearchData | None = None,
) -> tuple[ValidationRobustnessReport, list[RankScatterRow], str]:
    """Run the validation robustness suite for the locked eval combo."""

    _require_validation_robustness_enabled(config)
    train_start, train_end, val_start, val_end = _require_research_window(config)
    window = config.research_window
    assert window is not None

    data_start = min(config.start, window.train_start)
    data_end = max(config.end, window.val_end)
    load_config = replace(config, start=data_start, end=data_end)
    chosen_params = _resolve_chosen_params(config, selection_summary_df)
    rank_chosen_params = _rank_scatter_chosen_params(chosen_params)

    combos, _grid_data = _load_rank_correlation_grid(load_config)

    chosen_combo: Any | None = None
    if eval_research_data is not None:
        chosen_combo = _chosen_combo_from_eval_data(eval_research_data, chosen_params)
    if chosen_combo is None:
        chosen_combo = next(
            (combo for combo in combos if _params_match(combo.params, chosen_params)),
            None,
        )
    if chosen_combo is None and _is_filter_gate_module(chosen_params.get("_bias_module")):
        if eval_research_data is not None:
            chosen_combo = _chosen_combo_from_eval_data(eval_research_data, chosen_params)
        else:
            expanded_eval = expand_bias_specs(config.eval_bias_spec)
            if len(expanded_eval) != 1:
                raise ValueError(
                    "Eval filter-gate spec must expand to exactly one combo for validation robustness."
                )
            populate_cache_if_needed(load_config, bias_spec=config.eval_bias_spec)
            data_eval = load_signed_signal_research_data(
                load_config,
                expanded_eval,
                print_loaded=False,
            )
            chosen_combo = _chosen_combo_from_eval_data(data_eval, chosen_params)
        if chosen_combo is None:
            raise ValueError("Unable to resolve eval filter-gate combo for validation robustness.")
        if not any(_params_match(combo.params, rank_chosen_params) for combo in combos):
            raise ValueError(
                f"Eval signal leg {rank_chosen_params!r} is not present in the expanded IS grid."
            )
    elif chosen_combo is None:
        raise ValueError(
            f"Chosen eval combo {chosen_params!r} is not present in the expanded IS grid."
        )

    is_returns = _slice_returns(chosen_combo.returns, train_start, train_end)
    val_returns = _slice_returns(chosen_combo.returns, val_start, val_end)

    rank_rows, is_metrics, val_metrics = _build_rank_scatter_rows(
        config,
        combos,
        chosen_params=rank_chosen_params,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
    )
    # Rank correlation needs >= 3 IS grid points (Spearman p-value is non-finite at n=2).
    # Single-combo and 2-combo grids (e.g. EOF sma 200 vs 252) skip; other holdout checks still run.
    include_rank_correlation = is_metrics.shape[0] >= 3

    report = run_holdout_robustness_pipeline(
        is_returns,
        val_returns,
        config=holdout_config_from_research(config),
        include_rank_correlation=include_rank_correlation,
        rank_is_metrics=is_metrics.tolist() if include_rank_correlation else None,
        rank_holdout_metrics=val_metrics.tolist() if include_rank_correlation else None,
    )
    return report, rank_rows, chosen_combo.param_combo_label


def _rolling_z_mean(z_series: Sequence[float | None], *, window: int) -> list[float | None]:
    values = pd.Series(
        [np.nan if value is None else float(value) for value in z_series],
        dtype=float,
    )
    rolling = values.rolling(window, min_periods=1).mean()
    return [None if not math.isfinite(value) else float(value) for value in rolling.to_numpy()]


def _build_z_cusum_sr_frame(
    report: ValidationRobustnessReport,
    val_datetimes: Sequence[str],
) -> pd.DataFrame:
    cusum = report.cusum
    rolling = report.rolling_sharpe_zscore
    z_series = list(cusum.z_series)
    cusum_series = list(cusum.cusum_series)
    rolling_z = list(rolling.z_scores)
    z_rolling_mean = _rolling_z_mean(z_series, window=ROLLING_Z_MEAN_WINDOW)
    n_obs = max(len(z_series), len(cusum_series), len(rolling_z), len(val_datetimes))
    datetimes = [
        val_datetimes[index] if index < len(val_datetimes) else str(index)
        for index in range(n_obs)
    ]

    def _value_at(values: Sequence[float | None], index: int) -> float | None:
        if index >= len(values):
            return None
        value = values[index]
        if value is None:
            return None
        numeric = float(value)
        return None if not math.isfinite(numeric) else numeric

    return pd.DataFrame(
        {
            "bar_index": np.arange(n_obs, dtype=np.int64),
            "datetime": datetimes,
            "z_series": [_value_at(z_series, index) for index in range(n_obs)],
            "z_rolling_mean_60": [_value_at(z_rolling_mean, index) for index in range(n_obs)],
            "cusum_series": [_value_at(cusum_series, index) for index in range(n_obs)],
            "rolling_sharpe_z": [_value_at(rolling_z, index) for index in range(n_obs)],
            "cusum_statistic": cusum.statistic,
            "cusum_critical_value": cusum.critical_value,
            "rolling_z_threshold": rolling.z_threshold,
        }
    )


def _build_equity_bands_frame(report: ValidationRobustnessReport) -> pd.DataFrame:
    bands = report.equity_curve_bands
    return pd.DataFrame(
        {
            "bar_index": np.arange(len(bands.actual), dtype=np.int64),
            "actual": list(bands.actual),
            "expected": list(bands.expected),
            "upper_band": list(bands.upper_band),
            "lower_band": list(bands.lower_band),
            "fraction_below_lower": bands.fraction_below_lower,
            "outside_band": bands.outside_band,
        }
    )


def _markdown_summary(
    report: ValidationRobustnessReport,
    *,
    chosen_combo_label: str,
) -> str:
    sharpe = report.sharpe_comparison
    cusum = report.cusum
    bands = report.equity_curve_bands
    rolling = report.rolling_sharpe_zscore
    rank = report.rank_correlation
    lines = [
        "# Validation Robustness Summary",
        "",
        f"**Chosen combination:** {chosen_combo_label}",
        "",
        "## Sharpe comparison",
        "",
        f"- IS Sharpe: {sharpe.sr_is:.4f} [{sharpe.ci_is.lower:.4f}, {sharpe.ci_is.upper:.4f}]",
        f"- Val Sharpe: {sharpe.sr_val:.4f} [{sharpe.ci_val.lower:.4f}, {sharpe.ci_val.upper:.4f}]",
        f"- Degradation ratio: {sharpe.degradation_ratio:.4f}",
        f"- CI overlap: {'Yes' if sharpe.ci_overlap else 'No'}",
        f"- Pass: {'Yes' if sharpe.passed else 'No'}",
        "",
        "## CUSUM (IS parameters on validation returns)",
        "",
        f"- Statistic: {cusum.statistic:.4f} / critical {cusum.critical_value:.4f}",
        f"- Break detected: {'Yes' if cusum.break_detected else 'No'}",
        f"- Pass: {'Yes' if cusum.passed else 'No'}",
        "",
        "## Equity curve confidence bands",
        "",
        f"- Fraction below lower band: {bands.fraction_below_lower:.1%}",
        f"- Outside band: {'Yes' if bands.outside_band else 'No'}",
        f"- Pass: {'Yes' if bands.passed else 'No'}",
        "",
        "## Rolling Sharpe z-score",
        "",
        f"- Fraction below z={rolling.z_threshold:.1f}: {rolling.fraction_below_threshold:.1%}",
        f"- Unstable: {'Yes' if rolling.unstable else 'No'}",
        f"- Pass: {'Yes' if rolling.passed else 'No'}",
        "",
        "## Rank correlation",
        "",
        *(
            [f"- {rank.interpretation}", "- Pass: N/A (not computed)"]
            if "not applicable" in rank.interpretation.lower()
            else [
                f"- Spearman rho: {rank.spearman_rho:.4f} (p={rank.p_value:.4f})",
                f"- Pass: {'Yes' if rank.passed else 'No'}",
            ]
        ),
        "",
        f"**Overall:** {'PASS' if report.all_passed else 'FAIL'} — {report.interpretation}",
    ]
    return "\n".join(lines)


def write_validation_robustness_summary(
    report: ValidationRobustnessReport,
    output_dir: Path,
    *,
    rank_scatter_rows: Sequence[RankScatterRow],
    chosen_combo_label: str,
    val_datetimes: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Persist validation robustness JSON, markdown, and plot-input CSVs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_robustness_report.json"
    markdown_path = output_dir / "validation_robustness_summary.md"
    rank_csv = output_dir / "validation_rank_correlation.csv"
    z_cusum_csv = output_dir / "validation_z_cusum_sr.csv"
    bands_csv = output_dir / "validation_equity_bands.csv"

    json_path.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")
    markdown_path.write_text(
        _markdown_summary(report, chosen_combo_label=chosen_combo_label),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "param_combo_label": row.param_combo_label,
                "is_metric": row.is_metric,
                "val_metric": row.val_metric,
                "is_chosen": row.is_chosen,
            }
            for row in rank_scatter_rows
        ]
    ).to_csv(rank_csv, index=False)

    datetimes = (
        list(val_datetimes)
        if val_datetimes is not None
        else [str(index) for index in range(len(report.cusum.z_series))]
    )
    _build_z_cusum_sr_frame(report, datetimes).to_csv(z_cusum_csv, index=False)
    _build_equity_bands_frame(report).to_csv(bands_csv, index=False)

    return {
        "json": json_path,
        "markdown": markdown_path,
        "rank_correlation_csv": rank_csv,
        "z_cusum_sr_csv": z_cusum_csv,
        "equity_bands_csv": bands_csv,
    }


def refresh_validation_robustness_plots(
    input_dir: Path,
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    """Regenerate validation robustness plots from persisted JSON/CSV artifacts."""

    report_path = input_dir / "validation_robustness_report.json"
    rank_csv = input_dir / "validation_rank_correlation.csv"
    if not report_path.exists() or not rank_csv.exists():
        return []
    report = load_validation_robustness_report_from_json(report_path)
    rank_frame = pd.read_csv(rank_csv)
    chosen_rows = rank_frame.loc[rank_frame["is_chosen"].astype(bool), "param_combo_label"]
    chosen_label = str(chosen_rows.iloc[0]) if not chosen_rows.empty else ""
    plot_output_dir = output_dir or (input_dir / "matplotlib")
    from feature_research.visualization.validation_reports import write_validation_robustness_plots

    return write_validation_robustness_plots(
        report,
        plot_csv_dir=input_dir,
        output_dir=plot_output_dir,
        chosen_combo_label=chosen_label,
    )


def run_and_write_validation_robustness(
    config: ResearchConfig,
    *,
    output_dir: Path,
    selection_summary_df: pd.DataFrame | None = None,
    eval_research_data: FrozenSignalResearchData | None = None,
) -> dict[str, Path]:
    """Run validation robustness and write artifacts under ``output_dir``."""

    if not config.validation_robustness.enabled:
        return {}
    report, rank_rows, chosen_combo_label = run_validation_robustness_pipeline(
        config,
        selection_summary_df=selection_summary_df,
        eval_research_data=eval_research_data,
    )
    val_datetimes = [str(index) for index in range(len(report.cusum.z_series))]

    artifacts = write_validation_robustness_summary(
        report,
        output_dir,
        rank_scatter_rows=rank_rows,
        chosen_combo_label=chosen_combo_label,
        val_datetimes=val_datetimes,
    )

    from feature_research.visualization.validation_reports import write_validation_robustness_plots

    plot_dir = output_dir / "matplotlib"
    plot_paths = write_validation_robustness_plots(
        report,
        plot_csv_dir=output_dir,
        output_dir=plot_dir,
        chosen_combo_label=chosen_combo_label,
    )
    artifacts["plot_dir"] = plot_dir
    for index, plot_path in enumerate(plot_paths):
        artifacts[f"plot_{index}"] = plot_path
    return artifacts
