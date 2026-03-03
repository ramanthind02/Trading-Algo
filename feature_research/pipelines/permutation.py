from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from feature_research.config import FeatureType
from feature_research.core_helpers import expand_params_with_bin_count, normalize_timeframe_from_bias_spec
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_candles_for_config,
    load_features_for_combo,
    populate_cache_if_needed,
)
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validation.config import OutOfSamplePermutationConfig, PermutationTestConfig
from feature_selection.validation.objective_metrics import resolve_objective_metric
from feature_selection.validation.orchestration import run_permutation_test_suite

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig
    from feature_research.walkforward.runner import WalkforwardRunReport
    from feature_selection.validation.reports import PermutationTestSuite


def _build_fold_structure(
    start: datetime,
    end: datetime,
    fold_years: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    step_years = max(1, fold_years)
    fold_start = pd.Timestamp(start).normalize()
    end_exclusive = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)

    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while fold_start < end_exclusive:
        fold_end = min(fold_start + pd.DateOffset(years=step_years), end_exclusive)
        if fold_end <= fold_start:
            break
        folds.append((fold_start, fold_end))
        fold_start = fold_end
    return folds


def write_permutation_summary(suite: "PermutationTestSuite", output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for combo_name, s1 in suite.stage1_reports.items():
        s2 = suite.stage2_reports.get(combo_name)
        row: dict[str, object] = {
            "param_combo": combo_name,
            "stage1_metric": s1.original_metric,
            "stage1_pval": s1.p_value,
            "stage1_passed": s1.passed,
            "stage1_alpha": s1.alpha,
        }
        if s2 is not None:
            row["stage2_metric"] = s2.original_metric
            row["stage2_pval"] = s2.p_value
            row["stage2_passed"] = s2.passed
            row["stage2_mode"] = s2.permutation_mode
        else:
            row["stage2_metric"] = None
            row["stage2_pval"] = None
            row["stage2_passed"] = False
            row["stage2_mode"] = "skipped"
        rows.append(row)

    csv_path = output_dir / "permutation_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    fs = suite.funnel_stats
    md_lines = [
        "# In-Sample Permutation Summary",
        "",
        f"**Feature:** {suite.feature_name}  |  **Type:** {suite.feature_type}",
        "",
        "## Funnel (vector shuffle → pipeline permutation)",
        "",
        "| Stage | Tested | Passed |",
        "|-------|--------|--------|",
        f"| Stage 1 (vector shuffle) | {fs.total_params} | {fs.stage1_pass} |",
        f"| Stage 2 (pipeline/candle) | {fs.stage1_pass} | {fs.stage2_pass} |",
        "",
        f"**Computational savings (Stage 1 gate):** {fs.computational_savings_pct:.1f}%",
        "",
        "## Per-combo metrics and p-values",
        "",
        "| param_combo | S1 metric | S1 p-val | S1 pass | S2 metric | S2 p-val | S2 pass |",
        "|-------------|-----------|----------|--------|-----------|----------|--------|",
    ]
    for row in rows:
        s1_metric = row["stage1_metric"]
        s1_pvalue = row["stage1_pval"]
        s1_pass = "✓" if row["stage1_passed"] else "✗"
        s2_metric = row.get("stage2_metric")
        s2_pvalue = row.get("stage2_pval")
        s2_pass_cell = "✓" if row.get("stage2_passed") else ("—" if s2_metric is None and s2_pvalue is None else "✗")
        s2_metric_text = f"{s2_metric:.4f}" if s2_metric is not None else "—"
        s2_pvalue_text = f"{s2_pvalue:.4f}" if s2_pvalue is not None else "—"
        md_lines.append(
            f"| {row['param_combo']} | {s1_metric:.4f} | {s1_pvalue:.4f} | {s1_pass} | {s2_metric_text} | {s2_pvalue_text} | {s2_pass_cell} |"
        )
    md_lines.extend(["", "Full data: `permutation_summary.csv`", ""])

    md_path = output_dir / "permutation_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return (csv_path, md_path)


def run_permutation_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> "PermutationTestSuite":
    if not config.in_sample_permutation.enabled:
        raise ValueError("Permutation suite is disabled; set config.in_sample_permutation.enabled=True.")

    output_dir.mkdir(parents=True, exist_ok=True)
    populate_cache_if_needed(config)
    expanded = expand_bias_specs(config.bias_spec)
    if not expanded:
        raise ValueError("No parameter combinations available for permutation suite.")

    candles_df = load_candles_for_config(config)
    seed_spec = expanded[0]
    try:
        seed_feature_data = load_features_for_combo(seed_spec, config, candles_override=candles_df)
    except ValueError as exc:
        err_msg = str(exc)
        if "Tickers dropped" in err_msg or "Missing tickers" in err_msg or "missing ticker data" in err_msg:
            single_ticker_config = replace(config, tickers=[config.tickers[0]])
            print(
                f"  [permutation] Multi-ticker alignment failed; using single ticker: {single_ticker_config.tickers[0].name}"
            )
            candles_df = load_candles_for_config(single_ticker_config)
            config = single_ticker_config
            seed_feature_data = load_features_for_combo(seed_spec, config, candles_override=candles_df)
        else:
            raise
    if seed_feature_data is None:
        raise ValueError("Unable to load feature/target data. Ensure cache and candles are available.")

    _, target, feature_col = seed_feature_data
    module_name = config.bias_spec["module_name"]
    timeframe = normalize_timeframe_from_bias_spec(config.bias_spec)

    if config.feature_type == FeatureType.CONTINUOUS:
        param_grid = [
            combo_params
            for single_spec in expanded
            for combo_params in expand_params_with_bin_count(
                dict(single_spec["params"]),
                config.binning_params.bin_counts,
            )
        ]
    else:
        param_grid = [single_spec["params"] for single_spec in expanded]

    bias_only_keys = frozenset(config.bias_spec.get("params", {}).keys())

    def extractor_func(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        bias_params = {k: v for k, v in params.items() if k in bias_only_keys}
        single_spec = {
            "module_name": module_name,
            "timeframes": [timeframe],
            "params": bias_params,
        }
        loaded = load_features_for_combo(single_spec, config, candles_override=df)
        if loaded is None:
            raise ValueError(f"Feature extraction returned no data for params={params}.")
        feature, _, feature_name = loaded
        return feature.rename(feature_name)

    permutation_config = PermutationTestConfig(
        nreps=config.in_sample_permutation.nreps_stage2,
        alpha=config.in_sample_permutation.alpha,
        metric_threshold=config.in_sample_permutation.metric_threshold,
        top_k=config.in_sample_permutation.top_k,
        random_seed=config.in_sample_permutation.random_seed,
        permutation_mode_stage2=config.in_sample_permutation.permutation_mode_stage2,
        min_folds_stable=config.in_sample_permutation.min_folds_stable,
        n_jobs_stage2_reps=config.in_sample_permutation.n_jobs_stage2_reps,
        run_stage1=config.in_sample_permutation.run_stage1,
        run_stage2=config.in_sample_permutation.run_stage2,
        run_stage3_walkforward=False,
        out_of_sample=OutOfSamplePermutationConfig(
            objective_metric=config.in_sample_permutation.objective_metric,
            run_oos_permutation=False,
        ),
    )
    objective_func = resolve_objective_metric(config.in_sample_permutation.objective_metric)
    fold_structure = _build_fold_structure(
        config.start,
        config.end,
        config.in_sample_permutation.fold_years,
    )

    if config.feature_type == FeatureType.CONTINUOUS:
        bp = config.binning_params

        def binning_model_factory(params: dict[str, Any]) -> ContinuousBinningModel:
            bin_count = int(cast(int, params.get("bin_count", bp.bin_counts[0])))
            return ContinuousBinningModel(
                n_bins=bin_count,
                bin_counts=[bin_count],
                selection_metric=bp.selection_metric,
                strategy=bp.strategy,
                metric_threshold=bp.metric_threshold,
                t_threshold=bp.t_threshold,
                min_region_width=bp.min_region_width,
                shrinkage_k=bp.shrinkage_k,
                long_clip_min=bp.long_clip_min,
                long_clip_max=bp.long_clip_max,
                short_clip_min=bp.short_clip_min,
                short_clip_max=bp.short_clip_max,
                use_coverage_bonus=bp.use_coverage_bonus,
                coverage_bonus_per_10pct=bp.coverage_bonus_per_10pct,
                max_coverage_bonus=bp.max_coverage_bonus,
                bin_index_min=bp.bin_index_min,
                bin_index_max=bp.bin_index_max,
            )
    else:
        def binning_model_factory(_params: dict[str, Any]) -> None:
            return None

    return run_permutation_test_suite(
        candles_df=candles_df,
        feature_spec=config.bias_spec,
        target=target,
        param_grid=param_grid,
        objective_func=objective_func,
        fold_structure=fold_structure,
        config=permutation_config,
        extractor_func=extractor_func,
        binning_model_factory=binning_model_factory,
        feature_type=config.feature_type.value,
        feature_name=feature_col,
    )
