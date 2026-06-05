from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from research.feature.config import FeatureType, ResearchConfig, VectorShuffleScope
from research.feature.exploration.filter_gate_catalog import resolved_exploration_bias_spec
from research.feature.in_sample.data_loader import (
    expand_bias_specs,
    first_bias_spec,
    load_candles_for_config,
    populate_cache_if_needed,
)
from research.feature.research_table_exports import (
    canonical_in_sample_visualization_dir,
    objective_metric_display_label,
    permutation_vector_shuffle_records,
    write_permutation_vector_shuffle_exports,
)
from features.validation.objective_metrics import ObjectiveMetricSpec, resolve_objective_metric
from features.validation.orchestration import run_permutation_test_suite
from features.validation.stability_analysis import _param_combo_name
from research.feature._internal.core_helpers import combo_key
from research.feature.pipelines.permutation_signals import (
    load_quantile_binned_permutation_research_data,
)
from research.evaluation.walkforward.research_data import (
    build_reference_target,
    load_signed_signal_research_data,
)

if TYPE_CHECKING:
    from features.validation.reports import PermutationTestSuite
    from quantfoundry_core.robustness import PermutationTestResult


def _full_grid_permutation_markdown_lines(
    full_grid: "PermutationTestResult",
    *,
    selection_metric_label: str,
) -> list[str]:
    """Markdown section for SaaS full-grid search-bias results (robustness / Core)."""

    best = full_grid.observed_best_combination
    best_label = best.label or ", ".join(f"{key}={value}" for key, value in best.params.items())
    metric_obj = getattr(full_grid, "metric", None)
    metric_name = (
        getattr(metric_obj, "metric_name", selection_metric_label)
        if metric_obj is not None
        else selection_metric_label
    )
    return [
        "## Full-grid search-bias permutation (SaaS §3.1)",
        "",
        "Runs inside the **robustness** step: each null iteration shuffles the target "
        "return series and re-scores the **entire** parameter grid. Uses a **plain** "
        f"(non-HAC) metric (`{metric_name}`) for both observed and null scores so the "
        "return-shuffle null is calibrated fairly — NW/HAC correction collapses under IID "
        "shuffled returns and would inflate null max scores relative to the real-data path. "
        "Combo selection on real data still uses the NW-adjusted exploration metric "
        f"(`{selection_metric_label}`). This is **not** the per-combo vector-shuffle below.",
        "",
        "| Grid size | Observed best (plain metric) | p-value | Null iterations |",
        "|----------:|-----------------------------:|--------:|----------------:|",
        (
            f"| {full_grid.n_combinations} | {best_label} "
            f"({full_grid.observed_score:.4f}) | {full_grid.p_value:.4f} | "
            f"{full_grid.config.n_permutations} |"
        ),
        "",
        "Canonical detail: `robustness_summary.md`, `robustness_summary.csv` (`full_grid_p_value`), "
        "and `robustness_report.json` → `full_grid_permutation`.",
        "",
    ]


def _require_permutation_enabled(config: ResearchConfig) -> None:
    if not config.permutation.enabled:
        raise ValueError("Permutation is disabled.")
    if config.feature_type == FeatureType.CONTINUOUS:
        raise ValueError(
            "Permutation is only supported for native signed-signal bias nodes. "
            "Continuous features remain research-only and do not enter permutation testing."
        )


def _filter_param_grid_to_selected_combo(
    param_grid: list[dict[str, Any]],
    *,
    selected_combo_name: str,
) -> list[dict[str, Any]]:
    matched = [
        spec for spec in param_grid if _param_combo_name(spec) == selected_combo_name
    ]
    if len(matched) != 1:
        raise ValueError(
            f"Expected exactly one param-grid entry for selected combo {selected_combo_name!r}; "
            f"found {len(matched)} among {len(param_grid)} expanded specs."
        )
    return matched


def _resolve_vector_shuffle_param_grid(
    config: ResearchConfig,
    param_grid: list[dict[str, Any]],
    *,
    selected_combo_name: str | None,
) -> list[dict[str, Any]]:
    scope = config.permutation.vector_shuffle_scope
    if scope is VectorShuffleScope.FULL_GRID:
        return param_grid
    if selected_combo_name is None:
        raise ValueError(
            "vector_shuffle_scope=selected_combo requires a robustness winner "
            "(run with robustness.enabled=True before permutation)."
        )
    return _filter_param_grid_to_selected_combo(
        param_grid,
        selected_combo_name=selected_combo_name,
    )


def write_permutation_summary(
    suite: "PermutationTestSuite",
    output_dir: Path,
    *,
    objective_metric: ObjectiveMetricSpec | None = None,
    param_grid: list[dict[str, Any]] | None = None,
    full_grid_permutation: "PermutationTestResult | None" = None,
    selection_metric_label: str | None = None,
) -> tuple[Path, Path]:
    """Write vector-shuffle permutation tables and root ``permutation_summary`` artifacts.

  Vector shuffle is written here. When ``full_grid_permutation`` is provided (from the
  robustness step), a leading markdown section documents the search-bias test so readers
  are not misled into thinking vector-shuffle is the only permutation run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    label = objective_metric_display_label(objective_metric)
    visualization_exports = write_permutation_vector_shuffle_exports(
        suite, objective_metric_label=label, param_grid=param_grid
    )
    rows = permutation_vector_shuffle_records(suite, label, param_grid=param_grid)

    csv_path = output_dir / "permutation_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    fs = suite.funnel_stats
    metric_label = selection_metric_label or label
    md_lines = [
        "# In-Sample Permutation Summary",
        "",
        f"**Feature:** {suite.feature_name}  |  **Type:** {suite.feature_type}  |  **Objective:** `{label}`",
        "",
    ]
    if full_grid_permutation is not None:
        md_lines.extend(
            _full_grid_permutation_markdown_lines(
                full_grid_permutation,
                selection_metric_label=metric_label,
            )
        )
    md_lines.extend(
        [
        f"## Vector shuffle ({'selected combo' if fs.total_params == 1 else 'per combo'})",
        "",
        "| Tested | Passed |",
        "|--------|--------|",
        f"| {fs.total_params} | {fs.stage1_pass} |",
        "",
        ]
    )
    md_lines.extend(
        [
            "## P-value formula (vector shuffle)",
            "",
            "One-sided Monte Carlo with +1 pseudo-count: "
            "`p_value = (1 + null_ge_count) / (n_reps + 1)`, where `null_ge_count` is the "
            "number of null replicates with metric ≥ observed. With `n_reps=100`, only "
            "multiples of `1/101` are possible (not `1/100`). `passed` uses the empirical "
            f"`{1 - float(rows[0]['alpha']) if rows else 0.9:.0%}` null quantile (`critical_value`), "
            "not `p_value <= alpha` alone.",
            "",
            "## Scope",
            "",
            (
                "Single **selected** param combo — the robustness selection-metric winner."
                if fs.total_params == 1
                else "Each row is one **expanded** param combo from the same grid as in-sample EDA."
            )
            + " "
            + (
                "Signed-signal nodes use native discrete output."
                if fs.total_params == 1
                else (
                    "For ``feature_type=CONTINUOUS``, each combo is "
                    "**quantile-binned** per ticker (``binning_params.bin_counts[0]``) then mapped to "
                    "±1/0 from ``strategy`` (LONG: lowest bin long; SHORT: highest bin short; "
                    "LONG_SHORT: both tails). Signed-signal nodes use native discrete output."
                )
            ),
            "",
            "## Per-combo results",
            "",
            "| param_combo (readable) | observed_metric | p-value | pass | alpha | n_reps |",
            "|------------------------|-----------------|--------|------|-------|--------|",
        ]
    )
    for row in rows:
        obs = row["observed_metric"]
        pv = row["p_value"]
        ok = "✓" if row["passed"] else "✗"
        readable = str(row.get("param_combo_label") or row["param_combo"]).replace("|", "·")
        md_lines.append(
            f"| {readable} | {float(obs):.4f} | {float(pv):.4f} | {ok} | {row['alpha']} | {row['n_reps']} |"
        )
    md_lines.extend(
        [
            "",
            (
                "Visualization CSV: "
                f"`{visualization_exports['permutation_vector_shuffle_csv'].name}` in "
                f"`{canonical_in_sample_visualization_dir()}`"
            ),
            f"Flat summary: `permutation_summary.csv` (columns include `param_combo` = stable key, "
            "`param_combo_label` = readable).",
            "",
        ]
    )

    md_path = output_dir / "permutation_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return (csv_path, md_path)


def run_permutation_pipeline(
    config: ResearchConfig,
    output_dir: Path,
    *,
    selected_combo_name: str | None = None,
) -> tuple["PermutationTestSuite", list[dict[str, Any]]]:
    """Run vector-shuffle permutation on the exploration param grid or one selected combo.

    When ``config.permutation.vector_shuffle_scope`` is ``selected_combo``, pass
    ``selected_combo_name`` from the robustness selection-metric winner.

    Returns
    -------
    tuple
        ``(suite, param_grid)`` for ``write_permutation_summary(..., param_grid=...)``.
    """
    _require_permutation_enabled(config)
    tr_start, tr_end = config.training_window_bounds

    output_dir.mkdir(parents=True, exist_ok=True)
    # Full-span cache warmup (same as EDA); permutation loads use training slice below.
    exploration_spec = resolved_exploration_bias_spec(config)
    populate_cache_if_needed(config, bias_spec=exploration_spec)

    config = replace(config, start=tr_start, end=tr_end)
    perm_load_config = config
    expanded = expand_bias_specs(exploration_spec)
    if not expanded:
        raise ValueError("No parameter combinations available for permutation suite.")

    use_binned_continuous = config.feature_type == FeatureType.CONTINUOUS
    load_mode = (
        "quantile-binned ±1/0 (per ticker, same n_bins as binning_params)"
        if use_binned_continuous
        else "native signed signal"
    )
    print(
        f"Permutation: preloading signals for {len(expanded)} combos ({load_mode}) "
        f"(n_jobs={getattr(config, 'n_jobs', 1)})..."
    )
    try:
        data = (
            load_quantile_binned_permutation_research_data(perm_load_config, expanded, print_loaded=False)
            if use_binned_continuous
            else load_signed_signal_research_data(perm_load_config, expanded, print_loaded=False)
        )
    except ValueError as exc:
        err_msg = str(exc)
        if "Tickers dropped" in err_msg or "Missing tickers" in err_msg or "missing ticker data" in err_msg:
            single_ticker_config = replace(perm_load_config, tickers=[perm_load_config.tickers[0]])
            print(
                f"  [permutation] Multi-ticker alignment failed; using single ticker: {single_ticker_config.tickers[0].name}"
            )
            perm_load_config = single_ticker_config
            print(
                f"Permutation: retry preloading ({load_mode}) "
                f"(n_jobs={getattr(config, 'n_jobs', 1)})..."
            )
            data = (
                load_quantile_binned_permutation_research_data(perm_load_config, expanded, print_loaded=False)
                if use_binned_continuous
                else load_signed_signal_research_data(perm_load_config, expanded, print_loaded=False)
            )
        else:
            raise
    if not data.successful_param_grid or data.reference_index is None:
        raise ValueError("Unable to load feature/target data. Ensure cache and candles are available.")

    candles_df = load_candles_for_config(perm_load_config)
    target = build_reference_target(data.reference_index, data.reference_target_series)
    first_combo_key = next(iter(data.combo_signal_target))
    first_combo_frame = data.combo_signal_target[first_combo_key]
    signal_column = "signal" if "signal" in first_combo_frame.columns else "feature"
    feature_col = str(first_combo_frame[signal_column].name or "signal")
    param_grid = _resolve_vector_shuffle_param_grid(
        config,
        data.successful_param_grid,
        selected_combo_name=selected_combo_name,
    )
    scope_label = (
        f"selected combo {selected_combo_name!r}"
        if config.permutation.vector_shuffle_scope is VectorShuffleScope.SELECTED_COMBO
        else f"{len(param_grid)} combos"
    )
    print(f"Permutation: vector shuffle on {scope_label} ({load_mode}) ...")
    cached_signals_by_combo: dict[str, pd.Series] = {
        _param_combo_name(params): combo_frame["signal"]
        for params in param_grid
        for combo_frame in [data.combo_signal_target[combo_key(params)]]
    }

    def extractor_func(_df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        combo_name = _param_combo_name(params)
        cached_signal = cached_signals_by_combo.get(combo_name)
        if cached_signal is not None:
            return cached_signal
        raise KeyError(f"Missing preloaded permutation signal for params={params}.")

    permutation_config = config.permutation.to_permutation_test_config()
    objective_func = resolve_objective_metric(config.permutation.objective_metric)

    # Do not pass ``aligned_signals_by_combo``: the batched path permutes the **target**
    # (``run_vector_shuffle_target_perm_batch``), which is a different null than
    # ``run_vector_shuffle_test`` (permute feature values on the timeline, fixed target).
    # Feature shuffle is the intended "vector shuffle" for signal alignment.
    suite = run_permutation_test_suite(
        candles_df=candles_df,
        feature_spec=first_bias_spec(perm_load_config.bias_spec),
        target=target,
        param_grid=param_grid,
        objective_func=objective_func,
        config=permutation_config,
        extractor_func=extractor_func,
        feature_name=feature_col,
    )
    return suite, param_grid
