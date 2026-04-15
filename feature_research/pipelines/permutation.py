from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from feature_research.config import FeatureType, ResearchConfig
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_candles_for_config,
    populate_cache_if_needed,
)
from feature_research.research_table_exports import (
    canonical_in_sample_power_bi_dir,
    objective_metric_display_label,
    permutation_vector_shuffle_records,
    write_permutation_vector_shuffle_exports,
)
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec, resolve_objective_metric
from feature_selection.validation.orchestration import run_permutation_test_suite
from feature_selection.validation.stability_analysis import _param_combo_name
from feature_research.core_helpers import combo_key
from utils.evaluation.walkforward.research_data import (
    build_reference_target,
    load_signed_signal_research_data,
)

if TYPE_CHECKING:
    from feature_selection.validation.reports import PermutationTestSuite


def _require_permutation_enabled(config: ResearchConfig) -> None:
    if not config.permutation.enabled:
        raise ValueError("Permutation is disabled.")
    if config.feature_type == FeatureType.CONTINUOUS:
        raise ValueError(
            "Permutation is only supported for native signed-signal bias nodes. "
            "Continuous features remain research-only and do not enter permutation testing."
        )


def write_permutation_summary(
    suite: "PermutationTestSuite",
    output_dir: Path,
    *,
    objective_metric: ObjectiveMetricSpec | None = None,
    param_grid: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Write vector-shuffle permutation tables and root ``permutation_summary`` artifacts.

    ``feature_research`` runs vector shuffle only (no pipeline second phase). Detailed
    CSV for Power BI uses the fixed in-sample ``results/powerbi`` folder (see
    ``canonical_in_sample_power_bi_dir``); flat CSV/MD stay under ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    label = objective_metric_display_label(objective_metric)
    pbi = write_permutation_vector_shuffle_exports(
        suite, objective_metric_label=label, param_grid=param_grid
    )
    rows = permutation_vector_shuffle_records(suite, label, param_grid=param_grid)

    csv_path = output_dir / "permutation_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    fs = suite.funnel_stats
    md_lines = [
        "# In-Sample Permutation Summary",
        "",
        f"**Feature:** {suite.feature_name}  |  **Type:** {suite.feature_type}  |  **Objective:** `{label}`",
        "",
        "## Vector shuffle",
        "",
        "| Tested | Passed |",
        "|--------|--------|",
        f"| {fs.total_params} | {fs.stage1_pass} |",
        "",
    ]
    md_lines.extend(
        [
            "## Scope",
            "",
            "Each row is one **expanded** param combo from the same grid as in-sample EDA. "
            "For ``feature_type=CONTINUOUS``, each combo is "
            "**quantile-binned** per ticker (``binning_params.bin_counts[0]``) then mapped to "
            "±1/0 from ``strategy`` (LONG: lowest bin long; SHORT: highest bin short; "
            "LONG_SHORT: both tails). Signed-signal nodes use native discrete output.",
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
            f"Power BI: `{pbi['permutation_vector_shuffle_csv'].name}` in `{canonical_in_sample_power_bi_dir()}`",
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
) -> tuple["PermutationTestSuite", list[dict[str, Any]]]:
    """Run vector-shuffle permutation on the expanded research param grid.

    Uses the same ``bias_spec`` / ``feature_type`` as in-sample EDA (full grid).

    Returns
    -------
    tuple
        ``(suite, param_grid)`` for ``write_permutation_summary(..., param_grid=...)``.
    """
    _require_permutation_enabled(config)
    tr_start, tr_end = config.training_window_bounds

    output_dir.mkdir(parents=True, exist_ok=True)
    # Full-span cache warmup (same as EDA); permutation loads use training slice below.
    populate_cache_if_needed(config)

    config = replace(config, start=tr_start, end=tr_end)
    perm_load_config = config
    expanded = expand_bias_specs(perm_load_config.bias_spec)
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
    param_grid = data.successful_param_grid
    cached_signals_by_combo: dict[str, pd.Series] = {
        _param_combo_name(params): combo_frame["signal"]
        for params in data.successful_param_grid
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
        feature_spec=perm_load_config.bias_spec,
        target=target,
        param_grid=param_grid,
        objective_func=objective_func,
        config=permutation_config,
        extractor_func=extractor_func,
        feature_name=feature_col,
    )
    return suite, param_grid
