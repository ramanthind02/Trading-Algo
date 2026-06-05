"""Filter-gate A/B/C comparison on the exploration-winning signal combo."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research.feature.config import ResearchConfig
from research.feature.exploration.filter_gate_catalog import (
    exploration_filter_gate_combo_multiplier,
    resolved_filter_gate_bias_spec,
)
from research.feature.filter_research_labels import build_filter_exploration_long_pairs
from research.feature.in_sample.data_loader import (
    enrich_param_combo_with_module,
    expand_bias_specs,
    expanded_spec_combo_label,
    first_bias_spec,
    load_candles_for_config,
    load_features_for_combo,
    populate_cache_if_needed,
)
from research.feature.in_sample.metric_helpers import compute_param_sensitivity_metric
from research.feature.research_table_exports import (
    return_kind_for_target,
    write_filter_exploration_tables,
    write_in_sample_equity_curve_csv,
)
from research.feature._internal.core_helpers import normalize_timeframe_from_bias_spec
import pandas as pd


def run_filter_gate_exploration_for_winner(
    config: ResearchConfig,
    chosen_params: Mapping[str, Any],
) -> dict[str, Path]:
    """Load filter variants for the winning signal; write summary + equity curve CSVs."""
    gates = config.exploration_filter_gates
    if not gates.enabled:
        return {}

    filter_catalog = resolved_filter_gate_bias_spec(config, chosen_params)
    populate_cache_if_needed(config, bias_spec=filter_catalog)
    expanded = expand_bias_specs(filter_catalog)
    n_expected = exploration_filter_gate_combo_multiplier(
        rank_perturbation_pct=gates.rank_perturbation_pct
    )
    print(
        f"\nFilter-gate follow-up: {len(expanded)} combos on winning signal "
        f"(expected ~{n_expected})..."
    )

    timeframe = normalize_timeframe_from_bias_spec(first_bias_spec(filter_catalog))
    combo_store: dict[str, tuple[pd.Series, pd.Series, str, pd.Series, dict[str, object]]] = {}
    sensitivity_rows: list[dict[str, object]] = []

    for single_spec in expanded:
        combo = enrich_param_combo_with_module(
            single_spec["params"],
            single_spec.get("module_name"),
        )
        label = expanded_spec_combo_label(single_spec)
        data = load_features_for_combo(single_spec, config)
        if data is None:
            print(f"  [{label}] SKIP — no data")
            continue

        feature, target, feature_col, ticker_s = data
        paired = pd.concat(
            [
                feature.rename("signal"),
                target.rename("target"),
                ticker_s.rename("ticker"),
            ],
            axis=1,
        ).dropna(how="any")
        if paired.empty:
            print(f"  [{label}] SKIP — aligned feature/target empty")
            continue

        signal = paired["signal"]
        target = paired["target"]
        ticker_series = paired["ticker"]
        combo_store[label] = (signal, target, feature_col, ticker_series, dict(combo))

        strategy_returns = (signal * target).to_numpy(dtype=float)
        n_observations = int(len(strategy_returns))
        n_nonzero_signal = int((signal.fillna(0).to_numpy() != 0).sum())
        sharpe_v = float("nan")
        t_stat_v = float("nan")
        sortino_v = float("nan")
        if n_observations >= 5:
            try:
                sharpe_v = compute_param_sensitivity_metric(
                    strategy_returns, "sharpe", timeframe
                )
                t_stat_v = compute_param_sensitivity_metric(
                    strategy_returns, "t_stat", timeframe
                )
                sortino_v = compute_param_sensitivity_metric(
                    strategy_returns, "sortino", timeframe
                )
            except Exception:
                pass
        sensitivity_rows.append(
            {
                "param_combo_label": label,
                "feature_name": feature_col,
                "n_observations": n_observations,
                "n_nonzero_signal": n_nonzero_signal,
                "sharpe": sharpe_v,
                "t_stat": t_stat_v,
                "sortino": sortino_v,
            }
        )

    if not combo_store:
        print("Filter-gate follow-up: no combos loaded.")
        return {}

    paths = write_filter_exploration_tables(sensitivity_rows, combo_store)
    print(
        "Filter exploration summary: "
        f"{paths.get('filter_exploration_summary_csv', '')}"
    )

    try:
        portfolio_candles = load_candles_for_config(config)
    except Exception:
        portfolio_candles = None
    equity_path = write_in_sample_equity_curve_csv(
        combo_store,
        portfolio_candles=portfolio_candles,
        timeframe=timeframe,
        target_volatility=config.tearsheet_target_annual_volatility or 0.15,
        instrument_return_kind=return_kind_for_target(config.target_col),
        csv_stem="filter_exploration_equity_curve",
    )
    paths["filter_exploration_equity_curve_csv"] = equity_path
    print(f"Filter exploration equity curve CSV: {equity_path}")
    return paths
