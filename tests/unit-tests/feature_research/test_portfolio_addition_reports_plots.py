from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from quantfoundry_core.portfolio_gate import compute_portfolio_addition_gate

from research.feature.portfolio_addition.gate_runner import (
    PortfolioGateInputs,
    build_pairwise_peer_rows,
    build_portfolio_addition_context,
    build_weight_layer_hierarchy_detail,
    build_weight_layer_member_table,
    build_weight_layer_tree_from_members,
    write_portfolio_addition_context_csvs,
    write_portfolio_addition_summary,
)
from research.feature.visualization.portfolio_addition_reports import (
    _correlation_heatmap_axis_labels,
    write_portfolio_addition_plots,
)


def _synthetic_inputs(*, n: int = 260, seed: int = 11) -> PortfolioGateInputs:
    rng = np.random.default_rng(seed)
    existing_a = rng.normal(0.0004, 0.01, size=n)
    existing_b = rng.normal(0.0003, 0.011, size=n)
    new_strategy = rng.normal(0.0005, 0.012, size=n)
    portfolio_without = 0.5 * existing_a + 0.5 * existing_b
    portfolio_with = 0.4 * existing_a + 0.35 * existing_b + 0.25 * new_strategy
    labels = ["candidate", "ens_a", "ens_b"]
    matrix = np.array(
        [
            [1.0, float(np.corrcoef(new_strategy, existing_a)[0, 1]), float(np.corrcoef(new_strategy, existing_b)[0, 1])],
            [float(np.corrcoef(existing_a, new_strategy)[0, 1]), 1.0, float(np.corrcoef(existing_a, existing_b)[0, 1])],
            [float(np.corrcoef(existing_b, new_strategy)[0, 1]), float(np.corrcoef(existing_b, existing_a)[0, 1]), 1.0],
        ]
    )
    weight_without = pd.DataFrame(
        {
            "weight_layer_ticker": ["__GLOBAL__", "__GLOBAL__"],
            "stream_or_model_id": ["s1", "s2"],
            "stream_source_model_name": ["ens_a", "ens_b"],
            "stream_weight": [0.55, 0.45],
            "weight_method": ["ledoit_wolf_min_corr", "ledoit_wolf_min_corr"],
            "cluster_id": [
                "root/equity_indices/momentum/s1",
                "root/equity_indices/momentum/s2",
            ],
        }
    )
    weight_with = pd.DataFrame(
        {
            "weight_layer_ticker": ["__GLOBAL__", "__GLOBAL__", "__GLOBAL__"],
            "stream_or_model_id": ["s1", "s2", "s3"],
            "stream_source_model_name": ["ens_a", "ens_b", "candidate"],
            "stream_weight": [0.40, 0.35, 0.25],
            "weight_method": ["ledoit_wolf_min_corr", "ledoit_wolf_min_corr", "ledoit_wolf_min_corr"],
            "cluster_id": [
                "root/equity_indices/momentum/s1",
                "root/equity_indices/momentum/s2",
                "root/equity_indices/momentum/s3",
            ],
        }
    )
    return PortfolioGateInputs(
        new_strategy_returns=new_strategy,
        existing_strategy_returns={"ens_a": existing_a, "ens_b": existing_b},
        portfolio_returns_without=portfolio_without,
        portfolio_returns_with=portfolio_with,
        weight_assigned=0.25,
        candidate_key="candidate",
        eval_start=pd.Timestamp("2018-01-01"),
        eval_end=pd.Timestamp("2018-09-16"),
        weight_layer_without_df=weight_without,
        weight_layer_with_df=weight_with,
        pairwise_corr_matrix=pd.DataFrame(matrix, index=labels, columns=labels),
        sleeve_asset_class="equity_indices",
        sleeve_style_group="momentum",
        sleeve_baseline_ensemble_dirs={"ens_a": "vault/D/momentum/a"},
        sleeve_returns_without=0.5 * existing_a + 0.5 * existing_b,
        sleeve_returns_with=portfolio_with,
        sleeve_existing_strategy_returns={"ens_a": existing_a},
        sleeve_weight_assigned=0.25,
    )


def test_build_portfolio_addition_context_includes_peers_and_weight_members() -> None:
    inputs = _synthetic_inputs()
    report = compute_portfolio_addition_gate(
        inputs.new_strategy_returns,
        inputs.existing_strategy_returns,
        inputs.portfolio_returns_without,
        inputs.portfolio_returns_with,
        weight_assigned=inputs.weight_assigned,
        n_bootstrap=100,
        bootstrap_seed=1,
    )
    context = build_portfolio_addition_context(
        inputs,
        report,
        weight_layer_method="ledoit_wolf_min_corr",
        pairwise_flag_threshold=0.75,
    )
    assert len(context["pairwise_peers"]) == 2
    assert len(context["weight_layer_members"]) == 3
    assert "weight_layer_detail" in context
    assert "drawdown_detail" in context
    detail = context["weight_layer_detail"]
    assert isinstance(detail, dict)
    assert len(detail.get("tree", [])) >= 1
    peers = build_pairwise_peer_rows(
        inputs.pairwise_corr_matrix,
        candidate_key="candidate",
        flag_threshold=0.75,
    )
    assert peers[0]["peer_member"] in {"ens_a", "ens_b"}


def test_build_weight_layer_tree_groups_by_asset_and_style() -> None:
    members = [
        {
            "stream_id": "ES::D::m1",
            "model_name": "m1",
            "asset_class": "equity_indices",
            "style_group": "momentum",
            "weight_without": 0.2,
            "weight_with": 0.15,
            "weight_delta": -0.05,
            "is_candidate_stream": False,
        },
        {
            "stream_id": "GC::D::m2",
            "model_name": "m2",
            "asset_class": "commodities",
            "style_group": "momentum",
            "weight_without": 0.0,
            "weight_with": 0.25,
            "weight_delta": 0.25,
            "is_candidate_stream": True,
        },
    ]
    tree = build_weight_layer_tree_from_members(members)
    assert tree[0]["id"] == "commodities" or tree[0]["id"] == "equity_indices"
    assert any(node["id"] == "equity_indices" for node in tree)


def test_build_weight_layer_hierarchy_detail_includes_candidate_streams() -> None:
    inputs = _synthetic_inputs()
    members = build_weight_layer_member_table(
        inputs.weight_layer_without_df,
        inputs.weight_layer_with_df,
        candidate_key="candidate",
    ).to_dict(orient="records")
    detail = build_weight_layer_hierarchy_detail(
        inputs,
        weight_layer_method="ledoit_wolf_min_corr",
        weight_layer_kwargs={"fdm_max": 2.0},
        member_records=members,
    )
    assert detail["candidate_key"] == "candidate"
    assert float(detail["candidate_weight_total"]) == pytest.approx(1.0 / 3.0)


def test_write_portfolio_addition_plots_include_heatmap_and_drawdown(tmp_path: Path) -> None:
    inputs = _synthetic_inputs()
    report = compute_portfolio_addition_gate(
        inputs.new_strategy_returns,
        inputs.existing_strategy_returns,
        inputs.portfolio_returns_without,
        inputs.portfolio_returns_with,
        weight_assigned=inputs.weight_assigned,
        n_bootstrap=100,
        bootstrap_seed=2,
    )
    context = build_portfolio_addition_context(
        inputs,
        report,
        weight_layer_method="ledoit_wolf_min_corr",
        pairwise_flag_threshold=0.75,
    )
    payload = {"skipped": False, "context": context, **report.to_json_dict()}
    write_portfolio_addition_summary(payload, tmp_path, bootstrap_samples=[0.01, 0.02, 0.03])
    csv_paths = write_portfolio_addition_context_csvs(inputs, report, tmp_path)
    plot_paths = write_portfolio_addition_plots(
        payload,
        bootstrap_csv=tmp_path / "portfolio_addition_delta_sr_bootstrap.csv",
        corr_matrix_csv=csv_paths["pairwise_corr_matrix_csv"],
        drawdown_detail_csv=csv_paths["drawdown_detail_csv"],
        output_dir=tmp_path / "matplotlib",
    )
    names = {path.name for path in plot_paths}
    assert "portfolio_addition_delta_sr_histogram.png" in names
    assert "portfolio_addition_correlation_heatmap.png" in names
    assert "portfolio_addition_drawdown_correlation.png" in names
    weight_table = build_weight_layer_member_table(
        inputs.weight_layer_without_df,
        inputs.weight_layer_with_df,
        candidate_key="candidate",
    )
    assert len(weight_table) == 3
    assert float(weight_table.loc[weight_table["stream_id"] == "s3", "weight_with"].iloc[0]) == pytest.approx(
        1.0 / 3.0
    )


def test_build_weight_layer_member_table_uses_equal_split_within_style_group() -> None:
    """Portfolio gate UI shows dilution from adding streams, not signed SR-tilt artifacts."""
    cluster = "root/equity_indices/mean_reversion_indices"
    without = pd.DataFrame(
        {
            "weight_layer_ticker": ["__GLOBAL__"] * 3,
            "stream_or_model_id": ["s1", "s2", "s3"],
            "stream_source_model_name": ["m1", "m2", "m3"],
            "stream_weight": [0.032, 0.032, 0.030],
            "weight_method": ["hierarchy_equal"] * 3,
            "cluster_id": [f"{cluster}/s1", f"{cluster}/s2", f"{cluster}/s3"],
        }
    )
    with_portfolio = pd.DataFrame(
        {
            "weight_layer_ticker": ["__GLOBAL__"] * 5,
            "stream_or_model_id": ["s1", "s2", "s3", "cand_es", "cand_nq"],
            "stream_source_model_name": ["m1", "m2", "m3", "cand_es", "cand_nq"],
            "stream_weight": [0.156, 0.155, 0.139, -0.179, -0.179],
            "weight_method": ["hierarchy_equal"] * 5,
            "cluster_id": [
                f"{cluster}/s1",
                f"{cluster}/s2",
                f"{cluster}/s3",
                f"{cluster}/cand_es",
                f"{cluster}/cand_nq",
            ],
        }
    )
    table = build_weight_layer_member_table(
        without,
        with_portfolio,
        candidate_key="candidate",
    )
    group_budget_before = 0.032 + 0.032 + 0.030
    group_budget_after = 0.156 + 0.155 + 0.139 - 0.179 - 0.179
    existing = table[table["stream_id"] == "s1"].iloc[0]
    candidate = table[table["stream_id"] == "cand_es"].iloc[0]
    assert float(existing["weight_without"]) == pytest.approx(group_budget_before / 3)
    assert float(existing["weight_with"]) == pytest.approx(group_budget_after / 5)
    assert float(existing["weight_delta"]) < 0.0
    assert float(candidate["weight_without"]) == pytest.approx(0.0)
    assert float(candidate["weight_with"]) == pytest.approx(group_budget_after / 5)
    assert float(candidate["weight_delta"]) > 0.0


def test_correlation_heatmap_uses_short_codes_for_long_labels() -> None:
    labels = [
        "inclusion_candidate",
        "algomatic_momentum_signal_long_defaults_long",
        "filter_sma200_above_consec_momentum_lb40_cb3_buy_long",
    ]
    display, legend, use_codes = _correlation_heatmap_axis_labels(
        labels,
        candidate_key="inclusion_candidate",
    )
    assert use_codes is True
    assert display[0] == "S0*"
    assert legend[1][1].startswith("algomatic_momentum")


def test_write_correlation_heatmap_with_long_strategy_names(tmp_path: Path) -> None:
    labels = [
        "inclusion_candidate",
        "algomatic_momentum_signal_long_defaults_long",
        "buy_hold_long",
        "filter_sma200_above_consec_momentum_lb40_cb3_buy_long",
        "gc_sma_above_filter_d_period_200_long",
        "mr_indices_long",
    ]
    matrix = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    corr_csv = tmp_path / "portfolio_addition_pairwise_corr_matrix.csv"
    matrix.to_csv(corr_csv)
    payload = {
        "skipped": False,
        "context": {"candidate_key": "inclusion_candidate"},
    }
    out = write_portfolio_addition_plots(
        payload,
        bootstrap_csv=None,
        corr_matrix_csv=corr_csv,
        output_dir=tmp_path / "matplotlib",
    )
    heatmap = next(path for path in out if path.name == "portfolio_addition_correlation_heatmap.png")
    assert heatmap.exists()
    assert heatmap.stat().st_size > 10_000
