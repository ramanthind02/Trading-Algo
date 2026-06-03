"""Tests for portfolio_research.weight_layer_report."""

from __future__ import annotations

import pandas as pd

from portfolio_research.config import load_config
from portfolio_research.weight_layer_export import weight_layer_diagnostics_to_dataframe
from portfolio_research.weight_layer_report import (
    build_portfolio_weight_layer_report,
    build_weight_layer_member_records,
)


def _sample_diagnostics_df() -> pd.DataFrame:
    diagnostics = {
        "is_fitted": True,
        "weight_method": "hierarchy_equal",
        "fdm_max": 2.0,
        "tickers": {
            "__GLOBAL__": {
                "fdm": 2.0,
                "weights": {
                    "ES::M::buy_hold_signal_M::rule_based_3__M::ensemble_0": 0.03,
                    "GC::M::buy_hold_signal_M::rule_based_3__M::ensemble_0": 0.07,
                    "ES::D::calendar_ensemble_signal_D::signed_signal_x__D::ensemble_7": 0.12,
                },
                "mean_signal_correlation": 0.1,
                "mean_cluster_correlation": 0.1,
                "cluster_assignments": {
                    "ES::M::buy_hold_signal_M::rule_based_3__M::ensemble_0": (
                        "root/equity_indices/buy_hold/"
                        "ES::M::buy_hold_signal_M::rule_based_3__M::ensemble_0"
                    ),
                    "GC::M::buy_hold_signal_M::rule_based_3__M::ensemble_0": (
                        "root/commodities/buy_hold/"
                        "GC::M::buy_hold_signal_M::rule_based_3__M::ensemble_0"
                    ),
                    "ES::D::calendar_ensemble_signal_D::signed_signal_x__D::ensemble_7": (
                        "root/equity_indices/seasonal/"
                        "ES::D::calendar_ensemble_signal_D::signed_signal_x__D::ensemble_7"
                    ),
                },
            }
        },
    }
    return weight_layer_diagnostics_to_dataframe(
        diagnostics,
        phase="Validation",
        fit_start=pd.Timestamp("2000-01-01"),
        fit_end=pd.Timestamp("2018-12-31"),
        predict_start=pd.Timestamp("2019-01-01"),
        predict_end=pd.Timestamp("2022-12-31"),
    )


def test_build_weight_layer_member_records_includes_buy_hold_streams() -> None:
    members = build_weight_layer_member_records(_sample_diagnostics_df())
    buy_hold = [row for row in members if "buy_hold" in str(row["stream_id"])]
    assert len(buy_hold) == 2
    assert all(float(row["weight_with"]) > 0.0 for row in buy_hold)


def test_build_portfolio_weight_layer_report_has_ui_context() -> None:
    config = load_config()
    report = build_portfolio_weight_layer_report(
        _sample_diagnostics_df(),
        config=config,
        phase="Validation",
        fit_start=pd.Timestamp("2000-01-01"),
        fit_end=pd.Timestamp("2018-12-31"),
        predict_start=pd.Timestamp("2019-01-01"),
        predict_end=pd.Timestamp("2022-12-31"),
    )
    context = report["context"]
    assert context["weight_layer_method"] == "hierarchy_equal"
    assert "SR tilt L1–L2" in str(context["weight_layer_policy"])
    detail = context["weight_layer_detail"]
    assert detail["snapshot_mode"] is True
    assert detail["sr_tilt_max_depth"] == 2
    assert detail["within_group_method"] == "inverse_avg_pairwise_corr"
    assert "buy_hold" in {node["id"] for asset in detail["tree"] for node in asset["children"]}
