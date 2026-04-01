from __future__ import annotations

from pathlib import Path

import pandas as pd

from portfolio_research.weight_layer_report import export_global_weight_layer_report


class _ReportPortfolio:
    def get_diagnostics(self) -> dict:
        return {
            "weight_layer": {
                "is_fitted": True,
                "tf_weights": {"D": 0.6, "M": 0.4},
                "fdm": 1.42,
                "daily_grid_len": 2520,
                "adapter_diagnostics": {
                    "synthetic_ticker": "__GLOBAL__",
                    "ticker_rollups": {"ES": 0.7, "NQ": 0.3},
                    "timeframe_rollups": {"D": 0.6, "M": 0.4},
                    "stream_decode_map": {
                        "ES::M::buy_hold_signal_M::rule_based_3__M::ensemble_0": {
                            "ticker": "ES",
                            "timeframe": "M",
                            "original_model_name": "buy_hold_signal_M::rule_based_3__M::ensemble_0",
                        },
                        "ES::D::turnaround_signal_D::rule_based_3__D::ensemble_0": {
                            "ticker": "ES",
                            "timeframe": "D",
                            "original_model_name": "turnaround_signal_D::rule_based_3__D::ensemble_0",
                        },
                        "NQ::D::momentum_signal_D::rule_based_3__D::ensemble_1": {
                            "ticker": "NQ",
                            "timeframe": "D",
                            "original_model_name": "momentum_signal_D::rule_based_3__D::ensemble_1",
                        },
                    },
                },
                "diagnostics": {
                    "weight_method": "hrp_classic",
                    "tickers": {
                        "__GLOBAL__": {
                            "fdm": 1.31,
                            "n_models": 3,
                            "weights": {
                                "ES::M::buy_hold_signal_M::rule_based_3__M::ensemble_0": 0.40,
                                "ES::D::turnaround_signal_D::rule_based_3__D::ensemble_0": 0.30,
                                "NQ::D::momentum_signal_D::rule_based_3__D::ensemble_1": 0.30,
                            },
                            "cluster_assignments": {
                                "ES::M::buy_hold_signal_M::rule_based_3__M::ensemble_0": "cluster_1",
                                "ES::D::turnaround_signal_D::rule_based_3__D::ensemble_0": "cluster_2",
                                "NQ::D::momentum_signal_D::rule_based_3__D::ensemble_1": "cluster_2",
                            },
                            "cluster_weights": {"cluster_1": 0.40, "cluster_2": 0.60},
                            "cluster_metrics": {
                                "cluster_1": {
                                    "avg_positive_corr": 0.12,
                                    "ulcer_index": 0.08,
                                    "score": 1.91,
                                    "member_count": 1,
                                },
                                "cluster_2": {
                                    "avg_positive_corr": 0.05,
                                    "ulcer_index": 0.03,
                                    "score": 3.17,
                                    "member_count": 1,
                                },
                            },
                            "mean_cluster_correlation": 0.09,
                        },
                    },
                },
            }
        }


def test_export_global_weight_layer_report_includes_per_ticker_sections(tmp_path: Path) -> None:
    export_global_weight_layer_report(
        portfolio=_ReportPortfolio(),
        phase_name="train",
        output_dir=tmp_path,
    )

    html_text = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Ticker Rollups" in html_text
    assert 'href="#ticker-ES"' in html_text
    assert 'href="#ticker-NQ"' in html_text
    assert "Allocation Groups" in html_text
    assert "Model Weights" in html_text
    assert "hrp_classic" in html_text
    assert "buy_hold_signal_M::rule_based_3__M::ensemble_0" in html_text

    strategy_df = pd.read_csv(tmp_path / "global_strategy_weights.csv")
    assert set(strategy_df.columns) >= {
        "ticker",
        "cluster_id",
        "timeframe",
        "model_name",
        "stream_id",
        "model_weight",
        "cluster_weight",
        "avg_positive_corr",
        "ulcer_index",
        "score",
    }
    assert set(strategy_df["ticker"]) == {"ES", "NQ"}
    assert set(strategy_df["cluster_id"]) == {"cluster_1", "cluster_2"}
