from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_research.weight_layer_report import export_global_weight_layer_report


class _ReportPortfolio:
    def get_diagnostics(self) -> dict:
        return {
            "weight_layer": {
                "is_fitted": True,
                "tf_weights": {"D": 0.75, "M": 0.25},
                "fdm": 1.42,
                "mean_cross_tf_correlation": 0.18,
                "daily_grid_len": 2520,
                "diagnostics": {
                    "weight_method": "cluster_corr_ulcer",
                    "tickers": {
                        "Ticker.ES": {
                            "fdm": 1.31,
                            "n_models": 2,
                            "weights": {
                                "buy_hold_signal_M::rule_based_3__M::ensemble_0": 0.40,
                                "turnaround_signal_D::rule_based_3__D::ensemble_0": 0.60,
                            },
                            "cluster_assignments": {
                                "buy_hold_signal_M::rule_based_3__M::ensemble_0": "cluster_1",
                                "turnaround_signal_D::rule_based_3__D::ensemble_0": "cluster_2",
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
                        "Ticker.NQ": {
                            "fdm": 1.08,
                            "n_models": 1,
                            "weights": {
                                "momentum_signal_D::rule_based_3__D::ensemble_1": 1.0,
                            },
                            "cluster_assignments": {
                                "momentum_signal_D::rule_based_3__D::ensemble_1": "cluster_1",
                            },
                            "cluster_weights": {"cluster_1": 1.0},
                            "cluster_metrics": {
                                "cluster_1": {
                                    "avg_positive_corr": 0.00,
                                    "ulcer_index": 0.01,
                                    "score": 4.00,
                                    "member_count": 1,
                                }
                            },
                            "mean_cluster_correlation": 0.0,
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
    assert "Per-Ticker Weight Layers" in html_text
    assert 'href="#ticker-ES"' in html_text
    assert 'href="#ticker-NQ"' in html_text
    assert "Cluster Allocation" in html_text
    assert "Model Weights" in html_text
    assert "cluster_corr_ulcer" in html_text
    assert "buy_hold_signal_M::rule_based_3__M::ensemble_0" in html_text

    strategy_df = pd.read_csv(tmp_path / "global_strategy_weights.csv")
    assert set(strategy_df.columns) >= {
        "ticker",
        "cluster_id",
        "timeframe",
        "model_name",
        "model_weight",
        "cluster_weight",
        "avg_positive_corr",
        "ulcer_index",
        "score",
    }
    assert set(strategy_df["ticker"]) == {"ES", "NQ"}
    assert set(strategy_df["cluster_id"]) == {"cluster_1", "cluster_2"}
