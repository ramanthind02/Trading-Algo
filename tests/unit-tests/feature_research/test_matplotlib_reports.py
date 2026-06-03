"""Focused tests for CSV-to-Matplotlib research visualizations."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from feature_research.research_table_exports import AGGREGATE_EQUITY_TICKER
from feature_research.visualization.matplotlib_reports import (
    generate_detected_matplotlib_plots,
)


def test_generate_detected_matplotlib_plots_writes_aggregate_equity_only(
    tmp_path: Path,
) -> None:
    """Per-ticker equity PNGs are omitted; QuantStats tearsheets cover instrument detail."""
    input_dir = tmp_path / "visualization"
    output_dir = tmp_path / "plots"
    input_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    for ticker in ("ES", "NQ", "GC"):
        for period, offset in (("126", 0.0), ("252", 0.2), ("378", 0.4)):
            rows.extend(
                [
                    {
                        "datetime": f"2020-01-0{day}",
                        "param_combo_label": f"sma_regime_signal__period_{period}",
                        "ticker": ticker,
                        "cumulative_strategy_return": float(day) + offset,
                    }
                    for day in (1, 2, 3)
                ]
            )
    for period, offset in (("126", 0.0), ("252", 0.2), ("378", 0.4)):
        rows.extend(
            [
                {
                    "datetime": f"2020-01-0{day}",
                    "param_combo_label": f"sma_regime_signal__period_{period}",
                    "ticker": AGGREGATE_EQUITY_TICKER,
                    "cumulative_strategy_return": float(day) + offset + 0.5,
                }
                for day in (1, 2, 3)
            ]
        )
    pd.DataFrame(rows).to_csv(input_dir / "equity_curve.csv", index=False)

    generated = generate_detected_matplotlib_plots(input_dir, output_dir, top_n=6)

    aggregate_path = output_dir / "equity_curve.png"
    per_ticker_paths = [
        output_dir / f"equity_curve__ticker_{ticker}.png"
        for ticker in ("ES", "NQ", "GC")
    ]
    assert aggregate_path.exists()
    assert aggregate_path in generated
    assert not any(path.exists() for path in per_ticker_paths)


def test_generate_detected_matplotlib_plots_writes_supported_outputs(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "visualization"
    output_dir = tmp_path / "plots"
    (input_dir / "validation").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "datetime": "2020-01-01",
                "param_combo_label": "a=1",
                "ticker": "ES",
                "cumulative_strategy_return": 0.1,
            },
            {
                "datetime": "2020-01-02",
                "param_combo_label": "a=1",
                "ticker": "ES",
                "cumulative_strategy_return": 0.3,
            },
            {
                "datetime": "2020-01-01",
                "param_combo_label": "a=2",
                "ticker": "ES",
                "cumulative_strategy_return": 0.05,
            },
            {
                "datetime": "2020-01-02",
                "param_combo_label": "a=2",
                "ticker": "ES",
                "cumulative_strategy_return": 0.2,
            },
        ]
    ).to_csv(input_dir / "equity_curve.csv", index=False)
    pd.DataFrame(
        [
            {
                "datetime": "2020-01-03",
                "param_combo_label": "a=1",
                "ticker": "ES",
                "cumulative_strategy_return": 0.4,
            }
        ]
    ).to_csv(input_dir / "validation" / "equity_curve_validation_only.csv", index=False)
    pd.DataFrame(
        [
            {
                "param_combo_label": "a=1",
                "observed_metric": 0.7,
                "p_value": 0.04,
                "passed": True,
            },
            {
                "param_combo_label": "a=2",
                "observed_metric": 0.2,
                "p_value": 0.3,
                "passed": False,
            },
        ]
    ).to_csv(input_dir / "permutation_vector_shuffle.csv", index=False)
    pd.DataFrame(
        [
            {
                "vault_feature_name": "vault_a",
                "metric_name": "pearson_return_corr",
                "metric_value": 0.6,
                "ticker": "ES",
            }
        ]
    ).to_csv(input_dir / "vault_correlation_long.csv", index=False)
    pd.DataFrame(
        [
            {
                "research_display_label": "A: Baseline",
                "gate_type": "A",
                "sharpe": 0.4,
                "n_nonzero_signal": 120,
            },
            {
                "research_display_label": "C: Vol gate",
                "gate_type": "C",
                "sharpe": 0.8,
                "n_nonzero_signal": 90,
            },
        ]
    ).to_csv(input_dir / "filter_exploration_summary.csv", index=False)

    generated = generate_detected_matplotlib_plots(input_dir, output_dir, top_n=5)

    expected_outputs = [
        output_dir / "equity_curve.png",
        output_dir / "validation" / "equity_curve_validation_only.png",
        output_dir / "permutation_vector_shuffle.png",
        output_dir / "vault_correlation_long.png",
        output_dir / "filter_gate_comparison.png",
    ]
    assert all(path.exists() for path in expected_outputs)
    assert len(generated) >= len(expected_outputs)

    manifest = json.loads((output_dir / "matplotlib_manifest.json").read_text())
    assert manifest["input_dir"] == str(input_dir)
    assert (output_dir / "filter_gate_comparison.png").exists()


def test_generate_detected_matplotlib_plots_skips_param_sensitivity_csvs(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "visualization"
    output_dir = tmp_path / "plots"
    input_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "param_combo_label": "only_combo",
                "feature_name": "feat",
                "sharpe": 0.4,
                "t_stat": 1.8,
                "sortino": 0.5,
            }
        ]
    ).to_csv(input_dir / "param_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "param_combo_label": "only_combo",
                "param_key": "lookback",
                "param_value": 10,
                "param_sort_order": 0,
            }
        ]
    ).to_csv(input_dir / "param_combo_long.csv", index=False)

    generated = generate_detected_matplotlib_plots(input_dir, output_dir)

    assert generated == []
    assert not (output_dir / "param_sensitivity.png").exists()
    assert not (output_dir / "param_sensitivity_explorer").exists()


def test_generate_detected_matplotlib_plots_writes_filter_exploration_equity_curve(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "visualization"
    output_dir = tmp_path / "plots"
    input_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "datetime": "2020-01-01",
                "param_combo_label": "A: Baseline",
                "ticker": "NQ",
                "cumulative_strategy_return": 0.1,
            },
            {
                "datetime": "2020-01-02",
                "param_combo_label": "A: Baseline",
                "ticker": "NQ",
                "cumulative_strategy_return": 0.2,
            },
            {
                "datetime": "2020-01-01",
                "param_combo_label": "C: Vol gate",
                "ticker": "NQ",
                "cumulative_strategy_return": 0.05,
            },
            {
                "datetime": "2020-01-02",
                "param_combo_label": "C: Vol gate",
                "ticker": "NQ",
                "cumulative_strategy_return": 0.25,
            },
        ]
    ).to_csv(input_dir / "filter_exploration_equity_curve.csv", index=False)

    generated = generate_detected_matplotlib_plots(input_dir, output_dir, top_n=4)

    output_path = output_dir / "filter_exploration_equity_curve.png"
    assert output_path.exists()
    assert output_path in generated
