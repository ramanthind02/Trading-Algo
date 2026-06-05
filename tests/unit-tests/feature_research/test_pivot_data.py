"""Tests for parameter-sensitivity pivot payload helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.feature.ui.pivot_data import (
    build_parameter_sensitivity_pivot_payload,
    pivot_payload_available,
)


def _write_pivot_fixture(input_dir: Path) -> None:
    combos = [
        ("combo_a", 5, 0.1, 10, 1.6, 0.4),
        ("combo_b", 5, 0.1, 20, 2.2, 0.7),
        ("combo_c", 5, 0.2, 10, 1.9, 0.5),
        ("combo_d", 10, 0.2, 20, 3.6, 1.1),
    ]
    pd.DataFrame(
        [
            {
                "param_combo_label": label,
                "feature_name": "feat",
                "n_observations": 100,
                "n_nonzero_signal": 50,
                "sharpe": sharpe,
                "t_stat": t_stat,
                "sortino": sharpe + 0.2,
            }
            for label, _lookback, _threshold, _vol_window, t_stat, sharpe in combos
        ]
    ).to_csv(input_dir / "param_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "param_combo_label": label,
                "ticker": ticker,
                "feature_name": "feat",
                "n_observations": 100,
                "n_nonzero_signal": 50,
                "sharpe": sharpe - adjustment,
                "t_stat": t_stat - adjustment,
                "sortino": sharpe + 0.1 - adjustment,
            }
            for label, _lookback, _threshold, _vol_window, t_stat, sharpe in combos
            for ticker, adjustment in (("ES", 0.0), ("NQ", 0.3))
        ]
    ).to_csv(input_dir / "param_sensitivity_by_ticker.csv", index=False)
    pd.DataFrame(
        [
            {
                "param_combo_label": label,
                "param_key": param_key,
                "param_value": param_value,
                "param_sort_order": sort_order,
            }
            for label, lookback, threshold, vol_window, _t_stat, _sharpe in combos
            for param_key, param_value, sort_order in (
                ("lookback", lookback, 0),
                ("threshold", threshold, 1),
                ("vol_window", vol_window, 2),
            )
        ]
    ).to_csv(input_dir / "param_combo_long.csv", index=False)


def test_pivot_payload_available_requires_core_csvs(tmp_path: Path) -> None:
    input_dir = tmp_path / "visualization"
    input_dir.mkdir(parents=True)

    assert pivot_payload_available(input_dir) is False

    _write_pivot_fixture(input_dir)
    assert pivot_payload_available(input_dir) is True


def test_build_parameter_sensitivity_pivot_payload_shapes_records(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "visualization"
    input_dir.mkdir(parents=True)
    _write_pivot_fixture(input_dir)

    payload = build_parameter_sensitivity_pivot_payload(input_dir)

    assert payload is not None
    assert payload["default_metric"] == "sharpe"
    assert "t_stat" in payload["metrics"]
    assert payload["metric_labels"]["sharpe"] == "Sharpe"
    assert payload["field_labels"]["lookback"] == "Lookback"
    assert payload["field_labels"]["ticker"] == "Ticker"
    assert len(payload["datasets"]) == 2

    pooled = next(dataset for dataset in payload["datasets"] if dataset["id"] == "pooled")
    assert pooled["param_fields"] == ["lookback", "threshold", "vol_window"]
    assert pooled["filter_fields"] == []
    assert len(pooled["records"]) == 4
    assert pooled["records"][0]["lookback"] == "5"
    assert isinstance(pooled["records"][0]["t_stat"], float)

    by_ticker = next(dataset for dataset in payload["datasets"] if dataset["id"] == "by_ticker")
    assert by_ticker["filter_fields"] == ["ticker"]
    assert by_ticker["value_options"]["ticker"] == ["ES", "NQ"]
    assert len(by_ticker["records"]) == 8


def test_build_parameter_sensitivity_pivot_payload_json_safe_filter_summary(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "visualization"
    input_dir.mkdir(parents=True)
    _write_pivot_fixture(input_dir)
    pd.DataFrame(
        [
            {
                "param_combo_label": "filter_a",
                "research_display_label": "Trend · full gate",
                "filter_family": "trend",
                "gate_mode": "full_gate",
                "vol_max_rank": float("nan"),
                "gate_type": "C",
                "filter_detail": "SMA(252) above",
                "sharpe": 0.5,
                "t_stat": 1.2,
                "trade_reduction_pct": float("nan"),
            },
            {
                "param_combo_label": "filter_b",
                "research_display_label": "Vol · vol gate",
                "filter_family": "vol",
                "gate_mode": "vol_gate",
                "vol_max_rank": 0.33,
                "gate_type": "B",
                "filter_detail": "ATR pct",
                "sharpe": 0.8,
                "t_stat": 1.5,
                "trade_reduction_pct": 12.5,
            },
        ]
    ).to_csv(input_dir / "filter_exploration_summary.csv", index=False)

    payload = build_parameter_sensitivity_pivot_payload(input_dir)
    assert payload is not None
    filter_dataset = next(
        dataset for dataset in payload["datasets"] if dataset["id"] == "filter_comparison"
    )
    assert filter_dataset["records"][0]["vol_max_rank"] is None

    import json

    from flask import Flask

    app = Flask(__name__)
    with app.app_context():
        encoded = app.json.dumps(payload)
    json.loads(encoded)


def test_build_filter_exploration_dataset_includes_signal_params_from_metadata(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "visualization"
    input_dir.mkdir(parents=True)
    _write_pivot_fixture(input_dir)
    pd.DataFrame(
        [
            {
                "param_combo_label": "filter_gate__combo_abc",
                "research_display_label": "C: SMA(200) · full gate",
                "gate_type": "C",
                "filter_family": "trend",
                "gate_mode": "full_gate",
                "vol_max_rank": "n/a",
                "filter_detail": "SMA(200) price above",
                "sharpe": 0.5,
                "t_stat": 1.2,
            },
            {
                "param_combo_label": "filter_gate__combo_def",
                "research_display_label": "C: SMA(200) · full gate",
                "gate_type": "C",
                "filter_family": "trend",
                "gate_mode": "full_gate",
                "vol_max_rank": "n/a",
                "filter_detail": "SMA(200) price above",
                "sharpe": 0.8,
                "t_stat": 1.5,
            },
        ]
    ).to_csv(input_dir / "filter_exploration_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "param_combo_label": "filter_gate__combo_abc",
                "param_key": "gate_mode",
                "param_value": "full_gate",
                "param_sort_order": 0,
            },
            {
                "param_combo_label": "filter_gate__combo_def",
                "param_key": "gate_mode",
                "param_value": "full_gate",
                "param_sort_order": 0,
            },
        ]
    ).to_csv(input_dir / "filter_exploration_long.csv", index=False)

    combo_root = (
        tmp_path
        / "signed_signal"
        / "rsi_signal_filter_gate"
        / "filter_gate__combo_abc"
        / "feature"
        / "run"
    )
    combo_root.mkdir(parents=True)
    combo_root.joinpath("metadata.json").write_text(
        json.dumps(
            {
                "param_combo": {
                    "filter_module": "sma_above_filter",
                    "filter_params": {"period": 200},
                    "signal_module": "rsi_signal",
                    "signal_params": {"exit_bars": 5, "rsi_period": 2},
                }
            }
        ),
        encoding="utf-8",
    )
    other_combo = combo_root.parent.parent / "filter_gate__combo_def" / "feature" / "run"
    other_combo.mkdir(parents=True)
    other_combo.joinpath("metadata.json").write_text(
        json.dumps(
            {
                "param_combo": {
                    "filter_module": "sma_above_filter",
                    "filter_params": {"period": 200},
                    "signal_module": "rsi_signal",
                    "signal_params": {"exit_bars": 10, "rsi_period": 3},
                }
            }
        ),
        encoding="utf-8",
    )

    from research.feature.ui.pivot_data import _build_filter_exploration_dataset

    dataset = _build_filter_exploration_dataset(input_dir, ["sharpe", "t_stat"])
    assert dataset is not None
    assert "s_exit_bars" in dataset["param_fields"]
    assert "s_rsi_period" in dataset["param_fields"]


def test_build_parameter_sensitivity_pivot_payload_returns_none_without_long_table(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "visualization"
    input_dir.mkdir(parents=True)
    pd.DataFrame({"param_combo_label": ["a"], "t_stat": [1.0]}).to_csv(
        input_dir / "param_sensitivity.csv",
        index=False,
    )

    assert build_parameter_sensitivity_pivot_payload(input_dir) is None
