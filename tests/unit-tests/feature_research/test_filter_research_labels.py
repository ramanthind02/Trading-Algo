from __future__ import annotations

from feature_research.filter_research_labels import (
    build_filter_exploration_long_pairs,
    is_filter_exploration_modules,
    research_display_label,
)
from feature_research.in_sample.data_loader import BIAS_MODULE_COMBO_KEY, enrich_param_combo_with_module


def test_research_display_label_for_trend_entry_only() -> None:
    params = enrich_param_combo_with_module(
        {
            "filter_module": "sma_above_filter",
            "filter_params": {"period": 252},
            "signal_module": "cumulative_rsi_signal",
            "signal_params": {"lookback": 3, "avg_period": 3},
        },
        "filter_gate_entry_only",
    )
    label = research_display_label(params)
    assert label.startswith("B:")
    assert "252" in label
    assert "entry-only" in label


def test_research_display_label_for_vol_full_gate() -> None:
    params = enrich_param_combo_with_module(
        {
            "filter_module": "atr_percentile_filter",
            "filter_params": {
                "atr_period": 14,
                "lookback": 252,
                "max_rank_fraction": 0.4,
                "rank_metric": "atr_pct",
                "percentile_tail": "high",
            },
            "signal_module": "cumulative_rsi_signal",
            "signal_params": {"lookback": 3},
        },
        "filter_gate",
    )
    label = research_display_label(params)
    assert label.startswith("C:")
    assert "0.4" in label
    assert "full gate" in label


def test_filter_exploration_long_pairs_include_gate_and_signal_params() -> None:
    params = enrich_param_combo_with_module(
        {
            "filter_module": "sma_above_filter",
            "filter_params": {"period": 200},
            "signal_module": "cumulative_rsi_signal",
            "signal_params": {"lookback": 3, "avg_period": 3},
        },
        "filter_gate_entry_only",
    )
    pairs = build_filter_exploration_long_pairs("lbl", params)
    assert len(pairs) == 1
    _, flat = pairs[0]
    assert flat["filter_family"] == "trend"
    assert flat["f_period"] == 200
    assert flat["s_lookback"] == 3
    assert flat["s_avg_period"] == 3


def test_is_filter_exploration_modules() -> None:
    assert is_filter_exploration_modules({"filter_gate", "cumulative_rsi_signal"})
    assert not is_filter_exploration_modules({"cumulative_rsi_signal"})
