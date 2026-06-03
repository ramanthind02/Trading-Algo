"""Unit tests for pivot explorer display labels."""
from __future__ import annotations

from feature_research.ui.pivot_labels import (
    metric_display_label,
    param_field_display_label,
    param_field_group,
)


def test_param_field_display_label_maps_prefixes_and_gate_fields() -> None:
    assert param_field_display_label("s_entry_lookback") == "Signal · Entry Lookback"
    assert param_field_display_label("f_atr_period") == "Filter · Atr Period"
    assert param_field_display_label("filter_family") == "Filter family"
    assert param_field_display_label("entry_lookback") == "Entry Lookback"


def test_param_field_group_assigns_ui_buckets() -> None:
    assert param_field_group("filter_family") == "Gate & setup"
    assert param_field_group("s_entry_lookback") == "Signal parameters"
    assert param_field_group("f_atr_period") == "Filter parameters"
    assert param_field_group("entry_lookback") == "Strategy parameters"


def test_metric_display_label() -> None:
    assert metric_display_label("t_stat") == "T-stat"
    assert metric_display_label("custom_metric") == "Custom Metric"
