"""Tests for walkforward selected-params JSON codec."""
from __future__ import annotations

from feature_research.core_helpers import combo_key
from utils.core.enums import PositionMode
from utils.evaluation.walkforward.selected_params_codec import (
    decode_selected_params_list,
    serialize_selected_params,
)
from utils.evaluation.walkforward.walkforward_labels import canonical_param_label


def test_serialize_decode_round_trips_nested_filter_gate_params() -> None:
    params: dict[str, object] = {
        "filter_module": "adx_filter",
        "filter_params": {"length": 20, "threshold": 20.0, "compare": "below"},
        "signal_module": "cyclical_rsi",
        "signal_params": {"short_period": 4, "long_period": 120, "rsi_period": 2},
    }
    payload = serialize_selected_params(params)
    decoded = decode_selected_params_list(payload)
    assert len(decoded) == 1
    assert combo_key(decoded[0]) == combo_key(params)


def test_decode_null_and_invalid_yield_empty_list() -> None:
    assert decode_selected_params_list("null") == []
    assert decode_selected_params_list("not-json") == []


def test_serialize_decode_round_trips_position_mode_enum() -> None:
    """Walkforward JSON must round-trip enums as stable values (same combo_key as live params)."""
    params: dict[str, object] = {"mode": PositionMode.LONG_SHORT, "max_positions": 3}
    payload = serialize_selected_params(params)
    decoded = decode_selected_params_list(payload)
    assert len(decoded) == 1
    assert combo_key(decoded[0]) == combo_key(params)


def test_canonical_label_params_match_serialized_row_dict() -> None:
    """Winner row dict keys match expanded grid (combo_key alignment)."""
    params: dict[str, object] = {
        "filter_module": "adx_filter",
        "filter_params": {"length": 20, "compare": "below"},
        "signal_module": "cyclical_rsi",
        "signal_params": {"short_period": 4},
    }
    assert canonical_param_label(params)  # smoke: label is non-empty
    assert combo_key(decode_selected_params_list(serialize_selected_params(params))[0]) == combo_key(
        params
    )
