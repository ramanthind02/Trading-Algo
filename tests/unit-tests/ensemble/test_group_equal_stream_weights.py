"""Tests for vault group–equal global stream weights (ablation decode)."""
from __future__ import annotations

import pytest

from ensemble.portfolio_impl.global_weight_layer_adapter import build_group_equal_stream_weights


def test_two_groups_equal_split_within_group() -> None:
    m = {
        "a1": "G1",
        "a2": "G1",
        "b1": "G2",
    }
    w = build_group_equal_stream_weights(["a1", "a2", "b1"], m)
    assert w["b1"] == pytest.approx(0.5)
    assert w["a1"] == pytest.approx(0.25)
    assert w["a2"] == pytest.approx(0.25)


def test_remove_one_stream_redistributes_within_group_only() -> None:
    m = {"a1": "G1", "a2": "G1", "b1": "G2"}
    w_full = build_group_equal_stream_weights(["a1", "a2", "b1"], m)
    w_ab = build_group_equal_stream_weights(["a2", "b1"], m)
    # G1 had half the pie split between a1,a2; without a1, a2 gets full G1 share (0.5)
    assert w_ab["a2"] == pytest.approx(0.5)
    assert w_ab["b1"] == pytest.approx(0.5)
    assert w_full["a1"] == pytest.approx(0.25)


def test_unknown_streams_share_unknown_group() -> None:
    m = {"x": "G1"}
    w = build_group_equal_stream_weights(["x", "orphan"], m)
    # G1 and __unknown__: two groups → 0.5 each; x alone in G1 → 0.5; orphan → 0.5
    assert w["x"] == pytest.approx(0.5)
    assert w["orphan"] == pytest.approx(0.5)


def test_empty_returns_empty_series() -> None:
    s = build_group_equal_stream_weights([], {"a": "G"})
    assert len(s) == 0
