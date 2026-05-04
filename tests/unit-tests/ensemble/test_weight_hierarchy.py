"""Unit tests for manual hierarchy parsing and equal-split weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ensemble.portfolio_impl.global_weight_layer_adapter import build_global_stream_id
from ensemble.weight_hierarchy import (
    compute_equal_split_weights,
    load_hierarchy_from_path,
    parse_hierarchy_spec,
    validate_strict_stream_coverage,
)


def test_parse_hierarchy_spec_nested_groups() -> None:
    raw = {
        "type": "group",
        "id": "root",
        "children": [
            {
                "type": "group",
                "id": "g1",
                "children": [
                    {"type": "leaf", "stream_id": "a"},
                    {"type": "leaf", "stream_id": "b"},
                ],
            },
            {"type": "leaf", "stream_id": "c"},
        ],
    }
    root = parse_hierarchy_spec(raw)
    assert root.group_id == "root"
    assert len(root.children) == 2


def test_leaf_matcher_resolves_to_global_stream_id() -> None:
    raw = {
        "type": "group",
        "id": "root",
        "children": [
            {
                "type": "leaf",
                "ticker": "ES",
                "timeframe": "D",
                "model_name": "m1::x",
            },
        ],
    }
    root = parse_hierarchy_spec(raw)
    leaf = root.children[0]
    assert getattr(leaf, "stream_id", "") == build_global_stream_id("ES", "D", "m1::x")
    expected = build_global_stream_id("ES", "D", "m1::x")
    assert leaf.stream_id == expected


def test_equal_split_weights_three_leaves() -> None:
    raw = {
        "type": "group",
        "id": "root",
        "children": [
            {
                "type": "group",
                "id": "pair",
                "children": [
                    {"type": "leaf", "stream_id": "x"},
                    {"type": "leaf", "stream_id": "y"},
                ],
            },
            {"type": "leaf", "stream_id": "z"},
        ],
    }
    root = parse_hierarchy_spec(raw)
    series, assignments, cluster_weights, metrics = compute_equal_split_weights(
        root, ["x", "y", "z"]
    )
    assert series["x"] == pytest.approx(0.25)
    assert series["y"] == pytest.approx(0.25)
    assert series["z"] == pytest.approx(0.5)
    assert "root/pair" in cluster_weights
    assert metrics["root"]["member_count"] == 3


def test_validate_strict_stream_coverage_mismatch() -> None:
    with pytest.raises(ValueError, match="not in hierarchy"):
        validate_strict_stream_coverage(frozenset({"a"}), frozenset({"a", "b"}))


def test_load_hierarchy_from_path(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    spec = {
        "type": "group",
        "id": "r",
        "children": [
            {"type": "leaf", "stream_id": "u"},
            {"type": "leaf", "stream_id": "v"},
        ],
    }
    p.write_text(json.dumps(spec), encoding="utf-8")
    root = load_hierarchy_from_path(p)
    assert root.group_id == "r"
