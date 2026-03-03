"""Unit tests for Marginal Peak Selection (MPS)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_research.walkforward.marginal_peak_selection import (
    MarginalPeakConfig,
    _varying_dimensions,
    compute_pairwise_marginal_tables,
    run_marginal_peak_selection,
)


def _label(params: dict[str, object]) -> str:
    """Canonical param label matching runner/marginal_peak_selection."""
    return "|".join(f"{key}={params[key]}" for key in sorted(params))


# ---------------------------------------------------------------------------
# compute_pairwise_marginal_tables
# ---------------------------------------------------------------------------


def test_compute_pairwise_marginal_tables_2d_one_pair() -> None:
    """2D grid: one dimension pair, cell means correct."""
    param_grid: list[dict[str, object]] = [
        {"a": 1, "b": 10},
        {"a": 1, "b": 20},
        {"a": 2, "b": 10},
        {"a": 2, "b": 20},
    ]
    raw_objectives = {_label(p): float(i) for i, p in enumerate(param_grid)}  # 0, 1, 2, 3
    tables = compute_pairwise_marginal_tables(param_grid, raw_objectives, min_cell_size=1)
    assert list(tables.keys()) == [("a", "b")]
    df = tables[("a", "b")]
    assert len(df) == 4
    df = df.sort_values(["a", "b"]).reset_index(drop=True)
    # (1,10): one combo value 0; (1,20): 1; (2,10): 2; (2,20): 3
    assert df.loc[0, "cell_mean"] == pytest.approx(0.0)
    assert df.loc[1, "cell_mean"] == pytest.approx(1.0)
    assert df.loc[2, "cell_mean"] == pytest.approx(2.0)
    assert df.loc[3, "cell_mean"] == pytest.approx(3.0)
    assert list(df.columns) == ["a", "b", "cell_mean", "cell_size"]


def test_compute_pairwise_marginal_tables_3d_three_tables() -> None:
    """3D grid: three pairs; one table's cell means match hand-written expectations."""
    # lookback=2,3,4 x avg_period=2,3 x n_bins=8,9,10 -> 18 combos
    param_grid = [
        {"lookback": lb, "avg_period": ap, "n_bins": nb}
        for lb in (2, 3, 4)
        for ap in (2, 3)
        for nb in (8, 9, 10)
    ]
    # Set raw so (lookback=2, avg_period=2) cell has mean 3.0; others lower
    raw_objectives: dict[str, float] = {}
    for p in param_grid:
        lbl = _label(p)
        if p["lookback"] == 2 and p["avg_period"] == 2:
            raw_objectives[lbl] = 3.0
        else:
            raw_objectives[lbl] = 2.0
    tables = compute_pairwise_marginal_tables(param_grid, raw_objectives, min_cell_size=1)
    assert len(tables) == 3
    # Dimension keys are sorted: ("avg_period", "lookback"), ("avg_period", "n_bins"), ("lookback", "n_bins")
    assert ("avg_period", "lookback") in tables
    df_la = tables[("avg_period", "lookback")]
    row_22 = df_la[(df_la["lookback"] == 2) & (df_la["avg_period"] == 2)]
    assert len(row_22) == 1
    assert row_22["cell_mean"].iloc[0] == pytest.approx(3.0)
    row_23 = df_la[(df_la["lookback"] == 2) & (df_la["avg_period"] == 3)]
    assert len(row_23) == 1
    assert row_23["cell_mean"].iloc[0] == pytest.approx(2.0)


def test_compute_pairwise_marginal_tables_min_cell_size_excludes_small_cells() -> None:
    """Cells with fewer than min_cell_size combos are excluded."""
    param_grid = [
        {"x": 1, "y": 1},
        {"x": 1, "y": 1},  # (1,1) has 2
        {"x": 1, "y": 2},
        {"x": 2, "y": 1},  # (2,1) has 1
        {"x": 2, "y": 2},
    ]
    raw_objectives = {_label(p): 1.0 for p in param_grid}
    tables = compute_pairwise_marginal_tables(param_grid, raw_objectives, min_cell_size=2)
    df = tables[("x", "y")]
    assert len(df) == 1
    assert df.iloc[0]["x"] == 1 and df.iloc[0]["y"] == 1
    assert df.iloc[0]["cell_size"] == 2


def test_compute_pairwise_marginal_tables_empty_grid() -> None:
    """Empty param_grid returns empty dict."""
    assert compute_pairwise_marginal_tables([], {}, min_cell_size=2) == {}


def test_compute_pairwise_marginal_tables_single_dimension() -> None:
    """Single dimension: no pairs, returns empty dict."""
    param_grid = [{"a": 1}, {"a": 2}]
    raw_objectives = {_label(p): 1.0 for p in param_grid}
    assert compute_pairwise_marginal_tables(param_grid, raw_objectives) == {}


def test_compute_pairwise_marginal_tables_marginal_dim_3() -> None:
    """3D grid with marginal_dim=3: one table C(3,3)=1, columns and cell means correct."""
    param_grid = [
        {"a": 1, "b": 10, "c": 100},
        {"a": 1, "b": 10, "c": 101},
        {"a": 1, "b": 20, "c": 100},
        {"a": 2, "b": 10, "c": 100},
    ]
    raw_objectives = {_label(p): float(i) for i, p in enumerate(param_grid)}
    tables = compute_pairwise_marginal_tables(
        param_grid, raw_objectives, min_cell_size=1, marginal_dim=3
    )
    assert len(tables) == 1
    dims = ("a", "b", "c")
    assert dims in tables
    df = tables[dims].sort_values(list(dims)).reset_index(drop=True)
    assert list(df.columns) == ["a", "b", "c", "cell_mean", "cell_size"]
    assert len(df) == 4
    assert df.loc[0, "cell_mean"] == pytest.approx(0.0)
    assert df.loc[1, "cell_mean"] == pytest.approx(1.0)
    assert df.loc[2, "cell_mean"] == pytest.approx(2.0)
    assert df.loc[3, "cell_mean"] == pytest.approx(3.0)


def test_compute_pairwise_marginal_tables_marginal_dim_3_insufficient_dims() -> None:
    """2D grid with marginal_dim=3 returns empty dict."""
    param_grid = [
        {"a": 1, "b": 10},
        {"a": 2, "b": 20},
    ]
    raw_objectives = {_label(p): 1.0 for p in param_grid}
    tables = compute_pairwise_marginal_tables(
        param_grid, raw_objectives, min_cell_size=1, marginal_dim=3
    )
    assert tables == {}


# ---------------------------------------------------------------------------
# run_marginal_peak_selection
# ---------------------------------------------------------------------------


def test_run_marginal_peak_selection_dominant_regime() -> None:
    """One (dim_A, dim_B) cell has clearly highest mean; selection from that cell only."""
    param_grid = [
        {"lb": 2, "ap": 2, "nb": 8},
        {"lb": 2, "ap": 2, "nb": 9},
        {"lb": 2, "ap": 3, "nb": 8},
        {"lb": 3, "ap": 2, "nb": 8},
        {"lb": 3, "ap": 3, "nb": 8},
    ]
    raw_objectives = {
        _label({"lb": 2, "ap": 2, "nb": 8}): 3.0,
        _label({"lb": 2, "ap": 2, "nb": 9}): 3.1,
        _label({"lb": 2, "ap": 3, "nb": 8}): 2.0,
        _label({"lb": 3, "ap": 2, "nb": 8}): 2.0,
        _label({"lb": 3, "ap": 3, "nb": 8}): 2.0,
    }
    config = MarginalPeakConfig(k_max=5, min_gap=0.10, min_cell_size=1, fallback_k=2)
    result = run_marginal_peak_selection(raw_objectives, param_grid, config)
    assert result.fallback_used is False
    assert result.selected_pair == ("ap", "lb") or result.selected_pair == ("lb", "ap")
    assert result.peak_cell is not None
    assert result.gap >= config.min_gap
    # (lb=2, ap=2) has mean (3.0+3.1)/2 = 3.05; others 2.0 -> gap 1.05
    assert len(result.selected_labels) <= config.k_max
    dim_a, dim_b = result.selected_pair[0], result.selected_pair[1]
    peak_a, peak_b = result.peak_cell[0], result.peak_cell[1]
    label_to_params = dict(zip([_label(p) for p in param_grid], param_grid))
    for lbl in result.selected_labels:
        params = label_to_params[lbl]
        assert params[dim_a] == peak_a and params[dim_b] == peak_b


def test_run_marginal_peak_selection_fallback_when_gap_small() -> None:
    """All tables have gap < min_gap -> fallback to top fallback_k by raw."""
    param_grid = [
        {"a": 1, "b": 1},
        {"a": 1, "b": 2},
        {"a": 2, "b": 1},
        {"a": 2, "b": 2},
    ]
    # All cells similar mean -> small gaps
    raw_objectives = {
        _label({"a": 1, "b": 1}): 2.0,
        _label({"a": 1, "b": 2}): 2.05,
        _label({"a": 2, "b": 1}): 2.02,
        _label({"a": 2, "b": 2}): 2.01,
    }
    config = MarginalPeakConfig(k_max=2, min_gap=0.50, min_cell_size=1, fallback_k=2)
    result = run_marginal_peak_selection(raw_objectives, param_grid, config)
    assert result.fallback_used is True
    assert result.selected_pair is None
    assert result.peak_cell is None
    assert result.gap == 0.0
    assert len(result.selected_labels) == config.fallback_k
    # Top 2 by raw: 2.05, 2.02
    assert _label({"a": 1, "b": 2}) in result.selected_labels
    assert _label({"a": 2, "b": 1}) in result.selected_labels


def test_run_marginal_peak_selection_peak_cell_restriction_3d() -> None:
    """3D grid: selected labels all have (dim_A, dim_B) equal to peak cell."""
    param_grid = [
        {"lb": 2, "ap": 2, "nb": 8},
        {"lb": 2, "ap": 2, "nb": 9},
        {"lb": 2, "ap": 2, "nb": 10},
        {"lb": 2, "ap": 3, "nb": 8},
        {"lb": 3, "ap": 2, "nb": 8},
    ]
    raw_objectives = {
        _label(p): (3.5 if p["lb"] == 2 and p["ap"] == 2 else 2.0) for p in param_grid
    }
    config = MarginalPeakConfig(k_max=5, min_gap=0.10, min_cell_size=1, fallback_k=5)
    result = run_marginal_peak_selection(raw_objectives, param_grid, config)
    assert result.fallback_used is False
    assert result.selected_pair is not None and result.peak_cell is not None
    dim_a, dim_b = result.selected_pair[0], result.selected_pair[1]
    peak_a, peak_b = result.peak_cell[0], result.peak_cell[1]
    label_to_params = dict(zip([_label(p) for p in param_grid], param_grid))
    for lbl in result.selected_labels:
        params = label_to_params[lbl]
        assert params[dim_a] == peak_a
        assert params[dim_b] == peak_b


def test_run_marginal_peak_selection_marginal_dim_3() -> None:
    """With marginal_dim=3, selected_pair and peak_cell are 3-tuples; selection respects 3D peak cell."""
    param_grid = [
        {"lb": 2, "ap": 2, "nb": 8},
        {"lb": 2, "ap": 2, "nb": 9},
        {"lb": 2, "ap": 2, "nb": 10},
        {"lb": 2, "ap": 3, "nb": 8},
        {"lb": 3, "ap": 2, "nb": 8},
    ]
    # (2,2,8) cell has mean 4.0; (2,2,9),(2,2,10) have 3.0; others 2.0 -> clear gap 1.0, peak (2,2,8)
    raw_objectives = {
        _label({"lb": 2, "ap": 2, "nb": 8}): 4.0,
        _label({"lb": 2, "ap": 2, "nb": 9}): 3.0,
        _label({"lb": 2, "ap": 2, "nb": 10}): 3.0,
        _label({"lb": 2, "ap": 3, "nb": 8}): 2.0,
        _label({"lb": 3, "ap": 2, "nb": 8}): 2.0,
    }
    config = MarginalPeakConfig(
        k_max=5, min_gap=0.10, min_cell_size=1, fallback_k=5, marginal_dim=3
    )
    result = run_marginal_peak_selection(raw_objectives, param_grid, config)
    assert result.fallback_used is False
    assert result.selected_pair is not None and result.peak_cell is not None
    assert len(result.selected_pair) == 3
    assert len(result.peak_cell) == 3
    dims = result.selected_pair
    peak_vals = result.peak_cell
    label_to_params = dict(zip([_label(p) for p in param_grid], param_grid))
    for lbl in result.selected_labels:
        params = label_to_params[lbl]
        for i, d in enumerate(dims):
            assert params[d] == peak_vals[i]


def test_run_marginal_peak_selection_per_param_detail_columns_and_consistency() -> None:
    """per_param_detail has required columns; in_peak_cell and selected consistent with result."""
    param_grid = [
        {"x": 1, "y": 1},
        {"x": 1, "y": 2},
        {"x": 2, "y": 1},
        {"x": 2, "y": 2},
    ]
    raw_objectives = {
        _label({"x": 1, "y": 1}): 5.0,
        _label({"x": 1, "y": 2}): 1.0,
        _label({"x": 2, "y": 1}): 1.0,
        _label({"x": 2, "y": 2}): 1.0,
    }
    config = MarginalPeakConfig(k_max=2, min_gap=0.10, min_cell_size=1, fallback_k=2)
    result = run_marginal_peak_selection(raw_objectives, param_grid, config)
    df = result.per_param_detail
    assert set(df.columns) == {
        "param_label",
        "raw_objective",
        "cell_id",
        "cell_mean",
        "in_peak_cell",
        "selected",
    }
    assert set(df["param_label"]) == set(_label(p) for p in param_grid)
    assert df["selected"].sum() == len(result.selected_labels)
    assert set(df[df["selected"]]["param_label"]) == set(result.selected_labels)
    if not result.fallback_used:
        assert df["in_peak_cell"].sum() >= 1
        assert all(
            df.loc[df["param_label"] == lbl, "in_peak_cell"].iloc[0]
            for lbl in result.selected_labels
        )


# ---------------------------------------------------------------------------
# varying dimensions and dimension_ranges
# ---------------------------------------------------------------------------


def test_varying_dimensions_excludes_constant_selected_bin() -> None:
    """When selected_bin is constant (e.g. 0), marginal tables use only varying dims."""
    # Grid: lookback x avg_period x selected_bin=0 always
    param_grid = [
        {"lookback": lb, "avg_period": ap, "selected_bin": 0}
        for lb in (2, 3, 4)
        for ap in (2, 3)
    ]
    raw_objectives = {_label(p): float(i) for i, p in enumerate(param_grid)}
    tables = compute_pairwise_marginal_tables(param_grid, raw_objectives, min_cell_size=1)
    # Only varying dims are lookback and avg_period; selected_bin has one value
    assert "selected_bin" not in _varying_dimensions(param_grid)
    assert all("selected_bin" not in pair for pair in tables.keys())
    assert len(tables) == 1
    assert set(tables.keys()) == {("avg_period", "lookback")}
    # Run full MPS: selected_pair should not include selected_bin
    config = MarginalPeakConfig(k_max=5, min_gap=0.01, min_cell_size=1, fallback_k=5)
    result = run_marginal_peak_selection(raw_objectives, param_grid, config)
    if result.selected_pair is not None:
        assert "selected_bin" not in result.selected_pair


def test_dimension_ranges_restricts_combos_used_for_means() -> None:
    """With dimension_ranges selected_bin=(5,8), only combos in [5,8] are used for marginals/means."""
    # Full grid: lookback 2,3 x selected_bin 0..8
    param_grid = [
        {"lookback": lb, "selected_bin": sb}
        for lb in (2, 3)
        for sb in range(9)
    ]
    # Give high raw to selected_bin in [5,8] so peak is in range; low to 0..4
    raw_objectives: dict[str, float] = {}
    for p in param_grid:
        lbl = _label(p)
        sb = int(p["selected_bin"])
        raw_objectives[lbl] = 10.0 if 5 <= sb <= 8 else 1.0
    config = MarginalPeakConfig(
        k_max=10,
        min_gap=0.10,
        min_cell_size=1,
        fallback_k=5,
        dimension_ranges={"selected_bin": (5, 8)},
    )
    result = run_marginal_peak_selection(raw_objectives, param_grid, config)
    # All selected labels must have selected_bin in [5, 8]
    label_to_params = dict(zip([_label(p) for p in param_grid], param_grid))
    for lbl in result.selected_labels:
        params = label_to_params[lbl]
        assert 5 <= int(params["selected_bin"]) <= 8
    # per_param_detail should only include rows for combos in [5,8] (filtered grid)
    filtered_count = sum(1 for p in param_grid if 5 <= int(p["selected_bin"]) <= 8)
    assert len(result.per_param_detail) == filtered_count
    assert all(
        5 <= int(label_to_params[lbl]["selected_bin"]) <= 8
        for lbl in result.per_param_detail["param_label"]
    )
