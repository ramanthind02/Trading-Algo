"""
Tests for T010 — StableRegion dataclass and identify_stable_regions().

Covers 1D contiguous regions, 2D plateau detection (BFS), boundary detection,
multiple disjoint regions, and minimum size enforcement.
"""
import sys
import os
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from eda.parameter_analysis import (
    StableRegion,
    identify_stable_regions,
    compute_neighbor_smoothing,
)


def _make_1d_smoothed_df(param_values, raw_values, smoothed_values, stability_ratios):
    """Helper to build a pre-smoothed 1D DataFrame."""
    return pd.DataFrame({
        "param1_value": param_values,
        "sortino": raw_values,
        "smoothed_sortino": smoothed_values,
        "stability_ratio": stability_ratios,
        "n_neighbors": [0] * len(param_values),  # not used by region identification
    })


class TestStableRegionDataclass(unittest.TestCase):
    """Verify StableRegion is frozen and has expected fields."""

    def test_frozen(self):
        region = StableRegion(
            param_ranges={"p": (1, 3)},
            param_combinations=[(1,), (2,), (3,)],
            mean_stability_ratio=0.9,
            mean_objective=1.5,
            min_objective=1.2,
            max_objective=1.8,
            n_combinations=3,
            is_boundary_region=False,
        )
        with self.assertRaises(AttributeError):
            region.n_combinations = 99  # type: ignore


class Test1DStableRegions(unittest.TestCase):
    """Test contiguous region detection in 1D."""

    def test_single_contiguous_region(self):
        # stability_ratios:  0.6  0.85  0.9  0.88  0.7
        # With threshold=0.8: stable at indices 1,2,3 → region = [(3,),(4,),(5,)]
        df = _make_1d_smoothed_df(
            param_values=[2, 3, 4, 5, 6],
            raw_values=[0.5, 0.6, 0.8, 0.7, 0.4],
            smoothed_values=[0.3, 0.51, 0.72, 0.616, 0.28],
            stability_ratios=[0.6, 0.85, 0.9, 0.88, 0.7],
        )
        regions = identify_stable_regions(df, "sortino", stability_threshold=0.8)

        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(region.n_combinations, 3)
        self.assertCountEqual(region.param_combinations, [(3,), (4,), (5,)])

    def test_no_stable_region(self):
        df = _make_1d_smoothed_df(
            param_values=[1, 2, 3],
            raw_values=[1.0, 1.0, 1.0],
            smoothed_values=[0.5, 0.5, 0.5],
            stability_ratios=[0.5, 0.5, 0.5],
        )
        regions = identify_stable_regions(df, "sortino", stability_threshold=0.8)
        self.assertEqual(len(regions), 0)

    def test_minimum_size_enforced(self):
        # Only one point above threshold → should NOT form a region
        df = _make_1d_smoothed_df(
            param_values=[1, 2, 3],
            raw_values=[1.0, 1.0, 1.0],
            smoothed_values=[0.5, 0.9, 0.5],
            stability_ratios=[0.5, 0.9, 0.5],
        )
        regions = identify_stable_regions(df, "sortino", stability_threshold=0.8)
        self.assertEqual(len(regions), 0)


class Test2DStableRegions(unittest.TestCase):
    """Test BFS-based stable plateau detection in 2D."""

    def setUp(self):
        # 3x3 grid with stable plateau in center (4 connected cells)
        #   p2=1  p2=2  p2=3
        # p1=1: 0.5   0.5   0.5
        # p1=2: 0.5   0.9   0.85
        # p1=3: 0.5   0.85  0.9
        rows = []
        ratios = {
            (1, 1): 0.5, (1, 2): 0.5,  (1, 3): 0.5,
            (2, 1): 0.5, (2, 2): 0.9,  (2, 3): 0.85,
            (3, 1): 0.5, (3, 2): 0.85, (3, 3): 0.9,
        }
        for (p1, p2), sr in ratios.items():
            rows.append({
                "param1_value": p1,
                "param2_value": p2,
                "sortino": 1.0,
                "smoothed_sortino": sr * 1.0,
                "stability_ratio": sr,
                "n_neighbors": 0,
            })
        self.df = pd.DataFrame(rows)

    def test_plateau_detected(self):
        regions = identify_stable_regions(self.df, "sortino", stability_threshold=0.8)
        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(region.n_combinations, 4)
        expected_combos = {(2, 2), (2, 3), (3, 2), (3, 3)}
        self.assertEqual(set(region.param_combinations), expected_combos)

    def test_mean_stability_ratio(self):
        regions = identify_stable_regions(self.df, "sortino", stability_threshold=0.8)
        region = regions[0]
        expected_mean = np.mean([0.9, 0.85, 0.85, 0.9])
        self.assertAlmostEqual(region.mean_stability_ratio, expected_mean, places=10)


class TestBoundaryDetection(unittest.TestCase):
    """Test is_boundary_region flag."""

    def test_boundary_region(self):
        # Region touches grid boundary (p1=1 is min)
        df = _make_1d_smoothed_df(
            param_values=[1, 2, 3, 4, 5],
            raw_values=[1.0, 1.0, 1.0, 1.0, 1.0],
            smoothed_values=[0.9, 0.9, 0.5, 0.5, 0.5],
            stability_ratios=[0.9, 0.9, 0.5, 0.5, 0.5],
        )
        regions = identify_stable_regions(df, "sortino", stability_threshold=0.8)
        self.assertEqual(len(regions), 1)
        self.assertTrue(regions[0].is_boundary_region)

    def test_non_boundary_region(self):
        # Region in the middle, doesn't touch edges
        df = _make_1d_smoothed_df(
            param_values=[1, 2, 3, 4, 5],
            raw_values=[1.0, 1.0, 1.0, 1.0, 1.0],
            smoothed_values=[0.5, 0.9, 0.9, 0.9, 0.5],
            stability_ratios=[0.5, 0.9, 0.9, 0.9, 0.5],
        )
        regions = identify_stable_regions(df, "sortino", stability_threshold=0.8)
        self.assertEqual(len(regions), 1)
        self.assertFalse(regions[0].is_boundary_region)


class TestMultipleDisjointRegions(unittest.TestCase):
    """Test grid with two separate stable regions."""

    def test_two_separate_regions(self):
        # 1D grid: stable at [1,2] and [5,6], gap at [3,4]
        df = _make_1d_smoothed_df(
            param_values=[1, 2, 3, 4, 5, 6],
            raw_values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            smoothed_values=[0.9, 0.85, 0.5, 0.5, 0.85, 0.9],
            stability_ratios=[0.9, 0.85, 0.5, 0.5, 0.85, 0.9],
        )
        regions = identify_stable_regions(df, "sortino", stability_threshold=0.8)
        self.assertEqual(len(regions), 2)

        # Verify non-overlapping
        all_combos = []
        for r in regions:
            all_combos.extend(r.param_combinations)
        self.assertEqual(len(all_combos), len(set(all_combos)))

    def test_regions_sorted_by_objective(self):
        # Region A has higher mean objective than Region B
        df = _make_1d_smoothed_df(
            param_values=[1, 2, 3, 4, 5, 6],
            raw_values=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            smoothed_values=[0.9, 0.85, 0.5, 0.5, 1.7, 1.8],
            stability_ratios=[0.9, 0.85, 0.5, 0.5, 0.85, 0.9],
        )
        regions = identify_stable_regions(df, "sortino", stability_threshold=0.8)
        self.assertEqual(len(regions), 2)
        # First region should have higher mean_objective
        self.assertGreaterEqual(regions[0].mean_objective, regions[1].mean_objective)


class TestEndToEndSmoothedRegions(unittest.TestCase):
    """Integration: compute_neighbor_smoothing → identify_stable_regions."""

    def test_pipeline(self):
        # A 1D grid where the peak at param=5 is surrounded by good neighbors
        df = pd.DataFrame({
            "param1_value": [1, 2, 3, 4, 5, 6, 7],
            "sortino": [0.2, 0.3, 0.8, 1.0, 1.2, 1.0, 0.3],
        })
        smoothed = compute_neighbor_smoothing(df, ["lookback"], "sortino")
        regions = identify_stable_regions(smoothed, "sortino", stability_threshold=0.8)

        # There should be at least one stable region around the peak
        total_stable_combos = sum(r.n_combinations for r in regions)
        self.assertGreater(total_stable_combos, 0)


if __name__ == "__main__":
    unittest.main()
