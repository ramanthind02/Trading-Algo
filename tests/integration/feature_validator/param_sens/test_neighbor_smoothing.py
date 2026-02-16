"""
Tests for T009 — compute_neighbor_smoothing() and identify_neighbors().

Covers 1D, 2D, 3D grids, stability ratio computation, determinism,
boundary handling, and the identify_neighbors() helper.
"""
import sys
import os
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from eda.parameter_analysis import compute_neighbor_smoothing, identify_neighbors


class TestIdentifyNeighbors(unittest.TestCase):
    """Test the identify_neighbors() helper function."""

    def test_1d_interior(self):
        grid = {"lookback": [2, 3, 4, 5, 6]}
        neighbors = identify_neighbors((4,), grid)
        self.assertCountEqual(neighbors, [(3,), (5,)])

    def test_1d_left_boundary(self):
        grid = {"lookback": [2, 3, 4, 5]}
        neighbors = identify_neighbors((2,), grid)
        self.assertEqual(neighbors, [(3,)])

    def test_1d_right_boundary(self):
        grid = {"lookback": [2, 3, 4, 5]}
        neighbors = identify_neighbors((5,), grid)
        self.assertEqual(neighbors, [(4,)])

    def test_2d_interior(self):
        grid = {"fast": [8, 16, 32], "slow": [32, 64, 128]}
        neighbors = identify_neighbors((16, 64), grid)
        expected = [(8, 64), (32, 64), (16, 32), (16, 128)]
        self.assertCountEqual(neighbors, expected)

    def test_2d_corner(self):
        grid = {"fast": [8, 16, 32], "slow": [32, 64, 128]}
        neighbors = identify_neighbors((8, 32), grid)
        expected = [(16, 32), (8, 64)]
        self.assertCountEqual(neighbors, expected)

    def test_3d_interior(self):
        grid = {"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]}
        neighbors = identify_neighbors((2, 20, 200), grid)
        # 6 axis-aligned neighbors
        expected = [
            (1, 20, 200), (3, 20, 200),
            (2, 10, 200), (2, 30, 200),
            (2, 20, 100), (2, 20, 300),
        ]
        self.assertCountEqual(neighbors, expected)


class Test1DNeighborSmoothing(unittest.TestCase):
    """Test compute_neighbor_smoothing on a 1D grid."""

    def setUp(self):
        self.df = pd.DataFrame({
            "param1_value": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "sortino": [0.5, 0.6, 0.8, 1.2, 1.1, 0.9, 0.7, 0.4, 0.3],
        })

    def test_output_columns_present(self):
        result = compute_neighbor_smoothing(self.df, ["lookback"], "sortino")
        self.assertIn("smoothed_sortino", result.columns)
        self.assertIn("stability_ratio", result.columns)
        self.assertIn("n_neighbors", result.columns)

    def test_boundary_neighbors_count(self):
        result = compute_neighbor_smoothing(self.df, ["lookback"], "sortino")
        # First and last rows have 1 neighbor each
        self.assertEqual(result.iloc[0]["n_neighbors"], 1)
        self.assertEqual(result.iloc[-1]["n_neighbors"], 1)
        # Interior rows have 2 neighbors
        for i in range(1, len(result) - 1):
            self.assertEqual(result.iloc[i]["n_neighbors"], 2)

    def test_smoothed_values(self):
        result = compute_neighbor_smoothing(self.df, ["lookback"], "sortino")
        # First row: mean(0.5, 0.6) = 0.55
        self.assertAlmostEqual(result.iloc[0]["smoothed_sortino"], 0.55, places=10)
        # Second row: mean(0.6, 0.5, 0.8) = 0.6333...
        self.assertAlmostEqual(result.iloc[1]["smoothed_sortino"], np.mean([0.6, 0.5, 0.8]), places=10)

    def test_stability_ratio_computation(self):
        result = compute_neighbor_smoothing(self.df, ["lookback"], "sortino")
        for _, row in result.iterrows():
            raw = row["sortino"]
            smoothed = row["smoothed_sortino"]
            if raw != 0:
                expected_ratio = smoothed / raw
                self.assertAlmostEqual(row["stability_ratio"], expected_ratio, places=10)

    def test_stability_ratio_zero_raw(self):
        df = pd.DataFrame({
            "param1_value": [1, 2, 3],
            "metric": [0.0, 0.5, 1.0],
        })
        result = compute_neighbor_smoothing(df, ["p"], "metric")
        # raw=0 at index 0 → stability_ratio = NaN
        self.assertTrue(np.isnan(result.iloc[0]["stability_ratio"]))

    def test_immutability(self):
        original = self.df.copy()
        compute_neighbor_smoothing(self.df, ["lookback"], "sortino")
        pd.testing.assert_frame_equal(self.df, original)


class Test2DNeighborSmoothing(unittest.TestCase):
    """Test compute_neighbor_smoothing on a 2D grid."""

    def setUp(self):
        fast_vals = [8, 16, 32, 64]
        slow_vals = [32, 64, 128, 256]
        rows = []
        np.random.seed(42)
        for f in fast_vals:
            for s in slow_vals:
                rows.append({
                    "param1_value": f,
                    "param2_value": s,
                    "sortino": np.random.uniform(0.5, 2.0),
                })
        self.df = pd.DataFrame(rows)

    def test_corner_neighbor_count(self):
        result = compute_neighbor_smoothing(self.df, ["fast", "slow"], "sortino")
        # Corner (8, 32): 2 neighbors
        corner_row = result[(result["param1_value"] == 8) & (result["param2_value"] == 32)]
        self.assertEqual(int(corner_row["n_neighbors"].iloc[0]), 2)

    def test_edge_neighbor_count(self):
        result = compute_neighbor_smoothing(self.df, ["fast", "slow"], "sortino")
        # Edge (16, 32): 3 neighbors
        edge_row = result[(result["param1_value"] == 16) & (result["param2_value"] == 32)]
        self.assertEqual(int(edge_row["n_neighbors"].iloc[0]), 3)

    def test_interior_neighbor_count(self):
        result = compute_neighbor_smoothing(self.df, ["fast", "slow"], "sortino")
        # Interior (16, 64): 4 neighbors
        interior_row = result[(result["param1_value"] == 16) & (result["param2_value"] == 64)]
        self.assertEqual(int(interior_row["n_neighbors"].iloc[0]), 4)

    def test_all_rows_present(self):
        result = compute_neighbor_smoothing(self.df, ["fast", "slow"], "sortino")
        self.assertEqual(len(result), len(self.df))


class Test3DNeighborSmoothing(unittest.TestCase):
    """Test 3D grid stability ratio."""

    def test_3d_stability_ratio(self):
        rows = []
        vals = [1, 2, 3]
        for a in vals:
            for b in vals:
                for c in vals:
                    rows.append({
                        "param1_value": a,
                        "param2_value": b,
                        "param3_value": c,
                        "sortino": float(a + b + c),
                    })
        df = pd.DataFrame(rows)
        result = compute_neighbor_smoothing(df, ["a", "b", "c"], "sortino")

        # All stability_ratios should be defined (no zeros in raw)
        self.assertFalse(result["stability_ratio"].isna().any())

        # Interior point (2,2,2): 6 neighbors, raw=6.0
        interior = result[
            (result["param1_value"] == 2) &
            (result["param2_value"] == 2) &
            (result["param3_value"] == 2)
        ]
        self.assertEqual(int(interior["n_neighbors"].iloc[0]), 6)


class TestDeterminism(unittest.TestCase):
    """Identical inputs produce identical outputs."""

    def test_deterministic(self):
        df = pd.DataFrame({
            "param1_value": [1, 2, 3, 4, 5],
            "sortino": [0.5, 1.0, 1.5, 1.0, 0.5],
        })
        r1 = compute_neighbor_smoothing(df, ["p"], "sortino")
        r2 = compute_neighbor_smoothing(df, ["p"], "sortino")
        pd.testing.assert_frame_equal(r1, r2)


if __name__ == "__main__":
    unittest.main()
