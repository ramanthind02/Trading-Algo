"""
Unit tests for grid-aware neighbor averaging (utils/grid_smoothing.py).

Tests cover:
1. 2D worked example from spec section 6
2. 1D grid with boundary handling
3. 3D grid with interior/boundary points
4. Partial grid (missing cells)
5. Single row
6. Empty DataFrame
7. Duplicate param tuples
8. NaN handling in objectives
9. Custom output column name
10. Input immutability
11. Non-uniform grid spacing
"""

import unittest
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compute.grid_smoothing import add_smoothed_objective


class TestWorkedExample2D(unittest.TestCase):
    """Exact match to spec section 6 worked example."""

    def setUp(self):
        self.df = pd.DataFrame({
            "lookback": [7, 7, 7, 14, 14, 14, 21, 21, 21],
            "threshold": [30, 50, 70, 30, 50, 70, 30, 50, 70],
            "objective": [0.8, 1.0, 0.6, 1.1, 1.4, 1.0, 0.9, 1.2, 0.7],
        })
        self.param_columns = ["lookback", "threshold"]

    def test_center_point(self):
        """Row (14, 50) should have avg_objective = 1.14."""
        result = add_smoothed_objective(
            self.df, self.param_columns, "objective"
        )
        row = result[(result["lookback"] == 14) & (result["threshold"] == 50)]
        self.assertAlmostEqual(row["avg_objective"].iloc[0], 1.14, places=10)

    def test_corner_point(self):
        """Row (7, 30): neighbors are (14,30)=1.1, (7,50)=1.0. Mean of 0.8,1.1,1.0."""
        result = add_smoothed_objective(
            self.df, self.param_columns, "objective"
        )
        row = result[(result["lookback"] == 7) & (result["threshold"] == 30)]
        expected = np.mean([0.8, 1.1, 1.0])
        self.assertAlmostEqual(row["avg_objective"].iloc[0], expected, places=10)

    def test_edge_point(self):
        """Row (14, 30): neighbors are (7,30)=0.8, (21,30)=0.9, (14,50)=1.4.
        Mean of 1.1, 0.8, 0.9, 1.4."""
        result = add_smoothed_objective(
            self.df, self.param_columns, "objective"
        )
        row = result[(result["lookback"] == 14) & (result["threshold"] == 30)]
        expected = np.mean([1.1, 0.8, 0.9, 1.4])
        self.assertAlmostEqual(row["avg_objective"].iloc[0], expected, places=10)

    def test_output_column_present(self):
        result = add_smoothed_objective(
            self.df, self.param_columns, "objective"
        )
        self.assertIn("avg_objective", result.columns)

    def test_all_original_columns_preserved(self):
        result = add_smoothed_objective(
            self.df, self.param_columns, "objective"
        )
        for col in self.df.columns:
            self.assertIn(col, result.columns)

    def test_row_count_unchanged(self):
        result = add_smoothed_objective(
            self.df, self.param_columns, "objective"
        )
        self.assertEqual(len(result), len(self.df))


class Test1DGrid(unittest.TestCase):
    """1D grid: boundary points average over fewer neighbors."""

    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "obj": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    def test_boundary_left(self):
        """x=1: neighbors are (2,). Mean of [10, 20] = 15."""
        result = add_smoothed_objective(self.df, ["x"], "obj")
        row = result[result["x"] == 1]
        self.assertAlmostEqual(row["avg_objective"].iloc[0], 15.0)

    def test_interior(self):
        """x=3: neighbors are (2,) and (4,). Mean of [30, 20, 40] = 30."""
        result = add_smoothed_objective(self.df, ["x"], "obj")
        row = result[result["x"] == 3]
        self.assertAlmostEqual(row["avg_objective"].iloc[0], 30.0)

    def test_boundary_right(self):
        """x=5: neighbors are (4,). Mean of [50, 40] = 45."""
        result = add_smoothed_objective(self.df, ["x"], "obj")
        row = result[result["x"] == 5]
        self.assertAlmostEqual(row["avg_objective"].iloc[0], 45.0)


class Test3DGrid(unittest.TestCase):
    """3D grid: interior point has up to 6 neighbors."""

    def setUp(self):
        rows = [
            {"a": a, "b": b, "c": c, "obj": float(a + b + c)}
            for a in [1, 2, 3]
            for b in [10, 20, 30]
            for c in [100, 200, 300]
        ]
        self.df = pd.DataFrame(rows)

    def test_interior_point_neighbor_count(self):
        """Interior point (2, 20, 200) should have 6 neighbors + self = 7 values."""
        result = add_smoothed_objective(self.df, ["a", "b", "c"], "obj")
        row = result[
            (result["a"] == 2) & (result["b"] == 20) & (result["c"] == 200)
        ]
        # self = 222, neighbors: (1,20,200)=221, (3,20,200)=223,
        # (2,10,200)=212, (2,30,200)=232, (2,20,100)=122, (2,20,300)=322
        expected = np.mean([222, 221, 223, 212, 232, 122, 322])
        self.assertAlmostEqual(row["avg_objective"].iloc[0], expected, places=10)

    def test_corner_point_neighbor_count(self):
        """Corner (1, 10, 100) should have 3 neighbors + self = 4 values."""
        result = add_smoothed_objective(self.df, ["a", "b", "c"], "obj")
        row = result[
            (result["a"] == 1) & (result["b"] == 10) & (result["c"] == 100)
        ]
        # self = 111, neighbors: (2,10,100)=112, (1,20,100)=121, (1,10,200)=211
        expected = np.mean([111, 112, 121, 211])
        self.assertAlmostEqual(row["avg_objective"].iloc[0], expected, places=10)


class TestPartialGrid(unittest.TestCase):
    """Missing cells in grid; neighbors that don't exist are skipped."""

    def test_missing_neighbor(self):
        # 2D grid with some cells missing
        df = pd.DataFrame({
            "x": [1, 1, 2, 3, 3],
            "y": [1, 2, 1, 1, 2],
            "obj": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = add_smoothed_objective(df, ["x", "y"], "obj")

        # (2, 1): neighbors are (1,1)=1.0, (3,1)=4.0. (2,2) does NOT exist.
        row = result[(result["x"] == 2) & (result["y"] == 1)]
        expected = np.mean([3.0, 1.0, 4.0])
        self.assertAlmostEqual(row["avg_objective"].iloc[0], expected, places=10)


class TestSingleRow(unittest.TestCase):
    """Single row: output = self objective."""

    def test_single_row(self):
        df = pd.DataFrame({"p": [5], "obj": [3.14]})
        result = add_smoothed_objective(df, ["p"], "obj")
        self.assertAlmostEqual(result["avg_objective"].iloc[0], 3.14)


class TestEmptyDataFrame(unittest.TestCase):
    """Empty DataFrame returns copy with output column."""

    def test_empty(self):
        df = pd.DataFrame({"p": pd.Series(dtype=int), "obj": pd.Series(dtype=float)})
        result = add_smoothed_objective(df, ["p"], "obj")
        self.assertIn("avg_objective", result.columns)
        self.assertEqual(len(result), 0)


class TestDuplicateParamTuples(unittest.TestCase):
    """Multiple rows with same params; neighbor uses mean representative."""

    def test_duplicate_cells(self):
        df = pd.DataFrame({
            "x": [1, 1, 2],
            "obj": [10.0, 20.0, 30.0],
        })
        result = add_smoothed_objective(df, ["x"], "obj")

        # Representative for x=1 is mean(10, 20) = 15.0
        # Representative for x=2 is 30.0
        # Row 0 (x=1, obj=10): self=10, neighbor x=2 rep=30 -> mean(10, 30) = 20
        # Row 1 (x=1, obj=20): self=20, neighbor x=2 rep=30 -> mean(20, 30) = 25
        # Row 2 (x=2, obj=30): self=30, neighbor x=1 rep=15 -> mean(30, 15) = 22.5
        self.assertAlmostEqual(result["avg_objective"].iloc[0], 20.0)
        self.assertAlmostEqual(result["avg_objective"].iloc[1], 25.0)
        self.assertAlmostEqual(result["avg_objective"].iloc[2], 22.5)


class TestNaNHandling(unittest.TestCase):
    """NaN objectives excluded from neighbor averages."""

    def test_nan_neighbor_excluded(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "obj": [10.0, np.nan, 30.0],
        })
        result = add_smoothed_objective(df, ["x"], "obj")

        # Row x=1 (obj=10): neighbor x=2 is NaN -> excluded. Mean of [10] = 10
        self.assertAlmostEqual(result["avg_objective"].iloc[0], 10.0)

        # Row x=2 (obj=NaN): self=NaN, neighbors x=1 rep=10, x=3 rep=30
        # nanmean([NaN, 10, 30]) = 20
        self.assertAlmostEqual(result["avg_objective"].iloc[1], 20.0)

        # Row x=3 (obj=30): neighbor x=2 is NaN -> excluded. Mean of [30] = 30
        self.assertAlmostEqual(result["avg_objective"].iloc[2], 30.0)


class TestCustomOutputColumn(unittest.TestCase):
    """Custom output column name works."""

    def test_custom_name(self):
        df = pd.DataFrame({"p": [1, 2], "obj": [1.0, 2.0]})
        result = add_smoothed_objective(
            df, ["p"], "obj", output_column="smoothed_sortino"
        )
        self.assertIn("smoothed_sortino", result.columns)
        self.assertNotIn("avg_objective", result.columns)


class TestImmutability(unittest.TestCase):
    """Input DataFrame is not modified."""

    def test_input_unchanged(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "obj": [10.0, 20.0, 30.0],
        })
        original_columns = list(df.columns)
        original_values = df.values.copy()

        add_smoothed_objective(df, ["x"], "obj")

        self.assertEqual(list(df.columns), original_columns)
        np.testing.assert_array_equal(df.values, original_values)


class TestNonUniformGrid(unittest.TestCase):
    """Non-uniform spacing uses sorted adjacency."""

    def test_non_uniform_spacing(self):
        # Levels: [5, 10, 20, 40] — gaps are 5, 10, 20 (non-uniform)
        df = pd.DataFrame({
            "x": [5, 10, 20, 40],
            "obj": [1.0, 2.0, 3.0, 4.0],
        })
        result = add_smoothed_objective(df, ["x"], "obj")

        # x=10: neighbors are 5 (prev) and 20 (next) in sorted order
        # Mean of [2.0, 1.0, 3.0] = 2.0
        row = result[result["x"] == 10]
        self.assertAlmostEqual(row["avg_objective"].iloc[0], 2.0)

        # x=20: neighbors are 10 (prev) and 40 (next)
        # Mean of [3.0, 2.0, 4.0] = 3.0
        row = result[result["x"] == 20]
        self.assertAlmostEqual(row["avg_objective"].iloc[0], 3.0)

        # x=5: neighbor is 10 only
        # Mean of [1.0, 2.0] = 1.5
        row = result[result["x"] == 5]
        self.assertAlmostEqual(row["avg_objective"].iloc[0], 1.5)


class TestValidation(unittest.TestCase):
    """Input validation raises appropriate errors."""

    def test_missing_param_column(self):
        df = pd.DataFrame({"x": [1], "obj": [1.0]})
        with self.assertRaises(ValueError):
            add_smoothed_objective(df, ["x", "missing"], "obj")

    def test_missing_objective_column(self):
        df = pd.DataFrame({"x": [1], "obj": [1.0]})
        with self.assertRaises(ValueError):
            add_smoothed_objective(df, ["x"], "missing_obj")


if __name__ == "__main__":
    unittest.main()
    