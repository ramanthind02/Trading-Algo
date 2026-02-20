"""
Tests for T012 — ParameterSensitivityReport dataclass and
generate_parameter_sensitivity_report() orchestrator.

Covers 1D, 2D report generation, top-K selection, determinism,
and report field validation.
"""
import sys
import os
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from eda.parameter_analysis import (
    ParameterSensitivityReport,
    generate_parameter_sensitivity_report,
    StableRegion,
)


class TestParameterSensitivityReportDataclass(unittest.TestCase):
    """Verify the report dataclass structure."""

    def test_frozen(self):
        report = ParameterSensitivityReport(
            param_names=["lookback"],
            metric_name="sortino",
            stability_threshold=0.8,
            grid_results=pd.DataFrame(),
            stable_regions=[],
            mean_stability_ratio=0.0,
            median_stability_ratio=0.0,
            pct_stable_combinations=0.0,
            recommended_combinations=[],
            top_k_combinations=[],
            n_parameter_combinations=0,
            n_stable_regions=0,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        with self.assertRaises(AttributeError):
            report.metric_name = "sharpe"  # type: ignore

    def test_optional_plots_default_none(self):
        report = ParameterSensitivityReport(
            param_names=["p"],
            metric_name="sortino",
            stability_threshold=0.8,
            grid_results=pd.DataFrame(),
            stable_regions=[],
            mean_stability_ratio=0.0,
            median_stability_ratio=0.0,
            pct_stable_combinations=0.0,
            recommended_combinations=[],
            top_k_combinations=[],
        )
        self.assertIsNone(report.plot_1d)
        self.assertIsNone(report.plot_2d)
        self.assertIsNone(report.plot_3d)


class Test1DReportGeneration(unittest.TestCase):
    """Test full pipeline for a 1D grid."""

    def setUp(self):
        # RSI lookback sweep with clear stable plateau around 5-7
        self.df = pd.DataFrame({
            "param1_value": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "sortino": [0.2, 0.4, 0.7, 1.0, 1.1, 1.0, 0.7, 0.3, 0.1],
        })

    def test_report_fields_populated(self):
        report = generate_parameter_sensitivity_report(
            self.df, ["lookback"], "sortino",
        )
        self.assertEqual(report.param_names, ["lookback"])
        self.assertEqual(report.metric_name, "sortino")
        self.assertEqual(report.stability_threshold, 0.8)
        self.assertGreater(len(report.grid_results), 0)
        self.assertIsInstance(report.grid_results, pd.DataFrame)
        self.assertIn("smoothed_sortino", report.grid_results.columns)
        self.assertIn("stability_ratio", report.grid_results.columns)
        self.assertIn("n_neighbors", report.grid_results.columns)

    def test_plot_1d_populated(self):
        report = generate_parameter_sensitivity_report(
            self.df, ["lookback"], "sortino",
        )
        self.assertIsNotNone(report.plot_1d)
        self.assertIsInstance(report.plot_1d, go.Figure)
        self.assertIsNone(report.plot_2d)
        self.assertIsNone(report.plot_3d)

    def test_summary_statistics(self):
        report = generate_parameter_sensitivity_report(
            self.df, ["lookback"], "sortino",
        )
        self.assertGreater(report.mean_stability_ratio, 0)
        self.assertGreater(report.median_stability_ratio, 0)
        self.assertGreaterEqual(report.pct_stable_combinations, 0.0)
        self.assertLessEqual(report.pct_stable_combinations, 1.0)

    def test_timestamp_iso_format(self):
        report = generate_parameter_sensitivity_report(
            self.df, ["lookback"], "sortino",
        )
        self.assertIn("T", report.timestamp)
        self.assertGreater(len(report.timestamp), 10)

    def test_n_parameter_combinations(self):
        report = generate_parameter_sensitivity_report(
            self.df, ["lookback"], "sortino",
        )
        self.assertEqual(report.n_parameter_combinations, 9)


class Test2DReportGeneration(unittest.TestCase):
    """Test full pipeline for a 2D grid."""

    def setUp(self):
        rows = []
        np.random.seed(42)
        for f in [8, 16, 32, 64]:
            for s in [32, 64, 128, 256]:
                rows.append({
                    "param1_value": f,
                    "param2_value": s,
                    "sortino": np.random.uniform(0.5, 2.0),
                })
        self.df = pd.DataFrame(rows)

    def test_plot_2d_populated(self):
        report = generate_parameter_sensitivity_report(
            self.df, ["fast", "slow"], "sortino",
        )
        self.assertIsNotNone(report.plot_2d)
        self.assertIsInstance(report.plot_2d, go.Figure)
        self.assertIsNone(report.plot_1d)
        self.assertIsNone(report.plot_3d)

    def test_recommended_combinations_ranked(self):
        report = generate_parameter_sensitivity_report(
            self.df, ["fast", "slow"], "sortino",
        )
        if len(report.recommended_combinations) >= 2:
            # Verify descending order by checking the smoothed values
            smoothed_col = "smoothed_sortino"
            param_cols = ["param1_value", "param2_value"]
            grid = report.grid_results
            for i in range(len(report.recommended_combinations) - 1):
                combo_i = report.recommended_combinations[i]
                combo_next = report.recommended_combinations[i + 1]
                mask_i = (grid[param_cols[0]] == combo_i[0]) & (grid[param_cols[1]] == combo_i[1])
                mask_next = (grid[param_cols[0]] == combo_next[0]) & (grid[param_cols[1]] == combo_next[1])
                val_i = grid.loc[mask_i, smoothed_col].iloc[0]
                val_next = grid.loc[mask_next, smoothed_col].iloc[0]
                self.assertGreaterEqual(val_i, val_next)


class TestTopKSelection(unittest.TestCase):
    """Test top-K parameter combination selection."""

    def test_top_k_respects_limit(self):
        # Large 1D grid with many stable points
        n = 20
        df = pd.DataFrame({
            "param1_value": list(range(1, n + 1)),
            "sortino": [1.0] * n,  # all equal → all stable
        })
        report = generate_parameter_sensitivity_report(
            df, ["p"], "sortino", top_k=3,
        )
        self.assertLessEqual(len(report.top_k_combinations), 3)

    def test_top_k_fewer_than_k(self):
        # Only 2 stable combos, K=5 → should return 2
        df = pd.DataFrame({
            "param1_value": [1, 2, 3],
            "sortino": [0.1, 1.0, 0.1],
        })
        report = generate_parameter_sensitivity_report(
            df, ["p"], "sortino", top_k=5,
        )
        self.assertLessEqual(len(report.top_k_combinations), 5)

    def test_top_k_is_subset_of_recommended(self):
        df = pd.DataFrame({
            "param1_value": list(range(1, 10)),
            "sortino": [0.5, 0.8, 1.0, 1.2, 1.1, 0.9, 0.7, 0.4, 0.2],
        })
        report = generate_parameter_sensitivity_report(
            df, ["p"], "sortino", top_k=3,
        )
        for combo in report.top_k_combinations:
            self.assertIn(combo, report.recommended_combinations)


class TestReportDeterminism(unittest.TestCase):
    """Identical inputs produce identical reports (except timestamp)."""

    def test_deterministic(self):
        df = pd.DataFrame({
            "param1_value": [1, 2, 3, 4, 5],
            "sortino": [0.5, 1.0, 1.5, 1.0, 0.5],
        })
        r1 = generate_parameter_sensitivity_report(df, ["p"], "sortino")
        r2 = generate_parameter_sensitivity_report(df, ["p"], "sortino")

        pd.testing.assert_frame_equal(r1.grid_results, r2.grid_results)
        self.assertEqual(r1.stable_regions, r2.stable_regions)
        self.assertEqual(r1.recommended_combinations, r2.recommended_combinations)
        self.assertEqual(r1.top_k_combinations, r2.top_k_combinations)
        self.assertAlmostEqual(r1.mean_stability_ratio, r2.mean_stability_ratio)
        self.assertAlmostEqual(r1.pct_stable_combinations, r2.pct_stable_combinations)


class Test3DReportGeneration(unittest.TestCase):
    """Test 3D report routing modes."""

    def setUp(self):
        rows = []
        for p1 in [1, 2, 3]:
            for p2 in [10, 20, 30]:
                for p3 in [100, 200, 300]:
                    rows.append({
                        "param1_value": p1,
                        "param2_value": p2,
                        "param3_value": p3,
                        "sortino": float(p1 + p2 / 10 + p3 / 100),
                    })
        self.df = pd.DataFrame(rows)

    def test_plot_3d_surface_mode(self):
        report = generate_parameter_sensitivity_report(
            self.df, ["a", "b", "c"], "sortino", plot_3d_mode="surface_slices",
        )
        self.assertIsNotNone(report.plot_3d)
        self.assertIsInstance(report.plot_3d, go.Figure)
        self.assertTrue(all(isinstance(t, go.Surface) for t in report.plot_3d.data))

    def test_invalid_plot_3d_mode_raises(self):
        with self.assertRaises(ValueError):
            generate_parameter_sensitivity_report(
                self.df, ["a", "b", "c"], "sortino", plot_3d_mode="bad_mode",
            )


if __name__ == "__main__":
    unittest.main()
