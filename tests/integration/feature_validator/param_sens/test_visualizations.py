"""
Tests for T011 — Stability-aware visualization functions.

Validates figure structure, trace types, hover templates, stable region
shading, and the 3D slicing interface.
"""
import sys
import os
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from metrics.plotting.parameter_plots import (
    plot_parameter_sensitivity_with_stability,
    plot_2d_stability_heatmap,
    plot_3d_slices,
)
from eda.parameter_analysis import StableRegion


def _make_1d_df():
    return pd.DataFrame({
        "param1_value": [2, 3, 4, 5, 6, 7, 8],
        "sortino": [0.5, 0.7, 1.0, 1.2, 1.1, 0.8, 0.4],
        "smoothed_sortino": [0.6, 0.73, 0.97, 1.1, 1.03, 0.77, 0.6],
        "stability_ratio": [1.2, 1.04, 0.97, 0.92, 0.94, 0.96, 1.5],
        "n_neighbors": [1, 2, 2, 2, 2, 2, 1],
    })


def _make_2d_df():
    rows = []
    np.random.seed(42)
    for p1 in [1, 2, 3, 4]:
        for p2 in [10, 20, 30, 40]:
            raw = np.random.uniform(0.5, 2.0)
            rows.append({
                "param1_value": p1,
                "param2_value": p2,
                "sortino": raw,
                "smoothed_sortino": raw * 0.95,
                "stability_ratio": 0.95 if (p1 in [2, 3] and p2 in [20, 30]) else 0.5,
                "n_neighbors": 4,
            })
    return pd.DataFrame(rows)


def _make_3d_df():
    rows = []
    for a in [1, 2, 3]:
        for b in [10, 20, 30]:
            for c in [100, 200, 300]:
                raw = float(a + b / 10 + c / 100)
                rows.append({
                    "param1_value": a,
                    "param2_value": b,
                    "param3_value": c,
                    "sortino": raw,
                    "smoothed_sortino": raw * 0.9,
                    "stability_ratio": 0.9,
                })
    return pd.DataFrame(rows)


class Test1DStabilityPlot(unittest.TestCase):
    """Test plot_parameter_sensitivity_with_stability."""

    def test_returns_figure(self):
        fig = plot_parameter_sensitivity_with_stability(
            _make_1d_df(), "lookback", "sortino", show_plot=False,
        )
        self.assertIsInstance(fig, go.Figure)

    def test_has_raw_and_smoothed_traces(self):
        fig = plot_parameter_sensitivity_with_stability(
            _make_1d_df(), "lookback", "sortino", show_plot=False,
        )
        trace_names = [t.name for t in fig.data if hasattr(t, "name")]
        self.assertTrue(any("raw" in n.lower() for n in trace_names))
        self.assertTrue(any("smoothed" in n.lower() for n in trace_names))

    def test_has_stability_ratio_trace(self):
        fig = plot_parameter_sensitivity_with_stability(
            _make_1d_df(), "lookback", "sortino", show_plot=False,
        )
        trace_names = [t.name for t in fig.data if hasattr(t, "name")]
        self.assertTrue(any("stability" in n.lower() for n in trace_names))

    def test_stable_region_shading(self):
        region = StableRegion(
            param_ranges={"p": (4, 6)},
            param_combinations=[(4,), (5,), (6,)],
            mean_stability_ratio=0.94,
            mean_objective=1.03,
            min_objective=0.97,
            max_objective=1.1,
            n_combinations=3,
            is_boundary_region=False,
        )
        fig = plot_parameter_sensitivity_with_stability(
            _make_1d_df(), "lookback", "sortino",
            stable_regions=[region],
            show_plot=False,
        )
        # Should have vrect shapes for stable region shading
        shapes = fig.layout.shapes
        self.assertGreater(len(shapes), 0)

    def test_layout_template(self):
        fig = plot_parameter_sensitivity_with_stability(
            _make_1d_df(), "lookback", "sortino", show_plot=False,
        )
        # Verify plotly_white template is applied (check a known attribute)
        template = fig.layout.template
        self.assertIsNotNone(template)
        # plotly_white sets a white plot background
        self.assertEqual(fig.layout.template.layout.plot_bgcolor, "white")


class Test2DStabilityHeatmap(unittest.TestCase):
    """Test plot_2d_stability_heatmap."""

    def test_returns_figure(self):
        fig = plot_2d_stability_heatmap(
            _make_2d_df(), "fast", "slow", "sortino", show_plot=False,
        )
        self.assertIsInstance(fig, go.Figure)

    def test_has_heatmap_trace(self):
        fig = plot_2d_stability_heatmap(
            _make_2d_df(), "fast", "slow", "sortino", show_plot=False,
        )
        trace_types = [type(t).__name__ for t in fig.data]
        self.assertIn("Heatmap", trace_types)

    def test_has_contour_overlay(self):
        fig = plot_2d_stability_heatmap(
            _make_2d_df(), "fast", "slow", "sortino", show_plot=False,
        )
        trace_types = [type(t).__name__ for t in fig.data]
        self.assertIn("Contour", trace_types)

    def test_stable_region_markers(self):
        region = StableRegion(
            param_ranges={"fast": (2, 3), "slow": (20, 30)},
            param_combinations=[(2, 20), (2, 30), (3, 20), (3, 30)],
            mean_stability_ratio=0.95,
            mean_objective=1.5,
            min_objective=1.2,
            max_objective=1.8,
            n_combinations=4,
            is_boundary_region=False,
        )
        fig = plot_2d_stability_heatmap(
            _make_2d_df(), "fast", "slow", "sortino",
            stable_regions=[region],
            show_plot=False,
        )
        # Should have a Scatter trace for region markers
        trace_types = [type(t).__name__ for t in fig.data]
        self.assertIn("Scatter", trace_types)


class Test3DSlices(unittest.TestCase):
    """Test plot_3d_slices."""

    def test_returns_figure(self):
        fig = plot_3d_slices(
            _make_3d_df(), ["a", "b", "c"], "sortino", show_plot=False,
        )
        self.assertIsInstance(fig, go.Figure)

    def test_has_dropdown(self):
        fig = plot_3d_slices(
            _make_3d_df(), ["a", "b", "c"], "sortino", show_plot=False,
        )
        menus = fig.layout.updatemenus
        self.assertIsNotNone(menus)
        self.assertGreater(len(menus), 0)
        # Should have 3 buttons (Fix a, Fix b, Fix c)
        self.assertEqual(len(menus[0].buttons), 3)

    def test_has_slider(self):
        fig = plot_3d_slices(
            _make_3d_df(), ["a", "b", "c"], "sortino", show_plot=False,
        )
        sliders = fig.layout.sliders
        self.assertIsNotNone(sliders)
        self.assertGreater(len(sliders), 0)

    def test_traces_are_heatmaps(self):
        fig = plot_3d_slices(
            _make_3d_df(), ["a", "b", "c"], "sortino", show_plot=False,
        )
        for trace in fig.data:
            self.assertIsInstance(trace, go.Heatmap)

    def test_raises_for_less_than_3_params(self):
        with self.assertRaises(ValueError):
            plot_3d_slices(
                _make_3d_df(), ["a", "b"], "sortino", show_plot=False,
            )


class TestExportsAvailable(unittest.TestCase):
    """Verify the new functions are exported from metrics.plotting."""

    def test_imports(self):
        from metrics.plotting import (
            plot_parameter_sensitivity_with_stability,
            plot_2d_stability_heatmap,
            plot_3d_slices,
        )
        self.assertTrue(callable(plot_parameter_sensitivity_with_stability))
        self.assertTrue(callable(plot_2d_stability_heatmap))
        self.assertTrue(callable(plot_3d_slices))


if __name__ == "__main__":
    unittest.main()
