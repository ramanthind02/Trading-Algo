"""Unit tests for binning diagnostic plot functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validators.binning.diagnostics import RegionMetadata
from feature_selection.validators.binning.plots import (
    create_diagnostic_panel,
    plot_bin_heatmap,
    plot_position_multiplier_curve,
    plot_region_boundaries,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _build_fitted_model() -> ContinuousBinningModel:
    """Build a synthetic fitted ContinuousBinningModel with 5 bins."""
    model = ContinuousBinningModel(n_bins=5)
    model.is_fitted_ = True
    model.feature_column = "rsi_signal_D_lookback_5"
    model.bin_edges_ = [20.0, 40.0, 60.0, 80.0]
    model.bin_stats_ = {
        0: {
            "sharpe": -0.3, "t_stat": -1.0, "count": 50, "mean_return": -0.002,
            "selection_metric_long": -0.3, "selection_metric_short": 0.3,
            "feature_min": 0.0, "feature_max": 20.0,
            "adjusted_sharpe": -0.2, "adjusted_sharpe_short": 0.2,
            "volatility": 0.01,
        },
        1: {
            "sharpe": 0.1, "t_stat": 0.5, "count": 60, "mean_return": 0.0005,
            "selection_metric_long": 0.1, "selection_metric_short": -0.1,
            "feature_min": 20.0, "feature_max": 40.0,
            "adjusted_sharpe": 0.08, "adjusted_sharpe_short": -0.08,
            "volatility": 0.01,
        },
        2: {
            "sharpe": 0.0, "t_stat": 0.0, "count": 70, "mean_return": 0.0,
            "selection_metric_long": 0.0, "selection_metric_short": 0.0,
            "feature_min": 40.0, "feature_max": 60.0,
            "adjusted_sharpe": 0.0, "adjusted_sharpe_short": 0.0,
            "volatility": 0.01,
        },
        3: {
            "sharpe": 0.8, "t_stat": 2.5, "count": 55, "mean_return": 0.004,
            "selection_metric_long": 0.8, "selection_metric_short": -0.8,
            "feature_min": 60.0, "feature_max": 80.0,
            "adjusted_sharpe": 0.6, "adjusted_sharpe_short": -0.6,
            "volatility": 0.01,
        },
        4: {
            "sharpe": 1.2, "t_stat": 3.0, "count": 45, "mean_return": 0.006,
            "selection_metric_long": 1.2, "selection_metric_short": -1.2,
            "feature_min": 80.0, "feature_max": 100.0,
            "adjusted_sharpe": 0.9, "adjusted_sharpe_short": -0.9,
            "volatility": 0.01,
        },
    }
    model.significant_regions_ = [
        {"start_bin": 3, "end_bin": 4, "bins": [3, 4]},
    ]
    model.active_bins_by_strategy_ = {
        "long": [3, 4],
        "short": [0],
        "long_short": [0, 3, 4],
    }
    model.position_multipliers_by_strategy_ = {
        "long": {3: 1.6, 4: 1.9},
        "short": {0: -1.2},
        "long_short": {0: -1.2, 3: 1.6, 4: 1.9},
    }
    return model


def _build_feature_data() -> pd.Series:
    """Synthetic feature data spanning the full 0-100 range."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.uniform(0, 100, size=500), name="rsi_signal_D_lookback_5")


def _build_regions() -> list[RegionMetadata]:
    """Regions matching the synthetic model."""
    return [
        RegionMetadata(
            start_bin=3,
            end_bin=4,
            bins=[3, 4],
            mean_sharpe=1.0,
            mean_t_stat=2.75,
            sample_count=100,
            feature_range=(60.0, 100.0),
        ),
    ]


# ---------------------------------------------------------------------------
# plot_bin_heatmap tests
# ---------------------------------------------------------------------------

class TestPlotBinHeatmap:
    def test_returns_figure(self) -> None:
        model = _build_fitted_model()
        fig = plot_bin_heatmap(model)
        assert isinstance(fig, Figure)

    def test_metric_options(self) -> None:
        model = _build_fitted_model()
        for metric in ("sharpe", "t_stat", "sample_count", "mean_return"):
            fig = plot_bin_heatmap(model, metric=metric)
            assert isinstance(fig, Figure)

    def test_invalid_metric_raises(self) -> None:
        model = _build_fitted_model()
        with pytest.raises(ValueError, match="metric"):
            plot_bin_heatmap(model, metric="invalid_metric")

    def test_custom_cmap(self) -> None:
        model = _build_fitted_model()
        fig = plot_bin_heatmap(model, cmap="viridis")
        assert isinstance(fig, Figure)

    def test_axes_count(self) -> None:
        model = _build_fitted_model()
        fig = plot_bin_heatmap(model)
        axes = fig.get_axes()
        assert len(axes) >= 1


# ---------------------------------------------------------------------------
# plot_region_boundaries tests
# ---------------------------------------------------------------------------

class TestPlotRegionBoundaries:
    def test_returns_figure(self) -> None:
        model = _build_fitted_model()
        feature_data = _build_feature_data()
        regions = _build_regions()
        fig = plot_region_boundaries(model, feature_data, regions)
        assert isinstance(fig, Figure)

    def test_shaded_regions_count(self) -> None:
        model = _build_fitted_model()
        feature_data = _build_feature_data()
        regions = _build_regions()
        fig = plot_region_boundaries(model, feature_data, regions)
        ax = fig.get_axes()[0]
        # Each region produces an axvspan patch
        shaded_patches = [
            p for p in ax.patches
            if getattr(p, "get_alpha", lambda: None)() is not None
            and getattr(p, "get_alpha", lambda: None)() < 1.0
        ]
        assert len(shaded_patches) >= len(regions)

    def test_empty_regions(self) -> None:
        model = _build_fitted_model()
        feature_data = _build_feature_data()
        fig = plot_region_boundaries(model, feature_data, [])
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_position_multiplier_curve tests
# ---------------------------------------------------------------------------

class TestPlotPositionMultiplierCurve:
    def test_returns_figure(self) -> None:
        model = _build_fitted_model()
        fig = plot_position_multiplier_curve(model, strategy="long")
        assert isinstance(fig, Figure)

    def test_step_count(self) -> None:
        model = _build_fitted_model()
        fig = plot_position_multiplier_curve(model, strategy="long")
        ax = fig.get_axes()[0]
        lines = ax.get_lines()
        # Should have at least one line (the step function)
        assert len(lines) >= 1

    def test_short_strategy(self) -> None:
        model = _build_fitted_model()
        fig = plot_position_multiplier_curve(model, strategy="short")
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# create_diagnostic_panel tests
# ---------------------------------------------------------------------------

class TestCreateDiagnosticPanel:
    def test_subplot_count(self) -> None:
        model = _build_fitted_model()
        feature_data = _build_feature_data()
        regions = _build_regions()
        fig = create_diagnostic_panel(model, feature_data, regions)
        axes = fig.get_axes()
        assert len(axes) == 3

    def test_returns_figure(self) -> None:
        model = _build_fitted_model()
        feature_data = _build_feature_data()
        regions = _build_regions()
        fig = create_diagnostic_panel(model, feature_data, regions)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Save to tempdir test
# ---------------------------------------------------------------------------

class TestSaveDiagnosticPlots:
    def test_save_to_tempdir(self) -> None:
        model = _build_fitted_model()
        feature_data = _build_feature_data()
        regions = _build_regions()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_heatmap = plot_bin_heatmap(model)
            fig_boundaries = plot_region_boundaries(model, feature_data, regions)
            fig_multiplier = plot_position_multiplier_curve(model)
            fig_panel = create_diagnostic_panel(model, feature_data, regions)

            for name, fig in [
                ("heatmap.png", fig_heatmap),
                ("boundaries.png", fig_boundaries),
                ("multiplier.png", fig_multiplier),
                ("panel.png", fig_panel),
            ]:
                path = Path(tmpdir) / name
                fig.savefig(str(path), dpi=100)
                assert path.exists()
                assert path.stat().st_size > 0
