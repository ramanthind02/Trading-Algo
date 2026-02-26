"""Unit tests for binning diagnostics report generation (T008)."""

from __future__ import annotations

import json
import tempfile
from io import StringIO
from unittest.mock import patch

import pandas as pd
import pytest
from matplotlib.figure import Figure

from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validation.binning.diagnostics import (
    BinningSuccessCriteria,
    RegionMetadata,
)
from feature_selection.validation.binning.report import (
    BinningDiagnosticsReport,
    detect_failure_mode,
    display_report_summary,
    generate_binning_report,
    save_report,
)
from feature_selection.validation.binning.shape_analysis import (
    AdjacencyAnalysis,
    RegionCoverage,
)


# ---------------------------------------------------------------------------
# Mock helpers (temporary until T006/T007 are implemented)
# ---------------------------------------------------------------------------

def _mock_shape_summary() -> dict[str, int]:
    return {"long_tail": 1, "short_tail": 1}


def _mock_coverage_breakdown() -> list[RegionCoverage]:
    return [
        RegionCoverage(region_id=0, individual_coverage_pct=20.0, cumulative_coverage_pct=20.0),
        RegionCoverage(region_id=1, individual_coverage_pct=15.0, cumulative_coverage_pct=35.0),
    ]


def _mock_adjacency_analysis() -> AdjacencyAnalysis:
    return AdjacencyAnalysis(gap_sizes=[5, 3], is_connected=False, isolation_score=0.25)


def _mock_diagnostic_plots() -> dict[str, Figure]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, _ = plt.subplots()
    return {"heatmap": fig, "boundaries": fig, "multiplier_curve": fig, "panel": fig}


# ---------------------------------------------------------------------------
# Model stubs
# ---------------------------------------------------------------------------

def _build_model_with_valid_regions() -> ContinuousBinningModel:
    """Model with regions that pass all criteria."""
    model = ContinuousBinningModel(n_bins=5)
    model.is_fitted_ = True
    model.feature_column = "rsi_signal_D_lookback_14"
    model.bin_stats_ = {
        0: {
            "selection_metric_long": 0.1, "selection_metric_short": -0.1,
            "t_stat": 0.4, "sharpe": 0.1, "count": 50,
            "feature_min": 0.0, "feature_max": 20.0,
        },
        1: {
            "selection_metric_long": 0.8, "selection_metric_short": -0.8,
            "t_stat": 2.5, "sharpe": 1.2, "count": 60,
            "feature_min": 20.0, "feature_max": 40.0,
        },
        2: {
            "selection_metric_long": 0.7, "selection_metric_short": -0.7,
            "t_stat": 2.2, "sharpe": 1.0, "count": 55,
            "feature_min": 40.0, "feature_max": 60.0,
        },
        3: {
            "selection_metric_long": 0.2, "selection_metric_short": -0.2,
            "t_stat": 0.8, "sharpe": 0.2, "count": 65,
            "feature_min": 60.0, "feature_max": 80.0,
        },
    }
    model.significant_regions_ = [{"start_bin": 1, "end_bin": 2, "bins": [1, 2]}]
    return model


def _build_model_no_regions() -> ContinuousBinningModel:
    """Model with empty significant regions."""
    model = _build_model_with_valid_regions()
    model.significant_regions_ = []
    return model


def _build_model_isolated_spikes() -> ContinuousBinningModel:
    """Model with only single-bin spikes (width < min_region_width=2)."""
    model = _build_model_with_valid_regions()
    model.significant_regions_ = [
        {"start_bin": 1, "end_bin": 1, "bins": [1]},
        {"start_bin": 3, "end_bin": 3, "bins": [3]},
    ]
    return model


def _build_model_insufficient_edge() -> ContinuousBinningModel:
    """Model with wide regions but metrics below threshold."""
    model = ContinuousBinningModel(n_bins=5)
    model.is_fitted_ = True
    model.feature_column = "rsi_signal_D_lookback_14"
    model.bin_stats_ = {
        0: {
            "selection_metric_long": 0.1, "selection_metric_short": -0.1,
            "t_stat": 1.0, "sharpe": 0.1, "count": 50,
            "feature_min": 0.0, "feature_max": 20.0,
        },
        1: {
            "selection_metric_long": 0.3, "selection_metric_short": -0.3,
            "t_stat": 1.5, "sharpe": 0.3, "count": 60,
            "feature_min": 20.0, "feature_max": 40.0,
        },
        2: {
            "selection_metric_long": 0.2, "selection_metric_short": -0.2,
            "t_stat": 1.2, "sharpe": 0.2, "count": 55,
            "feature_min": 40.0, "feature_max": 60.0,
        },
    }
    # Wide region but t_stats all below threshold (2.0)
    model.significant_regions_ = [{"start_bin": 0, "end_bin": 2, "bins": [0, 1, 2]}]
    return model


def _build_synthetic_report(
    success: bool = True,
    failure_mode: str = "none",
) -> BinningDiagnosticsReport:
    """Build a synthetic report for testing serialization and display."""
    criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)
    regions = [
        RegionMetadata(
            start_bin=1, end_bin=2, bins=[1, 2],
            mean_sharpe=1.1, mean_t_stat=2.35,
            sample_count=115, feature_range=(20.0, 60.0),
        ),
    ]
    coverage = _mock_coverage_breakdown()
    adjacency = _mock_adjacency_analysis()

    return BinningDiagnosticsReport(
        feature_column="rsi_signal_D_lookback_14",
        parameter_combo={"lookback": 14, "n_bins": 5},
        success_verdict=success,
        failure_mode=failure_mode,
        criteria=criteria,
        regions=regions,
        shape_summary=_mock_shape_summary(),
        coverage_breakdown=coverage,
        total_coverage_pct=35.0,
        adjacency_analysis=adjacency,
        timestamp="2026-02-16T12:00:00",
    )


# ---------------------------------------------------------------------------
# Tests: generate_binning_report
# ---------------------------------------------------------------------------

class TestGenerateBinningReport:
    """Tests for generate_binning_report orchestration."""

    def test_generate_report_success_verdict(self) -> None:
        """Synthetic model with valid regions produces success_verdict=True."""
        model = _build_model_with_valid_regions()
        criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)
        feature_data = pd.Series(range(100), dtype=float, name="rsi_signal_D_lookback_14")

        with (
            patch(
                "feature_selection.validation.binning.report.analyze_multi_region_shapes",
                return_value=_mock_shape_summary(),
            ),
            patch(
                "feature_selection.validation.binning.report.calculate_region_coverage_breakdown",
                return_value=_mock_coverage_breakdown(),
            ),
            patch(
                "feature_selection.validation.binning.report.detect_region_adjacency",
                return_value=_mock_adjacency_analysis(),
            ),
            patch(
                "feature_selection.validation.binning.report.plot_bin_heatmap",
                return_value=_mock_diagnostic_plots()["heatmap"],
            ),
            patch(
                "feature_selection.validation.binning.report.plot_region_boundaries",
                return_value=_mock_diagnostic_plots()["boundaries"],
            ),
            patch(
                "feature_selection.validation.binning.report.plot_position_multiplier_curve",
                return_value=_mock_diagnostic_plots()["multiplier_curve"],
            ),
            patch(
                "feature_selection.validation.binning.report.create_diagnostic_panel",
                return_value=_mock_diagnostic_plots()["panel"],
            ),
        ):
            report, _ = generate_binning_report(model, feature_data, criteria, strategy="long")

        assert report.success_verdict is True
        assert report.failure_mode == "none"
        assert report.feature_column == "rsi_signal_D_lookback_14"
        assert len(report.regions) > 0

    def test_generate_report_no_regions_verdict(self) -> None:
        """Model with empty significant_regions produces failure."""
        model = _build_model_no_regions()
        criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)
        feature_data = pd.Series(range(100), dtype=float, name="rsi")

        with (
            patch(
                "feature_selection.validation.binning.report.analyze_multi_region_shapes",
                return_value={},
            ),
            patch(
                "feature_selection.validation.binning.report.calculate_region_coverage_breakdown",
                return_value=[],
            ),
            patch(
                "feature_selection.validation.binning.report.detect_region_adjacency",
                return_value=AdjacencyAnalysis(gap_sizes=[], is_connected=True, isolation_score=0.0),
            ),
            patch(
                "feature_selection.validation.binning.report.plot_bin_heatmap",
                return_value=_mock_diagnostic_plots()["heatmap"],
            ),
            patch(
                "feature_selection.validation.binning.report.plot_region_boundaries",
                return_value=_mock_diagnostic_plots()["boundaries"],
            ),
            patch(
                "feature_selection.validation.binning.report.plot_position_multiplier_curve",
                return_value=_mock_diagnostic_plots()["multiplier_curve"],
            ),
            patch(
                "feature_selection.validation.binning.report.create_diagnostic_panel",
                return_value=_mock_diagnostic_plots()["panel"],
            ),
        ):
            report, _ = generate_binning_report(model, feature_data, criteria)

        assert report.success_verdict is False
        assert report.failure_mode == "no_regions"


# ---------------------------------------------------------------------------
# Tests: detect_failure_mode
# ---------------------------------------------------------------------------

class TestDetectFailureMode:
    """Tests for failure mode classification."""

    def test_detect_failure_mode_no_regions(self) -> None:
        model = _build_model_no_regions()
        criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)
        assert detect_failure_mode(model, criteria) == "no_regions"

    def test_detect_failure_mode_isolated_spikes(self) -> None:
        model = _build_model_isolated_spikes()
        criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)
        assert detect_failure_mode(model, criteria) == "isolated_spikes"

    def test_detect_failure_mode_insufficient_edge(self) -> None:
        model = _build_model_insufficient_edge()
        criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)
        assert detect_failure_mode(model, criteria) == "insufficient_edge"

    def test_detect_failure_mode_none(self) -> None:
        model = _build_model_with_valid_regions()
        criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)
        assert detect_failure_mode(model, criteria) == "none"


# ---------------------------------------------------------------------------
# Tests: BinningDiagnosticsReport dataclass
# ---------------------------------------------------------------------------

class TestReportDataclass:
    """Tests for BinningDiagnosticsReport structure."""

    def test_report_dataclass_fields(self) -> None:
        """All required fields are present and types are correct."""
        report = _build_synthetic_report()

        assert isinstance(report.feature_column, str)
        assert isinstance(report.parameter_combo, dict)
        assert isinstance(report.success_verdict, bool)
        assert isinstance(report.failure_mode, str)
        assert isinstance(report.criteria, BinningSuccessCriteria)
        assert isinstance(report.regions, list)
        assert isinstance(report.shape_summary, dict)
        assert isinstance(report.coverage_breakdown, list)
        assert isinstance(report.total_coverage_pct, float)
        assert isinstance(report.adjacency_analysis, AdjacencyAnalysis)
        assert isinstance(report.timestamp, str)

    def test_report_is_frozen(self) -> None:
        """Frozen dataclass prevents mutation."""
        report = _build_synthetic_report()
        with pytest.raises(AttributeError):
            report.success_verdict = False  # type: ignore[misc]

    def test_report_parameter_combo_stored(self) -> None:
        """Parameter combo is stored correctly."""
        report = _build_synthetic_report()
        assert report.parameter_combo == {"lookback": 14, "n_bins": 5}


# ---------------------------------------------------------------------------
# Tests: save_report
# ---------------------------------------------------------------------------

class TestSaveReport:
    """Tests for JSON serialization and plot saving."""

    def test_save_report_creates_json(self) -> None:
        """save_report creates a JSON file at expected path."""
        report = _build_synthetic_report()
        plots = _mock_diagnostic_plots()
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = save_report(report, plots, tmpdir)
            assert result_path.endswith(".json")
            from pathlib import Path
            assert Path(result_path).exists()

    def test_save_report_json_fields(self) -> None:
        """Saved JSON contains all non-Figure fields with correct values."""
        report = _build_synthetic_report()
        plots = _mock_diagnostic_plots()
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = save_report(report, plots, tmpdir)
            with open(result_path) as f:
                data = json.load(f)

        assert data["feature_column"] == "rsi_signal_D_lookback_14"
        assert data["success_verdict"] is True
        assert data["failure_mode"] == "none"
        assert data["timestamp"] == "2026-02-16T12:00:00"
        assert data["total_coverage_pct"] == 35.0
        assert data["parameter_combo"] == {"lookback": 14, "n_bins": 5}

        # Plots should be paths, not Figure objects
        assert isinstance(data["diagnostic_plots"], dict)
        for plot_name in ("heatmap", "boundaries", "multiplier_curve", "panel"):
            assert plot_name in data["diagnostic_plots"]
            assert data["diagnostic_plots"][plot_name].endswith(".png")

        # Criteria preserved
        assert data["criteria"]["metric_threshold"] == 0.5
        assert data["criteria"]["t_threshold"] == 2.0
        assert data["criteria"]["min_region_width"] == 2

        # Regions preserved
        assert len(data["regions"]) == 1
        assert data["regions"][0]["start_bin"] == 1

    def test_save_report_creates_plot_files(self) -> None:
        """Plot PNG files are saved alongside JSON."""
        report = _build_synthetic_report()
        plots = _mock_diagnostic_plots()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_report(report, plots, tmpdir)
            from pathlib import Path
            plots_dir = Path(tmpdir) / report.feature_column / "plots"
            assert plots_dir.exists()
            png_files = list(plots_dir.glob("*.png"))
            assert len(png_files) == 4


# ---------------------------------------------------------------------------
# Tests: display_report_summary
# ---------------------------------------------------------------------------

class TestDisplayReportSummary:
    """Tests for terminal output formatting."""

    def test_display_report_summary_no_exception(self, capsys: pytest.CaptureFixture[str]) -> None:
        """display_report_summary runs without exception and prints key terms."""
        report = _build_synthetic_report()
        display_report_summary(report)
        captured = capsys.readouterr().out

        assert "Verdict" in captured or "verdict" in captured
        assert "Coverage" in captured or "coverage" in captured
        assert "Regions" in captured or "regions" in captured
        assert "rsi_signal_D_lookback_14" in captured

    def test_display_report_summary_fail_verdict(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Failed verdict displays FAIL."""
        report = _build_synthetic_report(success=False, failure_mode="no_regions")
        display_report_summary(report)
        captured = capsys.readouterr().out
        assert "FAIL" in captured


# ---------------------------------------------------------------------------
# Tests: timestamp format
# ---------------------------------------------------------------------------

class TestTimestamp:
    """Tests for ISO 8601 timestamp."""

    def test_timestamp_iso_format(self) -> None:
        """Timestamp matches ISO 8601 format."""
        report = _build_synthetic_report()
        # YYYY-MM-DDTHH:MM:SS
        from datetime import datetime
        parsed = datetime.fromisoformat(report.timestamp)
        assert parsed.year == 2026
