"""Enhanced shape detection and coverage analysis for binning diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from feature_selection.validators.binning.diagnostics import RegionMetadata


@dataclass(frozen=True)
class ShapeClassification:
    """Directional shape classification for a binning region."""

    shape_type: Literal["long_tail", "short_tail", "long_hump", "short_hump"]
    is_monotonic: bool
    touches_extreme: bool
    direction: Literal["long", "short"]


@dataclass(frozen=True)
class RegionCoverage:
    """Per-region and cumulative coverage percentages."""

    region_id: int
    individual_coverage_pct: float  # [0.0, 100.0]
    cumulative_coverage_pct: float  # [0.0, 100.0]


@dataclass(frozen=True)
class AdjacencyAnalysis:
    """Gap detection and connectivity metrics between regions."""

    gap_sizes: list[int]
    is_connected: bool
    isolation_score: float  # [0.0, 1.0]


def classify_region_shape(region: RegionMetadata, n_bins: int) -> ShapeClassification:
    """Classify region shape with directional info.

    Args:
        region: Region metadata from fitted binning model
        n_bins: Total number of bins in the model

    Returns:
        Shape classification with type, monotonicity, extremes, direction
    """
    raise NotImplementedError("Agent shape-analyzer will implement")


def analyze_multi_region_shapes(
    regions: list[RegionMetadata], n_bins: int
) -> dict[str, int]:
    """Return count of each shape_type across all regions.

    Args:
        regions: List of region metadata
        n_bins: Total number of bins

    Returns:
        Dictionary mapping shape_type to count
    """
    raise NotImplementedError("Agent shape-analyzer will implement")


def calculate_region_coverage_breakdown(
    regions: list[RegionMetadata], feature_data: pd.Series
) -> list[RegionCoverage]:
    """Calculate per-region and cumulative coverage percentages.

    Args:
        regions: List of region metadata
        feature_data: Original feature values

    Returns:
        List of RegionCoverage with individual and cumulative percentages
    """
    raise NotImplementedError("Agent shape-analyzer will implement")


def detect_region_adjacency(regions: list[RegionMetadata]) -> AdjacencyAnalysis:
    """Analyze gaps between regions and compute connectivity metrics.

    Args:
        regions: List of region metadata (assumed sorted by start_bin)

    Returns:
        Adjacency analysis with gap sizes, connectivity flag, isolation score
    """
    raise NotImplementedError("Agent shape-analyzer will implement")
