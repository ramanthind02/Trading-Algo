"""Enhanced shape detection and coverage analysis for binning diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from feature_selection.validation.binning.diagnostics import RegionMetadata


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
    direction: Literal["long", "short"] = "long" if region.mean_sharpe > 0 else "short"
    touches_extreme = region.start_bin == 0 or region.end_bin == n_bins - 1
    is_monotonic = touches_extreme

    shape_base: Literal["tail", "hump"] = "tail" if touches_extreme else "hump"
    shape_type = f"{direction}_{shape_base}"

    return ShapeClassification(
        shape_type=shape_type,  # type: ignore[arg-type]
        is_monotonic=is_monotonic,
        touches_extreme=touches_extreme,
        direction=direction,
    )


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
    classifications = [classify_region_shape(r, n_bins) for r in regions]
    return dict(Counter(c.shape_type for c in classifications))


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
    clean_data = feature_data.dropna()
    if clean_data.empty or not regions:
        return []

    n = len(clean_data)
    cumulative = 0.0
    coverages: list[RegionCoverage] = []

    for idx, region in enumerate(regions):
        low, high = region.feature_range
        count = int(((clean_data >= low) & (clean_data <= high)).sum())
        individual_pct = count / n * 100.0
        cumulative += individual_pct
        coverages.append(
            RegionCoverage(
                region_id=idx,
                individual_coverage_pct=individual_pct,
                cumulative_coverage_pct=cumulative,
            )
        )

    return coverages


def detect_region_adjacency(regions: list[RegionMetadata]) -> AdjacencyAnalysis:
    """Analyze gaps between regions and compute connectivity metrics.

    Args:
        regions: List of region metadata (assumed sorted by start_bin)

    Returns:
        Adjacency analysis with gap sizes, connectivity flag, isolation score
    """
    if len(regions) <= 1:
        return AdjacencyAnalysis(gap_sizes=[], is_connected=True, isolation_score=0.0)

    sorted_regions = sorted(regions, key=lambda r: r.start_bin)
    gap_sizes = [
        sorted_regions[i + 1].start_bin - sorted_regions[i].end_bin - 1
        for i in range(len(sorted_regions) - 1)
    ]

    is_connected = all(gap <= 2 for gap in gap_sizes)

    total_span = sorted_regions[-1].end_bin - sorted_regions[0].start_bin + 1
    mean_gap = sum(gap_sizes) / len(gap_sizes) if gap_sizes else 0.0
    isolation_score = min(mean_gap / total_span, 1.0) if total_span > 0 else 0.0

    return AdjacencyAnalysis(
        gap_sizes=gap_sizes,
        is_connected=is_connected,
        isolation_score=isolation_score,
    )
