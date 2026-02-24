"""Binning diagnostics utilities for continuous feature validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from feature_selection.base_models.base_model import BinningModelBase


@dataclass(frozen=True)
class BinningSuccessCriteria:
    """Criteria used to validate whether fitted binning is usable."""

    metric_threshold: float
    t_threshold: float
    min_region_width: int


@dataclass(frozen=True)
class RegionMetadata:
    """Aggregated statistics for one contiguous significant bin region."""

    start_bin: int
    end_bin: int
    bins: list[int]
    mean_sharpe: float
    mean_t_stat: float
    sample_count: int
    feature_range: tuple[float, float]


def _require_fitted_model(model: BinningModelBase) -> None:
    if not getattr(model, "is_fitted_", False):
        raise ValueError("Binning model must be fitted before diagnostics.")

    if not isinstance(getattr(model, "bin_stats_", None), dict):
        raise ValueError("Fitted model has invalid bin_stats_ state.")
    if not isinstance(getattr(model, "significant_regions_", None), list):
        raise ValueError("Fitted model has invalid significant_regions_ state.")


def _parse_region_bins(region: dict[str, object]) -> list[int]:
    bins_obj = region.get("bins")
    if not isinstance(bins_obj, list) or len(bins_obj) == 0:
        raise ValueError("Region must include a non-empty 'bins' list.")

    bins = [int(bin_idx) for bin_idx in bins_obj]
    if bins != sorted(bins):
        raise ValueError("Region bins must be ordered.")
    if len(set(bins)) != len(bins):
        raise ValueError("Region bins must be unique.")
    if bins[-1] - bins[0] + 1 != len(bins):
        raise ValueError("Region bins must be contiguous.")
    return bins


def extract_region_metadata(model: BinningModelBase) -> list[RegionMetadata]:
    """Extract per-region metadata from a fitted binning model."""
    _require_fitted_model(model)

    region_metadata: list[RegionMetadata] = []
    for region_obj in model.significant_regions_:
        if not isinstance(region_obj, dict):
            raise ValueError("Each region must be a dictionary.")

        bins = _parse_region_bins(region_obj)
        missing = [bin_idx for bin_idx in bins if bin_idx not in model.bin_stats_]
        if missing:
            raise ValueError(f"Region references bins missing from bin_stats_: {missing}")

        sharpe_values = [float(model.bin_stats_[bin_idx]["sharpe"]) for bin_idx in bins]
        t_values = [abs(float(model.bin_stats_[bin_idx]["t_stat"])) for bin_idx in bins]
        counts = [int(model.bin_stats_[bin_idx]["count"]) for bin_idx in bins]
        feature_mins = [float(model.bin_stats_[bin_idx]["feature_min"]) for bin_idx in bins]
        feature_maxs = [float(model.bin_stats_[bin_idx]["feature_max"]) for bin_idx in bins]

        region_metadata.append(
            RegionMetadata(
                start_bin=int(bins[0]),
                end_bin=int(bins[-1]),
                bins=bins,
                mean_sharpe=float(sum(sharpe_values) / len(sharpe_values)),
                mean_t_stat=float(sum(t_values) / len(t_values)),
                sample_count=int(sum(counts)),
                feature_range=(float(min(feature_mins)), float(max(feature_maxs))),
            )
        )

    return sorted(region_metadata, key=lambda region: region.start_bin)


def validate_binning_success(model: BinningModelBase, criteria: BinningSuccessCriteria) -> bool:
    """Return True when at least one contiguous region meets all criteria."""
    _require_fitted_model(model)
    regions = extract_region_metadata(model)
    if not regions:
        return False

    metric_threshold = float(criteria.metric_threshold)
    t_threshold = float(criteria.t_threshold)
    min_width = int(criteria.min_region_width)

    for region in regions:
        if len(region.bins) < min_width:
            continue

        is_valid = True
        for bin_idx in region.bins:
            stat = model.bin_stats_[bin_idx]
            metric_long = float(stat.get("selection_metric_long", float("-inf")))
            metric_short = float(stat.get("selection_metric_short", float("-inf")))
            metric_ok = metric_long >= metric_threshold or metric_short >= metric_threshold
            t_ok = abs(float(stat["t_stat"])) >= t_threshold
            if not (metric_ok and t_ok):
                is_valid = False
                break

        if is_valid:
            return True

    return False


def detect_region_shape(region: RegionMetadata, n_bins: int) -> Literal["tail", "hump"]:
    """Classify region shape based on whether the region touches extreme bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if region.start_bin < 0 or region.end_bin < region.start_bin:
        raise ValueError("Region bin bounds are invalid.")
    if region.end_bin >= n_bins:
        raise ValueError("Region end_bin cannot exceed n_bins - 1.")

    if region.start_bin == 0 or region.end_bin == n_bins - 1:
        return "tail"
    return "hump"


def calculate_coverage(regions: list[RegionMetadata], feature_data: pd.Series) -> float:
    """Calculate percent of feature values covered by tradeable region ranges."""
    clean_feature = feature_data.dropna()
    if clean_feature.empty or not regions:
        return 0.0

    covered = pd.Series(False, index=clean_feature.index)
    for region in regions:
        low, high = region.feature_range
        if low > high:
            raise ValueError("Region feature_range must have low <= high.")
        covered = covered | ((clean_feature >= low) & (clean_feature <= high))

    coverage = float(covered.sum()) / float(len(clean_feature)) * 100.0
    return max(0.0, min(100.0, coverage))
