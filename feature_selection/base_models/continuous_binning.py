"""Continuous quantile binning model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_selection.base_models.base_model import BinningModelBase


class ContinuousBinningModel(BinningModelBase):
    """Continuous feature model using quantile binning."""

    model_type = "continuous_binning"

    def __init__(
        self,
        n_bins: int = 15,
        selection_metric: str = "sharpe",
        strategy: str = "long",
        normalize_by: str | None = "ewsd",
        metric_threshold: float = 0.0,
        t_threshold: float = 2.0,
        min_region_width: int = 2,
        shrinkage_k: float = 20.0,
        long_clip_min: float = 0.5,
        long_clip_max: float = 2.0,
        short_clip_min: float = 0.5,
        short_clip_max: float = 2.0,
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            selection_metric=selection_metric,
            normalize_by=normalize_by,
            strategy=strategy,
            metric_threshold=metric_threshold,
            t_threshold=t_threshold,
            min_region_width=min_region_width,
            shrinkage_k=shrinkage_k,
            long_clip_min=long_clip_min,
            long_clip_max=long_clip_max,
            short_clip_min=short_clip_min,
            short_clip_max=short_clip_max,
        )

    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        """Create quantile bins with robust fallback handling."""
        try:
            bins = pd.qcut(feature_data, self.n_bins, labels=False, duplicates="drop")
            n_created_bins = int(bins.nunique())

            if n_created_bins < self.n_bins:
                unique_vals = feature_data.unique()
                n_unique = len(unique_vals)

                if n_unique <= self.n_bins:
                    sorted_unique = np.sort(unique_vals)
                    bin_map = {val: idx for idx, val in enumerate(sorted_unique)}
                    bins = feature_data.map(bin_map).astype(int)
                else:
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    unique_quantiles = np.quantile(unique_vals, quantiles)
                    unique_quantiles = np.unique(unique_quantiles)

                    if len(unique_quantiles) - 1 >= self.n_bins:
                        bins = pd.cut(
                            feature_data,
                            bins=unique_quantiles,
                            labels=False,
                            include_lowest=True,
                            duplicates="drop",
                        )
                        n_created_bins = int(bins.nunique())

                    if n_created_bins < self.n_bins:
                        bins = pd.cut(
                            feature_data,
                            bins=self.n_bins,
                            labels=False,
                            duplicates="drop",
                            include_lowest=True,
                        )
                        n_created_bins = int(bins.nunique())

                        if n_created_bins < self.n_bins:
                            min_val = float(feature_data.min())
                            max_val = float(feature_data.max())
                            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
                            bin_edges[0] = min_val - 1e-10
                            bin_edges[-1] = max_val + 1e-10
                            bins = pd.cut(
                                feature_data,
                                bins=bin_edges,
                                labels=False,
                                include_lowest=True,
                                duplicates="drop",
                            )
        except (ValueError, TypeError):
            min_val = float(feature_data.min())
            max_val = float(feature_data.max())
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            bin_edges[0] = min_val - 1e-10
            bin_edges[-1] = max_val + 1e-10
            bins = pd.cut(
                feature_data,
                bins=bin_edges,
                labels=False,
                include_lowest=True,
                duplicates="drop",
            )

        return bins.astype(float)

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"ContinuousBinningModel(n_bins={self.n_bins}, "
            f"selection_metric='{self.selection_metric}', {fitted})"
        )
