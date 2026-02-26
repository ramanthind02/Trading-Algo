"""Rule-based model for discrete -1/0/+1 features."""

from __future__ import annotations

import pandas as pd

from feature_selection.base_models.base_model import BinningModelBase


class RuleBasedModel(BinningModelBase):
    """Rule-based model with per-level statistics and multipliers."""

    model_type = "rule_based"

    def __init__(
        self,
        selection_metric: str = "sharpe",
        strategy: str = "long",
        normalize_by: str | None = None,
        metric_threshold: float = 0.0,
        shrinkage_k: float = 20.0,
        long_clip_min: float = 0.5,
        long_clip_max: float = 2.0,
        short_clip_min: float = 0.5,
        short_clip_max: float = 2.0,
        bin_counts: list[int] | None = None,  # Ignored (always 3 bins)
        use_coverage_bonus: bool = False,  # Ignored
        coverage_bonus_per_10pct: float = 0.02,  # Ignored
        max_coverage_bonus: float = 0.2,  # Ignored
    ) -> None:
        """Initialize rule-based model.

        Note: bin_counts and coverage bonus parameters are accepted for API
        consistency but ignored. RuleBasedModel always uses 3 bins {-1, 0, 1}.
        """
        # Store for API consistency, but override in parent
        self.bin_counts = [3]
        self.use_coverage_bonus = False

        super().__init__(
            n_bins=3,
            selection_metric=selection_metric,
            normalize_by=normalize_by,
            strategy=strategy,
            metric_threshold=metric_threshold,
            t_threshold=0.0,
            min_region_width=1,
            shrinkage_k=shrinkage_k,
            long_clip_min=long_clip_min,
            long_clip_max=long_clip_max,
            short_clip_min=short_clip_min,
            short_clip_max=short_clip_max,
        )

    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        mapping = {-1: 0, 0: 1, 1: 2}
        missing = feature_data[~feature_data.isin(mapping.keys())]
        if not missing.empty:
            bad_values = sorted(set(missing.astype(float).tolist()))
            raise ValueError(
                f"rule_based expects feature values in {{-1, 0, 1}}; got {bad_values}"
            )
        return feature_data.map(mapping).astype(float)

    def _assign_bins(self, feature_data: pd.Series) -> pd.Series:
        mapping = {-1: 0, 0: 1, 1: 2}
        missing = feature_data[~feature_data.isin(mapping.keys())]
        if not missing.empty:
            bad_values = sorted(set(missing.astype(float).tolist()))
            raise ValueError(
                f"rule_based expects feature values in {{-1, 0, 1}}; got {bad_values}"
            )
        return feature_data.map(mapping).astype(int)

    def _predict_bin_key(self, bin_idx: int) -> int:
        """Rule-based uses 0-based bin indices from _assign_bins; no conversion."""
        return bin_idx

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted_ else "unfitted"
        return f"RuleBasedModel(selection_metric='{self.selection_metric}', {fitted})"
