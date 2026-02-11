"""
Rule-Based Binning Model

No-op binning for rule-based bias nodes. Outputs the feature value as-is
(1 / 0 / -1 or other values from the node) with no binning or threshold logic.

Author: Trading Research Team
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from feature_selection.base_models.base_model import BinningModelBase


class RuleBasedBinningModel(BinningModelBase):
    """
    Pass-through "binning" model for rule-based bias nodes.

    Performs no binning. fit() only stores metadata; predict() returns the
    feature values unchanged. Node outputs (1, 0, -1) are already signed
    position fractions; they are used as-is. Strategy ('long', 'short', or
    'long_short') is accepted for API consistency but ignored (no negation).

    Use with rule-based nodes (e.g. TurtleTrading, DonchianChannel)
    whose outputs are already signed discrete signals.
    """

    def __init__(
        self,
        selection_metric: str = "sortino",
        strategy: str = "long",
        normalize_by: Optional[str] = None,
    ):
        super().__init__(
            n_bins=1,
            selection_metric=selection_metric,
            normalize_by=normalize_by,
            strategy=strategy,
        )

    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        """Not used; rule-based has no bins. Return zeros to satisfy ABC."""
        return pd.Series(0, index=feature_data.index)

    def _extract_thresholds(self, df: pd.DataFrame, n_bins: int) -> np.ndarray:
        """No thresholds. Return empty array."""
        return np.array([])

    def fit(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        normalization_data: Optional[pd.Series] = None,
    ) -> "RuleBasedBinningModel":
        """Store feature column and mark fitted; no binning."""
        if feature_data.name is None:
            raise ValueError(
                "feature_data must have a name attribute. "
                "Pass a named pd.Series (e.g., df['column_name'])."
            )
        self.feature_column = feature_data.name
        self.normalization_data_ = normalization_data
        self.thresholds_ = np.array([])
        self.best_long_bin_ = 0
        self.best_short_bin_ = 0
        self.bin_stats_ = {}
        self.is_fitted_ = True
        return self

    def predict(
        self,
        feature_data: pd.Series,
        strategy: str = "long",
        normalization_data: Optional[pd.Series] = None,
        scaled: bool = False,
    ) -> pd.Series:
        """Return feature values as-is (signed position: 1, 0, -1). No binning or negation."""
        # Lazy fit: rule-based has no learned state; mark fitted on first predict so
        # vault-loaded models work without an explicit fit step.
        if not self.is_fitted_:
            if feature_data.name is not None:
                self.feature_column = feature_data.name
            self.normalization_data_ = getattr(self, "normalization_data_", None)
            self.thresholds_ = np.array([])
            self.best_long_bin_ = 0
            self.best_short_bin_ = 0
            self.bin_stats_ = {}
            self.is_fitted_ = True

        out = feature_data.astype(float)

        if not scaled or self.normalize_by is None:
            return out

        if normalization_data is None:
            raise ValueError(
                f"scaled=True and normalize_by='{self.normalize_by}' but no normalization_data provided."
            )
        aligned = out.reindex(normalization_data.index, fill_value=0)
        min_vol = 1e-8 if self.normalize_by == "ewsd" else 1.0
        safe_vol = normalization_data.clip(lower=min_vol)
        return aligned / safe_vol

    def get_params(self, deep: bool = True) -> dict:
        """Return constructor params for vault/serialization."""
        return {
            "n_bins": self.n_bins,
            "selection_metric": self.selection_metric,
            "normalize_by": self.normalize_by,
            "strategy": self.strategy,
        }

    def set_params(self, **params: object) -> "RuleBasedBinningModel":
        """Set constructor params; reset fitted state if changed."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.is_fitted_ = False
        self.thresholds_ = None
        self.best_long_bin_ = None
        self.best_short_bin_ = None
        self.bin_stats_ = None
        return self

    def get_bin_stats(self) -> Dict:
        """No bins; return empty dict."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling get_bin_stats()")
        return self.bin_stats_
