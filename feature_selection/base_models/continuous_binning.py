"""Continuous quantile binning model with grid search over bin counts.

Outputs binary signals: 1 in selected long bin, -1 in selected short bin, 0 elsewhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_selection.base_models.base_model import BinningModelBase
from utils.core.enums import Direction, DirectionInput


class ContinuousBinningModel(BinningModelBase):
    """Continuous feature model using quantile binning with grid search.

    Selects the best bin per direction by t-statistic; predict() returns
    binary values (0, 1 for long; 0, -1 for short).
    """

    model_type = "continuous_binning"

    def __init__(
        self,
        n_bins: int = 15,
        bin_counts: list[int] | None = None,
        strategy: DirectionInput = Direction.LONG,
        bin_index_min: int = 0,
        bin_index_max: int | None = None,
    ) -> None:
        """Initialize continuous binning model.

        Parameters
        ----------
        n_bins : int
            Default bin count (used if bin_counts is None).
        bin_counts : list[int] | None
            List of bin counts to test in grid search. If None, uses [n_bins].
        strategy : DirectionInput
            Direction enum or canonical direction string.
        bin_index_min : int
            Minimum bin index to consider for selection (inclusive). Default 0.
        bin_index_max : int | None
            Maximum bin index to consider (inclusive). None means no cap (all bins).
        """
        self.bin_counts = bin_counts if bin_counts is not None else [n_bins]
        self.bin_index_min = bin_index_min
        self.bin_index_max = bin_index_max

        super().__init__(
            n_bins=n_bins,
            selection_metric="t_stat",
            normalize_by="ewsd",
            strategy=strategy,
            metric_threshold=0.0,
            t_threshold=2.0,
            min_region_width=2,
            shrinkage_k=20.0,
            long_clip_min=1.0,
            long_clip_max=1.0,
            short_clip_min=1.0,
            short_clip_max=1.0,
        )

    def clone(self) -> ContinuousBinningModel:
        """Return a new unfitted instance with the same constructor arguments."""
        return ContinuousBinningModel(
            n_bins=self.n_bins,
            bin_counts=list(self.bin_counts),
            strategy=self.strategy,
            bin_index_min=self.bin_index_min,
            bin_index_max=self.bin_index_max,
        )

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Expose constructor parameters for vault persistence (only accepted by __init__)."""
        return {
            "n_bins": self.n_bins,
            "bin_counts": list(self.bin_counts),
            "strategy": self.strategy.value,
            "bin_index_min": self.bin_index_min,
            "bin_index_max": self.bin_index_max,
        }

    def fit(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        normalization_data: pd.Series | None = None,
    ) -> None:
        """Fit model with grid search over bin counts.

        Parameters
        ----------
        feature_data : pd.Series
            Input feature values used to form quantile bins.
        target_data : pd.Series
            Prediction target aligned to ``feature_data``. When fitting on
            concatenated data from multiple tickers, this target must be
            volatility-normalized (for example ``log_return_ewsd``). Raw return targets across tickers with
            different volatility will bias quantile bin selection.
        normalization_data : pd.Series | None
            Optional normalization series consumed by shared base behaviors.

        Tests each bin count in self.bin_counts, selects best bin per direction
        based on t-statistic, and stores the winning configuration.
        """
        # Pre-fit validation
        self._reset_fitted_state()

        if not isinstance(feature_data, pd.Series):
            msg = f"feature_data must be pd.Series, got {type(feature_data)}"
            raise TypeError(msg)
        if not isinstance(target_data, pd.Series):
            msg = f"target_data must be pd.Series, got {type(target_data)}"
            raise TypeError(msg)
        if feature_data.name is None:
            msg = "feature_data must have a name attribute (pd.Series.name)"
            raise ValueError(msg)

        self.feature_column = feature_data.name
        self._training_feature_data = feature_data.copy()
        self._training_target_data = target_data.copy()

        df = pd.DataFrame({"feature": feature_data, "target": target_data}).dropna()

        if len(df) < max(10, self.n_bins * 5):
            msg = f"Insufficient data after dropna: {len(df)} rows"
            raise ValueError(msg)

        # Grid search over bin counts
        original_n_bins = self.n_bins
        best_result: dict[str, object] | None = None

        for n in self.bin_counts:
            try:
                self.n_bins = n
                bins = self._create_bins(df["feature"], df["target"])
            finally:
                # Ensure n_bins is not left in a partial state if _create_bins raises.
                self.n_bins = original_n_bins
            df_copy = df.copy()
            df_copy["bin"] = bins
            ordered_bins = sorted(df_copy["bin"].unique())

            bin_stats = self._calculate_bin_stats(df_copy, ordered_bins)

            # Restrict to allowed bin index range when bin_index_max is set
            if self.bin_index_max is None:
                allowed_bins = list(ordered_bins)
            else:
                allowed_bins = [
                    b for b in ordered_bins
                    if self.bin_index_min <= int(b) <= self.bin_index_max
                ]

            # Select best bin per direction by t-stat
            best_long_bin, best_short_bin = self._select_best_bins(bin_stats, allowed_bins)

            if best_long_bin is not None:
                long_t_stat = bin_stats[best_long_bin]["t_stat"]
            else:
                long_t_stat = float("-inf")

            if best_short_bin is not None:
                short_t_stat = bin_stats[best_short_bin]["t_stat"]
            else:
                short_t_stat = float("-inf")

            # Track best result by highest t-stat (sign-aware per strategy)
            if self.strategy == Direction.LONG:
                score = long_t_stat
            elif self.strategy == Direction.SHORT:
                score = -short_t_stat  # more negative short t-stat -> higher score
            else:  # long_short
                score = max(long_t_stat, -short_t_stat)

            if best_result is None or score > best_result["score"]:  # type: ignore[index]
                best_result = {
                    "n_bins": n,
                    "score": score,
                    "bins": bins,
                    "ordered_bins": ordered_bins,
                    "bin_stats": bin_stats,
                    "best_long_bin": best_long_bin,
                    "best_short_bin": best_short_bin,
                }

        if best_result is None:
            raise ValueError(
                "No bin counts to evaluate. Provide at least one value in bin_counts."
            )

        # Reject long-only fit when no profitable long bin exists
        if self.strategy == Direction.LONG and best_result["best_long_bin"] is None:
            raise ValueError(
                "No long bin with positive t-stat; cannot fit long-only model."
            )

        # Set fitted state from best result
        self.n_bins = int(best_result["n_bins"])
        df["bin"] = best_result["bins"]
        self.bin_edges_ = self._extract_bin_edges(df, best_result["ordered_bins"])
        self.bin_stats_ = best_result["bin_stats"]

        # Store selected bins (binary output: 1 / -1 in predict)
        self.selected_bins_ = {
            "long": best_result["best_long_bin"],
            "short": best_result["best_short_bin"],
        }

        # Build position multipliers for selected bins only
        self._build_position_multipliers_from_selected_bins()

        # Legacy: keep significant_regions_ empty for backwards compat
        self.significant_regions_ = []

        # Store fit config
        self.fit_config_ = {
            "n_bins": self.n_bins,
            "strategy": self.strategy.value,
            "bin_counts": self.bin_counts,
            "bin_index_min": self.bin_index_min,
            "bin_index_max": self.bin_index_max,
        }

        self.is_fitted_ = True

    def _select_best_bins(
        self, bin_stats: dict[int, dict], ordered_bins: list[int]
    ) -> tuple[int | None, int | None]:
        """Select best bin per direction based on t-statistic.

        Returns
        -------
        tuple[int | None, int | None]
            (best_long_bin, best_short_bin)
        """
        # Long: highest positive t-stat
        long_candidates = [b for b in ordered_bins if bin_stats[b]["t_stat"] > 0]
        best_long_bin = (
            max(long_candidates, key=lambda b: bin_stats[b]["t_stat"])
            if long_candidates
            else None
        )

        # Short: most negative t-stat
        short_candidates = [b for b in ordered_bins if bin_stats[b]["t_stat"] < 0]
        best_short_bin = (
            min(short_candidates, key=lambda b: bin_stats[b]["t_stat"])
            if short_candidates
            else None
        )

        return best_long_bin, best_short_bin

    def _build_position_multipliers_from_selected_bins(self) -> None:
        """Build binary position multipliers (1 / -1) from selected bins."""
        multipliers_long: dict[int, float] = {}
        multipliers_short: dict[int, float] = {}
        multipliers_long_short: dict[int, float] = {}

        if self.selected_bins_["long"] is not None:
            bin_idx = self.selected_bins_["long"]
            multipliers_long[bin_idx] = 1.0
            multipliers_long_short[bin_idx] = 1.0

        if self.selected_bins_["short"] is not None:
            bin_idx = self.selected_bins_["short"]
            multipliers_short[bin_idx] = -1.0
            multipliers_long_short[bin_idx] = -1.0

        self.position_multipliers_by_strategy_ = {
            "long": multipliers_long,
            "short": multipliers_short,
            "long_short": multipliers_long_short,
        }
        self.active_bins_by_strategy_ = {
            "long": list(multipliers_long.keys()),
            "short": list(multipliers_short.keys()),
            "long_short": list(multipliers_long_short.keys()),
        }

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
