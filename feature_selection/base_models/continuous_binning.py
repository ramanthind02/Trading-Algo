"""Continuous quantile binning model with grid search over bin counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_selection.base_models.base_model import BinningModelBase


class ContinuousBinningModel(BinningModelBase):
    """Continuous feature model using quantile binning with grid search.

    This model tests multiple bin counts and selects the single best bin per
    direction based on t-statistic. Supports coverage-adjusted Sharpe output.
    """

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
        bin_counts: list[int] | None = None,
        use_coverage_bonus: bool = False,
        coverage_bonus_per_10pct: float = 0.02,
        max_coverage_bonus: float = 0.2,
    ) -> None:
        """Initialize continuous binning model.

        Parameters
        ----------
        n_bins : int
            Default bin count (used if bin_counts is None).
        bin_counts : list[int] | None
            List of bin counts to test in grid search. If None, uses [n_bins].
        use_coverage_bonus : bool
            Whether to add coverage bonus to Sharpe ratio.
        coverage_bonus_per_10pct : float
            Bonus added per 10% coverage above 10% floor.
        max_coverage_bonus : float
            Maximum coverage bonus cap.
        """
        self.bin_counts = bin_counts if bin_counts is not None else [n_bins]
        self.use_coverage_bonus = use_coverage_bonus
        self.coverage_bonus_per_10pct = coverage_bonus_per_10pct
        self.max_coverage_bonus = max_coverage_bonus

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

    def fit(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        normalization_data: pd.Series | None = None,
    ) -> None:
        """Fit model with grid search over bin counts.

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

        df = pd.DataFrame({"feature": feature_data, "target": target_data}).dropna()

        if len(df) < max(10, self.n_bins * 5):
            msg = f"Insufficient data after dropna: {len(df)} rows"
            raise ValueError(msg)

        # Grid search over bin counts
        best_result = None
        for n in self.bin_counts:
            self.n_bins = n
            bins = self._create_bins(df["feature"], df["target"])
            df_copy = df.copy()
            df_copy["bin"] = bins
            ordered_bins = sorted(df_copy["bin"].unique())

            bin_stats = self._calculate_bin_stats(df_copy, ordered_bins)

            # Select best bin per direction
            best_long_bin, best_short_bin = self._select_best_bins(bin_stats, ordered_bins)

            # Compute coverage-adjusted Sharpe for winning bins
            total_obs = len(df_copy)
            if best_long_bin is not None:
                long_coverage = bin_stats[best_long_bin]["count"] / total_obs
                long_sharpe = self._compute_coverage_adjusted_sharpe(
                    bin_stats[best_long_bin], long_coverage
                )
                long_t_stat = bin_stats[best_long_bin]["t_stat"]
            else:
                long_sharpe = long_t_stat = float("-inf")

            if best_short_bin is not None:
                short_coverage = bin_stats[best_short_bin]["count"] / total_obs
                short_sharpe = self._compute_coverage_adjusted_sharpe(
                    bin_stats[best_short_bin], short_coverage
                )
                short_t_stat = abs(bin_stats[best_short_bin]["t_stat"])
            else:
                short_sharpe = short_t_stat = float("-inf")

            # Track best result by highest t-stat
            if self.strategy == "long":
                score = long_t_stat
            elif self.strategy == "short":
                score = short_t_stat
            else:  # long_short
                score = max(long_t_stat, short_t_stat)

            if best_result is None or score > best_result["score"]:
                best_result = {
                    "n_bins": n,
                    "score": score,
                    "bins": bins,
                    "ordered_bins": ordered_bins,
                    "bin_stats": bin_stats,
                    "best_long_bin": best_long_bin,
                    "best_short_bin": best_short_bin,
                    "long_sharpe": long_sharpe,
                    "short_sharpe": short_sharpe,
                }

        # Set fitted state from best result
        self.n_bins = best_result["n_bins"]
        df["bin"] = best_result["bins"]
        self.bin_edges_ = self._extract_bin_edges(df, best_result["ordered_bins"])
        self.bin_stats_ = best_result["bin_stats"]

        # Store selected bins and coverage-adjusted sharpes
        self.selected_bins_ = {
            "long": best_result["best_long_bin"],
            "short": best_result["best_short_bin"],
            "long_sharpe": best_result["long_sharpe"],
            "short_sharpe": best_result["short_sharpe"],
        }

        # Build position multipliers for selected bins only
        self._build_position_multipliers_from_selected_bins()

        # Legacy: keep significant_regions_ empty for backwards compat
        self.significant_regions_ = []

        # Store fit config
        self.fit_config_ = {
            "n_bins": self.n_bins,
            "selection_metric": self.selection_metric,
            "strategy": self.strategy,
            "metric_threshold": self.metric_threshold,
            "t_threshold": self.t_threshold,
            "min_region_width": self.min_region_width,
            "shrinkage_k": self.shrinkage_k,
            "bin_counts": self.bin_counts,
            "use_coverage_bonus": self.use_coverage_bonus,
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

    def _compute_coverage_adjusted_sharpe(self, bin_stat: dict, coverage: float) -> float:
        """Compute coverage-adjusted Sharpe ratio.

        Parameters
        ----------
        bin_stat : dict
            Per-bin statistics from _calculate_bin_stats.
        coverage : float
            Fraction of observations in this bin (0 to 1).

        Returns
        -------
        float
            Coverage-adjusted Sharpe ratio.
        """
        base_sharpe = bin_stat["sharpe"]

        if not self.use_coverage_bonus:
            return base_sharpe

        # Coverage bonus: +bonus per 10% above 10% floor
        coverage_pct = coverage * 100
        bonus = 0.0
        if coverage_pct > 10:
            bonus = ((coverage_pct - 10) / 10) * self.coverage_bonus_per_10pct
            bonus = min(bonus, self.max_coverage_bonus)

        return base_sharpe + bonus

    def _build_position_multipliers_from_selected_bins(self) -> None:
        """Build position multipliers dict from selected bins.

        Populates position_multipliers_by_strategy_ and active_bins_by_strategy_
        based on selected_bins_.
        """
        multipliers_long: dict[int, float] = {}
        multipliers_short: dict[int, float] = {}
        multipliers_long_short: dict[int, float] = {}

        if self.selected_bins_["long"] is not None:
            bin_idx = self.selected_bins_["long"]
            sharpe = self.selected_bins_["long_sharpe"]
            multipliers_long[bin_idx] = sharpe
            multipliers_long_short[bin_idx] = sharpe

        if self.selected_bins_["short"] is not None:
            bin_idx = self.selected_bins_["short"]
            sharpe = self.selected_bins_["short_sharpe"]
            # For short: coverage-adjusted Sharpe is based on raw Sharpe (which may be negative)
            # We want the multiplier to be negative, so just use the raw value
            # (if base Sharpe is negative, coverage bonus makes it less negative, but it stays negative)
            multipliers_short[bin_idx] = sharpe  # Keep as-is (should be negative or made negative)
            multipliers_long_short[bin_idx] = sharpe  # Keep as-is

        self.position_multipliers_by_strategy_ = {
            "long": multipliers_long,
            "short": multipliers_short,
            "long_short": multipliers_long_short,
        }

        # Also update active_bins_by_strategy_ for backwards compat
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
