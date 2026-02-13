"""
Base model classes for feature binning.

This module defines the shared fit/predict contract for binning models used in
feature selection and ensemble training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BinningModelBase(ABC):
    """Abstract base class for binning models."""

    model_type: str = "base"

    def __init__(
        self,
        n_bins: int = 3,
        selection_metric: str = "sharpe",
        normalize_by: Optional[str] = "ewsd",
        strategy: str = "long",
        metric_threshold: float = 0.0,
        t_threshold: float = 2.0,
        min_region_width: int = 2,
        shrinkage_k: float = 20.0,
        long_clip_min: float = 0.5,
        long_clip_max: float = 2.0,
        short_clip_min: float = 0.5,
        short_clip_max: float = 2.0,
    ) -> None:
        self.n_bins = n_bins
        self.selection_metric = selection_metric
        self.normalize_by = normalize_by
        self.strategy = strategy

        self.metric_threshold = metric_threshold
        self.t_threshold = t_threshold
        self.min_region_width = min_region_width
        self.shrinkage_k = shrinkage_k
        self.long_clip_min = long_clip_min
        self.long_clip_max = long_clip_max
        self.short_clip_min = short_clip_min
        self.short_clip_max = short_clip_max

        self.feature_column: Optional[str] = None
        self.normalization_data_ = None

        self.bin_edges_: Optional[List[float]] = None
        self.bin_stats_: Dict[int, Dict[str, float]] = {}
        self.significant_regions_: List[Dict[str, object]] = []
        self.active_bins_by_strategy_: Dict[str, List[int]] = {
            "long": [],
            "short": [],
            "long_short": [],
        }
        self.position_multipliers_by_strategy_: Dict[str, Dict[int, float]] = {
            "long": {},
            "short": {},
            "long_short": {},
        }
        self.fit_config_: Dict[str, object] = {}
        self.model_version_ = "binning_v2"
        self.is_fitted_ = False

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        if strategy == "long-short":
            return "long_short"
        return strategy

    def _reset_fitted_state(self) -> None:
        self.bin_edges_ = None
        self.bin_stats_ = {}
        self.significant_regions_ = []
        self.active_bins_by_strategy_ = {"long": [], "short": [], "long_short": []}
        self.position_multipliers_by_strategy_ = {"long": {}, "short": {}, "long_short": {}}
        self.fit_config_ = {}
        self.is_fitted_ = False

    def _normalize_feature(
        self,
        feature_data: pd.Series,
        normalization_data: Optional[pd.Series] = None,
    ) -> pd.Series:
        if self.normalize_by is None:
            return feature_data

        if normalization_data is None:
            raise ValueError(
                f"normalize_by='{self.normalize_by}' but no normalization_data provided. "
                f"Pass the {self.normalize_by.upper()} column when calling fit()/predict()."
            )

        aligned_feature = feature_data.reindex(normalization_data.index)
        min_vol = 1e-8 if self.normalize_by == "ewsd" else 1.0
        safe_vol = normalization_data.clip(lower=min_vol)
        return aligned_feature / safe_vol

    @abstractmethod
    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        """Create integer bin assignments for feature_data."""

    def _assign_bins(self, feature_data: pd.Series) -> pd.Series:
        """Assign bins during prediction using fitted bin edges."""
        if self.bin_edges_ is None or len(self.bin_edges_) == 0:
            return pd.Series(0, index=feature_data.index, dtype=int)
        bins = np.digitize(feature_data.to_numpy(), np.asarray(self.bin_edges_, dtype=float))
        return pd.Series(bins, index=feature_data.index, dtype=int)

    def _extract_bin_edges(self, df: pd.DataFrame, ordered_bins: List[int]) -> List[float]:
        if len(ordered_bins) <= 1:
            return []

        maxima = [
            float(df.loc[df["bin"] == bin_idx, "feature"].max())
            for bin_idx in ordered_bins[:-1]
        ]
        return maxima

    def _calculate_sortino(self, returns: pd.Series) -> float:
        downside = returns[returns < 0]
        downside_std = float(downside.std()) if len(downside) > 0 else 0.0
        if downside_std <= 1e-12:
            mean_return = float(returns.mean())
            return mean_return * np.sqrt(252.0) if mean_return > 0 else 0.0
        return float(returns.mean() / downside_std) * np.sqrt(252.0)

    def _calculate_bin_stats(self, df: pd.DataFrame, ordered_bins: List[int]) -> Dict[int, Dict[str, float]]:
        stats: Dict[int, Dict[str, float]] = {}

        for bin_idx in ordered_bins:
            bin_data = df.loc[df["bin"] == bin_idx]
            if bin_data.empty:
                continue

            returns = bin_data["target"]
            n_obs = int(len(returns))
            mean_return = float(returns.mean())
            volatility = float(returns.std())

            if not np.isfinite(volatility) or volatility <= 1e-12:
                if abs(mean_return) <= 1e-12:
                    sharpe = 0.0
                    t_stat = 0.0
                else:
                    # Deterministic-return bins still encode directional edge.
                    sharpe = float(np.sign(mean_return) * 10.0)
                    t_stat = float(np.sign(mean_return) * np.sqrt(n_obs))
            else:
                sharpe = float(mean_return / volatility)
                t_stat = float(mean_return / (volatility / np.sqrt(n_obs)))

            adjusted_sharpe = float(
                sharpe * np.sqrt(n_obs) / (np.sqrt(n_obs) + float(self.shrinkage_k))
            )

            sortino_long = self._calculate_sortino(returns)
            sortino_short = self._calculate_sortino(-returns)

            if self.selection_metric == "sharpe":
                metric_long = sharpe
                metric_short = -sharpe
            elif self.selection_metric == "mean":
                metric_long = mean_return
                metric_short = -mean_return
            elif self.selection_metric == "t_stat":
                metric_long = t_stat
                metric_short = -t_stat
            elif self.selection_metric == "sortino":
                metric_long = sortino_long
                metric_short = sortino_short
            else:
                raise ValueError(
                    f"Unknown selection_metric: {self.selection_metric}. "
                    "Use 'sharpe', 'mean', 't_stat', or 'sortino'."
                )

            stats[int(bin_idx)] = {
                "count": n_obs,
                "mean_return": mean_return,
                "volatility": volatility,
                "sharpe": sharpe,
                "adjusted_sharpe": adjusted_sharpe,
                "adjusted_sharpe_short": -adjusted_sharpe,
                "t_stat": t_stat,
                "selection_metric_long": float(metric_long),
                "selection_metric_short": float(metric_short),
                "feature_min": float(bin_data["feature"].min()),
                "feature_max": float(bin_data["feature"].max()),
            }

        return stats

    def _detect_significant_regions(self, ordered_bins: List[int]) -> List[Dict[str, object]]:
        if self.model_type != "continuous_binning":
            return []

        significant_bins = [
            bin_idx
            for bin_idx in ordered_bins
            if abs(float(self.bin_stats_[bin_idx]["t_stat"])) >= float(self.t_threshold)
        ]

        if not significant_bins:
            return []

        regions: List[List[int]] = []
        current_region: List[int] = [significant_bins[0]]

        for bin_idx in significant_bins[1:]:
            if bin_idx == current_region[-1] + 1:
                current_region.append(bin_idx)
            else:
                regions.append(current_region)
                current_region = [bin_idx]
        regions.append(current_region)

        filtered = [region for region in regions if len(region) >= int(self.min_region_width)]

        return [
            {"start_bin": int(region[0]), "end_bin": int(region[-1]), "bins": [int(b) for b in region]}
            for region in filtered
        ]

    def _allowed_bins_after_region_filter(self, ordered_bins: List[int]) -> List[int]:
        if self.model_type != "continuous_binning":
            return ordered_bins
        if not self.significant_regions_:
            return []

        region_bins = {
            int(bin_idx)
            for region in self.significant_regions_
            for bin_idx in region["bins"]  # type: ignore[index]
        }
        return [bin_idx for bin_idx in ordered_bins if bin_idx in region_bins]

    def _build_active_bins_and_multipliers(self, ordered_bins: List[int]) -> None:
        allowed_bins = set(self._allowed_bins_after_region_filter(ordered_bins))

        long_bins: List[int] = []
        short_bins: List[int] = []
        long_short_bins: List[int] = []

        long_mults: Dict[int, float] = {}
        short_mults: Dict[int, float] = {}
        long_short_mults: Dict[int, float] = {}

        for bin_idx in ordered_bins:
            # Rule-based level 0 is always flat.
            if self.model_type == "rule_based" and bin_idx == 1:
                continue

            if bin_idx not in allowed_bins:
                continue

            stat = self.bin_stats_[bin_idx]
            long_pass = stat["selection_metric_long"] >= float(self.metric_threshold)
            short_pass = stat["selection_metric_short"] >= float(self.metric_threshold)

            long_multiplier = 0.0
            short_multiplier = 0.0

            if long_pass:
                long_multiplier = float(
                    np.clip(
                        1.0 + stat["adjusted_sharpe"],
                        float(self.long_clip_min),
                        float(self.long_clip_max),
                    )
                )
                long_bins.append(bin_idx)
                long_mults[bin_idx] = long_multiplier

            if short_pass:
                short_multiplier = -float(
                    np.clip(
                        1.0 + stat["adjusted_sharpe_short"],
                        float(self.short_clip_min),
                        float(self.short_clip_max),
                    )
                )
                short_bins.append(bin_idx)
                short_mults[bin_idx] = short_multiplier

            combined_multiplier = 0.0
            if long_multiplier != 0.0 and short_multiplier != 0.0:
                if abs(long_multiplier) == abs(short_multiplier):
                    combined_multiplier = (
                        long_multiplier if stat["mean_return"] >= 0 else short_multiplier
                    )
                else:
                    combined_multiplier = (
                        long_multiplier
                        if abs(long_multiplier) > abs(short_multiplier)
                        else short_multiplier
                    )
            elif long_multiplier != 0.0:
                combined_multiplier = long_multiplier
            elif short_multiplier != 0.0:
                combined_multiplier = short_multiplier

            if combined_multiplier != 0.0:
                long_short_bins.append(bin_idx)
                long_short_mults[bin_idx] = float(np.clip(combined_multiplier, -2.0, 2.0))

        self.active_bins_by_strategy_ = {
            "long": sorted(long_bins),
            "short": sorted(short_bins),
            "long_short": sorted(long_short_bins),
        }
        self.position_multipliers_by_strategy_ = {
            "long": long_mults,
            "short": short_mults,
            "long_short": long_short_mults,
        }

    def fit(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        normalization_data: Optional[pd.Series] = None,
    ) -> "BinningModelBase":
        if feature_data.name is None:
            raise ValueError(
                "feature_data must have a name attribute. Pass a named pd.Series."
            )

        self._reset_fitted_state()

        self.feature_column = feature_data.name
        self.normalization_data_ = normalization_data

        df = pd.DataFrame({"feature": feature_data, "target": target_data}).dropna()
        if len(df) < max(10, self.n_bins * 5):
            raise ValueError(f"Insufficient data: got {len(df)} samples")

        bins = self._create_bins(df["feature"], df["target"]).reindex(df.index)
        df["bin"] = bins
        df = df.dropna(subset=["bin"]).copy()
        df["bin"] = df["bin"].astype(int)

        if df.empty:
            raise ValueError("Binning produced no valid rows")

        ordered_bins = sorted(int(v) for v in df["bin"].unique())

        self.bin_edges_ = self._extract_bin_edges(df, ordered_bins)
        self.bin_stats_ = self._calculate_bin_stats(df, ordered_bins)
        self.significant_regions_ = self._detect_significant_regions(ordered_bins)
        self._build_active_bins_and_multipliers(ordered_bins)

        self.fit_config_ = {
            "selection_metric": self.selection_metric,
            "metric_threshold": self.metric_threshold,
            "t_threshold": self.t_threshold,
            "min_region_width": self.min_region_width,
            "shrinkage_k": self.shrinkage_k,
            "long_clip_bounds": [self.long_clip_min, self.long_clip_max],
            "short_clip_bounds": [self.short_clip_min, self.short_clip_max],
        }

        self.is_fitted_ = True
        return self

    def predict(
        self,
        feature_data: pd.Series,
        strategy: str = "long",
        normalization_data: Optional[pd.Series] = None,
        scaled: bool = False,
    ) -> pd.Series:
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict()")

        normalized_strategy = self._normalize_strategy(strategy)
        if normalized_strategy not in {"long", "short", "long_short"}:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. Use 'long', 'short', or 'long_short'/'long-short'"
            )

        bins = self._assign_bins(feature_data)
        multipliers = self.position_multipliers_by_strategy_.get(normalized_strategy, {})

        raw = bins.map(lambda bin_idx: float(multipliers.get(int(bin_idx), 0.0))).astype(float)
        raw.index = feature_data.index

        if not scaled or self.normalize_by is None:
            return raw

        if normalization_data is None:
            raise ValueError(
                f"scaled=True and normalize_by='{self.normalize_by}' but no normalization_data provided. "
                f"Pass the {self.normalize_by.upper()} column when calling predict()."
            )

        aligned_signal = raw.reindex(normalization_data.index, fill_value=0.0)
        min_vol = 1e-8 if self.normalize_by == "ewsd" else 1.0
        safe_vol = normalization_data.clip(lower=min_vol)
        return aligned_signal / safe_vol

    def get_bin_stats(self) -> Dict[int, Dict[str, float]]:
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling get_bin_stats()")
        return self.bin_stats_

    def get_fitted_params(self) -> Dict[str, object]:
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling get_fitted_params()")
        return {
            "model_version": self.model_version_,
            "model_type": self.model_type,
            "bin_edges": self.bin_edges_,
            "bin_stats": self.bin_stats_,
            "significant_regions": self.significant_regions_,
            "active_bins_by_strategy": self.active_bins_by_strategy_,
            "position_multipliers_by_strategy": self.position_multipliers_by_strategy_,
            "fit_config": self.fit_config_,
        }

    def compute_objective_metric(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        objective_metric: "ObjectiveMetric",
        strategy: str = "long",
        normalization_data: Optional[pd.Series] = None,
    ) -> float:
        self.fit(feature_data, target_data, normalization_data=normalization_data)
        signals = self.predict(feature_data, strategy=strategy, normalization_data=normalization_data)
        selected_returns = (target_data * signals)[signals != 0]
        if len(selected_returns) == 0:
            return 0.0
        return float(objective_metric.compute(selected_returns))

    def save_to_feature_list(
        self,
        filepath: str,
        tickers: Optional[List[str]] = None,
    ) -> None:
        if self.feature_column is None:
            raise ValueError(
                "feature_column not set. Call fit() with a named pd.Series first, "
                "or ensure the Series has a name attribute (e.g., df['column_name'])."
            )

        import utils.helpers as helpers
        from utils.enums import TimeFrame

        parsed = helpers.parse_feature_column_name(self.feature_column)
        if parsed.get("module") is None or parsed.get("tf") is None:
            raise ValueError(
                f"Feature column '{self.feature_column}' cannot be parsed. "
                "Column names must follow standardized format: module_feature_tf_param1_val1_param2_val2."
            )

        tf = parsed.get("tf")
        if tf is not None:
            if isinstance(tf, TimeFrame):
                pass
            elif isinstance(tf, str):
                try:
                    TimeFrame[tf]
                except (KeyError, AttributeError) as exc:
                    raise ValueError(
                        f"Feature column '{self.feature_column}' has invalid timeframe '{tf}'."
                    ) from exc
            else:
                raise ValueError(
                    f"Feature column '{self.feature_column}' has invalid timeframe type '{type(tf)}'."
                )

        model_name = f"{self.feature_column}_{self.strategy}"
        params = self.get_params()

        feature_config = {
            "name": model_name,
            "model_type": self.model_type,
            "feature_column": self.feature_column,
            "strategy": self.strategy,
            "constructor_params": params,
        }

        try:
            from ensemble.ensemble_utils import add_feature_to_control_file
        except ImportError as exc:
            raise ImportError(
                "Cannot import ensemble.ensemble_utils. Make sure the ensemble package is properly installed."
            ) from exc

        add_feature_to_control_file(filepath, feature_config, tickers=tickers)

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        return {
            "n_bins": self.n_bins,
            "selection_metric": self.selection_metric,
            "normalize_by": self.normalize_by,
            "strategy": self.strategy,
            "metric_threshold": self.metric_threshold,
            "t_threshold": self.t_threshold,
            "min_region_width": self.min_region_width,
            "shrinkage_k": self.shrinkage_k,
            "long_clip_min": self.long_clip_min,
            "long_clip_max": self.long_clip_max,
            "short_clip_min": self.short_clip_min,
            "short_clip_max": self.short_clip_max,
        }

    def set_params(self, **params: object) -> "BinningModelBase":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}")
        self._reset_fitted_state()
        return self

    def score(self, X: pd.Series, y: pd.Series, strategy: str = "long") -> float:
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling score()")
        signals = self.predict(X, strategy=strategy)
        selected_returns = (y * signals)[signals != 0]
        if len(selected_returns) == 0:
            return 0.0
        return float(selected_returns.mean())
