"""Clustered weight layer for forecast combination.

This module combines model forecasts using one of two clustered allocation modes:

- ``cluster_equal``: cluster first, then equal weight across clusters.
- ``cluster_corr_ulcer``: cluster first, then tilt cluster weights by
  average positive correlation and ulcer index on forecast-weighted returns.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass as _dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPSILON = 1e-8


@_dataclass(frozen=True)
class WeightLayerConfig:
    """Configuration for the clustered weight layer."""

    weighting_method: str = "cluster_equal"
    rho_cut: float = 0.70
    fdm_max: float = 2.0
    group_weight_cap: float = 0.25
    risk_tilt_alpha: float = 0.5

    def __post_init__(self) -> None:
        valid_methods = {"cluster_equal", "cluster_corr_ulcer"}
        if self.weighting_method not in valid_methods:
            raise ValueError(
                f"weighting_method must be one of {sorted(valid_methods)}, "
                f"got '{self.weighting_method}'"
            )
        if not (0.0 <= self.rho_cut <= 1.0):
            raise ValueError(f"rho_cut must be in [0, 1], got {self.rho_cut}")
        if self.fdm_max <= 0.0:
            raise ValueError(f"fdm_max must be > 0, got {self.fdm_max}")
        if not (0.0 < self.group_weight_cap <= 1.0):
            raise ValueError(
                f"group_weight_cap must be in (0, 1], got {self.group_weight_cap}"
            )
        if self.risk_tilt_alpha < 0.0:
            raise ValueError(
                f"risk_tilt_alpha must be >= 0, got {self.risk_tilt_alpha}"
            )


class BaseWeightLayer(ABC):
    """Abstract interface for weight-layer implementations."""

    def __init__(self, fdm_max: float = 2.0) -> None:
        self.fdm_max = fdm_max
        self.fdm_: Dict[str, float] = {}
        self.weights_: Dict[str, pd.Series] = {}
        self.model_names_: Dict[str, List[str]] = {}
        self.mean_cluster_correlation_: Dict[str, float] = {}
        self.cluster_assignments_: Dict[str, Dict[str, str]] = {}
        self.cluster_weights_: Dict[str, Dict[str, float]] = {}
        self.cluster_metrics_: Dict[str, Dict[str, Dict[str, float | int | None]]] = {}
        self.is_fitted_: bool = False

    @property
    @abstractmethod
    def weight_method(self) -> str:
        """Canonical weighting method name."""

    @abstractmethod
    def fit(
        self,
        forecast_vectors: List[pd.DataFrame],
        signals: pd.DataFrame,
        returns: Optional[pd.Series] = None,
    ) -> "BaseWeightLayer":
        """Fit ticker-level weights from training data."""

    def combine(
        self,
        forecast_vectors: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """Combine forecasts using fitted ticker weights and FDM."""
        if not self.is_fitted_:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before calling combine()"
            )
        if not forecast_vectors:
            return pd.DataFrame(columns=["ticker", "forecast_score"])

        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        if all_forecasts.empty:
            return pd.DataFrame(columns=["ticker", "forecast_score"])

        combined_results = [
            self._combine_ticker(
                ticker=str(ticker),
                ticker_forecasts=all_forecasts[all_forecasts["ticker"] == ticker].copy(),
            )
            for ticker in all_forecasts["ticker"].unique()
        ]

        if not combined_results:
            return pd.DataFrame(columns=["ticker", "forecast_score"])

        combined = pd.concat(combined_results, ignore_index=True)
        if "datetime" in combined.columns:
            return combined[["ticker", "datetime", "forecast_score"]]
        return combined[["ticker", "forecast_score"]]

    def _combine_ticker(
        self,
        ticker: str,
        ticker_forecasts: pd.DataFrame,
    ) -> pd.DataFrame:
        if ticker in self.weights_:
            ticker_weights = self.weights_[ticker]
        else:
            available_models = ticker_forecasts["model_name"].unique()
            n_models = len(available_models)
            equal_weight = 1.0 / n_models if n_models > 0 else 1.0
            ticker_weights = pd.Series({m: equal_weight for m in available_models})
            logger.warning(
                "Ticker %s not seen during fit(), using equal weights (%0.4f per model)",
                ticker,
                equal_weight,
            )

        ticker_forecasts["weight"] = ticker_forecasts["model_name"].map(ticker_weights)
        missing_models = ticker_forecasts[ticker_forecasts["weight"].isna()][
            "model_name"
        ].unique()
        if len(missing_models) > 0:
            n_known = len(ticker_weights)
            fallback_weight = (
                1.0 / (n_known + len(missing_models))
                if n_known > 0
                else 1.0 / len(missing_models)
            )
            ticker_forecasts["weight"] = ticker_forecasts["weight"].fillna(fallback_weight)

        ticker_forecasts["weighted_forecast"] = (
            ticker_forecasts["forecast"] * ticker_forecasts["weight"]
        )

        if "datetime" in ticker_forecasts.columns:
            ticker_combined = ticker_forecasts.groupby(
                "datetime", as_index=False
            ).agg({"weighted_forecast": "sum"})
            ticker_combined["ticker"] = ticker
        else:
            ticker_combined = pd.DataFrame(
                {
                    "ticker": [ticker],
                    "weighted_forecast": [ticker_forecasts["weighted_forecast"].sum()],
                }
            )

        ticker_fdm = self.fdm_.get(ticker, 1.0)
        ticker_combined["forecast_score"] = (
            (ticker_combined["weighted_forecast"] * ticker_fdm).clip(upper=2.0, lower=-2.0)
        )
        return ticker_combined

    def get_diagnostics(self) -> Dict:
        """Return fitted diagnostics."""
        if not self.is_fitted_:
            return {"is_fitted": False}

        tickers_dict: Dict[str, Dict[str, object]] = {}
        for ticker in self.fdm_:
            ticker_weights = self.weights_.get(ticker)
            weights_dict = (
                None
                if ticker_weights is None
                else {
                    str(k): (0.0 if pd.isna(v) else float(v))
                    for k, v in ticker_weights.to_dict().items()
                }
            )
            tickers_dict[ticker] = {
                "fdm": self.fdm_.get(ticker, 1.0),
                "weights": weights_dict,
                "models": self.model_names_.get(ticker, []),
                "n_models": len(self.model_names_.get(ticker, [])),
                "cluster_assignments": self.cluster_assignments_.get(ticker, {}),
                "cluster_weights": self.cluster_weights_.get(ticker, {}),
                "cluster_metrics": self.cluster_metrics_.get(ticker, {}),
                "mean_cluster_correlation": self.mean_cluster_correlation_.get(ticker, 1.0),
            }

        fdm_values = list(self.fdm_.values())
        n_models_per_ticker = [len(self.model_names_.get(ticker, [])) for ticker in self.fdm_]
        cluster_counts = [
            len(self.cluster_weights_.get(ticker, {}))
            for ticker in self.fdm_
        ]

        return {
            "is_fitted": True,
            "tickers": tickers_dict,
            "summary": {
                "n_tickers": len(self.fdm_),
                "mean_fdm": float(np.mean(fdm_values)) if fdm_values else 1.0,
                "min_fdm": float(min(fdm_values)) if fdm_values else 1.0,
                "max_fdm": float(max(fdm_values)) if fdm_values else 1.0,
                "mean_models_per_ticker": (
                    float(np.mean(n_models_per_ticker)) if n_models_per_ticker else 0.0
                ),
                "mean_clusters_per_ticker": (
                    float(np.mean(cluster_counts)) if cluster_counts else 0.0
                ),
                "total_models": sum(n_models_per_ticker),
            },
            "weight_method": self.weight_method,
            "fdm_max": self.fdm_max,
        }


def _pivot_ticker_forecasts(
    ticker_forecasts: pd.DataFrame,
    available_models: List[str],
) -> pd.DataFrame:
    """Pivot raw forecasts into a date x model matrix."""
    if "datetime" not in ticker_forecasts.columns:
        return pd.DataFrame()

    df = ticker_forecasts.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.normalize()
    pivot = df.pivot_table(
        index="date",
        columns="model_name",
        values="forecast",
        aggfunc="mean",
    )
    existing_models = [model for model in available_models if model in pivot.columns]
    if not existing_models:
        return pd.DataFrame()
    pivot = pivot[existing_models]
    return pivot.dropna(how="all")


def _extract_cluster_assignments(
    forecast_pivot: pd.DataFrame,
    rho_cut: float,
) -> Dict[str, str]:
    """Cluster model forecast series using hierarchical correlation clustering."""
    model_names = list(forecast_pivot.columns)
    if len(model_names) <= 1 or len(forecast_pivot) < 2:
        return {model: "cluster_1" for model in model_names}

    from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    corr = forecast_pivot.corr().fillna(0.0).clip(lower=0.0)
    for column in forecast_pivot.columns:
        if forecast_pivot[column].nunique(dropna=True) <= 1:
            corr.loc[column, :] = 1.0
            corr.loc[:, column] = 1.0
    corr_values = corr.to_numpy(copy=True)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr_values), 0.0, None))
    np.fill_diagonal(dist, 0.0)

    if len(model_names) == 2:
        if corr_values[0, 1] >= rho_cut:
            return {model_names[0]: "cluster_1", model_names[1]: "cluster_1"}
        return {model_names[0]: "cluster_1", model_names[1]: "cluster_2"}

    condensed = squareform(dist, checks=False)
    if not np.all(np.isfinite(condensed)):
        return {model: "cluster_1" for model in model_names}

    linkage_matrix = scipy_linkage(condensed, method="ward")
    dist_threshold = float(np.sqrt(0.5 * (1.0 - rho_cut)))
    labels = fcluster(linkage_matrix, dist_threshold, criterion="distance")
    normalized_labels = {
        raw_label: f"cluster_{idx}"
        for idx, raw_label in enumerate(sorted(set(labels)), start=1)
    }
    return {
        model: normalized_labels[int(label)]
        for model, label in zip(model_names, labels)
    }


def _build_cluster_forecasts(
    forecast_pivot: pd.DataFrame,
    cluster_assignments: Dict[str, str],
) -> pd.DataFrame:
    """Aggregate model forecasts equally within each cluster."""
    clusters = sorted(set(cluster_assignments.values()))
    cluster_series = {
        cluster: forecast_pivot[
            [model for model, mapped_cluster in cluster_assignments.items() if mapped_cluster == cluster]
        ].mean(axis=1)
        for cluster in clusters
    }
    return pd.DataFrame(cluster_series)


def _compute_cluster_correlation(cluster_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compute cluster-level correlation matrix with negative values floored to zero."""
    if cluster_forecasts.empty or len(cluster_forecasts.columns) == 0:
        return pd.DataFrame()

    corr = cluster_forecasts.corr().fillna(0.0).clip(lower=0.0)
    for column in cluster_forecasts.columns:
        if cluster_forecasts[column].nunique(dropna=True) <= 1:
            corr.loc[column, :] = 1.0
            corr.loc[:, column] = 1.0
    np.fill_diagonal(corr.values, 1.0)
    return corr


def _mean_off_diagonal_correlation(corr_matrix: pd.DataFrame) -> float:
    """Mean upper-triangle correlation, assuming negative values already clipped away."""
    n_cols = len(corr_matrix.columns)
    if n_cols <= 1:
        return 1.0
    mask = np.triu(np.ones((n_cols, n_cols), dtype=bool), k=1)
    correlations = corr_matrix.to_numpy()[mask]
    if len(correlations) == 0:
        return 1.0
    return float(correlations.mean())


def _compute_fdm_from_corr_matrix(corr_matrix: pd.DataFrame, fdm_max: float) -> float:
    """Compute FDM from cluster-level correlation."""
    if len(corr_matrix.columns) <= 1:
        return 1.0
    mean_corr = _mean_off_diagonal_correlation(corr_matrix)
    fdm = float(np.sqrt(1.0 / (mean_corr + 0.01)))
    return min(fdm, fdm_max)


def _apply_group_weight_cap(weights: np.ndarray, cap: float) -> np.ndarray:
    """Iteratively cap cluster weights and redistribute residual weight."""
    n_weights = len(weights)
    if cap >= 1.0 or n_weights <= 1:
        return weights
    if n_weights * cap < 1.0 - 1e-9:
        return np.ones(n_weights) / n_weights

    clipped = weights.copy()
    permanently_capped = np.zeros(n_weights, dtype=bool)

    for _ in range(n_weights):
        over_cap = (~permanently_capped) & (clipped > cap)
        if not over_cap.any():
            break
        surplus = float((clipped[over_cap] - cap).sum())
        clipped[over_cap] = cap
        permanently_capped |= over_cap
        free = ~permanently_capped
        free_sum = clipped[free].sum()
        if free_sum <= 0.0:
            break
        clipped[free] += surplus * (clipped[free] / free_sum)

    total = clipped.sum()
    return clipped / total if total > 0.0 else np.ones(n_weights) / n_weights


def _compute_ulcer_index(series: pd.Series) -> float:
    """Ulcer index on a return proxy series."""
    clean = series.fillna(0.0).astype(float)
    if clean.empty:
        return 0.0
    cumulative = clean.cumsum()
    running_peak = cumulative.cummax()
    drawdown = cumulative - running_peak
    return float(np.sqrt(np.mean(np.square(drawdown.to_numpy()))))


def _distribute_cluster_weights_to_models(
    model_names: List[str],
    cluster_assignments: Dict[str, str],
    cluster_weights: Dict[str, float],
) -> pd.Series:
    """Split cluster weights equally within each cluster."""
    model_weights: Dict[str, float] = {}
    for cluster, cluster_weight in cluster_weights.items():
        members = [model for model in model_names if cluster_assignments[model] == cluster]
        per_model = cluster_weight / len(members) if members else 0.0
        for model in members:
            model_weights[model] = per_model

    total = sum(model_weights.values())
    if total <= 0.0:
        equal_weight = 1.0 / len(model_names) if model_names else 1.0
        return pd.Series({model: equal_weight for model in model_names})
    return pd.Series({model: weight / total for model, weight in model_weights.items()})


class ClusteredWeightLayer(BaseWeightLayer):
    """Always-clustered weight layer with two allocation modes."""

    def __init__(self, config: Optional[WeightLayerConfig] = None) -> None:
        self._wl_config = config or WeightLayerConfig()
        super().__init__(fdm_max=self._wl_config.fdm_max)

    @property
    def weight_method(self) -> str:
        return self._wl_config.weighting_method

    def fit(
        self,
        forecast_vectors: List[pd.DataFrame],
        signals: pd.DataFrame,
        returns: Optional[pd.Series] = None,
    ) -> "ClusteredWeightLayer":
        del signals

        if not forecast_vectors:
            raise ValueError("forecast_vectors cannot be empty")
        if self.weight_method == "cluster_corr_ulcer" and returns is None:
            raise ValueError(
                "cluster_corr_ulcer requires returns so ulcer index can be computed "
                "from forecast-weighted returns"
            )

        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        if all_forecasts.empty:
            raise ValueError("forecast_vectors contain no data")
        if "ticker" not in all_forecasts.columns:
            raise ValueError("forecast_vectors must contain 'ticker' column")

        normalized_returns: Optional[pd.Series] = None
        if returns is not None and not returns.empty:
            normalized_returns = returns.astype(float).copy()
            normalized_returns.index = pd.to_datetime(normalized_returns.index).normalize()
            if normalized_returns.index.duplicated().any():
                normalized_returns = normalized_returns.groupby(level=0).mean()

        self.fdm_.clear()
        self.weights_.clear()
        self.model_names_.clear()
        self.mean_cluster_correlation_.clear()
        self.cluster_assignments_.clear()
        self.cluster_weights_.clear()
        self.cluster_metrics_.clear()

        unique_tickers = sorted(all_forecasts["ticker"].unique(), key=str)
        logger.info(
            "Fitting %s for %d ticker(s): %s",
            self.__class__.__name__,
            len(unique_tickers),
            [str(ticker) for ticker in unique_tickers],
        )

        for raw_ticker in unique_tickers:
            ticker = str(raw_ticker)
            ticker_forecasts = all_forecasts[all_forecasts["ticker"] == raw_ticker].copy()
            available_models = sorted(str(model) for model in ticker_forecasts["model_name"].unique())
            if not available_models:
                continue

            forecast_pivot = _pivot_ticker_forecasts(ticker_forecasts, available_models)
            if len(available_models) == 1:
                self._store_single_cluster_result(
                    ticker=ticker,
                    model_names=available_models,
                )
                continue

            if forecast_pivot.empty or len(forecast_pivot) < 2:
                self._store_equal_fallback_result(
                    ticker=ticker,
                    model_names=available_models,
                    reason="insufficient forecast history for clustering",
                )
                continue

            cluster_assignments = _extract_cluster_assignments(
                forecast_pivot=forecast_pivot,
                rho_cut=self._wl_config.rho_cut,
            )
            cluster_forecasts = _build_cluster_forecasts(
                forecast_pivot=forecast_pivot,
                cluster_assignments=cluster_assignments,
            )

            if cluster_forecasts.empty:
                self._store_equal_fallback_result(
                    ticker=ticker,
                    model_names=available_models,
                    reason="unable to build cluster forecasts",
                )
                continue

            corr_matrix = _compute_cluster_correlation(cluster_forecasts)
            cluster_names = list(cluster_forecasts.columns)
            cluster_count = len(cluster_names)

            if cluster_count <= 1:
                self._store_single_cluster_result(
                    ticker=ticker,
                    model_names=available_models,
                    cluster_assignments=cluster_assignments,
                )
                continue

            if self.weight_method == "cluster_equal":
                raw_weights = np.ones(cluster_count) / cluster_count
                metrics = {
                    cluster: {
                        "avg_positive_corr": float(corr_matrix.loc[cluster].drop(cluster).mean()),
                        "ulcer_index": None,
                        "score": None,
                        "member_count": int(
                            sum(
                                1
                                for model in available_models
                                if cluster_assignments[model] == cluster
                            )
                        ),
                    }
                    for cluster in cluster_names
                }
            else:
                assert normalized_returns is not None
                aligned_returns = normalized_returns.reindex(cluster_forecasts.index).fillna(0.0)
                cluster_return_proxy = cluster_forecasts.multiply(aligned_returns, axis=0)
                raw_scores = []
                metrics = {}
                for cluster in cluster_names:
                    avg_positive_corr = float(corr_matrix.loc[cluster].drop(cluster).mean())
                    ulcer_index = _compute_ulcer_index(cluster_return_proxy[cluster])
                    score = 1.0 / (
                        (max(ulcer_index, _EPSILON) ** self._wl_config.risk_tilt_alpha)
                        * (1.0 + avg_positive_corr)
                    )
                    raw_scores.append(score)
                    metrics[cluster] = {
                        "avg_positive_corr": avg_positive_corr,
                        "ulcer_index": ulcer_index,
                        "score": float(score),
                        "member_count": int(
                            sum(
                                1
                                for model in available_models
                                if cluster_assignments[model] == cluster
                            )
                        ),
                    }
                raw_weights = np.asarray(raw_scores, dtype=float)
                if raw_weights.sum() <= 0.0:
                    raw_weights = np.ones(cluster_count) / cluster_count
                else:
                    raw_weights = raw_weights / raw_weights.sum()

            capped_weights = _apply_group_weight_cap(
                raw_weights,
                self._wl_config.group_weight_cap,
            )
            cluster_weights = {
                cluster: float(weight)
                for cluster, weight in zip(cluster_names, capped_weights)
            }
            model_weights = _distribute_cluster_weights_to_models(
                model_names=available_models,
                cluster_assignments=cluster_assignments,
                cluster_weights=cluster_weights,
            )

            self.weights_[ticker] = model_weights
            self.model_names_[ticker] = available_models
            self.cluster_assignments_[ticker] = cluster_assignments
            self.cluster_weights_[ticker] = cluster_weights
            self.cluster_metrics_[ticker] = metrics
            self.mean_cluster_correlation_[ticker] = _mean_off_diagonal_correlation(corr_matrix)
            self.fdm_[ticker] = _compute_fdm_from_corr_matrix(
                corr_matrix=corr_matrix,
                fdm_max=self._wl_config.fdm_max,
            )

        self.is_fitted_ = True
        return self

    def _store_single_cluster_result(
        self,
        ticker: str,
        model_names: List[str],
        cluster_assignments: Optional[Dict[str, str]] = None,
    ) -> None:
        cluster_map = cluster_assignments or {model: "cluster_1" for model in model_names}
        equal_weight = 1.0 / len(model_names) if model_names else 1.0
        self.weights_[ticker] = pd.Series({model: equal_weight for model in model_names})
        self.model_names_[ticker] = model_names
        self.cluster_assignments_[ticker] = cluster_map
        self.cluster_weights_[ticker] = {"cluster_1": 1.0}
        self.cluster_metrics_[ticker] = {
            "cluster_1": {
                "avg_positive_corr": 0.0,
                "ulcer_index": None if self.weight_method == "cluster_equal" else 0.0,
                "score": None if self.weight_method == "cluster_equal" else 1.0,
                "member_count": len(model_names),
            }
        }
        self.mean_cluster_correlation_[ticker] = 1.0
        self.fdm_[ticker] = 1.0

    def _store_equal_fallback_result(
        self,
        ticker: str,
        model_names: List[str],
        reason: str,
    ) -> None:
        logger.warning("Ticker %s falling back to one-cluster equal weights: %s", ticker, reason)
        self._store_single_cluster_result(ticker=ticker, model_names=model_names)


def WeightLayer(
    weight_method: str = "cluster_equal",
    fdm_max: float = 2.0,
    config: Optional[WeightLayerConfig] = None,
    **kwargs: float,
) -> BaseWeightLayer:
    """Factory for the clustered weight layer."""
    if config is None:
        config = WeightLayerConfig(
            weighting_method=weight_method,
            fdm_max=float(kwargs.pop("fdm_max", fdm_max)),
            rho_cut=float(kwargs.pop("rho_cut", 0.70)),
            group_weight_cap=float(kwargs.pop("group_weight_cap", 0.25)),
            risk_tilt_alpha=float(kwargs.pop("risk_tilt_alpha", 0.5)),
        )
    elif kwargs:
        raise ValueError("Pass either config or keyword overrides, not both")

    return ClusteredWeightLayer(config=config)


__all__ = [
    "WeightLayerConfig",
    "WeightLayer",
    "BaseWeightLayer",
    "ClusteredWeightLayer",
]
