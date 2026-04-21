"""Weight layer: equal, inverse-correlation, and manual hierarchy allocation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, fields
from dataclasses import dataclass as _dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ensemble.weight_hierarchy import (
    compute_equal_split_weights,
    resolve_hierarchy_for_fit,
)

logger = logging.getLogger(__name__)

_WEIGHT_METHODS = {
    "equal_signal",
    "inverse_avg_pairwise_corr",
    "hierarchy_equal",
    "inverse_corr_hierarchy",
    "ledoit_wolf_min_corr",
    "risk_parity_corr",
    "hierarchy_theme_inv_corr",
    "hierarchy_theme_ledoit",
    "ledoit_wolf_hierarchy_within",
}

_LEGACY_REMOVED_METHODS = frozenset(
    {
        "hrp_cluster_equal",
        "hrp_classic",
        "optimize_sortino_capped",
    }
)


@_dataclass(frozen=True)
class WeightLayerConfig:
    """Configuration for the weight layer."""

    weighting_method: str = "equal_signal"
    fdm_max: float = 2.0
    hierarchy_spec: Optional[dict[str, object]] = None
    hierarchy_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.weighting_method not in _WEIGHT_METHODS:
            raise ValueError(
                f"weighting_method must be one of {sorted(_WEIGHT_METHODS)}, "
                f"got '{self.weighting_method}'"
            )
        if self.fdm_max <= 0.0:
            raise ValueError(f"fdm_max must be > 0, got {self.fdm_max}")
        if self.weighting_method in (
            "hierarchy_equal",
            "inverse_corr_hierarchy",
            "hierarchy_theme_inv_corr",
            "hierarchy_theme_ledoit",
            "ledoit_wolf_hierarchy_within",
        ):
            has_spec = self.hierarchy_spec is not None and len(self.hierarchy_spec) > 0
            has_path = self.hierarchy_path is not None and str(self.hierarchy_path).strip() != ""
            if not has_spec and not has_path:
                raise ValueError(
                    f"{self.weighting_method} requires a non-empty hierarchy_spec or hierarchy_path"
                )


class BaseWeightLayer(ABC):
    """Abstract interface for weight-layer implementations."""

    def __init__(self, fdm_max: float = 2.0) -> None:
        self.fdm_max = fdm_max
        self.fdm_: Dict[str, float] = {}
        self.weights_: Dict[str, pd.Series] = {}
        self.model_names_: Dict[str, List[str]] = {}
        self.mean_signal_correlation_: Dict[str, float] = {}
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
        returns: Optional[pd.Series | pd.DataFrame] = None,
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
            ticker_weights = pd.Series({m: equal_weight for m in available_models}, dtype=float)
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
                "mean_signal_correlation": self.mean_signal_correlation_.get(ticker, 1.0),
                "mean_cluster_correlation": self.mean_cluster_correlation_.get(ticker, 1.0),
            }

        fdm_values = list(self.fdm_.values())
        n_models_per_ticker = [len(self.model_names_.get(ticker, [])) for ticker in self.fdm_]
        cluster_counts = [len(self.cluster_weights_.get(ticker, {})) for ticker in self.fdm_]

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
    return pivot[existing_models].dropna(how="all")


def _safe_normalize(weights: pd.Series | np.ndarray, labels: List[str]) -> pd.Series:
    values = weights.to_numpy(dtype=float) if isinstance(weights, pd.Series) else np.asarray(weights, dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        equal_weight = 1.0 / len(labels) if labels else 1.0
        return pd.Series({label: equal_weight for label in labels}, dtype=float)
    return pd.Series({label: float(value / total) for label, value in zip(labels, values)}, dtype=float)


def _positive_clipped_correlation(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    clipped = corr_matrix.clip(lower=0.0)
    np.fill_diagonal(clipped.values, 1.0)
    return clipped


def _mean_off_diagonal_correlation(corr_matrix: pd.DataFrame) -> float:
    n_cols = len(corr_matrix.columns)
    if n_cols <= 1:
        return 1.0
    mask = np.triu(np.ones((n_cols, n_cols), dtype=bool), k=1)
    correlations = corr_matrix.to_numpy(dtype=float)[mask]
    if len(correlations) == 0:
        return 1.0
    return float(correlations.mean())


def _correlation_multiplier_from_corr_matrix(
    corr_matrix: pd.DataFrame,
    *,
    cap: float,
    epsilon: float = 0.01,
) -> tuple[float, float]:
    """Return the mean off-diagonal correlation and capped diversification multiplier."""
    if len(corr_matrix.columns) <= 1:
        return 1.0, 1.0

    positive_corr = _positive_clipped_correlation(corr_matrix.copy())
    mean_corr = _mean_off_diagonal_correlation(positive_corr)
    multiplier = min(float(np.sqrt(1.0 / (mean_corr + epsilon))), cap)
    return mean_corr, multiplier


def _compute_fdm_from_corr_matrix(corr_matrix: pd.DataFrame, fdm_max: float) -> float:
    _, fdm = _correlation_multiplier_from_corr_matrix(corr_matrix, cap=fdm_max)
    return fdm


def _covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    std = np.sqrt(np.clip(np.diag(covariance.to_numpy(dtype=float)), 0.0, None))
    scale = np.outer(std, std)
    corr_values = np.divide(
        covariance.to_numpy(dtype=float),
        scale,
        out=np.zeros_like(covariance.to_numpy(dtype=float)),
        where=scale > 0.0,
    )
    corr_values = np.clip(corr_values, -1.0, 1.0)
    np.fill_diagonal(corr_values, 1.0)
    return pd.DataFrame(corr_values, index=covariance.index, columns=covariance.columns)


def _prepare_signal_matrix(forecast_pivot: pd.DataFrame) -> pd.DataFrame:
    filled = forecast_pivot.fillna(0.0).astype(float)
    sample_std = filled.std(axis=0, ddof=1)
    scale = sample_std.replace(0.0, 1.0).fillna(1.0)
    return filled.divide(scale, axis=1)


def _estimate_covariance_and_correlation(
    standardized_pivot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimator = LedoitWolf()
    estimator.fit(standardized_pivot.to_numpy(dtype=float))
    covariance = pd.DataFrame(
        estimator.covariance_,
        index=standardized_pivot.columns,
        columns=standardized_pivot.columns,
    )
    correlation = _covariance_to_correlation(covariance)
    positive_corr = _positive_clipped_correlation(correlation.copy())
    return covariance, correlation, positive_corr


def _build_singleton_assignments(model_names: List[str]) -> Dict[str, str]:
    return {model: f"cluster_{idx}" for idx, model in enumerate(model_names, start=1)}


def _build_singleton_metrics(
    weights: pd.Series,
    positive_corr: pd.DataFrame,
    *,
    include_scores: bool,
) -> tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float | int | None]]]:
    assignments = _build_singleton_assignments(list(weights.index))
    cluster_weights = {
        assignments[model_name]: float(weight)
        for model_name, weight in weights.items()
    }
    metrics = {
        assignments[model_name]: {
            "avg_positive_corr": _average_peer_correlation(positive_corr, model_name),
            "ulcer_index": None,
            "score": (
                float(1.0 / (1.0 + _average_peer_correlation(positive_corr, model_name)))
                if include_scores
                else None
            ),
            "member_count": 1,
        }
        for model_name in weights.index
    }
    return assignments, cluster_weights, metrics


def _average_peer_correlation(corr_matrix: pd.DataFrame, label: str) -> float:
    if label not in corr_matrix.columns or len(corr_matrix.columns) <= 1:
        return 0.0
    peers = corr_matrix.loc[label].drop(label)
    return float(peers.mean()) if len(peers) > 0 else 0.0


def _group_corr_scores(
    members: List[str],
    positive_corr: pd.DataFrame,
    *,
    mode: str,
) -> np.ndarray:
    """Per-member score from positive_corr using 'inv_avg' or 'ledoit' penalty."""
    if mode == "inv_avg":
        return np.asarray(
            [1.0 / (1.0 + _average_peer_correlation(positive_corr, m)) for m in members],
            dtype=float,
        )
    # ledoit: 1 / sum of row correlations
    return np.asarray(
        [1.0 / max(float(positive_corr.loc[m].sum()), 0.01) if m in positive_corr.index else 1.0
         for m in members],
        dtype=float,
    )


def _theme_aggregate_corr(
    theme_members: Dict[str, List[str]],
    positive_corr: pd.DataFrame,
) -> pd.DataFrame:
    """Build a theme × theme correlation matrix by averaging member signal correlations."""
    theme_ids = list(theme_members.keys())
    n = len(theme_ids)
    mat = np.ones((n, n), dtype=float)
    for i, ti in enumerate(theme_ids):
        for j, tj in enumerate(theme_ids):
            if i == j:
                continue
            cross_corrs = [
                float(positive_corr.loc[a, b])
                for a in theme_members[ti]
                for b in theme_members[tj]
                if a in positive_corr.index and b in positive_corr.columns
            ]
            mat[i, j] = float(np.mean(cross_corrs)) if cross_corrs else 0.0
    return pd.DataFrame(mat, index=theme_ids, columns=theme_ids)


def _equal_weights(model_names: List[str]) -> pd.Series:
    equal_weight = 1.0 / len(model_names) if model_names else 1.0
    return pd.Series({model: equal_weight for model in model_names}, dtype=float)


class ClusteredWeightLayer(BaseWeightLayer):
    """Weight layer: equal, inverse-correlation, or manual hierarchy (equal split)."""

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
        returns: Optional[pd.Series | pd.DataFrame] = None,
    ) -> "ClusteredWeightLayer":
        del signals
        del returns

        if not forecast_vectors:
            raise ValueError("forecast_vectors cannot be empty")

        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        if all_forecasts.empty:
            raise ValueError("forecast_vectors contain no data")
        if "ticker" not in all_forecasts.columns:
            raise ValueError("forecast_vectors must contain 'ticker' column")

        self.fdm_.clear()
        self.weights_.clear()
        self.model_names_.clear()
        self.mean_signal_correlation_.clear()
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

            if len(available_models) == 1:
                self._store_single_model_result(ticker=ticker, model_name=available_models[0])
                continue

            forecast_pivot = _pivot_ticker_forecasts(ticker_forecasts, available_models)
            if forecast_pivot.empty or len(forecast_pivot) < 2:
                self._store_equal_fallback_result(
                    ticker=ticker,
                    model_names=available_models,
                    reason="insufficient forecast history for covariance estimation",
                )
                continue
            if int(forecast_pivot.nunique(dropna=True).max()) <= 1:
                self._store_equal_fallback_result(
                    ticker=ticker,
                    model_names=available_models,
                    reason="all forecast vectors are constant in-sample",
                )
                continue

            try:
                signal_pivot = _prepare_signal_matrix(forecast_pivot)
                _covariance, _full_corr, positive_corr = _estimate_covariance_and_correlation(
                    signal_pivot
                )
            except Exception as exc:
                logger.warning(
                    "Ticker %s falling back to equal weights after covariance failure: %s",
                    ticker,
                    exc,
                )
                self._store_equal_fallback_result(
                    ticker=ticker,
                    model_names=available_models,
                    reason="invalid covariance inputs",
                )
                continue

            mean_signal_corr = _mean_off_diagonal_correlation(positive_corr)
            fdm = _compute_fdm_from_corr_matrix(
                corr_matrix=positive_corr,
                fdm_max=self._wl_config.fdm_max,
            )

            if self.weight_method == "equal_signal":
                model_weights = _equal_weights(available_models)
                cluster_assignments, cluster_weights, cluster_metrics = _build_singleton_metrics(
                    model_weights,
                    positive_corr,
                    include_scores=False,
                )
                mean_cluster_corr = mean_signal_corr
            elif self.weight_method == "inverse_avg_pairwise_corr":
                raw_scores = np.asarray(
                    [
                        1.0 / (1.0 + _average_peer_correlation(positive_corr, model_name))
                        for model_name in available_models
                    ],
                    dtype=float,
                )
                model_weights = _safe_normalize(raw_scores, available_models)
                cluster_assignments, cluster_weights, cluster_metrics = _build_singleton_metrics(
                    model_weights,
                    positive_corr,
                    include_scores=True,
                )
                mean_cluster_corr = mean_signal_corr
            elif self.weight_method == "inverse_corr_hierarchy":
                root = resolve_hierarchy_for_fit(
                    hierarchy_spec=self._wl_config.hierarchy_spec,
                    hierarchy_path=self._wl_config.hierarchy_path,
                )
                equal_weights, cluster_assignments, cluster_weights, cluster_metrics = (
                    compute_equal_split_weights(root, available_models)
                )
                # Invert each group's equal share by within-group corr; preserve group mass.
                group_to_members: Dict[str, List[str]] = {}
                for model_name, path in cluster_assignments.items():
                    group = "/".join(path.split("/")[:-1])
                    group_to_members.setdefault(group, []).append(model_name)
                adjusted: Dict[str, float] = {}
                for group, members in group_to_members.items():
                    group_mass = sum(float(equal_weights[m]) for m in members)
                    raw = np.asarray(
                        [1.0 / (1.0 + _average_peer_correlation(positive_corr, m)) for m in members],
                        dtype=float,
                    )
                    total = float(raw.sum())
                    for m, r in zip(members, raw):
                        adjusted[m] = group_mass * (r / total if total > 0.0 else 1.0 / len(members))
                model_weights = pd.Series(adjusted, dtype=float)
                mean_cluster_corr = mean_signal_corr
            elif self.weight_method == "ledoit_wolf_min_corr":
                col_sums = positive_corr.sum(axis=1)
                raw_scores = np.asarray(1.0 / col_sums.clip(lower=0.01), dtype=float)
                model_weights = _safe_normalize(raw_scores, available_models)
                cluster_assignments, cluster_weights, cluster_metrics = _build_singleton_metrics(
                    model_weights, positive_corr, include_scores=True
                )
                mean_cluster_corr = mean_signal_corr
            elif self.weight_method == "risk_parity_corr":
                col_sums = positive_corr.sum(axis=1)
                raw_scores = np.asarray(1.0 / np.sqrt(col_sums.clip(lower=0.01)), dtype=float)
                model_weights = _safe_normalize(raw_scores, available_models)
                cluster_assignments, cluster_weights, cluster_metrics = _build_singleton_metrics(
                    model_weights, positive_corr, include_scores=True
                )
                mean_cluster_corr = mean_signal_corr
            elif self.weight_method in (
                "hierarchy_theme_inv_corr",
                "hierarchy_theme_ledoit",
                "ledoit_wolf_hierarchy_within",
            ):
                root = resolve_hierarchy_for_fit(
                    hierarchy_spec=self._wl_config.hierarchy_spec,
                    hierarchy_path=self._wl_config.hierarchy_path,
                )
                equal_weights, cluster_assignments, cluster_weights, cluster_metrics = (
                    compute_equal_split_weights(root, available_models)
                )
                # Map each top-level theme to its member models.
                theme_to_members: Dict[str, List[str]] = {}
                for model_name, path in cluster_assignments.items():
                    # path is e.g. "root/theme/model" — theme is parts[1]
                    parts = path.split("/")
                    theme = parts[1] if len(parts) >= 2 else parts[0]
                    theme_to_members.setdefault(theme, []).append(model_name)

                theme_corr = _theme_aggregate_corr(theme_to_members, positive_corr)
                theme_ids = list(theme_to_members.keys())

                # Score themes by inter-theme correlation.
                penalty_mode = "ledoit" if self.weight_method in (
                    "hierarchy_theme_ledoit", "ledoit_wolf_hierarchy_within"
                ) else "inv_avg"
                theme_raw = _group_corr_scores(theme_ids, theme_corr, mode=penalty_mode)
                theme_raw_total = float(theme_raw.sum())
                theme_masses = {
                    t: (theme_raw[i] / theme_raw_total if theme_raw_total > 0.0 else 1.0 / len(theme_ids))
                    for i, t in enumerate(theme_ids)
                }

                # Distribute each theme's mass to its members.
                adjusted_weights: Dict[str, float] = {}
                for theme, members in theme_to_members.items():
                    mass = theme_masses[theme]
                    if self.weight_method == "ledoit_wolf_hierarchy_within":
                        within_raw = _group_corr_scores(members, positive_corr, mode="ledoit")
                    else:
                        # equal within each theme
                        within_raw = np.ones(len(members), dtype=float)
                    within_total = float(within_raw.sum())
                    for m, r in zip(members, within_raw):
                        adjusted_weights[m] = mass * (r / within_total if within_total > 0.0 else 1.0 / len(members))

                model_weights = pd.Series(adjusted_weights, dtype=float)
                mean_cluster_corr = mean_signal_corr
            else:
                root = resolve_hierarchy_for_fit(
                    hierarchy_spec=self._wl_config.hierarchy_spec,
                    hierarchy_path=self._wl_config.hierarchy_path,
                )
                model_weights, cluster_assignments, cluster_weights, cluster_metrics = (
                    compute_equal_split_weights(root, available_models)
                )
                mean_cluster_corr = mean_signal_corr

            self.weights_[ticker] = model_weights
            self.model_names_[ticker] = available_models
            self.cluster_assignments_[ticker] = cluster_assignments
            self.cluster_weights_[ticker] = cluster_weights
            self.cluster_metrics_[ticker] = cluster_metrics
            self.mean_signal_correlation_[ticker] = mean_signal_corr
            self.mean_cluster_correlation_[ticker] = mean_cluster_corr
            self.fdm_[ticker] = fdm

        self.is_fitted_ = True
        return self

    def _store_single_model_result(self, ticker: str, model_name: str) -> None:
        self.weights_[ticker] = pd.Series({model_name: 1.0}, dtype=float)
        self.model_names_[ticker] = [model_name]
        self.cluster_assignments_[ticker] = {model_name: "cluster_1"}
        self.cluster_weights_[ticker] = {"cluster_1": 1.0}
        self.cluster_metrics_[ticker] = {
            "cluster_1": {
                "avg_positive_corr": 0.0,
                "ulcer_index": None,
                "score": None,
                "member_count": 1,
            }
        }
        self.mean_signal_correlation_[ticker] = 1.0
        self.mean_cluster_correlation_[ticker] = 1.0
        self.fdm_[ticker] = 1.0

    def _store_equal_fallback_result(
        self,
        ticker: str,
        model_names: List[str],
        reason: str,
    ) -> None:
        logger.warning("Ticker %s falling back to equal weights: %s", ticker, reason)
        equal_weights = _equal_weights(model_names)
        cluster_assignments = _build_singleton_assignments(model_names)
        cluster_weights = {
            cluster_assignments[model_name]: float(equal_weights[model_name])
            for model_name in model_names
        }
        cluster_metrics = {
            cluster_assignments[model_name]: {
                "avg_positive_corr": 0.0,
                "ulcer_index": None,
                "score": None,
                "member_count": 1,
            }
            for model_name in model_names
        }
        self.weights_[ticker] = equal_weights
        self.model_names_[ticker] = model_names
        self.cluster_assignments_[ticker] = cluster_assignments
        self.cluster_weights_[ticker] = cluster_weights
        self.cluster_metrics_[ticker] = cluster_metrics
        self.mean_signal_correlation_[ticker] = 1.0
        self.mean_cluster_correlation_[ticker] = 1.0
        self.fdm_[ticker] = 1.0


def WeightLayer(
    weight_method: str = "equal_signal",
    fdm_max: float = 2.0,
    config: Optional[WeightLayerConfig] = None,
    **kwargs: object,
) -> BaseWeightLayer:
    """Factory for the weight layer."""
    if config is None:
        if "risk_tilt_alpha" in kwargs:
            raise ValueError("risk_tilt_alpha is no longer supported by WeightLayer")
        hierarchy_spec = kwargs.pop("hierarchy_spec", None)
        hierarchy_path_raw = kwargs.pop("hierarchy_path", None)
        hp: str | None = None
        if hierarchy_path_raw is not None:
            hp = str(hierarchy_path_raw).strip() or None
        config = WeightLayerConfig(
            weighting_method=weight_method,
            fdm_max=float(kwargs.pop("fdm_max", fdm_max)),
            hierarchy_spec=dict(hierarchy_spec) if isinstance(hierarchy_spec, dict) else None,
            hierarchy_path=hp,
        )
        if kwargs:
            raise TypeError(
                f"WeightLayer got unexpected keyword arguments: {sorted(kwargs.keys())}"
            )
    elif kwargs:
        raise ValueError("Pass either config or keyword overrides, not both")

    return ClusteredWeightLayer(config=config)


def serialize_weight_layer_state(weight_layer: BaseWeightLayer) -> Dict[str, Any]:
    """Serialize a fitted or unfitted weight layer to JSON-safe payload."""
    if not isinstance(weight_layer, ClusteredWeightLayer):
        raise TypeError(
            "Only ClusteredWeightLayer serialization is currently supported; "
            f"got {weight_layer.__class__.__name__}"
        )

    return {
        "class_name": weight_layer.__class__.__name__,
        "config": asdict(weight_layer._wl_config),
        "state": {
            "fdm": {str(k): float(v) for k, v in weight_layer.fdm_.items()},
            "weights": {
                str(ticker): {
                    str(model_name): float(weight)
                    for model_name, weight in series.to_dict().items()
                }
                for ticker, series in weight_layer.weights_.items()
            },
            "model_names": {
                str(ticker): [str(model_name) for model_name in model_names]
                for ticker, model_names in weight_layer.model_names_.items()
            },
            "mean_signal_correlation": {
                str(k): float(v) for k, v in weight_layer.mean_signal_correlation_.items()
            },
            "mean_cluster_correlation": {
                str(k): float(v) for k, v in weight_layer.mean_cluster_correlation_.items()
            },
            "cluster_assignments": {
                str(ticker): {str(model_name): str(cluster) for model_name, cluster in payload.items()}
                for ticker, payload in weight_layer.cluster_assignments_.items()
            },
            "cluster_weights": {
                str(ticker): {str(cluster): float(weight) for cluster, weight in payload.items()}
                for ticker, payload in weight_layer.cluster_weights_.items()
            },
            "cluster_metrics": {
                str(ticker): {
                    str(cluster): {
                        str(metric_name): (
                            None
                            if metric_value is None
                            else int(metric_value)
                            if isinstance(metric_value, (int, np.integer)) and not isinstance(metric_value, bool)
                            else float(metric_value)
                            if isinstance(metric_value, (float, np.floating))
                            else metric_value
                        )
                        for metric_name, metric_value in metrics.items()
                    }
                    for cluster, metrics in payload.items()
                }
                for ticker, payload in weight_layer.cluster_metrics_.items()
            },
            "is_fitted": bool(weight_layer.is_fitted_),
        },
    }


def deserialize_weight_layer_state(payload: Dict[str, Any]) -> BaseWeightLayer:
    """Restore a weight layer from ``serialize_weight_layer_state`` payload."""
    config_payload = dict(payload.get("config", {}))
    known = {f.name for f in fields(WeightLayerConfig)}
    filtered = {k: v for k, v in config_payload.items() if k in known}
    wm = filtered.get("weighting_method")
    if wm in _LEGACY_REMOVED_METHODS:
        raise ValueError(
            "Cannot deserialize weight layer: weighting_method "
            f"'{wm}' was removed (HRP / legacy). Re-fit the portfolio with "
            "equal_signal, inverse_avg_pairwise_corr, or hierarchy_equal."
        )
    config = WeightLayerConfig(**filtered)
    restored = WeightLayer(config=config)
    if not isinstance(restored, ClusteredWeightLayer):
        raise TypeError(
            "WeightLayer(config=...) did not return ClusteredWeightLayer during restore"
        )

    state = payload.get("state", {})
    restored.fdm_ = {
        str(ticker): float(value)
        for ticker, value in dict(state.get("fdm", {})).items()
    }
    restored.weights_ = {
        str(ticker): pd.Series(
            {str(model_name): float(weight) for model_name, weight in dict(weights).items()},
            dtype=float,
        )
        for ticker, weights in dict(state.get("weights", {})).items()
    }
    restored.model_names_ = {
        str(ticker): [str(model_name) for model_name in model_names]
        for ticker, model_names in dict(state.get("model_names", {})).items()
    }
    restored.mean_signal_correlation_ = {
        str(ticker): float(value)
        for ticker, value in dict(state.get("mean_signal_correlation", {})).items()
    }
    restored.mean_cluster_correlation_ = {
        str(ticker): float(value)
        for ticker, value in dict(state.get("mean_cluster_correlation", {})).items()
    }
    restored.cluster_assignments_ = {
        str(ticker): {str(model_name): str(cluster) for model_name, cluster in dict(assignments).items()}
        for ticker, assignments in dict(state.get("cluster_assignments", {})).items()
    }
    restored.cluster_weights_ = {
        str(ticker): {str(cluster): float(weight) for cluster, weight in dict(weights).items()}
        for ticker, weights in dict(state.get("cluster_weights", {})).items()
    }
    restored.cluster_metrics_ = {
        str(ticker): {
            str(cluster): {
                str(metric_name): metric_value
                for metric_name, metric_value in dict(metrics).items()
            }
            for cluster, metrics in dict(payload_by_cluster).items()
        }
        for ticker, payload_by_cluster in dict(state.get("cluster_metrics", {})).items()
    }
    restored.is_fitted_ = bool(state.get("is_fitted", False))
    return restored


__all__ = [
    "WeightLayerConfig",
    "WeightLayer",
    "BaseWeightLayer",
    "ClusteredWeightLayer",
    "deserialize_weight_layer_state",
    "serialize_weight_layer_state",
]
