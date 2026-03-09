"""
Weight Layer for Forecast Combination

This module implements the Weight Layer that combines forecasts from all base models
across all ensembles using configurable weight strategies and applies the Forecast
Diversification Multiplier (FDM).

The Weight Layer sits between Ensemble and Portfolio layers:
    Ensemble.predict() -> WeightLayer.combine() -> Portfolio.predict()

Supported weight strategies:
- InverseCorrelationWeightLayer: Inverse correlation weights (Carver's method)

Use the ``WeightLayer`` factory function to create instances by name:
    >>> layer = WeightLayer(weight_method='inverse_correlation')

Reference: Robert Carver's "Systematic Trading" and "Leveraged Trading"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass as _dataclass
from typing import Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WeightLayerConfig — pre-committed configuration for method selection
# ---------------------------------------------------------------------------

@_dataclass(frozen=True)
class WeightLayerConfig:
    """Pre-committed configuration for weight layer method selection.

    Attributes
    ----------
    weighting_method : str
        One of: ``"equal_flat"``, ``"equal_grouped"``,
        ``"inv_downside_vol_grouped"``, ``"downside_hrp_grouped"``,
        ``"downside_hrp_flat"``, ``"inverse_correlation"`` (backward-compat).
    group_method : str
        ``"feature_family"`` (extract module prefix from model name) or
        ``"correlation_clustering"`` (hierarchical clustering with rho_cut).
    rho_cut : float
        Correlation cutoff for ``correlation_clustering`` (default 0.70).
    within_group_weights : str
        ``"equal"`` — only supported option currently.
    linkage : str
        Linkage method for Ward-HRP clustering: ``"ward"``, ``"complete"``, ``"single"``.
    shrinkage : str
        ``"ledoit_wolf"`` or ``"none"`` for downside semi-covariance estimation.
    fdm_max : float
        Cap on FDM (default 2.0, per spec).
    fdm_correlation_source : str
        ``"downside"`` — use downside correlation matrix for FDM computation.
        ``"full_period"`` — use full-period correlation (for equal-weight methods).
    weight_stability_threshold : float
        Diagnostic: warn if any group weight shifts more than this across folds.
    """
    weighting_method: str = "inverse_correlation"
    group_method: str = "feature_family"
    rho_cut: float = 0.70
    within_group_weights: str = "equal"
    linkage: str = "ward"
    shrinkage: str = "ledoit_wolf"
    fdm_max: float = 2.0
    fdm_correlation_source: str = "downside"
    weight_stability_threshold: float = 0.20

    def __post_init__(self) -> None:
        valid_methods = {
            "equal_flat", "equal_grouped", "inv_downside_vol_grouped",
            "downside_hrp_grouped", "downside_hrp_flat", "inverse_correlation",
        }
        if self.weighting_method not in valid_methods:
            raise ValueError(
                f"weighting_method must be one of {sorted(valid_methods)}, "
                f"got '{self.weighting_method}'"
            )
        if self.group_method not in ("feature_family", "correlation_clustering"):
            raise ValueError(
                f"group_method must be 'feature_family' or 'correlation_clustering', "
                f"got '{self.group_method}'"
            )


# ---------------------------------------------------------------------------
# Weighter protocol (kept for backward compatibility with tests)
# ---------------------------------------------------------------------------

class Weighter(Protocol):
    """Protocol for weight calculation methods."""

    def fit(self, signals: pd.DataFrame) -> None:
        """Fit weights from binary signals."""
        ...

    def get_weights(self) -> pd.Series:
        """Get fitted weights."""
        ...


# ---------------------------------------------------------------------------
# InverseCorrelationWeighter (standalone helper, used by InverseCorrelationWeightLayer)
# ---------------------------------------------------------------------------

class InverseCorrelationWeighter:
    """
    Calculate weights based on inverse correlation between base model signals.

    Models with lower average correlation to other models get higher weights,
    as they provide more diversification benefit.

    Formula:
        d_i = 1 / (1 + avg_abs_corr_i)  (diversification score)
        w_i = d_i / sum(d_j)            (normalized weight)

    Attributes
    ----------
    weights_ : pd.Series
        Model name -> weight mapping after fitting
    correlation_matrix_ : pd.DataFrame
        Correlation matrix of signals (stored for diagnostics)
    is_fitted_ : bool
        Whether the weighter has been fitted
    """

    def __init__(self) -> None:
        self.weights_: Optional[pd.Series] = None
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False

    def fit(self, signals: pd.DataFrame) -> InverseCorrelationWeighter:
        """
        Fit weights from binary signals.

        Parameters
        ----------
        signals : pd.DataFrame
            Binary signals from all base models.
            Columns: model names, rows: samples
            Values should be 0 or 1

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If signals DataFrame is empty or has insufficient data
        """
        if signals.empty:
            raise ValueError("signals DataFrame cannot be empty")

        if len(signals) < 2:
            raise ValueError("Need at least 2 samples to calculate correlations")

        # Calculate correlation matrix (use absolute values)
        # Handle constant signals (all 0 or all 1) which result in NaN correlations
        self.correlation_matrix_ = signals.corr().abs()

        # Fill NaN correlations with 0 (constant signals = zero correlation = max diversification)
        # This happens when a signal is constant (all 0 or all 1)
        self.correlation_matrix_ = self.correlation_matrix_.fillna(0.0)

        # Handle single model case
        if len(signals.columns) == 1:
            model_name = signals.columns[0]
            self.weights_ = pd.Series({model_name: 1.0})
            self.is_fitted_ = True
            return self

        # For each model, calculate average correlation with others
        avg_correlations = {
            model_name: (
                self.correlation_matrix_.loc[
                    model_name,
                    [m for m in signals.columns if m != model_name],
                ].mean()
            )
            for model_name in signals.columns
        }
        # Replace NaN averages with 0 (maximum diversification)
        avg_correlations = {
            k: (0.0 if pd.isna(v) else v)
            for k, v in avg_correlations.items()
        }

        # Convert to diversification scores (inverse relationship)
        # Higher correlation = lower diversification score
        diversification_scores = {
            model: 1.0 / (1.0 + avg_corr)
            for model, avg_corr in avg_correlations.items()
        }

        # Normalize to sum to 1.0
        total = sum(diversification_scores.values())
        if total > 0:
            self.weights_ = pd.Series({
                model: score / total
                for model, score in diversification_scores.items()
            })
            # Ensure no NaN values in weights
            if self.weights_.isna().any():
                n_models = len(self.weights_)
                equal_weight = 1.0 / n_models
                self.weights_ = self.weights_.fillna(equal_weight)
                total = self.weights_.sum()
                if total > 0:
                    self.weights_ = self.weights_ / total
        else:
            # Fallback: equal weights if total is zero (shouldn't happen)
            equal_weight = 1.0 / len(signals.columns)
            self.weights_ = pd.Series({
                model: equal_weight
                for model in signals.columns
            })

        self.is_fitted_ = True
        return self

    def get_weights(self) -> pd.Series:
        """
        Get fitted weights.

        Returns
        -------
        pd.Series
            Model name -> weight mapping

        Raises
        ------
        ValueError
            If weighter has not been fitted
        """
        if not self.is_fitted_:
            raise ValueError("InverseCorrelationWeighter must be fitted before getting weights")
        return self.weights_


# ---------------------------------------------------------------------------
# BaseWeightLayer (Abstract Base – Template Method)
# ---------------------------------------------------------------------------

class BaseWeightLayer(ABC):
    """
    Abstract base for forecast combination weight layers.

    Shared responsibilities (implemented here):
    1. Orchestrate per-ticker weight fitting via ``fit()``
    2. Calculate FDM from forecast value correlations
    3. Combine forecasts: ``weighted_sum * FDM``
    4. Return diagnostics

    Subclasses implement one hook:
        ``_fit_ticker_weights(ticker_signals, ticker_forecast_vectors) -> pd.Series``

    Parameters
    ----------
    fdm_max : float, default=2.0
        Maximum FDM value (per project spec)

    Attributes
    ----------
    fdm_ : Dict[str, float]
        Ticker -> Fitted Forecast Diversification Multiplier
    weights_ : Dict[str, pd.Series]
        Ticker -> Model name -> weight mapping
    model_names_ : Dict[str, List[str]]
        Ticker -> List of model names in order
    mean_forecast_correlation_ : Dict[str, float]
        Ticker -> Mean correlation between forecasts
    is_fitted_ : bool
        Whether the layer has been fitted
    weight_method : str
        Name of the weighting strategy (set by subclasses)
    """

    def __init__(self, fdm_max: float = 2.0) -> None:
        self.fdm_max = fdm_max

        # Fitted attributes (per-ticker storage)
        self.fdm_: Dict[str, float] = {}
        self.weights_: Dict[str, pd.Series] = {}
        self.model_names_: Dict[str, List[str]] = {}
        self.mean_forecast_correlation_: Dict[str, float] = {}
        self.is_fitted_: bool = False

    # -- abstract hook --------------------------------------------------

    @property
    @abstractmethod
    def weight_method(self) -> str:
        """Return the canonical name of this weighting strategy."""
        ...

    @abstractmethod
    def _fit_ticker_weights(
        self,
        ticker_signals: pd.DataFrame,
        ticker_forecast_vectors: List[pd.DataFrame],
        ticker_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute model weights for a single ticker.

        Parameters
        ----------
        ticker_signals : pd.DataFrame
            Pivoted signals (index=date, columns=model_name, values=binary 0/1).
            Guaranteed to have >= 2 rows and >= 2 columns.
        ticker_forecast_vectors : list[pd.DataFrame]
            Forecast vectors filtered to this ticker.
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']
        ticker_returns : pd.Series, optional
            Aligned return series for this ticker (date index, float values).
            Required by downside-risk methods; ignored by correlation-based methods.

        Returns
        -------
        pd.Series
            Model name -> weight (must sum to 1.0, all >= 0).
        """
        ...

    # -- shared orchestration -------------------------------------------

    def fit(
        self,
        forecast_vectors: List[pd.DataFrame],
        signals: pd.DataFrame,
        returns: Optional[pd.Series] = None,
    ) -> BaseWeightLayer:
        """
        Fit weights and FDM from training data, separately for each ticker.

        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            List of forecast vectors from all ensembles (training data).
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']
        signals : pd.DataFrame
            Binary signals from all base models.
            Columns: model names, rows: samples (datetime index)
        returns : pd.Series, optional
            Instrument return series (date index, float values).  Passed through
            to ``_fit_ticker_weights`` for downside-risk weighting methods.
            Ignored by correlation-based methods.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If inputs are invalid or empty
        """
        if not forecast_vectors:
            raise ValueError("forecast_vectors cannot be empty")

        if signals.empty:
            raise ValueError("signals DataFrame cannot be empty")

        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)

        if all_forecasts.empty:
            raise ValueError("forecast_vectors contain no data")

        if 'ticker' not in all_forecasts.columns:
            raise ValueError("forecast_vectors must contain 'ticker' column")

        unique_tickers = sorted(all_forecasts['ticker'].unique())
        logger.info(
            f"Fitting {self.__class__.__name__} for "
            f"{len(unique_tickers)} ticker(s): {unique_tickers}"
        )

        for ticker in unique_tickers:
            logger.info(f"\nProcessing ticker: {ticker}")

            # Filter forecast_vectors to only this ticker
            ticker_forecast_vectors = [
                fv[fv['ticker'] == ticker].copy()
                for fv in forecast_vectors
                if isinstance(fv, pd.DataFrame)
                and 'ticker' in fv.columns
                and not fv[fv['ticker'] == ticker].empty
            ]

            if not ticker_forecast_vectors:
                logger.warning(f"No forecast vectors found for ticker {ticker}, skipping")
                continue

            ticker_all_forecasts = pd.concat(ticker_forecast_vectors, ignore_index=True)
            available_models = sorted(ticker_all_forecasts['model_name'].unique())

            logger.info(
                f"  Available models for {ticker}: {available_models} "
                f"({len(available_models)} model(s))"
            )

            if len(available_models) == 0:
                logger.warning(f"No models found for ticker {ticker}, skipping")
                continue

            # Extract per-model signals from forecast_vectors
            ticker_signals = self._extract_ticker_signals(
                ticker_all_forecasts, available_models
            )

            # Single model case – no diversification
            if len(available_models) == 1:
                model_name = available_models[0]
                self.weights_[ticker] = pd.Series({model_name: 1.0})
                self.model_names_[ticker] = [model_name]
                self.fdm_[ticker] = 1.0
                self.mean_forecast_correlation_[ticker] = 1.0
                logger.info(
                    f"  Ticker {ticker}: Single model '{model_name}', FDM=1.0, weight=1.0"
                )
                continue

            # Insufficient signal data – fall back to equal weights
            if ticker_signals.empty or len(ticker_signals) < 2:
                logger.warning(
                    f"Insufficient signals for ticker {ticker} "
                    f"({len(ticker_signals)} rows, need >= 2). "
                    f"Using equal weights and FDM=1.0"
                )
                equal_weight = 1.0 / len(available_models)
                self.weights_[ticker] = pd.Series(
                    {m: equal_weight for m in available_models}
                )
                self.model_names_[ticker] = available_models
                self.fdm_[ticker] = 1.0
                self.mean_forecast_correlation_[ticker] = 1.0
                continue

            # Extract ticker-aligned returns (same returns series for all tickers in single-ticker research)
            ticker_returns: Optional[pd.Series] = None
            if returns is not None:
                ticker_returns = returns  # pass the full series; subclasses align by index

            # --- Delegate weight calculation to subclass ---
            # Expose current ticker to subclasses that need per-ticker state
            self._current_ticker_for_group_returns = str(ticker)
            ticker_weights = self._fit_ticker_weights(
                ticker_signals, ticker_forecast_vectors, ticker_returns
            )

            self.weights_[ticker] = ticker_weights
            self.model_names_[ticker] = list(ticker_weights.index)

            # Calculate ticker-specific FDM
            ticker_fdm = self._calculate_fdm(
                ticker_forecast_vectors, ticker=ticker
            )
            self.fdm_[ticker] = ticker_fdm

            logger.info(
                f"  Ticker {ticker}: FDM={ticker_fdm:.4f}, "
                f"models={len(available_models)}, "
                f"mean_correlation="
                f"{self.mean_forecast_correlation_.get(ticker, 'N/A')}"
            )

        self.is_fitted_ = True
        return self

    # -- shared helpers -------------------------------------------------

    @staticmethod
    def _extract_ticker_signals(
        ticker_all_forecasts: pd.DataFrame,
        available_models: List[str],
    ) -> pd.DataFrame:
        """Pivot forecast rows into a (date x model) binary signal DataFrame."""
        if (
            'datetime' not in ticker_all_forecasts.columns
            or 'signal' not in ticker_all_forecasts.columns
        ):
            logger.warning(
                "Cannot extract signals (missing datetime or signal column)"
            )
            return pd.DataFrame()

        df = ticker_all_forecasts.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.normalize()

        signal_pivot = df.pivot_table(
            index='date',
            columns='model_name',
            values='signal',
            aggfunc='max',
        )

        existing_models = [m for m in available_models if m in signal_pivot.columns]
        return signal_pivot[existing_models] if existing_models else pd.DataFrame()

    def _calculate_fdm(
        self,
        forecast_vectors: List[pd.DataFrame],
        ticker: Optional[str] = None,
    ) -> float:
        """
        Calculate Forecast Diversification Multiplier from forecast value correlations.

        FDM accounts for diversification when combining multiple forecasts.
        Lower correlation between forecasts = higher FDM (more diversification benefit).

        Formula:
            FDM = sqrt(1 / (mean_corr + epsilon))
            Capped at fdm_max (default 2.0 per spec)

        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            Forecast vectors for a single ticker.
        ticker : str, optional
            Ticker name for storing mean_forecast_correlation_

        Returns
        -------
        float
            FDM value (capped at fdm_max)
        """
        default_corr = 1.0

        if not forecast_vectors:
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = default_corr
            return 1.0

        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)

        if all_forecasts.empty:
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = default_corr
            return 1.0

        if 'datetime' not in all_forecasts.columns or 'model_name' not in all_forecasts.columns:
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = default_corr
            return 1.0

        all_forecasts = all_forecasts.copy()
        all_forecasts['datetime'] = pd.to_datetime(all_forecasts['datetime'])
        all_forecasts['date'] = all_forecasts['datetime'].dt.normalize()

        if 'ticker' in all_forecasts.columns:
            forecast_pivot = all_forecasts.pivot_table(
                index=['date', 'ticker'],
                columns='model_name',
                values='forecast',
                aggfunc='first',
            )
        else:
            forecast_pivot = all_forecasts.pivot_table(
                index='date',
                columns='model_name',
                values='forecast',
                aggfunc='mean',
            )

        rows_before_drop = len(forecast_pivot)
        forecast_pivot = forecast_pivot.dropna()
        rows_after_drop = len(forecast_pivot)

        if rows_after_drop < 2 or len(forecast_pivot.columns) < 2:
            ticker_str = str(ticker) if ticker is not None else ''
            logger.warning(
                f"Insufficient data for FDM calculation"
                f"{' for ticker ' + ticker_str if ticker_str else ''}: "
                f"{rows_after_drop} rows (need >= 2), "
                f"{len(forecast_pivot.columns)} columns (need >= 2). "
                f"Before dropna: {rows_before_drop} rows."
            )
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = default_corr
            return 1.0

        corr_matrix = forecast_pivot.corr()

        logger.debug(f"\nForecast pivot for FDM calculation:")
        logger.debug(f"  Shape: {forecast_pivot.shape} (rows x columns)")
        logger.debug(f"  Columns: {list(forecast_pivot.columns)}")

        # Check for constant columns -> treat as perfectly correlated
        constant_cols = [
            col for col in forecast_pivot.columns
            if forecast_pivot[col].nunique() <= 1
        ]
        for col in constant_cols:
            corr_matrix.loc[col, :] = 1.0
            corr_matrix.loc[:, col] = 1.0

        if constant_cols:
            logger.warning(
                f"Found constant forecast columns: {constant_cols}. "
                f"Treated as perfectly correlated (corr=1.0) for FDM."
            )

        # Log NaN correlations before filling
        nan_mask = corr_matrix.isna()
        if nan_mask.any().any():
            nan_pairs = [
                (corr_matrix.index[i], corr_matrix.columns[j])
                for i in range(len(corr_matrix.index))
                for j in range(len(corr_matrix.columns))
                if nan_mask.iloc[i, j] and i != j
            ]
            if nan_pairs:
                logger.warning(
                    f"Found {len(nan_pairs)} NaN correlations: "
                    f"{nan_pairs[:5]}{'...' if len(nan_pairs) > 5 else ''}"
                )

        corr_matrix = corr_matrix.fillna(1.0)

        # Floor negative correlations at zero (Carver's recommendation)
        corr_matrix = corr_matrix.clip(lower=0.0)

        # Mean correlation from upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()

        if len(correlations) == 0:
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = default_corr
            return 1.0

        mean_correlation = correlations.mean()
        if ticker is not None:
            self.mean_forecast_correlation_[ticker] = mean_correlation

        logger.debug(f"\nIndividual forecast correlations:")
        for (model1, model2), corr_val in correlations.items():
            logger.debug(f"  {model1} vs {model2}: {corr_val:.4f}")

        ticker_str = str(ticker) if ticker is not None else ''
        logger.debug(
            f"FDM calculation{' for ticker ' + ticker_str if ticker_str else ''}: "
            f"{rows_after_drop} rows, {len(forecast_pivot.columns)} models, "
            f"mean correlation: {mean_correlation:.4f}"
        )

        epsilon = 0.01
        fdm = float(np.sqrt(1.0 / (mean_correlation + epsilon)))
        return min(fdm, self.fdm_max)

    # -- combine --------------------------------------------------------

    def combine(
        self,
        forecast_vectors: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Combine forecasts from all base models and apply ticker-specific FDM.

        For each ticker:
            forecast_score = (sum of weighted forecasts) * ticker_FDM

        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            Forecast vectors from all ensembles.
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']

        Returns
        -------
        pd.DataFrame
            Combined forecasts with columns:
            ['ticker', 'forecast_score'] or ['ticker', 'datetime', 'forecast_score']
            forecast_score is FDM-scaled and capped at +/-2.0

        Raises
        ------
        ValueError
            If the layer has not been fitted
        """
        if not self.is_fitted_:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before calling combine()"
            )

        if not forecast_vectors:
            return pd.DataFrame(columns=['ticker', 'forecast_score'])

        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)

        if all_forecasts.empty:
            return pd.DataFrame(columns=['ticker', 'forecast_score'])

        combined_results = [
            self._combine_ticker(ticker, all_forecasts[all_forecasts['ticker'] == ticker].copy())
            for ticker in all_forecasts['ticker'].unique()
        ]

        if combined_results:
            combined = pd.concat(combined_results, ignore_index=True)
            if 'datetime' in combined.columns:
                return combined[['ticker', 'datetime', 'forecast_score']]
            return combined[['ticker', 'forecast_score']]
        return pd.DataFrame(columns=['ticker', 'forecast_score'])

    def _combine_ticker(
        self,
        ticker: str,
        ticker_forecasts: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply weights + FDM for a single ticker."""
        # Resolve weights (fallback to equal if unseen)
        if ticker in self.weights_:
            ticker_weights = self.weights_[ticker]
        else:
            available_models = ticker_forecasts['model_name'].unique()
            n_models = len(available_models)
            equal_weight = 1.0 / n_models if n_models > 0 else 1.0
            ticker_weights = pd.Series({m: equal_weight for m in available_models})
            logger.warning(
                f"Ticker {ticker} not seen during fit(), using equal weights "
                f"({equal_weight:.4f} per model, {n_models} models)"
            )

        ticker_forecasts['weight'] = ticker_forecasts['model_name'].map(ticker_weights)

        # Handle missing weights (unseen models)
        missing_models = ticker_forecasts[ticker_forecasts['weight'].isna()]['model_name'].unique()
        if len(missing_models) > 0:
            n_known = len(ticker_weights)
            fallback_weight = (
                1.0 / (n_known + len(missing_models))
                if n_known > 0
                else 1.0 / len(missing_models)
            )
            ticker_forecasts['weight'] = ticker_forecasts['weight'].fillna(fallback_weight)

        ticker_forecasts['weighted_forecast'] = (
            ticker_forecasts['forecast'] * ticker_forecasts['weight']
        )

        if 'datetime' in ticker_forecasts.columns:
            ticker_combined = ticker_forecasts.groupby(
                'datetime', as_index=False
            ).agg({'weighted_forecast': 'sum'})
            ticker_combined['ticker'] = ticker
        else:
            ticker_combined = pd.DataFrame({
                'ticker': [ticker],
                'weighted_forecast': [ticker_forecasts['weighted_forecast'].sum()],
            })

        ticker_fdm = self.fdm_.get(ticker, 1.0)
        ticker_combined['forecast_score'] = (
            (ticker_combined['weighted_forecast'] * ticker_fdm).clip(upper=2.0, lower=-2.0)
        )
        return ticker_combined

    # -- diagnostics ----------------------------------------------------

    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about the fitted weight layer.

        Returns
        -------
        dict
            Dictionary containing:
            - is_fitted, tickers (per-ticker detail), summary, weight_method, fdm_max
        """
        if not self.is_fitted_:
            return {'is_fitted': False}

        tickers_dict = {}
        for ticker in self.fdm_:
            ticker_weights = self.weights_.get(ticker)
            weights_dict = None
            if ticker_weights is not None:
                weights_dict = {
                    k: (v if not pd.isna(v) else 0.0)
                    for k, v in ticker_weights.to_dict().items()
                }

            tickers_dict[ticker] = {
                'fdm': self.fdm_.get(ticker, 1.0),
                'weights': weights_dict,
                'models': self.model_names_.get(ticker, []),
                'n_models': len(self.model_names_.get(ticker, [])),
                'mean_forecast_correlation': self.mean_forecast_correlation_.get(ticker, 1.0),
            }

        fdm_values = list(self.fdm_.values())
        mean_fdm = float(np.mean(fdm_values)) if fdm_values else 1.0
        min_fdm = float(min(fdm_values)) if fdm_values else 1.0
        max_fdm = float(max(fdm_values)) if fdm_values else 1.0

        n_models_per_ticker = [
            len(self.model_names_.get(t, [])) for t in self.fdm_
        ]
        mean_models = float(np.mean(n_models_per_ticker)) if n_models_per_ticker else 0.0

        return {
            'is_fitted': True,
            'tickers': tickers_dict,
            'summary': {
                'n_tickers': len(self.fdm_),
                'mean_fdm': mean_fdm,
                'min_fdm': min_fdm,
                'max_fdm': max_fdm,
                'mean_models_per_ticker': mean_models,
                'total_models': sum(
                    len(self.model_names_.get(t, [])) for t in self.fdm_
                ),
            },
            'weight_method': self.weight_method,
            'fdm_max': self.fdm_max,
        }


# ---------------------------------------------------------------------------
# Private helper functions (used by new concrete subclasses)
# ---------------------------------------------------------------------------

def _extract_group_assignments(
    model_names: List[str],
    group_method: str,
    signals_df: pd.DataFrame,
    rho_cut: float = 0.70,
) -> Dict[str, str]:
    """Map model names to group IDs.

    Parameters
    ----------
    model_names : list[str]
        All model names for this ticker.
    group_method : str
        ``"feature_family"`` or ``"correlation_clustering"``.
    signals_df : pd.DataFrame
        (date × model) binary signal matrix, used only for correlation_clustering.
    rho_cut : float
        Correlation cutoff for clustering method.

    Returns
    -------
    dict mapping model_name → group_id (string)
    """
    if group_method == "feature_family":
        # Support "feature::member" names: use left part then first token before _
        def _family_id(name: str) -> str:
            left = name.split("::")[0].strip() if "::" in name else name
            return left.split("_")[0] if "_" in left else left
        return {name: _family_id(name) for name in model_names}

    # correlation_clustering: hierarchical Ward on full-period correlation
    from scipy.cluster.hierarchy import linkage as _scipy_linkage, fcluster
    from scipy.spatial.distance import squareform

    if signals_df.empty or len(signals_df.columns) < 2:
        return {name: name for name in model_names}

    available = [m for m in model_names if m in signals_df.columns]
    if len(available) < 2:
        return {name: name for name in model_names}

    corr = signals_df[available].corr().fillna(0.0).clip(lower=0.0)
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist)
    Z = _scipy_linkage(condensed, method="ward")
    # Cut at rho_cut: distance threshold = sqrt(0.5*(1-rho_cut))
    dist_threshold = float(np.sqrt(0.5 * (1.0 - rho_cut)))
    labels = fcluster(Z, dist_threshold, criterion="distance")
    group_map = {name: f"cluster_{lbl}" for name, lbl in zip(available, labels)}
    # Any models not in signals_df get their own group
    for name in model_names:
        if name not in group_map:
            group_map[name] = name
    return group_map


def _compute_group_signals_and_returns(
    ticker_signals: pd.DataFrame,
    ticker_returns: Optional[pd.Series],
    group_assignments: Dict[str, str],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Build (T×K) group signal and group return DataFrames.

    group_signal_k(t) = mean(signal_i(t) for i in group_k)
    group_return_k(t) = group_signal_k(t) × return(t)  [if returns provided]

    Returns
    -------
    group_signals_df : pd.DataFrame  columns = group IDs
    group_returns_df : pd.DataFrame or None  columns = group IDs
    """
    groups = sorted(set(group_assignments.values()))
    group_signals: Dict[str, pd.Series] = {}
    for grp in groups:
        members = [m for m, g in group_assignments.items() if g == grp and m in ticker_signals.columns]
        if not members:
            continue
        group_signals[grp] = ticker_signals[members].mean(axis=1)

    group_signals_df = pd.DataFrame(group_signals)

    if ticker_returns is None:
        return group_signals_df, None

    # Deduplicate indices so reindex does not raise (duplicate dates in signals or returns)
    if group_signals_df.index.duplicated().any():
        group_signals_df = group_signals_df.loc[~group_signals_df.index.duplicated(keep="first")]
    if ticker_returns.index.duplicated().any():
        ticker_returns = ticker_returns.loc[~ticker_returns.index.duplicated(keep="first")]

    aligned_returns = ticker_returns.reindex(group_signals_df.index)
    group_returns_df = group_signals_df.multiply(aligned_returns, axis=0)
    return group_signals_df, group_returns_df


def _compute_downside_semi_covariance(
    group_returns_df: pd.DataFrame,
    shrinkage: str = "ledoit_wolf",
) -> np.ndarray:
    """Compute K×K downside semi-covariance matrix.

    r_k^-(t) = min(group_return_k(t), 0)
    Σ^down_kl = (1/T) × Σ_t [r_k^-(t) × r_l^-(t)]

    Applies Ledoit-Wolf shrinkage if requested.
    """
    data = group_returns_df.values.copy()
    # Replace NaN/inf with 0 — missing dates have no downside contribution
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Lower semi-returns: clip positive values to zero
    semi = np.minimum(data, 0.0)
    T = semi.shape[0]
    semi_cov = (semi.T @ semi) / max(T, 1)

    if shrinkage == "ledoit_wolf" and semi_cov.shape[0] >= 2:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(assume_centered=True)
            lw.fit(semi)
            semi_cov = lw.covariance_
        except Exception:
            pass  # fall back to raw estimate

    return semi_cov


def _hrp_weights_from_semi_cov(
    semi_cov: np.ndarray,
    linkage_method: str = "ward",
) -> np.ndarray:
    """HRP weight vector from a K×K downside semi-covariance matrix.

    Steps: downside correlation → distance → Ward linkage → recursive bisection.

    Returns
    -------
    np.ndarray of length K, summing to 1.0
    """
    from scipy.cluster.hierarchy import linkage as _scipy_linkage, to_tree

    K = semi_cov.shape[0]
    if K == 1:
        return np.array([1.0])

    # Downside correlation
    diag = np.diag(semi_cov)
    denom = np.sqrt(np.outer(diag, diag))
    denom[denom == 0] = 1.0
    corr = semi_cov / denom
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    # Symmetrize corr before computing distance (guards against float-precision
    # asymmetry from Ledoit-Wolf shrinkage which can cause squareform to reject)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    # Distance — extract condensed form directly to skip squareform symmetry check
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    n = dist.shape[0]
    condensed = dist[np.triu_indices(n, k=1)]
    # Guard: if any non-finite values slipped through, fall back to equal weights
    if not np.all(np.isfinite(condensed)):
        return np.ones(K) / K
    Z = _scipy_linkage(condensed, method=linkage_method)

    # Leaf order from dendrogram
    root, _ = to_tree(Z, rd=True)

    def _leaf_order(node):
        if node.is_leaf():
            return [node.id]
        return _leaf_order(node.get_left()) + _leaf_order(node.get_right())

    leaf_order = _leaf_order(root)

    # Recursive bisection
    weights = np.ones(K, dtype=float)

    def _bisect(items: list) -> None:
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]

        def _cluster_var(idx_list: list) -> float:
            n = len(idx_list)
            w = np.ones(n) / n
            sub = semi_cov[np.ix_(idx_list, idx_list)]
            return float(w @ sub @ w)

        var_l = _cluster_var(left)
        var_r = _cluster_var(right)
        total = var_l + var_r
        alpha = (var_r / total) if total > 0 else 0.5  # fraction to left
        weights[left] *= alpha
        weights[right] *= (1.0 - alpha)
        _bisect(left)
        _bisect(right)

    _bisect(leaf_order)
    total = weights.sum()
    return weights / total if total > 0 else np.ones(K) / K


def _compute_fdm_from_corr_matrix(
    corr_matrix: np.ndarray,
    fdm_max: float = 2.0,
) -> float:
    """FDM from a K×K correlation matrix.

    mean_corr = mean of off-diagonal upper-triangle entries (clipped ≥ 0)
    FDM = min(sqrt(1 / (mean_corr + 0.01)), fdm_max)
    """
    K = corr_matrix.shape[0]
    if K <= 1:
        return 1.0
    mask = np.triu(np.ones((K, K), dtype=bool), k=1)
    off_diag = corr_matrix[mask]
    off_diag = np.clip(off_diag, 0.0, None)
    mean_corr = float(off_diag.mean()) if len(off_diag) > 0 else 1.0
    fdm = float(np.sqrt(1.0 / (mean_corr + 0.01)))
    return min(fdm, fdm_max)


# ---------------------------------------------------------------------------
# Concrete Strategy: Inverse Correlation
# ---------------------------------------------------------------------------

class InverseCorrelationWeightLayer(BaseWeightLayer):
    """
    Weight layer using inverse correlation weights.

    Models with lower average absolute correlation to other models receive
    higher weights, maximising diversification benefit.

    This is Robert Carver's recommended approach for small model counts.

    Parameters
    ----------
    fdm_max : float, default=2.0
        Maximum FDM value (per spec)
    """

    @property
    def weight_method(self) -> str:
        return 'inverse_correlation'

    def _fit_ticker_weights(
        self,
        ticker_signals: pd.DataFrame,
        ticker_forecast_vectors: List[pd.DataFrame],
        ticker_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Compute inverse-correlation weights from binary signals."""
        weighter = InverseCorrelationWeighter()
        weighter.fit(ticker_signals)
        return weighter.get_weights()

# ---------------------------------------------------------------------------
# Concrete Strategy: Equal Flat (Level 0)
# ---------------------------------------------------------------------------

class EqualFlatWeightLayer(BaseWeightLayer):
    """Level 0 — Equal weights across all N signals, no grouping.

    The hard baseline. FDM computed from full-period forecast correlations.
    """

    def __init__(self, config: Optional['WeightLayerConfig'] = None) -> None:
        cfg = config or WeightLayerConfig()
        super().__init__(fdm_max=cfg.fdm_max)
        self._wl_config = cfg

    @property
    def weight_method(self) -> str:
        return 'equal_flat'

    def _fit_ticker_weights(
        self,
        ticker_signals: pd.DataFrame,
        ticker_forecast_vectors: List[pd.DataFrame],
        ticker_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        n = len(ticker_signals.columns)
        equal_w = 1.0 / n if n > 0 else 1.0
        return pd.Series({m: equal_w for m in ticker_signals.columns})


# ---------------------------------------------------------------------------
# Concrete Strategy: Equal Grouped (Level 1)
# ---------------------------------------------------------------------------

class EqualGroupedWeightLayer(BaseWeightLayer):
    """Level 1 — Equal within group, equal across groups.

    Tests whether grouping structure alone adds value over flat equal weights.
    FDM from full-period correlation.
    """

    def __init__(self, config: Optional['WeightLayerConfig'] = None) -> None:
        cfg = config or WeightLayerConfig()
        super().__init__(fdm_max=cfg.fdm_max)
        self._wl_config = cfg

    @property
    def weight_method(self) -> str:
        return 'equal_grouped'

    def _fit_ticker_weights(
        self,
        ticker_signals: pd.DataFrame,
        ticker_forecast_vectors: List[pd.DataFrame],
        ticker_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        model_names = list(ticker_signals.columns)
        group_map = _extract_group_assignments(
            model_names, self._wl_config.group_method,
            ticker_signals, self._wl_config.rho_cut,
        )
        groups = sorted(set(group_map.values()))
        K = len(groups)
        group_weight = 1.0 / K if K > 0 else 1.0
        weights: Dict[str, float] = {}
        for grp in groups:
            members = [m for m in model_names if group_map[m] == grp]
            per_model = group_weight / len(members) if members else 0.0
            for m in members:
                weights[m] = per_model
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return pd.Series(weights)


# ---------------------------------------------------------------------------
# Concrete Strategy: Inverse Downside Vol, Grouped (Level 2)
# ---------------------------------------------------------------------------

class InvDownsideVolGroupedWeightLayer(BaseWeightLayer):
    """Level 2 — Equal within group, inverse downside vol across groups.

    Group weights ∝ 1/σ^down_k. No correlation matrix — targets drawdown
    without matrix estimation risk.
    """

    def __init__(self, config: Optional['WeightLayerConfig'] = None) -> None:
        cfg = config or WeightLayerConfig()
        super().__init__(fdm_max=cfg.fdm_max)
        self._wl_config = cfg

    @property
    def weight_method(self) -> str:
        return 'inv_downside_vol_grouped'

    def _fit_ticker_weights(
        self,
        ticker_signals: pd.DataFrame,
        ticker_forecast_vectors: List[pd.DataFrame],
        ticker_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        model_names = list(ticker_signals.columns)
        group_map = _extract_group_assignments(
            model_names, self._wl_config.group_method,
            ticker_signals, self._wl_config.rho_cut,
        )
        groups = sorted(set(group_map.values()))
        K = len(groups)

        if ticker_returns is None or K == 1:
            # Fallback: equal across groups
            group_weight = 1.0 / K if K > 0 else 1.0
            weights: Dict[str, float] = {}
            for grp in groups:
                members = [m for m in model_names if group_map[m] == grp]
                per_model = group_weight / len(members) if members else 0.0
                for m in members:
                    weights[m] = per_model
            total = sum(weights.values())
            return pd.Series({k: v / total for k, v in weights.items()})

        _, group_returns_df = _compute_group_signals_and_returns(
            ticker_signals, ticker_returns, group_map
        )
        if group_returns_df is None or group_returns_df.empty:
            equal_w = 1.0 / len(model_names)
            return pd.Series({m: equal_w for m in model_names})

        # Downside vol per group
        semi = np.minimum(group_returns_df.values, 0.0)
        downside_std = semi.std(axis=0, ddof=0)
        downside_std = np.where(downside_std == 0, 1e-8, downside_std)
        inv_vol = 1.0 / downside_std
        group_weights_arr = inv_vol / inv_vol.sum()
        group_cols = list(group_returns_df.columns)

        weights_out: Dict[str, float] = {}
        for i, grp in enumerate(group_cols):
            members = [m for m in model_names if group_map.get(m) == grp]
            per_model = group_weights_arr[i] / len(members) if members else 0.0
            for m in members:
                weights_out[m] = per_model

        total = sum(weights_out.values())
        if total > 0:
            weights_out = {k: v / total for k, v in weights_out.items()}
        return pd.Series(weights_out)


# ---------------------------------------------------------------------------
# Concrete Strategy: Downside HRP, Grouped (Level 3) — Default
# ---------------------------------------------------------------------------

class DownsideHRPGroupedWeightLayer(BaseWeightLayer):
    """Level 3 — Equal within group, Downside-HRP across groups.

    Full algorithm from weight_layer.md Section 4.  This is the default
    production method candidate.

    Steps
    -----
    1. Within-group aggregation (equal weights).
    2. Downside semi-covariance on group return streams.
    3. Ledoit-Wolf shrinkage.
    4. HRP recursive bisection on Ward dendrogram of downside distance.
    5. FDM from downside correlation of groups.
    """

    def __init__(self, config: Optional['WeightLayerConfig'] = None) -> None:
        cfg = config or WeightLayerConfig()
        super().__init__(fdm_max=cfg.fdm_max)
        self._wl_config = cfg
        # Per-ticker storage for group return streams used in FDM override
        self._group_returns_map: Dict[str, pd.DataFrame] = {}

    @property
    def weight_method(self) -> str:
        return 'downside_hrp_grouped'

    def _fit_ticker_weights(
        self,
        ticker_signals: pd.DataFrame,
        ticker_forecast_vectors: List[pd.DataFrame],
        ticker_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        model_names = list(ticker_signals.columns)
        group_map = _extract_group_assignments(
            model_names, self._wl_config.group_method,
            ticker_signals, self._wl_config.rho_cut,
        )
        groups = sorted(set(group_map.values()))
        K = len(groups)

        if K <= 1 or ticker_returns is None:
            # Single group or no returns: equal weights
            equal_w = 1.0 / len(model_names)
            return pd.Series({m: equal_w for m in model_names})

        _, group_returns_df = _compute_group_signals_and_returns(
            ticker_signals, ticker_returns, group_map
        )
        if group_returns_df is None or group_returns_df.empty or len(group_returns_df) < 2:
            equal_w = 1.0 / len(model_names)
            return pd.Series({m: equal_w for m in model_names})

        # Store group returns for this ticker so _calculate_fdm can use group-level
        # downside correlation instead of individual forecasts.
        ticker_key = getattr(self, "_current_ticker_for_group_returns", None)
        if isinstance(ticker_key, str):
            self._group_returns_map[ticker_key] = group_returns_df

        semi_cov = _compute_downside_semi_covariance(
            group_returns_df, shrinkage=self._wl_config.shrinkage
        )
        group_weights_arr = _hrp_weights_from_semi_cov(
            semi_cov, linkage_method=self._wl_config.linkage
        )
        group_cols = list(group_returns_df.columns)

        weights_out: Dict[str, float] = {}
        for i, grp in enumerate(group_cols):
            members = [m for m in model_names if group_map.get(m) == grp]
            per_model = group_weights_arr[i] / len(members) if members else 0.0
            for m in members:
                weights_out[m] = per_model

        total = sum(weights_out.values())
        if total > 0:
            weights_out = {k: v / total for k, v in weights_out.items()}
        return pd.Series(weights_out)

    def _calculate_fdm(
        self,
        forecast_vectors: List[pd.DataFrame],
        ticker: Optional[str] = None,
    ) -> float:
        """Override: FDM from group-level downside correlation, not individual forecasts."""
        # Retrieve stored group returns for this ticker (set during _fit_ticker_weights via fit).
        # Fallback to parent FDM if group data not available.
        group_returns: Optional[pd.DataFrame] = None
        if ticker is not None and hasattr(self, "_group_returns_map"):
            group_returns = self._group_returns_map.get(ticker)
        if group_returns is None or not isinstance(group_returns, pd.DataFrame):
            return super()._calculate_fdm(forecast_vectors, ticker)

        semi_cov = _compute_downside_semi_covariance(
            group_returns, shrinkage=self._wl_config.shrinkage
        )
        K = semi_cov.shape[0]
        diag = np.diag(semi_cov)
        denom = np.sqrt(np.outer(diag, diag))
        denom[denom == 0] = 1.0
        corr = semi_cov / denom
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, 0.0, None)
        fdm = _compute_fdm_from_corr_matrix(corr, fdm_max=self._wl_config.fdm_max)
        mean_corr = float(np.triu(corr, k=1).sum() / max(K * (K - 1) / 2, 1))
        if ticker is not None:
            self.mean_forecast_correlation_[ticker] = mean_corr
        return fdm


# ---------------------------------------------------------------------------
# Concrete Strategy: Downside HRP, Flat (Level 4)
# ---------------------------------------------------------------------------

class DownsideHRPFlatWeightLayer(BaseWeightLayer):
    """Level 4 — Downside-HRP directly on all N signals (no grouping).

    Control experiment: does grouping improve stability over raw HRP on
    individual signals?  Expect more weight instability than Level 3 as N grows.
    """

    def __init__(self, config: Optional['WeightLayerConfig'] = None) -> None:
        cfg = config or WeightLayerConfig()
        super().__init__(fdm_max=cfg.fdm_max)
        self._wl_config = cfg

    @property
    def weight_method(self) -> str:
        return 'downside_hrp_flat'

    def _fit_ticker_weights(
        self,
        ticker_signals: pd.DataFrame,
        ticker_forecast_vectors: List[pd.DataFrame],
        ticker_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        model_names = list(ticker_signals.columns)
        N = len(model_names)

        if N <= 1 or ticker_returns is None:
            equal_w = 1.0 / N if N > 0 else 1.0
            return pd.Series({m: equal_w for m in model_names})

        aligned_returns = ticker_returns.reindex(ticker_signals.index)
        # Per-model return streams: signal_i(t) × return(t)
        model_returns_df = ticker_signals.multiply(aligned_returns, axis=0)

        if len(model_returns_df.dropna()) < 2:
            equal_w = 1.0 / N
            return pd.Series({m: equal_w for m in model_names})

        semi_cov = _compute_downside_semi_covariance(
            model_returns_df.fillna(0.0), shrinkage=self._wl_config.shrinkage
        )
        weights_arr = _hrp_weights_from_semi_cov(
            semi_cov, linkage_method=self._wl_config.linkage
        )
        return pd.Series(dict(zip(model_names, weights_arr)))


# ---------------------------------------------------------------------------
# Backward-compatible factory function
# ---------------------------------------------------------------------------

def WeightLayer(
    weight_method: str = 'inverse_correlation',
    fdm_max: float = 2.0,
    config: Optional['WeightLayerConfig'] = None,
    **kwargs,
) -> BaseWeightLayer:
    """
    Factory that creates the appropriate weight layer by method name.

    Parameters
    ----------
    weight_method : str, default='inverse_correlation'
        Ignored when ``config`` is provided.
    fdm_max : float, default=2.0
        FDM cap (per spec). Used when ``config`` is None. When ``config`` is
        provided, use config.fdm_max instead.
    config : WeightLayerConfig, optional
        When provided, ``config.weighting_method`` takes precedence over
        ``weight_method``.
    **kwargs
        Ignored. Included for backward compatibility.

    Returns
    -------
    BaseWeightLayer

    Examples
    --------
    >>> layer = WeightLayer()                                     # inverse correlation (default)
    >>> layer = WeightLayer(config=WeightLayerConfig(weighting_method='downside_hrp_grouped'))
    """
    if config is not None:
        method = config.weighting_method
    else:
        method = weight_method

    _REGISTRY: Dict[str, type] = {
        'inverse_correlation': InverseCorrelationWeightLayer,
        'equal_flat': EqualFlatWeightLayer,
        'equal_grouped': EqualGroupedWeightLayer,
        'inv_downside_vol_grouped': InvDownsideVolGroupedWeightLayer,
        'downside_hrp_grouped': DownsideHRPGroupedWeightLayer,
        'downside_hrp_flat': DownsideHRPFlatWeightLayer,
    }
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown weight method: '{method}'. "
            f"Supported methods: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[method]
    if method == 'inverse_correlation':
        effective_fdm_max = config.fdm_max if config is not None else fdm_max
        return cls(fdm_max=effective_fdm_max)
    # For other methods, pass config so fdm_max is respected when caller passes it
    effective_config = config if config is not None else WeightLayerConfig(
        weighting_method=method, fdm_max=fdm_max
    )
    return cls(config=effective_config)


__all__ = [
    'WeightLayerConfig',
    'WeightLayer',
    'BaseWeightLayer',
    'InverseCorrelationWeightLayer',
    'EqualFlatWeightLayer',
    'EqualGroupedWeightLayer',
    'InvDownsideVolGroupedWeightLayer',
    'DownsideHRPGroupedWeightLayer',
    'DownsideHRPFlatWeightLayer',
    'InverseCorrelationWeighter',
    'Weighter',
]
