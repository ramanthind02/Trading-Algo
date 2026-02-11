"""
Weight Layer for Forecast Combination

This module implements the Weight Layer that combines forecasts from all base models
across all ensembles using configurable weight strategies and applies the Forecast
Diversification Multiplier (FDM).

The Weight Layer sits between Ensemble and Portfolio layers:
    Ensemble.predict() -> WeightLayer.combine() -> Portfolio.predict()

Supported weight strategies:
- InverseCorrelationWeightLayer: Inverse correlation weights (Carver's method)
- SortinoOptimizedWeightLayer: Constrained Sortino-ratio optimization with
  Ledoit-Wolf covariance shrinkage and equal-weight penalty

Use the ``WeightLayer`` factory function to create instances by name:
    >>> layer = WeightLayer(weight_method='inverse_correlation')
    >>> layer = WeightLayer(weight_method='sortino_optimized', gamma=1.0)

Reference: Robert Carver's "Systematic Trading" and "Leveraged Trading"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    fdm_max : float, default=2.5
        Maximum FDM value (Carver's recommendation)

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

    def __init__(self, fdm_max: float = 2.5) -> None:
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

            # --- Delegate weight calculation to subclass ---
            ticker_weights = self._fit_ticker_weights(
                ticker_signals, ticker_forecast_vectors
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
            Capped at fdm_max (typically 2.5)

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
# Concrete Strategy: Inverse Correlation (existing behaviour)
# ---------------------------------------------------------------------------

class InverseCorrelationWeightLayer(BaseWeightLayer):
    """
    Weight layer using inverse correlation weights.

    Models with lower average absolute correlation to other models receive
    higher weights, maximising diversification benefit.

    This is Robert Carver's recommended approach for small model counts.

    Parameters
    ----------
    fdm_max : float, default=2.5
        Maximum FDM value
    """

    @property
    def weight_method(self) -> str:
        return 'inverse_correlation'

    def _fit_ticker_weights(
        self,
        ticker_signals: pd.DataFrame,
        ticker_forecast_vectors: List[pd.DataFrame],
    ) -> pd.Series:
        """Compute inverse-correlation weights from binary signals."""
        weighter = InverseCorrelationWeighter()
        weighter.fit(ticker_signals)
        return weighter.get_weights()


# ---------------------------------------------------------------------------
# Concrete Strategy: Sortino-Optimized Constrained Portfolio
# ---------------------------------------------------------------------------

class SortinoOptimizedWeightLayer(BaseWeightLayer):
    """
    Constrained Sortino-ratio optimization with covariance shrinkage.

    Solves the quadratic program:

        max  w^T mu_shrunk
           - (gamma / 2) * w^T Sigma_down_shrunk w
           - shrinkage_lambda * ||w - w_eq||^2

        s.t.  w_i >= 0,  sum(w_i) == 1

    Where:
    - mu_shrunk  : mean forecast per model, shrunk toward zero
    - Sigma_down : downside semi-covariance (only below-zero returns)
    - Ledoit-Wolf shrinkage applied to Sigma_down
    - w_eq       : equal weights (1/N) -- strong diversification prior

    Parameters
    ----------
    fdm_max : float, default=2.5
        Maximum FDM value
    gamma : float, default=1.0
        Risk aversion parameter (higher = more conservative)
    shrinkage_lambda : float, default=0.5
        Strength of penalty toward equal weights
    mean_shrinkage : float, default=0.5
        Shrinkage factor for expected returns (0 = shrink to zero, 1 = no shrinkage)
    min_history : int, default=30
        Minimum observations required before optimizing (else equal weights)
    activity_adjust_means : bool, default=True
        If True, expected return (mu) per model is "mean forecast when active":
        mean(forecast) / (fraction_active + eps). This prevents always-on models
        (e.g. buy_hold) from dominating because they have non-zero forecast every day.
        When False, uses raw mean(forecast) over time (always-on gets higher mu).
    active_threshold : float, default=1e-8
        Minimum |forecast| to count as "active" for activity_adjust_means.
    """

    def __init__(
        self,
        fdm_max: float = 2.5,
        gamma: float = 1.0,
        shrinkage_lambda: float = 0.5,
        mean_shrinkage: float = 0.5,
        min_history: int = 30,
        activity_adjust_means: bool = True,
        active_threshold: float = 1e-8,
    ) -> None:
        super().__init__(fdm_max=fdm_max)
        self.gamma = gamma
        self.shrinkage_lambda = shrinkage_lambda
        self.mean_shrinkage = mean_shrinkage
        self.min_history = min_history
        self.activity_adjust_means = activity_adjust_means
        self.active_threshold = active_threshold

    @property
    def weight_method(self) -> str:
        return 'sortino_optimized'

    # -- hook -----------------------------------------------------------

    def _fit_ticker_weights(
        self,
        ticker_signals: pd.DataFrame,
        ticker_forecast_vectors: List[pd.DataFrame],
    ) -> pd.Series:
        """
        Solve constrained Sortino optimization for one ticker.

        Steps:
        1. Pivot forecasts to (date x model) matrix
        2. Compute downside semi-covariance
        3. Apply Ledoit-Wolf shrinkage
        4. Shrink expected returns toward zero
        5. Solve constrained QP (scipy SLSQP)
        6. Fallback to equal weights on failure
        """
        from scipy.optimize import minimize
        from sklearn.covariance import LedoitWolf

        # 1. Build (date x model) forecast matrix
        forecast_matrix = self._build_forecast_matrix(ticker_forecast_vectors)
        model_names = list(forecast_matrix.columns)
        n_models = len(model_names)

        # If insufficient data, fall back to equal weights
        if forecast_matrix.empty or len(forecast_matrix) < self.min_history:
            logger.info(
                f"  Sortino optimizer: insufficient history "
                f"({len(forecast_matrix)} < {self.min_history}), "
                f"using equal weights for {n_models} models"
            )
            equal_w = 1.0 / n_models
            return pd.Series({m: equal_w for m in model_names})

        values = forecast_matrix.values  # (T, N)

        # Fraction of days each model is active (for fallback when mu has no variation)
        active = np.abs(values) > self.active_threshold
        fraction_active = np.mean(active.astype(np.float64), axis=0)

        # 2. Downside semi-covariance (only periods with below-zero returns)
        sigma_down = self._build_downside_covariance(values)

        # 3. Ledoit-Wolf shrinkage on the downside covariance
        sigma_shrunk = self._shrink_covariance(sigma_down, n_models)

        # 4. Expected return proxy per model (activity-adjusted to avoid always-on bias)
        mu_raw = self._expected_forecast_per_model(values)
        mu_shrunk = self._shrink_means(mu_raw)

        # 5. Balance objective scale so return term can compete with risk (avoid equal-weight trap)
        w_eq = np.full(n_models, 1.0 / n_models)
        mu_balanced, sigma_balanced = self._balance_objective_scale(
            mu_shrunk, sigma_shrunk, w_eq
        )

        # 6. Return-seeking starting point (avoid getting stuck at equal weight)
        w0 = self._return_seeking_start(mu_shrunk)

        # 7. Solve constrained QP
        weights, solved = self._solve_qp(
            mu_balanced, sigma_balanced, w_eq, n_models, x0=w0
        )

        # 8. If QP returned nearly equal weights, replace with a rule-based allocation
        weight_range = float(weights.max() - weights.min())
        mu_std = float(np.std(mu_shrunk))
        if weight_range < 0.05:
            if mu_std > 1e-6:
                weights = self._weights_from_means(mu_shrunk)
                logger.info(
                    f"  Sortino optimizer: QP near-equal (range={weight_range:.4f}); "
                    f"using return-proportional weights (mu_std={mu_std:.4f})."
                )
            else:
                # No variation in activity-adjusted means: weight by inverse activity
                # so always-on (e.g. buy_hold) gets less weight, selective models more
                weights = self._weights_from_inverse_activity(fraction_active)
                logger.info(
                    f"  Sortino optimizer: QP near-equal, mu_std≈0; "
                    f"using inverse-activity weights (diversification)."
                )
        else:
            logger.info(
                f"  Sortino optimizer: {len(forecast_matrix)} obs, {n_models} models, "
                f"mu_std={mu_std:.4f}, converged={solved}, "
                f"weights={dict(zip(model_names, [round(w, 4) for w in weights]))}"
            )
            if not solved:
                logger.warning(
                    "  Sortino solver did not converge; returned weights may be equal or from fallback."
                )

        return pd.Series(dict(zip(model_names, weights)))

    # -- internal helpers -----------------------------------------------

    @staticmethod
    def _build_forecast_matrix(
        ticker_forecast_vectors: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """Pivot forecast vectors into a (date x model_name) matrix."""
        all_fv = pd.concat(ticker_forecast_vectors, ignore_index=True)

        if 'datetime' not in all_fv.columns or 'model_name' not in all_fv.columns:
            return pd.DataFrame()

        all_fv = all_fv.copy()
        all_fv['datetime'] = pd.to_datetime(all_fv['datetime'])
        all_fv['date'] = all_fv['datetime'].dt.normalize()

        pivot = all_fv.pivot_table(
            index='date',
            columns='model_name',
            values='forecast',
            aggfunc='first',
        )
        # Fill missing (date, model) with 0 = inactive; keep all dates so we have full history
        pivot = pivot.fillna(0.0)
        return pivot

    @staticmethod
    def _build_downside_covariance(values: np.ndarray) -> np.ndarray:
        """
        Compute the downside semi-covariance matrix.

        Only observations where at least one model has a below-zero forecast
        contribute.  Above-zero values are clipped to zero so they do not
        inflate the covariance estimate.

        Parameters
        ----------
        values : np.ndarray, shape (T, N)
            Forecast value matrix

        Returns
        -------
        np.ndarray, shape (N, N)
            Downside semi-covariance matrix
        """
        # Clip above-zero values to zero -- only downside matters
        downside = np.minimum(values, 0.0)

        n_obs = downside.shape[0]
        if n_obs < 2:
            return np.eye(downside.shape[1])

        # Centre around downside mean then compute covariance
        means = downside.mean(axis=0, keepdims=True)
        centred = downside - means
        cov = (centred.T @ centred) / (n_obs - 1)

        # Ensure positive semi-definite (numerical safety)
        cov = (cov + cov.T) / 2.0

        # Add tiny ridge if any eigenvalue is non-positive
        min_eig = np.linalg.eigvalsh(cov).min()
        if min_eig <= 0:
            cov += (abs(min_eig) + 1e-8) * np.eye(cov.shape[0])

        return cov

    @staticmethod
    def _shrink_covariance(
        cov: np.ndarray,
        n_models: int,
    ) -> np.ndarray:
        """
        Apply Ledoit-Wolf style shrinkage to a covariance matrix.

        Uses sklearn's LedoitWolf to estimate the optimal shrinkage intensity,
        blending the sample covariance toward a scaled identity (diagonal) target.

        Parameters
        ----------
        cov : np.ndarray, shape (N, N)
            Raw (downside) covariance matrix
        n_models : int
            Number of models (used for identity fallback)

        Returns
        -------
        np.ndarray, shape (N, N)
            Shrunk covariance matrix
        """
        from sklearn.covariance import LedoitWolf

        try:
            lw = LedoitWolf(assume_centered=False)
            # LedoitWolf.fit expects (n_samples, n_features).
            # We already have the covariance; use store_precision=False workaround:
            # Generate pseudo-samples from the covariance via Cholesky.
            # However it's simpler to just apply the analytic shrinkage formula
            # ourselves: Sigma_shrunk = (1 - alpha) * S + alpha * mu * I
            # where alpha is Ledoit-Wolf optimal and mu = trace(S)/p.
            n = n_models
            trace_s = np.trace(cov)
            mu_target = trace_s / n  # scalar target

            # Estimate shrinkage intensity analytically (simplified Oracle Approximating
            # Shrinkage when we only have the covariance, not raw samples).
            # Use a conservative default alpha when we can't run full LW.
            frobenius_sq = np.sum(cov ** 2)
            trace_sq = trace_s ** 2

            # Simplified OAS-style alpha (see Chen, Wiesel, Eldar & Hero 2010)
            rho_num = ((1.0 - 2.0 / n) * frobenius_sq + trace_sq)
            rho_den = ((n + 1.0 - 2.0 / n) * (frobenius_sq - trace_sq / n))

            if abs(rho_den) < 1e-12:
                alpha = 1.0  # highly singular -> full shrinkage
            else:
                alpha = float(np.clip(rho_num / rho_den, 0.0, 1.0))

            shrunk = (1.0 - alpha) * cov + alpha * mu_target * np.eye(n)

            logger.debug(f"  Covariance shrinkage alpha={alpha:.4f}")
            return shrunk

        except Exception as exc:
            logger.warning(
                f"Covariance shrinkage failed ({exc}), returning raw covariance"
            )
            return cov

    def _expected_forecast_per_model(self, values: np.ndarray) -> np.ndarray:
        """
        Expected return proxy per model (mu for the optimizer).

        When activity_adjust_means is True, use mean(forecast) / (fraction_active + eps)
        so that "always on" models (e.g. buy_hold) do not get an unfair advantage:
        we compare average contribution when the model is active, not raw mean over time.

        Parameters
        ----------
        values : np.ndarray, shape (T, N)
            Forecast matrix (time x models)

        Returns
        -------
        np.ndarray, shape (N,)
            Expected forecast per model (before mean_shrinkage).
        """
        if not self.activity_adjust_means:
            return np.mean(values, axis=0)

        eps = max(self.active_threshold, 1e-12)
        active = np.abs(values) > self.active_threshold  # (T, N)
        fraction_active = np.mean(active.astype(np.float64), axis=0)  # (N,)
        raw_mean = np.mean(values, axis=0)
        # Mean when active = sum(forecast) / count(active) = mean(forecast) / fraction_active
        # Avoid div by zero; if never active, treat as zero expected contribution
        mu = np.where(
            fraction_active > eps,
            raw_mean / (fraction_active + eps),
            np.zeros_like(raw_mean),
        )
        return mu

    def _shrink_means(self, mu: np.ndarray) -> np.ndarray:
        """
        Shrink expected returns toward zero.

        mu_shrunk = mean_shrinkage * mu

        A shrinkage factor of 0.5 is a strong prior that most signals
        are weaker than they appear in-sample.

        Parameters
        ----------
        mu : np.ndarray, shape (N,)
            Raw mean forecasts per model

        Returns
        -------
        np.ndarray, shape (N,)
            Shrunk means
        """
        return self.mean_shrinkage * mu

    def _balance_objective_scale(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        w_eq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Scale sigma so the return term w'mu can dominate; risk term is scaled down further.

        When risk dominates, the optimizer stays at equal weight. We scale sigma so that
        at w_eq the risk term is a fraction of the return term (return_dominance), so
        moving toward high-mu models is not overly penalized by risk.

        Returns
        -------
        mu_balanced, sigma_balanced
            Same mu; sigma scaled so at w_eq, (gamma/2)*w_eq'Sigma*w_eq = return_scale * return_dominance.
        """
        return_scale = np.abs(mu).max() + 1e-12
        risk_at_eq = float(w_eq @ sigma @ w_eq) + 1e-12
        # Scale so risk term at w_eq is return_dominance * return_scale (e.g. 0.5 -> risk half of return)
        return_dominance = 0.5  # risk term at w_eq = 50% of return term
        scale = (self.gamma / 2.0) * risk_at_eq / (return_scale * return_dominance)
        sigma_balanced = sigma / scale
        return mu, sigma_balanced

    @staticmethod
    def _return_seeking_start(mu: np.ndarray) -> np.ndarray:
        """
        Starting weights tilted toward higher expected-return models (sum=1, non-negative).

        Avoids starting at equal weight so the solver is less likely to stay there.
        """
        mu_min = mu.min()
        excess = np.maximum(mu - mu_min, 0.0) + 1e-12
        w0 = excess / excess.sum()
        return w0

    @staticmethod
    def _weights_from_means(mu: np.ndarray) -> np.ndarray:
        """
        Weights proportional to excess expected return (mu - min(mu)), normalized to sum 1.

        Used when the QP returns nearly equal weights but mu has variation, so we still
        differentiate by activity-adjusted mean. A floor ensures no model is excluded.
        """
        mu_min = mu.min()
        excess = np.maximum(mu - mu_min, 0.0) + 1e-12
        w = excess / excess.sum()
        n = len(w)
        min_w = 1.0 / (4 * n)  # floor = 25% of equal weight so no model is excluded
        w = np.maximum(w, min_w)
        w = w / w.sum()
        return w

    @staticmethod
    def _weights_from_inverse_activity(fraction_active: np.ndarray) -> np.ndarray:
        """
        Weights proportional to (1 - fraction_active), normalized to sum 1.

        Always-on models (fraction_active=1) get minimum weight; selective models get more.
        Used when activity-adjusted means have no variation (mu_std≈0) so we still
        diversify away from buy_hold. A floor (25% of equal weight) keeps always-on
        models in the portfolio rather than excluding them.
        """
        eps = 1e-12
        inv_activity = np.maximum(1.0 - fraction_active, 0.0) + eps
        w = inv_activity / inv_activity.sum()
        n = len(w)
        min_w = 1.0 / (4 * n)  # floor = 25% of equal weight (e.g. ~4.2% for n=6)
        w = np.maximum(w, min_w)
        w = w / w.sum()
        return w

    def _solve_qp(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        w_eq: np.ndarray,
        n: int,
        x0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Solve the constrained quadratic program via scipy SLSQP.

        Minimise (negative of the objective):
            -(w^T mu) + (gamma/2) w^T sigma w + shrinkage_lambda ||w - w_eq||^2

        Subject to:
            w_i >= 0,  sum(w_i) == 1

        Parameters
        ----------
        mu : np.ndarray (N,)
        sigma : np.ndarray (N, N)
        w_eq : np.ndarray (N,)   – equal-weight vector (fallback on failure)
        n : int                  – number of models
        x0 : np.ndarray (N,), optional – starting point (default w_eq)

        Returns
        -------
        weights : np.ndarray (N,)
            Optimal weights (sums to 1, all non-negative).
        solved : bool
            True if solver converged, False if fell back to equal weights.
        """
        from scipy.optimize import minimize

        gamma = self.gamma
        lam = self.shrinkage_lambda
        w0 = (x0 if x0 is not None else w_eq).copy()
        # Ensure x0 is feasible (sum=1, in bounds)
        w0 = np.maximum(w0, 0.0)
        total = w0.sum()
        if total <= 0:
            w0 = w_eq.copy()
        else:
            w0 = w0 / total

        def objective(w: np.ndarray) -> float:
            portfolio_return = w @ mu
            portfolio_risk = w @ sigma @ w
            eq_penalty = np.sum((w - w_eq) ** 2)
            return float(
                -portfolio_return
                + (gamma / 2.0) * portfolio_risk
                + lam * eq_penalty
            )

        def gradient(w: np.ndarray) -> np.ndarray:
            return (
                -mu
                + gamma * (sigma @ w)
                + 2.0 * lam * (w - w_eq)
            )

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0,
             'jac': lambda w: np.ones(n)},
        ]
        bounds = [(0.0, 1.0)] * n

        result = minimize(
            objective,
            x0=w0,
            jac=gradient,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12},
        )

        if result.success:
            weights = result.x
            weights = np.maximum(weights, 0.0)
            total = weights.sum()
            if total > 0:
                weights = weights / total
            else:
                weights = w_eq.copy()
            return weights, True

        logger.warning(
            f"  Sortino optimizer did not converge: {result.message}. "
            f"Falling back to equal weights."
        )
        return w_eq.copy(), False


# ---------------------------------------------------------------------------
# Backward-compatible factory function
# ---------------------------------------------------------------------------

def WeightLayer(
    weight_method: str = 'inverse_correlation',
    fdm_max: float = 2.5,
    **kwargs,
) -> BaseWeightLayer:
    """
    Factory that creates the appropriate weight layer by method name.

    This function preserves full backward compatibility: all existing code that
    calls ``WeightLayer(...)`` continues to work unchanged.

    Parameters
    ----------
    weight_method : str, default='inverse_correlation'
        Weight calculation strategy. Options:
        - ``'inverse_correlation'`` : Inverse correlation weights (Carver's method)
        - ``'sortino_optimized'``   : Constrained Sortino-ratio optimization
    fdm_max : float, default=2.5
        Maximum FDM value
    **kwargs
        Extra arguments forwarded to the concrete weight layer.
        For ``'sortino_optimized'``: gamma, shrinkage_lambda, mean_shrinkage, min_history.

    Returns
    -------
    BaseWeightLayer
        A fitted-ready weight layer instance.

    Examples
    --------
    >>> layer = WeightLayer()                                     # inverse correlation (default)
    >>> layer = WeightLayer(weight_method='sortino_optimized',
    ...                     gamma=1.0, shrinkage_lambda=0.5)      # Sortino optimized
    """
    match weight_method:
        case 'inverse_correlation':
            return InverseCorrelationWeightLayer(fdm_max=fdm_max)
        case 'sortino_optimized':
            return SortinoOptimizedWeightLayer(fdm_max=fdm_max, **kwargs)
        case _:
            raise ValueError(
                f"Unknown weight method: {weight_method}. "
                f"Supported methods: ['inverse_correlation', 'sortino_optimized']"
            )


__all__ = [
    'WeightLayer',
    'BaseWeightLayer',
    'InverseCorrelationWeightLayer',
    'SortinoOptimizedWeightLayer',
    'InverseCorrelationWeighter',
    'Weighter',
]
