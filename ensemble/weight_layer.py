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
        Weight calculation strategy. Only ``'inverse_correlation'`` is supported.
    fdm_max : float, default=2.5
        Maximum FDM value
    **kwargs
        Ignored. Included for backward-compatibility in call sites.

    Returns
    -------
    BaseWeightLayer
        A fitted-ready weight layer instance.

    Examples
    --------
    >>> layer = WeightLayer()                                     # inverse correlation (default)
    """
    if weight_method != 'inverse_correlation':
        raise ValueError(
            f"Unknown weight method: {weight_method}. "
            "Supported methods: ['inverse_correlation']"
        )
    return InverseCorrelationWeightLayer(fdm_max=fdm_max)


__all__ = [
    'WeightLayer',
    'BaseWeightLayer',
    'InverseCorrelationWeightLayer',
    'InverseCorrelationWeighter',
    'Weighter',
]
