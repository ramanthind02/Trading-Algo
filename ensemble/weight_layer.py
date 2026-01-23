"""
Weight Layer for Forecast Combination

This module implements the Weight Layer that combines forecasts from all base models
across all ensembles using inverse correlation weights and applies the Forecast
Diversification Multiplier (FDM).

The Weight Layer sits between Ensemble and Portfolio layers:
    Ensemble.predict() -> WeightLayer.combine() -> Portfolio.predict()

Key responsibilities:
1. Calculate inverse correlation weights from signal correlations
2. Calculate FDM from forecast value correlations
3. Combine all base model forecasts into a single forecast per instrument

Reference: Robert Carver's "Systematic Trading" and "Leveraged Trading"
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Protocol, Dict


class Weighter(Protocol):
    """Protocol for weight calculation methods."""

    def fit(self, signals: pd.DataFrame) -> None:
        """Fit weights from binary signals."""
        ...

    def get_weights(self) -> pd.Series:
        """Get fitted weights."""
        ...


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

    def __init__(self):
        self.weights_: Optional[pd.Series] = None
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False

    def fit(self, signals: pd.DataFrame) -> 'InverseCorrelationWeighter':
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
        self.correlation_matrix_ = signals.corr().abs()

        # Handle single model case
        if len(signals.columns) == 1:
            model_name = signals.columns[0]
            self.weights_ = pd.Series({model_name: 1.0})
            self.is_fitted_ = True
            return self

        # For each model, calculate average correlation with others
        avg_correlations = {}
        for model_name in signals.columns:
            other_models = [m for m in signals.columns if m != model_name]
            avg_corr = self.correlation_matrix_.loc[model_name, other_models].mean()
            avg_correlations[model_name] = avg_corr

        # Convert to diversification scores (inverse relationship)
        # Higher correlation = lower diversification score
        diversification_scores = {
            model: 1.0 / (1.0 + avg_corr)
            for model, avg_corr in avg_correlations.items()
        }

        # Normalize to sum to 1.0
        total = sum(diversification_scores.values())
        self.weights_ = pd.Series({
            model: score / total
            for model, score in diversification_scores.items()
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


class WeightLayer:
    """
    Combine forecasts from all base models using weight vector and apply FDM.

    The Weight Layer is responsible for:
    1. Fitting weights from signal correlations (delegated to Weighter)
    2. Calculating FDM from forecast value correlations
    3. Combining forecasts: weighted_sum * FDM

    Parameters
    ----------
    weight_method : str, default='inverse_correlation'
        Method for calculating weights. Options:
        - 'inverse_correlation': Use inverse correlation weights (default)
        - Future: 'linear', 'ml' for other methods
    fdm_max : float, default=2.0
        Maximum FDM value (Carver's recommendation)

    Attributes
    ----------
    weighter : Weighter
        The weighting strategy object
    fdm_ : float
        Fitted Forecast Diversification Multiplier
    weights_ : pd.Series
        Model name -> weight mapping (delegated from weighter)
    model_names_ : List[str]
        List of model names in order
    is_fitted_ : bool
        Whether the layer has been fitted

    Examples
    --------
    >>> weight_layer = WeightLayer(fdm_max=2.0)
    >>> weight_layer.fit(forecast_vectors, signals)
    >>> combined = weight_layer.combine(forecast_vectors)
    """

    def __init__(
        self,
        weight_method: str = 'inverse_correlation',
        fdm_max: float = 2.5
    ):
        self.weight_method = weight_method
        self.fdm_max = fdm_max

        # Initialize weighter based on method
        if weight_method == 'inverse_correlation':
            self.weighter: Weighter = InverseCorrelationWeighter()
        else:
            raise ValueError(
                f"Unknown weight method: {weight_method}. "
                f"Supported methods: ['inverse_correlation']"
            )

        # Fitted attributes
        self.fdm_: Optional[float] = None
        self.weights_: Optional[pd.Series] = None
        self.model_names_: Optional[List[str]] = None
        self.mean_forecast_correlation_: Optional[float] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        forecast_vectors: List[pd.DataFrame],
        signals: pd.DataFrame
    ) -> 'WeightLayer':
        """
        Fit weights and FDM from training data.

        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            List of forecast vectors from all ensembles (training data).
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']
        signals : pd.DataFrame
            Binary signals from all base models (for inverse correlation weight calculation).
            Columns: model names, rows: samples

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

        # Fit the weighter (uses signals for correlation)
        self.weighter.fit(signals)
        self.weights_ = self.weighter.get_weights()
        self.model_names_ = list(self.weights_.index)

        # Calculate FDM from forecast value correlations
        self.fdm_ = self._calculate_fdm(forecast_vectors)

        self.is_fitted_ = True
        return self

    def _calculate_fdm(self, forecast_vectors: List[pd.DataFrame]) -> float:
        """
        Calculate Forecast Diversification Multiplier from forecast value correlations.

        FDM accounts for diversification when combining multiple forecasts.
        Lower correlation between forecasts = higher FDM (more diversification benefit).

        Formula:
            FDM = sqrt(1 / (mean_corr + epsilon))
            Capped at fdm_max (typically 2.0)
        
        Steps:
        1. Extract forecast values for all base models
        2. Build correlation matrix of forecast values (not binary signals)
        3. Calculate mean correlation
        4. Calculate FDM: sqrt(1 / (mean_corr + epsilon))
        5. Cap at fdm_max

        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            List of forecast vectors from all ensembles.
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']

        Returns
        -------
        float
            FDM value (capped at fdm_max)
        """
        if not forecast_vectors:
            self.mean_forecast_correlation_ = 1.0
            return 1.0
        
        # Concatenate all forecast vectors
        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        
        if all_forecasts.empty:
            self.mean_forecast_correlation_ = 1.0
            return 1.0
        
        # Extract forecast values for all base models
        # Pivot by model_name (columns) and datetime (index)
        # Need to align by (datetime, ticker) or just datetime
        # For simplicity, group by datetime and take mean per model (across tickers)
        # Or better: group by (datetime, ticker) and pivot by model_name
        
        # Group by datetime and model_name, then pivot
        if 'datetime' in all_forecasts.columns and 'model_name' in all_forecasts.columns:
            # Pivot: datetime x model_name -> forecast values
            forecast_pivot = all_forecasts.pivot_table(
                index='datetime',
                columns='model_name',
                values='forecast',
                aggfunc='mean'  # If multiple tickers, take mean
            )
        else:
            # Fallback: cannot calculate correlation
            self.mean_forecast_correlation_ = 1.0
            return 1.0
        
        # Drop rows with any NaN (need complete data for correlation)
        forecast_pivot = forecast_pivot.dropna()
        
        if len(forecast_pivot) < 2 or len(forecast_pivot.columns) < 2:
            # Need at least 2 time periods and 2 models
            self.mean_forecast_correlation_ = 1.0
            return 1.0
        
        # Build correlation matrix of forecast values
        corr_matrix = forecast_pivot.corr()
        
        # Floor negative correlations at zero (Carver's recommendation)
        corr_matrix = corr_matrix.clip(lower=0.0)
        
        # Calculate mean correlation (excluding diagonal)
        # Get upper triangle (excluding diagonal) and calculate mean
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()
        
        if len(correlations) == 0:
            # No correlations to calculate
            self.mean_forecast_correlation_ = 1.0
            return 1.0
        
        mean_correlation = correlations.mean()
        self.mean_forecast_correlation_ = mean_correlation
        
        # Calculate FDM: sqrt(1 / (mean_correlation + epsilon))
        epsilon = 0.01  # Small epsilon to avoid division by zero
        fdm = np.sqrt(1.0 / (mean_correlation + epsilon))
        
        # Cap at fdm_max
        fdm = min(fdm, self.fdm_max)
        
        return float(fdm)

    def combine(
        self,
        forecast_vectors: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine forecasts from all base models and apply FDM.

        For each instrument:
            forecast_score = (sum of weighted forecasts) * FDM

        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            List of forecast vectors from all ensembles.
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']

        Returns
        -------
        pd.DataFrame
            Combined forecasts with columns: ['ticker', 'forecast_score']
            forecast_score is already FDM-scaled

        Raises
        ------
        ValueError
            If WeightLayer has not been fitted
        """
        if not self.is_fitted_:
            raise ValueError("WeightLayer must be fitted before calling combine()")

        if not forecast_vectors:
            return pd.DataFrame(columns=['ticker', 'forecast_score'])

        # Concatenate all forecast vectors
        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)

        if all_forecasts.empty:
            return pd.DataFrame(columns=['ticker', 'forecast_score'])

        # Map weights to forecasts
        all_forecasts['weight'] = all_forecasts['model_name'].map(self.weights_)

        # Handle missing weights (models not seen during training)
        missing_models = all_forecasts[all_forecasts['weight'].isna()]['model_name'].unique()
        if len(missing_models) > 0:
            # Assign equal weight to unknown models (fallback)
            n_known = len(self.weights_)
            fallback_weight = 1.0 / (n_known + len(missing_models)) if n_known > 0 else 1.0 / len(missing_models)
            all_forecasts['weight'] = all_forecasts['weight'].fillna(fallback_weight)

        # Calculate weighted forecast per row
        all_forecasts['weighted_forecast'] = all_forecasts['forecast'] * all_forecasts['weight']

        # Group by (datetime, ticker) and sum weighted forecasts
        # This preserves datetime information for proper alignment
        if 'datetime' in all_forecasts.columns:
            combined = all_forecasts.groupby(['datetime', 'ticker'], as_index=False).agg({
                'weighted_forecast': 'sum'
            })
        else:
            # Fallback: group by ticker only (loses datetime info)
            combined = all_forecasts.groupby('ticker', as_index=False).agg({
                'weighted_forecast': 'sum'
            })

        # Apply FDM
        combined['forecast_score'] = combined['weighted_forecast'] * self.fdm_
        
        # Cap forecast_score at 2.0 (per spec: max position is 2.0)
        combined['forecast_score'] = combined['forecast_score'].clip(upper=2.0, lower=-2.0)

        # Return required columns (preserve datetime if available)
        if 'datetime' in combined.columns:
            return combined[['ticker', 'datetime', 'forecast_score']]
        else:
            return combined[['ticker', 'forecast_score']]

    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about the fitted WeightLayer.

        Returns
        -------
        dict
            Dictionary containing:
            - weights: Model weights
            - fdm: Forecast Diversification Multiplier
            - mean_forecast_correlation: Mean correlation between forecasts
            - weight_method: Method used for weight calculation
        """
        if not self.is_fitted_:
            return {'is_fitted': False}

        return {
            'is_fitted': True,
            'weights': self.weights_.to_dict() if self.weights_ is not None else None,
            'fdm': self.fdm_,
            'mean_forecast_correlation': self.mean_forecast_correlation_,
            'weight_method': self.weight_method,
            'fdm_max': self.fdm_max,
            'n_models': len(self.model_names_) if self.model_names_ else 0
        }


__all__ = ['WeightLayer', 'InverseCorrelationWeighter', 'Weighter']
