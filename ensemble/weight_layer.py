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
        avg_correlations = {}
        for model_name in signals.columns:
            other_models = [m for m in signals.columns if m != model_name]
            if len(other_models) > 0:
                avg_corr = self.correlation_matrix_.loc[model_name, other_models].mean()
                # Handle NaN correlations (e.g., constant signals)
                if pd.isna(avg_corr):
                    avg_corr = 0.0  # Treat NaN as zero correlation (maximum diversification)
                avg_correlations[model_name] = avg_corr
            else:
                # Single model case (shouldn't happen here, but handle gracefully)
                avg_correlations[model_name] = 0.0

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
            # Ensure no NaN values in weights (shouldn't happen after fillna, but safety check)
            if self.weights_.isna().any():
                # Replace NaN with equal weights
                n_models = len(self.weights_)
                equal_weight = 1.0 / n_models
                self.weights_ = self.weights_.fillna(equal_weight)
                # Renormalize
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
    fdm_max : float, default=2.5
        Maximum FDM value (Carver's recommendation)

    Attributes
    ----------
    weighter : Weighter
        The weighting strategy object (created per ticker during fit)
    fdm_ : Dict[str, float]
        Ticker -> Fitted Forecast Diversification Multiplier
    weights_ : Dict[str, pd.Series]
        Ticker -> Model name -> weight mapping (delegated from weighter)
    model_names_ : Dict[str, List[str]]
        Ticker -> List of model names in order
    mean_forecast_correlation_ : Dict[str, float]
        Ticker -> Mean correlation between forecasts
    is_fitted_ : bool
        Whether the layer has been fitted

    Examples
    --------
    >>> weight_layer = WeightLayer(fdm_max=2.5)
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

        # Fitted attributes (per-ticker storage)
        self.fdm_: Dict[str, float] = {}  # ticker -> FDM
        self.weights_: Dict[str, pd.Series] = {}  # ticker -> weights Series
        self.model_names_: Dict[str, List[str]] = {}  # ticker -> list of model names
        self.mean_forecast_correlation_: Dict[str, float] = {}  # ticker -> mean correlation
        self.is_fitted_: bool = False

    def fit(
        self,
        forecast_vectors: List[pd.DataFrame],
        signals: pd.DataFrame
    ) -> 'WeightLayer':
        """
        Fit weights and FDM from training data, separately for each ticker.

        Groups forecast_vectors by ticker and calculates ticker-specific weights and FDM
        based on which models are available for each ticker.

        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            List of forecast vectors from all ensembles (training data).
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']
        signals : pd.DataFrame
            Binary signals from all base models (for inverse correlation weight calculation).
            Columns: model names, rows: samples (datetime index)
            Note: Signals are aggregated across tickers, so we need to extract ticker-specific signals

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

        # Concatenate all forecast vectors to extract ticker information
        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        
        if all_forecasts.empty:
            raise ValueError("forecast_vectors contain no data")

        # Extract unique tickers from forecast_vectors
        if 'ticker' not in all_forecasts.columns:
            raise ValueError("forecast_vectors must contain 'ticker' column")
        
        unique_tickers = sorted(all_forecasts['ticker'].unique())
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Fitting WeightLayer for {len(unique_tickers)} ticker(s): {unique_tickers}")
        
        # Fit per-ticker weights and FDM
        for ticker in unique_tickers:
            logger.info(f"\nProcessing ticker: {ticker}")
            
            # Filter forecast_vectors to only this ticker
            ticker_forecast_vectors = []
            for fv in forecast_vectors:
                if isinstance(fv, pd.DataFrame) and 'ticker' in fv.columns:
                    ticker_fv = fv[fv['ticker'] == ticker].copy()
                    if not ticker_fv.empty:
                        ticker_forecast_vectors.append(ticker_fv)
            
            if not ticker_forecast_vectors:
                logger.warning(f"No forecast vectors found for ticker {ticker}, skipping")
                continue
            
            # Determine which models are available for this ticker
            ticker_all_forecasts = pd.concat(ticker_forecast_vectors, ignore_index=True)
            available_models = sorted(ticker_all_forecasts['model_name'].unique())
            
            logger.info(f"  Available models for {ticker}: {available_models} ({len(available_models)} model(s))")
            
            if len(available_models) == 0:
                logger.warning(f"No models found for ticker {ticker}, skipping")
                continue
            
            # Extract signals from forecast_vectors for this ticker
            # This is more reliable than using the aggregated signals DataFrame
            ticker_signals = pd.DataFrame()
            if 'datetime' in ticker_all_forecasts.columns and 'signal' in ticker_all_forecasts.columns:
                ticker_all_forecasts['datetime'] = pd.to_datetime(ticker_all_forecasts['datetime'])
                ticker_all_forecasts['date'] = ticker_all_forecasts['datetime'].dt.normalize()
                
                # Pivot to get signals per model per date
                # Use 'max' to handle cases where same (date, model) appears multiple times
                signal_pivot = ticker_all_forecasts.pivot_table(
                    index='date',
                    columns='model_name',
                    values='signal',
                    aggfunc='max'  # Max signal (1 if any has signal)
                )
                
                # Ensure we only have columns for available models
                existing_models = [m for m in available_models if m in signal_pivot.columns]
                if existing_models:
                    ticker_signals = signal_pivot[existing_models]
                else:
                    ticker_signals = pd.DataFrame()
            else:
                logger.warning(f"Cannot extract signals for ticker {ticker} (missing datetime or signal column)")
                ticker_signals = pd.DataFrame()
            
            # Handle single model case
            if len(available_models) == 1:
                model_name = available_models[0]
                self.weights_[ticker] = pd.Series({model_name: 1.0})
                self.model_names_[ticker] = [model_name]
                self.fdm_[ticker] = 1.0  # No diversification benefit with 1 model
                self.mean_forecast_correlation_[ticker] = 1.0
                logger.info(f"  Ticker {ticker}: Single model '{model_name}', FDM=1.0, weight=1.0")
                continue
            
            # Multiple models: fit weighter and calculate FDM
            if ticker_signals.empty or len(ticker_signals) < 2:
                logger.warning(
                    f"Insufficient signals for ticker {ticker} ({len(ticker_signals)} rows, need >= 2). "
                    f"Using equal weights and FDM=1.0"
                )
                equal_weight = 1.0 / len(available_models)
                self.weights_[ticker] = pd.Series({m: equal_weight for m in available_models})
                self.model_names_[ticker] = available_models
                self.fdm_[ticker] = 1.0
                self.mean_forecast_correlation_[ticker] = 1.0
                continue
            
            # Create a new weighter instance for this ticker
            ticker_weighter = InverseCorrelationWeighter()
            ticker_weighter.fit(ticker_signals)
            ticker_weights = ticker_weighter.get_weights()
            
            self.weights_[ticker] = ticker_weights
            self.model_names_[ticker] = list(ticker_weights.index)
            
            # Calculate ticker-specific FDM
            ticker_fdm = self._calculate_fdm(ticker_forecast_vectors, ticker=ticker)
            self.fdm_[ticker] = ticker_fdm
            
            logger.info(
                f"  Ticker {ticker}: FDM={ticker_fdm:.4f}, "
                f"models={len(available_models)}, "
                f"mean_correlation={self.mean_forecast_correlation_.get(ticker, 'N/A')}"
            )

        self.is_fitted_ = True
        return self

    def _calculate_fdm(self, forecast_vectors: List[pd.DataFrame], ticker: Optional[str] = None) -> float:
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
        ticker : str, optional
            Ticker name for storing mean_forecast_correlation_ (if provided)

        Returns
        -------
        float
            FDM value (capped at fdm_max)
        """
        if not forecast_vectors:
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = 1.0
            return 1.0
        
        # Concatenate all forecast vectors
        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        
        if all_forecasts.empty:
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = 1.0
            return 1.0
        
        # Extract forecast values for all base models
        # Pivot by model_name (columns) and datetime (index)
        # Need to align by (datetime, ticker) or just datetime
        # For simplicity, group by datetime and take mean per model (across tickers)
        # Or better: group by (datetime, ticker) and pivot by model_name
        
        # Group by datetime and model_name, then pivot
        if 'datetime' in all_forecasts.columns and 'model_name' in all_forecasts.columns:
            # Normalize datetime to remove millisecond precision for proper alignment
            # This ensures we group by actual trading day, not millisecond timestamps
            all_forecasts = all_forecasts.copy()
            all_forecasts['datetime'] = pd.to_datetime(all_forecasts['datetime'])
            # Normalize to date-only (remove time component)
            all_forecasts['date'] = all_forecasts['datetime'].dt.normalize()
            
            # For FDM calculation, we want to preserve ticker-level granularity
            # because averaging across tickers can create artificial correlations (zeros are perfectly correlated)
            # Instead, pivot by (date, ticker) to keep ticker-level forecasts
            # This gives us more data points and better correlation estimates
            if 'ticker' in all_forecasts.columns:
                # Create a multi-index with (date, ticker) for more granular correlation
                forecast_pivot = all_forecasts.pivot_table(
                    index=['date', 'ticker'],
                    columns='model_name',
                    values='forecast',
                    aggfunc='first'  # Should be unique per (date, ticker, model)
                )
            else:
                # Fallback: group by date only
                forecast_pivot = all_forecasts.pivot_table(
                    index='date',
                    columns='model_name',
                    values='forecast',
                    aggfunc='mean'  # Mean across tickers for each date
                )
        else:
            # Fallback: cannot calculate correlation
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = 1.0
            return 1.0
        
        # Drop rows with any NaN (need complete data for correlation)
        # But first, check how many rows we have
        rows_before_drop = len(forecast_pivot)
        forecast_pivot = forecast_pivot.dropna()
        rows_after_drop = len(forecast_pivot)
        
        if rows_after_drop < 2 or len(forecast_pivot.columns) < 2:
            # Need at least 2 time periods and 2 models
            import logging
            logger = logging.getLogger(__name__)
            ticker_str = str(ticker) if ticker is not None else ''
            logger.warning(
                f"Insufficient data for FDM calculation{' for ticker ' + ticker_str if ticker_str else ''}: "
                f"{rows_after_drop} rows (need >= 2), {len(forecast_pivot.columns)} columns (need >= 2). "
                f"Before dropna: {rows_before_drop} rows. "
                f"This suggests models have no overlapping datetimes or all forecasts are NaN."
            )
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = 1.0
            return 1.0
        
        # Build correlation matrix of forecast values
        corr_matrix = forecast_pivot.corr()
        
        # Log debug information (not printed to console)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"\nForecast pivot for FDM calculation:")
        logger.debug(f"  Shape: {forecast_pivot.shape} (rows x columns)")
        logger.debug(f"  Index type: {type(forecast_pivot.index)}")
        logger.debug(f"  Columns: {list(forecast_pivot.columns)}")
        
        logger.debug(f"\nForecast pivot statistics:")
        for col in forecast_pivot.columns:
            col_data = forecast_pivot[col]
            non_zero = (col_data != 0).sum()
            unique_vals = col_data.nunique()
            stats_msg = (
                f"  {col}: "
                f"non-zero={non_zero}/{len(col_data)} ({100.0*non_zero/len(col_data):.1f}%), "
                f"unique_vals={unique_vals}, "
                f"mean={col_data.mean():.6f}, "
                f"std={col_data.std():.6f}"
            )
            logger.debug(stats_msg)
        
        logger.debug(f"\nRaw correlation matrix:\n{corr_matrix}")
        
        # Check for constant columns (all same value) which cause NaN correlations
        # Replace NaN correlations with 1.0 (perfect correlation for constant forecasts)
        constant_cols = []
        for col in forecast_pivot.columns:
            if forecast_pivot[col].nunique() <= 1:
                constant_cols.append(col)
                # Set correlation with other columns to 1.0 (constant = perfectly correlated)
                corr_matrix.loc[col, :] = 1.0
                corr_matrix.loc[:, col] = 1.0
        
        if constant_cols:
            logger.warning(
                f"Found constant forecast columns (all same value): {constant_cols}. "
                f"These will be treated as perfectly correlated (corr=1.0) for FDM calculation."
            )
        
        # Fill any remaining NaN with 1.0 (treat as perfect correlation)
        # But first log which correlations are NaN
        nan_mask = corr_matrix.isna()
        if nan_mask.any().any():
            nan_pairs = []
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    if nan_mask.iloc[i, j] and i != j:  # Exclude diagonal
                        nan_pairs.append((corr_matrix.index[i], corr_matrix.columns[j]))
            if nan_pairs:
                logger.warning(
                    f"Found {len(nan_pairs)} NaN correlations (likely due to constant or perfectly correlated forecasts): "
                    f"{nan_pairs[:5]}{'...' if len(nan_pairs) > 5 else ''}"
                )
        
        corr_matrix = corr_matrix.fillna(1.0)
        
        # Floor negative correlations at zero (Carver's recommendation)
        corr_matrix = corr_matrix.clip(lower=0.0)
        
        # Calculate mean correlation (excluding diagonal)
        # Get upper triangle (excluding diagonal) and calculate mean
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()
        
        if len(correlations) == 0:
            # No correlations to calculate (shouldn't happen with 2+ models)
            if ticker is not None:
                self.mean_forecast_correlation_[ticker] = 1.0
            return 1.0
        
        mean_correlation = correlations.mean()
        if ticker is not None:
            self.mean_forecast_correlation_[ticker] = mean_correlation
        
        # Log individual correlations to debug only (not printed)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"\nIndividual forecast correlations:")
        for (model1, model2), corr_val in correlations.items():
            corr_msg = f"  {model1} vs {model2}: {corr_val:.4f}"
            logger.debug(corr_msg)
        
        # Log FDM calculation details to debug only
        ticker_str = str(ticker) if ticker is not None else ''
        fdm_msg = (
            f"FDM calculation{' for ticker ' + ticker_str if ticker_str else ''}: "
            f"{rows_after_drop} rows, {len(forecast_pivot.columns)} models, "
            f"mean correlation: {mean_correlation:.4f}"
        )
        logger.debug(fdm_msg)
        
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
        Combine forecasts from all base models and apply ticker-specific FDM.

        For each ticker:
            forecast_score = (sum of weighted forecasts) * ticker_FDM

        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            List of forecast vectors from all ensembles.
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']

        Returns
        -------
        pd.DataFrame
            Combined forecasts with columns: ['ticker', 'forecast_score'] (or ['ticker', 'datetime', 'forecast_score'])
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

        # Group by ticker and apply ticker-specific weights and FDM
        combined_results = []
        
        for ticker in all_forecasts['ticker'].unique():
            ticker_forecasts = all_forecasts[all_forecasts['ticker'] == ticker].copy()
            
            # Get ticker-specific weights (fallback to equal weights if ticker not seen during fit)
            if ticker in self.weights_:
                ticker_weights = self.weights_[ticker]
            else:
                # Fallback: equal weights for all models for this ticker
                available_models = ticker_forecasts['model_name'].unique()
                n_models = len(available_models)
                equal_weight = 1.0 / n_models if n_models > 0 else 1.0
                ticker_weights = pd.Series({m: equal_weight for m in available_models})
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Ticker {ticker} not seen during fit(), using equal weights "
                    f"({equal_weight:.4f} per model, {n_models} models)"
                )
            
            # Map ticker-specific weights to forecasts
            ticker_forecasts['weight'] = ticker_forecasts['model_name'].map(ticker_weights)
            
            # Handle missing weights (models not in ticker's weight dict)
            missing_models = ticker_forecasts[ticker_forecasts['weight'].isna()]['model_name'].unique()
            if len(missing_models) > 0:
                # Assign equal weight to unknown models (fallback)
                n_known = len(ticker_weights)
                fallback_weight = 1.0 / (n_known + len(missing_models)) if n_known > 0 else 1.0 / len(missing_models)
                ticker_forecasts['weight'] = ticker_forecasts['weight'].fillna(fallback_weight)
            
            # Calculate weighted forecast per row
            ticker_forecasts['weighted_forecast'] = ticker_forecasts['forecast'] * ticker_forecasts['weight']
            
            # Group by datetime (if available) and sum weighted forecasts
            if 'datetime' in ticker_forecasts.columns:
                ticker_combined = ticker_forecasts.groupby('datetime', as_index=False).agg({
                    'weighted_forecast': 'sum'
                })
                ticker_combined['ticker'] = ticker
            else:
                # Fallback: single row per ticker
                ticker_combined = pd.DataFrame({
                    'ticker': [ticker],
                    'weighted_forecast': [ticker_forecasts['weighted_forecast'].sum()]
                })
            
            # Apply ticker-specific FDM (fallback to 1.0 if ticker not seen during fit)
            ticker_fdm = self.fdm_.get(ticker, 1.0)
            ticker_combined['forecast_score'] = ticker_combined['weighted_forecast'] * ticker_fdm
            
            # Cap forecast_score at 2.0 (per spec: max position is 2.0)
            ticker_combined['forecast_score'] = ticker_combined['forecast_score'].clip(upper=2.0, lower=-2.0)
            
            combined_results.append(ticker_combined)
        
        # Combine all ticker results
        if combined_results:
            combined = pd.concat(combined_results, ignore_index=True)
            # Return required columns (preserve datetime if available)
            if 'datetime' in combined.columns:
                return combined[['ticker', 'datetime', 'forecast_score']]
            else:
                return combined[['ticker', 'forecast_score']]
        else:
            return pd.DataFrame(columns=['ticker', 'forecast_score'])

    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about the fitted WeightLayer.

        Returns
        -------
        dict
            Dictionary containing:
            - is_fitted: Whether layer is fitted
            - tickers: Per-ticker information (fdm, weights, models, mean_correlation)
            - summary: Summary statistics (mean_fdm, total_tickers, etc.)
            - weight_method: Method used for weight calculation
            - fdm_max: Maximum FDM value
        """
        if not self.is_fitted_:
            return {'is_fitted': False}

        # Build per-ticker diagnostics
        tickers_dict = {}
        for ticker in self.fdm_.keys():
            ticker_weights = self.weights_.get(ticker)
            weights_dict = None
            if ticker_weights is not None:
                weights_dict = ticker_weights.to_dict()
                # Replace any NaN values with 0 (shouldn't happen, but safety check)
                weights_dict = {k: (v if not pd.isna(v) else 0.0) for k, v in weights_dict.items()}
            
            tickers_dict[ticker] = {
                'fdm': self.fdm_.get(ticker, 1.0),
                'weights': weights_dict,
                'models': self.model_names_.get(ticker, []),
                'n_models': len(self.model_names_.get(ticker, [])),
                'mean_forecast_correlation': self.mean_forecast_correlation_.get(ticker, 1.0)
            }
        
        # Calculate summary statistics
        fdm_values = list(self.fdm_.values())
        mean_fdm = np.mean(fdm_values) if fdm_values else 1.0
        min_fdm = min(fdm_values) if fdm_values else 1.0
        max_fdm = max(fdm_values) if fdm_values else 1.0
        
        # Count models per ticker
        n_models_per_ticker = [len(self.model_names_.get(t, [])) for t in self.fdm_.keys()]
        mean_models_per_ticker = np.mean(n_models_per_ticker) if n_models_per_ticker else 0.0
        
        return {
            'is_fitted': True,
            'tickers': tickers_dict,
            'summary': {
                'n_tickers': len(self.fdm_),
                'mean_fdm': float(mean_fdm),
                'min_fdm': float(min_fdm),
                'max_fdm': float(max_fdm),
                'mean_models_per_ticker': float(mean_models_per_ticker),
                'total_models': sum(len(self.model_names_.get(t, [])) for t in self.fdm_.keys())
            },
            'weight_method': self.weight_method,
            'fdm_max': self.fdm_max
        }


__all__ = ['WeightLayer', 'InverseCorrelationWeighter', 'Weighter']
