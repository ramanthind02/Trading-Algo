"""
Portfolio Testing Framework

A comprehensive testing framework for portfolios that handles the complete workflow
from candles to tearsheet analysis, with support for granular performance analysis
at portfolio, ensemble, and base model levels.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from metrics.plotting.graphing.quantstats_reports import generate_tearsheet


def _empty_returns_series(name: str) -> pd.Series:
    """Create an empty returns series with a consistent float dtype, DatetimeIndex, and name."""
    return pd.Series(dtype=float, name=name, index=pd.DatetimeIndex([]))


def resample_positions_to_daily(
    positions_df: pd.DataFrame,
    daily_dates_per_ticker: dict[object, pd.DatetimeIndex],
) -> pd.DataFrame:
    """Forward-fill position fractions from sparse/non-daily bars to daily datetimes.

    For each ticker, this function reindexes the signal to ticker daily dates, forward-fills
    active positions, and fills pre-signal dates with 0.0.
    """
    required_columns = {"ticker", "datetime", "position_fraction"}
    missing_columns = required_columns - set(positions_df.columns)
    if missing_columns:
        raise ValueError(
            f"positions_df missing required columns: {sorted(missing_columns)}"
        )

    if positions_df.empty or not daily_dates_per_ticker:
        return pd.DataFrame(columns=["ticker", "datetime", "position_fraction"])

    positions = positions_df.loc[:, ["ticker", "datetime", "position_fraction"]].copy()
    positions["datetime"] = pd.to_datetime(positions["datetime"]).dt.floor("s")

    output_frames = []
    grouped_positions = {
        ticker: grp.sort_values("datetime")
        for ticker, grp in positions.groupby("ticker", sort=False)
    }

    for ticker, ticker_daily_dates in daily_dates_per_ticker.items():
        if len(ticker_daily_dates) == 0:
            continue

        daily_index = pd.DatetimeIndex(pd.to_datetime(ticker_daily_dates)).sort_values()
        ticker_positions = grouped_positions.get(ticker)
        if ticker_positions is None or ticker_positions.empty:
            reindexed = pd.Series(0.0, index=daily_index, dtype=float, name="position_fraction")
        else:
            raw_series = (
                ticker_positions
                .set_index("datetime")["position_fraction"]
                .groupby(level=0)
                .last()
                .sort_index()
            )
            reindexed = raw_series.reindex(daily_index).ffill().fillna(0.0)
            reindexed.name = "position_fraction"

        output_frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "datetime": reindexed.index,
                    "position_fraction": reindexed.to_numpy(dtype=float),
                }
            )
        )

    if not output_frames:
        return pd.DataFrame(columns=["ticker", "datetime", "position_fraction"])

    return (
        pd.concat(output_frames, ignore_index=True)
        .sort_values(["ticker", "datetime"])
        .reset_index(drop=True)
    )


def aggregate_intraday_returns_to_daily(returns: pd.Series) -> pd.Series:
    """Aggregate intraday log-return observations to daily by summing per calendar day."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if returns.empty:
        return returns

    normalized_dates = returns.index.normalize()
    has_intraday_observations = normalized_dates.duplicated().any()
    if not has_intraday_observations:
        return returns

    aggregated = returns.groupby(normalized_dates).sum().sort_index()
    aggregated.name = returns.name
    return aggregated


def calculate_log_returns_from_candles(candles_df: pd.DataFrame) -> pd.Series:
    """
    Calculate log returns from candles DataFrame.
    
    Returns a Series indexed by datetime that can be aligned with candles.
    For multi-ticker data, calculates returns per ticker and combines.
    
    Parameters
    ----------
    candles_df : pd.DataFrame
        Candles DataFrame with columns: datetime, open, high, low, close, volume, ticker
        
    Returns
    -------
    pd.Series
        Returns series indexed by datetime (can be aligned with candles by datetime)
    """
    if candles_df.empty:
        return _empty_returns_series('returns')

    candles_sorted = candles_df.sort_values(['ticker', 'datetime']).copy()
    candles_sorted['datetime'] = pd.to_datetime(candles_sorted['datetime'])

    # Compute log returns per ticker using vectorized groupby + diff
    candles_sorted['log_close'] = np.log(candles_sorted['close'])
    candles_sorted['returns'] = (
        candles_sorted.groupby('ticker')['log_close'].diff()
    )

    # Drop NaNs from the first observation of each ticker
    result = (
        candles_sorted
        .set_index('datetime')['returns']
        .dropna()
        .sort_index()
    )

    result.name = 'returns'
    return result


def calculate_strategy_returns_from_positions(
    positions_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    strategy: str = 'long'
) -> pd.Series:
    """
    Calculate strategy returns from position fractions and candles.

    position_fraction is signed: 1 (long), -1 (short), 0 (flat).
    Strategy return = position_fraction * instrument_return (no strategy branching).

    IMPORTANT:
    - Positions are shifted forward by one period to avoid lookahead bias.
    - Returns are NOT scaled by 1/N; instrument weights account for capital allocation.

    Parameters
    ----------
    positions_df : pd.DataFrame
        Columns: ticker, datetime, position_fraction (signed: -1, 0, 1)
    candles_df : pd.DataFrame
        Columns: datetime, ticker, close
    strategy : str, default='long'
        Kept for backward compatibility; ignored (position is already signed).

    Returns
    -------
    pd.Series
        Daily strategy returns indexed by datetime (summed across tickers)
    """
    if positions_df.empty or candles_df.empty:
        return _empty_returns_series('strategy_return')

    candles_sorted = candles_df.sort_values(['ticker', 'datetime']).copy()
    candles_sorted['datetime'] = pd.to_datetime(candles_sorted['datetime'])
    # Normalize to bar granularity for (datetime, ticker) merge
    candles_sorted['datetime'] = candles_sorted['datetime'].dt.floor('s')

    # Compute log returns per ticker (reused for all strategies)
    candles_sorted['log_close'] = np.log(candles_sorted['close'])
    candles_sorted['instrument_return'] = (
        candles_sorted.groupby('ticker')['log_close'].diff()
    )

    # For each (ticker, datetime) in candles, compute the datetime of the next bar
    candles_sorted['next_datetime'] = (
        candles_sorted.groupby('ticker')['datetime'].shift(-1)
    )

    # Align positions with the *next* bar's return to avoid lookahead bias.
    positions = positions_df.copy()
    positions['datetime'] = pd.to_datetime(positions['datetime'])
    positions['datetime'] = positions['datetime'].dt.floor('s')

    # Merge to find, for each position at time t, the candle row and its next_datetime
    pos_with_next = positions.merge(
        candles_sorted[['ticker', 'datetime', 'next_datetime']],
        on=['ticker', 'datetime'],
        how='left',
    )

    # Drop positions that do not have a future bar
    pos_with_next = pos_with_next.dropna(subset=['next_datetime'])

    if pos_with_next.empty:
        return _empty_returns_series('strategy_return')

    pos_with_next = pos_with_next.rename(columns={'next_datetime': 'ret_datetime'})

    # Now join with instrument returns at ret_datetime
    returns_df = candles_sorted[['ticker', 'datetime', 'instrument_return']].dropna()

    merged = pos_with_next.merge(
        returns_df,
        left_on=['ticker', 'ret_datetime'],
        right_on=['ticker', 'datetime'],
        how='inner',
        suffixes=('', '_ret'),
    )

    if merged.empty:
        return _empty_returns_series('strategy_return')

    # Position is signed (1 / -1 / 0); return = position * instrument_return
    merged['strategy_return'] = merged['position_fraction'] * merged['instrument_return']

    # Group by datetime of the return (ret_datetime) and sum across tickers
    merged['ret_datetime'] = merged['ret_datetime'].astype('datetime64[ns]')
    strategy_returns = (
        merged.groupby('ret_datetime')['strategy_return']
        .sum()
        .sort_index()
    )

    strategy_returns.name = 'strategy_return'

    # #region agent log
    try:
        _pos_n = len(positions_df)
        _pos_zero = (positions_df["position_fraction"] == 0).sum() if "position_fraction" in positions_df.columns else 0
        _ret_n = len(strategy_returns)
        _ret_zero = (strategy_returns == 0.0).sum()
        _pct_zero_pos = float(_pos_zero) / _pos_n if _pos_n else 0.0
        _pct_zero_ret = float(_ret_zero) / _ret_n if _ret_n else 0.0
        with open("/home/raman/repos/Trading-Algo/.cursor/debug.log", "a") as _f:
            import json
            _f.write(
                json.dumps(
                    {
                        "hypothesisId": "B,E",
                        "location": "portfolio_tester.calculate_strategy_returns_from_positions",
                        "message": "strategy returns from positions",
                        "data": {"positions_n": _pos_n, "pct_position_zero": _pct_zero_pos, "returns_n": _ret_n, "pct_returns_zero": _pct_zero_ret},
                        "timestamp": __import__("time").time() * 1000,
                    },
                    default=str,
                )
                + "\n"
            )
    except Exception:  # noqa: S110
        pass
    # #endregion

    return strategy_returns


def calculate_baseline_returns(
    candles_df: pd.DataFrame,
    equal_weight: bool = True
) -> pd.Series:
    """
    Calculate baseline (buy-and-hold) returns from candles.
    
    Parameters
    ----------
    candles_df : pd.DataFrame
        Candles DataFrame with columns: datetime, ticker, close
    equal_weight : bool, default=True
        If True, equal weight all tickers. If False, use single ticker.
        
    Returns
    -------
    pd.Series
        Daily baseline returns indexed by datetime
    """
    if candles_df.empty:
        return _empty_returns_series('baseline_return')

    candles_sorted = candles_df.sort_values(['ticker', 'datetime']).copy()
    candles_sorted['datetime'] = pd.to_datetime(candles_sorted['datetime'])

    # Compute log returns per ticker
    candles_sorted['log_close'] = np.log(candles_sorted['close'])
    candles_sorted['returns'] = (
        candles_sorted.groupby('ticker')['log_close'].diff()
    )

    valid = candles_sorted.dropna(subset=['returns'])
    if valid.empty:
        return _empty_returns_series('baseline_return')

    if equal_weight:
        # Equal-weighted baseline: average across all tickers that have a return on that date
        baseline = (
            valid.groupby('datetime')['returns']
            .mean()
            .sort_index()
        )
    else:
        # Single-ticker buy-and-hold: use the first ticker's returns
        first_ticker = valid['ticker'].iloc[0]
        baseline = (
            valid[valid['ticker'] == first_ticker]
            .set_index('datetime')['returns']
            .sort_index()
        )

    baseline.name = 'baseline_return'
    return baseline


class PortfolioTester:
    """
    Portfolio Testing Framework
    
    Orchestrates portfolio testing workflow: fit portfolio, generate predictions,
    calculate returns, and generate tearsheets at multiple granularity levels.
    
    Parameters
    ----------
    portfolio : Portfolio
        The portfolio to test
    baseline_mode : str, default='equal_weight'
        Baseline strategy mode:
        - 'equal_weight': Equal weight all tickers
        - 'buy_hold': Single ticker buy-and-hold
    """
    
    def __init__(
        self,
        portfolio,
        baseline_mode: str = 'equal_weight'
    ):
        self.portfolio = portfolio
        self.baseline_mode = baseline_mode
        
        # Results storage
        self.positions_df: Optional[pd.DataFrame] = None
        self.ensemble_predictions: Optional[Dict[str, pd.DataFrame]] = None
        self.base_model_predictions: Optional[Dict[str, pd.DataFrame]] = None
        self.strategy_returns: Optional[pd.Series] = None
        self.baseline_returns: Optional[pd.Series] = None
        
    def fit(self, candles_df: pd.DataFrame) -> 'PortfolioTester':
        """
        Fit portfolio using candles DataFrame.
        
        Automatically calculates log returns as target variable.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
            
        Returns
        -------
        self
        """
        # Calculate log returns as target
        target_returns = calculate_log_returns_from_candles(candles_df)
        
        # Fit portfolio
        self.portfolio.fit_from_candles(candles_df, target_returns)
        
        return self
    
    def predict(
        self,
        candles_df: pd.DataFrame,
        return_ensemble_predictions: bool = False,
        return_base_model_predictions: bool = False
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Generate predictions with optional granularity.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame for prediction
        return_ensemble_predictions : bool, default=False
            If True, return ensemble-level predictions
        return_base_model_predictions : bool, default=False
            If True, return base model-level predictions
            
        Returns
        -------
        pd.DataFrame or Dict[str, Any]
            If both flags are False: DataFrame with portfolio positions
            If either flag is True: Dict with 'portfolio', 'ensembles', and/or 'base_models' keys
        """
        # Call portfolio predict with granularity flags
        result = self.portfolio.predict_from_candles(
            candles_df,
            return_ensemble_predictions=return_ensemble_predictions,
            return_base_model_predictions=return_base_model_predictions
        )
        
        # Store results
        if isinstance(result, dict):
            self.positions_df = result.get('portfolio')
            self.ensemble_predictions = result.get('ensembles')
            self.base_model_predictions = result.get('base_models')
        else:
            self.positions_df = result
            self.ensemble_predictions = None
            self.base_model_predictions = None
        
        return result
    
    def calculate_strategy_returns(
        self,
        candles_df: pd.DataFrame,
        positions_df: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Calculate strategy returns from position fractions.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame
        positions_df : pd.DataFrame, optional
            Position fractions. If None, uses self.positions_df
            
        Returns
        -------
        pd.Series
            Strategy returns indexed by datetime
        """
        if positions_df is None:
            positions_df = self.positions_df
        
        if positions_df is None:
            raise ValueError("No positions available. Call predict() first or provide positions_df.")
        
        # For portfolio-level, strategy is mixed (combines multiple ensembles)
        # Use 'long' as default - position_fraction sign already encodes direction
        # (positive = long, negative = short)
        self.strategy_returns = calculate_strategy_returns_from_positions(
            positions_df,
            candles_df,
            strategy='long'  # Portfolio combines strategies, position sign encodes direction
        )
        
        return self.strategy_returns
    
    def calculate_baseline_returns(
        self,
        candles_df: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate baseline returns.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame
            
        Returns
        -------
        pd.Series
            Baseline returns indexed by datetime
        """
        equal_weight = (self.baseline_mode == 'equal_weight')
        self.baseline_returns = calculate_baseline_returns(candles_df, equal_weight=equal_weight)
        
        return self.baseline_returns
    
    def generate_tearsheet(
        self,
        strategy_name: str = 'Portfolio',
        output_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        mode: str = 'full',
        candles_df: Optional[pd.DataFrame] = None,
        strategy_returns: Optional[pd.Series] = None,
        baseline_returns: Optional[pd.Series] = None
    ) -> None:
        """
        Generate QuantStats tearsheet for portfolio.

        Parameters
        ----------
        strategy_name : str, default='Portfolio'
            Name of the strategy for report title
        output_file : str, optional
            If provided, saves HTML report to this file
        output_dir : str, optional
            Directory to save HTML report. If provided, saves to {output_dir}/{strategy_name}_tearsheet.html
        mode : str, default='full'
            Tearsheet mode: 'html', 'full', 'basic', or 'metrics'
        candles_df : pd.DataFrame, optional
            Candles DataFrame. If provided and strategy_returns not calculated, will calculate it
        strategy_returns : pd.Series, optional
            Precomputed strategy returns (e.g. for training set). If provided, used instead of self.
        baseline_returns : pd.Series, optional
            Precomputed baseline returns (e.g. for training set). If provided, used instead of self.
        """
        use_strategy = strategy_returns
        if use_strategy is None:
            if self.strategy_returns is None:
                if candles_df is None:
                    raise ValueError("Either provide candles_df or call calculate_strategy_returns() first")
                self.calculate_strategy_returns(candles_df)
            use_strategy = self.strategy_returns

        use_baseline = baseline_returns
        if use_baseline is None:
            if self.baseline_returns is None:
                if candles_df is None:
                    raise ValueError("Either provide candles_df or call calculate_baseline_returns() first")
                self.calculate_baseline_returns(candles_df)
            use_baseline = self.baseline_returns

        # Determine output file path
        final_output_file = output_file
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            safe_name = strategy_name.replace(' ', '_').replace('::', '_').replace('/', '_')
            final_output_file = str(Path(output_dir) / f"{safe_name}_tearsheet.html")

        # If saving HTML file, save it first
        if final_output_file is not None:
            generate_tearsheet(
                strategy_returns=use_strategy,
                baseline_returns=use_baseline,
                feature_name=strategy_name,
                output_file=final_output_file,
                mode='html'
            )

        # If mode is not 'html' or no output file, also display in notebook
        if mode != 'html' or final_output_file is None:
            generate_tearsheet(
                strategy_returns=use_strategy,
                baseline_returns=use_baseline,
                feature_name=strategy_name,
                output_file=None,
                mode=mode
            )
    
    def generate_ensemble_tearsheets(
        self,
        output_dir: Optional[str] = None,
        mode: str = 'html',
        candles_df: Optional[pd.DataFrame] = None,
        baseline_returns: Optional[pd.Series] = None
    ) -> None:
        """
        Generate tearsheets for each ensemble.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save HTML reports. If provided, saves HTML files regardless of mode.
            If None, only displays in notebook based on mode.
        mode : str, default='full'
            Tearsheet mode: 'html', 'full', 'basic', or 'metrics'
        candles_df : pd.DataFrame, optional
            Candles DataFrame for calculating returns
        baseline_returns : pd.Series, optional
            Precomputed baseline returns (e.g. for training set). If provided, used instead of self.
        """
        if self.ensemble_predictions is None:
            raise ValueError("No ensemble predictions available. Call predict() with return_ensemble_predictions=True")

        if candles_df is None:
            raise ValueError("candles_df is required to calculate returns")

        use_baseline = baseline_returns
        if use_baseline is None and self.baseline_returns is None:
            self.calculate_baseline_returns(candles_df)
        if use_baseline is None:
            use_baseline = self.baseline_returns

        # Create output directory if saving HTML
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for ensemble_name, ensemble_positions in self.ensemble_predictions.items():
            # Extract ensemble index from name (e.g., "ensemble_0" -> 0)
            try:
                ensemble_idx = int(ensemble_name.replace('ensemble_', ''))
                if ensemble_idx < len(self.portfolio.ensembles):
                    ensemble = self.portfolio.ensembles[ensemble_idx]
                    # Determine dominant strategy from base models in this ensemble
                    # If all base models have same strategy, use that; otherwise default to 'long'
                    strategies = set()
                    if hasattr(ensemble, 'base_models'):
                        for base_model in ensemble.base_models.values():
                            if hasattr(base_model, 'binning_model'):
                                strategies.add(base_model.binning_model.strategy)
                            elif hasattr(base_model, 'strategy'):
                                strategies.add(base_model.strategy)

                    # Use strategy if all models agree, otherwise default to 'long'
                    ensemble_strategy = list(strategies)[0] if len(strategies) == 1 else 'long'
                else:
                    ensemble_strategy = 'long'  # Fallback
            except (ValueError, AttributeError, IndexError):
                ensemble_strategy = 'long'  # Fallback if can't determine

            # Calculate returns for this ensemble
            ensemble_returns = calculate_strategy_returns_from_positions(
                ensemble_positions,
                candles_df,
                strategy=ensemble_strategy
            )

            # Determine output file path
            output_file = None
            if output_dir is not None:
                output_file = str(Path(output_dir) / f"{ensemble_name}_tearsheet.html")

            # If saving HTML file, save it first
            if output_file is not None:
                generate_tearsheet(
                    strategy_returns=ensemble_returns,
                    baseline_returns=use_baseline,
                    feature_name=f"Ensemble: {ensemble_name}",
                    output_file=output_file,
                    mode='html'
                )

            # If mode is not 'html' or no output file, also display in notebook
            if mode != 'html' or output_file is None:
                generate_tearsheet(
                    strategy_returns=ensemble_returns,
                    baseline_returns=use_baseline,
                    feature_name=f"Ensemble: {ensemble_name}",
                    output_file=None,
                    mode=mode
                )
    
    def generate_base_model_tearsheets(
        self,
        output_dir: Optional[str] = None,
        mode: str = 'full',
        candles_df: Optional[pd.DataFrame] = None,
        baseline_returns: Optional[pd.Series] = None
    ) -> None:
        """
        Generate tearsheets for each base model.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save HTML reports. If provided, saves HTML files regardless of mode.
            If None, only displays in notebook based on mode.
        mode : str, default='full'
            Tearsheet mode: 'html', 'full', 'basic', or 'metrics'
        candles_df : pd.DataFrame, optional
            Candles DataFrame for calculating returns
        baseline_returns : pd.Series, optional
            Precomputed baseline returns (e.g. for training set). If provided, used instead of self.
        """
        if self.base_model_predictions is None:
            raise ValueError("No base model predictions available. Call predict() with return_base_model_predictions=True")

        if candles_df is None:
            raise ValueError("candles_df is required to calculate returns")

        use_baseline = baseline_returns
        if use_baseline is None and self.baseline_returns is None:
            self.calculate_baseline_returns(candles_df)
        if use_baseline is None:
            use_baseline = self.baseline_returns

        # Create output directory if saving HTML
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for model_name, model_positions in self.base_model_predictions.items():
            # Extract strategy from base model
            # Model name format: "ensemble_X::model_name" or "ensemble_X::feature_name::binning_method"
            model_strategy = 'long'  # Default
            
            try:
                if '::' in model_name:
                    # Parse "ensemble_X::model_name" or "ensemble_X::feature_name::binning_method"
                    parts = model_name.split('::')
                    ensemble_part = parts[0]  # "ensemble_X"
                    ensemble_idx = int(ensemble_part.replace('ensemble_', ''))
                    
                    if len(parts) >= 2:
                        # Try to find the base model - it could be parts[1] or parts[1]::parts[2]
                        base_model_name = '::'.join(parts[1:])  # Join everything after ensemble_X
                        
                        if ensemble_idx < len(self.portfolio.ensembles):
                            ensemble = self.portfolio.ensembles[ensemble_idx]
                            if hasattr(ensemble, 'base_models') and base_model_name in ensemble.base_models:
                                base_model = ensemble.base_models[base_model_name]
                                if hasattr(base_model, 'binning_model'):
                                    model_strategy = base_model.binning_model.strategy
                                elif hasattr(base_model, 'strategy'):
                                    model_strategy = base_model.strategy
            except (ValueError, AttributeError, IndexError, KeyError) as e:
                # Fallback to 'long' if can't determine strategy
                model_strategy = 'long'
            
            # Calculate returns for this base model
            model_returns = calculate_strategy_returns_from_positions(
                model_positions,
                candles_df,
                strategy=model_strategy
            )
            
            # Determine output file path
            output_file = None
            if output_dir is not None:
                # Sanitize model name for filename
                safe_name = model_name.replace('::', '_').replace('/', '_')
                output_file = str(Path(output_dir) / f"{safe_name}_tearsheet.html")
            
            # If saving HTML file, save it first
            if output_file is not None:
                generate_tearsheet(
                    strategy_returns=model_returns,
                    baseline_returns=use_baseline,
                    feature_name=f"Base Model: {model_name}",
                    output_file=output_file,
                    mode='html'
                )

            # If mode is not 'html' or no output file, also display in notebook
            if mode != 'html' or output_file is None:
                generate_tearsheet(
                    strategy_returns=model_returns,
                    baseline_returns=use_baseline,
                    feature_name=f"Base Model: {model_name}",
                    output_file=None,
                    mode=mode
                )
