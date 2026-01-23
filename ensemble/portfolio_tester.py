"""
Portfolio Testing Framework

A comprehensive testing framework for portfolios that handles the complete workflow
from candles to tearsheet analysis, with support for granular performance analysis
at portfolio, ensemble, and base model levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Any
from pathlib import Path

from metrics.plotting.graphing.quantstats_reports import generate_tearsheet


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
    candles_df = candles_df.sort_values(['ticker', 'datetime']).copy()
    
    # Calculate log returns per ticker and combine
    returns_list = []
    for ticker in candles_df['ticker'].unique():
        ticker_candles = candles_df[candles_df['ticker'] == ticker].copy()
        ticker_candles = ticker_candles.sort_values('datetime')
        
        # Calculate log returns
        ticker_candles['returns'] = np.log(ticker_candles['close'] / ticker_candles['close'].shift(1))
        
        # Set datetime as index
        ticker_candles = ticker_candles.set_index('datetime')
        
        # Extract returns
        ticker_returns = ticker_candles['returns'].dropna()
        returns_list.append(ticker_returns)
    
    # Combine all ticker returns (may have duplicate datetime indices from different tickers)
    if returns_list:
        combined_returns = pd.concat(returns_list)
        # Sort by datetime
        combined_returns = combined_returns.sort_index()
        return combined_returns
    else:
        return pd.Series(dtype=float, name='returns')


def calculate_strategy_returns_from_positions(
    positions_df: pd.DataFrame,
    candles_df: pd.DataFrame
) -> pd.Series:
    """
    Calculate strategy returns from position fractions and candles.
    
    Strategy return = position_fraction * instrument_return
    
    IMPORTANT: 
    - Positions are shifted forward by one period to avoid lookahead bias.
      A prediction made at time x using candle x is only available at time x+1.
    - Returns are NOT scaled by 1/N - instrument weights already account for capital allocation.
    
    Parameters
    ----------
    positions_df : pd.DataFrame
        Position fractions with columns: ticker, datetime, position_fraction
    candles_df : pd.DataFrame
        Candles DataFrame with columns: datetime, ticker, close
        
    Returns
    -------
    pd.Series
        Daily strategy returns indexed by datetime (summed across tickers)
    """
    # Calculate returns per ticker
    candles_sorted = candles_df.sort_values(['ticker', 'datetime']).copy()
    
    # Calculate log returns per ticker
    ticker_returns = {}
    for ticker in candles_sorted['ticker'].unique():
        ticker_candles = candles_sorted[candles_sorted['ticker'] == ticker].copy()
        ticker_candles = ticker_candles.sort_values('datetime')
        
        # Calculate log returns
        ticker_candles['returns'] = np.log(ticker_candles['close'] / ticker_candles['close'].shift(1))
        ticker_candles = ticker_candles.set_index('datetime')
        
        ticker_returns[ticker] = ticker_candles['returns'].dropna()
    
    # Note: No need to track num_tickers for scaling - instrument weights handle allocation
    
    # Shift positions forward by one period to avoid lookahead bias
    # A prediction at time x using candle x should only be available at time x+1
    # The return at time x+1 represents the return from x to x+1
    positions_shifted = positions_df.copy()
    positions_shifted['datetime'] = pd.to_datetime(positions_shifted['datetime'])
    
    # Group by ticker and shift datetime forward by one period
    positions_shifted_list = []
    for ticker in positions_shifted['ticker'].unique():
        ticker_positions = positions_shifted[positions_shifted['ticker'] == ticker].copy()
        ticker_positions = ticker_positions.sort_values('datetime')
        
        # Get the datetime index for this ticker's returns to find next valid datetime
        if ticker in ticker_returns:
            ticker_ret = ticker_returns[ticker]
            ticker_ret_index = ticker_ret.index
            
            # Shift each position to the next available datetime in returns
            # This ensures position at time x applies to return from x to x+1
            shifted_data = []
            for _, pos_row in ticker_positions.iterrows():
                dt = pos_row['datetime']
                position_fraction = pos_row['position_fraction']
                
                # Find next datetime in returns that is > current datetime
                # searchsorted with side='right' finds insertion point after any existing dt
                next_dt_idx = ticker_ret_index.searchsorted(dt, side='right')
                if next_dt_idx < len(ticker_ret_index):
                    next_dt = ticker_ret_index[next_dt_idx]
                    shifted_data.append({
                        'ticker': ticker,
                        'datetime': next_dt,
                        'position_fraction': position_fraction
                    })
                # If no future datetime, skip this position (it's at the end of the data)
            
            if shifted_data:
                ticker_positions_shifted = pd.DataFrame(shifted_data)
                positions_shifted_list.append(ticker_positions_shifted)
    
    if not positions_shifted_list:
        return pd.Series(dtype=float, name='strategy_return')
    
    positions_shifted = pd.concat(positions_shifted_list, ignore_index=True)
    
    # Merge shifted positions with returns
    strategy_returns_list = []
    
    for _, pos_row in positions_shifted.iterrows():
        ticker = pos_row['ticker']
        dt = pd.to_datetime(pos_row['datetime'])
        position_fraction = pos_row['position_fraction']
        
        # Get return for this ticker and datetime
        if ticker in ticker_returns:
            ticker_ret = ticker_returns[ticker]
            if dt in ticker_ret.index:
                # Strategy return = position_fraction * instrument_return
                # No 1/N scaling needed - instrument weights already account for capital allocation
                strategy_return = position_fraction * ticker_ret.loc[dt]
                strategy_returns_list.append({
                    'datetime': dt,
                    'return': strategy_return
                })
    
    if not strategy_returns_list:
        return pd.Series(dtype=float, name='strategy_return')
    
    # Convert to Series
    strategy_returns_df = pd.DataFrame(strategy_returns_list)
    strategy_returns_df = strategy_returns_df.set_index('datetime')
    strategy_returns_df = strategy_returns_df.sort_index()
    
    # Group by datetime and sum (if multiple tickers on same day)
    # Summing gives correct portfolio return (instrument weights already account for allocation)
    strategy_returns = strategy_returns_df.groupby('datetime')['return'].sum()
    
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
    candles_sorted = candles_df.sort_values(['ticker', 'datetime']).copy()
    
    # Calculate returns per ticker
    ticker_returns_dict = {}
    for ticker in candles_sorted['ticker'].unique():
        ticker_candles = candles_sorted[candles_sorted['ticker'] == ticker].copy()
        ticker_candles = ticker_candles.sort_values('datetime')
        
        # Calculate log returns
        ticker_candles['returns'] = np.log(ticker_candles['close'] / ticker_candles['close'].shift(1))
        ticker_candles = ticker_candles.set_index('datetime')
        
        ticker_returns_dict[ticker] = ticker_candles['returns'].dropna()
    
    if not ticker_returns_dict:
        return pd.Series(dtype=float, name='baseline_return')
    
    # Combine ticker returns
    if equal_weight:
        # Equal weight: average returns across tickers
        # Get all unique datetimes
        all_dates = set()
        for returns in ticker_returns_dict.values():
            all_dates.update(returns.index)
        all_dates = sorted(all_dates)
        
        # Average returns across tickers for each date
        baseline_returns_list = []
        for dt in all_dates:
            returns_on_date = []
            for ticker, returns in ticker_returns_dict.items():
                if dt in returns.index:
                    returns_on_date.append(returns.loc[dt])
            
            if returns_on_date:
                avg_return = np.mean(returns_on_date)
                baseline_returns_list.append({'datetime': dt, 'return': avg_return})
        
        baseline_df = pd.DataFrame(baseline_returns_list)
        baseline_df = baseline_df.set_index('datetime')
        baseline_returns = baseline_df['return'].sort_index()
    else:
        # Single ticker: use first ticker
        first_ticker = list(ticker_returns_dict.keys())[0]
        baseline_returns = ticker_returns_dict[first_ticker]
    
    return baseline_returns


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
        
        self.strategy_returns = calculate_strategy_returns_from_positions(
            positions_df,
            candles_df
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
        candles_df: Optional[pd.DataFrame] = None
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
        """
        # Calculate strategy returns if needed
        if self.strategy_returns is None:
            if candles_df is None:
                raise ValueError("Either provide candles_df or call calculate_strategy_returns() first")
            self.calculate_strategy_returns(candles_df)
        
        # Calculate baseline returns if needed
        if self.baseline_returns is None:
            if candles_df is None:
                raise ValueError("Either provide candles_df or call calculate_baseline_returns() first")
            self.calculate_baseline_returns(candles_df)
        
        # Determine output file path
        final_output_file = output_file
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            safe_name = strategy_name.replace(' ', '_').replace('::', '_').replace('/', '_')
            final_output_file = str(Path(output_dir) / f"{safe_name}_tearsheet.html")
        
        # If saving HTML file, save it first
        if final_output_file is not None:
            generate_tearsheet(
                strategy_returns=self.strategy_returns,
                baseline_returns=self.baseline_returns,
                feature_name=strategy_name,
                output_file=final_output_file,
                mode='html'
            )
        
        # If mode is not 'html' or no output file, also display in notebook
        if mode != 'html' or final_output_file is None:
            generate_tearsheet(
                strategy_returns=self.strategy_returns,
                baseline_returns=self.baseline_returns,
                feature_name=strategy_name,
                output_file=None,
                mode=mode
            )
    
    def generate_ensemble_tearsheets(
        self,
        output_dir: Optional[str] = None,
        mode: str = 'html',
        candles_df: Optional[pd.DataFrame] = None
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
        """
        if self.ensemble_predictions is None:
            raise ValueError("No ensemble predictions available. Call predict() with return_ensemble_predictions=True")
        
        if candles_df is None:
            raise ValueError("candles_df is required to calculate returns")
        
        # Create output directory if saving HTML
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for ensemble_name, ensemble_positions in self.ensemble_predictions.items():
            # Calculate returns for this ensemble
            ensemble_returns = calculate_strategy_returns_from_positions(
                ensemble_positions,
                candles_df
            )
            
            # Calculate baseline returns if not already done
            if self.baseline_returns is None:
                self.calculate_baseline_returns(candles_df)
            
            # Determine output file path
            output_file = None
            if output_dir is not None:
                output_file = str(Path(output_dir) / f"{ensemble_name}_tearsheet.html")
            
            # If saving HTML file, save it first
            if output_file is not None:
                generate_tearsheet(
                    strategy_returns=ensemble_returns,
                    baseline_returns=self.baseline_returns,
                    feature_name=f"Ensemble: {ensemble_name}",
                    output_file=output_file,
                    mode='html'
                )
            
            # If mode is not 'html' or no output file, also display in notebook
            if mode != 'html' or output_file is None:
                generate_tearsheet(
                    strategy_returns=ensemble_returns,
                    baseline_returns=self.baseline_returns,
                    feature_name=f"Ensemble: {ensemble_name}",
                    output_file=None,
                    mode=mode
                )
    
    def generate_base_model_tearsheets(
        self,
        output_dir: Optional[str] = None,
        mode: str = 'full',
        candles_df: Optional[pd.DataFrame] = None
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
        """
        if self.base_model_predictions is None:
            raise ValueError("No base model predictions available. Call predict() with return_base_model_predictions=True")
        
        if candles_df is None:
            raise ValueError("candles_df is required to calculate returns")
        
        # Create output directory if saving HTML
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate baseline returns if not already done
        if self.baseline_returns is None:
            self.calculate_baseline_returns(candles_df)
        
        for model_name, model_positions in self.base_model_predictions.items():
            # Calculate returns for this base model
            model_returns = calculate_strategy_returns_from_positions(
                model_positions,
                candles_df
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
                    baseline_returns=self.baseline_returns,
                    feature_name=f"Base Model: {model_name}",
                    output_file=output_file,
                    mode='html'
                )
            
            # If mode is not 'html' or no output file, also display in notebook
            if mode != 'html' or output_file is None:
                generate_tearsheet(
                    strategy_returns=model_returns,
                    baseline_returns=self.baseline_returns,
                    feature_name=f"Base Model: {model_name}",
                    output_file=None,
                    mode=mode
                )
