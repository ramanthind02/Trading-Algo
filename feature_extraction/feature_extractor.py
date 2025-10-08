
from datetime import datetime, timezone
from utils.enums import TimeFrame, Ticker
from feature_extraction.backtest import Backtest
import utils.helpers as helpers
from utils.candle_fetcher import CandleFetcher
import pandas as pd
import numpy as np
from typing import List


def get_backtester(name: str, start: datetime, end: datetime, middleman):
    """
    Create a backtester instance with the specified parameters.
    
    Parameters
    ----------
    name : str
        Name identifier for the backtest
    start : datetime
        Start date for the data
    end : datetime
        End date for the data
    middleman : object
        Middleman object for the backtest
        
    Returns
    -------
    Backtest
        Configured backtest instance
    """
    data = helpers.load_numpy_data(middleman.ticker, TimeFrame.D, start=start, end=end)
    candle_fetcher = CandleFetcher(
        ticker=middleman.ticker, 
        tfs=[TimeFrame.W, TimeFrame.M]
    )
    return Backtest(name, data, middleman, candle_fetcher)


def extract_bias(
        ticker: Ticker,
        start: datetime = datetime(1990, 1, 1),
        end: datetime = datetime.now()
) -> tuple[pd.DataFrame, dict]:
    """
    Process a single timeframe for bias feature extraction.
    
    Parameters
    ----------
    ticker : Ticker
        Ticker symbol to process
    start : datetime
        Start date for the data
    end : datetime
        End date for the data
        
    Returns
    -------
    tuple[pd.DataFrame, dict]
        Tuple containing:
        - Processed and normalized features dataframe
        - Columns dictionary for feature identification
    """
    # Build middleman and run backtest
    ml_manager = helpers.create_ml_manager(ticker)
    
    # Use the get_backtester function to create the backtest instance
    features_backtest = get_backtester("features", start, end, ml_manager)
    features_backtest.run()
    
    # Extract features and clear middleman
    # Use matrix_df property to ensure buffer is flushed
    features = ml_manager.matrix_df
    columns = ml_manager.columns
    
    return features, columns


def extract_feature(
        ticker: Ticker,
        start: datetime = datetime(2000, 1, 1),
        end: datetime = datetime.now()
) -> pd.DataFrame:
    """
    Extract and combine price data with bias features.
    
    Parameters
    ----------
    ticker : Ticker
        Ticker symbol to process
    start : datetime
        Start date for the data
    end : datetime
        End date for the data
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with price data, log return target, and bias features
    """
    # Load price data
    data = helpers.load_data(ticker, TimeFrame.D)
    
    # Calculate log return as target
    data['target'] = np.log(data['close'] / data['open'])
    
    # Set datetime as index for joining
    data.set_index('datetime', inplace=True)
    
    # Extract bias features
    features, cols = extract_bias(ticker=ticker, start=start, end=end)
    
    data.index = data.index.tz_localize('UTC')
    
    # Join price data with features
    full_df = data.join(features, how='inner')
    
    # Add ticker column
    full_df['ticker'] = ticker.name
    
    return full_df


def extract_features_multi(
        tickers: List[Ticker],
        start: datetime = datetime(2000, 1, 1),
        end: datetime = datetime.now(),
) -> pd.DataFrame:
    """
    Extract and combine features for multiple tickers by appending them together.
    Uses a simple index for compatibility with plotting code.
    
    Parameters
    ----------
    tickers : List[Ticker]
        List of ticker symbols to process
    start : datetime
        Start date for the data
    end : datetime
        End date for the data
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with data from all tickers using a simple index
    """
    if not tickers:
        raise ValueError("No tickers provided")
    
    # Extract features for each ticker
    all_dfs = []
    for ticker in tickers:
        try:
            df = extract_feature(ticker=ticker, start=start, end=end)
            
            # Reset the index to make datetime a regular column
            df = df.reset_index()
            
            # Make sure the ticker column exists and has the correct value
            df['ticker'] = ticker.name
            
            all_dfs.append(df)
        except Exception as e:
            print(f"Error extracting features for {ticker.name}: {str(e)}")
    
    if not all_dfs:
        raise ValueError("Failed to extract features for any ticker")
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    
    # Sort by datetime
    combined_df = combined_df.sort_values('datetime')
    
    # Reset index after sorting to ensure sequential indices for plotting
    combined_df = combined_df.reset_index(drop=True)
    
    return combined_df