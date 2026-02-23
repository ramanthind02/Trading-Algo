# Standard library imports
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Union

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

# Third party imports
import pandas as pd

# Local imports
from utils.core.enums import TimeFrame, Ticker


def txt_to_parquet(folder_path: str, all_tickers: bool = False, ticker: Optional[Ticker] = None) -> None:
    """
    Convert daily data text files to parquet format.
    Simplified version that doesn't use chunking or Dask for small daily data files.

    Parameters:
        folder_path: Path to the folder containing the text files
        all_tickers: Whether to process all tickers or just one
        ticker: Specific ticker to process if all_tickers is False

    Returns:
        None
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    if all_tickers:
        # Use the Ticker enum instead of hardcoding the list
        pairs = [t.name for t in Ticker]
    else:
        pairs = [ticker.name]

    for pair in pairs:
        # Use absolute paths
        input_path = os.path.join(folder_path, f'{pair}.txt')
        output_path = os.path.join(project_root, 'data', 'parquet_data', f'{pair}.parquet')

        # Ensure the data directories exist
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not os.path.exists(input_path):
            print(f"Warning: Input file not found: {input_path}")
            continue

        # Read the entire file at once - for daily data this should be manageable
        try:
            # Assuming the daily data format has date, open, high, low, close, volume columns
            df = pd.read_csv(input_path, names=['date', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert date to datetime format
            df['date'] = pd.to_datetime(df['date'])
            
            # Create timestamp column (seconds since epoch)
            df['timestamp'] = df['date'].astype(int) // 10**9
            
            # Create datetime string column for readability
            df['datetime'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # Drop the original date column
            df = df.drop('date', axis=1)
            
            # Save directly to parquet
            df.to_parquet(
                output_path,
                index=False,
                compression='snappy'
            )
            
            print(f"Processed {pair} and saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {pair}: {str(e)}")



def aggregate_parquet_data(folder_path: str,
                          ticker: Optional[Ticker] = None,
                          all_tickers: bool = False) -> None:
    """
    Aggregate daily parquet data into weekly and monthly timeframes.
    Simplified version that only handles daily -> weekly/monthly aggregation.

    Parameters:
        folder_path: Path to the folder containing the parquet files
        ticker: Specific ticker to process if all_tickers is False
        all_tickers: Whether to process all tickers or just one

    Returns:
        None
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    # Only process specified tickers
    tickers = [t.name for t in Ticker] if all_tickers else [ticker.name]
    
    # Daily, weekly, and monthly timeframes
    granularities = [TimeFrame.D, TimeFrame.W, TimeFrame.M]  # Store daily data and aggregate to weekly and monthly

    for market in tickers:
        input_path = os.path.join(folder_path, f'{market}.parquet')
        
        if not os.path.exists(input_path):
            print(f"Warning: Input file not found: {input_path}")
            continue

        print(f"\nProcessing {market}")
        
        # Read the daily data
        try:
            df = pd.read_parquet(input_path)
            
            # Convert timestamp to datetime if it's not already
            if 'datetime' not in df.columns:
                if 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    print(f"Error: No timestamp or datetime column found in {market} data")
                    continue
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Set datetime as index for resampling
            df = df.set_index('datetime')
            
            # Process each timeframe (weekly and monthly)
            for timeframe in granularities:
                output_path = os.path.join(
                    project_root,
                    'data',
                    'ohlc_data',
                    market,
                    f'{timeframe.name}_{market}.parquet'
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Resample to the target timeframe
                # For weekly data: 'W' means week ending on Sunday
                # For monthly data: 'M' means month ending on the last day
                aggregated = df.resample(timeframe.value).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum' if 'volume' in df.columns else None
                }).dropna()
                
                # Reset index to convert datetime back to a column
                aggregated = aggregated.reset_index()
                
                # Create timestamp column (seconds since epoch)
                aggregated['timestamp'] = aggregated['datetime'].astype(int) // 10**9
                
                # Format datetime as string
                aggregated['datetime'] = aggregated['datetime'].dt.strftime('%Y-%m-%d')
                
                # Save to parquet
                aggregated.to_parquet(
                    output_path,
                    index=False,
                    compression='snappy'
                )
                
                print(f"Processed {market}: {timeframe.name}")
                
        except Exception as e:
            print(f"Error processing {market}: {str(e)}")



def main():
    # Example usage of the simplified functions
    # Convert text files to parquet
    #txt_to_parquet(folder_path='/home/raman/school/CMPT_459/project/cmpt-459-project/data/daily_data', all_tickers=True)
    
    # Aggregate daily data to weekly and monthly
    aggregate_parquet_data(folder_path='/home/raman/school/CMPT_459/project/cmpt-459-project/data/parquet_data', ticker=None, all_tickers=True)


if __name__ == "__main__":
    main()
