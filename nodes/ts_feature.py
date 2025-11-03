from typing import List, Callable, Any, Optional, Dict
from collections import deque
import polars as pl
import numpy as np
from datetime import datetime
from nodes import BiasNode
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
import utils.helpers as _helpers


class TimeSeriesFeatureNode(BiasNode):
    """
    A dynamic BiasNode wrapper that applies time series feature extraction functions
    to the outputs of another BiasNode.
    
    This node:
    - Wraps another BiasNode and collects its outputs in a rolling window
    - Applies a specified time series transformation function from utils.functime to the window
    - Computes features lazily using Polars when features are extracted
    
    Parameters:
    - wrapped_node: Another BiasNode instance to wrap
    - lookback: Rolling window size for time series features
    - transformation: functime feature extraction function (from utils.functime or passed as function object)
    - transformation_name: String name for the transformation (for column naming)
    - transformation_args: Optional arguments for the transformation function
    
    Example functime functions available from utils.functime:
    - mean_abs_change: Compute mean absolute change
    - mean_change: Compute mean change
    - autocorrelation: Calculate autocorrelation at specified lag (requires n_lags arg)
    - number_crossings: Count crossings of a threshold value
    - linear_trend: Compute slope, intercept, RSS (returns dict; extracts slope by default)
    - number_peaks: Count peaks with specified support (requires support arg)
    - longest_streak_above_mean: Length of longest streak above mean
    - And many more - see utils/functime.py for full list
    
    Usage example:
        from utils.functime import mean_abs_change
        
        ts_node = TimeSeriesFeatureNode(
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            wrapped_node=rsi_node,
            lookback=60,
            transformation=mean_abs_change,
            transformation_name='meanAbsChange'
        )
        
    Or using string name (will lookup from utils.functime):
        # In bias_node_specs:
        {
            'module_name': 'ts_feature',
            'params': {
                'wrapped_module': 'rsi',
                'wrapped_params': {'lookback': 14},
                'lookback': 60,
                'transformation': 'mean_abs_change',
                'transformation_name': 'meanAbsChange'
            }
        }
    """
    
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        wrapped_node: BiasNode,
        lookback: int,
        transformation: Callable,
        transformation_name: str,
        transformation_args: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TimeSeriesFeatureNode
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - wrapped_node: The BiasNode to wrap
        - lookback: Rolling window size
        - transformation: functime feature extraction function
        - transformation_name: Name for column naming
        - transformation_args: Optional args for transformation function
        """
        super().__init__(ticker, tf)
        
        self.wrapped_node = wrapped_node
        self.lookback = lookback
        self.transformation = transformation
        self.transformation_name = transformation_name
        self.transformation_args = transformation_args or {}
        
        # Get base feature names from wrapped node
        try:
            base_column_names = wrapped_node.get_column_names()
        except Exception:
            base_column_names = getattr(wrapped_node, 'columns', [])
        
        if not base_column_names:
            # Fallback: use wrapped node name + generic feature names
            wrapped_name = wrapped_node.__class__.__name__.lower()
            num_outputs = len(wrapped_node.output) if hasattr(wrapped_node, 'output') and wrapped_node.output else 1
            base_column_names = [f"{wrapped_name}_output_{i}" for i in range(num_outputs)]
        
        # Maintain rolling windows for each output feature from wrapped node
        self.windows = {name: deque(maxlen=lookback) for name in base_column_names}
        self.datetimes = deque(maxlen=lookback)  # Track datetimes for alignment
        
        # Store outputs for later computation (lazy evaluation)
        self.stored_outputs = {name: [] for name in base_column_names}
        self.stored_datetimes = []
        
        # Number of candles needed before valid output
        self.front_bad = lookback
        
        # Standardized naming metadata
        self.module_name = 'tsFeature'
        self.output_features = []
        
        # Build output feature names: append _{transformationName}_{lookback} to original
        for base_name in base_column_names:
            new_name = f"{base_name}_{transformation_name}_{lookback}"
            self.output_features.append(new_name)
        
        # Store base column names for reference
        self.base_column_names = base_column_names
        
        self.params = {
            'wrapped_module': getattr(wrapped_node, 'module_name', wrapped_node.__class__.__name__),
            'lookback': lookback,
            'transformation': transformation_name,
            **self.transformation_args
        }
        
        # Define standardized columns
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute time series features for the given candle.
        
        During backtesting, this stores the wrapped node outputs in windows.
        Actual feature computation happens lazily at extraction time.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing placeholder values (will be computed later)
        """
        # Get outputs from wrapped node
        wrapped_outputs = self.wrapped_node.add_candle(candle)
        
        # Ensure we have the right number of outputs
        base_column_names = list(self.windows.keys())
        if len(wrapped_outputs) != len(base_column_names):
            # Adjust if mismatch
            if len(wrapped_outputs) > len(base_column_names):
                base_column_names = [f"output_{i}" for i in range(len(wrapped_outputs))]
                # Initialize windows if needed
                for name in base_column_names:
                    if name not in self.windows:
                        self.windows[name] = deque(maxlen=self.lookback)
                        self.stored_outputs[name] = []
            
            if len(wrapped_outputs) < len(base_column_names):
                wrapped_outputs = wrapped_outputs + [0.0] * (len(base_column_names) - len(wrapped_outputs))
        
        # Store outputs in windows and for later computation
        for i, (name, value) in enumerate(zip(base_column_names, wrapped_outputs)):
            if name in self.windows:
                self.windows[name].append(value)
                self.stored_outputs[name].append(value)
        
        self.datetimes.append(candle.datetime)
        self.stored_datetimes.append(candle.datetime)
        
        # Return placeholder values (NaN until window is full)
        # The actual computation will happen lazily at extraction time
        if len(self.datetimes) < self.front_bad:
            # Return NaN placeholders for initial period
            return [np.nan] * len(base_column_names)
        else:
            # Return wrapped node outputs directly for now
            # They will be replaced with computed features during extraction
            return list(wrapped_outputs)
    
    def compute_features_from_stored_data(self) -> Dict[str, List[float]]:
        """
        Compute time series features from stored outputs using Polars.
        
        This method should be called at feature extraction time (after backtest completes)
        to compute the actual time series features lazily.
        
        Returns:
        - Dict mapping feature names to computed feature values
        """
        if len(self.stored_datetimes) < self.lookback:
            # Not enough data
            return {name: [np.nan] * len(self.stored_datetimes) for name in self.windows.keys()}
        
        # Convert stored data to Polars DataFrame for efficient computation
        data_dict = {
            'datetime': self.stored_datetimes
        }
        
        # Add each output series
        for name in self.windows.keys():
            if name in self.stored_outputs:
                data_dict[name] = self.stored_outputs[name]
            else:
                data_dict[name] = [np.nan] * len(self.stored_datetimes)
        
        df = pl.DataFrame(data_dict).sort("datetime")
        
        # Compute features using rolling windows
        computed_features = {}
        base_column_names = list(self.windows.keys())
        
        for name in base_column_names:
            # Use Polars rolling window with custom aggregation
            # We'll compute the transformation for each rolling window
            feature_values = []
            
            # For each row, get the rolling window and compute transformation
            for i in range(len(df)):
                if i < self.lookback - 1:
                    # Not enough data for window
                    feature_values.append(np.nan)
                else:
                    # Get the rolling window
                    window_start = max(0, i - self.lookback + 1)
                    window_data = df.slice(window_start, self.lookback)
                    window_series = window_data[name]
                    
                    # Compute transformation
                    feature_value = self._compute_transformation(window_series)
                    feature_values.append(feature_value)
            
            computed_features[name] = feature_values
        
        # Map to output feature names
        output_features = {}
        for i, base_name in enumerate(base_column_names):
            output_name = self.output_features[i] if i < len(self.output_features) else f"{base_name}_{self.transformation_name}_{self.lookback}"
            if base_name in computed_features:
                output_features[output_name] = computed_features[base_name]
            else:
                output_features[output_name] = [np.nan] * len(self.stored_datetimes)
        
        return output_features
    
    def _compute_transformation(self, series: pl.Series) -> float:
        """
        Compute transformation on a Polars Series.
        
        Parameters:
        - series: Polars Series (window of data)
        
        Returns:
        - Computed feature value (for dict/struct results, returns a scalar summary)
        """
        try:
            # Handle NaN values
            if series.null_count() == len(series):
                return np.nan
            
            # Ensure series is not empty
            if len(series) == 0:
                return np.nan
            
            # Call transformation function
            # Most functime functions accept pl.Series from utils.functime
            if self.transformation_args:
                result = self.transformation(series, **self.transformation_args)
            else:
                result = self.transformation(series)
            
            # Handle different return types from functime functions
            if result is None:
                return np.nan
            elif isinstance(result, (int, float, np.integer, np.floating)):
                return float(result)
            elif isinstance(result, dict):
                # For functions like linear_trend that return dicts
                # Extract a summary value (e.g., slope, or first numeric value)
                # Users should specify which key to use via transformation_args if needed
                if 'slope' in result:
                    return float(result['slope'])
                elif 'rss' in result:
                    return float(result['rss'])
                elif 'intercept' in result:
                    return float(result['intercept'])
                else:
                    # Get first numeric value
                    for v in result.values():
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            return float(v)
                    return np.nan
            elif hasattr(result, 'struct') and hasattr(result.struct, 'field'):
                # Polars struct expression (shouldn't happen with Series input, but handle it)
                # Try to extract first field
                try:
                    field_names = result.struct.field_names() if hasattr(result.struct, 'field_names') else []
                    if field_names:
                        first_field = result.struct.field(field_names[0])
                        # If it's still an expression, we can't evaluate it here
                        # This case shouldn't occur with Series input
                        return np.nan
                    return np.nan
                except:
                    return np.nan
            elif hasattr(result, 'item'):
                # numpy scalar
                return float(result.item())
            elif hasattr(result, '__float__'):
                return float(result)
            elif isinstance(result, (list, tuple)):
                # For functions that return lists (e.g., energy_ratios)
                # Return first value or mean if numeric
                if result:
                    first = result[0]
                    if isinstance(first, (int, float, np.integer, np.floating)):
                        return float(first)
                    else:
                        return np.nan
                return np.nan
            else:
                # Unknown type - try to convert
                try:
                    return float(result)
                except (ValueError, TypeError):
                    return np.nan
                
        except Exception as e:
            # Log error for debugging but return NaN to continue processing
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Error computing transformation {self.transformation_name}: {e}")
            return np.nan

