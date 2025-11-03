
import utils.helpers as helpers
from utils.models import Candle
from utils.logger import get_logger
from typing import List, Dict, Tuple
from utils.enums import TimeFrame, Bias, Ticker
import pandas as pd
import numpy as np


logger = get_logger(__name__)


class MLManager:

    DEFAULT_LOOKBACK = 60

    def __init__(
        self,
        bias_strategies: Dict[Tuple[str, TimeFrame], Tuple[List[TimeFrame], Dict]],
        ticker: Ticker,
        base_tf: TimeFrame,
        build_matrix: bool = False, 
    ):
        """
        ML Manager controls the bias nodes for a specific model

        Parameters:
        - bias_strategies (Dict): Defines all the bias nodes for the ML manager
        - ticker (Ticker): The ticker of data being fed
        - base_tf (TimeFrame): Base timeframe for the ML model
        - build_matrix (bool): Whether or not to construct a matrix for training
        
        Returns: None
        """
        self.bias_strategies = bias_strategies
        self.ticker = ticker
        self.base_tf = base_tf
        self.build_matrix = build_matrix
        self.bias_nodes = []
        self.bias_values = []
        # Main matrix for all data - now a pandas DataFrame
        self.matrix = None  # Will be initialized in prepare_bias_nodes
        # Buffer for efficient DataFrame operations
        self.matrix_buffer = []
        self.buffer_size = 100  # Flush buffer when it reaches this size
        # Matrix organized by timeframe, with only relevant columns
        self.tf_matrix = {}
        # Track which columns belong to each timeframe
        self.tf_columns = {}
        # Track column indices for each timeframe
        self.tf_indices = {}
        self.vector_matrix = {}
        self.bias = Bias.NEUTRAL.value
        self.prepare_bias_nodes()

    def add_candle(self, candle: Candle, tf: TimeFrame) -> float:
        """
        Adds candle to all bias nodes and predicts bias with pre-trained ML model

        Parameters:
        - candle (Candle)
        - tf (TimeFrame)
        - is_historical (bool)
        - stacked_inputs (List[Tuple[str, float]])

        Returns:
        - float: Ranges between 0 and 1 -> 0 being bearish, 1 being bullish
        """

        if self.build_matrix and tf == self.base_tf:
            # Add to buffer instead of immediately appending to DataFrame
            self.matrix_buffer.append((candle.datetime, self.bias_values.copy()))
            
            # Flush buffer when it reaches the threshold size
            if len(self.matrix_buffer) >= self.buffer_size:
                self._flush_matrix_buffer()
        
        
        column_index = 0
        # Update all bias nodes and bias_values array
        for idx in range(len(self.bias_nodes)):
            timeframe, bias_node = self.bias_nodes[idx]
            num_columns = len(bias_node.columns)  # Get number of columns for this node

            if tf != timeframe:
                column_index += num_columns  # Skip the indices for this node
                continue

            vals = bias_node.add_candle(candle)

            for j in range(len(vals)):
                if isinstance(vals[j], Bias):
                    val = vals[j].value
                else:
                    val = vals[j]

                self.bias_values[column_index] = val
                column_index += 1
       
        return self.bias
    



    def _flush_matrix_buffer(self) -> None:
        """
        Flushes the matrix buffer to the DataFrame for efficient batch processing
        """
        if not self.matrix_buffer:
            return
            
        # Create a DataFrame from the buffer
        buffer_df = pd.DataFrame(
            [values for _, values in self.matrix_buffer],
            columns=self.columns,
            index=[dt for dt, _ in self.matrix_buffer]
        )
        
        # Append to the main DataFrame
        self.matrix = pd.concat([self.matrix, buffer_df])
        
        # Clear the buffer
        self.matrix_buffer = []
        
    @property
    def matrix_df(self):
        """
        Property that ensures buffer is flushed before returning the matrix DataFrame
        
        Returns:
            pd.DataFrame: The complete matrix DataFrame with all buffered data
        """
        # Ensure all buffered data is in the DataFrame
        self._flush_matrix_buffer()
        
        # Compute time series features if there are any TimeSeriesFeatureNode instances
        from nodes.ts_feature import TimeSeriesFeatureNode
        has_ts_nodes = any(isinstance(bias_node, TimeSeriesFeatureNode) 
                           for _, bias_node in self.bias_nodes)
        
        if has_ts_nodes:
            # Post-process to compute TS features
            updated_matrix = helpers.compute_ts_features_from_ml_manager(self)
            self.matrix = updated_matrix
        
        return self.matrix
        
    def prepare_bias_nodes(self) -> None:
        """
        Sets up the bias_nodes and bias_values arrays and lookup dictionary

        Parameters: None

        Returns: None
        """
        self.columns = []
        column_index = 0

        for bias_strategy, (tfs, params) in sorted(self.bias_strategies.items(), key=lambda x: str(x[0])):
            for tf in tfs:
                # Make sure the timeframe tracking structures are initialized
                if tf not in self.tf_columns:
                    self.tf_columns[tf] = []
                    self.tf_indices[tf] = []
                    self.tf_matrix[tf] = []
                    self.vector_matrix[tf] = []
                
                # Get the actual module name from params if stored there, otherwise use strategy key
                # This is important for ts_feature which needs the correct module name
                # Make a copy of params to avoid mutating the original
                params_copy = params.copy() if isinstance(params, dict) else {}
                module_name = params_copy.pop('_module_name', bias_strategy) if isinstance(params_copy, dict) else bias_strategy
                
                bias_node = helpers.create_bias_node(module_name, self.ticker, tf, params_copy)
                self.bias_nodes.append((tf, bias_node))

                # Prefer standardized names from node; fall back to legacy columns
                try:
                    column_names = bias_node.get_column_names() if hasattr(bias_node, 'get_column_names') else list(getattr(bias_node, 'columns', []))
                    if not column_names:
                        column_names = list(getattr(bias_node, 'columns', []))
                except Exception:
                    column_names = list(getattr(bias_node, 'columns', []))

                for column_name in column_names:
                    self.columns.append(column_name)
                    self.tf_columns[tf].append(column_name)
                    self.tf_indices[tf].append(column_index)
                    column_index += 1
                    self.bias_values.append(Bias.NEUTRAL.value)
        
        # Initialize an empty DataFrame with the correct columns
        self.matrix = pd.DataFrame(columns=self.columns)
        self.matrix_buffer = []  # Initialize buffer
        
        logger.info(f"MLManager: Prepared {len(self.bias_nodes)} bias nodes with {len(self.columns)} total columns")
        logger.info(f"MLManager: Timeframes configured: {list(self.tf_columns.keys())}")
