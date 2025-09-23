from datetime import datetime
import utils.helpers as helpers
from utils.models import Candle
from utils.logger import get_logger
from order.market import MarketOrder
from typing import List, Dict, Tuple, Optional
from machine_learning.artifact.enigma import EnigmaArtifact
from utils.enums import BiasStrategy, TimeFrame, Bias, Ticker, ModelType, TradeType
from machine_learning.preprocessing.type_conversion.list_to_polars import ListToPolars


logger = get_logger(__name__)


class MLManager:

    DEFAULT_LOOKBACK = 60

    def __init__(
        self,
        bias_strategies: Dict[BiasStrategy, Tuple[List[TimeFrame], Dict]],
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
        - model_type (ModelType): Classification or Regression
        - build_matrix (bool): Whether or not to construct a matrix for training
        
        Returns: None
        """
        self.bias_strategies = bias_strategies
        self.ticker = ticker
        self.base_tf = base_tf
        self.build_matrix = build_matrix
        self.bias_nodes = []
        self.bias_values = []
        # Main matrix for all data
        self.matrix = []
        # Matrix organized by timeframe, with only relevant columns
        self.tf_matrix = {}
        # Track which columns belong to each timeframe
        self.tf_columns = {}
        # Track column indices for each timeframe
        self.tf_indices = {}
        self.vector_matrix = {}
        self.bias = Bias.NEUTRAL.value
        self.prepare_bias_nodes()

    def add_candle(self, candle: Candle, tf: TimeFrame, is_historical: bool, stacked_inputs: List[Tuple[str, float]]) -> float:
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
        if not is_historical:
            if self.build_matrix and tf == self.base_tf:
                self.matrix.append((candle.datetime, self.bias_values.copy()))
            
            if tf in self.tf_indices:
                tf_values = [self.bias_values[i] for i in self.tf_indices[tf]]
                self.tf_matrix[tf].append((candle.datetime, tf_values))
                if self.build_matrix:
                    self.vector_matrix[tf].append((candle.datetime, tf_values))
                if len(self.tf_matrix[tf]) > self.DEFAULT_LOOKBACK:
                    self.tf_matrix[tf] = self.tf_matrix[tf][1:]
                logger.debug(f"Updated tf_matrix[{tf.name}]: {len(self.tf_matrix[tf])} entries, {len(tf_values)} features")
        
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

                self.bias_values[column_index] = val if not is_historical else self.bias_values[column_index]
                column_index += 1
       
        if tf == self.base_tf and not is_historical:
            # Predict bias if model exists
            if self.ml_model is not None:
                self.predict()
                
        return self.bias
    



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
                
                bias_node = helpers.create_bias_node(bias_strategy, self.ticker, tf, params)
                self.bias_nodes.append((tf, bias_node))

                for column in bias_node.columns:
                    column_name = f"{bias_strategy.name}_{tf.name}_{column}"
                    self.columns.append(column_name)
                    self.tf_columns[tf].append(column_name)
                    self.tf_indices[tf].append(column_index)
                    column_index += 1
                    self.bias_values.append(Bias.NEUTRAL.value)
        
        logger.info(f"MLManager: Prepared {len(self.bias_nodes)} bias nodes with {len(self.columns)} total columns")
        logger.info(f"MLManager: Timeframes configured: {list(self.tf_columns.keys())}")
        
