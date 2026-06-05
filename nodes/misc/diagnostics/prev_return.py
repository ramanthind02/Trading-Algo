from nodes import BiasNode
from lib.core.models import Candle
from lib.core.enums import Bias, Ticker, TimeFrame
from typing import ClassVar, List
import numpy as np

try:
    from lib.compute.fast_nodes import CYTHON_NODES_AVAILABLE, compute_return_fast
except ImportError:
    CYTHON_NODES_AVAILABLE = False
    compute_return_fast = None  # type: ignore[assignment]


class ReturnNode(BiasNode):
    """
    ReturnNode is a bias node that outputs the return of the candle.
    This serves as a canary value to identify lookahead bias.

    The return is calculated as the percentage change between the close price
    and the open price of the candle. Uses Cython compute_return_fast when
    available for consistency with other nodes.
    """

    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = ()
    
    def __init__(self, ticker: Ticker, tf: TimeFrame):
        """
        Initializes the ReturnNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        
        Returns: None
        """
        super().__init__(ticker, tf)
        # Standardized naming metadata
        self.module_name = 'prev_return'
        self.output_features = ['return', 'log_return', 'sign', 'is_bullish']
        self.params = {}
        # Define the columns attribute required by the MLManager
        self.columns = ['return', "log_return", "sign", "is_bullish"]
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the return of the candle as a percentage change.

        Parameters:
        - candle (Candle): The candle to compute the return for

        Returns:
        - List: [candle_return, log_return, sign, is_bullish]
        """
        if CYTHON_NODES_AVAILABLE and compute_return_fast is not None:
            candle_return, log_return = compute_return_fast(candle.open, candle.close)
        else:
            candle_return = ((candle.close - candle.open) / candle.open) * 100
            log_return = np.log(candle.close / candle.open) if candle.open > 0 else 0.0

        if candle_return > 0:
            self.bias = Bias.BULLISH
        elif candle_return < 0:
            self.bias = Bias.BEARISH
        else:
            self.bias = Bias.NEUTRAL

        is_bullish = candle_return > 0
        self.output = [candle_return, log_return, np.sign(candle_return), is_bullish]
        return self.output
    
