from typing import ClassVar, List, Tuple, Optional
from lib.core.enums import TimeFrame, Ticker
from nodes import BiasNode
from lib.core.models import Candle
import numpy as np

# Try to import Cython optimized version
try:
    from lib.compute.fast_nodes import compute_high_low_channel_fast, CYTHON_NODES_AVAILABLE
except ImportError:
    CYTHON_NODES_AVAILABLE = False


class DonchianChannel(BiasNode):
    """
    Implements the Donchian Channel strategy as a bias node.
    
    The Donchian Channel strategy is always in the market and uses a single channel:
    - Long when the bar **high** breaks above the prior window's highest high
    - Short when the bar **low** breaks below the prior window's lowest low
    
    Unlike the Turtle Trading strategy, this implementation:
    1. Is always in the market (no flat state)
    2. Uses a single lookback period for both entry and exit
    3. Does not filter trades based on previous results
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback"})
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int):
        """
        Initializes the DonchianChannel bias node.

        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - lookback (int): Number of periods to look back for channel calculation.

        Returns: None
        """
        super().__init__(ticker, tf)

        if not (lookback > 0):
            raise ValueError("Lookback period must be a positive integer.")

        self.lookback = lookback
        
        # Standardized naming metadata
        self.module_name = 'donchian'
        self.output_features = ['signal']
        self.params = {'lookback': lookback}
        
        # Number of candles needed before we can compute valid output
        # Need lookback + 1 candles (lookback for calculation + current candle)
        self.front_bad = lookback + 1
        
        if CYTHON_NODES_AVAILABLE:
            # Cython path: use numpy arrays for 5-10x speedup
            # Buffer size is lookback+1 to match the semantics of recent_candles
            self.highs = np.zeros(self.front_bad, dtype=np.float64)
            self.lows = np.zeros(self.front_bad, dtype=np.float64)
            self.buffer_idx = 0
            self.n_filled = 0
        else:
            # Fallback: use deque
            import collections
            self.candles_history = collections.deque(maxlen=self.front_bad)
        
        # For tracking current position
        self.current_position = 1  # Start with long position (1=long, -1=short)
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def calculate_donchian_channel(self, lookback: int) -> Tuple[float, float]:
        """
        Calculates the Donchian channel (highest high and lowest low) over the specified lookback period.

        Parameters:
        - lookback (int): Number of periods to look back.

        Returns:
        - Tuple[float, float]: (highest_high, lowest_low) over the lookback period.
        """
        if CYTHON_NODES_AVAILABLE:
            # Fast Cython path: use compute_high_low_channel_fast
            # We need lookback elements, excluding the current one
            # After writing, buffer_idx points to the next write position
            # The current candle is at (buffer_idx - 1) % front_bad
            # We want to exclude current, so look back from (buffer_idx - 2) % front_bad
            if self.n_filled < lookback + 1:  # Need lookback+1 total (including current)
                return 0.0, 0.0
            
            # The most recent element before current (where we want to start looking back)
            # buffer_idx was just incremented, so previous is at (buffer_idx - 2) % front_bad
            prev_idx = (self.buffer_idx - 2 + self.front_bad) % self.front_bad
            
            return compute_high_low_channel_fast(
                self.highs,
                self.lows,
                prev_idx,
                lookback,
                self.n_filled
            )
        else:
            # Fallback: use Python implementation
            if len(self.candles_history) < lookback:
                return 0.0, 0.0
                
            # Get the most recent candles excluding the current one
            recent_candles = list(self.candles_history)[-(lookback+1):-1]
            
            if not recent_candles:
                return 0.0, 0.0
                
            highest_high = max(candle.high for candle in recent_candles)
            lowest_low = min(candle.low for candle in recent_candles)
            
            return highest_high, lowest_low

    def _compute_candle(self, candle: Candle) -> List[float]:
        """
        Responds to a new candle being added and computes the Donchian Channel bias.

        Parameters:
        - candle (Candle): The latest candle data.

        Returns:
        - List[float]: A list containing the computed bias value (1=long, -1=short).
        """
        if CYTHON_NODES_AVAILABLE:
            # Fast Cython path: store in numpy arrays
            self.highs[self.buffer_idx] = candle.high
            self.lows[self.buffer_idx] = candle.low
            self.buffer_idx = (self.buffer_idx + 1) % self.front_bad
            self.n_filled = min(self.n_filled + 1, self.front_bad)
        else:
            # Fallback: use deque
            self.candles_history.append(candle)

        # Return current position if not enough data
        if (CYTHON_NODES_AVAILABLE and self.n_filled < self.front_bad) or \
           (not CYTHON_NODES_AVAILABLE and len(self.candles_history) < self.front_bad):
            position = float(self.current_position)
            self.output.append(position)
            return [position]

        # Calculate Donchian channel
        highest_high, lowest_low = self.calculate_donchian_channel(self.lookback)

        bar_high = float(candle.high)
        bar_low = float(candle.low)

        # Check for position changes (intrabar breakout: high / low vs prior channel)
        if self.current_position == 1:  # Currently long
            # Switch to short if the bar trades through the channel low
            if bar_low < lowest_low:
                self.current_position = -1

        elif self.current_position == -1:  # Currently short
            # Switch to long if the bar trades through the channel high
            if bar_high > highest_high:
                self.current_position = 1
        
        # Return the numeric position
        position = float(self.current_position)
        self.output.append(position)
        return [position]


class DonchianChannelLongOnly(BiasNode):
    """
    Long-only Donchian Channel breakout with optional SMA regime filter.

    Strategy behavior:
    - Enter long when bar **high** breaks above Donchian high (``entry_lookback`` window).
    - Exit to flat when bar **low** breaks below Donchian low (``exit_lookback`` window).
    - Never enters short.
    - Optional regime filter: when ``sma_period`` is a positive integer (>= 2),
      longs are only allowed when close > SMA(sma_period). If regime is bearish,
      signal is flat. Use ``None`` or ``0`` to disable the filter.
    - Optional ``channel_lookback``: when set, uses the same lookback for both entry
      and exit channels (symmetric Donchian). Overrides ``entry_lookback`` /
      ``exit_lookback`` for grid-friendly single-axis search.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"channel_lookback", "entry_lookback", "exit_lookback", "sma_period"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        entry_lookback: int = 20,
        exit_lookback: int = 20,
        sma_period: Optional[int] = None,
        channel_lookback: Optional[int] = None,
    ) -> None:
        super().__init__(ticker, tf)

        if channel_lookback is not None:
            ch = int(channel_lookback)
            if ch < 1:
                raise ValueError("channel_lookback must be a positive integer when set.")
            entry_lookback = ch
            exit_lookback = ch

        if not (entry_lookback > 0):
            raise ValueError("entry_lookback must be a positive integer.")
        if not (exit_lookback > 0):
            raise ValueError("exit_lookback must be a positive integer.")

        raw_sma_period = sma_period
        effective_sma = None if raw_sma_period in (None, 0) else raw_sma_period
        if effective_sma is not None and effective_sma < 2:
            raise ValueError(
                "sma_period must be 0 or None (filter off), or an integer >= 2."
            )

        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback
        self.sma_period = effective_sma

        # Standardized naming metadata
        self.module_name = "donchian_long_only"
        self.output_features = ["signal"]
        self.params = {
            "entry_lookback": entry_lookback,
            "exit_lookback": exit_lookback,
            # Only include sma_period when it is an active integer filter —
            # lookback_contributions requires int-like values and must not see None.
            **({"sma_period": raw_sma_period} if isinstance(raw_sma_period, int) else {}),
            **(
                {"channel_lookback": int(channel_lookback)}
                if channel_lookback is not None
                else {}
            ),
        }

        # Need enough bars for both Donchian windows + current bar; if SMA regime is
        # enabled, warmup must satisfy that as well.
        donchian_front_bad = max(entry_lookback, exit_lookback) + 1
        self.front_bad = max(donchian_front_bad, effective_sma or 0)

        if CYTHON_NODES_AVAILABLE:
            self.highs = np.zeros(self.front_bad, dtype=np.float64)
            self.lows = np.zeros(self.front_bad, dtype=np.float64)
            self.buffer_idx = 0
            self.n_filled = 0
        else:
            import collections

            self.candles_history = collections.deque(maxlen=self.front_bad)

        # SMA history is only used when regime filter is enabled.
        if self.sma_period is not None:
            import collections

            self.close_history = collections.deque(maxlen=self.sma_period)

        self.current_position = 0  # 1=long, 0=flat

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def calculate_donchian_channel(self, lookback: int) -> Tuple[float, float]:
        if CYTHON_NODES_AVAILABLE:
            if self.n_filled < lookback + 1:
                return 0.0, 0.0

            prev_idx = (self.buffer_idx - 2 + self.front_bad) % self.front_bad
            return compute_high_low_channel_fast(
                self.highs,
                self.lows,
                prev_idx,
                lookback,
                self.n_filled,
            )

        if len(self.candles_history) < lookback:
            return 0.0, 0.0

        recent_candles = list(self.candles_history)[-(lookback + 1) : -1]
        if not recent_candles:
            return 0.0, 0.0

        highest_high = max(candle.high for candle in recent_candles)
        lowest_low = min(candle.low for candle in recent_candles)
        return highest_high, lowest_low

    def _compute_sma_regime_allowed(self, close_price: float) -> bool:
        if self.sma_period is None:
            return True
        if len(self.close_history) < self.sma_period:
            return False
        sma = sum(self.close_history) / float(self.sma_period)
        return close_price > sma

    def _compute_candle(self, candle: Candle) -> List[float]:
        close_price = float(candle.close)
        if self.sma_period is not None:
            self.close_history.append(close_price)

        if CYTHON_NODES_AVAILABLE:
            self.highs[self.buffer_idx] = candle.high
            self.lows[self.buffer_idx] = candle.low
            self.buffer_idx = (self.buffer_idx + 1) % self.front_bad
            self.n_filled = min(self.n_filled + 1, self.front_bad)
        else:
            self.candles_history.append(candle)

        enough_data = (
            (CYTHON_NODES_AVAILABLE and self.n_filled >= self.front_bad)
            or (not CYTHON_NODES_AVAILABLE and len(self.candles_history) >= self.front_bad)
        )
        if not enough_data:
            position = float(self.current_position)
            self.output.append(position)
            return [position]

        highest_high, _ = self.calculate_donchian_channel(self.entry_lookback)
        _, lowest_low = self.calculate_donchian_channel(self.exit_lookback)
        regime_allows_longs = self._compute_sma_regime_allowed(close_price)

        bar_high = float(candle.high)
        bar_low = float(candle.low)

        if self.current_position == 1 and (bar_low < lowest_low or not regime_allows_longs):
            self.current_position = 0
        elif self.current_position == 0 and bar_high > highest_high and regime_allows_longs:
            self.current_position = 1

        position = float(self.current_position)
        self.output.append(position)
        return [position]
