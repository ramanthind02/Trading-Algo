from typing import ClassVar, List
from lib.core.models import Candle
from lib.core.enums import Ticker, TimeFrame
from nodes import BiasNode, LookbackWindow
from nodes.momentum import Momentum


class ConsecMomentum(BiasNode):
    """
    Consecutive Momentum Bias Node
    
    Wraps the Momentum node and tracks consecutive momentum changes to generate
    binary signals (0 or 1).
    
    When is_buy=True:
    - Outputs 1 when momentum has been increasing for x consecutive bars
    - Stays at 1 until momentum decreases for x consecutive bars, then returns to 0
    
    When is_buy=False:
    - Outputs 1 when momentum has been decreasing for x consecutive bars
    - Stays at 1 until momentum increases for x consecutive bars, then returns to 0
    
    Parameters:
    - lookback: Lookback period for underlying Momentum node (default: 10)
    - consecutive_bars: Number of consecutive bars needed (default: 3)
    - is_buy: Direction flag (default: True)
    """
    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback", "consecutive_bars"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 10,
        consecutive_bars: int = 3,
        is_buy: bool = True
    ):
        """
        Initialize ConsecMomentum node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: Lookback period for underlying Momentum node (default: 10)
        - consecutive_bars: Number of consecutive bars needed (default: 3)
        - is_buy: Direction flag (default: True)
        """
        super().__init__(ticker, tf)
        
        # Create wrapped Momentum node
        self.momentum_node = Momentum(ticker=ticker, tf=tf, lookback=lookback)
        
        self.consecutive_bars = consecutive_bars
        self.is_buy = is_buy
        
        # State tracking
        self.current_state = 0  # 0 = no signal, 1 = signal active
        self.consecutive_count = 0  # Track consecutive increases/decreases
        self.last_momentum = None  # Previous momentum value
        
        # Standardized naming metadata
        self.module_name = 'consecMomentum'
        self.output_features = ['signal']
        self.params = {
            'lookback': lookback,
            'consecutive_bars': consecutive_bars,
            'is_buy': is_buy
        }
        
        # Warm-up period: momentum node needs momentum_lookback bars,
        # then we need at least consecutive_bars to detect pattern
        # Add 1 more for comparison (need at least 2 momentum values to compare)
        self.front_bad = lookback + consecutive_bars
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _extra_lookback_contributions(self) -> tuple[LookbackWindow, ...]:
        return (LookbackWindow(label="stateful_warmup", bars=self.front_bad),)

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute consecutive momentum signal for the given candle.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the binary signal (0 or 1)
        """
        # Get momentum value from wrapped node
        momentum_result = self.momentum_node.add_candle(candle)
        if not momentum_result or len(momentum_result) == 0:
            self.output.append(0.0)
            return [0.0]
        
        current_momentum = momentum_result[0]
        
        # Return 0 during warm-up period
        if self.momentum_node.n_prices <= self.momentum_node.front_bad:
            self.last_momentum = current_momentum
            self.output.append(0.0)
            return [0.0]
        
        # Get previous momentum for comparison
        if self.last_momentum is None:
            # First valid momentum value, can't compare yet
            self.last_momentum = current_momentum
            self.output.append(float(self.current_state))
            return [float(self.current_state)]
        
        prev_momentum = self.last_momentum
        
        # Determine if momentum increased, decreased, or stayed the same
        if current_momentum > prev_momentum:
            change = 1  # Increased
        elif current_momentum < prev_momentum:
            change = -1  # Decreased
        else:
            change = 0  # No change
        
        # Update state based on is_buy flag
        if self.is_buy:
            # Buy logic: wait for x consecutive increases → 1, stay 1 until x consecutive decreases → 0
            if change == 1:  # Momentum increased
                if self.current_state == 0:
                    # Not in signal state, check if we have enough consecutive increases
                    self.consecutive_count += 1
                    if self.consecutive_count >= self.consecutive_bars:
                        self.current_state = 1
                        self.consecutive_count = 0  # Reset counter
                else:
                    # Already in signal state, reset counter (we're still increasing, so signal stays on)
                    self.consecutive_count = 0
            elif change == -1:  # Momentum decreased
                if self.current_state == 1:
                    # In signal state, check if we have enough consecutive decreases
                    self.consecutive_count += 1
                    if self.consecutive_count >= self.consecutive_bars:
                        self.current_state = 0
                        self.consecutive_count = 0  # Reset counter
                else:
                    # Not in signal state, reset counter (we're not in an uptrend anymore)
                    self.consecutive_count = 0
            else:  # change == 0, no change
                # No change breaks the consecutive pattern, reset counter
                self.consecutive_count = 0
        else:
            # Sell logic (is_buy=False): wait for x consecutive decreases → 1, stay 1 until x consecutive increases → 0
            if change == -1:  # Momentum decreased
                if self.current_state == 0:
                    # Not in signal state, check if we have enough consecutive decreases
                    self.consecutive_count += 1
                    if self.consecutive_count >= self.consecutive_bars:
                        self.current_state = 1
                        self.consecutive_count = 0  # Reset counter
                else:
                    # Already in signal state, reset counter (we're still decreasing, so signal stays on)
                    self.consecutive_count = 0
            elif change == 1:  # Momentum increased
                if self.current_state == 1:
                    # In signal state, check if we have enough consecutive increases
                    self.consecutive_count += 1
                    if self.consecutive_count >= self.consecutive_bars:
                        self.current_state = 0
                        self.consecutive_count = 0  # Reset counter
                else:
                    # Not in signal state, reset counter (we're not in a downtrend anymore)
                    self.consecutive_count = 0
            else:  # change == 0, no change
                # No change breaks the consecutive pattern, reset counter
                self.consecutive_count = 0
        
        # Update last momentum for next iteration
        self.last_momentum = current_momentum
        
        self.output.append(float(self.current_state))
        return [float(self.current_state)]
