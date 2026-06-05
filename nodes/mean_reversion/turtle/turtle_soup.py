"""
Turtle Soup bias node — inverse of Turtle Trading.

Reuses TurtleTrading and negates its positions so that:
- Turtle long → Turtle Soup short (-1)
- Turtle short → Turtle Soup long (1)
- Turtle flat → Turtle Soup flat (0)

Same parameters as Turtle (entry_lookback, stop_lookback, direction); no duplicated logic.
"""

from typing import ClassVar, List

from nodes import BiasNode, LookbackContribution
from nodes.turtle import TurtleTrading
from lib.core.enums import TimeFrame, Ticker
from lib.core.models import Candle


class TurtleSoup(BiasNode):
    """
    Turtle Soup: rule-based bias node that outputs the opposite of Turtle Trading.

    Outputs signed position: 1 (long), -1 (short), 0 (flat) = -(Turtle output).
    Uses the same entry/stop lookbacks and direction as Turtle; only the sign is flipped.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"entry_lookback", "stop_lookback"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        entry_lookback: int,
        stop_lookback: int,
        direction: str = "long",
    ) -> None:
        super().__init__(ticker, tf)

        self._turtle = TurtleTrading(
            ticker=ticker,
            tf=tf,
            entry_lookback=entry_lookback,
            stop_lookback=stop_lookback,
            direction=direction,
        )

        self.entry_lookback = entry_lookback
        self.stop_lookback = stop_lookback
        self.direction = direction

        self.module_name = "turtle_soup"
        self.output_features = ["signal"]
        self.params = {
            "entry_lookback": entry_lookback,
            "stop_lookback": stop_lookback,
            "direction": direction,
        }
        self.front_bad = self._turtle.front_bad

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        turtle_result = self._turtle.add_candle(candle)
        negated = [-x for x in turtle_result]
        signal = negated[0]
        self.output.append(signal)
        return negated

    def _extra_lookback_contributions(self) -> tuple[LookbackContribution, ...]:
        """Expose the wrapped Turtle warmup for cold cache rebuilds."""
        return self._turtle.lookback_contributions()
