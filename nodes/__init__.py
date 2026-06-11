"""Bias-node public API.

The ``BiasNode`` ABC and its lookback dataclasses live in :mod:`nodes.base`; this
package root re-exports them so ``from nodes import BiasNode`` keeps working. Concrete
indicators live in the themed sub-packages (``momentum``, ``mean_reversion``, …).
"""
from nodes.base import (
    BiasNode,
    LookbackContribution,
    LookbackWindow,
)

__all__ = ["BiasNode", "LookbackContribution", "LookbackWindow"]
