from nodes.mean_reversion.turtle import turtle_soup as _canonical_module
import sys as _sys

_canonical_module.TurtleSoup.lookback_param_names = frozenset({"entry_lookback", "stop_lookback"})
_sys.modules[__name__] = _canonical_module
