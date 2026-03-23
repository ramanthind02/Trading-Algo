from nodes.momentum.trend import supertrend_cross as _canonical_module
import sys as _sys

_canonical_module.SuperTrendCross.lookback_param_names = frozenset({"atrPeriod"})

_sys.modules[__name__] = _canonical_module
