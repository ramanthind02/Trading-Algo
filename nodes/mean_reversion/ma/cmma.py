from nodes.momentum.ma import cmma as _canonical_module
import sys as _sys

_canonical_module.CloseMaMinusMA.lookback_param_names = frozenset({"lookback", "atr_length"})
_sys.modules[__name__] = _canonical_module
