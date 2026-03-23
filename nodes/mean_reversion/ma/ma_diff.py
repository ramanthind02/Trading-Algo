from nodes.momentum.ma import ma_diff as _canonical_module
import sys as _sys

_canonical_module.MADiffNode.lookback_param_names = frozenset({"lookback"})

_sys.modules[__name__] = _canonical_module
