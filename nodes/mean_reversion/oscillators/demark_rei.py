from nodes.momentum.oscillators import demark_rei as _canonical_module
import sys as _sys

_canonical_module.DemarkREI.lookback_param_names = frozenset({"period"})

_sys.modules[__name__] = _canonical_module
