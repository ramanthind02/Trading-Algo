from nodes.momentum.oscillators import casey_c as _canonical_module
import sys as _sys

_canonical_module.CaseyC.lookback_param_names = frozenset({"changePeriod", "rankingPeriod"})

_sys.modules[__name__] = _canonical_module
