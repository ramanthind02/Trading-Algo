from nodes.mean_reversion.rsi import detrended_rsi as _canonical_module
import sys as _sys

_sys.modules[__name__] = _canonical_module
