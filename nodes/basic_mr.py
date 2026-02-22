from nodes.mean_reversion.price_action import basic_mr as _canonical_module
import sys as _sys

_sys.modules[__name__] = _canonical_module
