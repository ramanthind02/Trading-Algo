from nodes.buy_hold.core import buy_hold as _canonical_module
import sys as _sys

_sys.modules[__name__] = _canonical_module
