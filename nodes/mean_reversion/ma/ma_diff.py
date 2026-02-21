from nodes.momentum.ma import ma_diff as _canonical_module
import sys as _sys

_sys.modules[__name__] = _canonical_module
