from nodes.momentum.core import simple_momentum as _canonical_module
import sys as _sys

_sys.modules[__name__] = _canonical_module
