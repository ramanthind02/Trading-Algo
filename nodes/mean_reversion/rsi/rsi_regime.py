from nodes.regime.rsi import rsi_regime as _canonical_module
import sys as _sys

_canonical_module.RSIRegime.lookback_param_names = frozenset({"lookback", "ma_period"})

_sys.modules[__name__] = _canonical_module
