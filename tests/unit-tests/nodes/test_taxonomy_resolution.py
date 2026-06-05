import pytest

from nodes._taxonomy import CANONICAL_MODULE_CLASSES
from nodes.mean_reversion.rsi.rsi import RSI
from nodes.momentum.core.momentum import Momentum
from lib.core.enums import Ticker, TimeFrame
from lib.core.helpers import (
    _normalize_module_base_name,
    _resolve_bias_node_import_path,
    create_bias_node,
    create_fresh_bias_node,
)


def test_taxonomy_registry_declares_expected_class_names() -> None:
    assert CANONICAL_MODULE_CLASSES["atr_percentile_filter"] == "AtrPercentileFilterNode"
    assert CANONICAL_MODULE_CLASSES["rsi"] == "RSI"
    assert CANONICAL_MODULE_CLASSES["rsi_signal"] == "RSISignal"
    assert CANONICAL_MODULE_CLASSES["rsi_signal_atr_slope_entry"] == "RSISignalAtrSlopeEntry"
    assert CANONICAL_MODULE_CLASSES["cyclical_rsi_signal"] == "CyclicalRSISignal"
    assert CANONICAL_MODULE_CLASSES["cumulative_rsi_signal"] == "CumulativeRSISignal"
    assert CANONICAL_MODULE_CLASSES["casey_percent_c_signal"] == "CaseyPercentCSignal"
    assert CANONICAL_MODULE_CLASSES["percent_b_signal"] == "PercentBSignal"
    assert CANONICAL_MODULE_CLASSES["regime_lrsi_signal"] == "RegimeLrsiSignal"
    assert CANONICAL_MODULE_CLASSES["envelope_reversion_signal"] == "EnvelopeReversionSignal"
    assert CANONICAL_MODULE_CLASSES["williamsr_signal"] == "WilliamsRSignal"
    assert CANONICAL_MODULE_CLASSES["williamsrsignal"] == "WilliamsRSignal"
    assert CANONICAL_MODULE_CLASSES["turnaroundtuesday"] == "TurnaroundTuesday"


def test_taxonomy_mapping_resolves_rsi_to_canonical_import() -> None:
    assert _resolve_bias_node_import_path("rsi") == "nodes.mean_reversion.rsi.rsi"


def test_unregistered_module_raises_registry_error() -> None:
    with pytest.raises(ValueError, match="not registered"):
        _resolve_bias_node_import_path("temporary_node")


def test_create_bias_node_instantiates_flat_name_modules() -> None:
    rsi_node = create_bias_node("rsi", Ticker.ES, TimeFrame.D, {"lookback": 14})
    momentum_node = create_bias_node("momentum", Ticker.ES, TimeFrame.D, {"lookback": 10})

    assert isinstance(rsi_node, RSI)
    assert isinstance(momentum_node, Momentum)
    assert _normalize_module_base_name("momentum_10_D") == "momentum"


def test_create_fresh_bias_node_bypasses_singleton_reuse() -> None:
    shared = create_bias_node("rsi", Ticker.ES, TimeFrame.D, {"lookback": 14})
    fresh = create_fresh_bias_node("rsi", Ticker.ES, TimeFrame.D, {"lookback": 14})

    assert isinstance(fresh, RSI)
    assert fresh is not shared


def test_package_name_collisions_keep_legacy_imports() -> None:
    from nodes.buy_hold import BuyHold
    from nodes.momentum import Momentum as ImportedMomentum

    assert BuyHold.__name__ == "BuyHold"
    assert ImportedMomentum.__name__ == "Momentum"
