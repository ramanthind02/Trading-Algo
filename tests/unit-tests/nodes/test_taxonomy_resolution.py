from pathlib import Path

import pytest

from nodes.mean_reversion.rsi.rsi import RSI
from nodes.momentum.core.momentum import Momentum
from utils.enums import Ticker, TimeFrame
from utils.helpers import (
    _normalize_module_base_name,
    _resolve_bias_node_import_path,
    create_bias_node,
)


NODES_ROOT = Path(__file__).resolve().parents[3] / "nodes"


def test_taxonomy_mapping_resolves_rsi_to_canonical_import() -> None:
    assert _resolve_bias_node_import_path("rsi") == "nodes.mean_reversion.rsi.rsi"


def test_recursive_fallback_resolves_non_mapped_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_rglob(self: Path, pattern: str) -> list[Path]:
        if self == NODES_ROOT and pattern == "temporary_node.py":
            return [NODES_ROOT / "experimental" / "temporary_node.py"]
        return []

    monkeypatch.setattr(Path, "rglob", fake_rglob)
    assert _resolve_bias_node_import_path("temporary_node") == "nodes.experimental.temporary_node"


def test_recursive_fallback_ambiguity_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_rglob(self: Path, pattern: str) -> list[Path]:
        if self == NODES_ROOT and pattern == "dupe_node.py":
            return [
                NODES_ROOT / "group_a" / "dupe_node.py",
                NODES_ROOT / "group_b" / "dupe_node.py",
            ]
        return []

    monkeypatch.setattr(Path, "rglob", fake_rglob)

    with pytest.raises(ValueError, match="Ambiguous module resolution"):
        _resolve_bias_node_import_path("dupe_node")


def test_create_bias_node_instantiates_flat_name_modules() -> None:
    rsi_node = create_bias_node("rsi", Ticker.ES, TimeFrame.D, {"lookback": 14})
    momentum_node = create_bias_node("momentum", Ticker.ES, TimeFrame.D, {"lookback": 10})

    assert isinstance(rsi_node, RSI)
    assert isinstance(momentum_node, Momentum)
    assert _normalize_module_base_name("momentum_10_D") == "momentum"


def test_package_name_collisions_keep_legacy_imports() -> None:
    from nodes.buy_hold import BuyHold
    from nodes.momentum import Momentum as ImportedMomentum

    assert BuyHold.__name__ == "BuyHold"
    assert ImportedMomentum.__name__ == "Momentum"
