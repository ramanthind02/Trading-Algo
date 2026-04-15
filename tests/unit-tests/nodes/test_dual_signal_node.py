"""Unit tests for DualSignalNode.

Covers:
- AND-agreement logic: (+1,+1) → +1, (-1,-1) → -1, any disagreement → 0
- Warmup: front_bad = max(a.front_bad, b.front_bad)
- Column naming: encodes both module names and prefixed params
- Lookback contributions: union of both children
- Taxonomy resolution via helpers.create_bias_node
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pytest

from nodes import BiasNode
from nodes.composite.dual_signal import DualSignalNode
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candle(day: int = 0) -> Candle:
    return Candle(
        datetime=datetime(2023, 1, 1) + timedelta(days=day),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


class _FixedSignalNode(BiasNode):
    """Minimal bias node returning a constant signal for test isolation."""

    def __init__(self, ticker: Ticker, tf: TimeFrame, *, signal: float, front_bad: int = 0) -> None:
        super().__init__(ticker, tf)
        self.module_name = "fixed"
        self.output_features = ["signal"]
        self.params = {}
        self.front_bad = front_bad
        self._signal = signal
        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List[float]:
        self.output.append(self._signal)
        return [self._signal]


def _make_dual(
    signal_a: float,
    signal_b: float,
    front_bad_a: int = 0,
    front_bad_b: int = 0,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> DualSignalNode:
    """Construct a DualSignalNode whose inner nodes emit fixed signals."""
    node_a = _FixedSignalNode(Ticker.ES, TimeFrame.D, signal=signal_a, front_bad=front_bad_a)
    node_b = _FixedSignalNode(Ticker.ES, TimeFrame.D, signal=signal_b, front_bad=front_bad_b)

    call_count = 0

    def _fake_create(module_name: str, ticker: Ticker, tf: TimeFrame, params: dict) -> BiasNode:
        nonlocal call_count
        result = node_a if call_count == 0 else node_b
        call_count += 1
        return result

    import nodes.composite.dual_signal as _mod
    original = _mod.helpers.create_fresh_bias_node
    _mod.helpers.create_fresh_bias_node = _fake_create  # type: ignore[method-assign]
    try:
        node = DualSignalNode(
            Ticker.ES,
            TimeFrame.D,
            moduleA="fixed",
            paramsA={},
            moduleB="fixed",
            paramsB={},
        )
    finally:
        _mod.helpers.create_fresh_bias_node = original  # type: ignore[method-assign]

    return node


# ---------------------------------------------------------------------------
# Agreement logic
# ---------------------------------------------------------------------------

class TestAgreementLogic:
    def test_both_long_emits_long(self) -> None:
        node = _make_dual(signal_a=1.0, signal_b=1.0)
        result = node.add_candle(_candle())
        assert result == [1.0]

    def test_both_short_emits_short(self) -> None:
        node = _make_dual(signal_a=-1.0, signal_b=-1.0)
        result = node.add_candle(_candle())
        assert result == [-1.0]

    def test_disagreement_emits_neutral(self) -> None:
        node = _make_dual(signal_a=1.0, signal_b=-1.0)
        result = node.add_candle(_candle())
        assert result == [0.0]

    def test_node_a_flat_emits_neutral(self) -> None:
        node = _make_dual(signal_a=0.0, signal_b=1.0)
        result = node.add_candle(_candle())
        assert result == [0.0]

    def test_node_b_flat_emits_neutral(self) -> None:
        node = _make_dual(signal_a=1.0, signal_b=0.0)
        result = node.add_candle(_candle())
        assert result == [0.0]

    def test_both_flat_emits_neutral(self) -> None:
        node = _make_dual(signal_a=0.0, signal_b=0.0)
        result = node.add_candle(_candle())
        assert result == [0.0]

    def test_multiple_candles_consistent(self) -> None:
        node = _make_dual(signal_a=1.0, signal_b=1.0)
        results = [node.add_candle(_candle(i)) for i in range(5)]
        assert all(r == [1.0] for r in results)


# ---------------------------------------------------------------------------
# Warmup / front_bad
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_front_bad_is_max_of_children(self) -> None:
        node = _make_dual(signal_a=1.0, signal_b=1.0, front_bad_a=3, front_bad_b=7)
        assert node.front_bad == 7

    def test_front_bad_zero_when_both_zero(self) -> None:
        node = _make_dual(signal_a=1.0, signal_b=1.0, front_bad_a=0, front_bad_b=0)
        assert node.front_bad == 0

    def test_front_bad_uses_larger_child(self) -> None:
        node = _make_dual(signal_a=1.0, signal_b=1.0, front_bad_a=10, front_bad_b=2)
        assert node.front_bad == 10


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------

class TestColumnNaming:
    def test_column_name_contains_dual_signal(self) -> None:
        node = _make_dual(signal_a=1.0, signal_b=1.0)
        cols = node.get_column_names()
        assert len(cols) == 1
        assert "dual_signal" in cols[0]

    def test_column_name_contains_both_module_names(self) -> None:
        """module names are encoded in self.params and appear in the column."""
        node_a = _FixedSignalNode(Ticker.ES, TimeFrame.D, signal=1.0)
        node_b = _FixedSignalNode(Ticker.ES, TimeFrame.D, signal=1.0)

        import nodes.composite.dual_signal as _mod

        call_count = 0

        def _fake_create(module_name: str, ticker: Ticker, tf: TimeFrame, params: dict) -> BiasNode:
            nonlocal call_count
            result = node_a if call_count == 0 else node_b
            call_count += 1
            return result

        original = _mod.helpers.create_fresh_bias_node
        _mod.helpers.create_fresh_bias_node = _fake_create  # type: ignore[method-assign]
        try:
            node = DualSignalNode(
                Ticker.ES,
                TimeFrame.D,
                moduleA="rsi_signal",
                paramsA={"lookback": 14},
                moduleB="ewmac",
                paramsB={"span_fast": 16},
            )
        finally:
            _mod.helpers.create_fresh_bias_node = original  # type: ignore[method-assign]

        col = node.get_column_names()[0]
        assert "rsi_signal" in col
        assert "ewmac" in col

    def test_params_prefixed_in_column_name(self) -> None:
        """paramsA keys are prefixed with a_ and paramsB with b_."""
        node = _make_dual(signal_a=1.0, signal_b=1.0)
        # Params are {"moduleA": "fixed", "moduleB": "fixed"} — check both encoded
        assert node.params["moduleA"] == "fixed"
        assert node.params["moduleB"] == "fixed"

    def test_child_params_namespaced(self) -> None:
        """paramsA keys appear as a_<key> and paramsB keys as b_<key> in self.params."""
        node_a = _FixedSignalNode(Ticker.ES, TimeFrame.D, signal=1.0)
        node_b = _FixedSignalNode(Ticker.ES, TimeFrame.D, signal=1.0)

        import nodes.composite.dual_signal as _mod

        call_count = 0

        def _fake_create(module_name: str, ticker: Ticker, tf: TimeFrame, params: dict) -> BiasNode:
            nonlocal call_count
            result = node_a if call_count == 0 else node_b
            call_count += 1
            return result

        original = _mod.helpers.create_fresh_bias_node
        _mod.helpers.create_fresh_bias_node = _fake_create  # type: ignore[method-assign]
        try:
            node = DualSignalNode(
                Ticker.ES,
                TimeFrame.D,
                moduleA="rsi_signal",
                paramsA={"lookback": 14},
                moduleB="ewmac",
                paramsB={"span_fast": 16},
            )
        finally:
            _mod.helpers.create_fresh_bias_node = original  # type: ignore[method-assign]

        assert node.params.get("a_lookback") == 14
        assert node.params.get("b_span_fast") == 16


# ---------------------------------------------------------------------------
# Lookback contributions
# ---------------------------------------------------------------------------

class TestLookbackContributions:
    def test_includes_both_children_contributions(self) -> None:
        node = _make_dual(signal_a=1.0, signal_b=1.0, front_bad_a=5, front_bad_b=10)
        contribs = node.lookback_contributions()
        # front_bad contributions from both children should be included
        bars = {c.bars for c in contribs}
        assert 5 in bars
        assert 10 in bars


# ---------------------------------------------------------------------------
# Taxonomy resolution
# ---------------------------------------------------------------------------

class TestTaxonomyResolution:
    def test_registered_in_taxonomy(self) -> None:
        from nodes._taxonomy import CANONICAL_MODULE_IMPORTS, CANONICAL_MODULE_CLASSES
        assert "dual_signal" in CANONICAL_MODULE_IMPORTS
        assert "dual_signal" in CANONICAL_MODULE_CLASSES
        assert CANONICAL_MODULE_IMPORTS["dual_signal"] == "nodes.composite.dual_signal"
        assert CANONICAL_MODULE_CLASSES["dual_signal"] == "DualSignalNode"

    def test_create_via_helpers_with_real_nodes(self) -> None:
        """End-to-end: create DualSignalNode via taxonomy using buy_hold for both arms."""
        BiasNode._instances.clear()
        node = helpers.create_fresh_bias_node(
            "dual_signal",
            Ticker.ES,
            TimeFrame.D,
            {
                "moduleA": "buy_hold",
                "paramsA": {},
                "moduleB": "buy_hold",
                "paramsB": {},
            },
        )
        assert isinstance(node, DualSignalNode)
        # buy_hold always returns 1.0 after first candle
        candles = [_candle(i) for i in range(3)]
        results = [node.add_candle(c) for c in candles]
        # buy_hold has front_bad=1, so first candle is warmup; after that both agree on +1
        assert results[1] == [1.0]
        assert results[2] == [1.0]
