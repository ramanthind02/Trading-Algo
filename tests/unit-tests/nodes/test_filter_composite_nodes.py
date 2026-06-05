"""Unit tests for FilterGateNode and FilterAndSignalNode."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Union

from nodes import BiasNode
from nodes.composite.filter_and_signal import FilterAndSignalNode
from nodes.composite.filter_gate import FilterGateNode
from nodes.composite.filter_gate_entry_only import FilterGateEntryOnlyNode
from lib.core import helpers
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


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


class _SeqSignalNode(BiasNode):
    """Emits a fixed sequence of outputs (one per ``add_candle``), then repeats last value."""

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        *,
        sequence: list[float],
        front_bad: int = 0,
    ) -> None:
        super().__init__(ticker, tf)
        self.module_name = "seq"
        self.output_features = ["signal"]
        self.params = {}
        self.front_bad = front_bad
        self._sequence = list(sequence)
        self._idx = 0
        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List[float]:
        i = min(self._idx, len(self._sequence) - 1)
        v = float(self._sequence[i])
        self._idx += 1
        self.output.append(v)
        return [v]


class _FixedSignalNode(BiasNode):
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


def _patch_dual_create(
    filter_signal: float,
    signal_signal: float,
    *,
    front_f: int = 0,
    front_s: int = 0,
    target_module: str,
) -> Union[FilterGateNode, FilterAndSignalNode, FilterGateEntryOnlyNode]:
    node_f = _FixedSignalNode(Ticker.ES, TimeFrame.D, signal=filter_signal, front_bad=front_f)
    node_s = _FixedSignalNode(Ticker.ES, TimeFrame.D, signal=signal_signal, front_bad=front_s)
    call_count = 0

    def _fake_create(module_name: str, ticker: Ticker, tf: TimeFrame, params: dict) -> BiasNode:
        nonlocal call_count
        result = node_f if call_count == 0 else node_s
        call_count += 1
        return result

    if target_module == "filter_gate":
        import nodes.composite.filter_gate as _mod

        original = _mod.helpers.create_fresh_bias_node
        _mod.helpers.create_fresh_bias_node = _fake_create  # type: ignore[method-assign]
        try:
            return FilterGateNode(
                Ticker.ES,
                TimeFrame.D,
                filter_module="fixed",
                filter_params={},
                signal_module="fixed",
                signal_params={},
            )
        finally:
            _mod.helpers.create_fresh_bias_node = original  # type: ignore[method-assign]

    if target_module == "filter_gate_entry_only":
        import nodes.composite.filter_gate_entry_only as _m3

        original3 = _m3.helpers.create_fresh_bias_node
        _m3.helpers.create_fresh_bias_node = _fake_create  # type: ignore[method-assign]
        try:
            return FilterGateEntryOnlyNode(
                Ticker.ES,
                TimeFrame.D,
                filter_module="fixed",
                filter_params={},
                signal_module="fixed",
                signal_params={},
            )
        finally:
            _m3.helpers.create_fresh_bias_node = original3  # type: ignore[method-assign]

    import nodes.composite.filter_and_signal as _mod2

    original2 = _mod2.helpers.create_fresh_bias_node
    _mod2.helpers.create_fresh_bias_node = _fake_create  # type: ignore[method-assign]
    try:
        return FilterAndSignalNode(
            Ticker.ES,
            TimeFrame.D,
            filter_module="fixed",
            filter_params={},
            signal_module="fixed",
            signal_params={},
        )
    finally:
        _mod2.helpers.create_fresh_bias_node = original2  # type: ignore[method-assign]


class TestFilterGate:
    def test_filter_nonzero_passes_signal(self) -> None:
        node = _patch_dual_create(1.0, 0.5, target_module="filter_gate")
        assert node.add_candle(_candle()) == [0.5]

    def test_filter_zero_zeros_output(self) -> None:
        node = _patch_dual_create(0.0, 0.5, target_module="filter_gate")
        assert node.add_candle(_candle()) == [0.0]

    def test_negative_filter_still_passes_raw(self) -> None:
        node = _patch_dual_create(-1.0, 0.5, target_module="filter_gate")
        assert node.add_candle(_candle()) == [0.5]

    def test_front_bad_max(self) -> None:
        node = _patch_dual_create(1.0, 1.0, front_f=3, front_s=9, target_module="filter_gate")
        assert node.front_bad == 9


class TestFilterGateEntryOnly:
    def test_holds_when_filter_closes_after_entry(self) -> None:
        node_f = _SeqSignalNode(Ticker.ES, TimeFrame.D, sequence=[1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        node_s = _SeqSignalNode(Ticker.ES, TimeFrame.D, sequence=[0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        call_count = 0

        def _fake_create(module_name: str, ticker: Ticker, tf: TimeFrame, params: dict) -> BiasNode:
            nonlocal call_count
            result = node_f if call_count == 0 else node_s
            call_count += 1
            return result

        import nodes.composite.filter_gate_entry_only as _mod

        original = _mod.helpers.create_fresh_bias_node
        _mod.helpers.create_fresh_bias_node = _fake_create  # type: ignore[method-assign]
        try:
            node = FilterGateEntryOnlyNode(
                Ticker.ES,
                TimeFrame.D,
                filter_module="seq_f",
                filter_params={},
                signal_module="seq_s",
                signal_params={},
            )
        finally:
            _mod.helpers.create_fresh_bias_node = original  # type: ignore[method-assign]

        outs = [node.add_candle(_candle(i))[0] for i in range(6)]
        assert outs == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]

    def test_same_as_filter_gate_when_no_hold_transition(self) -> None:
        node = _patch_dual_create(1.0, 0.5, target_module="filter_gate_entry_only")
        assert node.add_candle(_candle()) == [0.5]


class TestFilterAndSignal:
    def test_agree_long(self) -> None:
        node = _patch_dual_create(1.0, 1.0, target_module="filter_and_signal")
        assert node.add_candle(_candle()) == [1.0]

    def test_agree_short(self) -> None:
        node = _patch_dual_create(-1.0, -1.0, target_module="filter_and_signal")
        assert node.add_candle(_candle()) == [-1.0]

    def test_disagree(self) -> None:
        node = _patch_dual_create(1.0, -1.0, target_module="filter_and_signal")
        assert node.add_candle(_candle()) == [0.0]

    def test_filter_flat(self) -> None:
        node = _patch_dual_create(0.0, 1.0, target_module="filter_and_signal")
        assert node.add_candle(_candle()) == [0.0]

    def test_front_bad_max(self) -> None:
        node = _patch_dual_create(1.0, 1.0, front_f=4, front_s=8, target_module="filter_and_signal")
        assert node.front_bad == 8


class TestTaxonomyFilterComposites:
    def test_registered(self) -> None:
        from nodes._taxonomy import CANONICAL_MODULE_IMPORTS, CANONICAL_MODULE_CLASSES

        assert CANONICAL_MODULE_IMPORTS["filter_gate"] == "nodes.composite.filter_gate"
        assert CANONICAL_MODULE_CLASSES["filter_gate"] == "FilterGateNode"
        assert CANONICAL_MODULE_IMPORTS["filter_gate_entry_only"] == (
            "nodes.composite.filter_gate_entry_only"
        )
        assert CANONICAL_MODULE_CLASSES["filter_gate_entry_only"] == "FilterGateEntryOnlyNode"
        assert CANONICAL_MODULE_IMPORTS["filter_and_signal"] == "nodes.composite.filter_and_signal"
        assert CANONICAL_MODULE_CLASSES["filter_and_signal"] == "FilterAndSignalNode"

    def test_create_filter_gate_fresh(self) -> None:
        BiasNode._instances.clear()
        node = helpers.create_fresh_bias_node(
            "filter_gate",
            Ticker.ES,
            TimeFrame.D,
            {
                "filter_module": "buy_hold",
                "filter_params": {},
                "signal_module": "buy_hold",
                "signal_params": {},
            },
        )
        assert isinstance(node, FilterGateNode)
        candles = [_candle(i) for i in range(3)]
        results = [node.add_candle(c) for c in candles]
        assert results[1] == [1.0]
        assert results[2] == [1.0]

    def test_create_filter_gate_entry_only_fresh(self) -> None:
        BiasNode._instances.clear()
        node = helpers.create_fresh_bias_node(
            "filter_gate_entry_only",
            Ticker.ES,
            TimeFrame.D,
            {
                "filter_module": "buy_hold",
                "filter_params": {},
                "signal_module": "buy_hold",
                "signal_params": {},
            },
        )
        assert isinstance(node, FilterGateEntryOnlyNode)
