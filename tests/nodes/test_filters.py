"""Unit tests for the composable signal filter framework.

Tests cover:
- SignalFilter ABC contract
- VolatilityFilter logic with synthetic candles
- FilteredBiasNode wrapping, neutral-value masking, and warmup behaviour
- Column naming with __f_ suffixes
- FilterSpec / create_filter factory
- Singleton (get_instance) compatibility with filters
- parse_feature_column_name round-trip with filter suffixes
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes import BiasNode
from nodes.filtered import FilteredBiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle
from utils.core import helpers
from utils.core.helpers import create_filtered_bias_node

from filters import (
    FilterSpec,
    SignalFilter,
    create_filter,
    register_filter,
)
from filters.volatility import VolatilityFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(
    close: float,
    day_index: int,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
) -> Candle:
    dt = datetime(2020, 1, 1) + timedelta(days=day_index)
    h = high if high is not None else close * 1.01
    lo = low if low is not None else close * 0.99
    return Candle(
        datetime=dt,
        open=close,
        high=h,
        low=lo,
        close=close,
        volume=volume,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


class _DummyNode(BiasNode):
    """Trivial node that echoes the close price as its signal."""

    def __init__(self, ticker: Ticker, tf: TimeFrame) -> None:
        super().__init__(ticker, tf)
        self.module_name = "dummy"
        self.output_features = ["signal"]
        self.params = {}
        self.front_bad = 1
        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List:
        return [candle.close]


class _AlwaysPassFilter(SignalFilter):
    """Filter that always passes — for testing the framework."""

    filter_name = "alwayspass"

    def __init__(self) -> None:
        self.params = {}
        self.warmup = 0

    def update(self, candle: Candle) -> None:
        pass

    def should_pass(self) -> bool:
        return True

    def reset(self) -> None:
        pass


class _AlwaysBlockFilter(SignalFilter):
    """Filter that always blocks — for testing neutral masking."""

    filter_name = "alwaysblock"

    def __init__(self) -> None:
        self.params = {}
        self.warmup = 0

    def update(self, candle: Candle) -> None:
        pass

    def should_pass(self) -> bool:
        return False

    def reset(self) -> None:
        pass


class _WarmupFilter(SignalFilter):
    """Filter that blocks during warmup and passes after."""

    filter_name = "warmuptest"

    def __init__(self, warmup_len: int = 5) -> None:
        self.params = {"warmup_len": warmup_len}
        self.warmup = warmup_len
        self._n = 0

    def update(self, candle: Candle) -> None:
        self._n += 1

    def should_pass(self) -> bool:
        return self._n >= self.warmup

    def reset(self) -> None:
        self._n = 0


# ---------------------------------------------------------------------------
# SignalFilter ABC contract
# ---------------------------------------------------------------------------

class TestSignalFilterABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            SignalFilter()  # type: ignore[abstract]

    def test_get_filter_suffix_empty_params(self) -> None:
        f = _AlwaysPassFilter()
        assert f.get_filter_suffix() == "__f_alwayspass"

    def test_get_filter_suffix_with_params(self) -> None:
        f = VolatilityFilter(atr_period=14, rank_period=252, regime="high", threshold=0.5)
        suffix = f.get_filter_suffix()
        assert suffix.startswith("__f_vol_")
        assert "atrPeriod_14" in suffix
        assert "rankPeriod_252" in suffix
        assert "regime_high" in suffix
        assert "threshold_0.5" in suffix


# ---------------------------------------------------------------------------
# FilterSpec + create_filter factory
# ---------------------------------------------------------------------------

class TestFilterFactory:
    def test_create_volatility_filter(self) -> None:
        spec = FilterSpec(filter_name="vol", params={"atr_period": 20, "regime": "low"})
        filt = create_filter(spec)
        assert isinstance(filt, VolatilityFilter)
        assert filt.atr_period == 20
        assert filt.regime == "low"

    def test_unknown_filter_raises(self) -> None:
        spec = FilterSpec(filter_name="nonexistent", params={})
        with pytest.raises(KeyError, match="nonexistent"):
            create_filter(spec)

    def test_register_filter_decorator(self) -> None:
        @register_filter
        class _TestFilter(SignalFilter):
            filter_name = "_test_register"

            def __init__(self) -> None:
                self.params = {}
                self.warmup = 0

            def update(self, candle: Candle) -> None:
                pass

            def should_pass(self) -> bool:
                return True

            def reset(self) -> None:
                pass

        spec = FilterSpec(filter_name="_test_register", params={})
        assert isinstance(create_filter(spec), _TestFilter)


# ---------------------------------------------------------------------------
# FilteredBiasNode
# ---------------------------------------------------------------------------

class TestFilteredBiasNode:
    def test_passthrough_with_always_pass(self) -> None:
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        filtered = FilteredBiasNode(node, [_AlwaysPassFilter()])
        results = [filtered.add_candle(_make_candle(100.0 + i, i)) for i in range(5)]
        # After warmup (front_bad=1), signals should pass through
        assert results[-1] == [104.0]

    def test_neutral_masking_with_always_block(self) -> None:
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        filtered = FilteredBiasNode(node, [_AlwaysBlockFilter()])
        # Feed enough candles to get past warmup
        results = [filtered.add_candle(_make_candle(100.0 + i, i)) for i in range(5)]
        # After warmup, blocked signals should be neutral (0.0)
        for r in results[1:]:  # skip warmup period
            assert r == [0.0]

    def test_custom_neutral_value(self) -> None:
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        filtered = FilteredBiasNode(node, [_AlwaysBlockFilter()], neutral_value=50.0)
        results = [filtered.add_candle(_make_candle(100.0 + i, i)) for i in range(5)]
        for r in results[1:]:
            assert r == [50.0]

    def test_nan_neutral_value(self) -> None:
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        filtered = FilteredBiasNode(
            node, [_AlwaysBlockFilter()], neutral_value=float("nan")
        )
        results = [filtered.add_candle(_make_candle(100.0 + i, i)) for i in range(5)]
        for r in results[1:]:
            assert len(r) == 1
            assert math.isnan(r[0])

    def test_warmup_passthrough(self) -> None:
        """During warmup, wrapped node's raw output is returned even if filter blocks."""
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        warmup_filter = _WarmupFilter(warmup_len=3)
        filtered = FilteredBiasNode(node, [warmup_filter])
        # front_bad = max(node.front_bad=1, filter.warmup=3) = 3
        assert filtered.front_bad == 3
        # First 3 candles are warmup — raw signal passes through
        r0 = filtered.add_candle(_make_candle(100.0, 0))
        r1 = filtered.add_candle(_make_candle(101.0, 1))
        r2 = filtered.add_candle(_make_candle(102.0, 2))
        assert r0 == [100.0]
        assert r1 == [101.0]
        assert r2 == [102.0]

    def test_stacked_filters_and_logic(self) -> None:
        """With AND logic, one blocking filter causes neutral output."""
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        filtered = FilteredBiasNode(node, [_AlwaysPassFilter(), _AlwaysBlockFilter()])
        results = [filtered.add_candle(_make_candle(100.0 + i, i)) for i in range(5)]
        for r in results[1:]:
            assert r == [0.0]


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------

class TestColumnNaming:
    def test_filtered_column_names(self) -> None:
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        vol_filter = VolatilityFilter(atr_period=14, regime="high", threshold=0.5)
        filtered = FilteredBiasNode(node, [vol_filter])
        cols = filtered.get_column_names()
        assert len(cols) == 1
        assert cols[0].startswith("dummy_signal_D")
        assert "__f_vol_" in cols[0]

    def test_stacked_filter_column_names(self) -> None:
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        f1 = _AlwaysPassFilter()
        f2 = VolatilityFilter(atr_period=20, regime="low")
        filtered = FilteredBiasNode(node, [f1, f2])
        cols = filtered.get_column_names()
        assert len(cols) == 1
        name = cols[0]
        assert "__f_alwayspass" in name
        assert "__f_vol_" in name
        # f1 suffix should come before f2 suffix
        assert name.index("__f_alwayspass") < name.index("__f_vol_")

    def test_parse_round_trip_with_filter(self) -> None:
        col = "rsi_signal_D_lookback_14__f_vol_atrPeriod_20_regime_high"
        parsed = helpers.parse_feature_column_name(col)
        assert parsed["module"] == "rsi"
        assert parsed["feature"] == "signal"
        assert parsed["params"]["lookback"] == 14
        assert len(parsed["filters"]) == 1
        f = parsed["filters"][0]
        assert f["filter_name"] == "vol"
        assert f["params"]["atrPeriod"] == 20
        assert f["params"]["regime"] == "high"

    def test_parse_no_filter_still_works(self) -> None:
        col = "rsi_signal_D_lookback_14"
        parsed = helpers.parse_feature_column_name(col)
        assert parsed["module"] == "rsi"
        assert parsed["filters"] == []

    def test_parse_stacked_filters(self) -> None:
        col = "ewmac_signal_D_spanFast_16__f_vol_atrPeriod_14_regime_high__f_alwayspass"
        parsed = helpers.parse_feature_column_name(col)
        assert parsed["module"] == "ewmac"
        assert len(parsed["filters"]) == 2
        assert parsed["filters"][0]["filter_name"] == "vol"
        assert parsed["filters"][1]["filter_name"] == "alwayspass"


# ---------------------------------------------------------------------------
# VolatilityFilter logic
# ---------------------------------------------------------------------------

class TestVolatilityFilter:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="atr_period"):
            VolatilityFilter(atr_period=0)
        with pytest.raises(ValueError, match="regime"):
            VolatilityFilter(regime="invalid")
        with pytest.raises(ValueError, match="threshold"):
            VolatilityFilter(threshold=1.5)

    def test_passes_during_warmup(self) -> None:
        filt = VolatilityFilter(atr_period=5, rank_period=10)
        for i in range(9):
            filt.update(_make_candle(100.0 + i, i))
            assert filt.should_pass() is True, f"Expected pass during warmup at candle {i}"

    def test_high_regime_passes_on_spike(self) -> None:
        """After warmup, a volatility spike should pass the 'high' filter."""
        filt = VolatilityFilter(atr_period=5, rank_period=20, regime="high", threshold=0.5)
        # Feed quiet candles to fill warmup
        for i in range(25):
            filt.update(_make_candle(100.0, i, high=100.5, low=99.5))
        # Now feed a high-vol candle
        filt.update(_make_candle(100.0, 26, high=110.0, low=90.0))
        assert filt.should_pass() is True

    def test_low_regime_blocks_on_spike(self) -> None:
        """A volatility spike should be blocked by the 'low' regime filter."""
        filt = VolatilityFilter(atr_period=5, rank_period=20, regime="low", threshold=0.5)
        for i in range(25):
            filt.update(_make_candle(100.0, i, high=100.5, low=99.5))
        filt.update(_make_candle(100.0, 26, high=110.0, low=90.0))
        assert filt.should_pass() is False

    def test_reset(self) -> None:
        filt = VolatilityFilter(atr_period=5, rank_period=10)
        for i in range(15):
            filt.update(_make_candle(100.0, i))
        filt.reset()
        assert filt._n_candles == 0
        assert len(filt._atr_history) == 0
        assert filt.should_pass() is True


# ---------------------------------------------------------------------------
# Singleton compatibility
# ---------------------------------------------------------------------------

class TestSingletonCompatibility:
    def test_filtered_node_get_instance(self) -> None:
        """get_instance with same args returns same object."""
        BiasNode._instances.clear()
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        f = _AlwaysPassFilter()
        a = FilteredBiasNode.get_instance(node, (f,))
        b = FilteredBiasNode.get_instance(node, (f,))
        assert a is b

    def test_different_filters_different_instances(self) -> None:
        BiasNode._instances.clear()
        node = _DummyNode(Ticker.ES, TimeFrame.D)
        a = FilteredBiasNode.get_instance(node, (_AlwaysPassFilter(),))
        b = FilteredBiasNode.get_instance(node, (_AlwaysBlockFilter(),))
        assert a is not b


# ---------------------------------------------------------------------------
# create_filtered_bias_node end-to-end
# ---------------------------------------------------------------------------

class TestCreateFilteredBiasNode:
    def test_no_filters_returns_raw_node(self) -> None:
        BiasNode._instances.clear()
        node = create_filtered_bias_node(
            module_name="momentum",
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            params={"lookback": 10},
            filter_specs=[],
        )
        assert not isinstance(node, FilteredBiasNode)

    def test_with_filter_returns_wrapped_node(self) -> None:
        BiasNode._instances.clear()
        spec = FilterSpec(filter_name="vol", params={"atr_period": 14, "regime": "high"})
        node = create_filtered_bias_node(
            module_name="momentum",
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            params={"lookback": 10},
            filter_specs=[spec],
        )
        assert isinstance(node, FilteredBiasNode)
        assert len(node.filters) == 1
        assert isinstance(node.filters[0], VolatilityFilter)


# ---------------------------------------------------------------------------
# BiasNodeSpec filters field
# ---------------------------------------------------------------------------

class TestBiasNodeSpec:
    def test_default_empty_filters(self) -> None:
        from feature_selection.base_models.feature_base_model import BiasNodeSpec

        spec = BiasNodeSpec(module_name="rsi", timeframes=[TimeFrame.D], params={"lookback": 14})
        assert spec.filters == ()

    def test_with_filters(self) -> None:
        from feature_selection.base_models.feature_base_model import BiasNodeSpec

        fs = FilterSpec(filter_name="vol", params={"atr_period": 14})
        spec = BiasNodeSpec(
            module_name="rsi",
            timeframes=[TimeFrame.D],
            params={"lookback": 14},
            filters=(fs,),
        )
        assert len(spec.filters) == 1
        assert spec.filters[0].filter_name == "vol"
