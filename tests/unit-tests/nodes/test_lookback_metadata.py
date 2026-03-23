from __future__ import annotations

import sys
from types import ModuleType

if "polars" not in sys.modules:
    polars_stub = ModuleType("polars")
    polars_stub.DataFrame = object
    polars_stub.Series = object
    sys.modules["polars"] = polars_stub

from nodes import LookbackContribution
from nodes.filtered import FilteredBiasNode
from nodes.mean_reversion.rsi.rsi import RSI
from nodes.misc.transforms.ts_feature import TimeSeriesFeatureNode
from nodes.momentum.oscillators.ultimate_c import UltimateC
from nodes.volatility.ewsd.ewsd import EWSDNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


class _WarmupOnlyFilter:
    filter_name = "warmup_only"

    def __init__(self, warmup: int) -> None:
        self.params = {"warmup": warmup}
        self.warmup = warmup

    def update(self, candle: Candle) -> None:
        _ = candle

    def should_pass(self) -> bool:
        return True

    def reset(self) -> None:
        return None

    def get_filter_suffix(self) -> str:
        return f"__f_{self.warmup}"


def _label_map(contributions: tuple[LookbackContribution, ...]) -> dict[str, int]:
    return {item.label: item.bars for item in contributions}


def test_rsi_declares_lookback_metadata() -> None:
    node = RSI(Ticker.ES, TimeFrame.D, lookback=14)

    contributions = _label_map(node.lookback_contributions())

    assert contributions["lookback"] == 14
    assert contributions["front_bad"] == 14
    assert node.max_lookback() == 14
    assert node.cold_rebuild_candle_count() == 17


def test_ultimate_c_uses_front_bad_for_max_lookback() -> None:
    node = UltimateC(Ticker.ES, TimeFrame.D, lookback=2, factor=2.0, smooth_lookback=2)

    contributions = _label_map(node.lookback_contributions())

    assert contributions["lookback"] == 2
    assert contributions["smoothLookback"] == 2
    assert contributions["front_bad"] == 11
    assert node.max_lookback() == 11
    assert node.cold_rebuild_candle_count() == 14


def test_filtered_bias_node_uses_max_of_wrapped_and_filter_warmups() -> None:
    wrapped = RSI(Ticker.ES, TimeFrame.D, lookback=14)
    filtered = FilteredBiasNode(wrapped, [_WarmupOnlyFilter(warmup=30)])

    assert filtered.max_lookback() == 30


def test_time_series_feature_node_includes_wrapped_warmup() -> None:
    wrapped = RSI(Ticker.ES, TimeFrame.D, lookback=14)
    node = TimeSeriesFeatureNode(
        ticker=Ticker.ES,
        tf=TimeFrame.D,
        wrapped_node=wrapped,
        lookback=60,
        transformation=lambda series: 0.0,
        transformation_name="dummy",
        transformation_args={},
    )

    assert node.max_lookback() == 74


def test_ewsd_declares_long_run_window_metadata() -> None:
    node = EWSDNode(Ticker.ES, TimeFrame.D, long_run_window=2520)

    contributions = _label_map(node.lookback_contributions())

    assert contributions["long_run_window"] == 2520
    assert node.max_lookback() >= 2520
