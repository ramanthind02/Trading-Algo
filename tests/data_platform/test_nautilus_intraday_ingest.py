"""WP-2 Unit 1b — NDX 2026 MT5 intraday ingest smoke test.

Ingests the NDX 1-minute bars + a bounded slice of bid/ask ticks into a temp
``ParquetDataCatalog``, reads them back, and asserts:
  * bars and quotes are non-empty,
  * each bar's ``ts_init`` equals its close (open + 1 minute),
  * all bar OHLC prices and quote bid/ask prices are positive.

Skips cleanly when ``data/mt5_data/NDX`` is absent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data_platform.nautilus.catalog import get_catalog
from data_platform.nautilus.ingest import (
    _read_partitions,
    ingest_mt5_intraday,
    mt5_data_root,
)
from data_platform.nautilus.instruments import to_nautilus_instrument
from data_platform.nautilus.ingest import _resolve_instrument

_SYMBOL = "NDX"
_ONE_MINUTE_NS = 60 * 1_000_000_000
_MAX_TICKS = 5_000  # the raw NDX tick history is ~54M rows; cap for the smoke test


pytestmark = pytest.mark.skipif(
    not (mt5_data_root() / _SYMBOL).exists(),
    reason=f"data/mt5_data/{_SYMBOL} not present",
)


def _catalog_with_instrument(tmp_path):
    catalog = get_catalog(tmp_path / "catalog")
    inst = _resolve_instrument(_SYMBOL)
    catalog.write_data([to_nautilus_instrument(inst)])
    return catalog, inst


def test_ndx_bars_ingest_ts_init_is_close(tmp_path) -> None:
    catalog, inst = _catalog_with_instrument(tmp_path)
    result = ingest_mt5_intraday(_SYMBOL, catalog, max_ticks=_MAX_TICKS)
    assert result.bars_written > 0

    bar_type = f"{inst.id}-1-MINUTE-LAST-EXTERNAL"
    bars = catalog.bars(bar_types=[bar_type])
    assert len(bars) > 0

    # Source open times -> expected close ns set.
    src = _read_partitions(mt5_data_root() / _SYMBOL / "bars_M1")
    expected_close_ns = {
        int(pd.Timestamp(t).value) + _ONE_MINUTE_NS for t in src["time"]
    }
    for bar in bars:
        # ts_init is the bar close and equals ts_event.
        assert bar.ts_init == bar.ts_event
        assert bar.ts_init in expected_close_ns
        assert float(bar.open) > 0
        assert float(bar.high) > 0
        assert float(bar.low) > 0
        assert float(bar.close) > 0
        # NDX precision is 2.
        assert bar.close.precision == 2


def test_ndx_quote_ticks_ingest(tmp_path) -> None:
    catalog, inst = _catalog_with_instrument(tmp_path)
    result = ingest_mt5_intraday(_SYMBOL, catalog, max_ticks=_MAX_TICKS)
    assert result.quotes_written > 0

    quotes = catalog.quote_ticks(instrument_ids=[str(inst.id)])
    assert len(quotes) > 0
    for q in quotes:
        assert float(q.bid_price) > 0
        assert float(q.ask_price) > 0
