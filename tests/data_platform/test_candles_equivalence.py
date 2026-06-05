"""WP-2 Unit-2 — legacy vs Nautilus research-candle equivalence (Option B).

Option B stores raw ``float64`` OHLC as the custom :class:`ResearchCandle` data
type, so the read-back adapter must reproduce the legacy loader output
**exactly** (``DataFrame.equals``), not merely within ``rtol``. This test ingests
the parity-fixture ``(ticker, timeframe)`` series into a session-scoped temp
``ParquetDataCatalog`` and asserts byte-identity for both ``load_data`` and
``load_data_multi_ticker``.

Fixtures covered (from WP-1 parity):
  * feature_research:   SI / D, 2000-01-01 -> 2022-12-31.
  * portfolio_research: ES, NQ, GC, CL (+ TLT) / D, W, M, 2000-01-01 -> 2026-05-13.

Skips cleanly when the repo OHLC data is absent.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from data_platform import loaders
from data_platform.loaders import load_data, load_data_multi_ticker, ohlc_data_dir
from data_platform.nautilus.candles import (
    load_data_multi_ticker_nautilus,
    load_data_nautilus,
)
from data_platform.nautilus.catalog import get_catalog
from data_platform.nautilus.research_ingest import ingest_research_candles
from lib.core.enums import Ticker, TimeFrame

# --- fixture windows / sets ---------------------------------------------------
_FEATURE_TICKERS = [Ticker.SI]
_FEATURE_TF = [TimeFrame.D]
_FEATURE_START = datetime(2000, 1, 1)
_FEATURE_END = datetime(2022, 12, 31)

_PORTFOLIO_TICKERS = [Ticker.ES, Ticker.NQ, Ticker.GC, Ticker.CL, Ticker.TLT]
_PORTFOLIO_TF = [TimeFrame.D, TimeFrame.W, TimeFrame.M]
_PORTFOLIO_START = datetime(2000, 1, 1)
_PORTFOLIO_END = datetime(2026, 5, 13)

_ALL_TICKERS = sorted(
    {*_FEATURE_TICKERS, *_PORTFOLIO_TICKERS}, key=lambda t: t.name
)
_ALL_TF = [TimeFrame.D, TimeFrame.W, TimeFrame.M]


def _has_ohlc_data() -> bool:
    root = ohlc_data_dir()
    return root.exists() and any(root.iterdir())


pytestmark = pytest.mark.skipif(
    not _has_ohlc_data(), reason="repo OHLC data not present in data/ohlc_data"
)


@pytest.fixture(scope="module")
def research_catalog(tmp_path_factory):
    """Ingest all fixture (ticker, timeframe) series into a temp catalog once."""
    root = tmp_path_factory.mktemp("research_candle_catalog")
    catalog = get_catalog(root)
    ingest_research_candles(_ALL_TICKERS, _ALL_TF, catalog=catalog)
    # Clear the legacy in-process cache so each load reads fresh (paranoia: the
    # cache is keyed by file path so it cannot leak across the flag, but keep the
    # comparison hermetic).
    loaders._LOAD_DATA_CACHE.clear()
    return catalog


def _assert_load_data_equal(
    ticker: Ticker, tf: TimeFrame, start: datetime, end: datetime, catalog
) -> None:
    legacy = load_data(ticker, tf, start=start, end=end)
    nautilus = load_data_nautilus(ticker, tf, start=start, end=end, catalog=catalog)
    assert legacy.equals(nautilus), (
        f"load_data mismatch for {ticker.name}/{tf.name}: "
        f"dtypes legacy={legacy.dtypes.to_dict()} nautilus={nautilus.dtypes.to_dict()}"
    )


# --- feature_research fixture -------------------------------------------------
def test_load_data_equivalence_feature_research(research_catalog) -> None:
    _assert_load_data_equal(
        Ticker.SI, TimeFrame.D, _FEATURE_START, _FEATURE_END, research_catalog
    )


def test_load_data_multi_ticker_equivalence_feature_research(research_catalog) -> None:
    legacy = load_data_multi_ticker(
        _FEATURE_TICKERS, TimeFrame.D, start=_FEATURE_START, end=_FEATURE_END
    )
    nautilus = load_data_multi_ticker_nautilus(
        _FEATURE_TICKERS,
        TimeFrame.D,
        start=_FEATURE_START,
        end=_FEATURE_END,
        catalog=research_catalog,
    )
    assert legacy.equals(nautilus)


# --- portfolio_research fixture ----------------------------------------------
@pytest.mark.parametrize("ticker", _PORTFOLIO_TICKERS, ids=lambda t: t.name)
@pytest.mark.parametrize("tf", _PORTFOLIO_TF, ids=lambda t: t.name)
def test_load_data_equivalence_portfolio_research(
    ticker, tf, research_catalog
) -> None:
    _assert_load_data_equal(
        ticker, tf, _PORTFOLIO_START, _PORTFOLIO_END, research_catalog
    )


@pytest.mark.parametrize("tf", _PORTFOLIO_TF, ids=lambda t: t.name)
def test_load_data_multi_ticker_equivalence_portfolio_research(
    tf, research_catalog
) -> None:
    legacy = load_data_multi_ticker(
        _PORTFOLIO_TICKERS, tf, start=_PORTFOLIO_START, end=_PORTFOLIO_END
    )
    nautilus = load_data_multi_ticker_nautilus(
        _PORTFOLIO_TICKERS,
        tf,
        start=_PORTFOLIO_START,
        end=_PORTFOLIO_END,
        catalog=research_catalog,
    )
    assert legacy.equals(nautilus), (
        f"multi-ticker mismatch for {tf.name}: "
        f"shapes legacy={legacy.shape} nautilus={nautilus.shape}"
    )
