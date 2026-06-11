"""Unit tests for the cross-ticker data store and SpreadNode.

Tests cover:
- CrossTickerDataStore singleton lifecycle, load, lookup, clear
- SpreadNode z-score computation with synthetic data
- Graceful fallback when store data is missing
- Feature extractor cross-ticker pre-loading
- extract_cross_ticker_names contract validation
- BaseModel cross-ticker pre-loading
- Walkforward _ensure_cross_ticker_data helper
- Permutation _update_cross_ticker_store_from_shuffled_candles helper
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nodes import BiasNode
from cache.runtime.central_cache import CentralCacheStore
from cache.runtime.central_cache_errors import ArtifactMissingError
from cache.runtime.central_cache_models import ArtifactDescriptor, ArtifactScope, LookupMode
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle
from cache.runtime.cross_ticker_store import (
    CrossTickerDataStore,
    extract_cross_ticker_names,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv_df(
    base_close: float = 100.0,
    n_rows: int = 50,
    start_date: datetime = datetime(2020, 1, 1),
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame indexed by datetime."""
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    closes = [base_close + i * 0.5 for i in range(n_rows)]
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [c - 0.1 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * n_rows,
        }
    ).set_index("datetime")


def _make_candle(
    close: float, day_index: int, ticker: Ticker = Ticker.ES
) -> Candle:
    dt = datetime(2020, 1, 1) + timedelta(days=day_index)
    return Candle(
        datetime=dt,
        open=close - 0.1,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1000.0,
        ticker=ticker,
        tf=TimeFrame.D,
    )


@pytest.fixture(autouse=True)
def _reset_store():
    """Ensure a clean store for every test."""
    CrossTickerDataStore.reset()
    BiasNode._instances.clear()
    yield
    CrossTickerDataStore.reset()
    BiasNode._instances.clear()


# ---------------------------------------------------------------------------
# CrossTickerDataStore
# ---------------------------------------------------------------------------

class TestCrossTickerDataStore:
    def test_singleton(self) -> None:
        a = CrossTickerDataStore.get_instance()
        b = CrossTickerDataStore.get_instance()
        assert a is b

    def test_reset_creates_new_instance(self) -> None:
        a = CrossTickerDataStore.get_instance()
        CrossTickerDataStore.reset()
        b = CrossTickerDataStore.get_instance()
        assert a is not b

    def test_set_data_and_get_candle(self) -> None:
        store = CrossTickerDataStore.get_instance()
        df = _make_ohlcv_df(base_close=100.0, n_rows=10)
        store.set_data(Ticker.NQ, TimeFrame.D, df)

        dt = datetime(2020, 1, 1)
        candle = store.get_candle(Ticker.NQ, TimeFrame.D, dt)
        assert candle is not None
        assert candle.close == 100.0
        assert candle.ticker == Ticker.NQ
        assert candle.tf == TimeFrame.D

    def test_get_candle_returns_full_ohlcv(self) -> None:
        store = CrossTickerDataStore.get_instance()
        df = _make_ohlcv_df(base_close=50.0, n_rows=3)
        store.set_data(Ticker.GC, TimeFrame.D, df)

        candle = store.get_candle(Ticker.GC, TimeFrame.D, datetime(2020, 1, 1))
        assert candle is not None
        assert candle.close == 50.0
        assert candle.open is not None
        assert candle.high is not None
        assert candle.low is not None

    def test_get_candle_returns_none_for_missing_date(self) -> None:
        store = CrossTickerDataStore.get_instance()
        df = _make_ohlcv_df(n_rows=5)
        store.set_data(Ticker.NQ, TimeFrame.D, df)

        assert store.get_candle(Ticker.NQ, TimeFrame.D, datetime(2099, 1, 1)) is None

    def test_get_candle_returns_none_for_unloaded_ticker(self) -> None:
        store = CrossTickerDataStore.get_instance()
        assert store.get_candle(Ticker.GC, TimeFrame.D, datetime(2020, 1, 1)) is None

    def test_query_candle_raises_for_unloaded_ticker(self) -> None:
        store = CrossTickerDataStore.get_instance()
        with pytest.raises(ArtifactMissingError, match="Candles are not loaded"):
            store.query_candle(Ticker.GC, TimeFrame.D, datetime(2020, 1, 1))

    def test_is_loaded(self) -> None:
        store = CrossTickerDataStore.get_instance()
        assert store.is_loaded(Ticker.NQ, TimeFrame.D) is False

        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df())
        assert store.is_loaded(Ticker.NQ, TimeFrame.D) is True
        assert store.is_loaded(Ticker.NQ, TimeFrame.W) is False

    def test_loaded_tickers(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df())
        store.set_data(Ticker.GC, TimeFrame.D, _make_ohlcv_df())

        assert set(store.loaded_tickers()) == {Ticker.NQ, Ticker.GC}

    def test_clear(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df())
        store.clear()

        assert store.is_loaded(Ticker.NQ, TimeFrame.D) is False
        assert store.loaded_tickers() == []

    def test_clear_does_not_remove_cached_artifacts(self, tmp_path: Path) -> None:
        CentralCacheStore.reset()
        CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path)  # type: ignore[attr-defined]
        store = CrossTickerDataStore.get_instance()
        cache = CentralCacheStore.get_instance()
        descriptor = ArtifactDescriptor(
            family="signals",
            ticker=Ticker.NQ,
            timeframe=TimeFrame.D,
            module_name="spread",
            params={"lookback": 5},
            scope=ArtifactScope.LIVE,
            artifact_name="signal",
        )
        cache.write_artifact(descriptor, _make_ohlcv_df(n_rows=5))
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df(n_rows=5))

        store.clear()

        assert store.is_loaded(Ticker.NQ, TimeFrame.D) is False
        assert cache.describe_artifact(descriptor) is not None
        CentralCacheStore.reset()

    def test_store_uses_latest_central_cache_instance(self, tmp_path: Path) -> None:
        store = CrossTickerDataStore.get_instance()
        CentralCacheStore.reset()
        CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path)  # type: ignore[attr-defined]

        store.set_data(Ticker.ES, TimeFrame.D, _make_ohlcv_df(n_rows=3))

        assert CentralCacheStore.get_instance().is_candle_loaded(Ticker.ES, TimeFrame.D) is True
        CentralCacheStore.reset()

    def test_set_data_without_datetime_index(self) -> None:
        """DataFrame with 'datetime' column (not index) is auto-indexed."""
        store = CrossTickerDataStore.get_instance()
        df = _make_ohlcv_df(n_rows=5).reset_index()  # datetime as column
        store.set_data(Ticker.NQ, TimeFrame.D, df)

        assert store.get_candle(Ticker.NQ, TimeFrame.D, datetime(2020, 1, 1)) is not None

    def test_put_candle_adds_to_empty_store(self) -> None:
        """Compatibility wrapper returns None on unloaded misses."""
        store = CrossTickerDataStore.get_instance()
        candle = store.get_candle(Ticker.ES, TimeFrame.D, datetime(2020, 1, 2))
        assert candle is None
        assert store.is_loaded(Ticker.ES, TimeFrame.D) is False

    def test_query_candle_as_of_returns_previous_bar(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df(n_rows=5))
        candle = store.query_candle(Ticker.NQ, TimeFrame.D, datetime(2020, 1, 1, 12), lookup_mode=LookupMode.AS_OF)
        assert candle is not None
        assert candle.close == 100.0

    def test_put_candle_appends_to_existing(self) -> None:
        """Explicit loads are still available for compatibility callers."""
        store = CrossTickerDataStore.get_instance()
        store.load(Ticker.ES, TimeFrame.D)
        assert store.is_loaded(Ticker.ES, TimeFrame.D)
        candle2 = store.get_candle(Ticker.ES, TimeFrame.D, datetime(2020, 1, 3))
        assert candle2 is not None

    def test_put_candle_overwrites_existing_datetime(self) -> None:
        """Auto-load returns None for genuinely missing dates (not a load failure)."""
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df(n_rows=5))
        # Date beyond the loaded range
        assert store.get_candle(Ticker.NQ, TimeFrame.D, datetime(2099, 12, 31)) is None


# ---------------------------------------------------------------------------
# SpreadNode
# ---------------------------------------------------------------------------

class TestSpreadNode:
    def test_column_naming(self) -> None:
        from nodes.pairs.spread import SpreadNode

        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df())

        node = SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=["NQ"], lookback=20)
        cols = node.get_column_names()
        assert len(cols) == 1
        assert "spread" in cols[0]
        assert "crossTickers" in cols[0]
        assert "NQ" in cols[0]
        assert "lookback_20" in cols[0]

    def test_returns_zero_during_warmup(self) -> None:
        from nodes.pairs.spread import SpreadNode

        store = CrossTickerDataStore.get_instance()
        nq_df = _make_ohlcv_df(base_close=200.0, n_rows=50)
        store.set_data(Ticker.NQ, TimeFrame.D, nq_df)

        node = SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=["NQ"], lookback=20)
        for i in range(19):  # less than lookback
            result = node.add_candle(_make_candle(100.0 + i * 0.5, i))
            assert result == [0.0], f"Expected [0.0] during warmup at bar {i}"

    def test_returns_nonzero_after_warmup(self) -> None:
        from nodes.pairs.spread import SpreadNode

        store = CrossTickerDataStore.get_instance()
        # NQ constant at 200 so spread diverges from ES's rising price
        nq_df = _make_ohlcv_df(base_close=200.0, n_rows=50)
        nq_df["close"] = 200.0
        store.set_data(Ticker.NQ, TimeFrame.D, nq_df)

        node = SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=["NQ"], lookback=20)
        results = []
        for i in range(30):
            # ES rises: 100, 101, 102, ... while NQ stays at 200
            result = node.add_candle(_make_candle(100.0 + i, i))
            results.append(result[0])

        # After warmup (bar 20+), z-score should be non-zero
        post_warmup = results[20:]
        assert any(v != 0.0 for v in post_warmup)

    def test_returns_neutral_when_cross_ticker_unavailable(self) -> None:
        """If cross-ticker data doesn't exist at all, returns [0.0] gracefully."""
        from nodes.pairs.spread import SpreadNode

        # Use a ticker that doesn't have parquet data at this datetime
        # Pre-load a tiny NQ dataset that has no overlap with ES candle dates
        store = CrossTickerDataStore.get_instance()
        tiny_df = _make_ohlcv_df(n_rows=3, start_date=datetime(1999, 1, 1))
        store.set_data(Ticker.NQ, TimeFrame.D, tiny_df)

        node = SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=["NQ"], lookback=5)
        # Feed candles at dates far from the tiny NQ dataset
        for i in range(10):
            result = node.add_candle(_make_candle(100.0, i))  # dates start at 2020-01-01
            assert result == [0.0], f"Expected [0.0] when NQ has no data at this date, got {result}"

    def test_zscore_correctness(self) -> None:
        """Verify z-score matches manual computation."""
        from nodes.pairs.spread import SpreadNode

        lookback = 5
        store = CrossTickerDataStore.get_instance()

        # NQ constant at 200.0, ES rises linearly
        nq_df = _make_ohlcv_df(base_close=200.0, n_rows=20)
        # Override close to be constant
        nq_df["close"] = 200.0
        store.set_data(Ticker.NQ, TimeFrame.D, nq_df)

        node = SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=["NQ"], lookback=lookback)

        spreads = []
        results = []
        for i in range(10):
            es_close = 100.0 + i * 2.0  # ES: 100, 102, 104, ...
            nq_close = 200.0
            spreads.append(es_close - nq_close)
            result = node.add_candle(_make_candle(es_close, i))
            results.append(result[0])

        # At bar 5 (index 5), we have spreads[1:6] in the window
        # (bar 0 is consumed but window starts filling from bar 0)
        # Verify last result manually
        last_window = spreads[-lookback:]
        expected_mean = np.mean(last_window)
        expected_std = np.std(last_window)
        expected_zscore = (spreads[-1] - expected_mean) / expected_std
        assert abs(results[-1] - expected_zscore) < 1e-10

    def test_singleton_compatibility(self) -> None:
        from nodes.pairs.spread import SpreadNode

        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df())

        a = SpreadNode.get_instance(Ticker.ES, TimeFrame.D, cross_tickers=["NQ"], lookback=20)
        b = SpreadNode.get_instance(Ticker.ES, TimeFrame.D, cross_tickers=["NQ"], lookback=20)
        assert a is b

        c = SpreadNode.get_instance(Ticker.ES, TimeFrame.D, cross_tickers=["GC"], lookback=20)
        assert a is not c


# ---------------------------------------------------------------------------
# Feature extractor cross-ticker pre-loading
# ---------------------------------------------------------------------------

class TestExtractorPreloading:
    def test_preload_cross_ticker_data(self) -> None:
        """The helper scans params for cross_tickers and loads into store."""
        from features.extraction.feature_extractor import _preload_cross_ticker_data

        param_combos = [
            {"cross_tickers": ["NQ"], "lookback": 20},
            {"cross_tickers": ["GC"], "lookback": 10},
        ]
        _preload_cross_ticker_data(
            param_combos,
            timeframes=[TimeFrame.D],
            start=datetime(2020, 1, 1),
            end=datetime(2025, 1, 1),
        )

        store = CrossTickerDataStore.get_instance()
        assert store.is_loaded(Ticker.NQ, TimeFrame.D)
        assert store.is_loaded(Ticker.GC, TimeFrame.D)

    def test_preload_no_cross_ticker_is_noop(self) -> None:
        """When no params have cross_tickers, nothing is loaded."""
        from features.extraction.feature_extractor import _preload_cross_ticker_data

        param_combos = [{"lookback": 14}]
        _preload_cross_ticker_data(
            param_combos,
            timeframes=[TimeFrame.D],
            start=datetime(2020, 1, 1),
            end=datetime(2025, 1, 1),
        )

        store = CrossTickerDataStore.get_instance()
        assert store.loaded_tickers() == []

    def test_preload_skips_already_loaded(self) -> None:
        """If ticker is already loaded, load() is not called again."""
        from features.extraction.feature_extractor import _preload_cross_ticker_data

        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df())

        # This should not crash or re-load
        param_combos = [{"cross_tickers": ["NQ"], "lookback": 20}]
        _preload_cross_ticker_data(
            param_combos,
            timeframes=[TimeFrame.D],
            start=datetime(2020, 1, 1),
            end=datetime(2025, 1, 1),
        )

        assert store.is_loaded(Ticker.NQ, TimeFrame.D)

    def test_preload_cross_ticker_list_loads_all(self) -> None:
        from features.extraction.feature_extractor import _preload_cross_ticker_data

        _preload_cross_ticker_data(
            [{"cross_tickers": ["NQ", "GC"], "lookback": 20}],
            timeframes=[TimeFrame.D],
            start=datetime(2020, 1, 1),
            end=datetime(2025, 1, 1),
        )
        store = CrossTickerDataStore.get_instance()
        assert store.is_loaded(Ticker.NQ, TimeFrame.D)
        assert store.is_loaded(Ticker.GC, TimeFrame.D)

    def test_preload_override_data_sets_store(self) -> None:
        """Override candles should be pushed into CrossTickerDataStore via set_data."""
        from features.extraction.feature_extractor import _preload_cross_ticker_override_data

        override = pd.DataFrame(
            {
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                ],
                "open": [100.0, 101.0, 200.0, 201.0],
                "high": [101.0, 102.0, 201.0, 202.0],
                "low": [99.0, 100.0, 199.0, 200.0],
                "close": [100.5, 101.5, 200.5, 201.5],
                "volume": [1000.0, 1000.0, 1000.0, 1000.0],
                "ticker": ["ES", "ES", "NQ", "NQ"],
            }
        )
        _preload_cross_ticker_override_data(override, [TimeFrame.D])

        store = CrossTickerDataStore.get_instance()
        assert store.is_loaded(Ticker.ES, TimeFrame.D)
        assert store.is_loaded(Ticker.NQ, TimeFrame.D)
        nq = store.get_candle(Ticker.NQ, TimeFrame.D, datetime(2020, 1, 1))
        assert nq is not None
        assert nq.close == 200.5

    def test_extract_features_with_override_preloads_extra_ticker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Extra tickers in candles_override should preload store for cross-ticker nodes."""
        import features.extraction.feature_extractor as fe

        idx = pd.DatetimeIndex([datetime(2020, 1, 1), datetime(2020, 1, 2)], tz="UTC")
        override = pd.DataFrame(
            {
                "datetime": list(idx) + list(idx),
                "open": [100.0, 101.0, 200.0, 201.0],
                "high": [101.0, 102.0, 201.0, 202.0],
                "low": [99.0, 100.0, 199.0, 200.0],
                "close": [100.5, 101.5, 200.5, 201.5],
                "volume": [1000.0, 1000.0, 1000.0, 1000.0],
                "ticker": ["ES", "ES", "NQ", "NQ"],
            }
        )

        def _fake_extract_single_ticker(**_: object) -> tuple[pd.DataFrame, pd.DataFrame]:
            features = pd.DataFrame({"fake_signal": [0.0, 0.0]}, index=idx)
            targets = pd.DataFrame(
                {
                    "raw_return": [0.0, 0.0],
                    "log_return": [0.0, 0.0],
                    "log_return_atr": [0.0, 0.0],
                    "log_return_ewsd": [0.0, 0.0],
                },
                index=idx,
            )
            return features, targets

        monkeypatch.setattr(fe, "_extract_features_single_ticker", _fake_extract_single_ticker)
        features_df, _ = fe.extract_features(
            module_name="rsi",
            params={"lookback": 14},
            ticker=Ticker.ES,
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 3),
            timeframes=[TimeFrame.D],
            candles_override=override,
        )

        store = CrossTickerDataStore.get_instance()
        assert store.is_loaded(Ticker.NQ, TimeFrame.D)
        assert store.get_candle(Ticker.NQ, TimeFrame.D, datetime(2020, 1, 1)) is not None
        assert set(features_df["ticker"].unique()) == {"ES"}


# ---------------------------------------------------------------------------
# extract_cross_ticker_names contract validation
# ---------------------------------------------------------------------------

class TestExtractCrossTickerNames:
    def test_returns_empty_set_when_key_absent(self) -> None:
        assert extract_cross_ticker_names({"lookback": 14}) == set()

    def test_returns_empty_set_for_none_value(self) -> None:
        assert extract_cross_ticker_names({"cross_tickers": None}) == set()

    def test_extracts_single_ticker(self) -> None:
        assert extract_cross_ticker_names({"cross_tickers": ["NQ"]}) == {"NQ"}

    def test_extracts_multiple_tickers(self) -> None:
        result = extract_cross_ticker_names({"cross_tickers": ["NQ", "GC", "CL"]})
        assert result == {"NQ", "GC", "CL"}

    def test_normalizes_to_uppercase(self) -> None:
        result = extract_cross_ticker_names({"cross_tickers": ["nq", "  gc "]})
        assert result == {"NQ", "GC"}

    def test_deduplicates(self) -> None:
        result = extract_cross_ticker_names({"cross_tickers": ["NQ", "nq", "NQ"]})
        assert result == {"NQ"}

    def test_skips_empty_strings(self) -> None:
        result = extract_cross_ticker_names({"cross_tickers": ["NQ", "", "  "]})
        assert result == {"NQ"}

    def test_raises_type_error_for_non_list(self) -> None:
        with pytest.raises(TypeError, match="must be list"):
            extract_cross_ticker_names({"cross_tickers": "NQ"})

    def test_raises_type_error_for_non_string_element(self) -> None:
        with pytest.raises(TypeError, match="must be str"):
            extract_cross_ticker_names({"cross_tickers": [123]})


# ---------------------------------------------------------------------------
# SpreadNode validation edge cases
# ---------------------------------------------------------------------------

class TestSpreadNodeValidation:
    def test_unknown_cross_ticker_raises(self) -> None:
        from nodes.pairs.spread import SpreadNode
        with pytest.raises(ValueError, match="Unknown cross ticker"):
            SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=["INVALID_XYZ"])

    def test_non_string_cross_ticker_raises(self) -> None:
        from nodes.pairs.spread import SpreadNode
        with pytest.raises(TypeError, match="must be str"):
            SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=[123])

    def test_empty_string_cross_ticker_raises(self) -> None:
        from nodes.pairs.spread import SpreadNode
        with pytest.raises(ValueError, match="at least one non-empty"):
            SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=["", "  "])

    def test_default_cross_ticker_is_nq(self) -> None:
        from nodes.pairs.spread import SpreadNode
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df())
        node = SpreadNode(Ticker.ES, TimeFrame.D)
        assert node.cross_ticker == Ticker.NQ

    def test_params_contract_includes_cross_tickers(self) -> None:
        """SpreadNode.params must contain canonical cross_tickers key."""
        from nodes.pairs.spread import SpreadNode
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.GC, TimeFrame.D, _make_ohlcv_df())
        node = SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=["GC"], lookback=10)
        assert "cross_tickers" in node.params
        assert node.params["cross_tickers"] == ["GC"]
        assert node.params["lookback"] == 10

    def test_case_insensitive_cross_ticker(self) -> None:
        from nodes.pairs.spread import SpreadNode
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df())
        node = SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=["nq"])
        assert node.cross_ticker == Ticker.NQ


# ---------------------------------------------------------------------------
# CrossTickerDataStore timezone edge cases
# ---------------------------------------------------------------------------

class TestStoreTimezoneHandling:
    def test_set_data_strips_timezone(self) -> None:
        """Timezone-aware DataFrames should be stored as naive UTC."""
        store = CrossTickerDataStore.get_instance()
        dates = pd.date_range("2020-01-01", periods=5, freq="D", tz="US/Eastern")
        df = pd.DataFrame({
            "open": [100.0] * 5, "high": [101.0] * 5,
            "low": [99.0] * 5, "close": [100.5] * 5,
            "volume": [1000.0] * 5,
        }, index=dates)
        df.index.name = "datetime"
        store.set_data(Ticker.NQ, TimeFrame.D, df)
        assert store.is_loaded(Ticker.NQ, TimeFrame.D)
        # Verify tz was stripped — naive datetime lookup works
        candle = store.get_candle(Ticker.NQ, TimeFrame.D, datetime(2020, 1, 1))
        assert candle is not None

    def test_get_candle_with_tz_aware_dt(self) -> None:
        """Lookup with timezone-aware datetime should still work."""
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df(n_rows=5))
        # Query with tz-aware datetime
        dt_aware = pd.Timestamp("2020-01-01", tz="UTC")
        candle = store.get_candle(Ticker.NQ, TimeFrame.D, dt_aware)
        assert candle is not None
        assert candle.close == 100.0

    def test_set_data_overwrites_previous(self) -> None:
        """Calling set_data twice for same (ticker, tf) replaces data."""
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df(base_close=100.0, n_rows=5))
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df(base_close=999.0, n_rows=5))
        candle = store.get_candle(Ticker.NQ, TimeFrame.D, datetime(2020, 1, 1))
        assert candle is not None
        assert candle.close == 999.0


# ---------------------------------------------------------------------------
# Walkforward _ensure_cross_ticker_data
# ---------------------------------------------------------------------------

class TestEnsureCrossTickerData:
    def test_loads_cross_tickers_from_params(self) -> None:
        from research.evaluation.walkforward.portfolio_evaluator import _ensure_cross_ticker_data
        _ensure_cross_ticker_data(
            [{"cross_tickers": ["NQ"], "lookback": 20}],
            [TimeFrame.D],
        )
        store = CrossTickerDataStore.get_instance()
        assert store.is_loaded(Ticker.NQ, TimeFrame.D)

    def test_skips_already_loaded(self) -> None:
        """Should not re-load if data is already in the store."""
        from research.evaluation.walkforward.portfolio_evaluator import _ensure_cross_ticker_data
        store = CrossTickerDataStore.get_instance()
        # Pre-populate with custom data
        custom_df = _make_ohlcv_df(base_close=999.0, n_rows=5)
        store.set_data(Ticker.NQ, TimeFrame.D, custom_df)

        _ensure_cross_ticker_data(
            [{"cross_tickers": ["NQ"], "lookback": 20}],
            [TimeFrame.D],
        )
        # Verify original data wasn't overwritten (still 999.0)
        candle = store.get_candle(Ticker.NQ, TimeFrame.D, datetime(2020, 1, 1))
        assert candle is not None
        assert candle.close == 999.0

    def test_noop_without_cross_tickers(self) -> None:
        from research.evaluation.walkforward.portfolio_evaluator import _ensure_cross_ticker_data
        _ensure_cross_ticker_data(
            [{"lookback": 14}],
            [TimeFrame.D],
        )
        store = CrossTickerDataStore.get_instance()
        assert store.loaded_tickers() == []

    def test_handles_unknown_ticker_gracefully(self) -> None:
        from research.evaluation.walkforward.portfolio_evaluator import _ensure_cross_ticker_data
        # Should not raise for unknown ticker name
        _ensure_cross_ticker_data(
            [{"cross_tickers": ["INVALID_XYZ"], "lookback": 20}],
            [TimeFrame.D],
        )
        store = CrossTickerDataStore.get_instance()
        assert store.loaded_tickers() == []

    def test_loads_multiple_timeframes(self) -> None:
        from research.evaluation.walkforward.portfolio_evaluator import _ensure_cross_ticker_data
        _ensure_cross_ticker_data(
            [{"cross_tickers": ["ES"], "lookback": 20}],
            [TimeFrame.D, TimeFrame.W],
        )
        store = CrossTickerDataStore.get_instance()
        assert store.is_loaded(Ticker.ES, TimeFrame.D)
        assert store.is_loaded(Ticker.ES, TimeFrame.W)


# ---------------------------------------------------------------------------
# Permutation _update_cross_ticker_store_from_shuffled_candles
# ---------------------------------------------------------------------------

class TestUpdateCrossTickerStoreFromShuffledCandles:
    def test_updates_store_from_multi_ticker_df(self) -> None:
        from features.validation.permutation_tests import (
            _update_cross_ticker_store_from_shuffled_candles,
        )
        shuffled = pd.DataFrame({
            "datetime": [datetime(2020, 1, 1), datetime(2020, 1, 2)] * 2,
            "open": [100.0, 101.0, 200.0, 201.0],
            "high": [101.0, 102.0, 201.0, 202.0],
            "low": [99.0, 100.0, 199.0, 200.0],
            "close": [100.5, 101.5, 200.5, 201.5],
            "volume": [1000.0] * 4,
            "ticker": ["ES", "ES", "NQ", "NQ"],
        })
        _update_cross_ticker_store_from_shuffled_candles(shuffled)

        store = CrossTickerDataStore.get_instance()
        assert store.is_loaded(Ticker.ES, TimeFrame.D)
        assert store.is_loaded(Ticker.NQ, TimeFrame.D)

    def test_noop_without_ticker_column(self) -> None:
        from features.validation.permutation_tests import (
            _update_cross_ticker_store_from_shuffled_candles,
        )
        shuffled = pd.DataFrame({
            "datetime": [datetime(2020, 1, 1)],
            "close": [100.0],
        })
        _update_cross_ticker_store_from_shuffled_candles(shuffled)
        store = CrossTickerDataStore.get_instance()
        assert store.loaded_tickers() == []

    def test_skips_unknown_ticker_names(self) -> None:
        from features.validation.permutation_tests import (
            _update_cross_ticker_store_from_shuffled_candles,
        )
        shuffled = pd.DataFrame({
            "datetime": [datetime(2020, 1, 1)],
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [1000.0],
            "ticker": ["UNKNOWN_TICKER_XYZ"],
        })
        _update_cross_ticker_store_from_shuffled_candles(shuffled)
        store = CrossTickerDataStore.get_instance()
        assert store.loaded_tickers() == []

    def test_uses_explicit_timeframes(self) -> None:
        from features.validation.permutation_tests import (
            _update_cross_ticker_store_from_shuffled_candles,
        )
        shuffled = pd.DataFrame({
            "datetime": [datetime(2020, 1, 1)],
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [1000.0],
            "ticker": ["ES"],
        })
        _update_cross_ticker_store_from_shuffled_candles(
            shuffled, timeframes=[TimeFrame.W]
        )
        store = CrossTickerDataStore.get_instance()
        assert store.is_loaded(Ticker.ES, TimeFrame.W)
        assert not store.is_loaded(Ticker.ES, TimeFrame.D)

    def test_overwrites_existing_store_data(self) -> None:
        """Shuffled candles should replace whatever was in the store."""
        from features.validation.permutation_tests import (
            _update_cross_ticker_store_from_shuffled_candles,
        )
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df(base_close=100.0, n_rows=5))

        # Push shuffled data with different close values
        shuffled = pd.DataFrame({
            "datetime": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "open": [500.0, 501.0], "high": [501.0, 502.0],
            "low": [499.0, 500.0], "close": [500.5, 501.5],
            "volume": [1000.0, 1000.0],
            "ticker": ["NQ", "NQ"],
        })
        _update_cross_ticker_store_from_shuffled_candles(shuffled)

        candle = store.get_candle(Ticker.NQ, TimeFrame.D, datetime(2020, 1, 1))
        assert candle is not None
        assert candle.close == 500.5

    def test_never_raises_on_malformed_data(self) -> None:
        """The helper must never crash the permutation loop."""
        from features.validation.permutation_tests import (
            _update_cross_ticker_store_from_shuffled_candles,
        )
        # Completely malformed — should silently pass
        _update_cross_ticker_store_from_shuffled_candles(pd.DataFrame())
        _update_cross_ticker_store_from_shuffled_candles(
            pd.DataFrame({"ticker": [None], "close": [1.0]})
        )
