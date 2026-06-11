"""Unit tests for ``scripts.mt5_data_fetch``.

The real ``MetaTrader5`` package is Windows-only. We patch
``scripts.mt5_data_fetch._mt5`` and ``_MT5_AVAILABLE`` directly so these tests
run on any platform.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# Ensure project root on sys.path for direct test invocation.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts import mt5_data_fetch as fetch_mod
from lib.core.enums import TimeFrame


# ---------------------------------------------------------------------------
# Fake MT5 module
# ---------------------------------------------------------------------------

@dataclass
class FakeMT5Tick:
    time: int
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class FakeMT5Account:
    login: int = 1513537520
    server: str = "FTMO-Demo"
    currency: str = "USD"


class FakeMT5Module:
    """Minimal stub of the ``MetaTrader5`` package surface that
    :mod:`scripts.mt5_data_fetch` actually touches."""

    TIMEFRAME_D1 = 16385

    def __init__(
        self,
        *,
        rates_by_symbol: Optional[Dict[str, np.ndarray]] = None,
        symbol_select_returns: Optional[Dict[str, bool]] = None,
        tick_unix: int = 1730419200,  # 2024-11-01 00:00 UTC
        initialize_returns: bool = True,
        last_error: Any = (0, "ok"),
        account: Optional[FakeMT5Account] = None,
    ) -> None:
        self._rates = rates_by_symbol or {}
        self._symbol_select_returns = symbol_select_returns or {}
        self._tick_unix = tick_unix
        self._init_returns = initialize_returns
        self._last_error_val = last_error
        self._account = account if account is not None else FakeMT5Account()
        # call recorders
        self.initialize_calls: List[tuple] = []
        self.shutdown_count = 0
        self.symbol_select_calls: List[Tuple[str, bool]] = []
        self.copy_rates_calls: List[Tuple[str, int, int, int]] = []

    # ---- module-level functions used by mt5_data_fetch
    def initialize(self, *args, **kwargs) -> bool:
        self.initialize_calls.append((args, kwargs))
        return self._init_returns

    def shutdown(self) -> None:
        self.shutdown_count += 1

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        self.symbol_select_calls.append((symbol, enable))
        return self._symbol_select_returns.get(symbol, True)

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start: int, count: int):
        self.copy_rates_calls.append((symbol, timeframe, start, count))
        return self._rates.get(symbol)

    def symbol_info_tick(self, symbol: str):
        return FakeMT5Tick(time=self._tick_unix)

    def account_info(self):
        return self._account

    def last_error(self):
        return self._last_error_val


def _make_rates(
    *,
    n: int,
    start_unix: int = 1700000000,  # 2023-11-14 22:13 UTC -> normalises to 2023-11-14
    price_open: float = 5000.0,
    step: float = 0.5,
    include_zero_vol_at: Optional[int] = None,
    include_today_unix: Optional[int] = None,
) -> np.ndarray:
    """Build a numpy structured array matching MT5's copy_rates_* return shape."""
    dtype = [
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
        ("spread", "i4"),
        ("real_volume", "i8"),
    ]
    rows = []
    for i in range(n):
        ts = start_unix + i * 86400
        o = price_open + i * step
        rows.append((ts, o, o + 1.0, o - 1.0, o + 0.5, 12345, 10, 0))
    if include_zero_vol_at is not None:
        idx = include_zero_vol_at
        ts = rows[idx][0]
        o = rows[idx][1]
        rows[idx] = (ts, o, o + 1.0, o - 1.0, o + 0.5, 0, 10, 0)  # zero vol
    if include_today_unix is not None:
        # Append a row dated "today" (the broker's current day from the tick)
        o = price_open + n * step
        rows.append((include_today_unix, o, o + 1.0, o - 1.0, o + 0.5, 999, 10, 0))
    return np.array(rows, dtype=dtype)


@pytest.fixture
def install_fake_mt5(monkeypatch: pytest.MonkeyPatch):
    """Returns a factory that installs a FakeMT5Module into mt5_data_fetch."""

    def _install(fake: FakeMT5Module) -> FakeMT5Module:
        monkeypatch.setattr(fetch_mod, "_mt5", fake)
        monkeypatch.setattr(fetch_mod, "_MT5_AVAILABLE", True)
        return fake

    return _install


# ---------------------------------------------------------------------------
# resolve_mt5_terminal_path
# ---------------------------------------------------------------------------

class TestResolveMt5TerminalPath:

    def test_explicit_data_path_wins(self):
        cfg = {
            "data": {"mt5_terminal_path": "C:/explicit/terminal64.exe"},
            "accounts": [{"enabled": True, "terminal_path": "C:/account/terminal64.exe"}],
        }
        assert fetch_mod.resolve_mt5_terminal_path(cfg) == "C:/explicit/terminal64.exe"

    def test_falls_back_to_first_enabled_account(self):
        cfg = {
            "data": {},
            "accounts": [
                {"enabled": False, "terminal_path": "C:/disabled.exe"},
                {"enabled": True, "terminal_path": "C:/enabled.exe"},
                {"enabled": True, "terminal_path": "C:/second_enabled.exe"},
            ],
        }
        assert fetch_mod.resolve_mt5_terminal_path(cfg) == "C:/enabled.exe"

    def test_returns_none_when_nothing_configured(self):
        cfg = {"accounts": []}
        assert fetch_mod.resolve_mt5_terminal_path(cfg) is None

    def test_blank_explicit_path_falls_through(self):
        cfg = {
            "data": {"mt5_terminal_path": "   "},
            "accounts": [{"enabled": True, "terminal_path": "C:/fallback.exe"}],
        }
        assert fetch_mod.resolve_mt5_terminal_path(cfg) == "C:/fallback.exe"


# ---------------------------------------------------------------------------
# fetch_mt5_daily_candles
# ---------------------------------------------------------------------------

class TestFetchMt5DailyCandles:

    def test_returns_standard_candle_shape(self, install_fake_mt5):
        rates = _make_rates(n=10)
        fake = install_fake_mt5(FakeMT5Module(rates_by_symbol={"US500.cash": rates}))
        df = fetch_mod.fetch_mt5_daily_candles(fake, "ES", "US500.cash", lookback_days=10)
        assert list(df.columns) == [
            "datetime", "open", "high", "low", "close", "volume", "ticker", "timeframe",
        ]
        assert (df["ticker"] == "ES").all()
        assert (df["timeframe"] == TimeFrame.D).all()
        # +5 buffer requested
        assert fake.copy_rates_calls[0][3] == 15

    def test_normalises_datetime_to_date_only_midnight(self, install_fake_mt5):
        rates = _make_rates(n=3, start_unix=1700000000)
        fake = install_fake_mt5(FakeMT5Module(rates_by_symbol={"US500.cash": rates}))
        df = fetch_mod.fetch_mt5_daily_candles(fake, "ES", "US500.cash", lookback_days=3)
        for ts in df["datetime"]:
            assert ts.hour == 0 and ts.minute == 0 and ts.second == 0

    def test_keeps_zero_volume_bars(self, install_fake_mt5):
        """We no longer drop zero-volume (holiday) bars — the bias pipeline
        handles them downstream."""
        rates = _make_rates(n=5, include_zero_vol_at=2)
        fake = install_fake_mt5(FakeMT5Module(rates_by_symbol={"US500.cash": rates}))
        df = fetch_mod.fetch_mt5_daily_candles(fake, "ES", "US500.cash", lookback_days=5)
        assert len(df) == 5  # all 5 kept (including the zero-volume bar)
        assert (df["volume"] == 0).any()

    def test_strips_today_partial_bar(self, install_fake_mt5):
        """Critical rubber-duck fix: never let the forming current-day D1 bar
        contaminate the cache."""
        today_unix = 1700604000  # 2023-11-21 22:00 UTC -> normalises to 2023-11-21
        rates = _make_rates(n=4, start_unix=1700000000, include_today_unix=today_unix)
        fake = install_fake_mt5(FakeMT5Module(
            rates_by_symbol={"US500.cash": rates},
            tick_unix=today_unix,  # broker "today" is the same date as the partial bar
        ))
        df = fetch_mod.fetch_mt5_daily_candles(fake, "ES", "US500.cash", lookback_days=5)
        # All returned bars must be strictly before today.
        today_date = pd.Timestamp(datetime.utcfromtimestamp(today_unix).date())
        assert (df["datetime"] < today_date).all()
        # Original 4 historical bars should remain (none happen to be today).
        assert len(df) == 4

    def test_symbol_select_false_returns_empty(self, install_fake_mt5):
        rates = _make_rates(n=5)
        fake = install_fake_mt5(FakeMT5Module(
            rates_by_symbol={"US500.cash": rates},
            symbol_select_returns={"US500.cash": False},
        ))
        df = fetch_mod.fetch_mt5_daily_candles(fake, "ES", "US500.cash", lookback_days=5)
        assert df.empty
        # copy_rates_from_pos must NOT be called if symbol_select failed.
        assert fake.copy_rates_calls == []

    def test_no_rates_returns_empty(self, install_fake_mt5):
        fake = install_fake_mt5(FakeMT5Module(rates_by_symbol={}))
        df = fetch_mod.fetch_mt5_daily_candles(fake, "ES", "US500.cash", lookback_days=5)
        assert df.empty

    def test_empty_mt5_symbol_returns_empty(self, install_fake_mt5):
        fake = install_fake_mt5(FakeMT5Module())
        df = fetch_mod.fetch_mt5_daily_candles(fake, "ES", "", lookback_days=5)
        assert df.empty


# ---------------------------------------------------------------------------
# sync_mt5_dailies_into_central_cache — orchestration
# ---------------------------------------------------------------------------

class TestSyncMt5DailiesIntoCentralCache:

    @pytest.fixture(autouse=True)
    def _patch_store_and_ctstore(self, monkeypatch: pytest.MonkeyPatch):
        """Patch CentralCacheStore + CrossTickerDataStore so the test does not
        touch the real cache singleton."""
        upserts: List[Tuple[str, str, int]] = []  # (ticker, timeframe, n_rows)
        ct_loads: List[str] = []

        class FakeRecord:
            class _Cov:
                start = datetime(2020, 1, 1)
                end = datetime(2024, 1, 1)
            coverage = _Cov()

        class FakeStore:
            def upsert_candles(self_, ticker, timeframe, df):
                upserts.append((ticker.name, timeframe.name, len(df)))

            def describe_candle(self_, ticker, timeframe):
                return FakeRecord()

            def query_candles(self_, ticker, timeframe, start=None, end=None):
                return pd.DataFrame(
                    {
                        "datetime": pd.date_range("2024-01-01", periods=5, freq="D"),
                        "open": [1.0] * 5,
                        "high": [1.0] * 5,
                        "low": [1.0] * 5,
                        "close": [1.0] * 5,
                        "volume": [1] * 5,
                    }
                )

        fake_store = FakeStore()

        import cache.runtime.central_cache as central_cache_mod
        monkeypatch.setattr(
            central_cache_mod.CentralCacheStore, "get_instance",
            classmethod(lambda cls: fake_store),
        )

        class FakeCTStore:
            def __init__(self_):
                self_._loaded: set = set()

            def is_loaded(self_, ticker, timeframe):
                return False

            def set_data(self_, ticker, timeframe, df):
                ct_loads.append(ticker.name)
                self_._loaded.add((ticker, timeframe))

        fake_ct = FakeCTStore()

        import cache.runtime.cross_ticker_store as cts_mod
        monkeypatch.setattr(
            cts_mod.CrossTickerDataStore, "get_instance",
            classmethod(lambda cls: fake_ct),
        )

        # Suppress monthly rebuild by stubbing resample_daily_to_monthly to empty.
        import scripts.enigma_live_forecast as live_mod
        monkeypatch.setattr(
            live_mod, "resample_daily_to_monthly",
            lambda df: pd.DataFrame(),
        )

        self.upserts = upserts
        self.ct_loads = ct_loads

    def _config(self, **data_overrides) -> Dict[str, Any]:
        data = {
            "source": "mt5",
            "max_lookback_days": 50,
            "mt5_min_bars_per_ticker": 5,
        }
        data.update(data_overrides)
        return {
            "instruments": {
                "ES": {"mt5_symbol": "US500.cash"},
                "NQ": {"mt5_symbol": "US100.cash"},
            },
            "data": data,
            "accounts": [
                {"enabled": True, "terminal_path": "C:/Program Files/FTMO/terminal64.exe"}
            ],
        }

    def test_happy_path_writes_each_ticker_to_cache(self, install_fake_mt5):
        rates_es = _make_rates(n=10)
        rates_nq = _make_rates(n=10, price_open=18000.0)
        install_fake_mt5(FakeMT5Module(
            rates_by_symbol={"US500.cash": rates_es, "US100.cash": rates_nq},
        ))
        cfg = self._config()
        runtime_daily, partial = fetch_mod.sync_mt5_dailies_into_central_cache(
            config=cfg, required_tickers={"ES", "NQ"}
        )
        assert partial is None
        assert set(runtime_daily["ticker"]) == {"ES", "NQ"}
        # Each ticker should get upserted to TimeFrame.D
        d_upserts = [u for u in self.upserts if u[1] == "D"]
        assert sorted(t for t, _, _ in d_upserts) == ["ES", "NQ"]

    def test_initialize_failure_raises(self, install_fake_mt5):
        install_fake_mt5(FakeMT5Module(initialize_returns=False, last_error=(-1, "no terminal")))
        cfg = self._config()
        with pytest.raises(RuntimeError, match="mt5.initialize"):
            fetch_mod.sync_mt5_dailies_into_central_cache(
                config=cfg, required_tickers={"ES"}
            )

    def test_missing_required_ticker_fails_fast(self, install_fake_mt5):
        """ES has data, NQ does not — must raise rather than silently produce ES-only signal."""
        rates_es = _make_rates(n=10)
        install_fake_mt5(FakeMT5Module(
            rates_by_symbol={"US500.cash": rates_es},  # US100.cash missing
        ))
        cfg = self._config()
        with pytest.raises(RuntimeError, match="MT5 data fetch failed"):
            fetch_mod.sync_mt5_dailies_into_central_cache(
                config=cfg, required_tickers={"ES", "NQ"}
            )

    def test_insufficient_bars_fails_fast(self, install_fake_mt5):
        rates_es = _make_rates(n=10)
        rates_nq = _make_rates(n=2)  # only 2 bars, below mt5_min_bars_per_ticker=5
        install_fake_mt5(FakeMT5Module(
            rates_by_symbol={"US500.cash": rates_es, "US100.cash": rates_nq},
        ))
        cfg = self._config()
        with pytest.raises(RuntimeError, match="insufficient bars"):
            fetch_mod.sync_mt5_dailies_into_central_cache(
                config=cfg, required_tickers={"ES", "NQ"}
            )

    def test_ticker_missing_from_instruments_fails_fast(self, install_fake_mt5):
        rates_es = _make_rates(n=10)
        install_fake_mt5(FakeMT5Module(rates_by_symbol={"US500.cash": rates_es}))
        cfg = self._config()
        # GC not in config.instruments
        with pytest.raises(RuntimeError, match="no entry in config.instruments"):
            fetch_mod.sync_mt5_dailies_into_central_cache(
                config=cfg, required_tickers={"ES", "GC"}
            )

    def test_shutdown_called_even_on_error(self, install_fake_mt5):
        fake = install_fake_mt5(FakeMT5Module(rates_by_symbol={}))  # all empty -> raises
        cfg = self._config()
        with pytest.raises(RuntimeError):
            fetch_mod.sync_mt5_dailies_into_central_cache(
                config=cfg, required_tickers={"ES"}
            )
        assert fake.shutdown_count == 1

    def test_shutdown_called_on_success(self, install_fake_mt5):
        rates = _make_rates(n=10)
        fake = install_fake_mt5(FakeMT5Module(
            rates_by_symbol={"US500.cash": rates, "US100.cash": rates},
        ))
        cfg = self._config()
        fetch_mod.sync_mt5_dailies_into_central_cache(
            config=cfg, required_tickers={"ES", "NQ"}
        )
        assert fake.shutdown_count == 1

    def test_terminal_path_from_first_enabled_account(self, install_fake_mt5):
        rates = _make_rates(n=10)
        fake = install_fake_mt5(FakeMT5Module(rates_by_symbol={"US500.cash": rates}))
        cfg = self._config()
        cfg["accounts"] = [
            {"enabled": False, "terminal_path": "C:/disabled.exe"},
            {"enabled": True, "terminal_path": "C:/FTMO/terminal64.exe"},
        ]
        fetch_mod.sync_mt5_dailies_into_central_cache(
            config=cfg, required_tickers={"ES"}
        )
        # initialize() should have been called with the enabled account's path.
        assert fake.initialize_calls
        assert fake.initialize_calls[0][0] == ("C:/FTMO/terminal64.exe",)


# ---------------------------------------------------------------------------
# Profile guard (data.source="mt5" only allowed for cfd_prop)
# ---------------------------------------------------------------------------
# This guard lives in enigma_live_forecast.main; here we verify it triggers
# a SystemExit when the wrong profile is paired with data.source=mt5.

class TestProfileGuard:

    def test_mt5_source_with_non_cfd_profile_exits(self, capsys, tmp_path):
        from scripts import enigma_live_forecast as live_mod
        import json

        cfg = {
            "_comment": "test",
            "portfolio": {"vault_root": "vault"},
            "account": {"capital_usd": 100000},
            "connection": {"host": "127.0.0.1", "port": 7497, "client_id": 1},
            "tradeable_tickers": ["ES"],
            "instruments": {"ES": {"mt5_symbol": "US500.cash", "sec_type": "CONTFUT"}},
            "data": {"source": "mt5"},
        }
        cfg_path = tmp_path / "live_forecast_test.json"
        cfg_path.write_text(json.dumps(cfg))

        argv = [
            "enigma_live_forecast.py",
            "--profile", "futures_prop",
            "--config", str(cfg_path),
            "--dry-run",
        ]
        old = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as exc:
                live_mod.main()
        finally:
            sys.argv = old
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "only supported for profile='cfd_prop'" in captured.out
