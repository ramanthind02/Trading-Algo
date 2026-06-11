"""Unit tests for the CFD daily de-stale guard (cfd_candles).

The de-stale step overwrites a session's OPEN with the first M1 bar's CLOSE to
step over the 00:00-01:00 financing dead zone. That is only valid on TRUE
intraday days; on sparse days (pre-2016 daily-as-M1, 2016-17 hourly-as-M1) the
first M1 close IS the daily close, so the overwrite would collapse open==close.
``MIN_INTRADAY_BARS_FOR_DESTALE`` gates it: dense days de-stale, sparse days keep
their genuine open.

These tests build a small synthetic broker store on disk (D1 + M1 parquet) and
point ``_sym_dir`` at it, so no real data is touched.
"""
from __future__ import annotations

import pandas as pd
import pytest

import data_platform.providers.mt5.cfd_candles as C

_SYM = "TEST_SYM"


def _write_store(root, d1: pd.DataFrame, m1: pd.DataFrame) -> None:
    """Write a synthetic broker store: bars_D1/part.parquet + bars_M1/year=*/part.parquet."""
    sym_dir = root / _SYM
    (sym_dir / "bars_D1").mkdir(parents=True, exist_ok=True)
    d1.to_parquet(sym_dir / "bars_D1" / "part.parquet")
    for year, grp in m1.assign(_y=pd.to_datetime(m1["time"]).dt.year).groupby("_y"):
        ydir = sym_dir / "bars_M1" / f"year={year}"
        ydir.mkdir(parents=True, exist_ok=True)
        grp.drop(columns=["_y"]).to_parquet(ydir / "part.parquet")


@pytest.fixture()
def synthetic_symbol(tmp_path, monkeypatch):
    """A two-day synthetic symbol: one sparse (1 bar/day) day, one dense (many bars) day.

    Day A (sparse): broker D1 open != close; M1 has a single bar whose close equals
                    the daily close. The guard must KEEP the real open (open != close).
    Day B (dense) : M1 has >MIN_INTRADAY_BARS_FOR_DESTALE bars; the first bar's open
                    is the stale dead-zone price (== prior close) and its close is the
                    real session open. The guard must DE-STALE (open == first M1 close).
    """
    day_a = pd.Timestamp("2015-06-01")  # sparse era
    day_b = pd.Timestamp("2020-06-01")  # dense era

    # Broker D1: real opens that differ from closes on BOTH days.
    d1 = pd.DataFrame(
        {
            "time": [day_a, day_b],
            "open": [100.0, 200.0],   # genuine session opens
            "high": [110.0, 220.0],
            "low": [95.0, 195.0],
            "close": [105.0, 210.0],  # day A: open(100) != close(105)
            "tick_volume": [10, 5000],
        }
    )

    n_dense = C.MIN_INTRADAY_BARS_FOR_DESTALE + 5
    # Day A M1: ONE bar; its close == daily close (105) — de-staling would zero return.
    a_rows = pd.DataFrame(
        {
            "time": [day_a + pd.Timedelta(hours=1)],
            "open": [99.0],
            "high": [106.0],
            "low": [98.0],
            "close": [105.0],
            "tick_volume": [10],
        }
    )
    # Day B M1: many minute bars. Bar 1 opens stale (180 == "prior close") and closes
    # at the real session open (201). Later bars walk up to the daily close (210).
    b_times = [day_b + pd.Timedelta(hours=1, minutes=i) for i in range(n_dense)]
    b_closes = [201.0] + [201.0 + (210.0 - 201.0) * (i / (n_dense - 1)) for i in range(1, n_dense)]
    b_opens = [180.0] + b_closes[:-1]  # bar 1 open is the stale dead-zone carry
    b_rows = pd.DataFrame(
        {
            "time": b_times,
            "open": b_opens,
            "high": [max(o, c) + 0.5 for o, c in zip(b_opens, b_closes)],
            "low": [min(o, c) - 0.5 for o, c in zip(b_opens, b_closes)],
            "close": b_closes,
            "tick_volume": [50] * n_dense,
        }
    )
    m1 = pd.concat([a_rows, b_rows], ignore_index=True)

    _write_store(tmp_path, d1, m1)
    monkeypatch.setattr(C, "_sym_dir", lambda symbol: tmp_path / symbol)
    C._cfd_daily_frame.cache_clear()
    yield {"day_a": day_a, "day_b": day_b}
    C._cfd_daily_frame.cache_clear()


def test_sparse_day_keeps_real_open(synthetic_symbol):
    """A 1-bar/day (sparse) session must KEEP its genuine open (open != close)."""
    df = C._cfd_daily_frame(_SYM).set_index("datetime")
    row = df.loc[synthetic_symbol["day_a"]]
    assert row["open"] == pytest.approx(100.0), "sparse-day open must be the real D1 open"
    assert row["close"] == pytest.approx(105.0)
    assert row["open"] != row["close"], "sparse day must not be collapsed by de-stale"


def test_dense_day_is_destaled(synthetic_symbol):
    """A many-bar (dense) session must be DE-STALED to the first M1 bar's close."""
    df = C._cfd_daily_frame(_SYM).set_index("datetime")
    row = df.loc[synthetic_symbol["day_b"]]
    # First M1 bar's close is 201.0 (the real session open), not the stale 180/200.
    assert row["open"] == pytest.approx(201.0), "dense-day open must be first M1 close"
    assert row["open"] != pytest.approx(200.0), "dense-day open must not stay the stale D1 open"
    assert row["close"] == pytest.approx(210.0)
    assert row["open"] != row["close"]


def test_first_tradeable_open_reports_bar_counts(synthetic_symbol):
    """_first_tradeable_open returns per-day first close AND bar count for the guard."""
    t = C._first_tradeable_open(_SYM)
    assert set(t.columns) == {"first_close", "bars"}
    assert int(t.loc[synthetic_symbol["day_a"], "bars"]) == 1
    assert int(t.loc[synthetic_symbol["day_b"], "bars"]) > C.MIN_INTRADAY_BARS_FOR_DESTALE


def test_splice_default_is_intraday_cutover():
    """The spliced-feed default boundary is the documented 2018 intraday cutover."""
    from datetime import date

    assert C.CFD_INTRADAY_CUTOVER == date(2018, 1, 1)
    assert C._SPLICE_DEFAULT == pd.Timestamp(C.CFD_INTRADAY_CUTOVER)


def test_ratio_loader_returns_faithful_pct(monkeypatch):
    """load_futures_ratio_candles_raw preserves close/open % (faithful) on synthetic data.

    The ``futures_ratio`` research feed is backed by this loader, so the proportional-back-adjust
    %-returns must pass through unchanged.
    """
    import numpy as np

    from lib.core.enums import TimeFrame

    synth = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=10, freq="D"),
            "open": np.full(10, 100.0),
            "high": np.full(10, 103.0),
            "low": np.full(10, 100.0),
            "close": np.full(10, 103.0),  # close/open == 1.03 every bar
            "volume": np.zeros(10, dtype="int64"),
        }
    )
    monkeypatch.setattr(C, "_read_ratio_parquet", lambda path: synth.copy())
    monkeypatch.setattr(C.Path, "exists", lambda self: True)

    out = C.load_futures_ratio_candles_raw("ES", TimeFrame.D)
    ratios = (out["close"] / out["open"]).to_numpy()
    np.testing.assert_allclose(ratios, np.full(len(out), 1.03), atol=1e-12)
