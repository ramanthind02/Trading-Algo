"""Tests for data_platform.nautilus.materialize.

Covers:
  1. materialize 1 symbol → catalog.bars() returns rows with correct precision
  2. re-materialize → idempotent (second call skips the symbol)
  3. append job ingests only bars newer than coverage_end
  4. instruments written to catalog carry the fixed increments
     (price_precision=1, price_increment=0.1 for the NDX stand-in)

Fixture strategy (reuses the test_mt5_scraper_writes idiom):
  - Synthetic M1 parquet files are created in tmp_path to mimic
    ``data/mt5_data/{SYM}/bars_M1/year={Y}/part.parquet``.
  - ``data_platform.nautilus.ingest.mt5_data_root`` AND
    ``data_platform.nautilus.materialize.mt5_data_root`` are both monkeypatched
    to the synthetic store root so no real MT5 data is required.
  - ``data_platform.core.catalog.load_catalog`` is monkeypatched to return a
    one-instrument InstrumentCatalog with the NDX stand-in (price_precision=1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nautilus_trader.model.data import Bar, QuoteTick

from data_platform.nautilus.catalog import get_catalog


# ---------------------------------------------------------------------------
# Helpers: synthetic MT5 store + catalog
# ---------------------------------------------------------------------------

def _make_mt5_store(
    tmp_path: Path,
    symbol: str = "NDX",
    years: tuple[int, ...] = (2024,),
    bars_per_year: int = 120,
) -> Path:
    """Create synthetic M1 bars under ``{root}/{symbol}/bars_M1/year={Y}/part.parquet``.

    OHLC invariants guaranteed: low <= min(open, close), high >= max(open, close).
    """
    root = tmp_path / "mt5_data"
    for year in years:
        year_dir = root / symbol / "bars_M1" / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        _write_synthetic_bars(year_dir / "part.parquet", year=year, n=bars_per_year)
    return root


def _write_synthetic_bars(path: Path, year: int, n: int, seed_offset: int = 0) -> None:
    """Write ``n`` synthetic M1 bars to *path* with valid OHLC relationships."""
    rng = np.random.default_rng(year + seed_offset)
    times = pd.date_range(
        f"{year}-01-02 01:00:00", periods=n, freq="1min", tz="UTC"
    )
    close = 20_000.0 + np.clip(
        rng.uniform(-5, 5, n).cumsum(), -2_000, 2_000
    )
    open_ = close + rng.uniform(-2, 2, n)
    # Ensure high >= max(open, close) and low <= min(open, close).
    high = np.maximum(open_, close) + rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, n)
    pd.DataFrame({
        "time": times,
        "open": open_.astype("float64"),
        "high": high.astype("float64"),
        "low": low.astype("float64"),
        "close": close.astype("float64"),
        "tick_volume": rng.integers(10, 100, n).astype("int32"),
        "spread": rng.integers(3, 8, n).astype("int16"),
        "real_volume": np.zeros(n, dtype="int64"),
    }).to_parquet(path, index=False)


def _make_instrument_catalog(
    symbol: str = "NDX",
    venue: str = "XNAS",
    price_precision: int = 1,
    price_increment: float = 0.1,
):
    """Return an InstrumentCatalog with one synthetic CFD instrument."""
    from data_platform.core.catalog import InstrumentCatalog
    from data_platform.core.enums import AssetClass, InstrumentClass
    from data_platform.core.identifiers import InstrumentId
    from data_platform.core.instruments import Instrument

    inst = Instrument(
        id=InstrumentId.from_str(f"{symbol}.{venue}"),
        raw_symbol=symbol,
        asset_class=AssetClass.INDEX,
        instrument_class=InstrumentClass.CFD,
        price_precision=price_precision,
        price_increment=price_increment,
        data_source="mt5",
    )
    cat = InstrumentCatalog()
    cat.add(inst)
    return cat


# ---------------------------------------------------------------------------
# 1. Materialize → catalog has bars
# ---------------------------------------------------------------------------

def test_materialize_bars_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """materialize_symbol writes M1 bars into the catalog."""
    import data_platform.nautilus.ingest as _ingest
    import data_platform.core.catalog as _core_cat

    mt5_root = _make_mt5_store(tmp_path, symbol="NDX", years=(2024,), bars_per_year=120)
    monkeypatch.setattr(_ingest, "mt5_data_root", lambda: mt5_root)
    monkeypatch.setattr("data_platform.nautilus.materialize.mt5_data_root", lambda: mt5_root)

    inst_cat = _make_instrument_catalog(price_precision=1, price_increment=0.1)
    monkeypatch.setattr(_core_cat, "load_catalog", lambda: inst_cat)

    catalog = get_catalog(tmp_path / "catalog")
    from data_platform.nautilus.materialize import materialize_symbol

    result = materialize_symbol("NDX", catalog, from_year=2024)

    assert not result["skipped"], f"Unexpected skip: {result.get('reason')}"
    assert result["bars_written"] == 120
    assert result["quotes_written"] == 120

    bar_type = "NDX.XNAS-1-MINUTE-LAST-EXTERNAL"
    bars = catalog.bars(bar_types=[bar_type])
    assert len(bars) == 120
    # All bars have the fixed precision.
    assert all(b.close.precision == 1 for b in bars)
    # All prices are positive.
    assert all(float(b.close) > 0 for b in bars)


# ---------------------------------------------------------------------------
# 2. Idempotency: second materialize call skips the symbol
# ---------------------------------------------------------------------------

def test_materialize_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second materialize_symbol call for the same symbol skips it."""
    import data_platform.nautilus.ingest as _ingest
    import data_platform.core.catalog as _core_cat

    mt5_root = _make_mt5_store(tmp_path, symbol="NDX", years=(2024,), bars_per_year=60)
    monkeypatch.setattr(_ingest, "mt5_data_root", lambda: mt5_root)
    monkeypatch.setattr("data_platform.nautilus.materialize.mt5_data_root", lambda: mt5_root)

    inst_cat = _make_instrument_catalog(price_precision=1, price_increment=0.1)
    monkeypatch.setattr(_core_cat, "load_catalog", lambda: inst_cat)

    catalog = get_catalog(tmp_path / "catalog")
    from data_platform.nautilus.materialize import materialize_symbol

    # First run — should ingest
    r1 = materialize_symbol("NDX", catalog, from_year=2024)
    assert not r1["skipped"]
    assert r1["bars_written"] == 60

    # Second run — same symbol, same catalog → skip
    r2 = materialize_symbol("NDX", catalog, from_year=2024)
    assert r2["skipped"]
    assert r2.get("reason") == "already in catalog"
    assert r2["bars_written"] == 0

    # Catalog still has exactly the original 60 bars (no duplicates)
    bar_type = "NDX.XNAS-1-MINUTE-LAST-EXTERNAL"
    bars = catalog.bars(bar_types=[bar_type])
    assert len(bars) == 60


# ---------------------------------------------------------------------------
# 3. Append: only newer bars are ingested
# ---------------------------------------------------------------------------

def test_append_ingests_only_newer_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """append_symbol adds only bars after the catalog's current coverage end."""
    import data_platform.nautilus.ingest as _ingest
    import data_platform.core.catalog as _core_cat

    # Build a 2-year MT5 store (2024: 60 bars, 2025: 60 bars) using the OHLC helper.
    mt5_root = tmp_path / "mt5_data"
    yr24_dir = mt5_root / "NDX" / "bars_M1" / "year=2024"
    yr25_dir = mt5_root / "NDX" / "bars_M1" / "year=2025"
    yr24_dir.mkdir(parents=True, exist_ok=True)
    yr25_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_bars(yr24_dir / "part.parquet", year=2024, n=60)
    _write_synthetic_bars(yr25_dir / "part.parquet", year=2025, n=60)

    monkeypatch.setattr(_ingest, "mt5_data_root", lambda: mt5_root)
    monkeypatch.setattr(
        "data_platform.nautilus.materialize.mt5_data_root", lambda: mt5_root
    )

    inst_cat = _make_instrument_catalog(price_precision=1, price_increment=0.1)
    monkeypatch.setattr(_core_cat, "load_catalog", lambda: inst_cat)

    catalog = get_catalog(tmp_path / "catalog")
    from data_platform.nautilus.materialize import materialize_symbol, append_symbol

    # Initial materialize: ingest all available data (2024 + 2025 = 120 bars).
    r1 = materialize_symbol("NDX", catalog, from_year=2024)
    assert not r1["skipped"]
    assert r1["bars_written"] == 120

    bar_type = "NDX.XNAS-1-MINUTE-LAST-EXTERNAL"
    bars_before = catalog.bars(bar_types=[bar_type])
    assert len(bars_before) == 120

    # Simulate 10 new M1 bars scraped today: extend the 2025 partition.
    rng = np.random.default_rng(9999)
    existing_2025 = pd.read_parquet(yr25_dir / "part.parquet")
    last_time = existing_2025["time"].iloc[-1]
    if not isinstance(last_time, pd.Timestamp):
        last_time = pd.Timestamp(last_time)
    extra_times = pd.date_range(
        last_time + pd.Timedelta(minutes=1), periods=10, freq="1min", tz="UTC"
    )
    close_extra = 21_000.0 + rng.uniform(-2, 2, 10).cumsum()
    open_extra = close_extra + rng.uniform(-1, 1, 10)
    high_extra = np.maximum(open_extra, close_extra) + rng.uniform(0, 0.5, 10)
    low_extra = np.minimum(open_extra, close_extra) - rng.uniform(0, 0.5, 10)
    df_extra = pd.DataFrame({
        "time": extra_times,
        "open": open_extra.astype("float64"),
        "high": high_extra.astype("float64"),
        "low": low_extra.astype("float64"),
        "close": close_extra.astype("float64"),
        "tick_volume": rng.integers(10, 100, 10).astype("int32"),
        "spread": rng.integers(3, 8, 10).astype("int16"),
        "real_volume": np.zeros(10, dtype="int64"),
    })
    pd.concat([existing_2025, df_extra], ignore_index=True).to_parquet(
        yr25_dir / "part.parquet", index=False
    )

    # Append should only ingest the 10 new bars (strictly after coverage_end).
    r2 = append_symbol("NDX", catalog)
    assert not r2["skipped"], f"Unexpected skip in append: {r2.get('reason')}"
    assert r2["bars_written"] == 10

    bars_after = catalog.bars(bar_types=[bar_type])
    assert len(bars_after) == 120 + 10


# ---------------------------------------------------------------------------
# 4. Fixed increments round-trip into the catalog instrument
# ---------------------------------------------------------------------------

def test_instruments_carry_fixed_increments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Instruments written to the catalog carry price_precision=1 / price_increment=0.1.

    This verifies the Step-1 fix (NDX stand-in) — after write_instruments_to_catalog,
    the instrument read back from the catalog has the corrected values.
    """
    import data_platform.core.catalog as _core_cat

    inst_cat = _make_instrument_catalog(price_precision=1, price_increment=0.1)
    monkeypatch.setattr(_core_cat, "load_catalog", lambda: inst_cat)
    # instruments.py imports load_catalog at module-level; patch its own namespace
    # too so build_all_nautilus_instruments() sees the synthetic 1-instrument catalog.
    monkeypatch.setattr("data_platform.nautilus.instruments.load_catalog", lambda: inst_cat)

    catalog = get_catalog(tmp_path / "catalog")
    from data_platform.nautilus.instruments import write_instruments_to_catalog

    written = write_instruments_to_catalog(catalog)
    assert len(written) == 1

    read_back = catalog.instruments()
    assert len(read_back) == 1

    inst = read_back[0]
    # price_precision=1 and price_increment=0.1 must survive the catalog round-trip.
    assert inst.price_precision == 1
    assert float(inst.price_increment) == pytest.approx(0.1, abs=1e-9)
