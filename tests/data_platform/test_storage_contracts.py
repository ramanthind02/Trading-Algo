"""Schema contracts: writer schemas match the canonical contracts, and the
validate-on-write gates catch the invariant violations they claim to.

The writer-equality tests are the Phase-0 drift detector from
docs/library/Data/data_platform_migration_plan.md §3 — `_D1_SCHEMA` was xfail
until M0.3 (ADR-6); it is now unified onto `MT5_BARS_SCHEMA` and passes.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from data_platform.providers.mt5 import daily_scraper, rollover_tick_scraper, scraper
from data_platform.storage import contracts


# ---------------------------------------------------------------------------
# Writer schemas == contracts (the drift detector)
# ---------------------------------------------------------------------------


def test_scraper_bars_schema_matches_contract() -> None:
    assert contracts.fields_equal(scraper._BARS_SCHEMA, contracts.MT5_BARS_SCHEMA)


def test_scraper_ticks_schema_matches_contract() -> None:
    assert contracts.fields_equal(scraper._TICKS_SCHEMA, contracts.MT5_TICKS_SCHEMA)


def test_rollover_schema_matches_ticks_contract() -> None:
    assert contracts.fields_equal(
        rollover_tick_scraper._ROLLOVER_SCHEMA, contracts.MT5_TICKS_SCHEMA
    )


def test_daily_scraper_schema_matches_contract() -> None:
    assert contracts.fields_equal(daily_scraper._D1_SCHEMA, contracts.MT5_D1_BARS_SCHEMA)


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------


def _mt5_bars_table(**overrides: list) -> pa.Table:
    columns: dict[str, list] = {
        "time": [1_700_000_000, 1_700_000_060, 1_700_000_120],
        "open": [1.0, 2.0, 3.0],
        "high": [1.5, 2.5, 3.5],
        "low": [0.5, 1.5, 2.5],
        "close": [1.2, 2.2, 3.2],
        "tick_volume": [10, 20, 30],
        "spread": [2, 3, 4],
        "real_volume": [0, 0, 0],
    }
    columns.update(overrides)
    arrays = [
        pa.array(columns[field.name]).cast(field.type)
        for field in contracts.MT5_BARS_SCHEMA
    ]
    return pa.Table.from_arrays(arrays, schema=contracts.MT5_BARS_SCHEMA)


def _ticks_table(**overrides: list) -> pa.Table:
    columns: dict[str, list] = {
        "time_msc": [1_700_000_000_000, 1_700_000_000_500],
        "bid": [1.0999, 1.1000],
        "ask": [1.1001, 1.1002],
        "last": [0.0, 0.0],
        "volume": [0, 0],
        "time": [1_700_000_000, 1_700_000_000],
        "flags": [6, 6],
    }
    columns.update(overrides)
    arrays = [
        pa.array(columns[field.name]).cast(field.type)
        for field in contracts.MT5_TICKS_SCHEMA
    ]
    return pa.Table.from_arrays(arrays, schema=contracts.MT5_TICKS_SCHEMA)


def _norgate_table(**overrides: list) -> pa.Table:
    columns: dict[str, list] = {
        "date": [19_700, 19_701, 19_702],  # date32 = days since epoch
        "open": [10.0, 11.0, 12.0],
        "high": [10.5, 11.5, 12.5],
        "low": [9.5, 10.5, 11.5],
        "close": [10.2, 11.2, 12.2],
        "volume": [100, 200, 300],
    }
    columns.update(overrides)
    arrays = [
        pa.array(columns[field.name], type=pa.int32()).cast(field.type)
        if field.type == pa.date32()
        else pa.array(columns[field.name]).cast(field.type)
        for field in contracts.NORGATE_BAR_SCHEMA
    ]
    return pa.Table.from_arrays(arrays, schema=contracts.NORGATE_BAR_SCHEMA)


def test_valid_tables_pass_all_gates() -> None:
    contracts.validate_mt5_bars(_mt5_bars_table())
    contracts.validate_mt5_ticks(_ticks_table())
    contracts.validate_norgate_bars(_norgate_table())


def test_wrong_schema_rejected() -> None:
    table = _mt5_bars_table().drop_columns(["spread"])
    with pytest.raises(contracts.SchemaContractError, match="schema"):
        contracts.validate_mt5_bars(table)


def test_duplicate_time_rejected() -> None:
    table = _mt5_bars_table(time=[1_700_000_000, 1_700_000_000, 1_700_000_120])
    with pytest.raises(contracts.SchemaContractError, match="strictly increasing"):
        contracts.validate_mt5_bars(table)


def test_unsorted_time_rejected() -> None:
    table = _mt5_bars_table(time=[1_700_000_120, 1_700_000_060, 1_700_000_000])
    with pytest.raises(contracts.SchemaContractError, match="strictly increasing"):
        contracts.validate_mt5_bars(table)


def test_low_above_high_rejected() -> None:
    table = _mt5_bars_table(low=[2.0, 1.5, 2.5])  # first bar: low 2.0 > high 1.5? high=1.5
    with pytest.raises(contracts.SchemaContractError, match="OHLC sanity"):
        contracts.validate_mt5_bars(table)


def test_nan_close_rejected() -> None:
    table = _mt5_bars_table(close=[1.2, float("nan"), 3.2])
    with pytest.raises(contracts.SchemaContractError, match="close"):
        contracts.validate_mt5_bars(table)


def test_spread_overflow_rejected() -> None:
    values = pa.array([1, 40_000], type=pa.int64())
    with pytest.raises(contracts.SchemaContractError, match="int16"):
        contracts.check_spread_fits_int16(values, store="mt5_bars")


def test_negative_spread_rejected() -> None:
    values = pa.array([-1, 3], type=pa.int64())
    with pytest.raises(contracts.SchemaContractError, match="negative"):
        contracts.check_spread_fits_int16(values, store="mt5_bars")


def test_negative_bid_rejected() -> None:
    table = _ticks_table(bid=[-0.1, 1.1])
    with pytest.raises(contracts.SchemaContractError, match="negative"):
        contracts.validate_mt5_ticks(table)


def test_norgate_duplicate_date_rejected() -> None:
    table = _norgate_table(date=[19_700, 19_700, 19_702])
    with pytest.raises(contracts.SchemaContractError, match="strictly increasing"):
        contracts.validate_norgate_bars(table)


def test_timezone_stamp_present_on_mt5_contracts() -> None:
    for schema in (contracts.MT5_BARS_SCHEMA, contracts.MT5_D1_BARS_SCHEMA, contracts.MT5_TICKS_SCHEMA):
        assert schema.metadata[contracts.TIMEZONE_METADATA_KEY] == contracts.BROKER_EET_AS_UTC


def test_stamp_metadata_merges_contract_metadata() -> None:
    bare = _mt5_bars_table().replace_schema_metadata({b"existing": b"kept"})
    stamped = contracts.stamp_metadata(bare, contracts.MT5_BARS_SCHEMA)
    assert stamped.schema.metadata[b"existing"] == b"kept"
    assert stamped.schema.metadata[contracts.TIMEZONE_METADATA_KEY] == contracts.BROKER_EET_AS_UTC


def test_adjustment_series_skips_ohlc_sanity() -> None:
    """ADJUSTMENT kind: a table with high < low passes the gate; default PRICE kind rejects it."""
    # ratio coefficients where ordering is inverted (legitimate near a futures roll)
    table = _norgate_table(open=[0.85] * 3, high=[0.90] * 3, low=[0.95] * 3, close=[0.88] * 3)
    # PRICE (default) must reject the OHLC-ordering violation
    with pytest.raises(contracts.SchemaContractError, match="OHLC sanity"):
        contracts.validate_norgate_bars(table)
    # ADJUSTMENT must accept it — ordering invariant does not apply to coefficients
    contracts.validate_norgate_bars(table, kind=contracts.NorgateSeriesKind.ADJUSTMENT)


# ---------------------------------------------------------------------------
# D1 contract — int64 tick_volume gate (amended ADR-6)
# ---------------------------------------------------------------------------


def _mt5_d1_bars_table(**overrides: list) -> pa.Table:
    """Build a minimal valid MT5_D1_BARS_SCHEMA table (daily spacing, 3 rows)."""
    columns: dict[str, list] = {
        "time": [1_700_000_000, 1_700_086_400, 1_700_172_800],  # ~1-day apart
        "open": [1.0, 2.0, 3.0],
        "high": [1.5, 2.5, 3.5],
        "low": [0.5, 1.5, 2.5],
        "close": [1.2, 2.2, 3.2],
        "tick_volume": [10, 20, 30],
        "spread": [2, 3, 4],
        "real_volume": [0, 0, 0],
    }
    columns.update(overrides)
    arrays = [
        pa.array(columns[field.name]).cast(field.type)
        for field in contracts.MT5_D1_BARS_SCHEMA
    ]
    return pa.Table.from_arrays(arrays, schema=contracts.MT5_D1_BARS_SCHEMA)


def test_d1_large_tick_volume_passes_d1_gate() -> None:
    """D1 table with tick_volume > int32-max passes validate_mt5_d1_bars (amended ADR-6).

    BAC D1 volumes reach 20.3B on high-volume days — int32 max is ~2.1B, so any
    US-stock D1 file at peak volume must use the D1 contract's int64 tick_volume.
    """
    large_tv = 2_147_483_648  # int32-max + 1
    table = _mt5_d1_bars_table(
        tick_volume=[large_tv, large_tv * 2, large_tv * 3],
    )
    # Must not raise — int64 has ample headroom for daily stock volumes.
    contracts.validate_mt5_d1_bars(table)


def test_d1_large_tick_volume_rejected_by_m1_gate() -> None:
    """MT5_BARS_SCHEMA gate (M1 / int32 tick_volume) rejects a D1 table with int64 tick_volume.

    This confirms that the two contracts are genuinely distinct: a writer
    that accidentally routes D1 data through write_mt5_bars will be stopped
    at the schema-equality check before any data is written.
    """
    large_tv = 2_147_483_648
    table = _mt5_d1_bars_table(
        tick_volume=[large_tv, large_tv * 2, large_tv * 3],
    )
    with pytest.raises(contracts.SchemaContractError, match="schema"):
        contracts.validate_mt5_bars(table)
