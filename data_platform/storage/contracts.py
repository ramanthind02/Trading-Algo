"""Schema contracts for the parquet stores — one enforced schema per store.

This module is the single declaration of what each bulk store's parquet schema
IS, replacing the per-writer copies that drifted (`scraper._BARS_SCHEMA` vs
`daily_scraper._D1_SCHEMA` wrote the same logical row with different dtypes,
and every Norgate `_normalize` docstring promised ``date32`` while
``Table.from_pandas`` persisted ``timestamp[ns]``).

Writers validate against these contracts at write time via the gate functions
below; readers keep coercing through ``loaders._normalize_loaded_frame``
exactly as before. Pure module — no I/O.

Decisions of record (docs/library/Data/data_platform_migration_plan.md):
- ADR-2: MT5 timestamps are broker EET/EEST wall-clock stored with a UTC label.
  That contract is stamped machine-readably into the schema metadata
  (``timezone=broker_eet_as_utc``) instead of living only in
  docs/library/Data/mt5_timezones.md.
- ADR-6 (amended 2026-06-09, M0.3): M1 uses ``MT5_BARS_SCHEMA`` (int32
  tick_volume / int16 spread); D1 gets its own ``MT5_D1_BARS_SCHEMA`` (int64
  tick_volume / int16 spread).  10 US-stock D1 files carry daily volumes up to
  20.3B (BAC) which can never fit int32; per-minute M1 volumes are verified
  ~100× under int32 headroom.  The drift fix is the contract being explicit, not
  the widths being identical.
"""
from __future__ import annotations

from enum import Enum

import pyarrow as pa
import pyarrow.compute as pc

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3

#: ADR-2 stamp — MT5 timestamps are broker wall-clock mislabelled UTC.
TIMEZONE_METADATA_KEY = b"timezone"
BROKER_EET_AS_UTC = b"broker_eet_as_utc"
_MT5_METADATA: dict[bytes, bytes] = {TIMEZONE_METADATA_KEY: BROKER_EET_AS_UTC}

_SPREAD_INT16_MAX = 32_767  # silent overflow would corrupt spread; assert loudly

#: Canonical MT5 bar row — M1 only (ADR-6 amended). Matches scraper._BARS_SCHEMA.
#: Per-minute volumes are verified ~100× under int32 headroom, so int32 suffices.
MT5_BARS_SCHEMA = pa.schema(
    [
        ("time", pa.timestamp("s", tz="UTC")),  # broker wall-clock, see metadata
        ("open", pa.float32()),
        ("high", pa.float32()),
        ("low", pa.float32()),
        ("close", pa.float32()),
        ("tick_volume", pa.int32()),
        ("spread", pa.int16()),
        ("real_volume", pa.int64()),
    ],
    metadata=_MT5_METADATA,
)

#: Canonical MT5 bar row — D1 only (ADR-6 amended 2026-06-09, M0.3).
#: tick_volume is int64 because daily stock-CFD volume exceeds int32 (BAC observed
#: 20.3B on a single day); per-minute M1 volume does not — see migration plan.
MT5_D1_BARS_SCHEMA = pa.schema(
    [
        ("time", pa.timestamp("s", tz="UTC")),  # broker wall-clock, see metadata
        ("open", pa.float32()),
        ("high", pa.float32()),
        ("low", pa.float32()),
        ("close", pa.float32()),
        ("tick_volume", pa.int64()),  # int64: daily stock-CFD volume exceeds int32
        ("spread", pa.int16()),
        ("real_volume", pa.int64()),
    ],
    metadata=_MT5_METADATA,
)

#: Canonical MT5 tick row — bulk ticks, tick_cache chunks, rollover windows.
MT5_TICKS_SCHEMA = pa.schema(
    [
        ("time_msc", pa.int64()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("last", pa.float64()),
        ("volume", pa.int64()),
        ("time", pa.timestamp("s", tz="UTC")),  # broker wall-clock, see metadata
        ("flags", pa.int32()),
    ],
    metadata=_MT5_METADATA,
)

#: Canonical Norgate daily bar row — ohlc_data, stock_data, contract archive,
#: continuous archive. A REAL date32, not the timestamp[ns] the old writers
#: emitted (readers coerce either form, so the fix is read-compatible).
NORGATE_BAR_SCHEMA = pa.schema(
    [
        ("date", pa.date32()),
        ("open", pa.float32()),
        ("high", pa.float32()),
        ("low", pa.float32()),
        ("close", pa.float32()),
        ("volume", pa.int32()),
    ]
)


class NorgateSeriesKind(Enum):
    """Whether a Norgate bar frame contains prices or back-adjustment coefficients.

    Ratio-adjustment (``*_ratio.parquet``) files store per-field multiplicative
    coefficients, not prices.  OHLC ordering does not apply to coefficients (e.g.
    near a roll event the coefficient ``high`` can be numerically less than ``low``).
    Pass ``kind=NorgateSeriesKind.ADJUSTMENT`` to skip the OHLC-sanity gate while
    keeping schema equality, sorted-unique dates, and the no-NaN-close checks.
    """

    PRICE = "price"
    ADJUSTMENT = "adjustment"


class SchemaContractError(ValueError):
    """A table violates its store's schema contract or data invariants."""


def fields_equal(actual: pa.Schema, contract: pa.Schema) -> bool:
    """Field-level equality (names, types, order) — metadata excluded."""
    return actual.remove_metadata().equals(contract.remove_metadata())


def require_schema(table: pa.Table, contract: pa.Schema, *, store: str) -> None:
    """Raise unless ``table``'s fields exactly match the contract."""
    if not fields_equal(table.schema, contract):
        raise SchemaContractError(
            f"{store}: schema does not match contract.\n"
            f"  actual:   {table.schema.remove_metadata()}\n"
            f"  contract: {contract.remove_metadata()}"
        )


def stamp_metadata(table: pa.Table, contract: pa.Schema) -> pa.Table:
    """Merge the contract's metadata (e.g. the ADR-2 timezone stamp) onto a table."""
    merged = {**(table.schema.metadata or {}), **(contract.metadata or {})}
    return table.replace_schema_metadata(merged)


def _require(condition: bool, store: str, message: str) -> None:
    if not condition:
        raise SchemaContractError(f"{store}: {message}")


def _check_sorted_unique(table: pa.Table, column: str, store: str) -> None:
    values = table.column(column)
    if len(values) <= 1:
        return
    diffs = pc.pairwise_diff(values.combine_chunks())
    valid = diffs.drop_null()  # first element of pairwise_diff is null
    zero = pa.scalar(0, type=valid.type)  # timestamp diffs are duration-typed
    _require(
        bool(pc.all(pc.greater(valid, zero)).as_py()),
        store,
        f"'{column}' must be strictly increasing and unique",
    )


def _check_ohlc_sanity(table: pa.Table, store: str) -> None:
    low, high = table.column("low"), table.column("high")
    for name in ("open", "close", "high"):
        _require(
            not bool(pc.any(pc.less(table.column(name), low)).as_py()),
            store,
            f"'{name}' below 'low' — OHLC sanity violated",
        )
    for name in ("open", "close"):
        _require(
            not bool(pc.any(pc.greater(table.column(name), high)).as_py()),
            store,
            f"'{name}' above 'high' — OHLC sanity violated",
        )
    close = table.column("close")
    _require(close.null_count == 0, store, "'close' contains nulls")
    _require(
        not bool(pc.any(pc.is_nan(close)).as_py()),
        store,
        "'close' contains NaN",
    )


def check_spread_fits_int16(values: pa.ChunkedArray | pa.Array, *, store: str) -> None:
    """Loud guard against silent int16 truncation of wide spreads (ADR-6)."""
    if len(values) == 0:
        return
    as_int64 = values.cast(pa.int64()) if values.type != pa.int64() else values
    _require(
        bool(pc.all(pc.less_equal(as_int64, _SPREAD_INT16_MAX)).as_py()),
        store,
        f"'spread' exceeds int16 max ({_SPREAD_INT16_MAX} points)",
    )
    _require(
        bool(pc.all(pc.greater_equal(as_int64, 0)).as_py()),
        store,
        "'spread' is negative",
    )


def validate_mt5_bars(table: pa.Table, *, store: str = "mt5_bars") -> None:
    """Write gate for the MT5 M1 bar store (int32 tick_volume / int16 spread)."""
    require_schema(table, MT5_BARS_SCHEMA, store=store)
    _check_sorted_unique(table, "time", store)
    _check_ohlc_sanity(table, store)
    check_spread_fits_int16(table.column("spread"), store=store)


def validate_mt5_d1_bars(table: pa.Table, *, store: str = "mt5_d1_bars") -> None:
    """Write gate for the MT5 D1 bar store (int64 tick_volume / int16 spread).

    Identical invariants to validate_mt5_bars but validates against MT5_D1_BARS_SCHEMA
    so that daily stock-CFD volumes exceeding int32 (e.g. BAC at 20.3B) are accepted.
    """
    require_schema(table, MT5_D1_BARS_SCHEMA, store=store)
    _check_sorted_unique(table, "time", store)
    _check_ohlc_sanity(table, store)
    check_spread_fits_int16(table.column("spread"), store=store)


def validate_mt5_ticks(table: pa.Table, *, store: str = "mt5_ticks") -> None:
    """Write gate for the MT5 tick stores (bulk, cache chunks, rollover)."""
    require_schema(table, MT5_TICKS_SCHEMA, store=store)
    _check_sorted_unique(table, "time_msc", store)
    for name in ("bid", "ask"):
        column = table.column(name)
        _require(column.null_count == 0, store, f"'{name}' contains nulls")
        _require(
            not bool(pc.any(pc.less(column, 0)).as_py()),
            store,
            f"'{name}' is negative",
        )


def validate_norgate_bars(
    table: pa.Table,
    *,
    store: str = "norgate_bars",
    kind: NorgateSeriesKind = NorgateSeriesKind.PRICE,
) -> None:
    """Write gate for the Norgate daily bar stores.

    Pass ``kind=NorgateSeriesKind.ADJUSTMENT`` for ratio/coefficient frames where
    OHLC ordering does not apply; all other invariants (schema, sorted-unique dates,
    no-NaN close, non-negative volume) still run.
    """
    require_schema(table, NORGATE_BAR_SCHEMA, store=store)
    _check_sorted_unique(table, "date", store)
    if kind is NorgateSeriesKind.PRICE:
        _check_ohlc_sanity(table, store)
    volume = table.column("volume")
    _require(
        not bool(pc.any(pc.less(volume, 0)).as_py()),
        store,
        "'volume' is negative",
    )
