"""Single-writer storage layer over the parquet stores (migration plan §3).

``contracts`` declares one enforced pyarrow schema per store; the entrypoints
here are the ONLY sanctioned way to write those stores. Every write:

1. converts/accepts a ``pyarrow.Table`` in the store's contract schema,
2. runs the validate-on-write gate (schema equality + data invariants),
3. stamps contract metadata (e.g. the ADR-2 broker-timezone marker), and
4. writes atomically (temp sibling + ``os.replace`` via ``lib.core.atomic_io``)
   so a crash mid-write never leaves a torn parquet.

Norgate note: the contract stores ``date`` as a REAL ``date32`` (the dtype every
docstring promised) instead of the ``timestamp[ns]`` the legacy ``_write``
bodies emitted. The read side (``loaders._normalize_loaded_frame``) accepts
both, so old and new files coexist; parity is asserted on loader output, not
raw bytes (plan §13).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_platform.storage.contracts import (
    MT5_BARS_SCHEMA,
    MT5_D1_BARS_SCHEMA,
    MT5_TICKS_SCHEMA,
    NORGATE_BAR_SCHEMA,
    NorgateSeriesKind,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    SchemaContractError,
    stamp_metadata,
    validate_mt5_bars,
    validate_mt5_d1_bars,
    validate_mt5_ticks,
    validate_norgate_bars,
)
from lib.core.atomic_io import atomic_write

__all__ = [
    "MT5_BARS_SCHEMA",
    "MT5_D1_BARS_SCHEMA",
    "MT5_TICKS_SCHEMA",
    "NORGATE_BAR_SCHEMA",
    "NorgateSeriesKind",
    "SchemaContractError",
    "norgate_bars_table",
    "validate_mt5_bars",
    "validate_mt5_d1_bars",
    "validate_mt5_ticks",
    "validate_norgate_bars",
    "write_mt5_bars",
    "write_mt5_d1_bars",
    "write_mt5_ticks",
    "write_norgate_bars",
]


def _write_atomic(
    table: pa.Table, path: Path, *, compression_level: int | None = None
) -> None:
    atomic_write(
        path,
        lambda tmp: pq.write_table(
            table, tmp, compression=PARQUET_COMPRESSION, compression_level=compression_level
        ),
    )


def norgate_bars_table(frame: pd.DataFrame) -> pa.Table:
    """Convert a normalized Norgate daily frame (date index, OHLCV) to the contract.

    The cast to ``date32`` requires midnight-normalized dates — a non-midnight
    timestamp raises ``SchemaContractError``, which is itself a contract
    violation worth surfacing loudly.
    """
    index = pd.DatetimeIndex(frame.index)
    if not (index == index.normalize()).all():
        raise SchemaContractError(
            "norgate_bars: cast to date32 requires midnight-normalized dates"
        )
    table = pa.Table.from_pandas(frame, preserve_index=True)
    missing = [name for name in NORGATE_BAR_SCHEMA.names if name not in table.column_names]
    if missing:
        raise SchemaContractError(f"norgate_bars: missing columns {missing}")
    table = table.select(NORGATE_BAR_SCHEMA.names)
    try:
        return table.cast(NORGATE_BAR_SCHEMA)
    except pa.ArrowInvalid as exc:
        raise SchemaContractError(f"norgate_bars: cast to contract failed — {exc}") from exc


def write_norgate_bars(
    frame: pd.DataFrame,
    path: Path,
    *,
    store: str = "norgate_bars",
    extra_metadata: dict[bytes, bytes] | None = None,
    kind: NorgateSeriesKind = NorgateSeriesKind.PRICE,
) -> None:
    """THE writer for Norgate daily bar stores (ohlc_data, stock_data, archives)."""
    table = norgate_bars_table(frame)
    if extra_metadata:
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), **extra_metadata}
        )
    validate_norgate_bars(table, store=store, kind=kind)
    _write_atomic(table, path, compression_level=PARQUET_COMPRESSION_LEVEL)


def write_mt5_bars(table: pa.Table, path: Path, *, store: str = "mt5_bars") -> None:
    """THE writer for the MT5 M1 bar partitions (int32 tick_volume / int16 spread)."""
    table = stamp_metadata(table, MT5_BARS_SCHEMA)
    validate_mt5_bars(table, store=store)
    _write_atomic(table, path)


def write_mt5_d1_bars(table: pa.Table, path: Path, *, store: str = "mt5_d1_bars") -> None:
    """THE writer for the MT5 D1 bar partitions (int64 tick_volume / int16 spread).

    Uses MT5_D1_BARS_SCHEMA so that daily stock-CFD volumes exceeding int32 are
    accepted without silent truncation (amended ADR-6, see contracts.py).
    """
    table = stamp_metadata(table, MT5_D1_BARS_SCHEMA)
    validate_mt5_d1_bars(table, store=store)
    _write_atomic(table, path)


def write_mt5_ticks(table: pa.Table, path: Path, *, store: str = "mt5_ticks") -> None:
    """THE writer for the MT5 tick partitions (bulk, cache chunks, rollover windows)."""
    table = stamp_metadata(table, MT5_TICKS_SCHEMA)
    validate_mt5_ticks(table, store=store)
    _write_atomic(table, path)
