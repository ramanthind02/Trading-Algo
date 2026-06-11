"""Single-writer entrypoints: atomic writes, validate-on-write, and — the
load-bearing case — date32 output reading back identically to the legacy
timestamp[ns] files through ``loaders._normalize_loaded_frame`` on BOTH
parquet engines (ohlc_data is read with fastparquet, stocks with pyarrow).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_platform import storage
from data_platform.loaders import _normalize_loaded_frame
from data_platform.storage import contracts

_START, _END = datetime(1990, 1, 1), datetime(2099, 12, 31)


def _norgate_frame() -> pd.DataFrame:
    """A frame exactly as the legacy `_normalize` bodies produce it."""
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "volume": [100, 200, 300],
        },
        index=index,
    )
    frame.index.name = "date"
    return frame.astype(
        {c: "float32" for c in ("open", "high", "low", "close")} | {"volume": "int32"}
    )


def _legacy_write(frame: pd.DataFrame, path: Path) -> None:
    """Byte pattern of the old `_write` bodies: from_pandas, timestamp[ns] date."""
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=True),
        path,
        compression="zstd",
        compression_level=3,
    )


def test_norgate_writer_emits_real_date32(tmp_path: Path) -> None:
    target = tmp_path / "D_ES.parquet"
    storage.write_norgate_bars(_norgate_frame(), target)
    schema = pq.read_schema(target)
    assert schema.field("date").type == pa.date32()


@pytest.mark.parametrize("engine", ["pyarrow", "fastparquet"])
def test_date32_reads_back_identical_to_legacy(tmp_path: Path, engine: str) -> None:
    frame = _norgate_frame()
    legacy_path, new_path = tmp_path / "legacy.parquet", tmp_path / "new.parquet"
    _legacy_write(frame, legacy_path)
    storage.write_norgate_bars(frame, new_path)

    legacy = _normalize_loaded_frame(pd.read_parquet(legacy_path, engine=engine), _START, _END)
    fresh = _normalize_loaded_frame(pd.read_parquet(new_path, engine=engine), _START, _END)
    pd.testing.assert_frame_equal(fresh, legacy)


def test_failed_validation_leaves_existing_file_untouched(tmp_path: Path) -> None:
    target = tmp_path / "D_ES.parquet"
    storage.write_norgate_bars(_norgate_frame(), target)
    before = target.read_bytes()

    bad = _norgate_frame()
    bad.loc[bad.index[1], "low"] = 99.0  # low above high
    with pytest.raises(storage.SchemaContractError, match="OHLC sanity"):
        storage.write_norgate_bars(bad, target)

    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_column_rejected(tmp_path: Path) -> None:
    frame = _norgate_frame().drop(columns=["volume"])
    with pytest.raises(storage.SchemaContractError, match="missing columns"):
        storage.write_norgate_bars(frame, tmp_path / "x.parquet")


def test_non_midnight_date_rejected(tmp_path: Path) -> None:
    frame = _norgate_frame()
    frame.index = frame.index + pd.Timedelta(hours=9)
    with pytest.raises(storage.SchemaContractError, match="midnight"):
        storage.write_norgate_bars(frame, tmp_path / "x.parquet")


def test_extra_metadata_stamped(tmp_path: Path) -> None:
    target = tmp_path / "stock.parquet"
    storage.write_norgate_bars(
        _norgate_frame(), target, extra_metadata={b"norgate_raw_symbol": b"AGM.A"}
    )
    assert pq.read_schema(target).metadata[b"norgate_raw_symbol"] == b"AGM.A"


def _mt5_bars_table() -> pa.Table:
    columns = {
        "time": [1_700_000_000, 1_700_000_060],
        "open": [1.0, 2.0],
        "high": [1.5, 2.5],
        "low": [0.5, 1.5],
        "close": [1.2, 2.2],
        "tick_volume": [10, 20],
        "spread": [2, 3],
        "real_volume": [0, 0],
    }
    arrays = [
        pa.array(columns[field.name]).cast(field.type) for field in contracts.MT5_BARS_SCHEMA
    ]
    return pa.Table.from_arrays(arrays, schema=contracts.MT5_BARS_SCHEMA.remove_metadata())


def test_mt5_writer_stamps_timezone_metadata(tmp_path: Path) -> None:
    target = tmp_path / "part.parquet"
    storage.write_mt5_bars(_mt5_bars_table(), target)
    metadata = pq.read_schema(target).metadata
    assert metadata[contracts.TIMEZONE_METADATA_KEY] == contracts.BROKER_EET_AS_UTC
    # Parquet stores timestamp[s] as ms on disk, so compare after casting back to the contract.
    read_back = pq.read_table(target).cast(contracts.MT5_BARS_SCHEMA.remove_metadata())
    assert read_back.equals(_mt5_bars_table())
