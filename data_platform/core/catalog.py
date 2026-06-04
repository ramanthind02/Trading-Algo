"""Instrument catalog: the single registry resolving InstrumentId -> Instrument.

Stored at ``data/instruments/catalog.parquet`` (+ a human-readable
``catalog.json``). One row per instrument across all asset classes and all
data sources (norgate futures, norgate stocks, ib, mt5). This is the metadata
layer that sits on top of the existing parquet bar stores — bar data stays in
``data/ohlc_data/{TICKER}/`` (futures) and ``data/stock_data/{SYMBOL}/``
(stocks); the catalog tells consumers each instrument's class, venue,
precision, increments, multiplier, and which adjusted series exist.

When we migrate to NautilusTrader, this catalog is what feeds
``ParquetDataCatalog.write_data([...instruments...])`` — each row maps to the
matching nautilus instrument type by ``instrument_class``.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .enums import AssetClass, InstrumentClass
from .identifiers import InstrumentId
from .instruments import Instrument

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[1],
    )


def catalog_dir() -> Path:
    return _repo_root() / "data" / "instruments"


def catalog_parquet_path() -> Path:
    return catalog_dir() / "catalog.parquet"


def catalog_json_path() -> Path:
    return catalog_dir() / "catalog.json"


def _instrument_to_row(inst: Instrument) -> dict:
    row = asdict(inst)
    row["id"] = str(inst.id)
    row["asset_class"] = inst.asset_class.value
    row["instrument_class"] = inst.instrument_class.value
    row["activation"] = inst.activation.isoformat() if inst.activation else None
    row["expiration"] = inst.expiration.isoformat() if inst.expiration else None
    row["price_adjustments"] = json.dumps(inst.price_adjustments)
    row["info"] = json.dumps(inst.info, default=str)
    return row


def _row_to_instrument(row: dict) -> Instrument:
    def _as_date(v: object) -> date | None:
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            return None
        return date.fromisoformat(str(v))

    return Instrument(
        id=InstrumentId.from_str(str(row["id"])),
        raw_symbol=str(row["raw_symbol"]),
        asset_class=AssetClass(str(row["asset_class"])),
        instrument_class=InstrumentClass(str(row["instrument_class"])),
        price_precision=int(row["price_precision"]),
        price_increment=float(row["price_increment"]),
        size_precision=int(row.get("size_precision", 0) or 0),
        size_increment=float(row.get("size_increment", 1.0) or 1.0),
        quote_currency=str(row.get("quote_currency", "USD")),
        multiplier=float(row.get("multiplier", 1.0) or 1.0),
        lot_size=float(row.get("lot_size", 1.0) or 1.0),
        margin_init=_opt_float(row.get("margin_init")),
        margin_maint=_opt_float(row.get("margin_maint")),
        maker_fee=float(row.get("maker_fee", 0.0) or 0.0),
        taker_fee=float(row.get("taker_fee", 0.0) or 0.0),
        underlying=_opt_str(row.get("underlying")),
        activation=_as_date(row.get("activation")),
        expiration=_as_date(row.get("expiration")),
        venue_name=_opt_str(row.get("venue_name")),
        description=_opt_str(row.get("description")),
        data_source=_opt_str(row.get("data_source")),
        price_adjustments=json.loads(row.get("price_adjustments") or "{}"),
        info=json.loads(row.get("info") or "{}"),
    )


def _opt_float(v: object) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return float(v)


def _opt_str(v: object) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    return str(v)


class InstrumentCatalog:
    """Read/write registry of instruments, persisted to parquet + json."""

    def __init__(self, instruments: dict[str, Instrument] | None = None) -> None:
        self._by_id: dict[str, Instrument] = instruments or {}

    # ── lookups ──────────────────────────────────────────────────────────
    def find(self, instrument_id: InstrumentId | str) -> Instrument | None:
        key = str(instrument_id)
        return self._by_id.get(key)

    def __contains__(self, instrument_id: InstrumentId | str) -> bool:
        return str(instrument_id) in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def all(self) -> list[Instrument]:
        return list(self._by_id.values())

    def filter(self, *, asset_class: AssetClass | None = None,
               instrument_class: InstrumentClass | None = None,
               data_source: str | None = None) -> list[Instrument]:
        out = self._by_id.values()
        if asset_class is not None:
            out = [i for i in out if i.asset_class == asset_class]
        if instrument_class is not None:
            out = [i for i in out if i.instrument_class == instrument_class]
        if data_source is not None:
            out = [i for i in out if i.data_source == data_source]
        return list(out)

    # ── mutation ─────────────────────────────────────────────────────────
    def add(self, instrument: Instrument) -> None:
        self._by_id[str(instrument.id)] = instrument

    def add_many(self, instruments: list[Instrument]) -> None:
        for inst in instruments:
            self.add(inst)

    # ── persistence ──────────────────────────────────────────────────────
    def save(self) -> None:
        catalog_dir().mkdir(parents=True, exist_ok=True)
        rows = [_instrument_to_row(i) for i in self._by_id.values()]
        df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            catalog_parquet_path(),
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
        )
        with open(catalog_json_path(), "w") as f:
            json.dump(rows, f, indent=2, default=str)

    @classmethod
    def load(cls) -> "InstrumentCatalog":
        path = catalog_parquet_path()
        if not path.exists():
            return cls({})
        df = pd.read_parquet(path)
        by_id: dict[str, Instrument] = {}
        for row in df.to_dict("records"):
            inst = _row_to_instrument(row)
            by_id[str(inst.id)] = inst
        return cls(by_id)


def load_catalog() -> InstrumentCatalog:
    """Convenience: load the on-disk catalog (empty if not yet built)."""
    return InstrumentCatalog.load()
