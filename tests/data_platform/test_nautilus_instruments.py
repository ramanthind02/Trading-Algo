"""WP-2 Unit 1 — instrument round-trip into the Nautilus catalog.

Every on-disk ``InstrumentCatalog`` row maps to a valid Nautilus instrument,
writes to a temp ``ParquetDataCatalog``, and reads back with counts + ids
reconciling. The class distribution is asserted against the known catalog
state (69 rows: FuturesContract 23, Equity 30, CurrencyPair 7, Cfd 9).
"""
from __future__ import annotations

from collections import Counter

from nautilus_trader.model.instruments import (
    Cfd,
    CurrencyPair,
    Equity,
    FuturesContract,
)

from data_platform.nautilus.catalog import get_catalog
from data_platform.nautilus.instruments import (
    build_all_nautilus_instruments,
    write_instruments_to_catalog,
)

# Expected mapped class distribution for the 69-row on-disk catalog.
# FUTURE(23)->FuturesContract; SPOT+EQUITY(26) + WARRANT(4) -> Equity(30);
# SPOT+FX(7)->CurrencyPair; CFD(9)->Cfd (added XNGUSD + XTIUSD).
_EXPECTED_DISTRIBUTION = {
    "FuturesContract": 23,
    "Equity": 30,
    "CurrencyPair": 7,
    "Cfd": 9,
}
_EXPECTED_TOTAL = 69


def test_all_rows_map_with_no_skips() -> None:
    mapped, skipped = build_all_nautilus_instruments()
    assert skipped == [], f"unexpected skipped rows: {skipped}"
    assert len(mapped) == _EXPECTED_TOTAL


def test_class_distribution() -> None:
    mapped, _ = build_all_nautilus_instruments()
    dist = Counter(type(i).__name__ for i in mapped)
    assert dict(dist) == _EXPECTED_DISTRIBUTION, (
        f"class distribution mismatch: got {dict(dist)}, "
        f"expected {_EXPECTED_DISTRIBUTION}"
    )
    # Every mapped instrument is one of the four expected Nautilus types.
    assert all(
        isinstance(i, (FuturesContract, Equity, CurrencyPair, Cfd)) for i in mapped
    )


def test_round_trip_counts_and_ids_reconcile(tmp_path) -> None:
    mapped, skipped = build_all_nautilus_instruments()
    catalog = get_catalog(tmp_path / "catalog")
    written = write_instruments_to_catalog(catalog)

    # mapped count == written count.
    assert len(written) == len(mapped)
    assert len(written) == _EXPECTED_TOTAL

    read_back = catalog.instruments()
    # written count == read-back count.
    assert len(read_back) == len(written)

    written_ids = sorted(str(i.id) for i in written)
    read_ids = sorted(str(i.id) for i in read_back)
    # Every InstrumentId string round-trips exactly.
    assert written_ids == read_ids
    assert len(set(read_ids)) == len(read_ids)  # ids unique


def test_skips_reported_when_any() -> None:
    # On the current catalog there are no skips; if that changes this surfaces
    # the reasons rather than silently passing.
    _, skipped = build_all_nautilus_instruments()
    assert not skipped, f"skipped rows present: {[(s.instrument_id, s.reason) for s in skipped]}"
