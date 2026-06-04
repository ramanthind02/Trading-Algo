"""Cross-source symbol resolution over the InstrumentCatalog.

The same economic instrument has different native symbols per source:
  ES.XCME  ->  norgate &ES_CCB, ib CONTFUT "ES"@CME, mt5 "US500.cash"

Canonical identity is the catalog ``InstrumentId``; each instrument carries an
``info["source_symbols"]`` dict (populated by the catalog seeders). These helpers
resolve in both directions so adapters never hardcode per-source symbol tables.
"""
from __future__ import annotations

from .catalog import InstrumentCatalog
from .identifiers import InstrumentId


def source_symbol(
    catalog: InstrumentCatalog,
    instrument_id: InstrumentId | str,
    source: str,
) -> str | None:
    """Native symbol for ``instrument_id`` at ``source`` (e.g. 'norgate_adj',
    'ib_contfut', 'mt5'), or None if the instrument or key is absent."""
    inst = catalog.find(instrument_id)
    if inst is None:
        return None
    value = inst.info.get("source_symbols", {}).get(source)
    return str(value) if value is not None else None


def instrument_from_source_symbol(
    catalog: InstrumentCatalog,
    source: str,
    native_symbol: str,
) -> InstrumentId | None:
    """Reverse lookup: a source's native symbol -> canonical InstrumentId."""
    for inst in catalog.all():
        if inst.info.get("source_symbols", {}).get(source) == native_symbol:
            return inst.id
    return None
