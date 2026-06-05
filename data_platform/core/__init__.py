"""data_core — Nautilus-aligned instrument model and catalog.

Intermediate model that mirrors NautilusTrader's identifiers, instruments, and
BarType semantics without depending on ``nautilus_trader``. All adapters
(Norgate, IB, MT5) target this model; the instrument catalog is the single
registry resolving ``InstrumentId`` -> ``Instrument`` spec.

See data_core/README.md for the Nautilus migration mapping.
"""
from .bar_type import BarSpecification, BarType
from .catalog import InstrumentCatalog, catalog_dir, load_catalog
from .enums import (
    FALLBACK_MIC,
    NORGATE_EXCHANGE_TO_MIC,
    AggregationSource,
    AssetClass,
    BarAggregation,
    InstrumentClass,
    PriceAdjustment,
    PriceType,
    venue_mic_for_exchange,
)
from .bar_record import CanonicalBarRecord
from .identifiers import InstrumentId, Symbol, Venue
from .instruments import Instrument
from .reconciler import (
    ConflictRecord,
    ProvenanceRecord,
    ReconcileResult,
    SourcePriorityReconciler,
)
from .source_priority import PriorityRule, SourcePriorityConfig, SourceRule
from .symbol_map import instrument_from_source_symbol, source_symbol

__all__ = [
    # identifiers
    "Venue",
    "Symbol",
    "InstrumentId",
    # enums
    "AssetClass",
    "InstrumentClass",
    "PriceType",
    "BarAggregation",
    "AggregationSource",
    "PriceAdjustment",
    "NORGATE_EXCHANGE_TO_MIC",
    "FALLBACK_MIC",
    "venue_mic_for_exchange",
    # instruments
    "Instrument",
    # bar types
    "BarSpecification",
    "BarType",
    # catalog
    "InstrumentCatalog",
    "load_catalog",
    "catalog_dir",
    # symbol map
    "source_symbol",
    "instrument_from_source_symbol",
    # multi-source reconciliation
    "CanonicalBarRecord",
    "SourcePriorityConfig",
    "PriorityRule",
    "SourceRule",
    "SourcePriorityReconciler",
    "ReconcileResult",
    "ProvenanceRecord",
    "ConflictRecord",
]
