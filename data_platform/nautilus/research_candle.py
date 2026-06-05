"""Custom Nautilus data type for **byte-exact** research OHLC candles.

WP-2 Unit-2, Option B. The native Nautilus :class:`~nautilus_trader.model.data.Bar`
stores prices as :class:`~nautilus_trader.model.objects.Price`, which *quantizes*
to the instrument's ``price_precision``. Back-adjusted futures prices are not
tick-aligned, so a native-``Bar`` round-trip perturbs the raw values and breaks
the EXACT (``rtol=1e-8``) research-parity gate.

``ResearchCandle`` sidesteps that: it is a ``@customdataclass`` carrying raw
``float64`` OHLC and ``int64`` volume. The ``@customdataclass`` Arrow schema maps
Python ``float`` -> ``pa.float64()`` and ``int`` -> ``pa.int64()``, so values
persist to and query back from the :class:`ParquetDataCatalog` **bit-for-bit**
(verified: ``np.float32(x).astype(np.float64)`` survives the round-trip exactly).

Timestamp convention
--------------------
``ts_event == ts_init`` is the legacy **date-label** (the tz-naive ``datetime``
the homegrown loaders index by) expressed as unix nanoseconds:
``pd.Timestamp(date_label).value``. For a daily bar that is midnight of the bar
date. This is deliberately the *label*, not the bar close — the read-back adapter
reconstructs the exact ``datetime`` the legacy loaders return, so downstream
features see identical timestamps. (Native execution ``Bar``s, which use
``ts_init = close``, are a separate Unit-1b concern and are untouched.)

Instrument identity
-------------------
Each candle carries an :class:`InstrumentId` of the form
``{TICKER}.RESEARCH_{TF}`` (e.g. ``SI.RESEARCH_D``). The ``RESEARCH_*`` venue
namespaces the research candle store apart from the execution instruments/bars in
the same catalog, and folding the timeframe into the venue keeps ``D``/``W``/``M``
series for one ticker on distinct instrument ids so they never collide on
read-back within a single catalog.

NOTE: this module must **not** use ``from __future__ import annotations`` — the
``@customdataclass`` Arrow-schema builder introspects the live annotation objects
(``InstrumentId``/``float``/``int``); stringized (PEP 563) annotations break it.
"""

from nautilus_trader.core.data import Data
from nautilus_trader.model.custom import customdataclass
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue

#: Venue prefix used to namespace research candles away from execution instruments.
RESEARCH_VENUE_PREFIX = "RESEARCH"


def research_venue(timeframe_name: str) -> Venue:
    """The timeframe-scoped research venue, e.g. ``"D"`` -> ``RESEARCH_D``."""
    return Venue(f"{RESEARCH_VENUE_PREFIX}_{timeframe_name}")


def research_instrument_id(ticker_name: str, timeframe_name: str) -> InstrumentId:
    """Build the namespaced research :class:`InstrumentId`.

    e.g. ``("SI", "D")`` -> ``InstrumentId(SI.RESEARCH_D)``.
    """
    return InstrumentId(Symbol(ticker_name), research_venue(timeframe_name))


@customdataclass
class ResearchCandle(Data):
    """Raw-float64 research OHLC candle persisted losslessly in the catalog.

    Fields:
        instrument_id: namespaced ``{TICKER}.RESEARCH`` identifier.
        open/high/low/close: raw ``float64`` prices (no tick quantization).
        volume: ``int64`` volume.
        ts_event / ts_init: legacy date-label as unix nanoseconds (see module doc).
    """

    instrument_id: InstrumentId
    open: float
    high: float
    low: float
    close: float
    volume: int
