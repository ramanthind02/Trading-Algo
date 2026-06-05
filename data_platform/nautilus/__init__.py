"""Nautilus-backed data layer (WP-2).

Additive bridge from the homegrown ``data_platform.core`` model to a
NautilusTrader ``ParquetDataCatalog``. This package only *adds* a Nautilus
storage/IO path alongside the existing loaders — it does not modify or remove
any legacy code.

Modules:
  - ``catalog``     — open/locate the ``ParquetDataCatalog``.
  - ``instruments`` — map ``data_platform.core.instruments.Instrument`` rows to
                      Nautilus ``Instrument`` subtypes and write them.
  - ``ingest``      — wrangle MT5 intraday parquet into Nautilus ``Bar`` /
                      ``QuoteTick`` and write them to the catalog.
"""
from __future__ import annotations

from .catalog import default_catalog_path, get_catalog

__all__ = ["get_catalog", "default_catalog_path"]
