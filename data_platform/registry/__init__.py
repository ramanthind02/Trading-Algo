"""data_platform.registry — queryable metadata index for the trading platform.

Public API summary
------------------
Connection management (db module):
  connect()            -- read-write; creates schema on first use
  connect_readonly()   -- read-only URI connection (agent/reader path, ADR-9)
  transaction(conn)    -- context manager: commit on success, rollback on exception
  registry_path()      -- canonical path: <repo>/data/registry.db
  RegistryVersionError -- raised when DB user_version != SCHEMA_VERSION

Mutation (writer module — single-writer rule, ADR-3):
  writer.upsert_instrument / replace_source_symbols
  writer.record_blob
  writer.record_job_run / finish_job_run
  writer.upsert_spec / upsert_run / add_result / upsert_vault_entry
  writer.upsert_account / insert_order / insert_deal
  writer.insert_forecast / insert_equity_snapshot

Queries (reader module — always read-only):
  reader.instrument_by_source_symbol
  reader.coverage / freshness
  reader.runs_by_spec / run / headline_metrics
  reader.vault_lineage
  reader.accounts_active / deals_for_account / slippage
  reader.job_history

CLI:
  python -m data_platform.registry [rebuild|backup|report]
"""
from data_platform.registry.db import (
    SCHEMA_VERSION,
    RegistryVersionError,
    connect,
    connect_readonly,
    registry_path,
    transaction,
)
from data_platform.registry import reader, writer

__all__ = [
    "SCHEMA_VERSION",
    "RegistryVersionError",
    "connect",
    "connect_readonly",
    "registry_path",
    "transaction",
    "reader",
    "writer",
]
