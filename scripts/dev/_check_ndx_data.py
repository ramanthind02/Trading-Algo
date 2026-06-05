"""Sanity check NDX scraped data and catalog lookup."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data_platform.core import load_catalog, source_symbol, instrument_from_source_symbol

catalog = load_catalog()

# 1. Catalog lookup
inst = catalog.find("NDX.XNAS")
print(f"Catalog entry: {inst.id}  class={inst.instrument_class.value}  source={inst.data_source}")
print(f"  raw_symbol={inst.raw_symbol}  asset={inst.asset_class.value}")
print(f"  source_symbols: {inst.info.get('source_symbols', {})}")
print(f"  tick_history_start: {inst.info.get('tick_history_start')}")
print()

# 2. Reverse lookup: NDX symbol -> InstrumentId
iid = instrument_from_source_symbol(catalog, "mt5", "NDX")
print(f"Reverse lookup mt5:NDX -> {iid}")
print()

# 3. M1 bars
m1 = pd.read_parquet("data/mt5_data/NDX/bars_M1/year=2026/part.parquet")
print(f"M1 bars: {len(m1):,} rows")
print(f"  Columns: {m1.columns.tolist()}")
print(f"  Date range: {m1['time'].min()} -> {m1['time'].max()}")
print(f"  Sample spread values (points): {m1['spread'].describe().to_dict()}")
print()

# 4. Ticks
ticks = pd.read_parquet("data/mt5_data/NDX/ticks/year=2026/part.parquet")
print(f"Ticks: {len(ticks):,} rows")
print(f"  Columns: {ticks.columns.tolist()}")
print(f"  Date range: {ticks['time'].min()} -> {ticks['time'].max()}")
print(f"  Bid/ask spread sample (first 5 rows):")
print(ticks[['time', 'bid', 'ask']].head().to_string(index=False))
print(f"  Avg bid-ask spread: {(ticks['ask'] - ticks['bid']).mean():.4f}")
