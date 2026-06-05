"""Ticker and symbol constants for the Norgate adapter.

Single source of truth for every mapping between repo tickers and Norgate
symbols.  All other modules import from here.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Repo ticker  →  Norgate back-adjusted continuous symbol (&XX_CCB)
# ---------------------------------------------------------------------------
TICKER_TO_CCB: dict[str, str] = {
    "ES":  "&ES_CCB",
    "NQ":  "&NQ_CCB",
    "YM":  "&YM_CCB",
    "RTY": "&RTY_CCB",
    "CL":  "&CL_CCB",
    "HO":  "&HO_CCB",
    "GC":  "&GC_CCB",
    "HG":  "&HG_CCB",
    "SI":  "&SI_CCB",
    "PL":  "&PL_CCB",
    "EU":  "&6E_CCB",
    "JY":  "&6J_CCB",
    "BP":  "&6B_CCB",
    "CD":  "&6C_CCB",
    "SF":  "&6S_CCB",
    "C":   "&ZC_CCB",
    "S":   "&ZS_CCB",
    "W":   "&ZW_CCB",
    "GF":  "&GF_CCB",
    "TY":  "&ZN_CCB",
    "FV":  "&ZF_CCB",
    "US":  "&ZB_CCB",
    "TU":  "&ZT_CCB",
}

# Repo ticker  →  Norgate unadjusted continuous symbol (&XX)
TICKER_TO_RAW: dict[str, str] = {
    ticker: symbol.replace("_CCB", "")
    for ticker, symbol in TICKER_TO_CCB.items()
}

# Repo ticker  →  Norgate Futures DB prefix for individual contracts
# e.g. "ES" → contracts named "ES-2024H", "ES-2024M", ...
TICKER_TO_CONTRACT_PREFIX: dict[str, str] = {
    "ES":  "ES",
    "NQ":  "NQ",
    "YM":  "YM",
    "RTY": "RTY",
    "CL":  "CL",
    "HO":  "HO",
    "GC":  "GC",
    "HG":  "HG",
    "SI":  "SI",
    "PL":  "PL",
    "EU":  "6E",
    "JY":  "6J",
    "BP":  "6B",
    "CD":  "6C",
    "SF":  "6S",
    "C":   "ZC",
    "S":   "ZS",
    "W":   "ZW",
    "GF":  "GF",
    "TY":  "ZN",
    "FV":  "ZF",
    "US":  "ZB",
    "TU":  "ZT",
}

# Parquet write settings used across all archive and working stores.
# zstd level 3: ~27% smaller than snappy with negligible read overhead.
PARQUET_COMPRESSION: str = "zstd"
PARQUET_COMPRESSION_LEVEL: int = 3
