"""Canonical path layout for the Norgate data store.

All paths are derived from the repo root so the module works regardless of
the current working directory.

Layout
------
data/norgate/
  working/
    continuous/
      adjusted/     {TICKER}.parquet   ← CCB series, refreshed on every rebuild
      unadjusted/   {TICKER}.parquet   ← raw continuous, refreshed on every rebuild
  archive/
    continuous/     {TICKER}.parquet   ← CCB series, permanent (never purged)
    contracts/
      {TICKER}/     {NORGATE_SYMBOL}.parquet  ← individual expiries, permanent

data/ohlc_data/
  {TICKER}/
    D_{TICKER}.parquet        ← daily from CCB
    W_{TICKER}.parquet        ← weekly
    M_{TICKER}.parquet        ← monthly
    D_{TICKER}_unadj.parquet  ← daily from unadjusted (σ denominator)
"""
from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[3],
    )


def norgate_root() -> Path:
    return _repo_root() / "data" / "norgate"


# ── working dirs (ephemeral, purged on full rebuild) ──────────────────────
def working_adjusted_dir() -> Path:
    return norgate_root() / "working" / "continuous" / "adjusted"


def working_unadjusted_dir() -> Path:
    return norgate_root() / "working" / "continuous" / "unadjusted"


# ── archive dirs (permanent, never purged) ────────────────────────────────
def archive_continuous_dir() -> Path:
    return norgate_root() / "archive" / "continuous"


def archive_contracts_dir() -> Path:
    return norgate_root() / "archive" / "contracts"


def archive_contracts_ticker_dir(ticker: str) -> Path:
    return archive_contracts_dir() / ticker


# ── canonical ohlc store ──────────────────────────────────────────────────
def ohlc_root() -> Path:
    return _repo_root() / "data" / "ohlc_data"


def ohlc_ticker_dir(ticker: str) -> Path:
    return ohlc_root() / ticker
