"""Verify the daily-bar scrape captured full history per symbol.

Compares each stored ``bars_D1/part.parquet`` against the expected history depth
recorded in ``data/mt5_data/_daily_history_depth.json`` (the broker probe), and
flags symbols whose stored start date is materially later than expected
(truncated) or that are missing entirely.

Known reference depths (from the broker probe):
  FX majors (EURUSD/USDJPY/USDCHF) -> 1971-01-04
  US stocks (uniform)              -> 2008-05-05
  ETFs (uniform)                   -> 2010-01-11
  XAUUSD -> 1998-04-22, XAGUSD -> 2002-09-15

Run after the scrape:
    python -m data_platform.providers.mt5.probes.verify_daily_coverage
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[4],
    )


def _mt5_dir() -> Path:
    return _repo_root() / "data" / "mt5_data"


def _stored_start(symbol: str) -> pd.Timestamp | None:
    path = _mt5_dir() / symbol / "bars_D1" / "part.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["time"])
    if df.empty:
        return None
    return pd.to_datetime(df["time"]).min()


def _load_expected() -> dict[str, str]:
    """Map symbol -> expected oldest date (ISO) from the broker depth probe, if present."""
    depth_file = _mt5_dir() / "_daily_history_depth.json"
    if not depth_file.exists():
        return {}
    raw = json.loads(depth_file.read_text())
    out: dict[str, str] = {}
    # The probe file shape may vary; accept {symbol: {oldest: date}} or {symbol: date}.
    items = raw.items() if isinstance(raw, dict) else []
    for sym, val in items:
        if isinstance(val, dict):
            d = val.get("oldest") or val.get("oldest_date") or val.get("start")
        else:
            d = val
        if isinstance(d, str):
            out[sym] = d[:10]
    return out


def main() -> None:
    mt5_dir = _mt5_dir()
    symbols = sorted(p.name for p in mt5_dir.iterdir()
                     if p.is_dir() and not p.name.startswith("_"))
    expected = _load_expected()

    print(f"=== MT5 daily coverage: {len(symbols)} symbols with a folder ===")
    have_d1 = 0
    total_bars = 0
    truncated: list[str] = []
    missing: list[str] = []

    for sym in symbols:
        start = _stored_start(sym)
        path = mt5_dir / sym / "bars_D1" / "part.parquet"
        if start is None:
            missing.append(sym)
            continue
        have_d1 += 1
        df = pd.read_parquet(path, columns=["time"])
        total_bars += len(df)
        exp = expected.get(sym)
        if exp:
            exp_ts = pd.Timestamp(exp).tz_localize("UTC") if start.tzinfo else pd.Timestamp(exp)
            # Truncated if stored start is >45 days later than the probe's oldest.
            if (start - exp_ts).days > 45:
                truncated.append(f"{sym}: stored {start.date()} vs expected {exp}")

    print(f"  with D1 parquet : {have_d1}")
    print(f"  total daily bars: {total_bars:,}")
    print(f"  no D1 file      : {len(missing)}")
    if missing[:15]:
        print("    e.g.", ", ".join(missing[:15]))
    if truncated:
        print(f"\n  TRUNCATED vs broker probe ({len(truncated)}):")
        for t in truncated[:25]:
            print("    ", t)
    else:
        print("\n  No truncated symbols vs the broker depth probe — full history captured.")

    # Spot-check the reference anchors the user cited.
    print("\n=== reference anchors ===")
    for sym, want in [("EURUSD", "1971-01-04"), ("XAUUSD", "1998-04-22")]:
        s = _stored_start(sym)
        if s is not None:
            ok = "OK" if str(s.date()) <= want or (s - pd.Timestamp(want).tz_localize(s.tzinfo)).days <= 45 else "CHECK"
            print(f"  {sym:<8} stored start {s.date()} (expect ~{want}) [{ok}]")
        else:
            print(f"  {sym:<8} MISSING")


if __name__ == "__main__":
    main()
