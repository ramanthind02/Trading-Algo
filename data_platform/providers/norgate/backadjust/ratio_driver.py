"""Emit proportional (RATIO) continuous-future series into the canonical store.

For every futures ticker in ``data/ohlc_data/{T}/`` that has both an additive
``D_{T}.parquet`` and an unadjusted ``D_{T}_unadj.parquet``, this driver:

  1. recovers the roll events from the additive/unadjusted pair,
  2. multiplicatively back-adjusts the unadjusted OHLC,
  3. writes ``D_{T}_ratio.parquet`` (daily) and resamples to
     ``W_{T}_ratio.parquet`` (W-SUN) and ``M_{T}_ratio.parquet`` (ME),

alongside the existing files — nothing is overwritten or moved. The ratio
series is the **%-return / σ** reference (additive stays the signal input; the
unadjusted stays the real-price reference). See
``docs/library/Data/feed_comparison_and_adjustment.md``.

Schema matches ``migrate.py`` exactly: date32 index, float32 OHLC, int32 volume.

Run directly:
    python -m data_platform.providers.norgate.backadjust.ratio_driver
    python -m data_platform.providers.norgate.backadjust.ratio_driver --tickers ES NQ GC
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_platform.storage import NorgateSeriesKind, write_norgate_bars
from .._paths import ohlc_root, ohlc_ticker_dir
from .ratio_adjuster import apply_ratio_adjustment, recover_rolls

OHLC_COLS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class RatioResult:
    ticker: str
    n_rolls: int
    daily_rows: int
    min_price: float
    source_neg: bool   # the unadjusted source itself printed a negative (genuine, e.g. WTI Apr-2020)


# ── schema helpers (mirror migrate.py) ─────────────────────────────────────


def _read_ohlc(path: Path) -> pd.DataFrame:
    """Read a canonical ohlc parquet into a date-indexed frame."""
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    elif "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return df.sort_index()


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OHLC_COLS:
        if col in out.columns:
            out[col] = out[col].astype("float32")
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0).astype("int32")
    else:
        out["volume"] = 0
        out["volume"] = out["volume"].astype("int32")
    keep = [c for c in (*OHLC_COLS, "volume") if c in out.columns]
    out = out[keep]
    out.index.name = "date"
    return out


def _aggregate(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = (
        daily.resample(freq)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open"])
    )
    agg.index.name = "date"
    for col in OHLC_COLS:
        agg[col] = agg[col].astype("float32")
    agg["volume"] = agg["volume"].astype("int32")
    return agg


def _write(df: pd.DataFrame, path: Path) -> None:
    write_norgate_bars(df, path, store="ohlc_data", kind=NorgateSeriesKind.ADJUSTMENT)


# ── per-ticker generation ──────────────────────────────────────────────────


def has_ratio_inputs(ticker: str) -> bool:
    """True when both additive and unadjusted daily files exist for *ticker*."""
    tdir = ohlc_ticker_dir(ticker)
    return (tdir / f"D_{ticker}.parquet").exists() and (
        tdir / f"D_{ticker}_unadj.parquet"
    ).exists()


def generate_ratio_for_ticker(ticker: str) -> RatioResult:
    tdir = ohlc_ticker_dir(ticker)
    adj_df = _read_ohlc(tdir / f"D_{ticker}.parquet")
    unadj_df = _read_ohlc(tdir / f"D_{ticker}_unadj.parquet")

    rolls = recover_rolls(adj_df["close"].astype("float64"), unadj_df["close"].astype("float64"))
    ratio_daily = _coerce_schema(apply_ratio_adjustment(unadj_df, rolls))

    weekly = _aggregate(ratio_daily, "W-SUN")
    monthly = _aggregate(ratio_daily, "ME")

    _write(ratio_daily, tdir / f"D_{ticker}_ratio.parquet")
    _write(weekly, tdir / f"W_{ticker}_ratio.parquet")
    _write(monthly, tdir / f"M_{ticker}_ratio.parquet")

    source_min = float(unadj_df["close"].min())
    return RatioResult(
        ticker=ticker,
        n_rolls=len(rolls),
        daily_rows=len(ratio_daily),
        min_price=float(ratio_daily["close"].min()),
        source_neg=source_min < 0.0,
    )


def _discover_tickers() -> list[str]:
    """All ticker dirs in ohlc_data that have both adjusted + unadjusted daily."""
    root = ohlc_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and has_ratio_inputs(p.name))


def generate_all(tickers: list[str] | None = None) -> list[RatioResult]:
    targets = tickers or _discover_tickers()
    results: list[RatioResult] = []
    for ticker in targets:
        if not has_ratio_inputs(ticker):
            print(f"  SKIP  {ticker}: missing additive or unadjusted daily (no ratio inputs)")
            continue
        result = generate_ratio_for_ticker(ticker)
        results.append(result)
        flag = "  (source neg — genuine, not a method bug)" if result.source_neg else ""
        print(
            f"  {result.ticker:<5}  rolls={result.n_rolls:>4}  D={result.daily_rows:>6}"
            f"  min={result.min_price:>12.4f}{flag}"
        )
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Emit RATIO continuous-future series into ohlc_data.")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Subset of tickers (default: every ticker with ratio inputs).")
    args = p.parse_args(argv)

    print("=== RATIO back-adjustment -> ohlc_data (D/W/M _ratio) ===")
    results = generate_all(args.tickers)
    n_method_bug = 0  # method never introduces negatives on a positive source; flagged below
    for r in results:
        if r.min_price < 0.0 and not r.source_neg:
            print(f"  WARN {r.ticker}: ratio min<0 with positive source — METHOD BUG")
            n_method_bug += 1
    print(f"\nDone. {len(results)} tickers written (D/W/M _ratio each).")
    if n_method_bug:
        print(f"  {n_method_bug} ticker(s) failed the no-negative invariant.")
    return 1 if n_method_bug else 0


if __name__ == "__main__":
    raise SystemExit(main())
