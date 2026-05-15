#!/usr/bin/env python3
"""Prop discrete micro exposure on the **live** GlobalPortfolio path.

Optional ``--refresh-candles`` reloads repository OHLC parquets into the central
cache (D+M) for all **required** vault tickers so ``build_cache_query`` overlap
extends to the latest common parquet end (fixes short 2026 windows when one
series lagged).

``--compare-years`` prints 2025 full-year vs 2026 year-to-cache-end stats:
days with >=1 micro (tradeable legs), and distribution of daily
``max(|position_fraction|)`` to explain low discrete exposure at $50k.

Cross-checks contract rounding vs ``portfolio_research.futures_sim`` internals.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\report_prop_micro_exposure_ytd.py --refresh-candles --compare-years
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import FrozenSet

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ensemble.portfolio_impl.portfolio_cache import _query_candles_from_cache
from portfolio_research.futures_sim import (
    _compute_simple_instrument_returns,
    _simulate_ticker_bars,
)
from scripts import enigma_live_forecast as elf
from utils.cache.runtime.cache_manager import CacheManager
from utils.cache.runtime.central_cache import CentralCacheStore
from utils.core.enums import TimeFrame, Ticker
from utils.futures_micro_specs import (
    canonical_listed_micro_futures,
    listed_micro_futures_row,
    micro_contract_fractional_and_whole,
)
from utils.vault_paths import resolve_vault_root


def _ticker_str(value: object) -> str:
    return value.name if hasattr(value, "name") else str(value)


def _print_query_end_limiter(required: set[str], q_end: pd.Timestamp) -> None:
    store = CentralCacheStore.get_instance()
    ends: list[tuple[str, pd.Timestamp]] = []
    for ts in sorted(required):
        if ts not in Ticker.__members__:
            continue
        rec = store.describe_candle(Ticker[ts], TimeFrame.D)
        if rec and rec.coverage.end:
            ends.append((ts, pd.Timestamp(rec.coverage.end).normalize()))
    if not ends:
        return
    limiting = min(ends, key=lambda x: x[1])
    print(
        f"  Shortest daily series (sets query end {q_end.date()}): "
        f"{limiting[0]} through {limiting[1].date()}"
    )


def _refresh_candles_from_repo(required: set[str]) -> None:
    """Reload D+M candles from repo parquets into the central cache for ``required``."""
    tickers = [Ticker[t] for t in sorted(required) if t in Ticker.__members__]
    result = CacheManager().bootstrap_source_candles(
        tickers=tickers,
        timeframes=[TimeFrame.D, TimeFrame.M],
        reset_existing=False,
    )
    print(
        f"  Candle bootstrap: {result['success']}/{result['total']} series loaded "
        f"({result['failed']} failed)"
    )


def _print_cache_coverage(required: set[str]) -> None:
    store = CentralCacheStore.get_instance()
    print("  Central cache candle coverage (bounds of query_end = min of D ends):")
    for ts in sorted(required):
        if ts not in Ticker.__members__:
            continue
        te = Ticker[ts]
        parts = []
        for tf in (TimeFrame.D, TimeFrame.M):
            rec = store.describe_candle(te, tf)
            if rec is None or rec.coverage.end is None:
                parts.append(f"{tf.name}=missing")
            else:
                parts.append(
                    f"{tf.name}={pd.Timestamp(rec.coverage.start).date()}.."
                    f"{pd.Timestamp(rec.coverage.end).date()}"
                )
        print(f"    {ts}: {' | '.join(parts)}")


def _print_repo_parquet_hints(required: set[str]) -> None:
    """Show on-disk parquet span per ticker (source of bootstrap)."""
    mgr = CacheManager()
    per_tf = [TimeFrame.D, TimeFrame.M]
    print("  Repo parquet span (CacheManager.get_available_date_range_per_ticker):")
    ranges = mgr.get_available_date_range_per_ticker(
        [Ticker[t] for t in sorted(required) if t in Ticker.__members__],
        list(per_tf),
    )
    for tk, span in sorted(ranges.items(), key=lambda x: x[0].name):
        lo, hi = span
        print(f"    {tk.name}: {pd.Timestamp(lo).date()} .. {pd.Timestamp(hi).date()}")


def _discrete_stats(
    positions: pd.DataFrame,
    daily_closes: pd.DataFrame,
    *,
    capital_usd: float,
    tradeable: FrozenSet[str] | None,
) -> tuple[int, int, float]:
    if positions.empty or daily_closes.empty:
        return 0, 0, 0.0

    pos = positions.loc[:, ["ticker", "datetime", "position_fraction"]].copy()
    pos["ticker_s"] = pos["ticker"].map(_ticker_str)
    pos["dt"] = pd.to_datetime(pos["datetime"]).dt.normalize()
    if tradeable is not None:
        pos = pos.loc[pos["ticker_s"].isin(tradeable)].reset_index(drop=True)

    c = daily_closes.loc[:, ["ticker", "datetime", "close"]].copy()
    c["ticker_s"] = c["ticker"].map(_ticker_str)
    c["dt"] = pd.to_datetime(c["datetime"]).dt.normalize()

    merged = pos.merge(c[["ticker_s", "dt", "close"]], on=["ticker_s", "dt"], how="inner")
    if merged.empty:
        return 0, 0, 0.0

    wholes: list[int] = []
    fracs: list[float] = []
    for row in merged.itertuples(index=False):
        spec = listed_micro_futures_row(row.ticker_s)
        if spec is None:
            wholes.append(0)
            fracs.append(0.0)
            continue
        frac, whole = micro_contract_fractional_and_whole(
            futures_index_price=float(row.close),
            position_fraction=float(row.position_fraction),
            capital_usd=capital_usd,
            micro_dollars_per_point=spec.micro_dollars_per_point,
        )
        wholes.append(whole)
        fracs.append(abs(frac))
    merged = merged.assign(contracts_whole=wholes, frac_ct=fracs)
    merged["abs_w"] = merged["contracts_whole"].abs()
    by_day = merged.groupby("dt", as_index=False).agg(
        max_abs_pf=("position_fraction", lambda s: float(s.abs().max())),
        max_abs_whole=("abs_w", "max"),
        max_frac_ct=("frac_ct", "max"),
    )
    n_dates = len(by_day)
    n_pos = int((by_day["max_abs_whole"] >= 1).sum())
    pct = 100.0 * n_pos / n_dates if n_dates else 0.0
    return n_dates, n_pos, pct


def _diagnostics_table(
    positions: pd.DataFrame,
    daily_closes: pd.DataFrame,
    *,
    capital_usd: float,
    tradeable: FrozenSet[str],
) -> pd.DataFrame:
    """Per-day: max abs position_fraction, max fractional contracts (any leg)."""
    pos = positions.loc[:, ["ticker", "datetime", "position_fraction"]].copy()
    pos["ticker_s"] = pos["ticker"].map(_ticker_str)
    pos["dt"] = pd.to_datetime(pos["datetime"]).dt.normalize()
    pos = pos.loc[pos["ticker_s"].isin(tradeable)].reset_index(drop=True)
    c = daily_closes.loc[:, ["ticker", "datetime", "close"]].copy()
    c["ticker_s"] = c["ticker"].map(_ticker_str)
    c["dt"] = pd.to_datetime(c["datetime"]).dt.normalize()
    merged = pos.merge(c[["ticker_s", "dt", "close"]], on=["ticker_s", "dt"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["dt", "max_abs_pf", "max_frac_ct"])

    fracs: list[float] = []
    for row in merged.itertuples(index=False):
        spec = listed_micro_futures_row(row.ticker_s)
        if spec is None:
            fracs.append(0.0)
            continue
        frac, _ = micro_contract_fractional_and_whole(
            futures_index_price=float(row.close),
            position_fraction=float(row.position_fraction),
            capital_usd=capital_usd,
            micro_dollars_per_point=spec.micro_dollars_per_point,
        )
        fracs.append(abs(frac))
    merged = merged.assign(frac_ct=fracs)
    return (
        merged.groupby("dt", as_index=False)
        .agg(
            max_abs_pf=("position_fraction", lambda s: float(s.abs().max())),
            max_frac_ct=("frac_ct", "max"),
        )
        .sort_values("dt")
        .reset_index(drop=True)
    )


def _futures_sim_exposure_days(
    *,
    live_slice: pd.DataFrame,
    daily_for_sim: pd.DataFrame,
    capital: float,
    tradeable: FrozenSet[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None,
) -> tuple[int, int, float]:
    micro = canonical_listed_micro_futures()
    instrument_returns = _compute_simple_instrument_returns(daily_for_sim)
    positions_norm = live_slice.loc[:, ["ticker", "datetime", "position_fraction"]].copy()
    positions_norm["datetime"] = pd.to_datetime(positions_norm["datetime"]).dt.floor("s")
    positions_norm["ticker"] = positions_norm["ticker"].map(_ticker_str)

    all_rows: list[tuple[pd.Timestamp, int]] = []
    for ticker in sorted(tradeable):
        if ticker not in micro:
            continue
        spec = micro[ticker]
        sub = positions_norm.loc[positions_norm["ticker"] == ticker]
        if sub.empty:
            continue
        bars = _simulate_ticker_bars(
            ticker=ticker,
            ticker_positions=sub,
            instrument_returns=instrument_returns,
            multiplier=spec.micro_dollars_per_point,
            margin_long=spec.illustrative_margin_long_usd,
            margin_short=spec.illustrative_margin_short_usd,
            account_capital=capital,
            finite_leverage=True,
        )
        all_rows.extend((b.date, b.contracts) for b in bars)

    if not all_rows:
        return 0, 0, 0.0
    diag = pd.DataFrame(all_rows, columns=["date", "contracts"])
    diag["date"] = pd.to_datetime(diag["date"]).dt.normalize()
    sub = diag.loc[diag["date"] >= start_ts.normalize()].reset_index(drop=True)
    if end_ts is not None:
        sub = sub.loc[sub["date"] <= end_ts.normalize()].reset_index(drop=True)
    if sub.empty:
        return 0, 0, 0.0
    sub["ac"] = sub["contracts"].abs()
    by_day = sub.groupby("date", as_index=False)["ac"].max()
    n_dates = len(by_day)
    n_pos = int((by_day["ac"] >= 1).sum())
    pct = 100.0 * n_pos / n_dates if n_dates else 0.0
    return n_dates, n_pos, pct


def _slice_by_dates(
    df: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"]).dt.normalize()
    mask = dt >= start.normalize()
    if end is not None:
        mask &= dt <= end.normalize()
    return df.loc[mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO / "configs" / "live_forecast_config_prop.json",
    )
    parser.add_argument(
        "--refresh-candles",
        action="store_true",
        help="Reload D+M OHLC from repo parquets into central cache for required tickers.",
    )
    parser.add_argument(
        "--compare-years",
        action="store_true",
        help="Print 2025 vs 2026 (through cache overlap) exposure and PF diagnostics.",
    )
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--capital", type=float, default=50_000.0)
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else _REPO / args.config
    prop = json.loads(cfg_path.read_text(encoding="utf-8"))
    prop["portfolio"]["vault_root"] = str(
        resolve_vault_root(prop["portfolio"].get("vault_root") or "vault")
    )
    prop["portfolio"]["target_volatility"] = 0.20
    capital = float(args.capital)
    tradeable_raw = prop.get("tradeable_tickers") or []
    tradeable: FrozenSet[str] = frozenset(str(t) for t in tradeable_raw)
    start_ts = pd.Timestamp(args.start).normalize()

    print("=" * 72)
    print("Prop micro exposure (live GlobalPortfolio + discrete cross-check)")
    print(f"  Config: {cfg_path}")
    print(f"  target_volatility: {prop['portfolio']['target_volatility']}")
    print(f"  max_position_pct: {prop['portfolio']['max_position_pct']}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Tradeable: {sorted(tradeable)}")
    print("=" * 72)

    portfolio = elf.build_portfolio(prop)
    required = elf.discover_required_tickers(portfolio)
    data_cfg = prop.get("data") if isinstance(prop.get("data"), dict) else None

    if args.refresh_candles:
        print("Refreshing candles from repo parquets (D+M, required tickers)...")
        _print_repo_parquet_hints(required)
        _refresh_candles_from_repo(required)

    _print_cache_coverage(required)

    print("  Refreshing bias caches...")
    elf.refresh_bias_caches(prop["portfolio"]["vault_root"], required)

    query, instrument_returns = elf.build_cache_query(
        required,
        data_config=data_cfg,
        daily_overlay=None,
    )
    portfolio.fit_from_cache(query, instrument_returns)
    positions_live = portfolio.predict_from_cache(query)
    positions_live["datetime"] = pd.to_datetime(positions_live["datetime"]).dt.floor("s")
    q_end = pd.Timestamp(query.end).normalize()

    daily_live = _query_candles_from_cache(query, TimeFrame.D)
    daily_tradeable = daily_live.loc[
        daily_live["ticker"].map(_ticker_str).isin(tradeable)
    ].reset_index(drop=True)

    print()
    print(f"LIVE cache overlap end: {q_end.date()}")
    _print_query_end_limiter(required, q_end)

    if args.compare_years:
        windows: list[tuple[str, pd.Timestamp, pd.Timestamp | None]] = [
            ("2025 (calendar)", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
            ("2026 (Jan 1 .. cache end)", pd.Timestamp("2026-01-01"), q_end),
        ]
        for label, w_start, w_end in windows:
            sl = _slice_by_dates(positions_live, start=w_start, end=w_end)
            n_d, n_m, pct = _discrete_stats(
                sl, daily_live, capital_usd=capital, tradeable=tradeable
            )
            n_fs, n_m_fs, pct_fs = _futures_sim_exposure_days(
                live_slice=sl,
                daily_for_sim=daily_tradeable,
                capital=capital,
                tradeable=tradeable,
                start_ts=w_start,
                end_ts=w_end,
            )
            diag_tbl = _diagnostics_table(
                sl, daily_live, capital_usd=capital, tradeable=tradeable
            )
            med_pf = float(diag_tbl["max_abs_pf"].median()) if not diag_tbl.empty else float("nan")
            p90_pf = float(diag_tbl["max_abs_pf"].quantile(0.9)) if not diag_tbl.empty else float("nan")
            med_fc = float(diag_tbl["max_frac_ct"].median()) if not diag_tbl.empty else float("nan")
            days_half = int((diag_tbl["max_frac_ct"] >= 0.5).sum()) if not diag_tbl.empty else 0
            print()
            print(f"--- {label} ---")
            print(f"  Rows (tradeable slice): {len(sl)}")
            print(
                f"  Days in grid: {n_d} | >=1 micro any leg: {n_m} ({pct:.1f}%) "
                f"| futures_sim-style: {n_m_fs}/{n_fs} ({pct_fs:.1f}%)"
            )
            print(
                f"  Daily max |position_fraction| (over ES/NQ/GC): "
                f"median={med_pf:.4f} p90={p90_pf:.4f}"
            )
            print(
                f"  Daily max fractional micro count (pre-round): median={med_fc:.3f} "
                f"| days with max_frac>=0.5 (can round to 1): {days_half}"
            )
        print()
        print(
            "Note: Low 2026 discrete exposure is usually (1) smaller |position_fraction| "
            "than needed for 0.5+ fractional micros at $50k, and/or (2) a shorter cache "
            "window until all required tickers have D candles through the same end date."
        )
        return

    live_slice = _slice_by_dates(positions_live, start=start_ts, end=None)
    n_live, pos_live, pct_live = _discrete_stats(
        live_slice,
        daily_live,
        capital_usd=capital,
        tradeable=tradeable,
    )
    n_fs, pos_fs, pct_fs = _futures_sim_exposure_days(
        live_slice=live_slice,
        daily_for_sim=daily_tradeable,
        capital=capital,
        tradeable=tradeable,
        start_ts=start_ts,
        end_ts=None,
    )
    print(f"  From {start_ts.date()} through {q_end.date()}: {len(live_slice)} rows")
    print(
        f"  Direct merge: {n_live} days, >=1 micro on {pos_live} ({pct_live:.1f}%) "
        f"| futures_sim: {n_fs} days, {pos_fs} ({pct_fs:.1f}%)"
    )


if __name__ == "__main__":
    main()
