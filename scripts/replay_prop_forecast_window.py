#!/usr/bin/env python3
"""Replay prop-firm GlobalPortfolio forecasts over a recent cache window.

Fits the portfolio exactly like ``enigma_prop_forecast`` (same
``build_cache_query`` / instrument returns), runs ``predict_from_cache`` on
the **full** overlap query (so monthly + daily cache reads stay consistent),
then **filters** rows to the last ``--trading-days`` business days for display
or CSV export.

This is **not** a causal walk-forward: the WeightLayer is fit once using the
full overlap implied by ``build_cache_query``. Predictions still use the full
candle/volatility context from cache; only the printed/exported rows are
trimmed to the replay window.

Usage::

    python scripts/replay_prop_forecast_window.py --trading-days 10
    python scripts/replay_prop_forecast_window.py --trading-days 10 --as-of 2026-05-13
    python scripts/replay_prop_forecast_window.py --trading-days 10 --output deploy/prop_replay.csv

``--as-of`` sets the **last** calendar day of the strip (normalized); the first day
is ``as_of`` minus ``trading_days - 1`` **business** days, clipped to the cache
query start. If ``--as-of`` is omitted, the strip ends at the cache overlap
(``query.end``).

By default this script **connects to TWS** (same host/port/client_id as the
config), **fetches** missing/recent daily history from Interactive Brokers, **upserts**
it into the central cache, refreshes bias caches, then runs the replay (same as
the live forecast data step). Use ``--no-ib-sync`` to use only whatever is already
in cache (e.g. CI or offline). For automation only, ``TRADING_ALGO_SKIP_IB_CANDLE_SYNC=1``
still disables the IB step but prints a notice to stdout.

Also prints a per-day **max abs whole micro** table (same sizing as live) and lists
dates with ``>= 1`` or ``> 1`` contract on any tradeable leg.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, Set

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ensemble.portfolio import GlobalPortfolio, PortfolioCacheQuery
from ensemble.portfolio_impl.portfolio_cache import _query_candles_from_cache
from scripts import enigma_live_forecast as _elf
from scripts.demo_ib_data_fetch import IBDataClient, IBConfig
from lib.core.enums import TimeFrame
from lib.core.futures_micro_specs import listed_micro_futures_row, micro_contract_fractional_and_whole
from lib.core.vault_paths import resolve_vault_root


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _tradeable_set(config: Dict[str, Any]) -> FrozenSet[str] | None:
    raw = config.get("tradeable_tickers")
    if not raw:
        return None
    return frozenset(str(t) for t in raw)


def _ticker_str(value: object) -> str:
    return value.name if hasattr(value, "name") else str(value)


def _daily_max_abs_whole_micros(
    positions: pd.DataFrame,
    query: PortfolioCacheQuery,
    *,
    capital_usd: float,
    tradeable: FrozenSet[str] | None,
) -> pd.DataFrame:
    """Per calendar day: max abs whole micro contracts across tradeable legs (same rounding as live)."""
    if positions.empty:
        return pd.DataFrame(columns=["dt", "max_abs_whole"])

    pos = positions.loc[:, ["ticker", "datetime", "position_fraction"]].copy()
    pos["ticker_s"] = pos["ticker"].map(_ticker_str)
    pos["dt"] = pd.to_datetime(pos["datetime"]).dt.normalize()
    if tradeable is not None:
        pos = pos.loc[pos["ticker_s"].isin(tradeable)].reset_index(drop=True)

    daily = _query_candles_from_cache(query, TimeFrame.D)
    daily = daily.loc[:, ["ticker", "datetime", "close"]].copy()
    daily["ticker_s"] = daily["ticker"].map(_ticker_str)
    daily["dt"] = pd.to_datetime(daily["datetime"]).dt.normalize()
    if tradeable is not None:
        daily = daily.loc[daily["ticker_s"].isin(tradeable)].reset_index(drop=True)

    merged = pos.merge(daily[["ticker_s", "dt", "close"]], on=["ticker_s", "dt"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["dt", "max_abs_whole"])

    def _whole(row: pd.Series) -> int:
        spec = listed_micro_futures_row(str(row["ticker_s"]))
        if spec is None:
            return 0
        _frac, whole = micro_contract_fractional_and_whole(
            futures_index_price=float(row["close"]),
            position_fraction=float(row["position_fraction"]),
            capital_usd=capital_usd,
            micro_dollars_per_point=spec.micro_dollars_per_point,
        )
        return int(whole)

    merged = merged.assign(contracts_whole=merged.apply(_whole, axis=1))
    merged["abs_w"] = merged["contracts_whole"].abs()
    return (
        merged.groupby("dt", as_index=False)["abs_w"]
        .max()
        .rename(columns={"abs_w": "max_abs_whole"})
        .sort_values("dt")
        .reset_index(drop=True)
    )


def _replay_window_bounds(
    *,
    query_start: pd.Timestamp,
    window_end: pd.Timestamp,
    trading_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive replay strip [start, end] on normalized dates."""
    if trading_days < 1:
        raise ValueError("trading_days must be >= 1")
    we = window_end.normalize()
    span = max(trading_days - 1, 0)
    candidate_start = we - pd.offsets.BDay(span)
    qs = pd.Timestamp(query_start).normalize()
    narrow_start = max(qs, candidate_start.normalize())
    return narrow_start, we


def _print_daily_coverage_banner(required: Set[str], title: str) -> None:
    """Print min daily ``coverage.end`` across required tickers (sets portfolio query end)."""
    from lib.cache.runtime.central_cache import CentralCacheStore
    from lib.core.enums import Ticker

    store = CentralCacheStore.get_instance()
    ends: list[pd.Timestamp] = []
    print(title)
    for ts in sorted(required):
        try:
            te = Ticker[ts]
        except KeyError:
            continue
        rec = store.describe_candle(te, TimeFrame.D)
        if rec is None or rec.coverage.end is None:
            print(f"  {ts} D: (missing)")
            continue
        end = pd.Timestamp(rec.coverage.end).normalize()
        ends.append(end)
        start = rec.coverage.start
        lo = pd.Timestamp(start).date() if start is not None else None
        print(f"  {ts} D: {lo} .. {end.date()}")
    if ends:
        print(f"  -> query end floor (min of above): {min(ends).date()}")


def _replay_sync_ib_candles_then_bias(
    *,
    config: Dict[str, Any],
    required: Set[str],
    no_ib_sync: bool,
) -> None:
    """Pull IB dailies into central cache and refresh bias artifacts (best-effort)."""
    if no_ib_sync:
        print("\n(IB sync disabled: --no-ib-sync; using existing central cache.)")
        _print_daily_coverage_banner(required, "Current central cache:")
        return
    skip = os.environ.get("TRADING_ALGO_SKIP_IB_CANDLE_SYNC", "").lower() in ("1", "true", "yes")
    if skip:
        print(
            "\n(IB sync disabled: TRADING_ALGO_SKIP_IB_CANDLE_SYNC is set; "
            "using existing central cache.)"
        )
        _print_daily_coverage_banner(required, "Current central cache:")
        return

    conn = config.get("connection") or {}
    host = str(conn.get("host", "127.0.0.1"))
    port = int(conn.get("port", 7497))
    client_id = int(conn.get("client_id", 1))

    _print_daily_coverage_banner(required, "Central cache before IB upsert:")

    client: IBDataClient | None = None
    try:
        print(
            f"\nSyncing daily candles from Interactive Brokers @ {host}:{port} "
            f"(client_id={client_id}), then bias refresh..."
        )
        ib_config = IBConfig(host=host, port=port, client_id=client_id)
        client = IBDataClient(ib_config)
        client.connect_to_ib()
        waited = 0.0
        max_wait = 30.0
        while not client.connected and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
        if not client.connected:
            print(
                f"  Warning: TWS not connected within {max_wait:.0f}s; "
                "continuing with existing cache only.",
                file=sys.stderr,
            )
            _print_daily_coverage_banner(required, "Central cache (unchanged):")
            return
        time.sleep(1.0)
        _elf.sync_ib_fetched_dailies_into_central_cache(
            config=config,
            required_tickers=required,
            client=client,
            profile="prop",
        )
        _elf.refresh_bias_caches(
            str(resolve_vault_root(config["portfolio"].get("vault_root") or "vault")),
            required,
        )
        print("IB history upserted; bias caches refreshed for vault.")
        _print_daily_coverage_banner(required, "Central cache after IB upsert:")
    except Exception as exc:
        print(
            f"  Warning: IB sync failed ({type(exc).__name__}: {exc}); using existing cache.",
            file=sys.stderr,
        )
        _print_daily_coverage_banner(required, "Central cache after failed IB sync:")
    finally:
        if client is not None:
            try:
                client.disconnect_from_ib()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "live_forecast_config_prop.json",
        help="Prop live JSON (default: configs/live_forecast_config_prop.json)",
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=10,
        help=(
            "Number of **business** days in the replay strip ending at ``--as-of`` "
            "(or at cache ``query.end`` if omitted)."
        ),
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help=(
            "Last calendar day of the replay window (e.g. 2026-05-13). "
            "Default: cache overlap end."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for the replayed position rows",
    )
    parser.add_argument(
        "--no-ib-sync",
        action="store_true",
        help="Do not connect to TWS; use only candles already in central cache.",
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else _REPO_ROOT / args.config
    config = _load_config(cfg_path)
    config["portfolio"]["vault_root"] = str(
        resolve_vault_root(config["portfolio"].get("vault_root") or "vault")
    )

    portfolio: GlobalPortfolio = _elf.build_portfolio(config)
    required: Set[str] = _elf.discover_required_tickers(portfolio)
    data_cfg = config.get("data") or {}

    _replay_sync_ib_candles_then_bias(
        config=config,
        required=required,
        no_ib_sync=bool(args.no_ib_sync),
    )

    query, instrument_returns = _elf.build_cache_query(
        required,
        data_config=data_cfg if isinstance(data_cfg, dict) else None,
        daily_overlay=None,
    )

    portfolio.fit_from_cache(query, instrument_returns)

    q_start = pd.Timestamp(query.start)
    q_end = pd.Timestamp(query.end)
    q_end_n = q_end.normalize()
    requested_end = (
        pd.Timestamp(args.as_of).normalize()
        if args.as_of is not None
        else q_end_n
    )
    effective_end = min(requested_end, q_end_n)
    if args.as_of is not None and requested_end > q_end_n:
        print(
            f"Note: --as-of {requested_end.date()} is after cache query end {q_end_n.date()}; "
            f"using last {args.trading_days} business day(s) through {effective_end.date()}.\n",
            file=sys.stderr,
        )
    narrow_start, window_end = _replay_window_bounds(
        query_start=q_start,
        window_end=effective_end,
        trading_days=args.trading_days,
    )

    positions = portfolio.predict_from_cache(query)
    if positions.empty:
        print("No positions returned.", file=sys.stderr)
        sys.exit(1)

    dt_series = pd.to_datetime(positions["datetime"]).dt.normalize()
    mask = (dt_series >= narrow_start) & (dt_series <= window_end)
    positions = positions.loc[mask].reset_index(drop=True)

    if positions.empty:
        print(
            "No position rows in the requested replay window (extend cache / check as-of).",
            file=sys.stderr,
        )
        sys.exit(1)

    tradeable = _tradeable_set(config)
    if tradeable is not None:
        key_series = positions["ticker"].map(
            lambda v: v.name if hasattr(v, "name") else str(v)
        )
        positions = positions.loc[key_series.isin(tradeable)].reset_index(drop=True)

    if positions.empty:
        print(
            "No tradeable-ticker rows in the requested replay window.",
            file=sys.stderr,
        )
        sys.exit(1)

    positions = positions.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    capital = float(config.get("account", {}).get("capital_usd", 50_000.0))

    print(
        f"Replay window: {narrow_start.date()} .. {window_end.date()} "
        f"({args.trading_days} business days requested; fit used full cache query "
        f"{q_start.date()} .. {q_end.date()})."
    )
    print(positions.to_string(index=False))

    micro_days = _daily_max_abs_whole_micros(
        positions,
        query,
        capital_usd=capital,
        tradeable=tradeable,
    )
    print()
    print(f"--- Discrete micro sizing (${capital:,.0f} notional, banker's round) ---")
    if micro_days.empty:
        print("No overlapping daily closes for micro sizing in this window.")
    else:
        print(micro_days.to_string(index=False))
        ge1 = micro_days.loc[micro_days["max_abs_whole"] >= 1, "dt"]
        gt1 = micro_days.loc[micro_days["max_abs_whole"] > 1, "dt"]
        print()
        print(
            f"Days with >= 1 micro (any leg): {len(ge1)} | dates: "
            f"{', '.join(d.strftime('%Y-%m-%d') for d in ge1)}"
        )
        if gt1.empty:
            print("Days with > 1 micro on any leg: none in this window.")
        else:
            print(
                f"Days with > 1 micro on any leg: {len(gt1)} | dates: "
                f"{', '.join(d.strftime('%Y-%m-%d') for d in gt1)}"
            )

    if args.output is not None:
        out = args.output if args.output.is_absolute() else _REPO_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        positions.to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
