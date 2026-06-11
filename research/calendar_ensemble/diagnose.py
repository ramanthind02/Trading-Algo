#!/usr/bin/env python3
"""Run calendar-ensemble diagnostics (raw vs vol, bond FOMC, events, windows)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from lib.core.runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

from data_platform.events.diagnostics import (  # noqa: E402
    BLOG_NARRATIVE_WINDOWS,
    BLOG_PARAMETER_WINDOWS,
    aggregate_event_stats,
    event_trade_table,
    load_calendar_bundle,
    load_ticker_log_returns,
    raw_strategy_returns,
    summarize_return_series,
    vol_scaled_portfolio_returns,
)
from data_platform.events.trading_day_index import TradingDayIndex, load_es_trading_sessions
from lib.core.enums import Ticker

_OUTPUT_DIR = (
    Path("research/feature/in_sample/results/calendar_ensemble/diagnostics")
)


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for key, val in metrics.items():
        if isinstance(val, float):
            if (
                "return" in key
                or "drawdown" in key
                or "vol" in key
                or "sharpe" in key
                or "market" in key
                or "active" in key
            ) and "year" not in key:
                print(f"  {key}: {val*100:.2f}%")
            else:
                print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


def _fomc_only_returns(
    fomc_tickers: tuple[Ticker, ...],
    start: datetime,
    end: datetime,
    windows: object = BLOG_PARAMETER_WINDOWS,
) -> pd.DataFrame:
    """Long only on FOMC window days (no holiday legs)."""
    bundle = load_calendar_bundle()
    index = TradingDayIndex(load_es_trading_sessions(start.date(), end.date()))
    fomc_sessions = index.active_sessions(
        bundle.fomc_decision_dates,
        windows.fomc_entry,
        windows.fomc_exit,
    )
    legs = {"equity_holiday": frozenset(), "gold_holiday": frozenset(), "fomc": fomc_sessions}
    from data_platform.events.diagnostics import combined_signal_series

    frames = []
    for ticker in fomc_tickers:
        lr = load_ticker_log_returns(ticker, start, end)
        sig = combined_signal_series(lr.index, ticker=ticker, legs=legs)
        df = pd.DataFrame(
            {
                "ticker": ticker.name,
                "signal": sig,
                "log_return": lr,
                "strategy_return": sig * lr,
            }
        )
        frames.append(df)
    long = pd.concat(frames).sort_index()
    port = long.groupby(long.index)["strategy_return"].mean().rename("portfolio")
    return port


def main() -> int:
    repo = ensure_project_root_on_path()
    out = repo / _OUTPUT_DIR
    start = datetime(2005, 1, 1)
    end = datetime(2022, 12, 31)

    print("Calendar ensemble diagnostics")
    print(f"Period: {start.date()} .. {end.date()}")
    print(f"Output: {out}")

    # --- 1. Raw vs vol-scaled (ES+GC vault sleeve) ---
    vault_tickers = (Ticker.ES, Ticker.GC)
    config_tickers = (Ticker.ES, Ticker.NQ, Ticker.GC)

    for label, tickers in (
        ("ES+GC (vault)", vault_tickers),
        ("ES+NQ+GC (config)", config_tickers),
    ):
        _, raw_port = raw_strategy_returns(tickers, start, end, BLOG_PARAMETER_WINDOWS)
        raw_m = summarize_return_series(raw_port["portfolio"])
        _print_metrics(f"RAW signal×return — {label}", raw_m)

        vol_port = vol_scaled_portfolio_returns(
            tickers, start, end, BLOG_PARAMETER_WINDOWS, target_volatility=0.15
        )
        vol_m = summarize_return_series(vol_port)
        _print_metrics(f"VOL-SCALED 15% — {label}", vol_m)

    comparison_rows = []
    for label, tickers in (
        ("ES+GC_raw", vault_tickers),
        ("ES+GC_vol15", vault_tickers),
        ("ES+NQ+GC_raw", config_tickers),
        ("ES+NQ+GC_vol15", config_tickers),
    ):
        is_vol = "vol15" in label
        if is_vol:
            s = vol_scaled_portfolio_returns(
                tickers, start, end, BLOG_PARAMETER_WINDOWS, target_volatility=0.15
            )
        else:
            _, port = raw_strategy_returns(tickers, start, end, BLOG_PARAMETER_WINDOWS)
            s = port["portfolio"]
        m = summarize_return_series(s)
        m["scenario"] = label
        comparison_rows.append(m)
    _write_df(pd.DataFrame(comparison_rows), out / "01_raw_vs_vol_comparison.csv")

    # --- 2. Bond FOMC leg (TY / US vs NQ) ---
    fomc_scenarios: list[tuple[str, tuple[Ticker, ...]]] = [
        ("fomc_ES_GC_NQ", (Ticker.ES, Ticker.GC, Ticker.NQ)),
        ("fomc_ES_GC_TY", (Ticker.ES, Ticker.GC, Ticker.TY)),
        ("fomc_ES_GC_US", (Ticker.ES, Ticker.GC, Ticker.US)),
        ("fomc_ES_GC_TY_US", (Ticker.ES, Ticker.GC, Ticker.TY, Ticker.US)),
        ("fomc_TY_only", (Ticker.TY,)),
        ("fomc_ES_GC_only", (Ticker.ES, Ticker.GC)),
    ]
    fomc_rows = []
    for name, fomc_tickers in fomc_scenarios:
        port = _fomc_only_returns(fomc_tickers, start, end)
        m = summarize_return_series(port)
        m["scenario"] = name
        fomc_rows.append(m)
        print(f"\nFOMC-only RAW (D-2..D0) {name}: total={m.get('total_return', 0)*100:.1f}% maxDD={m.get('max_drawdown', 0)*100:.1f}%")
    _write_df(pd.DataFrame(fomc_rows), out / "02_fomc_leg_comparison.csv")

    # Full ensemble with blog-like FOMC on bonds: ES/NQ equity holidays, GC gold, FOMC on ES+GC+TY
    # Approximated by per-ticker calendar_ensemble where TY only gets fomc via combined_signal
    ty_combo_port = _fomc_only_returns((Ticker.ES, Ticker.GC, Ticker.TY), start, end)
    # Add rough "full ensemble" proxy: merge ES calendar_ensemble + GC + TY fomc-only weighted
    _, esgc_raw = raw_strategy_returns((Ticker.ES, Ticker.GC), start, end, BLOG_PARAMETER_WINDOWS)
    ty_fomc = _fomc_only_returns((Ticker.TY,), start, end)
    # ES+GC full calendar + TY on FOMC days only (manual OR on portfolio returns by day)
    proxy = pd.concat(
        [
            esgc_raw["portfolio"].rename("esgc"),
            ty_fomc.rename("ty_fomc"),
        ],
        axis=1,
    ).fillna(0.0)
    proxy["blog_proxy_es_gc_ty_fomc"] = proxy["esgc"].copy()
    fomc_days = ty_fomc[ty_fomc != 0].index
    proxy.loc[fomc_days, "blog_proxy_es_gc_ty_fomc"] = (
        proxy.loc[fomc_days, "esgc"] + proxy.loc[fomc_days, "ty_fomc"]
    ) / 2.0
    m_proxy = summarize_return_series(proxy["blog_proxy_es_gc_ty_fomc"])
    _print_metrics("PROXY full calendar ES+GC + TY on FOMC (raw)", m_proxy)

    # --- 3. Event-level attribution ---
    bundle = load_calendar_bundle()
    index = TradingDayIndex(load_es_trading_sessions(start.date(), end.date()))
    ticker_lr = {
        t.name: load_ticker_log_returns(t, start, end)
        for t in (Ticker.ES, Ticker.NQ, Ticker.GC, Ticker.TY, Ticker.US)
    }
    events = event_trade_table(
        bundle, index, BLOG_PARAMETER_WINDOWS, ticker_lr, start_year=2005, end_year=2022
    )
    _write_df(events, out / "03_event_trades.csv")

    agg = aggregate_event_stats(events)
    _write_df(agg, out / "04_event_stats_by_year.csv")

    # Worst events 2010-2013
    bad = events[(events["year"] >= 2010) & (events["year"] <= 2013)].sort_values(
        "trade_log_return"
    )
    _write_df(bad.head(30), out / "05_worst_events_2010_2013.csv")

    print("\n=== Worst 10 event trades (2010-2013) ===")
    print(
        bad.head(10)[
            ["year", "event_type", "event_id", "d0", "ticker", "trade_log_return"]
        ].to_string(index=False)
    )

    # --- 4. Window variants (narrative vs parameter section) ---
    window_rows = []
    for windows in (BLOG_NARRATIVE_WINDOWS, BLOG_PARAMETER_WINDOWS):
        _, port = raw_strategy_returns(vault_tickers, start, end, windows)
        m = summarize_return_series(port["portfolio"])
        m["window_set"] = windows.label
        window_rows.append(m)
        vol = vol_scaled_portfolio_returns(
            vault_tickers, start, end, windows, target_volatility=0.15
        )
        mv = summarize_return_series(vol)
        mv["window_set"] = windows.label + "_vol15"
        window_rows.append(mv)
    _write_df(pd.DataFrame(window_rows), out / "06_window_variant_comparison.csv")

    for windows in (BLOG_NARRATIVE_WINDOWS, BLOG_PARAMETER_WINDOWS):
        _, port = raw_strategy_returns(vault_tickers, start, end, windows)
        print(f"\n=== Annual RAW returns ({windows.label}) ES+GC ===")
        annual = port["portfolio"].groupby(port.index.year).apply(lambda s: (1 + s).prod() - 1)
        for y, r in annual.items():
            if 2008 <= y <= 2016:
                print(f"  {y}: {r*100:6.2f}%")

    summary = {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "outputs": str(out),
        "notes": [
            "Raw = signal x log_return, equal-weight across tickers.",
            "Vol15 = TFPortfolio vol-targeted position_fraction at 15% tau.",
            "Event trades = sum of log returns over each holiday/FOMC window.",
        ],
    }
    (out / "README_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. See {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
