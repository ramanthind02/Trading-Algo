"""Calendar ensemble diagnostics: event PnL, window variants, return paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from data_platform.events.calendar_loader import CalendarBundle, HolidayAssetBucket, load_calendar_bundle
from data_platform.events.trading_day_index import TradingDayIndex, load_es_trading_sessions
from lib.core.enums import Ticker, TimeFrame
from lib.core import helpers


@dataclass(frozen=True)
class WindowParams:
    """Calendar window offsets (trading days relative to D0)."""

    equity_entry: int
    equity_exit: int
    gold_entry: int
    gold_exit: int
    fomc_entry: int = -2
    fomc_exit: int = 0
    label: str = "default"


BLOG_NARRATIVE_WINDOWS = WindowParams(
    equity_entry=-3,
    equity_exit=0,
    gold_entry=-1,
    gold_exit=1,
    label="blog_narrative",
)

BLOG_PARAMETER_WINDOWS = WindowParams(
    equity_entry=-4,
    equity_exit=0,
    gold_entry=-2,
    gold_exit=1,
    label="blog_parameters",
)


def load_ticker_log_returns(
    ticker: Ticker,
    start: datetime,
    end: datetime,
) -> pd.Series:
    """Daily log returns indexed by session date."""
    df = helpers.load_data(ticker, TimeFrame.D, start=start, end=end)
    df = df.sort_values("datetime")
    dt = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
    close = df["close"].astype(float)
    lr = np.log(close / close.shift(1))
    out = pd.Series(lr.to_numpy(), index=dt, name="log_return")
    return out.dropna()


def active_sessions_for_windows(
    bundle: CalendarBundle,
    windows: WindowParams,
    trading_index: TradingDayIndex,
) -> dict[str, frozenset[date]]:
    """Active session sets keyed by leg name."""
    equity_d0 = tuple(
        e.d0 for e in bundle.holiday_events if e.asset_bucket == HolidayAssetBucket.EQUITY
    )
    gold_d0 = tuple(
        e.d0 for e in bundle.holiday_events if e.asset_bucket == HolidayAssetBucket.GOLD
    )
    return {
        "equity_holiday": trading_index.active_sessions(
            equity_d0, windows.equity_entry, windows.equity_exit
        ),
        "gold_holiday": trading_index.active_sessions(
            gold_d0, windows.gold_entry, windows.gold_exit
        ),
        "fomc": trading_index.active_sessions(
            bundle.fomc_decision_dates, windows.fomc_entry, windows.fomc_exit
        ),
    }


def combined_signal_series(
    session_dates: pd.DatetimeIndex,
    *,
    ticker: Ticker,
    legs: dict[str, frozenset[date]],
) -> pd.Series:
    """OR-combine leg active sets for one ticker."""
    date_only = {d.date() if hasattr(d, "date") else d for d in session_dates}
    signals: list[int] = []
    for dt in session_dates:
        d = dt.date() if hasattr(dt, "date") else dt
        active = 0
        if ticker in (Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY):
            if d in legs["equity_holiday"]:
                active = 1
        if ticker == Ticker.GC and d in legs["gold_holiday"]:
            active = 1
        if d in legs["fomc"]:
            active = 1
        signals.append(active)
    return pd.Series(signals, index=session_dates, dtype=int)


def raw_strategy_returns(
    tickers: Sequence[Ticker],
    start: datetime,
    end: datetime,
    windows: WindowParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ticker long frame and equal-weight portfolio raw signal × log_return."""
    bundle = load_calendar_bundle()
    index = TradingDayIndex(load_es_trading_sessions(start.date(), end.date()))
    legs = active_sessions_for_windows(bundle, windows, index)

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        lr = load_ticker_log_returns(ticker, start, end)
        sig = combined_signal_series(lr.index, ticker=ticker, legs=legs)
        aligned = pd.DataFrame(
            {
                "ticker": ticker.name,
                "signal": sig.reindex(lr.index).fillna(0).astype(int),
                "log_return": lr,
            }
        )
        aligned["strategy_return"] = aligned["signal"] * aligned["log_return"]
        frames.append(aligned)

    long = pd.concat(frames).sort_index()
    port = (
        long.groupby(long.index)["strategy_return"]
        .mean()
        .rename("portfolio")
        .to_frame()
    )
    return long, port


def summarize_return_series(daily: pd.Series) -> dict[str, float]:
    """Key metrics from a daily return series."""
    if daily.empty:
        return {}
    equity = (1.0 + daily).cumprod()
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    by_year = daily.groupby(daily.index.year).apply(lambda s: (1.0 + s).prod() - 1.0)
    active = daily[daily != 0]
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(equity.iloc[-1] ** (252.0 / max(len(daily), 1)) - 1.0),
        "max_drawdown": float(dd.min()),
        "max_dd_date": str(dd.idxmin().date()) if hasattr(dd.idxmin(), "date") else str(dd.idxmin()),
        "vol_ann": float(daily.std() * np.sqrt(252)),
        "sharpe_daily": float(daily.mean() / daily.std() * np.sqrt(252))
        if daily.std() > 1e-12
        else float("nan"),
        "sharpe_active_days": float(active.mean() / active.std() * np.sqrt(252))
        if len(active) > 1 and active.std() > 1e-12
        else float("nan"),
        "time_in_market": float((daily != 0).mean()),
        "mean_return_on_active_days": float(active.mean()) if len(active) else float("nan"),
        "worst_year": int(by_year.idxmin()),
        "worst_year_return": float(by_year.min()),
    }


def event_trade_table(
    bundle: CalendarBundle,
    trading_index: TradingDayIndex,
    windows: WindowParams,
    ticker_returns: dict[str, pd.Series],
    start_year: int = 2005,
    end_year: int = 2025,
) -> pd.DataFrame:
    """One row per holiday or FOMC event with cumulative log return over the trade window."""
    rows: list[dict[str, object]] = []

    def sessions_in_window(d0: date, entry: int, exit_off: int) -> list[date]:
        active = trading_index.active_sessions((d0,), entry, exit_off)
        return sorted(active)

    for event in bundle.holiday_events:
        if not (start_year <= event.d0.year <= end_year):
            continue
        if event.asset_bucket == HolidayAssetBucket.EQUITY:
            entry, exit_off = windows.equity_entry, windows.equity_exit
            tickers = ("ES", "NQ")
        else:
            entry, exit_off = windows.gold_entry, windows.gold_exit
            tickers = ("GC",)
        sess = sessions_in_window(event.d0, entry, exit_off)
        for tname in tickers:
            lr = ticker_returns.get(tname)
            if lr is None:
                continue
            trade_rets = [float(lr.get(pd.Timestamp(s), 0.0)) for s in sess if pd.Timestamp(s) in lr.index]
            rows.append(
                {
                    "event_type": "holiday",
                    "event_id": event.holiday_id,
                    "d0": event.d0.isoformat(),
                    "closure": event.closure_date.isoformat(),
                    "ticker": tname,
                    "n_sessions": len(sess),
                    "trade_log_return": float(np.sum(trade_rets)),
                    "year": event.d0.year,
                }
            )

    for d0 in bundle.fomc_decision_dates:
        if not (start_year <= d0.year <= end_year):
            continue
        sess = sessions_in_window(d0, windows.fomc_entry, windows.fomc_exit)
        for tname in ticker_returns:
            lr = ticker_returns[tname]
            trade_rets = [float(lr.get(pd.Timestamp(s), 0.0)) for s in sess if pd.Timestamp(s) in lr.index]
            rows.append(
                {
                    "event_type": "fomc",
                    "event_id": "fomc_decision",
                    "d0": d0.isoformat(),
                    "closure": "",
                    "ticker": tname,
                    "n_sessions": len(sess),
                    "trade_log_return": float(np.sum(trade_rets)),
                    "year": d0.year,
                }
            )

    return pd.DataFrame(rows)


def aggregate_event_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize event trades by year, event_type, ticker."""
    if events.empty:
        return events
    return (
        events.groupby(["year", "event_type", "ticker"], as_index=False)
        .agg(
            n_trades=("trade_log_return", "count"),
            total_log_return=("trade_log_return", "sum"),
            mean_log_return=("trade_log_return", "mean"),
            win_rate=("trade_log_return", lambda s: float((s > 0).mean())),
        )
        .sort_values(["year", "event_type", "ticker"])
    )


def vol_scaled_portfolio_returns(
    tickers: Sequence[Ticker],
    start: datetime,
    end: datetime,
    windows: WindowParams,
    *,
    target_volatility: float = 0.15,
    train_end: datetime | None = None,
) -> pd.Series:
    """Equal-weight vol-targeted portfolio daily returns (production-style)."""
    from research.feature.in_sample.data_loader import expand_bias_specs
    from research.feature.research_table_exports import _build_portfolio_positions_df
    from ensemble.portfolio_impl.portfolio_tester import calculate_strategy_returns_from_positions

    spec = {
        "module_name": "calendar_ensemble",
        "timeframes": [TimeFrame.D],
        "params": {
            "equity_entry_offset": windows.equity_entry,
            "equity_exit_offset": windows.equity_exit,
            "gold_entry_offset": windows.gold_entry,
            "gold_exit_offset": windows.gold_exit,
            "fomc_entry_offset": windows.fomc_entry,
            "fomc_exit_offset": windows.fomc_exit,
        },
    }
    train_cutoff = train_end or datetime(2018, 12, 31)
    candles = helpers.load_data_multi_ticker(list(tickers), TimeFrame.D, start=start, end=end)
    train_c = candles[pd.to_datetime(candles["datetime"]) <= train_cutoff]

    leg_returns: list[pd.Series] = []
    for ticker in tickers:
        from features.extraction.feature_extractor import extract_features_for_bias_node

        feats, _ = extract_features_for_bias_node(
            spec, [ticker], start, end, "log_return", use_cache=True
        )
        sig_col = [c for c in feats.columns if c != "ticker"][0]
        pos = _build_portfolio_positions_df(
            feats[sig_col],
            feats["ticker"],
            train_c,
            candles,
            TimeFrame.D,
            target_volatility=target_volatility,
        )
        rets = calculate_strategy_returns_from_positions(
            pos, candles, instrument_return_kind="log_intraday"
        )
        if isinstance(rets.index, pd.MultiIndex):
            rets = rets.groupby(level=0).sum()
        leg_returns.append(pd.Series(rets, name=ticker.name))

    combined = pd.concat(leg_returns, axis=1).mean(axis=1).sort_index()
    combined.index = pd.to_datetime(combined.index)
    return combined
