"""Conditional-returns binning — slice a strategy's signal returns by an indicator.

Given a **best-performing** signal (a single bias-node combo) and a **conditioning**
indicator (RSI, ADX, ATR, EWMAC, …), bucket the strategy's per-bar returns by the value of
the conditioning indicator and report per-bucket stats (mean return, Sharpe, t-stat, hit rate,
…). An optional **binary regime filter** (an indicator + threshold) cross-cuts each bucket into
on/off so you can read "the strategy's edge in this regime, on vs off".

This mirrors the in-sample decile path
(:func:`research.feature.binning.pipeline._attach_investigation_strategy_returns`): the strategy
return per (ticker, date) is ``strategy_signal[t] × forward_return[t]`` — exactly the quantity
the exploration phase scores combos on — so the buckets describe the *same* returns that picked
the winner, never a re-derived proxy. Both the conditioning indicator (observed at decision time
``t``) and the forward return are aligned lookahead-free by ``extract_features_for_bias_node``.

Pure / in-memory: no file writes, no charts. The frontend API adapter shapes the result to JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from features.extraction.feature_extractor import extract_features_for_bias_node
from lib.core.enums import Direction, Ticker, TimeFrame
from research.feature.in_sample.metric_helpers import compute_param_sensitivity_metric

BinMode = Literal["quantile", "fixed"]

_TARGET_COL = "log_return"


@dataclass(frozen=True)
class IndicatorSpec:
    """A conditioning / regime indicator: a bias-node module + its (scalar) params."""

    module_name: str
    params: dict[str, Any] = field(default_factory=dict)

    def bias_spec(self, timeframe: TimeFrame) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "timeframes": [timeframe],
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class RegimeSpec:
    """A binary on/off regime: indicator value compared to a threshold."""

    indicator: IndicatorSpec
    threshold: float
    above: bool = True  # on when value >= threshold (above) else value <= threshold

    def label(self) -> str:
        op = "≥" if self.above else "≤"
        return f"{_indicator_label(self.indicator)} {op} {_fmt_num(self.threshold)}"


@dataclass(frozen=True)
class ConditionalReturnsRequest:
    """All inputs for one conditional-returns analysis."""

    tickers: tuple[Ticker, ...]
    timeframe: TimeFrame
    start: datetime
    end: datetime
    data_feed: str  # "cfd" | "futures" — set as the process-global research feed before extraction
    direction: Direction
    strategy: IndicatorSpec  # the best-performing combo (scalar params)
    condition: IndicatorSpec  # the indicator whose value defines the buckets
    bin_mode: BinMode = "quantile"
    n_bins: int = 5
    edges: tuple[float, ...] = ()  # interior thresholds for bin_mode="fixed"
    regime: RegimeSpec | None = None
    per_ticker: bool = False


@dataclass(frozen=True)
class BinStat:
    """Per-bucket return statistics (one row of the result)."""

    bin_index: int
    label: str
    lo: float | None
    hi: float | None
    regime: str | None  # "on" | "off" | None
    ticker: str | None  # set only when per_ticker
    n_obs: int
    mean_return: float
    sharpe: float
    sortino: float
    t_stat: float
    hit_rate: float
    cumulative: float
    instrument_mean_return: float


@dataclass(frozen=True)
class ConditionalReturnsResult:
    available: bool
    reason: str | None
    bins: list[BinStat]
    condition_label: str
    strategy_label: str
    regime_label: str | None
    bin_mode: BinMode
    n_bins: int
    tickers: list[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_conditional_returns(req: ConditionalReturnsRequest) -> ConditionalReturnsResult:
    """Bucket the strategy's signal returns by the conditioning indicator's value."""

    if req.bin_mode == "fixed" and not req.edges:
        return _empty(req, "Fixed-threshold mode requires at least one edge.")
    if req.bin_mode == "quantile" and req.n_bins < 2:
        return _empty(req, "Quantile mode requires n_bins >= 2.")

    # The conditioning indicator's value + forward return drive everything; the EWSD blend is
    # irrelevant here (we score on raw log_return, not the EWSD-normalised variant), so we do not
    # touch set_ewsd_blend_weights. We DO set the research feed so the right candles are loaded.
    _set_feed(req.data_feed)

    tickers = list(req.tickers)
    panel = _condition_panel(req.condition, tickers, req)
    if panel.empty:
        return _empty(req, "No conditioning-indicator data over the run's window.")

    strat = _signal_frame(req.strategy, tickers, req, value_name="strategy_signal")
    if strat.empty:
        return _empty(req, "No strategy-signal data over the run's window.")

    merged = panel.merge(strat, on=["datetime", "ticker"], how="inner")
    if req.regime is not None:
        regime = _signal_frame(req.regime.indicator, tickers, req, value_name="regime_value")
        merged = merged.merge(regime, on=["datetime", "ticker"], how="inner")

    merged = merged.dropna(subset=["feature", "target", "strategy_signal"])
    # Conditional *strategy* returns: only the bars the strategy actually holds a position. Flat
    # bars (signal == 0) carry no information about the strategy's edge in a regime.
    merged = merged[pd.to_numeric(merged["strategy_signal"], errors="coerce").fillna(0.0) != 0.0]
    if merged.empty:
        return _empty(req, "Strategy is never active over the analysed window.")

    merged["strategy_return"] = merged["strategy_signal"].astype(float) * merged["target"].astype(float)
    if req.regime is not None:
        merged["regime"] = _regime_state(merged["regime_value"], req.regime)

    bins = _bucketize(merged, req)
    if not bins:
        return _empty(req, "Binning produced no populated buckets.")
    return ConditionalReturnsResult(
        available=True,
        reason=None,
        bins=bins,
        condition_label=_indicator_label(req.condition),
        strategy_label=_indicator_label(req.strategy),
        regime_label=req.regime.label() if req.regime is not None else None,
        bin_mode=req.bin_mode,
        n_bins=req.n_bins,
        tickers=sorted(merged["ticker"].unique().tolist()),
    )


# ---------------------------------------------------------------------------
# Extraction → aligned (datetime, ticker) frames
# ---------------------------------------------------------------------------


def _condition_panel(indicator: IndicatorSpec, tickers: list[Ticker], req: ConditionalReturnsRequest) -> pd.DataFrame:
    """``[datetime, ticker, feature, target]`` for the conditioning indicator."""

    features_df, targets_df = _extract(indicator, tickers, req)
    value_col = _pick_value_column(features_df, indicator.module_name)
    if value_col is None or _TARGET_COL not in targets_df.columns:
        return pd.DataFrame()
    feat = _as_dt_frame(features_df[[value_col, "ticker"]].rename(columns={value_col: "feature"}))
    tgt = _as_dt_frame(targets_df[[_TARGET_COL, "ticker"]].rename(columns={_TARGET_COL: "target"}))
    return feat.merge(tgt, on=["datetime", "ticker"], how="inner")


def _signal_frame(
    indicator: IndicatorSpec,
    tickers: list[Ticker],
    req: ConditionalReturnsRequest,
    *,
    value_name: str,
) -> pd.DataFrame:
    """``[datetime, ticker, <value_name>]`` for a signal / regime indicator."""

    features_df, _ = _extract(indicator, tickers, req)
    value_col = _pick_value_column(features_df, indicator.module_name)
    if value_col is None:
        return pd.DataFrame()
    return _as_dt_frame(features_df[[value_col, "ticker"]].rename(columns={value_col: value_name}))


def _extract(indicator: IndicatorSpec, tickers: list[Ticker], req: ConditionalReturnsRequest) -> tuple[pd.DataFrame, pd.DataFrame]:
    return extract_features_for_bias_node(
        bias_spec=indicator.bias_spec(req.timeframe),
        ticker=tickers,
        start=req.start,
        end=req.end,
        target_col=_TARGET_COL,
        use_cache=True,
        populate_on_miss=True,
    )


def _as_dt_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reset the datetime index to a ``datetime`` column and normalise ``ticker`` to a str name."""

    out = frame.rename_axis("datetime").reset_index()
    out["ticker"] = out["ticker"].map(lambda x: x.name if hasattr(x, "name") else str(x))
    return out


def _pick_value_column(features_df: pd.DataFrame, module_name: str) -> str | None:
    """The indicator's value column: prefer ``{module}_…``; never the auto-added EWSD column."""

    cols = [c for c in features_df.columns if c != "ticker"]
    own = [c for c in cols if c.startswith(f"{module_name}_")]
    if own:
        return own[0]
    non_ewsd = [c for c in cols if "ewsd" not in c.lower()]
    return non_ewsd[0] if non_ewsd else (cols[0] if cols else None)


# ---------------------------------------------------------------------------
# Bucketing + per-bucket metrics
# ---------------------------------------------------------------------------


def _regime_state(values: pd.Series, regime: RegimeSpec) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    on = v >= regime.threshold if regime.above else v <= regime.threshold
    return on.map(lambda flag: "on" if bool(flag) else "off")


def _bucketize(merged: pd.DataFrame, req: ConditionalReturnsRequest) -> list[BinStat]:
    """Assign buckets (quantile/fixed), optionally per ticker, then summarise each bucket."""

    group_frames: list[tuple[str | None, pd.DataFrame]] = (
        [(str(tk), sub) for tk, sub in merged.groupby("ticker", sort=True)]
        if req.per_ticker
        else [(None, merged)]
    )

    stats: list[BinStat] = []
    for ticker, frame in group_frames:
        binned = _assign_bins(frame, req)
        if binned.empty:
            continue
        regimes = ["on", "off"] if req.regime is not None else [None]
        for regime in regimes:
            scope = binned if regime is None else binned[binned["regime"] == regime]
            for bin_index, bucket in scope.groupby("bin_index", sort=True):
                stats.append(_bin_stat(int(bin_index), bucket, regime, ticker, req.timeframe))
    return stats


def _assign_bins(frame: pd.DataFrame, req: ConditionalReturnsRequest) -> pd.DataFrame:
    feature = pd.to_numeric(frame["feature"], errors="coerce")
    out = frame.copy()
    if req.bin_mode == "fixed":
        edges = sorted(float(e) for e in req.edges)
        cuts = [-np.inf, *edges, np.inf]
        cat = pd.cut(feature, bins=cuts, labels=False, right=False)
        out["bin_index"] = cat.astype("Int64")
        out["__lo"] = cat.map(lambda i: cuts[int(i)] if pd.notna(i) else np.nan)
        out["__hi"] = cat.map(lambda i: cuts[int(i) + 1] if pd.notna(i) else np.nan)
    else:
        try:
            codes, bins = pd.qcut(feature, q=req.n_bins, labels=False, duplicates="drop", retbins=True)
        except ValueError:
            return pd.DataFrame()
        out["bin_index"] = pd.Series(codes, index=frame.index).astype("Int64")
        edge_lo = {i: float(bins[i]) for i in range(len(bins) - 1)}
        edge_hi = {i: float(bins[i + 1]) for i in range(len(bins) - 1)}
        out["__lo"] = out["bin_index"].map(lambda i: edge_lo.get(int(i)) if pd.notna(i) else np.nan)
        out["__hi"] = out["bin_index"].map(lambda i: edge_hi.get(int(i)) if pd.notna(i) else np.nan)
    return out.dropna(subset=["bin_index"])


def _bin_stat(
    bin_index: int,
    bucket: pd.DataFrame,
    regime: str | None,
    ticker: str | None,
    timeframe: TimeFrame,
) -> BinStat:
    strat = pd.to_numeric(bucket["strategy_return"], errors="coerce").dropna()
    instrument = pd.to_numeric(bucket["target"], errors="coerce").dropna()
    lo = _scalar(bucket["__lo"])
    hi = _scalar(bucket["__hi"])
    return BinStat(
        bin_index=bin_index,
        label=_bin_label(lo, hi),
        lo=lo,
        hi=hi,
        regime=regime,
        ticker=ticker,
        n_obs=int(len(strat)),
        mean_return=_metric(strat, "mean", timeframe),
        sharpe=_metric(strat, "sharpe", timeframe),
        sortino=_metric(strat, "sortino", timeframe),
        t_stat=_metric(strat, "t_stat", timeframe),
        hit_rate=float((strat > 0).mean()) if len(strat) else float("nan"),
        cumulative=float(strat.sum()) if len(strat) else float("nan"),
        instrument_mean_return=_metric(instrument, "mean", timeframe),
    )


def _metric(series: pd.Series, name: str, timeframe: TimeFrame) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return float(s.iloc[0]) if (name == "mean" and len(s) == 1) else float("nan")
    try:
        return float(compute_param_sensitivity_metric(s, name, timeframe))
    except Exception:  # noqa: BLE001 — a degenerate bucket should not 500 the whole analysis
        return float("nan")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _set_feed(data_feed: str) -> None:
    from lib.core.research_feed import set_research_feed

    set_research_feed(data_feed)


def _scalar(col: pd.Series) -> float | None:
    s = pd.to_numeric(col, errors="coerce").dropna()
    if s.empty:
        return None
    v = float(s.iloc[0])
    return None if not np.isfinite(v) else v


def _bin_label(lo: float | None, hi: float | None) -> str:
    lo_inf = lo is None or not np.isfinite(lo)
    hi_inf = hi is None or not np.isfinite(hi)
    if lo_inf and not hi_inf:
        return f"< {_fmt_num(hi)}"
    if hi_inf and not lo_inf:
        return f"≥ {_fmt_num(lo)}"
    if lo_inf and hi_inf:
        return "all"
    return f"{_fmt_num(lo)} – {_fmt_num(hi)}"


def _indicator_label(indicator: IndicatorSpec) -> str:
    if not indicator.params:
        return indicator.module_name
    body = ", ".join(f"{k}={_fmt_num(v)}" for k, v in indicator.params.items())
    return f"{indicator.module_name}({body})"


def _fmt_num(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f == int(f) else f"{f:.2f}"


def _empty(req: ConditionalReturnsRequest, reason: str) -> ConditionalReturnsResult:
    return ConditionalReturnsResult(
        available=False,
        reason=reason,
        bins=[],
        condition_label=_indicator_label(req.condition),
        strategy_label=_indicator_label(req.strategy),
        regime_label=req.regime.label() if req.regime is not None else None,
        bin_mode=req.bin_mode,
        n_bins=req.n_bins,
        tickers=[],
    )
