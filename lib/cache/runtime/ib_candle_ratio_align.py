"""Ratio-align IBKR daily OHLC to existing cache before ``upsert_candles``.

Norgate-backed history in the central cache uses continuous back-adjusted
symbols (``&*_CCB``). Interactive Brokers ``CONTFUT`` uses a different roll and
adjustment rule set. When appending IB rows, we:

1. **Append-only overlap policy:** keep only sessions strictly *after* the
   current cache end date so parquet-backed levels are not bulk-overwritten by
   a long IB lookback.
2. **Junction ratio:**** multiply the appended block's OHLC by a
   single positive factor so the first appended close matches the last
   pre-existing close (returns within the block are unchanged).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

_OHLC: tuple[str, str, str, str] = ("open", "high", "low", "close")
_EPS = 1e-12


@dataclass(frozen=True)
class IbCacheAppendResult:
    """Outcome of filtering + optional ratio alignment for an IB daily batch."""

    candles_df: pd.DataFrame
    applied_ratio: bool
    ratio: float | None
    rows_in: int
    rows_kept: int
    anchor_datetime: pd.Timestamp | None
    join_datetime: pd.Timestamp | None
    skip_reason: str | None


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "datetime" in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.drop(columns=["datetime"])
    if "datetime" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out = out.set_index(pd.to_datetime(out["datetime"], errors="coerce"))
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    if "datetime" in out.columns:
        out = out.drop(columns=["datetime"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "datetime"
    return out


def _output_like_incoming(incoming_template: pd.DataFrame, indexed: pd.DataFrame) -> pd.DataFrame:
    """Reset to ``datetime`` column if the template used columns (not index)."""
    if incoming_template.empty and indexed.empty:
        return incoming_template.copy()
    reset = indexed.reset_index()
    if "datetime" not in reset.columns:
        first = reset.columns[0]
        reset = reset.rename(columns={first: "datetime"})
    return reset


def prepare_ib_rows_for_central_cache_append(
    existing_cache_df: pd.DataFrame,
    incoming_ib_df: pd.DataFrame,
    *,
    apply_junction_ratio: bool,
    ohlc_cols: tuple[str, str, str, str] = _OHLC,
) -> IbCacheAppendResult:
    """Filter IB rows to append-only sessions and optionally ratio-scale OHLC.

    Parameters
    ----------
    existing_cache_df
        Current daily frame for the ticker (index or ``datetime`` column). May be
        empty when the cache has no prior data for this series.
    incoming_ib_df
        IB-fetched daily rows for one ticker (same shape conventions).
    apply_junction_ratio
        When ``True``, scale kept OHLC by
        ``close_anchor / close_join``. When ``False`` (e.g. ``STK``), only the
        append-only filter runs.
    """
    rows_in = len(incoming_ib_df)
    if incoming_ib_df.empty or rows_in == 0:
        return IbCacheAppendResult(
            candles_df=incoming_ib_df.copy(),
            applied_ratio=False,
            ratio=None,
            rows_in=rows_in,
            rows_kept=0,
            anchor_datetime=None,
            join_datetime=None,
            skip_reason="empty_incoming",
        )

    inc = _ensure_datetime_index(incoming_ib_df)
    exist = _ensure_datetime_index(existing_cache_df) if not existing_cache_df.empty else pd.DataFrame()

    if exist.empty:
        out_idx = inc.copy()
        for col in ohlc_cols:
            if col in out_idx.columns:
                out_idx[col] = out_idx[col].astype("float64", copy=False)
        result_df = _output_like_incoming(incoming_ib_df, out_idx)
        return IbCacheAppendResult(
            candles_df=result_df,
            applied_ratio=False,
            ratio=None,
            rows_in=rows_in,
            rows_kept=len(result_df),
            anchor_datetime=None,
            join_datetime=None,
            skip_reason=None,
        )

    existing_max = pd.Timestamp(exist.index.max()).normalize()
    inc_day = pd.to_datetime(inc.index, errors="coerce").normalize()
    mask = inc_day > existing_max
    inc_kept = inc.loc[mask].copy()
    rows_kept = len(inc_kept)

    if rows_kept == 0:
        return IbCacheAppendResult(
            candles_df=incoming_ib_df.iloc[0:0].copy(),
            applied_ratio=False,
            ratio=None,
            rows_in=rows_in,
            rows_kept=0,
            anchor_datetime=None,
            join_datetime=None,
            skip_reason="no_new_sessions",
        )

    anchor_ts = pd.Timestamp(exist.index.max())
    anchor_close = float(exist.loc[anchor_ts, "close"])
    join_ts = pd.Timestamp(inc_kept.index.min())
    join_close = float(inc_kept.loc[join_ts, "close"])

    ratio: float | None = None
    applied = False
    if apply_junction_ratio:
        if abs(join_close) < _EPS:
            logger.warning(
                "IB junction ratio skipped: join close ~0 (ticker join_ts=%s)",
                join_ts,
            )
        else:
            ratio = anchor_close / join_close
            for col in ohlc_cols:
                if col in inc_kept.columns:
                    inc_kept[col] = inc_kept[col].astype("float64", copy=False) * ratio
            applied = True
            logger.info(
                "IB append ratio-align: anchor=%s close=%.6f join=%s ib_close=%.6f r=%.8f rows=%d",
                anchor_ts,
                anchor_close,
                join_ts,
                join_close,
                ratio,
                rows_kept,
            )

    result_df = _output_like_incoming(incoming_ib_df, inc_kept)
    return IbCacheAppendResult(
        candles_df=result_df,
        applied_ratio=applied,
        ratio=ratio,
        rows_in=rows_in,
        rows_kept=rows_kept,
        anchor_datetime=anchor_ts,
        join_datetime=join_ts,
        skip_reason=None,
    )
