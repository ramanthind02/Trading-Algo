"""Daily EWSD volatility service with incremental state persistence.

This module centralizes EWSD volatility computation for execution and research:
- Compute causal daily EWSD volatility series from daily candles.
- Align daily volatility to arbitrary candle timeframes by ticker/date forward-fill.
- Incrementally update persisted per-ticker EWSD state for live trading.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from utils.core.ticker_key import normalize_ticker_key

DEFAULT_ANNUAL_VOL = 0.20
LAMBDA_SHORT_DEFAULT = 0.06061
LONG_RUN_WINDOW_DEFAULT = 2520
BLEND_SHORT_WEIGHT_DEFAULT = 0.7
BLEND_LONG_WEIGHT_DEFAULT = 0.3


@dataclass(frozen=True)
class EWSDVolatilityConfig:
    """Configuration for EWSD volatility estimation."""

    lambda_short: float = LAMBDA_SHORT_DEFAULT
    long_run_window: int = LONG_RUN_WINDOW_DEFAULT
    blend_short_weight: float = BLEND_SHORT_WEIGHT_DEFAULT
    blend_long_weight: float = BLEND_LONG_WEIGHT_DEFAULT
    annualization_factor: float = 16.0
    default_annual_vol: float = DEFAULT_ANNUAL_VOL


@dataclass
class _TickerState:
    """Mutable incremental EWSD state for a single ticker."""

    last_datetime: Optional[pd.Timestamp] = None
    last_close: Optional[float] = None
    ewma_variance: Optional[float] = None
    sigma_long: float = 0.01
    returns_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=LONG_RUN_WINDOW_DEFAULT)
    )
    last_long_run_month: Optional[str] = None


class DailyEWSDVolatilityService:
    """Compute, align, and persist causal daily EWSD volatility."""

    def __init__(
        self,
        config: Optional[EWSDVolatilityConfig] = None,
        store_dir: Optional[str] = None,
    ) -> None:
        self.config = config or EWSDVolatilityConfig()
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self._state_path = self.store_dir / "ewsd_state.json" if self.store_dir else None
        self._history_path = self.store_dir / "daily_ewsd_volatility.parquet" if self.store_dir else None

        if self.store_dir is not None:
            self.store_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_candles(candles_df: pd.DataFrame) -> pd.DataFrame:
        if candles_df.empty:
            return candles_df.copy()

        required = {"datetime", "ticker", "close"}
        missing = required - set(candles_df.columns)
        if missing:
            raise ValueError(f"Candles missing required columns: {sorted(missing)}")

        out = candles_df[["datetime", "ticker", "close"]].copy()
        out["datetime"] = pd.to_datetime(out["datetime"]).dt.tz_localize(None)
        out["ticker"] = out["ticker"].map(normalize_ticker_key)
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out = out.dropna(subset=["datetime", "ticker", "close"])

        # Collapse to one close per ticker/day (last close).
        out["date"] = out["datetime"].dt.normalize()
        out = (
            out.sort_values(["ticker", "datetime"])
            .groupby(["ticker", "date"], as_index=False)
            .agg(close=("close", "last"))
            .rename(columns={"date": "datetime"})
        )
        return out[["datetime", "ticker", "close"]]

    def _safe_annual_from_state(self, state: _TickerState) -> float:
        if state.ewma_variance is None or not np.isfinite(state.ewma_variance) or state.ewma_variance <= 0.0:
            return float(self.config.default_annual_vol)

        sigma_short = float(np.sqrt(max(state.ewma_variance, 0.0)))
        blended_daily = (
            self.config.blend_short_weight * sigma_short
            + self.config.blend_long_weight * float(max(state.sigma_long, 0.0))
        )
        if not np.isfinite(blended_daily) or blended_daily <= 0.0:
            return float(self.config.default_annual_vol)

        annual = blended_daily * self.config.annualization_factor
        if not np.isfinite(annual) or annual <= 0.0:
            return float(self.config.default_annual_vol)
        return float(annual)

    def _update_long_run_if_needed(self, state: _TickerState, ts: pd.Timestamp) -> None:
        month_key = ts.strftime("%Y-%m")
        if state.last_long_run_month == month_key:
            return

        history = np.asarray(state.returns_history, dtype=np.float64)
        if history.size >= 2:
            sigma = float(np.std(history, ddof=1))
            if np.isfinite(sigma) and sigma > 0.0:
                state.sigma_long = sigma
        elif history.size == 1:
            state.sigma_long = float(abs(history[0]))

        state.last_long_run_month = month_key

    def _update_state_for_row(self, state: _TickerState, ts: pd.Timestamp, close: float) -> float:
        if state.last_close is None or not np.isfinite(state.last_close) or state.last_close <= 0.0:
            state.last_close = float(close)
            state.last_datetime = ts
            self._update_long_run_if_needed(state, ts)
            return float(self.config.default_annual_vol)

        ret = float(np.log(float(close) / float(state.last_close)))
        if np.isfinite(ret):
            state.returns_history.append(ret)
            r_sq = ret * ret
            if state.ewma_variance is None or not np.isfinite(state.ewma_variance):
                state.ewma_variance = r_sq
            else:
                state.ewma_variance = (
                    self.config.lambda_short * float(state.ewma_variance)
                    + (1.0 - self.config.lambda_short) * r_sq
                )

        self._update_long_run_if_needed(state, ts)

        state.last_close = float(close)
        state.last_datetime = ts
        return self._safe_annual_from_state(state)

    def _new_state(self) -> _TickerState:
        return _TickerState(
            returns_history=deque(maxlen=self.config.long_run_window)
        )

    def compute_daily_series(self, daily_candles_df: pd.DataFrame) -> pd.DataFrame:
        """Compute full causal daily EWSD volatility series from daily candles."""
        normalized = self._normalize_candles(daily_candles_df)
        if normalized.empty:
            return pd.DataFrame(columns=["datetime", "ticker", "ewsd_annual_vol"])

        rows: list[dict[str, Any]] = []
        for ticker, grp in normalized.groupby("ticker", sort=False):
            state = self._new_state()
            for row in grp.sort_values("datetime").itertuples(index=False):
                annual = self._update_state_for_row(
                    state=state,
                    ts=pd.Timestamp(row.datetime),
                    close=float(row.close),
                )
                rows.append(
                    {
                        "datetime": pd.Timestamp(row.datetime),
                        "ticker": str(ticker),
                        "ewsd_annual_vol": float(annual),
                    }
                )

        return (
            pd.DataFrame(rows)
            .sort_values(["ticker", "datetime"])
            .reset_index(drop=True)
        )

    def align_daily_volatility_to_candles(
        self,
        daily_volatility_df: pd.DataFrame,
        candles_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align daily EWSD volatility to arbitrary timeframe candles.

        Uses per-ticker date-based forward fill and fails fast on unresolved gaps.
        """
        required_vol = {"datetime", "ticker", "ewsd_annual_vol"}
        missing_vol = required_vol - set(daily_volatility_df.columns)
        if missing_vol:
            raise ValueError(
                "daily_volatility_df missing required columns: "
                f"{sorted(missing_vol)}"
            )

        required_candles = {"datetime", "ticker"}
        missing_candles = required_candles - set(candles_df.columns)
        if missing_candles:
            raise ValueError(
                "candles_df missing required columns: "
                f"{sorted(missing_candles)}"
            )

        if candles_df.empty:
            return pd.DataFrame(columns=["datetime", "ticker", "ewsd_annual_vol"])

        vol = daily_volatility_df[["datetime", "ticker", "ewsd_annual_vol"]].copy()
        vol["datetime"] = pd.to_datetime(vol["datetime"]).dt.tz_localize(None).dt.normalize()
        vol["ticker"] = vol["ticker"].map(normalize_ticker_key)
        vol["ewsd_annual_vol"] = pd.to_numeric(vol["ewsd_annual_vol"], errors="coerce")
        vol = vol.dropna(subset=["datetime", "ticker", "ewsd_annual_vol"])
        vol = (
            vol.sort_values(["ticker", "datetime"])
            .drop_duplicates(subset=["ticker", "datetime"], keep="last")
        )

        candles = candles_df[["datetime", "ticker"]].copy()
        candles["datetime"] = pd.to_datetime(candles["datetime"]).dt.tz_localize(None)
        candles["ticker"] = candles["ticker"].map(normalize_ticker_key)
        candles["date"] = candles["datetime"].dt.normalize()

        aligned_parts: list[pd.DataFrame] = []
        for ticker, grp in candles.groupby("ticker", sort=False):
            ticker_vol = vol[vol["ticker"] == ticker][["datetime", "ewsd_annual_vol"]].copy()
            if ticker_vol.empty:
                raise ValueError(
                    f"Missing daily EWSD volatility for ticker '{ticker}'."
                )

            ticker_vol = (
                ticker_vol.sort_values("datetime")
                .drop_duplicates(subset=["datetime"], keep="last")
                .set_index("datetime")
            )

            required_dates = pd.DatetimeIndex(grp["date"].unique()).sort_values()
            union_index = ticker_vol.index.union(required_dates).sort_values()
            aligned_values = (
                ticker_vol["ewsd_annual_vol"]
                .reindex(union_index)
                .ffill()
                .bfill()
                .reindex(required_dates)
            )

            aligned = grp.copy()
            aligned["ewsd_annual_vol"] = aligned["date"].map(aligned_values)

            if aligned["ewsd_annual_vol"].isna().any():
                missing_dates = (
                    aligned.loc[aligned["ewsd_annual_vol"].isna(), "date"]
                    .drop_duplicates()
                    .sort_values()
                )
                first_missing = missing_dates.iloc[0]
                raise ValueError(
                    "Missing aligned EWSD volatility after forward-fill for "
                    f"ticker '{ticker}' at date {first_missing.date()}."
                )

            aligned_parts.append(
                aligned[["datetime", "ticker", "ewsd_annual_vol"]]
            )

        return (
            pd.concat(aligned_parts, ignore_index=True)
            .sort_values(["ticker", "datetime"])
            .reset_index(drop=True)
        )

    def _load_state(self) -> dict[str, _TickerState]:
        if self._state_path is None or not self._state_path.exists():
            return {}

        raw = json.loads(self._state_path.read_text())
        out: dict[str, _TickerState] = {}
        for ticker, payload in raw.items():
            history = deque(payload.get("returns_history", []), maxlen=self.config.long_run_window)
            out[ticker] = _TickerState(
                last_datetime=pd.to_datetime(payload.get("last_datetime")) if payload.get("last_datetime") else None,
                last_close=payload.get("last_close"),
                ewma_variance=payload.get("ewma_variance"),
                sigma_long=float(payload.get("sigma_long", 0.01)),
                returns_history=history,
                last_long_run_month=payload.get("last_long_run_month"),
            )
        return out

    def _save_state(self, state_by_ticker: dict[str, _TickerState]) -> None:
        if self._state_path is None:
            return

        serializable: dict[str, dict[str, Any]] = {}
        for ticker, state in state_by_ticker.items():
            serializable[ticker] = {
                "last_datetime": state.last_datetime.isoformat() if state.last_datetime is not None else None,
                "last_close": state.last_close,
                "ewma_variance": state.ewma_variance,
                "sigma_long": state.sigma_long,
                "returns_history": list(state.returns_history),
                "last_long_run_month": state.last_long_run_month,
            }

        self._state_path.write_text(json.dumps(serializable, indent=2, sort_keys=True))

    def load_history(self) -> pd.DataFrame:
        if self._history_path is None or not self._history_path.exists():
            return pd.DataFrame(columns=["datetime", "ticker", "ewsd_annual_vol"])
        history = pd.read_parquet(self._history_path)
        history["datetime"] = pd.to_datetime(history["datetime"]).dt.tz_localize(None)
        history["ticker"] = history["ticker"].map(normalize_ticker_key)
        history = history.sort_values(["ticker", "datetime"]).reset_index(drop=True)
        return history

    def update_incremental(self, daily_candles_df: pd.DataFrame) -> pd.DataFrame:
        """Update persisted EWSD state/history with only new daily candles."""
        if self.store_dir is None:
            raise ValueError("update_incremental requires store_dir to be configured.")

        normalized = self._normalize_candles(daily_candles_df)
        if normalized.empty:
            return self.load_history()

        state_by_ticker = self._load_state()
        new_rows: list[dict[str, Any]] = []

        for ticker, grp in normalized.groupby("ticker", sort=False):
            state = state_by_ticker.get(ticker, self._new_state())
            grp_sorted = grp.sort_values("datetime")

            if state.last_datetime is not None:
                grp_sorted = grp_sorted[grp_sorted["datetime"] > state.last_datetime]

            for row in grp_sorted.itertuples(index=False):
                annual = self._update_state_for_row(
                    state=state,
                    ts=pd.Timestamp(row.datetime),
                    close=float(row.close),
                )
                new_rows.append(
                    {
                        "datetime": pd.Timestamp(row.datetime),
                        "ticker": str(ticker),
                        "ewsd_annual_vol": float(annual),
                    }
                )

            state_by_ticker[ticker] = state

        history = self.load_history()
        if new_rows:
            appended = pd.DataFrame(new_rows)
            if history.empty:
                history = appended
            else:
                history = pd.concat([history, appended], ignore_index=True)
            history = (
                history.sort_values(["ticker", "datetime"])
                .drop_duplicates(subset=["ticker", "datetime"], keep="last")
                .reset_index(drop=True)
            )

        if self._history_path is not None:
            history.to_parquet(self._history_path, index=False)
        self._save_state(state_by_ticker)
        return history

    def latest_volatility_map(self) -> Dict[str, float]:
        """Return latest annualized EWSD volatility per ticker from persisted history."""
        history = self.load_history()
        if history.empty:
            return {}

        latest = (
            history.sort_values(["ticker", "datetime"]) 
            .groupby("ticker", as_index=False)
            .last()
        )
        return {
            str(row.ticker): float(row.ewsd_annual_vol)
            for row in latest.itertuples(index=False)
        }


def compute_daily_ewsd_volatility(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience API for full-series daily EWSD computation."""
    return DailyEWSDVolatilityService().compute_daily_series(candles_df)


def align_daily_ewsd_volatility_to_candles(
    daily_volatility_df: pd.DataFrame,
    candles_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convenience API for daily EWSD alignment to arbitrary candle bars."""
    return DailyEWSDVolatilityService().align_daily_volatility_to_candles(
        daily_volatility_df=daily_volatility_df,
        candles_df=candles_df,
    )
