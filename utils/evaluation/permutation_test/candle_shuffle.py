from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


class CandleShuffleMode(Enum):
    """Mode selector for candle shuffling."""

    AUTO = "auto"
    DAILY = "daily"
    INTRADAY = "intraday"


class GapType(Enum):
    """Gap categories used by intraday shuffling."""

    WEEKEND = "weekend"
    MAINTENANCE = "maintenance"
    REGULAR = "regular"


GapClassifier = Callable[[pd.Timestamp, pd.Timestamp], GapType]


@dataclass(frozen=True)
class IntradayGapConfig:
    """
    Configuration for classifying intraday transition gaps.

    The default rules target ES-style 24/5 data:
    - Weekend: Friday -> Monday
    - Maintenance: around 17:xx -> 18:xx with a 30-180 minute break
    - Regular: everything else
    """

    maintenance_prev_hour: int = 17
    maintenance_next_hour: int = 18
    maintenance_min_gap_minutes: int = 30
    maintenance_max_gap_minutes: int = 180
    custom_classifier: Optional[GapClassifier] = None


def _normalize_mode(mode: str | CandleShuffleMode) -> CandleShuffleMode:
    if isinstance(mode, CandleShuffleMode):
        return mode

    normalized = str(mode).strip().lower()
    for item in CandleShuffleMode:
        if item.value == normalized:
            return item

    raise ValueError(f"Unknown candle shuffle mode: {mode}")


def _normalize_gap_type(gap_type: GapType | str) -> GapType:
    if isinstance(gap_type, GapType):
        return gap_type

    normalized = str(gap_type).strip().lower()
    for item in GapType:
        if item.value == normalized:
            return item

    raise ValueError(f"Unknown gap type: {gap_type}")


def classify_daily_gap(prev_ts: pd.Timestamp, curr_ts: pd.Timestamp) -> GapType:
    """Classify daily transition as weekend or regular (weekday)."""
    if prev_ts.weekday() == 4 and curr_ts.weekday() == 0:
        return GapType.WEEKEND
    return GapType.REGULAR


def classify_intraday_gap(
    prev_ts: pd.Timestamp,
    curr_ts: pd.Timestamp,
    config: Optional[IntradayGapConfig] = None,
) -> GapType:
    """Classify intraday transition using configurable rules."""
    cfg = config or IntradayGapConfig()

    if cfg.custom_classifier is not None:
        return _normalize_gap_type(cfg.custom_classifier(prev_ts, curr_ts))

    delta_minutes = (curr_ts - prev_ts).total_seconds() / 60.0

    if (
        prev_ts.weekday() == 4
        and curr_ts.weekday() == 0
        and (curr_ts - prev_ts) >= pd.Timedelta(days=2)
    ):
        return GapType.WEEKEND

    is_maintenance_hour = (
        prev_ts.hour == cfg.maintenance_prev_hour
        and curr_ts.hour == cfg.maintenance_next_hour
    )
    is_maintenance_gap = (
        cfg.maintenance_min_gap_minutes <= delta_minutes <= cfg.maintenance_max_gap_minutes
    )
    if is_maintenance_hour and is_maintenance_gap:
        return GapType.MAINTENANCE

    return GapType.REGULAR


class CandleShuffler:
    """
    Candle-level shuffler with daily and intraday variants.

    The implementation preserves the sequential datetime timeline and only
    permutes relative quantities (triplets and gaps), then reconstructs OHLC.
    """

    REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "datetime")

    def __init__(
        self,
        df: pd.DataFrame,
        permute_start_idx: int = 0,
        mode: str | CandleShuffleMode = CandleShuffleMode.AUTO,
        intraday_gap_config: Optional[IntradayGapConfig] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

        self._df = df.copy()
        self._df["datetime"] = pd.to_datetime(self._df["datetime"])
        self._df = self._df.sort_values("datetime", kind="stable").reset_index(drop=True)

        n_rows = len(self._df)
        start_idx = max(0, int(permute_start_idx))
        self._permute_start_idx = min(start_idx, n_rows)
        self._mode = _normalize_mode(mode)
        self._intraday_gap_config = intraday_gap_config or IntradayGapConfig()
        self._rng = np.random.default_rng(random_seed) if random_seed is not None else None

    def permute(self) -> pd.DataFrame:
        if len(self._df) <= 1:
            return self._df.copy()

        mode = self._resolve_mode()
        if mode is CandleShuffleMode.DAILY:
            opens, highs, lows, closes = self._permute_daily()
        else:
            opens, highs, lows, closes = self._permute_intraday()

        out = self._df.copy()
        out["open"] = opens
        out["high"] = highs
        out["low"] = lows
        out["close"] = closes
        return out

    def _resolve_mode(self) -> CandleShuffleMode:
        if self._mode is not CandleShuffleMode.AUTO:
            return self._mode

        dt = self._df["datetime"]
        if len(dt) <= 1:
            return CandleShuffleMode.DAILY

        deltas = dt.diff().dropna().dt.total_seconds().to_numpy()
        if deltas.size == 0:
            return CandleShuffleMode.DAILY

        return CandleShuffleMode.DAILY if np.median(deltas) >= 86400.0 else CandleShuffleMode.INTRADAY

    def _shuffle_indices(self, size: int) -> np.ndarray:
        if size <= 1:
            return np.arange(size, dtype=int)

        if self._rng is None:
            return np.random.permutation(size)

        return self._rng.permutation(size)

    def _shuffle_values(self, values: np.ndarray) -> np.ndarray:
        if values.size <= 1:
            return values.copy()

        idx = self._shuffle_indices(values.size)
        return values[idx]

    def _permute_daily(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        opens = self._df["open"].to_numpy(dtype=float).copy()
        highs = self._df["high"].to_numpy(dtype=float).copy()
        lows = self._df["low"].to_numpy(dtype=float).copy()
        closes = self._df["close"].to_numpy(dtype=float).copy()
        datetimes = self._df["datetime"]

        n = opens.size
        start = max(1, self._permute_start_idx)
        if start >= n:
            return opens, highs, lows, closes

        triplets = np.column_stack(
            (
                highs[start:] - opens[start:],
                lows[start:] - opens[start:],
                closes[start:] - opens[start:],
            )
        )
        gaps = opens[start:] - closes[start - 1 : n - 1]

        transition_types = [
            classify_daily_gap(datetimes.iloc[i - 1], datetimes.iloc[i]) for i in range(start, n)
        ]

        weekend_mask = np.array([g is GapType.WEEKEND for g in transition_types], dtype=bool)
        weekday_mask = ~weekend_mask

        shuffled_triplets = triplets[self._shuffle_indices(triplets.shape[0])]
        shuffled_weekend = self._shuffle_values(gaps[weekend_mask])
        shuffled_weekday = self._shuffle_values(gaps[weekday_mask])

        weekend_pos = 0
        weekday_pos = 0
        for offset, i in enumerate(range(start, n)):
            if transition_types[offset] is GapType.WEEKEND:
                gap_value = shuffled_weekend[weekend_pos]
                weekend_pos += 1
            else:
                gap_value = shuffled_weekday[weekday_pos]
                weekday_pos += 1

            opens[i] = closes[i - 1] + gap_value
            highs[i] = opens[i] + shuffled_triplets[offset, 0]
            lows[i] = opens[i] + shuffled_triplets[offset, 1]
            closes[i] = opens[i] + shuffled_triplets[offset, 2]

        return opens, highs, lows, closes

    def _intraday_transition_types(self) -> list[GapType]:
        datetimes = self._df["datetime"]
        gap_types: list[GapType] = [GapType.REGULAR]
        gap_types.extend(
            classify_intraday_gap(datetimes.iloc[i - 1], datetimes.iloc[i], self._intraday_gap_config)
            for i in range(1, len(datetimes))
        )
        return gap_types

    @staticmethod
    def _day_boundaries(transition_types: Sequence[GapType], n: int) -> tuple[list[int], list[int]]:
        starts = [0]
        starts.extend(
            i
            for i in range(1, n)
            if transition_types[i] in {GapType.MAINTENANCE, GapType.WEEKEND}
        )
        ends = starts[1:] + [n]
        return starts, ends

    @staticmethod
    def _round_up_to_day_start(day_starts: Sequence[int], start_idx: int, n: int) -> int:
        if start_idx <= 0:
            return 0

        next_starts = [idx for idx in day_starts if idx >= start_idx]
        return next_starts[0] if next_starts else n

    def _permute_intraday(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        opens = self._df["open"].to_numpy(dtype=float).copy()
        highs = self._df["high"].to_numpy(dtype=float).copy()
        lows = self._df["low"].to_numpy(dtype=float).copy()
        closes = self._df["close"].to_numpy(dtype=float).copy()
        n = opens.size

        transition_types = self._intraday_transition_types()
        day_starts, day_ends = self._day_boundaries(transition_types, n)

        effective_start = self._round_up_to_day_start(day_starts, self._permute_start_idx, n)
        if effective_start >= n:
            return opens, highs, lows, closes

        day_pos_by_start = {start: pos for pos, start in enumerate(day_starts)}
        if effective_start == 0:
            first_perm_day_pos = 1
        else:
            first_perm_day_pos = day_pos_by_start[effective_start]

        if first_perm_day_pos >= len(day_starts):
            return opens, highs, lows, closes

        basis_day_pos = first_perm_day_pos - 1
        basis_end_idx = day_ends[basis_day_pos]
        if basis_end_idx >= n:
            return opens, highs, lows, closes

        perm_day_positions = list(range(first_perm_day_pos, len(day_starts)))
        if not perm_day_positions:
            return opens, highs, lows, closes

        day_payloads = [
            self._permute_single_day(
                opens[day_starts[pos] : day_ends[pos]],
                highs[day_starts[pos] : day_ends[pos]],
                lows[day_starts[pos] : day_ends[pos]],
                closes[day_starts[pos] : day_ends[pos]],
            )
            for pos in perm_day_positions
        ]
        shuffled_day_payloads = [day_payloads[idx] for idx in self._shuffle_indices(len(day_payloads))]

        transition_indices = [day_starts[pos] for pos in perm_day_positions]
        schedule_types = [transition_types[idx] for idx in transition_indices]
        transition_gaps = np.array([opens[idx] - closes[idx - 1] for idx in transition_indices], dtype=float)

        pools = {
            GapType.WEEKEND: self._shuffle_values(
                transition_gaps[np.array([t is GapType.WEEKEND for t in schedule_types], dtype=bool)]
            ),
            GapType.MAINTENANCE: self._shuffle_values(
                transition_gaps[np.array([t is GapType.MAINTENANCE for t in schedule_types], dtype=bool)]
            ),
            GapType.REGULAR: self._shuffle_values(
                transition_gaps[np.array([t is GapType.REGULAR for t in schedule_types], dtype=bool)]
            ),
        }
        pool_pos = {GapType.WEEKEND: 0, GapType.MAINTENANCE: 0, GapType.REGULAR: 0}

        write_idx = day_starts[first_perm_day_pos]
        prev_close = closes[write_idx - 1]

        for slot_idx, payload in enumerate(shuffled_day_payloads):
            gap_type = schedule_types[slot_idx]
            gap_value = pools[gap_type][pool_pos[gap_type]]
            pool_pos[gap_type] += 1

            day_open = prev_close + gap_value
            day_len = payload["open"].size
            end_idx = write_idx + day_len

            opens[write_idx:end_idx] = day_open + payload["open"]
            highs[write_idx:end_idx] = day_open + payload["high"]
            lows[write_idx:end_idx] = day_open + payload["low"]
            closes[write_idx:end_idx] = day_open + payload["close"]

            prev_close = closes[end_idx - 1]
            write_idx = end_idx

        return opens, highs, lows, closes

    def _permute_single_day(
        self,
        day_opens: np.ndarray,
        day_highs: np.ndarray,
        day_lows: np.ndarray,
        day_closes: np.ndarray,
    ) -> dict[str, np.ndarray]:
        n_day = day_opens.size
        if n_day == 0:
            return {
                "open": np.array([], dtype=float),
                "high": np.array([], dtype=float),
                "low": np.array([], dtype=float),
                "close": np.array([], dtype=float),
            }

        rel_high = day_highs - day_opens
        rel_low = day_lows - day_opens
        rel_close = day_closes - day_opens
        rel_gaps = np.zeros(n_day, dtype=float)
        if n_day > 1:
            rel_gaps[1:] = day_opens[1:] - day_closes[:-1]

        shuffled_triplets = np.column_stack((rel_high, rel_low, rel_close))[
            self._shuffle_indices(n_day)
        ]
        shuffled_gaps = self._shuffle_values(rel_gaps[1:]) if n_day > 1 else np.array([], dtype=float)

        anchored_open = np.zeros(n_day, dtype=float)
        anchored_high = np.zeros(n_day, dtype=float)
        anchored_low = np.zeros(n_day, dtype=float)
        anchored_close = np.zeros(n_day, dtype=float)

        for i in range(n_day):
            if i > 0:
                anchored_open[i] = anchored_close[i - 1] + shuffled_gaps[i - 1]

            anchored_high[i] = anchored_open[i] + shuffled_triplets[i, 0]
            anchored_low[i] = anchored_open[i] + shuffled_triplets[i, 1]
            anchored_close[i] = anchored_open[i] + shuffled_triplets[i, 2]

        return {
            "open": anchored_open,
            "high": anchored_high,
            "low": anchored_low,
            "close": anchored_close,
        }
