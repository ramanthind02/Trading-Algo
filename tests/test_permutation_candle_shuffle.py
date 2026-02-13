import numpy as np
import pandas as pd

from utils.permutation_test.candle_shuffle import (
    CandleShuffler,
    CandleShuffleMode,
    GapType,
    IntradayGapConfig,
    classify_intraday_gap,
)


REQUIRED_COLUMNS = ["open", "high", "low", "close", "datetime"]


def _in_pool(value: float, pool: set[float], tol: float = 1e-9) -> bool:
    return any(np.isclose(value, candidate, atol=tol, rtol=0.0) for candidate in pool)


def _make_daily_df(n_bars: int = 15) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-06", periods=n_bars)

    opens: list[float] = [100.0]
    highs: list[float] = [101.0]
    lows: list[float] = [99.0]
    closes: list[float] = [100.5]

    for i in range(1, n_bars):
        prev = dates[i - 1]
        curr = dates[i]
        is_weekend_gap = prev.weekday() == 4 and curr.weekday() == 0
        gap = 1000.0 + i if is_weekend_gap else 10.0 + i

        curr_open = closes[-1] + gap
        curr_close = curr_open + ((i % 5) - 2) * 0.2
        curr_high = max(curr_open, curr_close) + 1.0
        curr_low = min(curr_open, curr_close) - 1.0

        opens.append(curr_open)
        highs.append(curr_high)
        lows.append(curr_low)
        closes.append(curr_close)

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.arange(n_bars) + 1,
        }
    )


def _daily_types(df: pd.DataFrame) -> list[str]:
    dt = pd.to_datetime(df["datetime"])
    gap_types: list[str] = ["basis"]
    for i in range(1, len(df)):
        prev = dt.iloc[i - 1]
        curr = dt.iloc[i]
        gap_types.append("weekend" if prev.weekday() == 4 and curr.weekday() == 0 else "weekday")
    return gap_types


def _make_intraday_df() -> pd.DataFrame:
    start_days = pd.to_datetime(
        [
            "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09",
            "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16",
        ]
    )

    timestamps: list[pd.Timestamp] = []
    for day in start_days:
        timestamps.extend(
            [
                day.replace(hour=18, minute=0),
                day.replace(hour=19, minute=0),
                (day + pd.Timedelta(days=1)).replace(hour=17, minute=15),
            ]
        )

    config = IntradayGapConfig()

    opens: list[float] = [100.0]
    closes: list[float] = [100.3]
    highs: list[float] = [100.8]
    lows: list[float] = [99.8]

    for i in range(1, len(timestamps)):
        prev_ts = timestamps[i - 1]
        curr_ts = timestamps[i]
        gap_type = classify_intraday_gap(prev_ts, curr_ts, config)

        if gap_type is GapType.WEEKEND:
            gap = 2000.0 + i
        elif gap_type is GapType.MAINTENANCE:
            gap = 200.0 + i
        else:
            gap = 2.0 + i / 100.0

        curr_open = closes[-1] + gap
        curr_close = curr_open + (0.25 if i % 2 == 0 else -0.15)
        curr_high = max(curr_open, curr_close) + 0.4
        curr_low = min(curr_open, curr_close) - 0.4

        opens.append(curr_open)
        closes.append(curr_close)
        highs.append(curr_high)
        lows.append(curr_low)

    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.arange(len(timestamps)) + 1,
        }
    )


def _intraday_types(df: pd.DataFrame, config: IntradayGapConfig | None = None) -> list[GapType]:
    cfg = config or IntradayGapConfig()
    dt = pd.to_datetime(df["datetime"])
    out: list[GapType] = [GapType.REGULAR]
    for i in range(1, len(df)):
        out.append(classify_intraday_gap(dt.iloc[i - 1], dt.iloc[i], cfg))
    return out


def _intraday_day_starts(df: pd.DataFrame, config: IntradayGapConfig | None = None) -> list[int]:
    gap_types = _intraday_types(df, config)
    starts = [0]
    starts.extend(
        i for i in range(1, len(df))
        if gap_types[i] in {GapType.MAINTENANCE, GapType.WEEKEND}
    )
    return starts


def test_daily_reproducibility_with_fixed_seed() -> None:
    df = _make_daily_df()

    shuffled_1 = CandleShuffler(
        df,
        permute_start_idx=0,
        mode=CandleShuffleMode.DAILY,
        random_seed=7,
    ).permute()
    shuffled_2 = CandleShuffler(
        df,
        permute_start_idx=0,
        mode=CandleShuffleMode.DAILY,
        random_seed=7,
    ).permute()

    pd.testing.assert_frame_equal(shuffled_1, shuffled_2)


def test_daily_prefix_is_unchanged_when_permute_start_idx_is_set() -> None:
    df = _make_daily_df()
    start_idx = 6

    shuffled = CandleShuffler(
        df,
        permute_start_idx=start_idx,
        mode=CandleShuffleMode.DAILY,
        random_seed=11,
    ).permute()

    pd.testing.assert_frame_equal(
        shuffled.loc[: start_idx - 1, REQUIRED_COLUMNS],
        df.loc[: start_idx - 1, REQUIRED_COLUMNS],
        check_dtype=False,
    )


def test_daily_preserves_first_open_last_close_and_ohlc_validity() -> None:
    df = _make_daily_df()

    shuffled = CandleShuffler(
        df,
        permute_start_idx=0,
        mode=CandleShuffleMode.DAILY,
        random_seed=13,
    ).permute()

    assert shuffled["open"].iloc[0] == df["open"].iloc[0]
    assert np.isclose(shuffled["close"].iloc[-1], df["close"].iloc[-1])

    assert (shuffled["high"] >= shuffled[["open", "close"]].max(axis=1)).all()
    assert (shuffled["low"] <= shuffled[["open", "close"]].min(axis=1)).all()


def test_daily_weekend_and_weekday_gap_pools_do_not_mix() -> None:
    df = _make_daily_df()
    gap_types = _daily_types(df)

    original_gaps = df["open"].values[1:] - df["close"].values[:-1]
    weekend_gaps = {original_gaps[i - 1] for i in range(1, len(df)) if gap_types[i] == "weekend"}
    weekday_gaps = {original_gaps[i - 1] for i in range(1, len(df)) if gap_types[i] == "weekday"}

    shuffled = CandleShuffler(
        df,
        permute_start_idx=0,
        mode=CandleShuffleMode.DAILY,
        random_seed=21,
    ).permute()

    shuffled_gaps = shuffled["open"].values[1:] - shuffled["close"].values[:-1]

    for i in range(1, len(df)):
        gap = shuffled_gaps[i - 1]
        if gap_types[i] == "weekend":
            assert _in_pool(gap, weekend_gaps)
        else:
            assert _in_pool(gap, weekday_gaps)
            assert not _in_pool(gap, weekend_gaps)


def test_daily_handles_empty_and_single_bar_inputs() -> None:
    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    single_df = _make_daily_df(1)[REQUIRED_COLUMNS].copy()

    shuffled_empty = CandleShuffler(empty_df, mode=CandleShuffleMode.DAILY).permute()
    shuffled_single = CandleShuffler(single_df, mode=CandleShuffleMode.DAILY).permute()

    assert shuffled_empty.empty
    assert list(shuffled_empty.columns) == REQUIRED_COLUMNS
    pd.testing.assert_frame_equal(shuffled_single, single_df, check_dtype=False)


def test_intraday_rounds_up_permute_start_idx_to_next_day_boundary() -> None:
    df = _make_intraday_df()
    start_idx = 1  # middle of the first day block

    day_starts = _intraday_day_starts(df)
    expected_start = next(idx for idx in day_starts if idx >= start_idx)

    shuffled = CandleShuffler(
        df,
        permute_start_idx=start_idx,
        mode=CandleShuffleMode.INTRADAY,
        random_seed=31,
    ).permute()

    pd.testing.assert_frame_equal(
        shuffled.loc[: expected_start - 1, REQUIRED_COLUMNS],
        df.loc[: expected_start - 1, REQUIRED_COLUMNS],
        check_dtype=False,
    )


def test_intraday_keeps_basis_day_unchanged() -> None:
    df = _make_intraday_df()
    day_starts = _intraday_day_starts(df)
    start_idx = day_starts[1]
    basis_end = day_starts[1] - 1

    shuffled = CandleShuffler(
        df,
        permute_start_idx=start_idx,
        mode=CandleShuffleMode.INTRADAY,
        random_seed=37,
    ).permute()

    pd.testing.assert_frame_equal(
        shuffled.loc[:basis_end, REQUIRED_COLUMNS],
        df.loc[:basis_end, REQUIRED_COLUMNS],
        check_dtype=False,
    )


def test_intraday_preserves_multiset_of_per_day_net_moves() -> None:
    df = _make_intraday_df()
    day_starts = _intraday_day_starts(df)
    day_ends = day_starts[1:] + [len(df)]
    start_idx = day_starts[1]

    shuffled = CandleShuffler(
        df,
        permute_start_idx=start_idx,
        mode=CandleShuffleMode.INTRADAY,
        random_seed=41,
    ).permute()

    original_moves = [
        df["close"].iloc[e - 1] - df["open"].iloc[s]
        for s, e in zip(day_starts[1:], day_ends[1:])
    ]
    shuffled_moves = [
        shuffled["close"].iloc[e - 1] - shuffled["open"].iloc[s]
        for s, e in zip(day_starts[1:], day_ends[1:])
    ]

    assert np.allclose(np.sort(original_moves), np.sort(shuffled_moves))


def test_intraday_gap_type_pools_do_not_mix() -> None:
    df = _make_intraday_df()
    gap_types = _intraday_types(df)

    original_gaps = df["open"].values[1:] - df["close"].values[:-1]
    original_pools: dict[GapType, set[float]] = {
        GapType.WEEKEND: set(),
        GapType.MAINTENANCE: set(),
        GapType.REGULAR: set(),
    }
    for i in range(1, len(df)):
        original_pools[gap_types[i]].add(original_gaps[i - 1])

    shuffled = CandleShuffler(
        df,
        permute_start_idx=3,
        mode=CandleShuffleMode.INTRADAY,
        random_seed=43,
    ).permute()

    shuffled_gaps = shuffled["open"].values[1:] - shuffled["close"].values[:-1]

    for i in range(1, len(df)):
        gap = shuffled_gaps[i - 1]
        expected_pool = original_pools[gap_types[i]]
        assert _in_pool(gap, expected_pool)


def test_intraday_reproducibility_with_fixed_seed() -> None:
    df = _make_intraday_df()

    shuffled_1 = CandleShuffler(
        df,
        permute_start_idx=3,
        mode=CandleShuffleMode.INTRADAY,
        random_seed=47,
    ).permute()
    shuffled_2 = CandleShuffler(
        df,
        permute_start_idx=3,
        mode=CandleShuffleMode.INTRADAY,
        random_seed=47,
    ).permute()

    pd.testing.assert_frame_equal(shuffled_1, shuffled_2)
