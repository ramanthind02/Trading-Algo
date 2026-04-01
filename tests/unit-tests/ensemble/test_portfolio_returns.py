from __future__ import annotations

import pandas as pd
import pytest

from ensemble.portfolio_returns import calculate_idm_from_returns, calculate_returns_from_candles
from utils.core.enums import TimeFrame, Ticker


def test_calculate_returns_from_candles_normalizes_tickers_and_computes_returns() -> None:
    candles_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "ticker": [Ticker.ES, Ticker.ES, Ticker.ES, "NQ", "NQ", "NQ"],
            "close": [100.0, 110.0, 121.0, 200.0, 220.0, 242.0],
            "open": [0.0] * 6,
            "high": [0.0] * 6,
            "low": [0.0] * 6,
            "volume": [1.0] * 6,
            "timeframe": [TimeFrame.D] * 6,
        }
    )

    result = calculate_returns_from_candles(candles_df)

    assert list(result.columns) == ["ES", "NQ"]
    assert result["ES"].tolist() == pytest.approx([0.1, 0.1])
    assert result["NQ"].tolist() == pytest.approx([0.1, 0.1])


def test_calculate_returns_from_candles_drops_non_overlapping_rows_for_multi_ticker_idm() -> None:
    candles_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "ticker": ["ES", "ES", "ES", "NQ", "NQ", "NQ"],
            "close": [100.0, 110.0, 121.0, 200.0, 220.0, 242.0],
            "open": [0.0] * 6,
            "high": [0.0] * 6,
            "low": [0.0] * 6,
            "volume": [1.0] * 6,
            "timeframe": [TimeFrame.D] * 6,
        }
    )

    result = calculate_returns_from_candles(candles_df)

    assert result.empty
    assert list(result.columns) == ["ES", "NQ"]


def test_calculate_returns_from_candles_keeps_single_ticker_returns() -> None:
    candles_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["ES", "ES", "ES"],
            "close": [100.0, 110.0, 99.0],
            "open": [0.0] * 3,
            "high": [0.0] * 3,
            "low": [0.0] * 3,
            "volume": [1.0] * 3,
            "timeframe": [TimeFrame.D] * 3,
        }
    )

    result = calculate_returns_from_candles(candles_df)

    assert list(result.columns) == ["ES"]
    assert result["ES"].tolist() == pytest.approx([0.1, -0.1])


def test_calculate_idm_from_returns_defaults_to_one_for_single_instrument() -> None:
    instrument_returns = pd.DataFrame({"ES": [0.1, -0.2, 0.3]})

    mean_corr, idm = calculate_idm_from_returns(instrument_returns, idm_max=2.5)

    assert mean_corr == pytest.approx(1.0)
    assert idm == pytest.approx(1.0)


def test_calculate_idm_from_returns_caps_multiplier() -> None:
    instrument_returns = pd.DataFrame(
        {
            "ES": [0.01, -0.01, 0.01, -0.01],
            "NQ": [-0.01, 0.01, -0.01, 0.01],
        }
    )

    mean_corr, idm = calculate_idm_from_returns(instrument_returns, idm_max=1.2)

    assert mean_corr == pytest.approx(0.0)
    assert idm == pytest.approx(1.2)
