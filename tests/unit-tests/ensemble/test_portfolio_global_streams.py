from __future__ import annotations

import pandas as pd
import pytest

from ensemble.portfolio_global_streams import (
    align_forecast_vectors_to_daily_grid,
    build_daily_grid,
    normalize_global_signals_by_downside_vol,
)


def test_build_daily_grid_prefers_reference_index() -> None:
    forecast_vectors = [
        pd.DataFrame(
            {
                "ticker": ["ES"],
                "datetime": pd.to_datetime(["2024-01-02"]),
                "model_name": ["alpha"],
                "forecast": [0.5],
                "signal": [0.5],
                "timeframe": ["D"],
            }
        )
    ]

    grid = build_daily_grid(
        forecast_vectors,
        reference_index=pd.to_datetime(["2024-01-03 14:00", "2024-01-01 09:00"]),
    )

    assert list(grid) == list(pd.to_datetime(["2024-01-01", "2024-01-03"]))


def test_align_forecast_vectors_to_daily_grid_forward_fills_streams() -> None:
    forecast_vectors = [
        pd.DataFrame(
            {
                "ticker": ["ES", "ES"],
                "datetime": pd.to_datetime(["2024-01-01", "2024-01-03"]),
                "model_name": ["alpha", "alpha"],
                "forecast": [1.0, 2.0],
                "signal": [1.0, 2.0],
                "timeframe": ["D", "D"],
            }
        )
    ]

    aligned = align_forecast_vectors_to_daily_grid(
        forecast_vectors,
        pd.date_range("2024-01-01", "2024-01-03", freq="D"),
    )[0]

    assert aligned["datetime"].tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    )
    assert aligned["forecast"].tolist() == pytest.approx([1.0, 1.0, 2.0])
    assert aligned["signal"].tolist() == pytest.approx([1.0, 1.0, 2.0])


def test_normalize_global_signals_by_downside_vol_scales_per_stream() -> None:
    forecast_vectors = [
        pd.DataFrame(
            {
                "ticker": ["ES", "ES", "ES"],
                "datetime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "model_name": ["alpha", "alpha", "alpha"],
                "forecast": [1.0, 1.0, 1.0],
                "signal": [1.0, -2.0, 3.0],
                "timeframe": ["D", "D", "D"],
            }
        )
    ]
    global_returns = pd.Series(
        [0.1, 0.2, -0.3],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    normalized = normalize_global_signals_by_downside_vol(forecast_vectors, global_returns)[0]

    expected_scale = 0.5686240703077328
    assert normalized["signal"].tolist() == pytest.approx(
        [1.0 / expected_scale, -2.0 / expected_scale, 3.0 / expected_scale]
    )
