"""Unit tests for rolling/CUSUM stability visualization helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from quantfoundry_core.robustness import rolling_is_performance

from research.feature.visualization.cusum_stability import (
    build_rolling_cusum_frame,
    plot_rolling_cusum_stability_csv,
    rolling_sharpe_end_indices,
    stability_rolling_window_label,
    write_rolling_cusum_artifacts,
)


def test_stability_rolling_window_label_defaults_to_six_months() -> None:
    assert stability_rolling_window_label(126) == "6-month rolling"


def test_rolling_sharpe_end_indices_aligns_windows() -> None:
    assert rolling_sharpe_end_indices(window=4, n_obs=10).tolist() == [3, 4, 5, 6, 7, 8, 9]


def test_build_rolling_cusum_frame_uses_full_series_with_rolling_window() -> None:
    returns = np.random.default_rng(0).normal(0.0, 0.01, size=40)
    rolling = rolling_is_performance(returns, window=8, periods_per_year=252)
    datetimes = [f"2020-01-{index + 1:02d}" for index in range(len(returns))]
    frame = build_rolling_cusum_frame(rolling, datetimes)
    assert frame["bar_index"].min() == 0
    assert frame["bar_index"].max() == 39
    assert int(frame["rolling_window_days"].iloc[0]) == 8
    assert frame["rolling_window_label"].iloc[0] == stability_rolling_window_label(8)
    assert frame["cusum"].notna().all()
    assert frame["rolling_sharpe"].notna().any()


def test_write_rolling_cusum_artifacts_writes_csv_and_png(tmp_path: Path) -> None:
    returns = np.random.default_rng(1).normal(0.0, 0.01, size=30)
    rolling = rolling_is_performance(returns, window=6, periods_per_year=252)
    artifacts = write_rolling_cusum_artifacts(
        rolling,
        tmp_path,
        [f"2021-02-{index + 1:02d}" for index in range(len(returns))],
    )
    assert artifacts["rolling_cusum_csv"].exists()
    assert artifacts["cusum_plot_png"].exists()
    replotted = plot_rolling_cusum_stability_csv(
        artifacts["rolling_cusum_csv"],
        tmp_path / "matplotlib" / "cusum_stability_regen.png",
    )
    assert replotted is not None
    assert replotted.exists()
