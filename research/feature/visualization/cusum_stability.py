"""Rolling IS Sharpe and CUSUM stability plots for exploration robustness."""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quantfoundry_core.robustness import RollingISResult

DEFAULT_STABILITY_ROLLING_WINDOW_DAYS = 126
TRADING_DAYS_PER_YEAR = 252


def stability_rolling_window_label(
    rolling_window_days: int,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> str:
    """Human-readable label for the configured rolling window."""

    if rolling_window_days == DEFAULT_STABILITY_ROLLING_WINDOW_DAYS and periods_per_year == TRADING_DAYS_PER_YEAR:
        return "6-month rolling"
    months = rolling_window_days * 12.0 / float(periods_per_year)
    if abs(months - round(months)) < 0.05:
        return f"{int(round(months))}-month rolling"
    return f"{rolling_window_days}-bar rolling"


def rolling_sharpe_end_indices(*, window: int, n_obs: int) -> np.ndarray:
    """Bar indices (0-based) where each rolling Sharpe window ends."""

    if n_obs < window:
        return np.empty(0, dtype=np.int64)
    return np.arange(window - 1, n_obs, dtype=np.int64)


def build_rolling_cusum_frame(
    rolling_is: RollingISResult,
    observation_datetimes: Sequence[str] | None,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Build a wide CSV frame for rolling Sharpe and CUSUM series."""

    cusum_values = np.asarray(rolling_is.cusum_series, dtype=np.float64)
    rolling_values = np.asarray(rolling_is.rolling_sharpe, dtype=np.float64)
    n_obs = int(cusum_values.shape[0])
    end_indices = rolling_sharpe_end_indices(window=rolling_is.window, n_obs=n_obs)

    datetimes = (
        list(observation_datetimes)
        if observation_datetimes is not None
        else [str(index) for index in range(n_obs)]
    )
    if len(datetimes) != n_obs:
        datetimes = [str(index) for index in range(n_obs)]

    cusum_frame = pd.DataFrame(
        {
            "bar_index": np.arange(n_obs, dtype=np.int64),
            "datetime": datetimes,
            "cusum": cusum_values,
        }
    )
    rolling_frame = pd.DataFrame(
        {
            "bar_index": end_indices,
            "datetime": [datetimes[int(index)] for index in end_indices],
            "rolling_sharpe": rolling_values,
        }
    )
    merged = cusum_frame.merge(rolling_frame, on=["bar_index", "datetime"], how="outer")
    merged["rolling_window_days"] = rolling_is.window
    merged["rolling_window_label"] = stability_rolling_window_label(
        rolling_is.window,
        periods_per_year=periods_per_year,
    )
    merged["cusum_statistic"] = rolling_is.cusum_statistic
    merged["cusum_critical_value"] = rolling_is.cusum_critical_value
    merged["cusum_break_detected"] = rolling_is.cusum_break_detected
    merged["cusum_break_index"] = rolling_is.cusum_break_index
    return merged.sort_values("bar_index", kind="mergesort").reset_index(drop=True)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rolling_cusum_stability_csv(
    csv_path: Path,
    output_path: Path,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Path | None:
    """Render 6-month rolling Sharpe + CUSUM chart from ``robustness_rolling_cusum.csv``."""

    if not csv_path.exists():
        return None
    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None
    required = {
        "bar_index",
        "datetime",
        "cusum",
        "rolling_sharpe",
        "cusum_statistic",
        "cusum_critical_value",
        "cusum_break_detected",
        "cusum_break_index",
        "rolling_window_days",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None

    work = frame.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work["cusum"] = pd.to_numeric(work["cusum"], errors="coerce")
    work["rolling_sharpe"] = pd.to_numeric(work["rolling_sharpe"], errors="coerce")
    work = work.dropna(subset=["datetime"])
    if work.empty:
        return None

    rolling_window_days = int(pd.to_numeric(work["rolling_window_days"].iloc[0], errors="coerce"))
    window_label = (
        str(work["rolling_window_label"].iloc[0])
        if "rolling_window_label" in work.columns
        else stability_rolling_window_label(rolling_window_days, periods_per_year=periods_per_year)
    )
    cusum_statistic = float(work["cusum_statistic"].iloc[0])
    cusum_critical = float(work["cusum_critical_value"].iloc[0])
    break_detected = bool(work["cusum_break_detected"].iloc[0])
    break_index = int(work["cusum_break_index"].iloc[0])
    n_obs = int(pd.to_numeric(work["bar_index"], errors="coerce").max()) + 1
    envelope = cusum_critical * math.sqrt(max(n_obs, 1))

    rolling = work.dropna(subset=["rolling_sharpe"]).sort_values("datetime")
    cusum = work.dropna(subset=["cusum"]).sort_values("datetime")

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)
    fig.patch.set_facecolor("#ffffff")
    for axis in axes:
        axis.set_facecolor("#fbfbfc")
        axis.grid(alpha=0.2, linestyle="--")

    title = f"{window_label.title()} IS Sharpe and CUSUM Stability ({rolling_window_days} bars)"
    axes[0].set_title(title, fontsize=12, fontweight="bold")
    if not rolling.empty:
        axes[0].plot(
            rolling["datetime"],
            rolling["rolling_sharpe"],
            color="#2563eb",
            linewidth=1.8,
            label=f"{window_label.title()} Sharpe",
        )
        axes[0].axhline(0.0, color="#64748b", linewidth=1.0, linestyle=":")
    axes[0].set_ylabel("Annualized Sharpe")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(
        cusum["datetime"],
        cusum["cusum"],
        color="#ea580c",
        linewidth=1.8,
        label="CUSUM S(t)",
    )
    axes[1].axhline(envelope, color="#94a3b8", linewidth=1.0, linestyle="--", label="±5% envelope")
    axes[1].axhline(-envelope, color="#94a3b8", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("CUSUM")
    axes[1].set_xlabel("Datetime")

    if break_detected and 0 <= break_index < n_obs:
        break_rows = work[work["bar_index"] == break_index]
        if not break_rows.empty:
            break_time = break_rows["datetime"].iloc[0]
            for axis in axes:
                axis.axvline(
                    break_time,
                    color="#dc2626",
                    linewidth=1.2,
                    linestyle="--",
                    label="Break",
                )

    status = "BREAK" if break_detected else "PASS"
    fig.text(
        0.99,
        0.02,
        f"{window_label.title()} CUSUM {cusum_statistic:.2f} / {cusum_critical:.2f} ({status})",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#334155",
    )
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def write_rolling_cusum_artifacts(
    rolling_is: RollingISResult,
    output_dir: Path,
    observation_datetimes: Sequence[str] | None,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, Path]:
    """Write CSV + PNG rolling/CUSUM artifacts under ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "robustness_rolling_cusum.csv"
    frame = build_rolling_cusum_frame(
        rolling_is,
        observation_datetimes,
        periods_per_year=periods_per_year,
    )
    frame.to_csv(csv_path, index=False)

    plot_dir = output_dir / "matplotlib"
    plot_path = plot_dir / "cusum_stability.png"
    plotted = plot_rolling_cusum_stability_csv(
        csv_path,
        plot_path,
        periods_per_year=periods_per_year,
    )

    artifacts: dict[str, Path] = {"rolling_cusum_csv": csv_path}
    if plotted is not None:
        artifacts["cusum_plot_png"] = plotted
    return artifacts
