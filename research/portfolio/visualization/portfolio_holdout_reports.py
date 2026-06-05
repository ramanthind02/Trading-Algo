"""Matplotlib reports for portfolio holdout validation (SaaS §8)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quantfoundry_core.robustness.portfolio_holdout import PortfolioHoldoutReport



def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_correlation_heatmaps(
    research_csv: Path,
    holdout_csv: Path,
    output_path: Path,
) -> Path:
    """Side-by-side research vs holdout correlation heatmaps."""

    c_research = pd.read_csv(research_csv, index_col=0)
    c_holdout = pd.read_csv(holdout_csv, index_col=0)
    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, frame, title in (
        (axes[0], c_research, "Research correlations"),
        (axes[1], c_holdout, "Holdout correlations"),
    ):
        values = frame.to_numpy(dtype=float)
        im = ax.imshow(values, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        ax.set_xticks(range(len(frame.columns)))
        ax.set_yticks(range(len(frame.index)))
        ax.set_xticklabels(frame.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(frame.index, fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_contribution_waterfall(contributions_csv: Path, output_path: Path) -> Path:
    """Horizontal bars of realised vs expected strategy contributions."""

    frame = pd.read_csv(contributions_csv)
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.4 * frame.shape[0])))
    y = np.arange(frame.shape[0])
    height = 0.35
    ax.barh(y - height / 2, frame["realised_contribution"], height=height, label="Realised", color="#2563eb")
    ax.barh(
        y + height / 2,
        frame["expected_contribution"],
        height=height,
        label="Expected",
        color="#94a3b8",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["strategy"])
    ax.set_xlabel("Contribution share")
    ax.set_title("Holdout strategy contributions")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_holdout_equity_with_culling(
    holdout_returns: pd.Series,
    output_path: Path,
    *,
    culling_dates: list[pd.Timestamp] | None = None,
) -> Path:
    """Portfolio equity curve with optional monitoring culling markers."""

    returns = holdout_returns.dropna().sort_index()
    equity = (1.0 + returns).cumprod()
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(equity.index, equity.to_numpy(dtype=float), color="#111827", linewidth=1.6)
    for event in culling_dates or []:
        ax.axvline(event, color="#ef4444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Portfolio holdout equity")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_portfolio_holdout_plots(
    report: PortfolioHoldoutReport,
    *,
    output_dir: Path,
    correlation_research_csv: Path,
    correlation_holdout_csv: Path,
    contributions_csv: Path,
    holdout_portfolio_returns: pd.Series,
) -> list[Path]:
    """Render portfolio-specific Matplotlib charts (monitoring plots are written separately)."""

    _ = report
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_correlation_heatmaps(
            correlation_research_csv,
            correlation_holdout_csv,
            output_dir / "correlation_heatmaps.png",
        ),
        plot_contribution_waterfall(
            contributions_csv,
            output_dir / "contribution_waterfall.png",
        ),
        plot_holdout_equity_with_culling(
            holdout_portfolio_returns,
            output_dir / "portfolio_equity_holdout.png",
        ),
    ]
