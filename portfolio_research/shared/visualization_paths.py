"""Path helpers for portfolio holdout artifacts."""
from __future__ import annotations

from pathlib import Path


def holdout_root(output_root: Path) -> Path:
    return Path(output_root) / "holdout"


def holdout_returns_dir(output_root: Path) -> Path:
    return holdout_root(output_root) / "returns"


def holdout_test_dir(output_root: Path) -> Path:
    return holdout_root(output_root) / "test"


def holdout_strategy_dir(output_root: Path, strategy_name: str) -> Path:
    safe = strategy_name.replace("/", "_").replace(" ", "_")
    return holdout_root(output_root) / "strategies" / safe


def holdout_portfolio_visualization_dir(output_root: Path) -> Path:
    return holdout_root(output_root) / "visualization" / "portfolio"
