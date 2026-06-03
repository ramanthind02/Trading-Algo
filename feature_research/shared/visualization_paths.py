"""Pure path helpers for CSV-first feature_research artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

VISUALIZATION_SUBDIR_NAME = "visualization"


def walkforward_visualization_csv_dir(
    output_root: Path,
    phase: Literal["validation", "oos"],
) -> Path:
    """Stable folder for validation / OOS visualization CSVs."""

    return Path(output_root) / VISUALIZATION_SUBDIR_NAME / phase


def validation_walkforward_output_dir(output_root: Path, module_name: str) -> Path:
    """Walk-forward validation artifacts (fold scores, tearsheets, report.json)."""

    return Path(output_root) / "signed_signal" / module_name.strip() / "validation"


def canonical_in_sample_visualization_dir() -> Path:
    """Stable in-repo folder for in-sample visualization CSV inputs."""

    return (
        Path(__file__).resolve().parents[1]
        / "in_sample"
        / "results"
        / VISUALIZATION_SUBDIR_NAME
    )
