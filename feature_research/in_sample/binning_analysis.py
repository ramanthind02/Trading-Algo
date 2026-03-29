"""Legacy binning-analysis entry point.

Frozen signed-signal research no longer supports fitted binning analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig


def binning_model_from_config(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("Legacy binning analysis is unsupported in frozen-signal research.")


def run_binning_analysis_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Path]:
    _ = (config, output_dir, dry_run)
    raise RuntimeError("Legacy binning analysis is unsupported in frozen-signal research.")
