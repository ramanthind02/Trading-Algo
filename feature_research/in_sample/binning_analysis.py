"""Legacy binning-analysis entry point.

Frozen signed-signal research no longer supports fitted binning analysis.
Dry-run and config helpers remain for tests and tooling.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig


def _extract_binning_params(config: "ResearchConfig") -> dict:
    return asdict(config.binning_params)


def binning_model_from_config(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("Legacy binning analysis is unsupported in frozen-signal research.")


def run_binning_analysis_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Path]:
    if dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        return {}
    _ = config
    raise RuntimeError("Legacy binning analysis is unsupported in frozen-signal research.")
