"""
Walk-forward research engine (fold scoring, portfolio eval, artifacts).

Rolling-window helpers previously re-exported here live alongside their callers;
import submodules explicitly (e.g. ``config``, ``runner``).
"""

from . import (
    config,
    evaluators,
    io,
    portfolio_evaluator,
    research_data,
    runner,
)

from .config import (
    MemberPredictionMode,
    WalkforwardResearchConfig,
    WeightLayerAlgorithm,
)
from .io import (
    resolve_walkforward_output_dir,
    write_walkforward_artifacts,
)
from .runner import (
    WalkforwardRunReport,
    build_fold_rows_from_explicit_specs,
    run_walkforward_research,
)

__all__ = [
    "WalkforwardResearchConfig",
    "WeightLayerAlgorithm",
    "MemberPredictionMode",
    "WalkforwardRunReport",
    "run_walkforward_research",
    "build_fold_rows_from_explicit_specs",
    "resolve_walkforward_output_dir",
    "write_walkforward_artifacts",
    "config",
    "evaluators",
    "io",
    "portfolio_evaluator",
    "research_data",
    "runner",
]
