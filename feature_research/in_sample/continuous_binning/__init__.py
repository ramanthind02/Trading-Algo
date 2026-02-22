from __future__ import annotations

from typing import TYPE_CHECKING

from feature_research.in_sample.continuous_binning.config import ResearchConfig, load_config

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from feature_research.in_sample.continuous_binning import pipeline as _pipeline

__all__ = [
    "ResearchConfig",
    "load_config",
    "run_continuous_eda_pipeline",
    "run_continuous_permutation_pipeline",
    "run_continuous_walkforward_pipeline",
]

_LAZY_EXPORTS = {
    "run_continuous_eda_pipeline",
    "run_continuous_permutation_pipeline",
    "run_continuous_walkforward_pipeline",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from feature_research.in_sample.continuous_binning import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
