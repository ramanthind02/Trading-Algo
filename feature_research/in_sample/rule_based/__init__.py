from __future__ import annotations

from typing import TYPE_CHECKING

from feature_research.in_sample.rule_based.config import RuleBasedResearchConfig, load_config

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from feature_research.in_sample.rule_based import pipeline as _pipeline

__all__ = [
    "RuleBasedResearchConfig",
    "load_config",
    "run_rule_based_eda_pipeline",
    "run_rule_based_permutation_pipeline",
    "run_rule_based_walkforward_pipeline",
]

_LAZY_EXPORTS = {
    "run_rule_based_eda_pipeline",
    "run_rule_based_permutation_pipeline",
    "run_rule_based_walkforward_pipeline",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from feature_research.in_sample.rule_based import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
