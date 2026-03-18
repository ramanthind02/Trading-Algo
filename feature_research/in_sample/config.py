"""Re-export shim for backward compatibility.

The unified ResearchConfig now lives in feature_research.config.
This module exists only to maintain existing import paths.
"""
from feature_research.config import (
    BinningAnalysisConfig,
    FeatureType,
    ParamSensitivityConfig,
    PermutationResearchConfig,
    ResearchConfig,
    build_objective_metric_presets,
    load_config,
)

__all__ = [
    "BinningAnalysisConfig",
    "FeatureType",
    "ParamSensitivityConfig",
    "PermutationResearchConfig",
    "ResearchConfig",
    "build_objective_metric_presets",
    "load_config",
]
