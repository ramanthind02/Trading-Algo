"""Feature Validator EDA pipeline — fresh implementation (T001–T004)."""
from .eda_dataclasses import (
    DescriptiveStats, TemporalStability, CorrelationAnalysis,
    ICDecay, FeatureACF,
    CommonEDAPlots, CommonEDAStats,
    DecileBinStats, DecileAnalysis,
    QuintileSpread,
    DistributionDiagnostics, ContinuousEDAPlots, ContinuousEDAStats,
    LevelStats, PerLevelStats, BootstrapCI, BootstrapCIResults,
    RuleBasedEDAPlots, RuleBasedEDAStats,
    EDAMetadata, EDAConfig, DiagnosticFlags,
    ContinuousEDAReport, RuleBasedEDAReport,
)

__all__ = [
    "DescriptiveStats", "TemporalStability", "CorrelationAnalysis",
    "ICDecay", "FeatureACF",
    "CommonEDAPlots", "CommonEDAStats",
    "DecileBinStats", "DecileAnalysis",
    "QuintileSpread",
    "DistributionDiagnostics", "ContinuousEDAPlots", "ContinuousEDAStats",
    "LevelStats", "PerLevelStats", "BootstrapCI", "BootstrapCIResults",
    "RuleBasedEDAPlots", "RuleBasedEDAStats",
    "EDAMetadata", "EDAConfig", "DiagnosticFlags",
    "ContinuousEDAReport", "RuleBasedEDAReport",
]
