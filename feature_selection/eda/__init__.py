"""Feature Validator EDA pipeline — fresh implementation (T001–T004)."""
from .eda_dataclasses import (
    DescriptiveStats, TemporalStability, CorrelationAnalysis,
    CommonEDAPlots, CommonEDAStats,
    DecileBinStats, DecileAnalysis, MonotonicityTest,
    DistributionDiagnostics, ContinuousEDAPlots, ContinuousEDAStats,
    LevelStats, PerLevelStats, BootstrapCI, BootstrapCIResults,
    TransitionMatrix, RuleBasedEDAPlots, RuleBasedEDAStats,
    EDAMetadata, EDAConfig, DiagnosticFlags,
    ContinuousEDAReport, RuleBasedEDAReport,
)

__all__ = [
    "DescriptiveStats", "TemporalStability", "CorrelationAnalysis",
    "CommonEDAPlots", "CommonEDAStats",
    "DecileBinStats", "DecileAnalysis", "MonotonicityTest",
    "DistributionDiagnostics", "ContinuousEDAPlots", "ContinuousEDAStats",
    "LevelStats", "PerLevelStats", "BootstrapCI", "BootstrapCIResults",
    "TransitionMatrix", "RuleBasedEDAPlots", "RuleBasedEDAStats",
    "EDAMetadata", "EDAConfig", "DiagnosticFlags",
    "ContinuousEDAReport", "RuleBasedEDAReport",
]
