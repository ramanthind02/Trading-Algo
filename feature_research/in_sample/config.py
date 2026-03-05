"""Unified in-sample research config. Consolidates continuous_binning and rule_based paths.

This module provides a single ResearchConfig that handles both feature types.
The feature_type field determines validation behavior:
  - CONTINUOUS: uses BinningAnalysisConfig, supports Phase 2 binning analysis
  - RULE_BASED: uses fixed 3-level binning, skips Phase 2 analysis

Researcher-editable phase presets (bias specs, targets, reports dirs) live in
``feature_research/config.py``. This module mainly provides the runtime
``ResearchConfig`` shape, and re-exports ``BinningAnalysisConfig`` for backward
compatibility while assembling shared defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from feature_research.config import (
    BinningAnalysisConfig,
    OBJECTIVE_METRIC_PRESETS,
    RAW_TARGET_COLS,
    FeatureType,
    OOSWindowConfig,
    ParamSensitivityConfig,
    PermutationResearchConfig,
    load_config as load_base_config,
)
from utils.core.enums import Ticker

IN_SAMPLE_OBJECTIVE_METRIC_PRESETS = OBJECTIVE_METRIC_PRESETS


__all__ = [
    "BinningAnalysisConfig",
    "FeatureType",
    "IN_SAMPLE_OBJECTIVE_METRIC_PRESETS",
    "ParamSensitivityConfig",
    "PermutationResearchConfig",
    "ResearchConfig",
    "load_config",
]

@dataclass(frozen=True)
class ResearchConfig:
    """Unified in-sample research config for both continuous and rule-based features.

    Attributes
    ----------
    tickers : list[Ticker]
        Instruments to include. Data is concatenated across tickers.
    start : datetime
        In-sample period start (inclusive).
    end : datetime
        In-sample period end (inclusive).
    bias_spec : dict[str, Any]
        Bias-node specification. ``params`` values may be lists for grid search.
        For CONTINUOUS: typically RSI with lookback params.
        For RULE_BASED: typically RSI_SIGNAL with rsi_period, oversold, overbought, etc.
    target_col : str
        Column from ``targets_df`` to use as the prediction target.
        Options: ``"log_return"``, ``"log_return_atr"``, ``"log_return_ewsd"``.
    strategy : str
        Passed to the EDA summary label only.
        Options: ``"long"``, ``"short"``, ``"long-short"``.
    use_cache : bool
        Whether to use pre-computed caches for feature extraction.
    populate_cache : bool
        If True, run ``CacheManager.populate_cache()`` before extraction.
    reports_dir : Path
        Root output directory for EDA reports.
    feature_type : FeatureType
        CONTINUOUS or RULE_BASED. Determines validation behavior and binning approach.
    in_sample_permutation : PermutationResearchConfig
        In-sample permutation settings (if enabled).
    binning_params : BinningAnalysisConfig
        Binning hyperparameters. Only used for CONTINUOUS; rule-based uses fixed 3 levels.
    param_sensitivity : ParamSensitivityConfig
        Parameter sensitivity / stable region selection (notebook and reports).
    validation_window : OOSWindowConfig | None
        Explicit train/test window for validation run. When None, validation is disabled.
    oos_window : OOSWindowConfig | None
        Explicit train/test window for out-of-sample run. When None, OOS is disabled.
    top_k : int
        Number of top parameter combinations to select.
    objective_metric_name : str
        Key into objective metric presets (e.g. "t_stat", "sharpe").
    smoothing_self_weight : float
        Center param weight relative to each 1-step neighbor in smoothing.
    n_jobs : int
        Parallel jobs for scoring param combos; -1 = all CPUs.
    output_root : Path
        Root directory for research output artifacts.
    """

    # --- Required fields (no defaults) ---
    tickers: list[Ticker]
    start: datetime
    end: datetime
    bias_spec: dict[str, Any]
    target_col: str
    strategy: str
    use_cache: bool
    populate_cache: bool
    reports_dir: Path
    # --- Optional fields (with defaults) ---
    feature_type: FeatureType = FeatureType.CONTINUOUS
    in_sample_permutation: PermutationResearchConfig = field(
        default_factory=lambda: PermutationResearchConfig(
            objective_metric=OBJECTIVE_METRIC_PRESETS["t_stat"],
        )
    )
    binning_params: BinningAnalysisConfig = field(default_factory=BinningAnalysisConfig)
    param_sensitivity: ParamSensitivityConfig = field(default_factory=ParamSensitivityConfig)
    validation_window: OOSWindowConfig | None = None
    oos_window: OOSWindowConfig | None = None
    # Flat evaluation fields
    top_k: int = 1
    objective_metric_name: str = "t_stat"
    smoothing_self_weight: float = 3.0
    n_jobs: int = 8
    output_root: Path = field(default_factory=lambda: Path("feature_research/shared_results"))

    @property
    def permutation_suite(self) -> PermutationResearchConfig:
        """Backward-compatible alias for in-sample permutation settings."""
        return self.in_sample_permutation

    def __post_init__(self) -> None:
        if self.target_col in RAW_TARGET_COLS and len(self.tickers) > 1:
            raise ValueError(
                f"target_col='{self.target_col}' uses raw (unnormalized) returns with "
                f"multiple tickers ({len(self.tickers)} configured). "
                "Raw returns cannot be compared across tickers with different volatility. "
                "Use 'log_return_ewsd' or 'log_return_atr' for multi-ticker research."
            )


def load_config() -> ResearchConfig:
    """Return unified research config with feature_type dispatched from base config.

    This is the single entrypoint for loading in-sample research config.
    It checks base.feature_type to determine which phase-specific defaults to apply.
    """
    base = load_base_config()
    phase_defaults = base.in_sample_defaults.for_feature_type(base.feature_type)

    # Param-sensitivity / binning metric source of truth lives in shared
    # BinningAnalysisConfig (feature_research.config), not the permutation presets.
    binning_overrides = dict(phase_defaults.binning_params_overrides)
    binning_overrides.setdefault("selection_metric", BinningAnalysisConfig().selection_metric)
    binning_overrides.setdefault("strategy", phase_defaults.strategy)
    binning_params = BinningAnalysisConfig(**binning_overrides)

    return ResearchConfig(
        feature_type=base.feature_type,
        tickers=base.tickers,
        start=base.start,
        end=base.end,
        bias_spec=phase_defaults.bias_spec,
        target_col=phase_defaults.target_col,
        strategy=phase_defaults.strategy,
        use_cache=base.use_cache,
        populate_cache=base.populate_cache,
        reports_dir=phase_defaults.reports_dir,
        in_sample_permutation=base.permutation,
        binning_params=binning_params,
        param_sensitivity=base.param_sensitivity,
        validation_window=base.validation_window,
        oos_window=base.oos_window,
        top_k=base.top_k,
        objective_metric_name=base.objective_metric_key,
        smoothing_self_weight=base.smoothing_self_weight,
        n_jobs=base.n_jobs,
        output_root=base.output_root,
    )
