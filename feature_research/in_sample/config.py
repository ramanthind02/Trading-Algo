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
    GlobalResearchDefaults,
    OBJECTIVE_METRIC_PRESETS,
    RAW_TARGET_COLS,
    FeatureType,
    OOSWindowConfig,
    ParamSensitivityConfig,
    PermutationResearchConfig,
    load_config as load_base_config,
)
from feature_research.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
    WeightLayerAlgorithm,
    _coerce_enum_or_raise,
)
from utils.core.enums import Ticker

IN_SAMPLE_OBJECTIVE_METRIC_PRESETS = OBJECTIVE_METRIC_PRESETS


def _default_permutation_from_global() -> PermutationResearchConfig:
    """Build permutation config from GlobalResearchDefaults so tests/defaults stay aligned."""
    g = GlobalResearchDefaults()
    return PermutationResearchConfig(
        OBJECTIVE_METRIC_PRESETS[g.objective_metric_key],
        g.top_k,
        g.min_folds_stable,
        g.fold_years,
    )


__all__ = [
    "BinningAnalysisConfig",
    "FeatureType",
    "IN_SAMPLE_OBJECTIVE_METRIC_PRESETS",
    "ParamSensitivityConfig",
    "PermutationResearchConfig",
    "ResearchConfig",
    "WalkforwardResearchConfig",
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
    walkforward_selection_method : WalkforwardSelectionMethod | str
        Selection algorithm. Must match ``walkforward.selection_method``.
        Set at the top level here so researchers don't need to dig into
        ``walkforward`` to find/change it. Accepts string values for convenience.
    weight_layer_algorithm : WeightLayerAlgorithm | str
        Weight layer method. Must match ``walkforward.weight_layer_algorithm``.
        Accepts string values for convenience.
    in_sample_permutation : PermutationResearchConfig
        In-sample permutation settings (if enabled).
    binning_params : BinningAnalysisConfig
        Binning hyperparameters. Only used for CONTINUOUS; rule-based uses fixed 3 levels.
    walkforward : WalkforwardResearchConfig
        Walkforward config (window, selection, etc.). In-sample never runs walkforward;
        the standalone script ``feature_research/walkforward/run_walkforward.py`` uses
        this and sets enabled=True when running walkforward.
    param_sensitivity : ParamSensitivityConfig
        Parameter sensitivity / stable region selection (notebook and reports).
    oos_window : OOSWindowConfig | None
        Explicit train/test window for out-of-sample run. When None, OOS is disabled.
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
    walkforward_selection_method: WalkforwardSelectionMethod | str = (
        WalkforwardSelectionMethod.TOP_K
    )
    weight_layer_algorithm: WeightLayerAlgorithm | str = (
        WeightLayerAlgorithm.INVERSE_CORRELATION
    )
    in_sample_permutation: PermutationResearchConfig = field(
        default_factory=lambda: _default_permutation_from_global()
    )
    binning_params: BinningAnalysisConfig = field(default_factory=BinningAnalysisConfig)
    walkforward: WalkforwardResearchConfig = field(
        default_factory=lambda: WalkforwardResearchConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2023, 12, 31),
            enabled=False,
        )
    )
    param_sensitivity: ParamSensitivityConfig = field(default_factory=ParamSensitivityConfig)
    validation_window: OOSWindowConfig | None = None
    oos_window: OOSWindowConfig | None = None

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

        # Coerce and validate walkforward_selection_method
        normalized_sm = _coerce_enum_or_raise(
            self.walkforward_selection_method,
            WalkforwardSelectionMethod,
            "walkforward_selection_method",
        )
        object.__setattr__(self, "walkforward_selection_method", normalized_sm)
        if normalized_sm != self.walkforward.selection_method:
            raise ValueError(
                f"walkforward_selection_method ({normalized_sm.value!r}) does not match "
                f"walkforward.selection_method ({self.walkforward.selection_method.value!r}). "
                "Set both consistently or build walkforward via BaseResearchConfig.build_walkforward()."
            )

        # Coerce and validate weight_layer_algorithm
        normalized_wla = _coerce_enum_or_raise(
            self.weight_layer_algorithm,
            WeightLayerAlgorithm,
            "weight_layer_algorithm",
        )
        object.__setattr__(self, "weight_layer_algorithm", normalized_wla)
        if normalized_wla != self.walkforward.weight_layer_algorithm:
            raise ValueError(
                f"weight_layer_algorithm ({normalized_wla.value!r}) does not match "
                f"walkforward.weight_layer_algorithm ({self.walkforward.weight_layer_algorithm.value!r}). "
                "Set both consistently or build walkforward via BaseResearchConfig.build_walkforward()."
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

    walkforward = base.build_walkforward(
        train_window_years=phase_defaults.walkforward_train_window_years,
        test_window_years=phase_defaults.walkforward_test_window_years,
        num_steps=phase_defaults.walkforward_num_steps,
        enabled=False,  # In-sample never runs walkforward; standalone run_walkforward.py sets enabled=True
    )

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
        walkforward_selection_method=base.walkforward_defaults.selection_method,
        weight_layer_algorithm=base.walkforward_defaults.weight_layer_algorithm,
        in_sample_permutation=base.permutation,
        binning_params=binning_params,
        walkforward=walkforward,
        param_sensitivity=base.param_sensitivity,
        validation_window=base.validation_window,
        oos_window=base.oos_window,
    )
