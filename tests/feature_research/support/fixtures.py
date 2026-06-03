"""Explicit ResearchConfig fixtures for tests — never import ``load_config()`` here."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from feature_research.config import (
    BinningAnalysisConfig,
    EvaluationDefaultsCatalog,
    EvaluationPhaseDefaultsConfig,
    FeatureType,
    InSampleDefaultsCatalog,
    InSamplePhaseDefaultsConfig,
    PermutationResearchConfig,
    PortfolioInclusionConfig,
    PortfolioSourceConfig,
    ResearchConfig,
    ResearchWindowConfig,
    VaultSaveConfig,
    build_objective_metric_presets,
)
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from utils.core.enums import Direction, Ticker, TimeFrame


def minimal_research_config(**overrides: Any) -> ResearchConfig:
    """Minimal signed-signal ``ResearchConfig`` for unit/integration tests."""
    timeframe = overrides.pop("timeframe", TimeFrame.D)
    objective_metric_presets = build_objective_metric_presets(timeframe)
    research_window = ResearchWindowConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2018, 12, 31),
        val_start=datetime(2019, 1, 1),
        val_end=datetime(2022, 12, 31),
    )
    bias_spec: dict[str, Any] = {
        "module_name": "buy_hold",
        "timeframes": [timeframe],
        "params": {},
    }
    reports_dir = Path("feature_research/in_sample/results/signed_signal/test_fixture")
    in_sample_defaults = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=bias_spec,
            target_col="log_return",
            strategy=Direction.LONG,
            reports_dir=reports_dir,
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=bias_spec,
            target_col="log_return",
            strategy=Direction.LONG,
            reports_dir=reports_dir,
        ),
    )
    evaluation_defaults = EvaluationDefaultsCatalog(
        continuous=EvaluationPhaseDefaultsConfig(bias_spec=bias_spec),
        signed_signal=EvaluationPhaseDefaultsConfig(bias_spec=bias_spec),
    )
    base = ResearchConfig(
        tickers=[Ticker.ES],
        start=research_window.train_start,
        end=research_window.val_end,
        permutation=PermutationResearchConfig(
            objective_metric=objective_metric_presets["t_stat"],
        ),
        objective_metric_presets=objective_metric_presets,
        binning_params=BinningAnalysisConfig(strategy=Direction.LONG),
        in_sample_defaults=in_sample_defaults,
        timeframe=timeframe,
        feature_type=FeatureType.SIGNED_SIGNAL,
        evaluation_defaults=evaluation_defaults,
        research_window=research_window,
        output_root=Path("feature_research/shared_results/test_fixture"),
        portfolio_source=PortfolioSourceConfig(tickers=(Ticker.ES,)),
        portfolio_inclusion=PortfolioInclusionConfig(
            ephemeral_weight_hierarchy_group="buy_hold",
            candidate_tickers=(Ticker.ES,),
        ),
        vault_save=VaultSaveConfig(
            direction=Direction.LONG,
            ensemble_name="fixture_ensemble",
            weight_hierarchy_group="buy_hold",
            tickers=(Ticker.ES,),
            dry_run=True,
        ),
    )
    return replace(base, **overrides) if overrides else base
