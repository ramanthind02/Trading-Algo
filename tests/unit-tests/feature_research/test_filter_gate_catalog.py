"""Unit tests for automatic exploration filter-gate catalog expansion."""
from __future__ import annotations

import pytest

from research.feature.exploration.filter_gate_catalog import (
    ExplorationFilterGatesConfig,
    atr_rank_fraction_grid,
    build_exploration_catalog_with_filter_gates,
    exploration_filter_gate_combo_multiplier,
    exploration_pass1_vol_regime_enabled,
    resolved_exploration_bias_spec,
)
from research.feature.in_sample.data_loader import expand_bias_specs
from lib.core.enums import TimeFrame


def _signal_spec() -> dict[str, object]:
    return {
        "module_name": "donchian_long_only",
        "timeframes": [TimeFrame.D],
        "params": {"entry_lookback": [10, 20], "exit_lookback": [5, 15]},
    }


def test_atr_rank_fraction_grid_center_and_perturbation() -> None:
    low = atr_rank_fraction_grid(0.2, pct=0.3)
    high = atr_rank_fraction_grid(0.6, pct=0.3)
    assert low == pytest.approx(sorted([0.14, 0.2, 0.26]), rel=1e-6)
    assert high == pytest.approx(sorted([0.42, 0.6, 0.78]), rel=1e-6)


def test_exploration_filter_gate_combo_multiplier_is_15() -> None:
    assert exploration_filter_gate_combo_multiplier() == 15


def test_build_exploration_catalog_branch_count() -> None:
    catalog = build_exploration_catalog_with_filter_gates(_signal_spec())
    assert len(catalog) == 4
    assert catalog[0]["module_name"] == "donchian_long_only"
    assert catalog[1]["module_name"] == ["filter_gate_entry_only", "filter_gate"]
    assert catalog[1]["params"]["filter_module"] == "sma_above_filter"
    assert catalog[2]["params"]["filter_module"] == "atr_percentile_filter"
    assert catalog[2]["params"]["filter_params"]["percentile_tail"] == "low"
    assert catalog[3]["params"]["filter_params"]["percentile_tail"] == "high"


def test_build_exploration_catalog_combo_count_formula() -> None:
    catalog = build_exploration_catalog_with_filter_gates(_signal_spec())
    signal_combos = len(expand_bias_specs(_signal_spec()))
    expanded = len(expand_bias_specs(catalog))
    assert signal_combos == 4
    assert expanded == signal_combos * exploration_filter_gate_combo_multiplier()


def test_build_exploration_catalog_disabled_returns_signal_only() -> None:
    gates = ExplorationFilterGatesConfig(enabled=False)
    catalog = build_exploration_catalog_with_filter_gates(_signal_spec(), gate_config=gates)
    assert len(catalog) == 1
    assert catalog[0]["module_name"] == "donchian_long_only"


def test_build_exploration_catalog_rejects_filter_gate_input() -> None:
    gated = {
        "module_name": "filter_gate",
        "timeframes": [TimeFrame.D],
        "params": {
            "filter_module": "sma_above_filter",
            "filter_params": {"period": 252},
            "signal_module": "donchian_long_only",
            "signal_params": {"entry_lookback": 20},
        },
    }
    with pytest.raises(ValueError, match="must not already be a filter gate"):
        build_exploration_catalog_with_filter_gates(gated)


def test_resolved_exploration_bias_spec_on_minimal_config() -> None:
    from datetime import datetime

    from research.feature.config import (
        BinningAnalysisConfig,
        ExplorationFilterGatesConfig,
        InSampleDefaultsCatalog,
        InSamplePhaseDefaultsConfig,
        PermutationResearchConfig,
        ResearchConfig,
    )
    from features.validation.objective_metrics import ObjectiveMetricSpec
    from pathlib import Path
    from research.feature.config import FeatureType
    from lib.core.enums import Direction, Ticker

    spec = _signal_spec()
    in_sample = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=spec,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/signed_signal/test"),
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=spec,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/signed_signal/test"),
        ),
    )
    config = ResearchConfig(
        tickers=[Ticker.GC],
        start=datetime(2000, 1, 1),
        end=datetime(2018, 12, 31),
        permutation=PermutationResearchConfig(
            objective_metric=ObjectiveMetricSpec(builtin="t_stat"),
        ),
        objective_metric_presets={},
        binning_params=BinningAnalysisConfig(),
        in_sample_defaults=in_sample,
        feature_type=FeatureType.SIGNED_SIGNAL,
        exploration_filter_gates=ExplorationFilterGatesConfig(
            scope="full_signal_grid",
        ),
    )
    resolved = resolved_exploration_bias_spec(config)
    assert isinstance(resolved, list)
    assert len(resolved) == 4
    assert exploration_pass1_vol_regime_enabled(config)


def test_resolved_exploration_bias_spec_respects_disabled_gates() -> None:
    from datetime import datetime
    from pathlib import Path

    from research.feature.config import (
        BinningAnalysisConfig,
        InSampleDefaultsCatalog,
        InSamplePhaseDefaultsConfig,
        PermutationResearchConfig,
        ResearchConfig,
    )
    from research.feature.exploration.filter_gate_catalog import ExplorationFilterGatesConfig
    from features.validation.objective_metrics import ObjectiveMetricSpec
    from research.feature.config import FeatureType
    from lib.core.enums import Direction, Ticker

    spec = _signal_spec()
    in_sample = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=spec,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/signed_signal/test"),
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=spec,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/signed_signal/test"),
        ),
    )
    config = ResearchConfig(
        tickers=[Ticker.GC],
        start=datetime(2000, 1, 1),
        end=datetime(2018, 12, 31),
        permutation=PermutationResearchConfig(
            objective_metric=ObjectiveMetricSpec(builtin="t_stat"),
        ),
        objective_metric_presets={},
        binning_params=BinningAnalysisConfig(),
        in_sample_defaults=in_sample,
        feature_type=FeatureType.SIGNED_SIGNAL,
        exploration_filter_gates=ExplorationFilterGatesConfig(enabled=False),
    )
    assert resolved_exploration_bias_spec(config) == spec
    assert not exploration_pass1_vol_regime_enabled(config)


def test_resolve_exploration_winner_params_single_combo_grid() -> None:
    from datetime import datetime
    from pathlib import Path

    from research.feature.config import (
        BinningAnalysisConfig,
        FeatureType,
        InSampleDefaultsCatalog,
        InSamplePhaseDefaultsConfig,
        PermutationResearchConfig,
        ResearchConfig,
    )
    from research.feature.exploration.filter_gate_catalog import (
        resolve_exploration_winner_params,
    )
    from features.validation.objective_metrics import ObjectiveMetricSpec
    from lib.core.enums import Direction, Ticker, TimeFrame

    spec = {
        "module_name": "close_breakout",
        "timeframes": [TimeFrame.D],
        "params": {},
    }
    in_sample = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=spec,
            target_col="log_return",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/signed_signal/test"),
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=spec,
            target_col="log_return",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/signed_signal/test"),
        ),
    )
    config = ResearchConfig(
        tickers=[Ticker.NQ],
        start=datetime(2000, 1, 1),
        end=datetime(2022, 12, 31),
        permutation=PermutationResearchConfig(
            objective_metric=ObjectiveMetricSpec(builtin="t_stat"),
        ),
        objective_metric_presets={},
        binning_params=BinningAnalysisConfig(),
        in_sample_defaults=in_sample,
        feature_type=FeatureType.SIGNED_SIGNAL,
    )
    winner = resolve_exploration_winner_params(config, robustness_report=None)
    assert winner.get("_bias_module") == "close_breakout"
