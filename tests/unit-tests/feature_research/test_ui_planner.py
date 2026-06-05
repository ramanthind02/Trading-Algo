from __future__ import annotations

from datetime import datetime

import pytest

from research.feature.config import ResearchWindowConfig
from research.feature.shared import FeatureResearchPhase
from research.feature.ui.contracts import ComboDiagnostics
from research.feature.ui.module_catalog import module_options
from research.feature.ui.contracts import UiRunAction
from research.feature.ui.planner import (
    apply_ui_request,
    build_phase_command,
    build_phase_plan,
    build_ui_defaults,
    build_ui_request,
)
from nodes._taxonomy import CANONICAL_MODULE_CLASSES
from tests.feature_research.support.fixtures import minimal_research_config
from lib.core.enums import Direction, Ticker
from research.evaluation.walkforward.io import resolve_walkforward_output_dir

def test_build_ui_request_uses_default_tickers_when_blank() -> None:
    config = minimal_research_config()
    defaults = build_ui_defaults(config)

    request = build_ui_request(
        phase_name=FeatureResearchPhase.EXPLORATION.value,
        tickers_text="",
        fallback_tickers=defaults.default_tickers,
    )

    assert request.phase is FeatureResearchPhase.EXPLORATION
    assert request.tickers == defaults.default_tickers
    assert request.window_override is None


def test_build_ui_defaults_exposes_configured_bias_summary() -> None:
    defaults = build_ui_defaults(minimal_research_config())

    assert defaults.configured_module == "buy_hold"
    assert defaults.configured_search_space
    assert defaults.workflow_steps
    assert defaults.config_source_path == "feature_research/config.py"
    assert defaults.workflow_steps[0].phase is FeatureResearchPhase.EXPLORATION
    assert defaults.configured_module_description == CANONICAL_MODULE_CLASSES["buy_hold"]


def test_module_options_use_taxonomy_backed_labels() -> None:
    options = {option.value: option for option in module_options()}

    assert options["stacked_sma_long_only"].label == "stacked_sma_long_only"
    assert options["stacked_sma_long_only"].description == CANONICAL_MODULE_CLASSES["stacked_sma_long_only"]
    assert options["cyclical_rsi_signal"].label == "cyclical_rsi_signal"
    assert options["buy_hold"].description == CANONICAL_MODULE_CLASSES["buy_hold"]


def test_module_options_hide_explicit_long_short_controls() -> None:
    options = {option.value: option for option in module_options()}
    stacked_field_keys = {field.key for field in options["stacked_sma_long_only"].fields}
    cyclical_strategy_field = next(
        field for field in options["cyclical_rsi_signal"].fields if field.key == "strategy_mode"
    )

    assert "mode" not in stacked_field_keys
    assert {option.value for option in cyclical_strategy_field.options} == {"long", "short"}


def test_build_ui_request_rejects_partial_window_override() -> None:
    config = minimal_research_config()
    defaults = build_ui_defaults(config)

    with pytest.raises(ValueError, match="Fill all four window dates"):
        build_ui_request(
            phase_name=FeatureResearchPhase.VALIDATION.value,
            tickers_text="ES",
            fallback_tickers=defaults.default_tickers,
            train_start="2010-01-01",
            train_end="2015-01-01",
        )


def test_build_ui_request_parses_comma_separated_tickers() -> None:
    config = minimal_research_config()
    defaults = build_ui_defaults(config)

    request = build_ui_request(
        phase_name=FeatureResearchPhase.EXPLORATION.value,
        tickers_text="ES, NQ , GC",
        fallback_tickers=defaults.default_tickers,
    )

    assert request.tickers == (Ticker.ES, Ticker.NQ, Ticker.GC)


def test_build_phase_command_joins_multiple_tickers() -> None:
    config = minimal_research_config()
    defaults = build_ui_defaults(config)
    request = build_ui_request(
        phase_name=FeatureResearchPhase.EXPLORATION.value,
        tickers_text="ES,NQ,GC",
        fallback_tickers=defaults.default_tickers,
    )

    command = build_phase_command(request, UiRunAction.EXECUTE)
    plan = build_phase_plan(config, request)

    assert "--tickers ES,NQ,GC" in command
    assert any(
        field.label == "Research tickers" and field.value == "ES, NQ, GC"
        for field in plan.fields
    )


def test_apply_ui_request_updates_phase_window_and_related_tickers() -> None:
    config = minimal_research_config()
    original_bias_spec = config.bias_spec
    original_eval_bias_spec = config.eval_bias_spec
    request = build_ui_request(
        phase_name=FeatureResearchPhase.VALIDATION.value,
        tickers_text="ES,NQ",
        fallback_tickers=tuple(config.tickers),
        train_start="2001-01-01",
        train_end="2017-12-31",
        val_start="2018-01-01",
        val_end="2020-12-31",
    )

    updated = apply_ui_request(config, request)

    assert updated.tickers == [Ticker.ES, Ticker.NQ]
    assert updated.portfolio_source is not None
    assert updated.portfolio_source.tickers == config.portfolio_source.tickers
    assert updated.portfolio_inclusion is not None
    assert updated.portfolio_inclusion.candidate_tickers == (Ticker.ES, Ticker.NQ)
    assert updated.vault_save is not None
    assert updated.vault_save.tickers == (Ticker.ES, Ticker.NQ)
    assert updated.bias_spec == original_bias_spec
    assert updated.eval_bias_spec == original_eval_bias_spec
    assert updated.research_window == ResearchWindowConfig(
        train_start=datetime(2001, 1, 1),
        train_end=datetime(2017, 12, 31),
        val_start=datetime(2018, 1, 1),
        val_end=datetime(2020, 12, 31),
    )


def test_apply_ui_request_syncs_single_ticker_portfolio_context() -> None:
    from dataclasses import replace

    from research.feature.config import PortfolioInclusionConfig

    base = minimal_research_config()
    config = replace(
        base,
        portfolio_inclusion=PortfolioInclusionConfig(
            ephemeral_weight_hierarchy_group="crude_oil_mr",
            candidate_tickers=(Ticker.CL,),
            emit_tearsheets=True,
        ),
    )
    request = build_ui_request(
        phase_name=FeatureResearchPhase.VALIDATION.value,
        tickers_text="CL",
        fallback_tickers=tuple(config.tickers),
    )

    updated = apply_ui_request(config, request)

    assert updated.tickers == [Ticker.CL]
    assert updated.portfolio_source is not None
    assert updated.portfolio_source.tickers == base.portfolio_source.tickers
    assert updated.portfolio_inclusion is not None
    assert updated.portfolio_inclusion.candidate_tickers == (Ticker.CL,)
    assert updated.portfolio_inclusion.ephemeral_weight_hierarchy_group == "crude_oil_mr"


def test_apply_ui_request_portfolio_tickers_override() -> None:
    config = minimal_research_config()
    request = build_ui_request(
        phase_name=FeatureResearchPhase.VALIDATION.value,
        tickers_text="GC",
        portfolio_tickers_text="ES,NQ",
        fallback_tickers=tuple(config.tickers),
    )

    updated = apply_ui_request(config, request)

    assert updated.tickers == [Ticker.GC]
    assert updated.portfolio_source is not None
    assert updated.portfolio_source.tickers == (Ticker.ES, Ticker.NQ)
    assert updated.portfolio_inclusion is not None
    assert updated.portfolio_inclusion.candidate_tickers == (Ticker.GC,)
    assert request.portfolio_tickers == (Ticker.ES, Ticker.NQ)


def test_apply_ui_request_tickers_override_config_universe() -> None:
    from dataclasses import replace

    from research.feature.config import (
        PortfolioInclusionConfig,
        PortfolioSourceConfig,
        VaultSaveConfig,
    )

    portfolio_universe = (Ticker.ES, Ticker.GC, Ticker.NQ)
    config = replace(
        minimal_research_config(),
        tickers=[Ticker.ES, Ticker.GC, Ticker.NQ],
        portfolio_source=PortfolioSourceConfig(tickers=portfolio_universe),
        portfolio_inclusion=PortfolioInclusionConfig(
            ephemeral_weight_hierarchy_group="momentum",
            candidate_tickers=portfolio_universe,
        ),
        vault_save=VaultSaveConfig(
            direction=Direction.LONG,
            ensemble_name="close_breakout",
            weight_hierarchy_group="momentum",
            tickers=portfolio_universe,
            dry_run=True,
        ),
    )
    request = build_ui_request(
        phase_name=FeatureResearchPhase.VALIDATION.value,
        tickers_text="NQ",
        fallback_tickers=tuple(config.tickers),
    )

    updated = apply_ui_request(config, request)

    assert updated.tickers == [Ticker.NQ]
    assert updated.portfolio_source is not None
    assert updated.portfolio_source.tickers == portfolio_universe
    assert updated.portfolio_inclusion is not None
    assert updated.portfolio_inclusion.candidate_tickers == (Ticker.NQ,)
    assert updated.vault_save is not None
    assert updated.vault_save.tickers == (Ticker.NQ,)
    plan = build_phase_plan(config, request)
    gate_field = next(
        field for field in plan.fields if field.label == "Portfolio gate tickers"
    )
    assert gate_field.value == "ES, GC, NQ"


def test_build_phase_plan_for_validation_uses_real_output_dir(monkeypatch) -> None:
    config = minimal_research_config()
    monkeypatch.setattr(
        "research.feature.ui.planner.build_combo_diagnostics",
        lambda _config: ComboDiagnostics(
            raw_combo_count=6,
            loaded_combo_count=6,
            effective_combo_count=2.5,
            evaluation_combo_label="period_1_28__period_2_48",
            surface_interpretation="Smooth surface.",
        ),
    )
    request = build_ui_request(
        phase_name=FeatureResearchPhase.VALIDATION.value,
        tickers_text="ES",
        fallback_tickers=tuple(config.tickers),
    )

    plan = build_phase_plan(config, request)

    expected_output = resolve_walkforward_output_dir(
        feature_type=config.feature_type.value,
        module_name="buy_hold",
        root_dir=config.output_root,
        output_subdir="validation",
    )

    assert plan.phase is FeatureResearchPhase.VALIDATION
    assert plan.output_path == expected_output
    assert "research.feature.ui.run_phase" in plan.command
    assert "--action execute" in plan.command
    assert "--module" not in plan.command
    assert any(
        field.label == "Configured bias node" and field.value == "buy_hold"
        for field in plan.fields
    )
    assert any(field.label == "Validation window" for field in plan.fields)
    assert plan.combo_diagnostics is not None
    assert plan.combo_diagnostics.effective_combo_count == 2.5


def test_build_phase_plan_for_portfolio_addition_reviews_validation_gate_artifacts(monkeypatch) -> None:
    config = minimal_research_config()
    monkeypatch.setattr(
        "research.feature.ui.planner.build_combo_diagnostics",
        lambda _config: ComboDiagnostics(
            raw_combo_count=1,
            loaded_combo_count=1,
            effective_combo_count=1.0,
            evaluation_combo_label="empty_params",
        ),
    )
    request = build_ui_request(
        phase_name=FeatureResearchPhase.PORTFOLIO_ADDITION.value,
        tickers_text="GC",
        fallback_tickers=tuple(config.tickers),
    )

    plan = build_phase_plan(config, request)

    assert plan.phase is FeatureResearchPhase.PORTFOLIO_ADDITION
    assert any("visualization/validation" in step for step in plan.will_run)
    assert any("Validation completes" in note for note in plan.notes)
