from __future__ import annotations

import importlib

import feature_research
import feature_research.__main__ as feature_research_main
from feature_research import exploration, portfolio_addition, validation
from feature_research.exploration import (
    run_exploration_permutation_pipeline,
    run_exploration_pipeline,
    run_exploration_robustness_pipeline,
    write_exploration_permutation_summary,
    write_exploration_robustness_summary,
)
from feature_research.pipeline import (
    run_eda_pipeline,
    run_oos_pipeline,
    run_oos_pipeline_with_bundle,
    run_permutation_pipeline,
    run_robustness_pipeline,
    run_validation_pipeline,
    write_permutation_summary,
    write_robustness_summary,
)
from feature_research.portfolio_addition import (
    run_portfolio_addition_pipeline,
    run_portfolio_addition_pipeline_with_bundle,
)
from feature_research.shared import FeatureResearchPhase
from feature_research.shared.cli import build_usage, resolve_command


def test_root_package_exports_canonical_phase_modules() -> None:
    assert feature_research.exploration is exploration
    assert feature_research.validation is validation
    assert feature_research.portfolio_addition is portfolio_addition
    assert feature_research.FeatureResearchPhase is FeatureResearchPhase


def test_legacy_helper_imports_resolve_to_internal_modules() -> None:
    legacy_core_helpers = importlib.import_module("feature_research.core_helpers")
    legacy_bootstrap = importlib.import_module("feature_research.bootstrap")

    _ = legacy_core_helpers.combo_key
    _ = legacy_bootstrap.ensure_repo_root_on_syspath
    _ = legacy_bootstrap.find_repo_root
    legacy_core_helpers = importlib.import_module("feature_research.core_helpers")
    legacy_bootstrap = importlib.import_module("feature_research.bootstrap")

    assert legacy_core_helpers.__name__ == "feature_research._internal.core_helpers"
    assert legacy_bootstrap.__name__ == "feature_research._internal.bootstrap"


def test_pipeline_module_reexports_canonical_phase_wrappers() -> None:
    assert run_eda_pipeline is run_exploration_pipeline
    assert run_robustness_pipeline is run_exploration_robustness_pipeline
    assert write_robustness_summary is write_exploration_robustness_summary
    assert run_permutation_pipeline is run_exploration_permutation_pipeline
    assert write_permutation_summary is write_exploration_permutation_summary
    assert run_validation_pipeline is validation.run_validation_pipeline
    assert run_oos_pipeline is run_portfolio_addition_pipeline
    assert run_oos_pipeline_with_bundle is run_portfolio_addition_pipeline_with_bundle


def test_cli_usage_and_aliases_expose_canonical_phase_names() -> None:
    usage = build_usage()

    assert "Primary phases:" in usage
    assert "exploration" in usage
    assert "validation" in usage
    assert "portfolio_addition" in usage
    assert resolve_command("exploration") is not None
    assert resolve_command("in_sample").name == FeatureResearchPhase.EXPLORATION.value
    assert resolve_command("oos").name == FeatureResearchPhase.PORTFOLIO_ADDITION.value


def test_package_main_dispatches_canonical_and_legacy_aliases(
    monkeypatch,
) -> None:
    seen: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        feature_research_main,
        "run_command",
        lambda command, argv: seen.append((command.name, list(argv))) or 7,
    )

    assert feature_research_main.main(["portfolio_addition"]) == 7
    assert feature_research_main.main(["oos"]) == 7
    assert seen == [
        (FeatureResearchPhase.PORTFOLIO_ADDITION.value, []),
        (FeatureResearchPhase.PORTFOLIO_ADDITION.value, []),
    ]
