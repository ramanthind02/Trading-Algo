from __future__ import annotations

import importlib

import research.feature
import research.feature.__main__ as feature_research_main
from research.feature import exploration, portfolio_addition, validation
from research.feature.exploration import (
    run_exploration_permutation_pipeline,
    run_exploration_pipeline,
    run_exploration_robustness_pipeline,
    write_exploration_permutation_summary,
    write_exploration_robustness_summary,
)
from research.feature.pipeline import (
    run_eda_pipeline,
    run_oos_pipeline,
    run_oos_pipeline_with_bundle,
    run_permutation_pipeline,
    run_robustness_pipeline,
    run_validation_pipeline,
    write_permutation_summary,
    write_robustness_summary,
)
from research.feature.portfolio_addition import (
    run_portfolio_addition_pipeline,
    run_portfolio_addition_pipeline_with_bundle,
)
from research.feature.shared import FeatureResearchPhase
from research.feature.shared.cli import build_usage, resolve_command


def test_root_package_exports_canonical_phase_modules() -> None:
    assert research.feature.exploration is exploration
    assert research.feature.validation is validation
    assert research.feature.portfolio_addition is portfolio_addition
    assert research.feature.FeatureResearchPhase is FeatureResearchPhase


def test_legacy_helper_imports_resolve_to_internal_modules() -> None:
    legacy_core_helpers = importlib.import_module("research.feature.core_helpers")
    legacy_bootstrap = importlib.import_module("research.feature.bootstrap")

    _ = legacy_core_helpers.combo_key
    _ = legacy_bootstrap.ensure_repo_root_on_syspath
    _ = legacy_bootstrap.find_repo_root
    legacy_core_helpers = importlib.import_module("research.feature.core_helpers")
    legacy_bootstrap = importlib.import_module("research.feature.bootstrap")

    assert legacy_core_helpers.__name__ == "research.feature._internal.core_helpers"
    assert legacy_bootstrap.__name__ == "research.feature._internal.bootstrap"


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
