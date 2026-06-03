from __future__ import annotations

from pathlib import Path

from feature_research.shared import FeatureResearchPhase
from feature_research.ui.workspace_manifest import (
    build_workspace_manifest,
    exploration_viz_matches_config,
    manifest_matches_config,
    validation_artifacts_current,
    validation_viz_matches_config,
    write_workspace_manifest,
)
from tests.feature_research.support.fixtures import minimal_research_config


def test_manifest_matches_config_requires_same_eval_target() -> None:
    config = minimal_research_config()
    manifest = build_workspace_manifest(config, FeatureResearchPhase.VALIDATION)
    assert manifest_matches_config(manifest, config)

    stale = dict(manifest)
    stale["module_name"] = "donchian_long_only"
    assert not manifest_matches_config(stale, config)


def test_validation_artifacts_current_false_without_manifest() -> None:
    config = minimal_research_config()
    assert not validation_artifacts_current(config)


def test_write_workspace_manifest_round_trip(tmp_path) -> None:
    config = minimal_research_config()
    manifest_path = write_workspace_manifest(
        tmp_path / "workspace_manifest.json",
        config,
        FeatureResearchPhase.EXPLORATION,
    )
    payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_matches_config(payload, config)


def test_exploration_viz_matches_config_allows_run_ticker_subset() -> None:
    from dataclasses import replace

    from utils.core.enums import Ticker

    config = minimal_research_config()
    manifest = build_workspace_manifest(config, FeatureResearchPhase.EXPLORATION)
    nq_only = dict(manifest)
    nq_only["tickers"] = ["NQ"]
    full_config = replace(config, tickers=[Ticker.ES, Ticker.GC, Ticker.NQ])

    assert not manifest_matches_config(nq_only, full_config)
    assert exploration_viz_matches_config(nq_only, full_config)


def test_exploration_viz_requires_matching_reports_dir() -> None:
    config = minimal_research_config()
    manifest = build_workspace_manifest(config, FeatureResearchPhase.EXPLORATION)
    stale = dict(manifest)
    stale["reports_dir"] = "feature_research/in_sample/results/signed_signal/stale"
    assert not exploration_viz_matches_config(stale, config)


def test_validation_viz_matches_config_allows_run_ticker_subset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dataclasses import replace

    from utils.core.enums import Ticker

    config = minimal_research_config()
    manifest = build_workspace_manifest(config, FeatureResearchPhase.VALIDATION)
    nq_only = dict(manifest)
    nq_only["tickers"] = ["NQ"]
    full_config = replace(config, tickers=[Ticker.ES, Ticker.GC, Ticker.NQ])

    assert not manifest_matches_config(nq_only, full_config)
    assert validation_viz_matches_config(nq_only, full_config)

    manifest_path = tmp_path / "workspace_manifest.json"
    manifest_path.write_text(__import__("json").dumps(nq_only), encoding="utf-8")
    monkeypatch.setattr(
        "feature_research.ui.workspace_manifest.validation_manifest_path",
        lambda _config: manifest_path,
    )
    assert validation_artifacts_current(full_config)
